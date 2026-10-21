// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SwiGLU, routing weight and MXFP8 cast with the exact arithmetic of DeepGEMM's
// fp8_fp4_mega_moe L1 epilogue (sm100_fp8_fp4_mega_moe.cuh), so an unfused
// MegaMoE (grouped GEMM -> this op -> grouped GEMM) reproduces its bits:
// BF16 clamp (gate from above, up both sides), sigmoid via __expf and
// rcp.approx.ftz under fast math, (silu * up) * weight with two FP32 RN
// products, no BF16 rounding before the cast, then one UE8M0 scale per 32
// columns from the FP32 amax and a saturating E4M3 cast.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <optional>

namespace vllm::batch_invariant {
namespace {

constexpr int kGroup = 32;

__device__ __forceinline__ float fast_rcp(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// DeepGEMM math::get_ue8m0_sf_exp<float, e4m3>: ceil(log2(amax / 448)) as a
// biased exponent, floored at 2^-22 (the 1e-4 amax floor).
__device__ __forceinline__ uint32_t ue8m0_exp(float amax) {
  const uint32_t bits = __float_as_uint(amax);
  const uint32_t rounded = (bits + ((1u << 23) - 1) - (0x60u << 16)) >> 23;
  return max(rounded, 105u + 8u) - 8u;
}

template <bool kFastMath>
__global__ void swiglu_apply_weight_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ input, const float* __restrict__ weights,
    const int32_t* __restrict__ avail_rows, __nv_fp8_e4m3* __restrict__ out_q,
    uint8_t* __restrict__ out_s, __nv_bfloat16* __restrict__ out_bf16,
    int64_t rows, int64_t inter, int64_t sf_stride, bool has_clamp,
    float clamp) {
  const int64_t groups = inter / kGroup;
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= rows * groups) return;
  const int64_t row = gid / groups, group = gid % groups;
  if (avail_rows != nullptr && row >= avail_rows[0]) return;
  const float weight = weights != nullptr ? weights[row] : 1.0f;
  const __nv_bfloat16 hi = __float2bfloat16_rn(clamp);
  const __nv_bfloat16 lo = __float2bfloat16_rn(-clamp);
  const __nv_bfloat16* gate_row = input + row * 2 * inter + group * kGroup;
  const __nv_bfloat16* up_row = gate_row + inter;

  float values[kGroup];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < kGroup; ++j) {
    __nv_bfloat16 g = gate_row[j], u = up_row[j];
    if (has_clamp) {
      g = __hmin(g, hi);
      u = __hmin(__hmax(u, lo), hi);
    }
    const float gate = __bfloat162float(g), up = __bfloat162float(u);
    const float e = kFastMath ? __expf(-gate) : expf(-gate);
    const float d = __fadd_rn(1.0f, e);
    const float s =
        kFastMath ? __fmul_rn(gate, fast_rcp(d)) : __fdiv_rn(gate, d);
    values[j] = __fmul_rn(__fmul_rn(s, up), weight);
    amax = fmaxf(amax, fabsf(values[j]));
  }
  const uint32_t exponent = ue8m0_exp(amax);
  const float inverse = __uint_as_float((254u - exponent) << 23);
  __nv_fp8_e4m3* q_row = out_q + row * inter + group * kGroup;
#pragma unroll
  for (int j = 0; j < kGroup; ++j)
    q_row[j] = __nv_fp8_e4m3(__fmul_rn(values[j], inverse));  // RN, satfinite
  if (out_bf16 != nullptr) {
    __nv_bfloat16* b_row = out_bf16 + row * inter + group * kGroup;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) b_row[j] = __float2bfloat16_rn(values[j]);
  }
  // Packed UE8M0, four per int32 along the columns, MN-major int32 words.
  out_s[((group / 4) * sf_stride + row) * 4 + group % 4] = uint8_t(exponent);
}

}  // namespace

void swiglu_apply_weight_to_fp8(
    const torch::Tensor& input, const std::optional<torch::Tensor>& weights,
    const std::optional<torch::Tensor>& avail_rows, torch::Tensor& output_q,
    torch::Tensor& output_s, const std::optional<torch::Tensor>& output_bf16,
    std::optional<double> clamp_limit, bool fast_math) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
                  input.dim() == 2 && input.is_contiguous() &&
                  input.size(1) % (2 * 4 * kGroup) == 0,
              "swiglu_apply_weight_to_fp8 requires contiguous CUDA BF16 "
              "[M, 2N] input with N a multiple of 128");
  const int64_t rows = input.size(0), inter = input.size(1) / 2;
  TORCH_CHECK(output_q.device() == input.device() &&
                  output_q.scalar_type() == at::kFloat8_e4m3fn &&
                  output_q.dim() == 2 && output_q.size(0) == rows &&
                  output_q.size(1) == inter && output_q.is_contiguous(),
              "output_q must be contiguous FP8 E4M3 [M, N]");
  TORCH_CHECK(output_s.device() == input.device() &&
                  output_s.scalar_type() == at::kInt && output_s.dim() == 2 &&
                  output_s.size(0) == rows &&
                  output_s.size(1) == inter / (4 * kGroup) &&
                  output_s.stride(0) == 1 && output_s.stride(1) >= rows &&
                  output_s.stride(1) % 4 == 0,
              "output_s must be MN-major TMA-aligned packed UE8M0 int32 "
              "[M, N/128]");
  if (weights) {
    TORCH_CHECK(weights->device() == input.device() &&
                    weights->scalar_type() == at::kFloat &&
                    weights->dim() == 1 && weights->size(0) == rows &&
                    weights->is_contiguous(),
                "weights must be contiguous FP32 [M]");
  }
  if (avail_rows) {
    TORCH_CHECK(avail_rows->device() == input.device() &&
                    avail_rows->scalar_type() == at::kInt &&
                    avail_rows->numel() == 1,
                "avail_rows must be a one-element CUDA int32 tensor");
  }
  if (output_bf16) {
    TORCH_CHECK(output_bf16->device() == input.device() &&
                    output_bf16->scalar_type() == at::kBFloat16 &&
                    output_bf16->sizes() == output_q.sizes() &&
                    output_bf16->is_contiguous(),
                "output_bf16 must be contiguous BF16 [M, N]");
  }
  TORCH_CHECK(
      !clamp_limit || (std::isfinite(*clamp_limit) && *clamp_limit >= 0),
      "SwiGLU clamp must be finite and non-negative");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t threads = rows * (inter / kGroup);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto launch = fast_math ? swiglu_apply_weight_to_fp8_kernel<true>
                                : swiglu_apply_weight_to_fp8_kernel<false>;
  launch<<<(threads + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      weights ? weights->data_ptr<float>() : nullptr,
      avail_rows ? avail_rows->data_ptr<int32_t>() : nullptr,
      reinterpret_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
      reinterpret_cast<uint8_t*>(output_s.data_ptr()),
      output_bf16 ? reinterpret_cast<__nv_bfloat16*>(output_bf16->data_ptr())
                  : nullptr,
      rows, inter, output_s.stride(1), clamp_limit.has_value(),
      float(clamp_limit.value_or(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
