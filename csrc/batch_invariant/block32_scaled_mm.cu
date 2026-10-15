// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <climits>

namespace vllm::batch_invariant {
namespace {

struct MatrixLayout {
  int64_t row, group, k;
};

__device__ uint32_t load_fragment(const uint8_t* values, MatrixLayout stride,
                                  int64_t row, int group, int k, int rows) {
  if (row >= rows) return 0;
  const uint8_t* base =
      values + row * stride.row + group * stride.group + k * stride.k;
  if (stride.k == 1 && reinterpret_cast<uintptr_t>(base) % 4 == 0) {
    return *reinterpret_cast<const uint32_t*>(base);
  }
  uint32_t packed = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    packed |= uint32_t(values[row * stride.row + group * stride.group +
                              (k + byte) * stride.k])
              << (8 * byte);
  }
  return packed;
}

template <typename Scale>
__device__ float load_scale(const Scale* scales, MatrixLayout stride,
                            int64_t row, int group, int block) {
  int column = sizeof(Scale) == 4 ? block / 4 : block;
  uint32_t bits =
      scales[row * stride.row + group * stride.group + column * stride.k];
  if constexpr (sizeof(Scale) == 4) bits = (bits >> (8 * (block % 4))) & 255;
  bits = bits == 0 ? 0x00400000 : bits == 255 ? 0x7FC00000 : bits << 23;
  return __uint_as_float(bits);
}

__device__ float accumulate_block(float result, float product, float a_scale,
                                  float b_scale) {
  float scaled;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(product), "f"(a_scale));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(scaled), "f"(b_scale));
  asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(result), "f"(scaled));
  return result;
}

template <typename Scale, typename Output>
__global__ void block32_mm_kernel(const uint8_t* a, const Scale* a_scale,
                                  const uint8_t* weight,
                                  const Scale* weight_scale, Output* output,
                                  int m, int n, int k, int groups,
                                  int scale_rows, MatrixLayout a_stride,
                                  MatrixLayout as_stride, MatrixLayout w_stride,
                                  MatrixLayout ws_stride) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row = blockIdx.x * 16 + lane / 4;
  const int column = blockIdx.y * 32 + warp * 8;
  const int group = blockIdx.z;
  float result[4] = {};
  for (int block = 0; block < k / 32; ++block) {
    const int offset = block * 32 + (lane % 4) * 4;
    uint32_t a0 = load_fragment(a, a_stride, row, group, offset, m);
    uint32_t a1 = load_fragment(a, a_stride, row + 8, group, offset, m);
    uint32_t a2 = load_fragment(a, a_stride, row, group, offset + 16, m);
    uint32_t a3 = load_fragment(a, a_stride, row + 8, group, offset + 16, m);
    uint32_t b0 =
        load_fragment(weight, w_stride, column + lane / 4, group, offset, n);
    uint32_t b1 = load_fragment(weight, w_stride, column + lane / 4, group,
                                offset + 16, n);
    float product[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
#else
    asm volatile("trap;");
#endif
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int out_row = row + (element >= 2 ? 8 : 0);
      const int out_column = column + (lane % 4) * 2 + (element % 2);
      if (out_row < m && out_column < n) {
        result[element] = accumulate_block(
            result[element], product[element],
            load_scale(a_scale, as_stride, out_row, group, block),
            load_scale(weight_scale, ws_stride, out_column / scale_rows, group,
                       block));
      }
    }
  }
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    const int out_row = row + (element >= 2 ? 8 : 0);
    const int out_column = column + (lane % 4) * 2 + (element % 2);
    if (out_row < m && out_column < n) {
      output[(int64_t(out_row) * groups + group) * n + out_column] =
          Output(result[element]);
    }
  }
}

MatrixLayout layout(const torch::Tensor& tensor, bool grouped, bool weight) {
  if (!grouped) return {tensor.stride(0), 0, tensor.stride(1)};
  return weight ? MatrixLayout{tensor.stride(1), tensor.stride(0),
                               tensor.stride(2)}
                : MatrixLayout{tensor.stride(0), tensor.stride(1),
                               tensor.stride(2)};
}

}  // namespace

void block32_scaled_mm(const torch::Tensor& a, const torch::Tensor& a_scale,
                       const torch::Tensor& weight,
                       const torch::Tensor& weight_scale, torch::Tensor& output,
                       bool grouped, int64_t weight_scale_group_rows) {
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a_scale.device() == a.device() &&
                  weight.device() == a.device() &&
                  weight_scale.device() == a.device() &&
                  output.device() == a.device(),
              "K32 tensors must share a CUDA device");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  weight.scalar_type() == a.scalar_type(),
              "K32 operands must be E4M3");
  TORCH_CHECK(a.dim() == (grouped ? 3 : 2) && weight.dim() == a.dim() &&
                  a_scale.dim() == a.dim() && weight_scale.dim() == a.dim(),
              "Invalid K32 operand ranks");
  TORCH_CHECK(a_scale.scalar_type() == weight_scale.scalar_type() &&
                  (a_scale.scalar_type() == at::kByte ||
                   a_scale.scalar_type() == at::kInt),
              "K32 scales must be paired E8M0 bytes or packed INT32");
  const int64_t m = a.size(0), groups = grouped ? a.size(1) : 1;
  const int64_t n = weight.size(grouped ? 1 : 0), k = a.size(-1);
  TORCH_CHECK(k > 0 && k % 32 == 0 && weight.size(-1) == k &&
                  weight_scale_group_rows > 0 &&
                  weight_scale_group_rows <= INT_MAX,
              "Invalid K32 geometry");
  const int64_t scale_columns =
      a_scale.scalar_type() == at::kInt ? (k / 32 + 3) / 4 : k / 32;
  TORCH_CHECK(
      a_scale.size(0) >= m && a_scale.size(-1) >= scale_columns &&
          weight_scale.size(grouped ? 1 : 0) >=
              (n + weight_scale_group_rows - 1) / weight_scale_group_rows &&
          weight_scale.size(-1) >= scale_columns,
      "K32 scale capacity is too small");
  TORCH_CHECK(
      !grouped || (weight.size(0) == groups && a_scale.size(1) == groups &&
                   weight_scale.size(0) == groups),
      "K32 groups must match");
  TORCH_CHECK(output.is_contiguous() && output.dim() == a.dim() &&
                  output.size(0) == m && output.size(-1) == n &&
                  (!grouped || output.size(1) == groups),
              "Invalid K32 output shape");
  TORCH_CHECK(output.scalar_type() == at::kFloat ||
                  output.scalar_type() == at::kHalf ||
                  output.scalar_type() == at::kBFloat16,
              "Invalid K32 output dtype");
  TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX && groups <= 65535 &&
                  (n + 31) / 32 <= 65535,
              "K32 launch geometry is too large");
  if (output.numel() == 0) return;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major * 10 + properties->minor >= 89,
              "K32 FP8 MMA requires SM89 or newer");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((m + 15) / 16, (n + 31) / 32, groups);
  const auto a_stride = layout(a, grouped, false),
             as_stride = layout(a_scale, grouped, false);
  const auto w_stride = layout(weight, grouped, true),
             ws_stride = layout(weight_scale, grouped, true);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, output.scalar_type(), "block32_scaled_mm", [&] {
        if (a_scale.scalar_type() == at::kInt) {
          block32_mm_kernel<int32_t, scalar_t><<<grid, 128, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(a.data_ptr()),
              a_scale.data_ptr<int32_t>(),
              reinterpret_cast<const uint8_t*>(weight.data_ptr()),
              weight_scale.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), m,
              n, k, groups, weight_scale_group_rows, a_stride, as_stride,
              w_stride, ws_stride);
        } else {
          block32_mm_kernel<uint8_t, scalar_t><<<grid, 128, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(a.data_ptr()),
              a_scale.data_ptr<uint8_t>(),
              reinterpret_cast<const uint8_t*>(weight.data_ptr()),
              weight_scale.data_ptr<uint8_t>(), output.data_ptr<scalar_t>(), m,
              n, k, groups, weight_scale_group_rows, a_stride, as_stride,
              w_stride, ws_stride);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::batch_invariant
