// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include <optional>

namespace vllm::batch_invariant {
void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& row_starts,
                           const torch::Tensor& row_ends,
                           torch::Tensor& indices, int64_t num_rows,
                           int64_t stride0, int64_t stride1, int64_t top_k);
void silu_mul_bf16(const torch::Tensor& input, torch::Tensor& output,
                   std::optional<double> clamp_limit);
void hc_collapse(const torch::Tensor& hidden, const torch::Tensor& pre,
                 torch::Tensor& output);
void sparse_metadata(const torch::Tensor& mask,
                     const std::optional<torch::Tensor>& main,
                     int64_t main_size, int64_t window, torch::Tensor& output,
                     torch::Tensor& lengths);

void cast_indexer_logits(const torch::Tensor& input,
                         const torch::Tensor& row_ends, torch::Tensor& output);
void dequant_gather_nvfp4(const torch::Tensor& cache,
                          const torch::Tensor& seq_lens,
                          const std::optional<torch::Tensor>& gather_lens,
                          const torch::Tensor& block_table, int64_t block_size,
                          int64_t offset, torch::Tensor& output);

void expert_permute(const torch::Tensor& a, const torch::Tensor& scale,
                    const torch::Tensor& routes,
                    const std::optional<torch::Tensor>& expert_map,
                    torch::Tensor& output, torch::Tensor& output_scale,
                    torch::Tensor& layout, torch::Tensor& inverse,
                    torch::Tensor& counts, torch::Tensor& starts,
                    int64_t alignment);
void gather_expert_slots(const torch::Tensor& values,
                         const torch::Tensor& routes,
                         const torch::Tensor& inverse,
                         const torch::Tensor& expert_map, torch::Tensor& slots);
void ordered_topk_combine(const torch::Tensor& slots,
                          const torch::Tensor& weights,
                          const std::optional<torch::Tensor>& routes,
                          int64_t experts, torch::Tensor& output);

void block32_scaled_mm(const torch::Tensor& a, const torch::Tensor& a_scale,
                       const torch::Tensor& weight,
                       const torch::Tensor& weight_scale, torch::Tensor& output,
                       bool grouped, int64_t weight_scale_group_rows);

void fused_silu_mul_per_token_group_quant(
    torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s,
    int64_t group_size, double eps, double min_8bit, double max_8bit,
    double clamp_limit, bool round_scale, bool scale_ue8m0,
    bool fuse_silu_and_mul, const std::optional<torch::Tensor>& masked_m,
    const std::optional<torch::Tensor>& row_scale);

}  // namespace vllm::batch_invariant

TORCH_LIBRARY(vllm_batch_invariant, ops) {
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor row_starts, Tensor "
      "row_ends, "
      "Tensor(a!) indices, int num_rows, int stride0, int stride1, int top_k) "
      "-> ()");
  ops.def(
      "silu_mul_bf16(Tensor input, Tensor(a!) output, float? clamp_limit) -> "
      "()");
  ops.def("hc_collapse(Tensor hidden, Tensor pre, Tensor(a!) output) -> ()");
  ops.def(
      "sparse_metadata(Tensor mask, Tensor? main, int main_size, int window, "
      "Tensor(a!) output, Tensor(b!) lengths) -> ()");
  ops.def(
      "cast_indexer_logits(Tensor input, Tensor row_ends, Tensor(a!) output) "
      "-> ()");
  ops.def(
      "dequant_gather_nvfp4(Tensor cache, Tensor seq_lens, Tensor? "
      "gather_lens, "
      "Tensor block_table, int block_size, int offset, Tensor(a!) output) -> "
      "()");
  ops.def(
      "expert_permute(Tensor a, Tensor scale, Tensor routes, Tensor? "
      "expert_map, "
      "Tensor(a!) output, Tensor(b!) output_scale, Tensor(c!) layout, "
      "Tensor(d!) inverse, Tensor(e!) counts, Tensor(f!) starts, int "
      "alignment) -> ()");
  ops.def(
      "gather_expert_slots(Tensor values, Tensor routes, Tensor inverse, "
      "Tensor expert_map, Tensor(a!) slots) -> ()");
  ops.def(
      "ordered_topk_combine(Tensor slots, Tensor weights, Tensor? routes, "
      "int experts, Tensor(a!) output) -> ()");
  ops.def(
      "block32_scaled_mm(Tensor a, Tensor a_scale, Tensor weight, "
      "Tensor weight_scale, Tensor(a!) output, bool grouped, "
      "int weight_scale_group_rows) -> ()");
  ops.def(
      "fused_silu_mul_per_token_group_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float min_8bit, "
      "float max_8bit, float clamp_limit, bool round_scale, bool scale_ue8m0, "
      "bool fuse_silu_and_mul, Tensor? masked_m, Tensor? row_scale=None) -> "
      "()");
}

TORCH_LIBRARY_IMPL(vllm_batch_invariant, CUDA, ops) {
  ops.impl("top_k_per_row_prefill",
           &vllm::batch_invariant::top_k_per_row_prefill);
  ops.impl("silu_mul_bf16", &vllm::batch_invariant::silu_mul_bf16);
  ops.impl("hc_collapse", &vllm::batch_invariant::hc_collapse);
  ops.impl("sparse_metadata", &vllm::batch_invariant::sparse_metadata);
  ops.impl("cast_indexer_logits", &vllm::batch_invariant::cast_indexer_logits);
  ops.impl("dequant_gather_nvfp4",
           &vllm::batch_invariant::dequant_gather_nvfp4);
  ops.impl("expert_permute", &vllm::batch_invariant::expert_permute);
  ops.impl("gather_expert_slots", &vllm::batch_invariant::gather_expert_slots);
  ops.impl("ordered_topk_combine",
           &vllm::batch_invariant::ordered_topk_combine);
  ops.impl("block32_scaled_mm", &vllm::batch_invariant::block32_scaled_mm);
  ops.impl("fused_silu_mul_per_token_group_quant",
           &vllm::batch_invariant::fused_silu_mul_per_token_group_quant);
}
