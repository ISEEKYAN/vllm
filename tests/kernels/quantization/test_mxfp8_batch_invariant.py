# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 products remain quantized and invariant to token partitioning."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import block32_scaled_mm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FP8 GEMM")
def test_block32_scaled_mm_known_values_and_token_partition():
    torch.manual_seed(723)
    for n, k in ((32, 64), (256, 512)):
        a = torch.randint(-2, 3, (7, k), device="cuda").float()
        b = torch.randint(-2, 3, (n, k), device="cuda").float()
        sf = torch.tensor([126, 127, 128], device="cuda", dtype=torch.uint8)
        sa = sf[torch.arange(7 * k // 32, device="cuda") % 3].reshape(7, -1)
        sb = sf[torch.arange(n * k // 32, device="cuda") % 3].reshape(n, -1)
        qa, qb = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        actual = block32_scaled_mm(qa, sa, qb, sb, torch.bfloat16)
        dense_a = a * sa.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        dense_b = b * sb.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        expected = (dense_a.double() @ dense_b.double().T).bfloat16()
        assert torch.equal(actual, expected)
        for chunk in (1, 3):
            parts = [
                block32_scaled_mm(
                    qa[i : i + chunk], sa[i : i + chunk], qb, sb, torch.bfloat16
                )
                for i in range(0, 7, chunk)
            ]
            assert torch.equal(actual, torch.cat(parts))
