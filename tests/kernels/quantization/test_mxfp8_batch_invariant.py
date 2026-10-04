# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 products remain quantized and invariant to token partitioning."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import (
    block32_scaled_mm,
    packed_block32_grouped_mm,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FP8 GEMM")
def test_packed_grouped_mm_padding_strides_known_values_and_token_partition():
    torch.manual_seed(727)
    for groups, n, k in ((1, 32, 128), (2, 512, 4096)):
        m, padded_m = 7, 8
        a = torch.randint(-2, 3, (m, groups, k), device="cuda").float()
        weight = torch.randint(-2, 3, (groups, n, k), device="cuda").float()
        sf = torch.tensor([126, 127, 128], device="cuda", dtype=torch.uint8)
        sa = sf[torch.arange(m * groups * k // 32, device="cuda") % 3].reshape(
            m, groups, -1
        )
        sw = sf[torch.arange(groups * n * k // 32, device="cuda") % 3].reshape(
            groups, n, -1
        )
        # Genuine logical MN-major strides, four packed exponents per INT32.
        pa = torch.empty(
            (groups, k // 128, padded_m), device="cuda", dtype=torch.int32
        ).permute(2, 0, 1)
        pa.fill_(-1)  # NaN exponent bytes in padding must never be consumed.
        pa[:m].copy_(sa.contiguous().view(torch.int32))
        pw = torch.empty(
            (groups, k // 128, n), device="cuda", dtype=torch.int32
        ).permute(0, 2, 1)
        pw.copy_(sw.contiguous().view(torch.int32))
        qa, qw = a.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
        actual = packed_block32_grouped_mm(qa, pa, qw, pw, torch.bfloat16)
        da = a * sa.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        dw = weight * sw.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        expected = torch.stack(
            [da[:, g].double() @ dw[g].double().T for g in range(groups)], dim=1
        ).bfloat16()
        assert torch.equal(actual, expected)
        for chunk in (1, 3):
            parts = [
                packed_block32_grouped_mm(
                    qa[i : i + chunk], pa[i : i + chunk], qw, pw, torch.bfloat16
                )
                for i in range(0, m, chunk)
            ]
            assert torch.equal(actual, torch.cat(parts))
