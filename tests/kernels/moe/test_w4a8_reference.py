# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from tests.kernels.moe.w4a8_reference import quantize_a8, swiglu_clamp, topk_fma


class TestW4A8Reference(unittest.TestCase):
    def test_clamp_precedes_silu_and_leaves_negative_gate_unclamped(self):
        x = torch.tensor([[20, -20, 1, 20, 20, -20]], dtype=torch.bfloat16)
        expected = torch.tensor(
            [
                [
                    10 / (1 + torch.exp(torch.tensor(-10.0))) * 10,
                    -20 / (1 + torch.exp(torch.tensor(20.0))) * 10,
                    -10 / (1 + torch.exp(torch.tensor(-1.0))),
                ]
            ],
            dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(swiglu_clamp(x), expected))
        self.assertLess(swiglu_clamp(x)[0, 1].abs(), 1e-6)

    def test_a8_zero_tiny_and_exact_power_of_two_scales(self):
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        x[1, 0] = 1e-12
        x[2, 0] = 448
        x[3, 0] = 450
        codes, scales = quantize_a8(x)
        self.assertTrue(torch.equal(scales[:, 0], torch.tensor([2**-33, 2**-33, 1, 2])))
        self.assertEqual(codes[0].view(torch.uint8).count_nonzero(), 0)
        # 225 rounds to 224 in E4M3, ties-to-even.
        self.assertEqual(codes[3, 0].float(), 224)

    def test_topk_rounds_after_all_slots_and_preserves_slot_order(self):
        rows = torch.tensor([[[1], [2**-8], [2**-8]]], dtype=torch.bfloat16)
        weights = torch.ones(1, 3)
        self.assertEqual(topk_fma(rows, weights).item(), 1 + 2**-7)
        large = torch.tensor([[[2**30], [1], [-(2**30)]]], dtype=torch.bfloat16)
        self.assertEqual(topk_fma(large, weights).item(), 0)
        self.assertEqual(topk_fma(large[:, [0, 2, 1]], weights).item(), 1)

    def test_topk_uses_fma_not_rounded_product_then_add(self):
        value = 1 + 2**-7
        rows = torch.tensor([[[-value], [value]]], dtype=torch.bfloat16)
        weights = torch.tensor([[1, 1 + 2**-23]], dtype=torch.float32)
        self.assertEqual(topk_fma(rows, weights).item(), value * 2**-23)


if __name__ == "__main__":
    unittest.main()
