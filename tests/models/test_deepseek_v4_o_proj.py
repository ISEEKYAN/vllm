from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4.nvidia.ops import o_proj


@pytest.mark.parametrize(
    ("major", "recipe", "tma_aligned", "use_triton"),
    [
        (9, (1, 128, 128), False, True),
        (10, (1, 1, 128), True, False),
    ],
)
def test_batch_invariant_o_proj_backend_selection(
    monkeypatch,
    major: int,
    recipe: tuple[int, int, int],
    tma_aligned: bool,
    use_triton: bool,
) -> None:
    monkeypatch.setattr(
        o_proj.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=major),
    )
    assert o_proj.compute_fp8_einsum_recipe() == (recipe, tma_aligned)
    assert (
        o_proj.use_triton_w8a8_fallback(
            batch_invariant=True,
            tma_aligned_scales=tma_aligned,
        )
        is use_triton
    )
    assert not o_proj.use_triton_w8a8_fallback(
        batch_invariant=False,
        tma_aligned_scales=tma_aligned,
    )
