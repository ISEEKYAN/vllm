# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
from pathlib import Path

import torch

_SHAPES = {
    (2688, 12864),
    (5376, 2688),
    (2688, 4608),
    (4096, 2688),
    (2688, 3712),
    (3712, 2688),
}
_INSTALLED = False
_DISABLED = False


def _load_extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "cublaslt_pinned.cpp"
    build_dir = Path.home() / ".cache" / "vllm_batch_invariant_cublaslt"
    build_dir.mkdir(parents=True, exist_ok=True)
    include_dirs = []
    library_dirs = []
    rpaths = []
    for base in (
        Path(torch.__file__).parent.parent / "nvidia" / "cu13",
        Path(torch.__file__).parent.parent / "nvidia" / "cublas",
        Path("/usr/local/cuda"),
    ):
        if (base / "include" / "cublasLt.h").exists():
            include_dirs.append(str(base / "include"))
        for subdir in ("lib", "lib64"):
            if list((base / subdir).glob("libcublasLt.so*")):
                library_dirs.append(str(base / subdir))
                rpaths.append(str(base / subdir))

    shim = build_dir / "libshim"
    shim.mkdir(exist_ok=True)
    for library in ("cublasLt", "cublas"):
        matches = []
        for directory in library_dirs:
            matches.extend(glob.glob(str(Path(directory) / f"lib{library}.so*")))
        if not matches:
            raise RuntimeError(f"lib{library} is unavailable")
        target = shim / f"lib{library}.so"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(sorted(matches)[0])

    return load(
        name="vllm_batch_invariant_cublaslt",
        sources=[str(source)],
        extra_include_paths=include_dirs,
        extra_ldflags=[f"-L{shim}"]
        + [f"-Wl,-rpath,{path}" for path in rpaths]
        + ["-lcublasLt", "-lcublas"],
        extra_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=False,
    )


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.view(torch.int16), right.view(torch.int16)))


def _check_shape(extension, stock, k: int, n: int) -> bool:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(k * 100_000 + n)
    weight = (
        torch.randn(
            n,
            k,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    for m in (1, 17, 320, 1024):
        x = (
            torch.randn(
                m,
                k,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.05
        )
        reference = stock(x, weight.t())
        output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        extension.mm(x, weight, output)
        repeated = torch.empty_like(output)
        extension.mm(x, weight, repeated)
        if not _bitwise_equal(reference, output) or not _bitwise_equal(
            output, repeated
        ):
            return False
    return True


def install_pinned_cublaslt() -> bool:
    global _DISABLED, _INSTALLED
    if _INSTALLED:
        return True
    if _DISABLED or os.environ.get("NEMOTRON_BATCH_INVARIANT_CUBLASLT") != "1":
        return False

    from vllm.model_executor.layers import batch_invariant

    stock = batch_invariant.matmul_persistent
    try:
        extension = _load_extension()
        extension.init()
        admitted = {
            shape
            for shape in _SHAPES
            if _check_shape(extension, stock, shape[0], shape[1])
        }
    except Exception as error:
        print(
            "Pinned batch-invariant cuBLASLt initialization failed; "
            f"using Triton: {error}",
            flush=True,
        )
        _DISABLED = True
        return False

    if admitted != _SHAPES:
        print(
            "Pinned batch-invariant cuBLASLt self-check rejected shapes "
            f"{sorted(_SHAPES - admitted)}; using Triton.",
            flush=True,
        )
        _DISABLED = True
        return False

    @torch.library.custom_op(
        "vllm::nemotron_batch_invariant_cublaslt",
        mutates_args=(),
    )
    def cublaslt_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        output = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
        extension.mm(a.contiguous(), b.t(), output)
        return output

    @cublaslt_op.register_fake
    def cublaslt_op_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

    def matmul_pinned(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = (a.shape[1], b.shape[1])
        if (
            bias is None
            and a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16
            and key in admitted
            and a.stride(1) == 1
            and b.stride() == (1, a.shape[1])
        ):
            return cublaslt_op(a, b)
        return stock(a, b, bias=bias)

    batch_invariant.matmul_persistent = matmul_pinned
    _INSTALLED = True
    print(
        "Pinned batch-invariant cuBLASLt enabled for Nemotron dense shapes: "
        f"{sorted(admitted)}",
        flush=True,
    )
    return True
