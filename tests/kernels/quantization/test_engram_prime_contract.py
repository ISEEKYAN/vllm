# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU prime contract, independent sympy oracle and release layout preservation."""

import ast
from pathlib import Path
from typing import Any

import pytest
from sympy import isprime, nextprime


def production_functions():
    # Import only the actual pure scalar functions, avoiding CUDA model-layer
    # imports on CPU contract workers. No substitute implementation is used.
    import vllm

    path = Path(vllm.__file__).parent / "models/deepseek_v41/common/engram.py"
    tree = ast.parse(path.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_is_prime", "find_next_prime")
    ]
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        2,
        7,
        37,
        59,
        61,
        67,
        71,
        341,
        561,
        16000001,
        16000057,
        4294967291,
        4294967295,
    ],
)
def test_native_prime_matches_independent_oracle(value):
    assert production_functions()["_is_prime"](value) == isprime(value)


def test_all_small_primes_and_composites():
    functions = production_functions()
    for value in range(1000):
        assert functions["_is_prime"](value) == isprime(value), value


@pytest.mark.parametrize(
    "start, count, expected_sum", [(32, 8, 408), (15999999, 24, 384006168)]
)
def test_tiny_and_release_bucket_layouts(start, count, expected_sum):
    functions = production_functions()
    native: list[int] = []
    reference: list[int] = []
    seen: set[int] = set()
    current = independent = start
    for _ in range(count):
        current = functions["find_next_prime"](current, seen)
        independent = int(nextprime(independent))
        native.append(current)
        reference.append(independent)
        seen.add(current)
    assert native == reference
    assert sum(native) == expected_sum
