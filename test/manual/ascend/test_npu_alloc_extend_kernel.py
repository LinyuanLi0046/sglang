#!/usr/bin/env python3
"""Exercise the framework NPU allocator across real Prefill shape changes.

This deliberately launches one small warmup allocation, then changes request
batch size, extend length, free-page pointer alignment, and page count.  With
Triton compile logging enabled there must be no new ``alloc_extend_kernel_npu``
compile after the ``WARMUP_DONE`` marker.

The allocator starts with the captured production shape ``free_pages[17658]``
and is reused across calls, exactly like the serving process.  Therefore every
allocation also re-slices ``free_pages`` and changes its live shape/data_ptr.

Run on the A5 host:

    python test/manual/ascend/test_npu_alloc_extend_kernel.py --device npu:5

Native operator profiling (do not use event timing):

    msprof op --kernel-name=alloc_extend_kernel_npu \
        --output=./msprof_alloc_extend_framework \
        python test/manual/ascend/test_npu_alloc_extend_kernel.py \
        --device npu:5
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: E402

from sglang.kernels.ops.memory.allocator import (  # noqa: E402
    NPU_ALLOC_EXTEND_BLOCK_SIZE,
    NPU_ALLOC_EXTEND_MAX_BATCH_SIZE,
)
from sglang.srt.hardware_backend.npu.allocator_npu import (  # noqa: E402
    NPUPagedTokenToKVPoolAllocator,
)


PAGE_SIZE = 128
FREE_PAGE_COUNT = 17658
RESERVED_PREFIX_PAGES = 256


@dataclass(frozen=True)
class Case:
    name: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]


def _make_cases() -> list[Case]:
    return [
        Case("warmup_bs1_m1", (0,), (1,)),
        Case("formal_bs1_m16384", (65,), (16384,)),
        Case("formal_bs1_m16385", (0,), (16385,)),
        Case("formal_bs1_m20001", (17,), (20001,)),
        Case(
            "formal_bs8_uneven",
            (0, 1, 65, 127, 128, 129, 255, 513),
            (1, 127, 128, 129, 2047, 2048, 4097, 20001),
        ),
        Case(
            "formal_bs128_uneven",
            tuple(index % 257 for index in range(128)),
            tuple(1 + (index * 193) % 4097 for index in range(128)),
        ),
    ]


def _set_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type != "npu":
        raise ValueError(f"This test requires an NPU device, got {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    return device


def _last_locs(prefix_lens: tuple[int, ...]) -> list[int]:
    return [
        -1
        if prefix_len == 0
        else (1 + request_id) * PAGE_SIZE
        + (prefix_len - 1) % PAGE_SIZE
        for request_id, prefix_len in enumerate(prefix_lens)
    ]


def _reference(
    case: Case,
    free_pages: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    last_locs = _last_locs(case.prefix_lens)
    output: list[int] = []
    free_page_cursor = 0
    for prefix_len, extend_len, last_loc in zip(
        case.prefix_lens,
        case.extend_lens,
        last_locs,
        strict=True,
    ):
        part1 = 0
        if prefix_len % PAGE_SIZE:
            part1 = min(extend_len, PAGE_SIZE - prefix_len % PAGE_SIZE)
        output.extend(last_loc + 1 + offset for offset in range(part1))

        remaining = extend_len - part1
        request_pages = math.ceil(remaining / PAGE_SIZE)
        for offset in range(remaining):
            page_id = int(free_pages[free_page_cursor + offset // PAGE_SIZE])
            output.append(page_id * PAGE_SIZE + offset % PAGE_SIZE)
        free_page_cursor += request_pages

    return torch.tensor(output, dtype=torch.int32), free_page_cursor


def _make_allocator(
    device: torch.device,
    free_page_offset: int,
) -> tuple[NPUPagedTokenToKVPoolAllocator, torch.Tensor]:
    # Use the real framework constructor/clear path.  ``kvcache`` is unused by
    # allocation itself, so no model-sized KV tensors are needed for this test.
    # Prefix last_loc values refer to pages 1..128; reserve 1..256 so those
    # existing pages cannot alias the live free-page pool.
    allocator = NPUPagedTokenToKVPoolAllocator(
        size=(
            FREE_PAGE_COUNT
            + RESERVED_PREFIX_PAGES
            + free_page_offset
        )
        * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=False,
    )
    allocator.free_pages = allocator.free_pages[
        RESERVED_PREFIX_PAGES + free_page_offset :
    ]
    return allocator, allocator.free_pages.cpu()


def _audit_source_contract() -> None:
    kernel_source_path = (
        PYTHON_ROOT / "sglang/kernels/ops/memory/allocator.py"
    )
    source = kernel_source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "alloc_extend_kernel_npu"
    )
    argument_names = {argument.arg for argument in function.args.args}
    forbidden = {"extend_num_tokens", "max_num_extend_tokens", "batch_size"}
    if forbidden & argument_names:
        raise AssertionError(
            "Request-varying scalar leaked into JIT signature: "
            f"{sorted(forbidden & argument_names)}"
        )
    if "tl.program_id(1)" not in source:
        raise AssertionError("NPU allocator kernel is not chunk-grid based")


def _run_case(
    allocator: NPUPagedTokenToKVPoolAllocator,
    free_pages_cpu: torch.Tensor,
    case: Case,
) -> tuple[torch.Tensor, int]:
    prefix_cpu = torch.tensor(case.prefix_lens, dtype=torch.int64)
    extend_cpu = torch.tensor(case.extend_lens, dtype=torch.int64)
    seq_cpu = prefix_cpu + extend_cpu
    last_locs_cpu = torch.tensor(_last_locs(case.prefix_lens), dtype=torch.int64)
    expected, expected_new_pages = _reference(case, free_pages_cpu)

    pointer_mod_16 = allocator.free_pages.data_ptr() % 16
    free_pages_len = allocator.free_pages.numel()
    output = allocator.alloc_extend(
        prefix_lens=prefix_cpu.to(allocator.device),
        prefix_lens_cpu=prefix_cpu,
        seq_lens=seq_cpu.to(allocator.device),
        seq_lens_cpu=seq_cpu,
        last_loc=last_locs_cpu.to(allocator.device),
        extend_num_tokens=int(extend_cpu.sum().item()),
    )
    if output is None:
        raise AssertionError(f"allocator unexpectedly returned None for {case.name}")
    torch_npu.npu.synchronize()
    actual = output.cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected).flatten()
        first = mismatch[:10].tolist()
        details = [
            (index, int(actual[index]), int(expected[index]))
            for index in first
        ]
        raise AssertionError(
            f"{case.name} mismatch_count={mismatch.numel()}, "
            f"first(index, actual, expected)={details}"
        )
    if torch.unique(actual).numel() != actual.numel():
        raise AssertionError(f"{case.name} produced duplicate KV slot indices")

    print(
        f"PASS {case.name}: bs={len(case.prefix_lens)}, "
        f"input_shapes={len(case.prefix_lens)};{len(case.prefix_lens)};"
        f"{len(case.prefix_lens)};{free_pages_len}, "
        f"total_extend={sum(case.extend_lens)}, "
        f"max_extend={max(case.extend_lens)}, "
        f"new_pages={expected_new_pages}, "
        f"free_page_ptr_mod16={pointer_mod_16}"
    )
    return free_pages_cpu[expected_new_pages:], expected_new_pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:5")
    parser.add_argument(
        "--free-page-offset",
        type=int,
        default=0,
        help="Initial int64 slice offset; use 1 to start from an unaligned pointer.",
    )
    args = parser.parse_args()
    if args.free_page_offset < 0:
        raise ValueError("--free-page-offset must be non-negative")

    _audit_source_contract()
    device = _set_device(args.device)
    allocator, free_pages_cpu = _make_allocator(
        device,
        args.free_page_offset,
    )
    if allocator.free_pages.numel() != FREE_PAGE_COUNT:
        raise AssertionError(
            f"Expected captured free_pages[{FREE_PAGE_COUNT}], got "
            f"free_pages[{allocator.free_pages.numel()}]"
        )
    cases = _make_cases()
    if max(len(case.prefix_lens) for case in cases) > NPU_ALLOC_EXTEND_MAX_BATCH_SIZE:
        raise AssertionError("test case exceeds the fast-kernel batch envelope")

    free_pages_cpu, _ = _run_case(allocator, free_pages_cpu, cases[0])
    print(
        "WARMUP_DONE: subsequent cases vary N, BS, chunk count, page count, "
        "and free-page alignment; no further alloc_extend_kernel_npu JIT is expected."
    )
    for case in cases[1:]:
        free_pages_cpu, _ = _run_case(allocator, free_pages_cpu, case)

    print(
        "PASS framework sequence: one warmup plus all formal cases; "
        f"BLOCK_SIZE={NPU_ALLOC_EXTEND_BLOCK_SIZE}, "
        f"MAX_BATCH_SIZE={NPU_ALLOC_EXTEND_MAX_BATCH_SIZE}"
    )


if __name__ == "__main__":
    main()
