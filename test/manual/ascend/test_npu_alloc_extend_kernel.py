#!/usr/bin/env python3
"""Correctness and native msprof-op probe for NPU alloc_extend_kernel_npu.

The default tensors reproduce the captured input shapes:

    prefix_lens[1]; seq_lens[1]; last_loc[1]; free_pages[17658]

Correctness on NPU 5:

    python test/manual/ascend/test_npu_alloc_extend_kernel.py \
        --mode check --device npu:5

Authoritative A5 latency (no event timing):

    msprof op --warm-up=5 --launch-count=20 \
        --kernel-name=alloc_extend_kernel_npu \
        --output=./msprof_alloc_extend_framework \
        python test/manual/ascend/test_npu_alloc_extend_kernel.py \
        --mode profile --device npu:5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402

from sglang.srt.hardware_backend.npu.allocator_npu import (
    NPU_ALLOC_EXTEND_MAX_TOKENS,
    alloc_extend_kernel_npu,
)  # noqa: E402


KERNEL_NAME = "alloc_extend_kernel_npu"
DEFAULT_FREE_PAGES_LEN = 17658
FREE_PAGE_BASE = 1000
EXISTING_PAGE_ID = 17


@dataclass
class Case:
    prefix_lens: torch.Tensor
    seq_lens: torch.Tensor
    last_loc: torch.Tensor
    free_pages: torch.Tensor
    expected_cpu: torch.Tensor
    extend_len: int
    num_new_pages: int
    page_size: int


def set_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type != "npu":
        raise ValueError(f"This test requires an NPU device, got {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    return device


def make_reference(
    prefix_len: int,
    extend_len: int,
    page_size: int,
    last_loc: int,
    free_pages_cpu: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if prefix_len % page_size == 0:
        num_part1 = 0
    else:
        num_part1 = min(extend_len, page_size - prefix_len % page_size)

    result: list[int] = []
    for offset in range(num_part1):
        result.append(last_loc + 1 + offset)

    remaining = extend_len - num_part1
    num_new_pages = (remaining + page_size - 1) // page_size
    if num_new_pages > free_pages_cpu.numel():
        raise ValueError(
            f"Need {num_new_pages} free pages, only have {free_pages_cpu.numel()}"
        )

    for page_offset in range(num_new_pages):
        page_id = int(free_pages_cpu[page_offset])
        count = min(page_size, remaining - page_offset * page_size)
        page_base = page_id * page_size
        result.extend(page_base + token_offset for token_offset in range(count))

    if len(result) != extend_len:
        raise AssertionError(
            f"Reference generated {len(result)} slots, expected {extend_len}"
        )
    return torch.tensor(result, dtype=torch.int64), num_new_pages


def make_case(args: argparse.Namespace, device: torch.device) -> Case:
    if args.prefix_len < 0:
        raise ValueError("--prefix-len must be non-negative")
    if args.extend_len <= 0:
        raise ValueError("--extend-len must be positive")
    if args.extend_len > NPU_ALLOC_EXTEND_MAX_TOKENS:
        raise ValueError(
            f"The fast kernel supports at most {NPU_ALLOC_EXTEND_MAX_TOKENS} "
            f"extend tokens, got {args.extend_len}"
        )
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")
    if args.free_pages_len <= 0:
        raise ValueError("--free-pages-len must be positive")
    if args.free_page_offset < 0:
        raise ValueError("--free-page-offset must be non-negative")

    prefix_len = args.prefix_len
    seq_len = prefix_len + args.extend_len
    if prefix_len == 0:
        last_loc_value = -1
    else:
        last_loc_value = (
            EXISTING_PAGE_ID * args.page_size + (prefix_len - 1) % args.page_size
        )

    backing_len = args.free_pages_len + args.free_page_offset
    free_pages_backing_cpu = torch.arange(
        FREE_PAGE_BASE,
        FREE_PAGE_BASE + backing_len,
        dtype=torch.int64,
    )
    free_pages_backing = free_pages_backing_cpu.to(device)
    free_pages = free_pages_backing[
        args.free_page_offset : args.free_page_offset + args.free_pages_len
    ]
    free_pages_cpu = free_pages_backing_cpu[
        args.free_page_offset : args.free_page_offset + args.free_pages_len
    ]

    expected_cpu, num_new_pages = make_reference(
        prefix_len,
        args.extend_len,
        args.page_size,
        last_loc_value,
        free_pages_cpu,
    )
    return Case(
        prefix_lens=torch.tensor([prefix_len], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
        last_loc=torch.tensor([last_loc_value], dtype=torch.int64, device=device),
        free_pages=free_pages,
        expected_cpu=expected_cpu,
        extend_len=args.extend_len,
        num_new_pages=num_new_pages,
        page_size=args.page_size,
    )


def bind_launch(case: Case) -> tuple[Callable[[], None], torch.Tensor]:
    output = torch.empty(
        (case.extend_len,), dtype=torch.int64, device=case.prefix_lens.device
    )
    bs = case.prefix_lens.numel()
    bs_upper = triton.next_power_of_2(bs)

    def launch() -> None:
        alloc_extend_kernel_npu[(bs,)](
            case.prefix_lens,
            case.seq_lens,
            case.last_loc,
            case.free_pages,
            output,
            bs_upper,
            case.page_size,
        )

    return launch, output


def print_case(case: Case, args: argparse.Namespace) -> None:
    print(
        "input_shapes="
        f"{tuple(case.prefix_lens.shape)};{tuple(case.seq_lens.shape)};"
        f"{tuple(case.last_loc.shape)};{tuple(case.free_pages.shape)}"
    )
    print("input_dtypes=INT64;INT64;INT64;INT64")
    print(
        f"prefix_len={args.prefix_len}, extend_len={args.extend_len}, "
        f"seq_len={args.prefix_len + args.extend_len}, page_size={args.page_size}, "
        f"num_new_pages={case.num_new_pages}, "
        f"free_page_offset={args.free_page_offset}"
    )


def run_check(case: Case) -> None:
    launch, output = bind_launch(case)
    launch()
    torch_npu.npu.synchronize()
    actual_cpu = output.cpu()
    if torch.equal(actual_cpu, case.expected_cpu):
        print(f"PASS correctness: output_shape={tuple(output.shape)}")
        return

    mismatch = torch.nonzero(actual_cpu != case.expected_cpu).flatten()
    first = mismatch[:10].tolist()
    details = [
        (index, int(actual_cpu[index]), int(case.expected_cpu[index]))
        for index in first
    ]
    raise AssertionError(
        f"Result mismatch: mismatch_count={mismatch.numel()}, "
        f"first(index, actual, expected)={details}"
    )


def run_profile(case: Case, args: argparse.Namespace) -> None:
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    launch, _ = bind_launch(case)

    # Compile once, then launch the profiled body.  Task Duration(us) from
    # native msprof op is the sole performance authority; no event timing.
    launch()
    torch_npu.npu.synchronize()
    for _ in range(args.iters):
        launch()
    torch_npu.npu.synchronize()
    print(
        f"completed {args.iters} launches after one compile launch; "
        f"kernel_name={KERNEL_NAME}, device={args.device}"
    )
    print(
        "Read Task Duration(us) from msprof op output using "
        f"--kernel-name={KERNEL_NAME}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check/profile the framework NPU alloc_extend Triton kernel"
    )
    parser.add_argument("--mode", choices=("check", "profile"), default="check")
    parser.add_argument("--device", default="npu:5")
    parser.add_argument("--prefix-len", type=int, default=65)
    parser.add_argument("--extend-len", type=int, default=16384)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--free-pages-len", type=int, default=DEFAULT_FREE_PAGES_LEN)
    parser.add_argument(
        "--free-page-offset",
        type=int,
        default=0,
        help="Slice backing storage by N int64 elements; visible shape stays [17658]",
    )
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = set_device(args.device)
    case = make_case(args, device)
    print_case(case, args)
    if args.mode == "check":
        run_check(case)
    else:
        run_profile(case, args)


if __name__ == "__main__":
    main()
