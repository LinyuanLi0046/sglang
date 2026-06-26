from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.hardware_backend.npu.triton import (
    init_replay_block_tables_npu_triton,
)


def _parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("empty integer list")
    return items


def _seq_len_offset(mode: str, draft_tokens: int, draft_step: int) -> int:
    if mode == "none":
        return 0
    if mode == "target_verify":
        return draft_tokens
    if mode == "draft_decode":
        return draft_step + 1
    raise ValueError(f"unsupported mode: {mode}")


def _compute_max_seq_pages(seq_lens_cpu: torch.Tensor, page_size: int, offset: int) -> int:
    max_len = int(seq_lens_cpu.max().item()) + offset
    return (max_len + page_size - 1) // page_size


def _baseline_pytorch(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_tables: torch.Tensor,
    out_seq_lens: torch.Tensor,
    bs: int,
    page_size: int,
    seq_len_offset: int,
) -> int:
    max_seq_pages = _compute_max_seq_pages(seq_lens_cpu[:bs], page_size, seq_len_offset)
    max_len = max_seq_pages * page_size

    if max_seq_pages > 0:
        block_tables[:bs, :max_seq_pages].copy_(
            req_to_token[req_pool_indices[:bs], :max_len][:, ::page_size] // page_size
        )
    block_tables[:bs, max_seq_pages:].fill_(0)
    out_seq_lens[:bs].copy_(seq_lens[:bs] + seq_len_offset)
    return max_seq_pages


def _triton_impl(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_tables: torch.Tensor,
    out_seq_lens: torch.Tensor,
    bs: int,
    page_size: int,
    seq_len_offset: int,
) -> int:
    max_seq_pages = _compute_max_seq_pages(seq_lens_cpu[:bs], page_size, seq_len_offset)
    init_replay_block_tables_npu_triton(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        block_tables=block_tables,
        out_seq_lens=out_seq_lens,
        bs=bs,
        max_seq_pages=max_seq_pages,
        page_size=page_size,
        seq_len_offset=seq_len_offset,
    )
    return max_seq_pages


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _make_case(
    bs: int,
    req_pool_size: int,
    capture_pages: int,
    active_pages: int,
    page_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(seed)

    max_context_len = capture_pages * page_size
    kv_pool_pages = max(capture_pages * 8, 512)
    kv_pool_tokens = kv_pool_pages * page_size

    req_to_token_cpu = torch.randint(
        0,
        kv_pool_tokens,
        (req_pool_size, max_context_len),
        dtype=torch.int32,
        generator=g_cpu,
    )
    req_pool_indices = torch.randperm(req_pool_size, generator=g_cpu)[:bs].to(torch.int32)
    seq_lens_cpu = torch.randint(
        (active_pages - 1) * page_size + 1,
        active_pages * page_size + 1,
        (bs,),
        dtype=torch.int32,
        generator=g_cpu,
    )
    req_to_token = req_to_token_cpu.to(device="npu")
    seq_lens = seq_lens_cpu.to(device="npu")
    return req_to_token, req_pool_indices.to("npu"), seq_lens, seq_lens_cpu


def _check_correctness(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    bs: int,
    capture_pages: int,
    page_size: int,
    seq_len_offset: int,
) -> int:
    baseline_block_tables = torch.full(
        (bs, capture_pages), -1, dtype=torch.int32, device="npu"
    )
    baseline_seq_lens = torch.full((bs,), -1, dtype=torch.int32, device="npu")
    triton_block_tables = torch.full(
        (bs, capture_pages), -1, dtype=torch.int32, device="npu"
    )
    triton_seq_lens = torch.full((bs,), -1, dtype=torch.int32, device="npu")

    baseline_pages = _baseline_pytorch(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        block_tables=baseline_block_tables,
        out_seq_lens=baseline_seq_lens,
        bs=bs,
        page_size=page_size,
        seq_len_offset=seq_len_offset,
    )
    triton_pages = _triton_impl(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        block_tables=triton_block_tables,
        out_seq_lens=triton_seq_lens,
        bs=bs,
        page_size=page_size,
        seq_len_offset=seq_len_offset,
    )
    torch.npu.synchronize()

    if baseline_pages != triton_pages:
        raise AssertionError(
            f"max_seq_pages mismatch: baseline={baseline_pages}, triton={triton_pages}"
        )
    if not torch.equal(baseline_block_tables, triton_block_tables):
        raise AssertionError("block_tables mismatch between baseline and triton")
    if not torch.equal(baseline_seq_lens, triton_seq_lens):
        raise AssertionError("seq_lens mismatch between baseline and triton")
    return baseline_pages


def _bench_one_case(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    bs: int,
    capture_pages: int,
    page_size: int,
    seq_len_offset: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    baseline_block_tables = torch.empty(
        (bs, capture_pages), dtype=torch.int32, device="npu"
    )
    baseline_seq_lens = torch.empty((bs,), dtype=torch.int32, device="npu")
    triton_block_tables = torch.empty(
        (bs, capture_pages), dtype=torch.int32, device="npu"
    )
    triton_seq_lens = torch.empty((bs,), dtype=torch.int32, device="npu")

    def run_baseline():
        _baseline_pytorch(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_tables=baseline_block_tables,
            out_seq_lens=baseline_seq_lens,
            bs=bs,
            page_size=page_size,
            seq_len_offset=seq_len_offset,
        )

    def run_triton():
        _triton_impl(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_tables=triton_block_tables,
            out_seq_lens=triton_seq_lens,
            bs=bs,
            page_size=page_size,
            seq_len_offset=seq_len_offset,
        )

    baseline_ms = _time_ms(run_baseline, warmup=warmup, iters=iters)
    triton_ms = _time_ms(run_triton, warmup=warmup, iters=iters)
    return baseline_ms, triton_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark replay block-table init: PyTorch baseline vs Triton."
    )
    parser.add_argument("--bs-list", default="1,2,4,8,16,24,32")
    parser.add_argument("--active-pages-list", default="1,4,16,64,128")
    parser.add_argument("--capture-pages", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--req-pool-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=("none", "target_verify", "draft_decode"),
        default="none",
    )
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--draft-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("NPU is required to run this benchmark.")

    bs_list = _parse_int_list(args.bs_list)
    active_pages_list = _parse_int_list(args.active_pages_list)
    seq_len_offset = _seq_len_offset(args.mode, args.draft_tokens, args.draft_step)

    if max(active_pages_list) > args.capture_pages:
        raise ValueError("active-pages-list contains value larger than capture-pages")
    if max(bs_list) > args.req_pool_size:
        raise ValueError("req-pool-size must be >= max(bs-list)")

    torch.set_default_device("npu")
    print(
        f"# mode={args.mode} page_size={args.page_size} capture_pages={args.capture_pages} "
        f"warmup={args.warmup} iters={args.iters} seq_len_offset={seq_len_offset}"
    )
    print(
        "bs active_pages max_seq_pages pytorch_ms triton_ms speedup correctness"
    )

    case_id = 0
    for bs in bs_list:
        for active_pages in active_pages_list:
            case_id += 1
            req_to_token, req_pool_indices, seq_lens, seq_lens_cpu = _make_case(
                bs=bs,
                req_pool_size=args.req_pool_size,
                capture_pages=args.capture_pages,
                active_pages=active_pages,
                page_size=args.page_size,
                seed=args.seed + case_id,
            )

            max_seq_pages = _check_correctness(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                bs=bs,
                capture_pages=args.capture_pages,
                page_size=args.page_size,
                seq_len_offset=seq_len_offset,
            )
            baseline_ms, triton_ms = _bench_one_case(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                bs=bs,
                capture_pages=args.capture_pages,
                page_size=args.page_size,
                seq_len_offset=seq_len_offset,
                warmup=args.warmup,
                iters=args.iters,
            )
            speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
            print(
                f"{bs:2d} {active_pages:12d} {max_seq_pages:13d} "
                f"{baseline_ms:10.4f} {triton_ms:9.4f} {speedup:7.3f}x OK"
            )


if __name__ == "__main__":
    main()
