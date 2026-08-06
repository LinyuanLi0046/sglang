#!/usr/bin/env python3
"""Compare WeLM activation dumps produced by SGLANG_DUMP_ACTIVATIONS.

The dump layout used by ``welmv4.py`` is::

    <root>/TP0_PP0_Rank0_pid<PID>/Pass<NNNNN>/<tensor-name>.pt

This tool compares CUDA and NPU tensors in model execution order, reports
per-side statistics and numerical differences, and highlights the earliest
significant divergence.

Examples:

    # Inspect processes/passes before comparing.
    python scripts/compare_welm_activations.py \
        --cuda-dir /tmp/welm_dump_cuda \
        --npu-dir /tmp/welm_dump_npu \
        --list-passes

    # Auto-select the latest metadata-matched prefill pass.
    python scripts/compare_welm_activations.py \
        --cuda-dir /tmp/welm_dump_cuda \
        --npu-dir /tmp/welm_dump_npu \
        --stage prefill \
        --csv /tmp/welm_activation_compare.csv

    # Select rank/process and pass explicitly.
    python scripts/compare_welm_activations.py \
        --cuda-dir /tmp/welm_dump_cuda \
        --npu-dir /tmp/welm_dump_npu \
        --cuda-process 0 --npu-process 0 \
        --cuda-pass Pass00042 --npu-pass Pass00037 \
        --csv /tmp/welm_activation_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


PASS_RE = re.compile(r"^Pass(\d+)$")
PID_RE = re.compile(r"pid(\d+)")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


@dataclass
class TensorStats:
    numel: int
    finite: int
    nan: int
    pos_inf: int
    neg_inf: int
    mean: Optional[float]
    std: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]


@dataclass
class DiffStats:
    compared: int
    cosine: Optional[float]
    rel_l2: Optional[float]
    mean_abs: Optional[float]
    max_abs: Optional[float]
    rmse: Optional[float]
    mean_rel: Optional[float]
    max_rel: Optional[float]
    equal_fraction: Optional[float]


@dataclass
class PassInfo:
    path: Path
    pass_id: int
    stage: str
    num_files: int
    input_tokens: Optional[int]
    positions_shape: Optional[Tuple[int, ...]]
    position_min: Optional[int]
    position_max: Optional[int]
    extend_seq_lens: Optional[List[int]]
    fingerprint: str


@dataclass
class ComparisonResult:
    order: int
    name: str
    status: str
    reasons: List[str]
    cuda_path: Optional[str]
    npu_path: Optional[str]
    cuda_shape: Optional[Tuple[int, ...]]
    npu_shape: Optional[Tuple[int, ...]]
    cuda_dtype: Optional[str]
    npu_dtype: Optional[str]
    cuda_stats: Optional[TensorStats]
    npu_stats: Optional[TensorStats]
    diff: Optional[DiffStats]
    hint: str


class StatsAccumulator:
    def __init__(self) -> None:
        self.numel = 0
        self.finite = 0
        self.nan = 0
        self.pos_inf = 0
        self.neg_inf = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, raw: torch.Tensor) -> None:
        x = raw.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        self.numel += x.numel()
        if x.numel() == 0:
            return

        nan_mask = torch.isnan(x)
        pos_inf_mask = torch.isposinf(x)
        neg_inf_mask = torch.isneginf(x)
        finite_mask = torch.isfinite(x)
        self.nan += int(nan_mask.sum().item())
        self.pos_inf += int(pos_inf_mask.sum().item())
        self.neg_inf += int(neg_inf_mask.sum().item())

        finite_x = x[finite_mask]
        if finite_x.numel() == 0:
            return
        self.finite += finite_x.numel()
        self.total += float(finite_x.sum().item())
        self.total_sq += float(torch.dot(finite_x, finite_x).item())
        self.minimum = min(self.minimum, float(finite_x.min().item()))
        self.maximum = max(self.maximum, float(finite_x.max().item()))

    def finish(self) -> TensorStats:
        if self.finite == 0:
            mean = std = minimum = maximum = None
        else:
            mean = self.total / self.finite
            variance = max(self.total_sq / self.finite - mean * mean, 0.0)
            std = math.sqrt(variance)
            minimum = self.minimum
            maximum = self.maximum
        return TensorStats(
            numel=self.numel,
            finite=self.finite,
            nan=self.nan,
            pos_inf=self.pos_inf,
            neg_inf=self.neg_inf,
            mean=mean,
            std=std,
            minimum=minimum,
            maximum=maximum,
        )


class DiffAccumulator:
    def __init__(self, relative_floor: float) -> None:
        self.relative_floor = relative_floor
        self.compared = 0
        self.total_abs = 0.0
        self.total_diff_sq = 0.0
        self.total_ref_sq = 0.0
        self.total_npu_sq = 0.0
        self.dot = 0.0
        self.max_abs = 0.0
        self.total_rel = 0.0
        self.max_rel = 0.0
        self.equal = 0
        self.total_for_equal = 0

    def update(self, cuda_raw: torch.Tensor, npu_raw: torch.Tensor) -> None:
        cuda_flat = cuda_raw.detach().reshape(-1).cpu()
        npu_flat = npu_raw.detach().reshape(-1).cpu()
        self.total_for_equal += cuda_flat.numel()

        equal = torch.eq(cuda_flat, npu_flat)
        if cuda_flat.is_floating_point() and npu_flat.is_floating_point():
            equal = equal | (torch.isnan(cuda_flat) & torch.isnan(npu_flat))
        self.equal += int(equal.sum().item())

        cuda = cuda_flat.to(torch.float64)
        npu = npu_flat.to(torch.float64)
        finite_pair = torch.isfinite(cuda) & torch.isfinite(npu)
        cuda = cuda[finite_pair]
        npu = npu[finite_pair]
        if cuda.numel() == 0:
            return

        diff = npu - cuda
        abs_diff = diff.abs()
        rel = abs_diff / cuda.abs().clamp_min(self.relative_floor)
        self.compared += cuda.numel()
        self.total_abs += float(abs_diff.sum().item())
        self.total_diff_sq += float(torch.dot(diff, diff).item())
        self.total_ref_sq += float(torch.dot(cuda, cuda).item())
        self.total_npu_sq += float(torch.dot(npu, npu).item())
        self.dot += float(torch.dot(cuda, npu).item())
        self.max_abs = max(self.max_abs, float(abs_diff.max().item()))
        self.total_rel += float(rel.sum().item())
        self.max_rel = max(self.max_rel, float(rel.max().item()))

    def finish(self) -> DiffStats:
        if self.compared == 0:
            cosine = rel_l2 = mean_abs = max_abs = rmse = None
            mean_rel = max_rel = None
        else:
            ref_norm = math.sqrt(self.total_ref_sq)
            npu_norm = math.sqrt(self.total_npu_sq)
            diff_norm = math.sqrt(self.total_diff_sq)
            if ref_norm == 0.0 and npu_norm == 0.0:
                cosine = 1.0
            elif ref_norm == 0.0 or npu_norm == 0.0:
                cosine = 0.0
            else:
                cosine = max(-1.0, min(1.0, self.dot / (ref_norm * npu_norm)))
            rel_l2 = diff_norm / max(ref_norm, self.relative_floor)
            mean_abs = self.total_abs / self.compared
            max_abs = self.max_abs
            rmse = diff_norm / math.sqrt(self.compared)
            mean_rel = self.total_rel / self.compared
            max_rel = self.max_rel

        equal_fraction = (
            self.equal / self.total_for_equal if self.total_for_equal else 1.0
        )
        return DiffStats(
            compared=self.compared,
            cosine=cosine,
            rel_l2=rel_l2,
            mean_abs=mean_abs,
            max_abs=max_abs,
            rmse=rmse,
            mean_rel=mean_rel,
            max_rel=max_rel,
            equal_fraction=equal_fraction,
        )


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} contains {type(value).__name__}, not a Tensor")
    return value.detach().cpu()


def tensor_chunks(tensor: torch.Tensor, chunk_elements: int) -> Iterable[torch.Tensor]:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        yield flat[start : start + chunk_elements]


def collect_stats(tensor: torch.Tensor, chunk_elements: int) -> TensorStats:
    acc = StatsAccumulator()
    for chunk in tensor_chunks(tensor, chunk_elements):
        acc.update(chunk)
    return acc.finish()


def analyze_same_shape(
    cuda: torch.Tensor,
    npu: torch.Tensor,
    chunk_elements: int,
    relative_floor: float,
) -> Tuple[TensorStats, TensorStats, DiffStats]:
    cuda_acc = StatsAccumulator()
    npu_acc = StatsAccumulator()
    diff_acc = DiffAccumulator(relative_floor)
    cuda_flat = cuda.reshape(-1)
    npu_flat = npu.reshape(-1)
    for start in range(0, cuda_flat.numel(), chunk_elements):
        cuda_chunk = cuda_flat[start : start + chunk_elements]
        npu_chunk = npu_flat[start : start + chunk_elements]
        cuda_acc.update(cuda_chunk)
        npu_acc.update(npu_chunk)
        diff_acc.update(cuda_chunk, npu_chunk)
    return cuda_acc.finish(), npu_acc.finish(), diff_acc.finish()


def discover_processes(root: Path) -> List[Tuple[Path, List[Path]]]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if root.is_dir() and PASS_RE.match(root.name):
        return [(root.parent, [root])]
    # Also accept two manually copied/renamed directories that directly contain
    # the .pt files, without the process/Pass wrapper directories.
    if root.is_dir() and any(root.glob("*.pt")):
        return [(root, [root])]

    pass_dirs = [
        path
        for path in root.rglob("Pass*")
        if path.is_dir() and PASS_RE.match(path.name) and any(path.glob("*.pt"))
    ]
    grouped: Dict[Path, List[Path]] = {}
    for pass_dir in pass_dirs:
        grouped.setdefault(pass_dir.parent, []).append(pass_dir)

    def process_key(item: Tuple[Path, List[Path]]) -> Tuple[int, str]:
        process_dir = item[0]
        match = PID_RE.search(process_dir.name)
        return (int(match.group(1)) if match else sys.maxsize, str(process_dir))

    processes = sorted(grouped.items(), key=process_key)
    for _, passes in processes:
        passes.sort(key=lambda path: int(PASS_RE.match(path.name).group(1)))
    if not processes:
        raise FileNotFoundError(
            f"No non-empty PassNNNNN directories found under {root}"
        )
    return processes


def select_process(
    processes: Sequence[Tuple[Path, List[Path]]], selector: str, side: str
) -> Tuple[Path, List[Path]]:
    if selector.isdigit():
        index = int(selector)
        if index >= len(processes):
            raise ValueError(
                f"{side} process index {index} is out of range; found "
                f"{len(processes)} processes"
            )
        return processes[index]

    matches = [item for item in processes if selector in str(item[0])]
    if len(matches) != 1:
        raise ValueError(
            f"{side} process selector {selector!r} matched {len(matches)} processes"
        )
    return matches[0]


def first_matching_file(pass_dir: Path, suffix: str) -> Optional[Path]:
    matches = sorted(pass_dir.glob(f"*{suffix}.pt"))
    return matches[0] if matches else None


def small_tensor_values(tensor: torch.Tensor) -> List[int]:
    return [int(value) for value in tensor.reshape(-1).to(torch.int64).tolist()]


def metadata_fingerprint(named_tensors: Sequence[Tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        # Metadata consists of token IDs, positions, and sequence lengths. Cast
        # to int64 so an int32/int64 backend difference does not prevent pairing.
        canonical = tensor.detach().cpu().reshape(-1).to(torch.int64)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(repr(canonical.tolist()).encode("ascii"))
    return digest.hexdigest()


def describe_pass(pass_dir: Path) -> PassInfo:
    pass_match = PASS_RE.match(pass_dir.name)
    pass_id = int(pass_match.group(1)) if pass_match else 0
    files = list(pass_dir.glob("*.pt"))
    input_path = pass_dir / "model.oe.input_ids.pt"
    positions_path = first_matching_file(pass_dir, ".self_attn.positions")
    extend_path = first_matching_file(pass_dir, ".self_attn.extend_seq_lens")

    metadata: List[Tuple[str, torch.Tensor]] = []
    input_ids = load_tensor(input_path) if input_path.exists() else None
    positions = load_tensor(positions_path) if positions_path else None
    extend_seq_lens = load_tensor(extend_path) if extend_path else None
    if input_ids is not None:
        metadata.append(("input_ids", input_ids))
    if positions is not None:
        metadata.append(("positions", positions))
    if extend_seq_lens is not None:
        metadata.append(("extend_seq_lens", extend_seq_lens))

    if extend_seq_lens is not None:
        stage = "prefill"
    elif positions is not None or input_ids is not None:
        stage = "decode"
    else:
        stage = "unknown"

    position_values = small_tensor_values(positions) if positions is not None else []
    return PassInfo(
        path=pass_dir,
        pass_id=pass_id,
        stage=stage,
        num_files=len(files),
        input_tokens=input_ids.numel() if input_ids is not None else None,
        positions_shape=tuple(positions.shape) if positions is not None else None,
        position_min=min(position_values) if position_values else None,
        position_max=max(position_values) if position_values else None,
        extend_seq_lens=(
            small_tensor_values(extend_seq_lens)
            if extend_seq_lens is not None
            else None
        ),
        fingerprint=metadata_fingerprint(metadata) if metadata else "",
    )


def print_processes(
    side: str, processes: Sequence[Tuple[Path, List[Path]]]
) -> None:
    print(f"\n{side} processes:")
    for process_index, (process_dir, pass_dirs) in enumerate(processes):
        print(f"  [{process_index}] {process_dir}")
        for pass_dir in pass_dirs:
            info = describe_pass(pass_dir)
            seq_lens = (
                str(info.extend_seq_lens[:8])
                + ("..." if len(info.extend_seq_lens) > 8 else "")
                if info.extend_seq_lens is not None
                else "-"
            )
            print(
                f"      {pass_dir.name}: stage={info.stage:<7} "
                f"files={info.num_files:<4} input_tokens={info.input_tokens!s:<6} "
                f"positions={info.positions_shape} "
                f"range=[{info.position_min},{info.position_max}] "
                f"extend_seq_lens={seq_lens} fp={info.fingerprint[:12]}"
            )


def resolve_pass_selector(passes: Sequence[PassInfo], selector: str) -> PassInfo:
    match = PASS_RE.match(selector)
    if match:
        requested = int(match.group(1))
    elif selector.isdigit():
        requested = int(selector)
    else:
        candidate = Path(selector).resolve()
        path_matches = [info for info in passes if info.path.resolve() == candidate]
        if len(path_matches) == 1:
            return path_matches[0]
        raise ValueError(f"Invalid pass selector: {selector!r}")

    matches = [info for info in passes if info.pass_id == requested]
    if len(matches) != 1:
        raise ValueError(f"Pass {requested} was not found")
    return matches[0]


def latest_for_stage(passes: Sequence[PassInfo], stage: Optional[str]) -> PassInfo:
    candidates = [info for info in passes if stage is None or info.stage == stage]
    if not candidates:
        raise ValueError(f"No pass with stage={stage!r} was found")
    return max(candidates, key=lambda info: info.pass_id)


def latest_matching_pass(
    passes: Sequence[PassInfo], fingerprint: str, stage: Optional[str]
) -> Optional[PassInfo]:
    matches = [
        info
        for info in passes
        if fingerprint
        and info.fingerprint == fingerprint
        and (stage is None or info.stage == stage)
    ]
    return max(matches, key=lambda info: info.pass_id) if matches else None


def select_pass_pair(
    cuda_passes: Sequence[PassInfo],
    npu_passes: Sequence[PassInfo],
    cuda_selector: Optional[str],
    npu_selector: Optional[str],
    stage: str,
) -> Tuple[PassInfo, PassInfo, List[str]]:
    warnings: List[str] = []
    cuda_selected = (
        resolve_pass_selector(cuda_passes, cuda_selector) if cuda_selector else None
    )
    npu_selected = (
        resolve_pass_selector(npu_passes, npu_selector) if npu_selector else None
    )

    desired_stage = None if stage in ("auto", "latest") else stage
    if cuda_selected and not npu_selected:
        npu_selected = latest_matching_pass(
            npu_passes, cuda_selected.fingerprint, desired_stage
        )
        if npu_selected is None:
            npu_selected = latest_for_stage(npu_passes, desired_stage)
            warnings.append("No NPU pass had the selected CUDA metadata fingerprint.")
    elif npu_selected and not cuda_selected:
        cuda_selected = latest_matching_pass(
            cuda_passes, npu_selected.fingerprint, desired_stage
        )
        if cuda_selected is None:
            cuda_selected = latest_for_stage(cuda_passes, desired_stage)
            warnings.append("No CUDA pass had the selected NPU metadata fingerprint.")
    elif not cuda_selected and not npu_selected:
        if stage == "latest":
            cuda_selected = latest_for_stage(cuda_passes, None)
            npu_selected = latest_for_stage(npu_passes, None)
        else:
            stages_to_try = [stage] if stage != "auto" else ["prefill", "decode", None]
            for candidate_stage in stages_to_try:
                cuda_candidates = [
                    info
                    for info in cuda_passes
                    if candidate_stage is None or info.stage == candidate_stage
                ]
                common_pairs: List[Tuple[PassInfo, PassInfo]] = []
                for cuda_info in cuda_candidates:
                    npu_info = latest_matching_pass(
                        npu_passes, cuda_info.fingerprint, candidate_stage
                    )
                    if npu_info is not None:
                        common_pairs.append((cuda_info, npu_info))
                if common_pairs:
                    cuda_selected, npu_selected = max(
                        common_pairs,
                        key=lambda pair: (pair[0].pass_id, pair[1].pass_id),
                    )
                    break
            if cuda_selected is None or npu_selected is None:
                fallback_stage = None if stage == "auto" else stage
                cuda_selected = latest_for_stage(cuda_passes, fallback_stage)
                npu_selected = latest_for_stage(npu_passes, fallback_stage)
                warnings.append(
                    "No common metadata fingerprint was found; selected the latest "
                    "pass independently on each side. Verify that they belong to the "
                    "same request."
                )

    assert cuda_selected is not None and npu_selected is not None
    if (
        cuda_selected.fingerprint
        and npu_selected.fingerprint
        and cuda_selected.fingerprint != npu_selected.fingerprint
    ):
        warnings.append(
            "Selected passes have different input/position/sequence-length metadata."
        )
    return cuda_selected, npu_selected, warnings


LAYER_ORDER = (
    ".__input__.0",
    ".input_layernorm.0",
    ".attn.mixer.1",
    ".self_attn.positions",
    ".self_attn.extend_seq_lens",
    ".self_attn.q_pre_rope",
    ".self_attn.k_pre_rope",
    ".self_attn.v",
    ".self_attn.q_after_norm",
    ".self_attn.k_after_norm",
    ".self_attn.q_post_rope",
    ".self_attn.k_post_rope",
    ".self_attn.attn_output",
    ".attn.router.0",
    ".self_attn.gated_attn_output",
    ".attn.mixer.o_proj_out",
    ".attn.mixer.o_norm_out",
    ".attn.mixer.0",
    ".norm_after_attn.output",
    ".norm_after_attn.output_fp32",
    ".norm_after_attn.residual",
    ".mlp.router.input",
    ".mlp.router.input_fp32",
    ".mlp.router.logits",
    ".mlp.router.scores",
    ".mlp.router.topk_scores",
    ".mlp.router.topk_ids",
    ".mlp.experts_output",
    ".mlp.shared_output",
    ".mlp.output",
    ".mlp.output_with_residual",
)


def semantic_order(name: str) -> Tuple[int, int, int, str]:
    layer_match = LAYER_RE.match(name)
    if not layer_match:
        if name == "model.embed_tokens.output":
            return (0, 0, 0, name)
        if name == "model.oe.input_ids":
            return (0, 1, 0, name)
        if name == "model.oe.base_hidden_states":
            return (0, 2, 0, name)
        gram_match = re.match(r"model\.oe\.gram(\d+)\.ids$", name)
        if gram_match:
            return (0, 3, int(gram_match.group(1)), name)
        vocab_match = re.match(
            r"model\.oe\.vocab(\d+)\.(hashed_ids|embedding)$", name
        )
        if vocab_match:
            vocab_index = int(vocab_match.group(1))
            within_vocab = 0 if vocab_match.group(2) == "hashed_ids" else 1
            return (0, 4, vocab_index * 2 + within_vocab, name)
        if name == "model.oe.projected":
            return (0, 5, 0, name)
        if name == "model.oe.output":
            return (0, 6, 0, name)
        return (0, 7, 0, name)

    layer = int(layer_match.group(1))
    suffix = "." + layer_match.group(2)
    stage = next(
        (index for index, marker in enumerate(LAYER_ORDER) if suffix == marker),
        len(LAYER_ORDER),
    )
    return (1, layer, stage, name)


def diagnostic_hint(name: str) -> str:
    hints = (
        ("embed_tokens.output", "token embedding lookup / embedding weight shard"),
        ("oe.gram", "ngram token-table indexing and history positions"),
        ("oe.vocab", "ngram hashing or ngram embedding lookup"),
        ("oe.projected", "over-encoding projection"),
        ("oe.output", "base embedding + over-encoding merge"),
        ("input_layernorm", "decoder input RMSNorm / residual path"),
        ("q_pre_rope", "QKV projection (Q branch)"),
        ("k_pre_rope", "QKV projection (K branch)"),
        (".self_attn.v", "QKV projection (V branch)"),
        ("q_after_norm", "per-head Q RMSNorm"),
        ("k_after_norm", "per-head K RMSNorm"),
        ("q_post_rope", "RoPE/YARN on Q"),
        ("k_post_rope", "RoPE/YARN on K"),
        ("attn_output", "attention backend, KV cache, mask/window, or attention sink"),
        ("gated_attn_output", "head-wise attention gate"),
        ("attn.router", "head-wise attention gate projection"),
        ("o_proj_out", "attention output projection / TP reduction boundary"),
        ("o_norm_out", "attention output RMSNorm"),
        ("norm_after_attn", "post-attention residual + RMSNorm"),
        ("router.logits", "MoE router linear"),
        ("router.scores", "MoE router sigmoid/softmax"),
        ("router.topk", "MoE routing bias/top-k/normalization"),
        ("experts_output", "fused MoE expert computation and routing weights"),
        ("shared_output", "shared expert and shared-expert gate"),
        ("mlp.output_with_residual", "MoE + residual merge"),
        ("mlp.output", "MoE expert/shared-expert merge or TP reduction"),
    )
    return next(
        (hint for marker, hint in hints if marker in name), "upstream layer state"
    )


def tensor_file_map(pass_dir: Path, include_weights: bool) -> Dict[str, Path]:
    result = {}
    for path in pass_dir.glob("*.pt"):
        name = path.name[:-3]
        if not include_weights and ".__weights__" in name:
            continue
        result[name] = path
    return result


def is_integral_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() and not tensor.is_complex()


def classify_numeric_result(
    cuda: torch.Tensor,
    npu: torch.Tensor,
    cuda_stats: TensorStats,
    npu_stats: TensorStats,
    diff: DiffStats,
    args: argparse.Namespace,
) -> Tuple[str, List[str]]:
    status = "OK"
    reasons: List[str] = []

    nonfinite = (
        cuda_stats.nan
        + cuda_stats.pos_inf
        + cuda_stats.neg_inf
        + npu_stats.nan
        + npu_stats.pos_inf
        + npu_stats.neg_inf
    )
    if nonfinite:
        return "FAIL", ["non_finite_values"]

    if is_integral_tensor(cuda) and is_integral_tensor(npu):
        if diff.equal_fraction != 1.0:
            return "FAIL", ["integer_values_differ"]
    elif diff.compared:
        reference_scale = max(
            abs(cuda_stats.minimum or 0.0),
            abs(cuda_stats.maximum or 0.0),
            abs(npu_stats.minimum or 0.0),
            abs(npu_stats.maximum or 0.0),
        )
        close_by_abs_rel = (diff.max_abs or 0.0) <= (
            args.atol + args.rtol * reference_scale
        )
        metric_ok = (
            diff.cosine is not None
            and diff.rel_l2 is not None
            and diff.cosine >= args.ok_cosine
            and diff.rel_l2 <= args.ok_rel_l2
        )
        metric_warn = (
            diff.cosine is not None
            and diff.rel_l2 is not None
            and diff.cosine >= args.fail_cosine
            and diff.rel_l2 <= args.fail_rel_l2
        )
        if not close_by_abs_rel and not metric_ok:
            reasons.append("numeric_metrics_outside_ok_threshold")
            if metric_warn:
                status = "WARN"
            else:
                status = "FAIL"

    if cuda.dtype != npu.dtype and not args.ignore_dtype:
        reasons.append("dtype_mismatch")
        if status == "OK":
            status = "WARN"
    return status, reasons


def compare_passes(
    cuda_pass: Path, npu_pass: Path, args: argparse.Namespace
) -> List[ComparisonResult]:
    cuda_files = tensor_file_map(cuda_pass, args.include_weights)
    npu_files = tensor_file_map(npu_pass, args.include_weights)
    names = sorted(set(cuda_files) | set(npu_files), key=semantic_order)
    if args.name_regex:
        pattern = re.compile(args.name_regex)
        names = [name for name in names if pattern.search(name)]

    results: List[ComparisonResult] = []
    for order, name in enumerate(names):
        cuda_path = cuda_files.get(name)
        npu_path = npu_files.get(name)
        hint = diagnostic_hint(name)
        if cuda_path is None or npu_path is None:
            missing_side = "CUDA" if cuda_path is None else "NPU"
            results.append(
                ComparisonResult(
                    order=order,
                    name=name,
                    status="FAIL",
                    reasons=[f"missing_on_{missing_side.lower()}"],
                    cuda_path=str(cuda_path) if cuda_path else None,
                    npu_path=str(npu_path) if npu_path else None,
                    cuda_shape=None,
                    npu_shape=None,
                    cuda_dtype=None,
                    npu_dtype=None,
                    cuda_stats=None,
                    npu_stats=None,
                    diff=None,
                    hint=hint,
                )
            )
            continue

        try:
            cuda = load_tensor(cuda_path)
            npu = load_tensor(npu_path)
        except Exception as exc:
            results.append(
                ComparisonResult(
                    order=order,
                    name=name,
                    status="ERROR",
                    reasons=[f"load_error: {exc}"],
                    cuda_path=str(cuda_path),
                    npu_path=str(npu_path),
                    cuda_shape=None,
                    npu_shape=None,
                    cuda_dtype=None,
                    npu_dtype=None,
                    cuda_stats=None,
                    npu_stats=None,
                    diff=None,
                    hint=hint,
                )
            )
            continue

        cuda_shape = tuple(cuda.shape)
        npu_shape = tuple(npu.shape)
        cuda_dtype = str(cuda.dtype).removeprefix("torch.")
        npu_dtype = str(npu.dtype).removeprefix("torch.")
        if cuda_shape != npu_shape:
            cuda_stats = collect_stats(cuda, args.chunk_elements)
            npu_stats = collect_stats(npu, args.chunk_elements)
            results.append(
                ComparisonResult(
                    order=order,
                    name=name,
                    status="FAIL",
                    reasons=["shape_mismatch"],
                    cuda_path=str(cuda_path),
                    npu_path=str(npu_path),
                    cuda_shape=cuda_shape,
                    npu_shape=npu_shape,
                    cuda_dtype=cuda_dtype,
                    npu_dtype=npu_dtype,
                    cuda_stats=cuda_stats,
                    npu_stats=npu_stats,
                    diff=None,
                    hint=hint,
                )
            )
            continue

        try:
            cuda_stats, npu_stats, diff = analyze_same_shape(
                cuda,
                npu,
                args.chunk_elements,
                args.relative_floor,
            )
            status, reasons = classify_numeric_result(
                cuda, npu, cuda_stats, npu_stats, diff, args
            )
        except Exception as exc:
            cuda_stats = npu_stats = diff = None
            status = "ERROR"
            reasons = [f"comparison_error: {exc}"]

        results.append(
            ComparisonResult(
                order=order,
                name=name,
                status=status,
                reasons=reasons,
                cuda_path=str(cuda_path),
                npu_path=str(npu_path),
                cuda_shape=cuda_shape,
                npu_shape=npu_shape,
                cuda_dtype=cuda_dtype,
                npu_dtype=npu_dtype,
                cuda_stats=cuda_stats,
                npu_stats=npu_stats,
                diff=diff,
                hint=hint,
            )
        )
    return results


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6e}"


def shape_text(shape: Optional[Tuple[int, ...]]) -> str:
    return "-" if shape is None else "[" + ",".join(map(str, shape)) + "]"


def print_results(results: Sequence[ComparisonResult], brief: bool) -> None:
    print("\nComparison (semantic model execution order):")
    if brief:
        print(
            f"{'#':>3} {'STATUS':<6} {'COSINE':>12} {'REL_L2':>12} "
            f"{'MEAN_ABS':>12} {'MAX_ABS':>12}  NAME"
        )
    for result in results:
        diff = result.diff
        if brief:
            print(
                f"{result.order:03d} {result.status:<6} "
                f"{format_float(diff.cosine if diff else None):>12} "
                f"{format_float(diff.rel_l2 if diff else None):>12} "
                f"{format_float(diff.mean_abs if diff else None):>12} "
                f"{format_float(diff.max_abs if diff else None):>12}  "
                f"{result.name}"
            )
            continue

        print(f"\n[{result.order:03d}] {result.status:<5} {result.name}")
        print(
            f"      shape CUDA={shape_text(result.cuda_shape)} "
            f"NPU={shape_text(result.npu_shape)}; "
            f"dtype CUDA={result.cuda_dtype or '-'} NPU={result.npu_dtype or '-'}"
        )
        if result.cuda_stats:
            stats = result.cuda_stats
            print(
                "      CUDA "
                f"mean={format_float(stats.mean)} std={format_float(stats.std)} "
                f"min={format_float(stats.minimum)} max={format_float(stats.maximum)} "
                f"nan={stats.nan} inf={stats.pos_inf + stats.neg_inf}"
            )
        if result.npu_stats:
            stats = result.npu_stats
            print(
                "      NPU  "
                f"mean={format_float(stats.mean)} std={format_float(stats.std)} "
                f"min={format_float(stats.minimum)} max={format_float(stats.maximum)} "
                f"nan={stats.nan} inf={stats.pos_inf + stats.neg_inf}"
            )
        if diff:
            print(
                "      DIFF "
                f"cos={format_float(diff.cosine)} rel_l2={format_float(diff.rel_l2)} "
                f"mean_abs={format_float(diff.mean_abs)} "
                f"max_abs={format_float(diff.max_abs)} rmse={format_float(diff.rmse)} "
                f"mean_rel={format_float(diff.mean_rel)} "
                f"equal={format_float(diff.equal_fraction)}"
            )
        if result.reasons:
            print(f"      reason={','.join(result.reasons)}; inspect={result.hint}")


def flatten_result(result: ComparisonResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "order": result.order,
        "status": result.status,
        "name": result.name,
        "reasons": ";".join(result.reasons),
        "hint": result.hint,
        "cuda_shape": shape_text(result.cuda_shape),
        "npu_shape": shape_text(result.npu_shape),
        "cuda_dtype": result.cuda_dtype,
        "npu_dtype": result.npu_dtype,
        "cuda_path": result.cuda_path,
        "npu_path": result.npu_path,
    }
    for side, stats in (("cuda", result.cuda_stats), ("npu", result.npu_stats)):
        for key in TensorStats.__dataclass_fields__:
            row[f"{side}_{key}"] = getattr(stats, key) if stats else None
    for key in DiffStats.__dataclass_fields__:
        row[key] = getattr(result.diff, key) if result.diff else None
    return row


def write_csv(path: Path, results: Sequence[ComparisonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_result(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, results: Sequence[ComparisonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        item = asdict(result)
        item["cuda_shape"] = list(result.cuda_shape) if result.cuda_shape else None
        item["npu_shape"] = list(result.npu_shape) if result.npu_shape else None
        payload.append(item)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)


def first_with_status(
    results: Sequence[ComparisonResult], statuses: Sequence[str]
) -> Optional[ComparisonResult]:
    return next((result for result in results if result.status in statuses), None)


def print_summary(results: Sequence[ComparisonResult]) -> None:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    count_text = ", ".join(
        f"{key}={value}" for key, value in sorted(counts.items())
    )
    print("\nSummary: " + count_text)

    first_issue = first_with_status(results, ("WARN", "FAIL", "ERROR"))
    first_fail = first_with_status(results, ("FAIL", "ERROR"))
    first_numeric = next(
        (
            result
            for result in results
            if result.status in ("WARN", "FAIL", "ERROR")
            and result.reasons != ["dtype_mismatch"]
        ),
        None,
    )

    if first_issue is None:
        print("No issue crossed the configured thresholds.")
        return

    print(
        f"FIRST ISSUE: [{first_issue.order:03d}] {first_issue.name} "
        f"({first_issue.status}: {','.join(first_issue.reasons)})"
    )
    print(f"  Inspect first: {first_issue.hint}")
    if first_numeric and first_numeric is not first_issue:
        print(
            f"FIRST NUMERIC/STRUCTURAL DIVERGENCE: [{first_numeric.order:03d}] "
            f"{first_numeric.name} ({first_numeric.status}: "
            f"{','.join(first_numeric.reasons)})"
        )
        print(f"  Inspect first: {first_numeric.hint}")
    if first_fail and first_fail is not first_issue and first_fail is not first_numeric:
        print(
            f"FIRST HARD FAILURE: [{first_fail.order:03d}] {first_fail.name} "
            f"({','.join(first_fail.reasons)})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CUDA and NPU WeLM activation dump tensors."
    )
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--npu-dir", type=Path, required=True)
    parser.add_argument(
        "--cuda-process",
        default="0",
        help="CUDA process index from --list-passes, or a unique path substring",
    )
    parser.add_argument(
        "--npu-process",
        default="0",
        help="NPU process index from --list-passes, or a unique path substring",
    )
    parser.add_argument("--cuda-pass", help="CUDA pass number/name/path")
    parser.add_argument("--npu-pass", help="NPU pass number/name/path")
    parser.add_argument(
        "--stage",
        choices=("auto", "prefill", "decode", "latest"),
        default="auto",
        help=(
            "Pass selection when explicit passes are omitted. auto prefers the "
            "latest metadata-matched prefill pass."
        ),
    )
    parser.add_argument(
        "--list-passes", action="store_true", help="List discovered processes/passes"
    )
    parser.add_argument("--csv", type=Path, help="Write a detailed CSV report")
    parser.add_argument("--json", type=Path, help="Write a detailed JSON report")
    parser.add_argument(
        "--brief", action="store_true", help="Print a one-line summary per tensor"
    )
    parser.add_argument(
        "--name-regex", help="Only compare tensor names matching this regex"
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Include .__weights__ tensors if weight dumping was enabled",
    )
    parser.add_argument(
        "--ignore-dtype",
        action="store_true",
        help="Do not mark an otherwise matching tensor WARN for dtype mismatch",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--ok-cosine",
        type=float,
        default=0.9999,
        help="Minimum cosine similarity for OK (default: 0.9999)",
    )
    parser.add_argument(
        "--ok-rel-l2",
        type=float,
        default=0.02,
        help="Maximum relative L2 error for OK (default: 0.02)",
    )
    parser.add_argument(
        "--fail-cosine",
        type=float,
        default=0.99,
        help="Below this cosine similarity is FAIL (default: 0.99)",
    )
    parser.add_argument(
        "--fail-rel-l2",
        type=float,
        default=0.10,
        help="Above this relative L2 error is FAIL (default: 0.10)",
    )
    parser.add_argument(
        "--relative-floor",
        type=float,
        default=1e-12,
        help="Denominator floor for relative errors",
    )
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=1_000_000,
        help="Elements processed per CPU chunk (default: 1000000)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.chunk_elements <= 0:
        parser.error("--chunk-elements must be positive")
    if args.fail_cosine > args.ok_cosine:
        parser.error("--fail-cosine must be <= --ok-cosine")
    if args.fail_rel_l2 < args.ok_rel_l2:
        parser.error("--fail-rel-l2 must be >= --ok-rel-l2")

    try:
        cuda_processes = discover_processes(args.cuda_dir)
        npu_processes = discover_processes(args.npu_dir)
        if args.list_passes:
            print_processes("CUDA", cuda_processes)
            print_processes("NPU", npu_processes)
            return 0

        cuda_process_dir, cuda_pass_dirs = select_process(
            cuda_processes, args.cuda_process, "CUDA"
        )
        npu_process_dir, npu_pass_dirs = select_process(
            npu_processes, args.npu_process, "NPU"
        )
        cuda_passes = [describe_pass(path) for path in cuda_pass_dirs]
        npu_passes = [describe_pass(path) for path in npu_pass_dirs]
        cuda_pass, npu_pass, warnings = select_pass_pair(
            cuda_passes,
            npu_passes,
            args.cuda_pass,
            args.npu_pass,
            args.stage,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    if len(cuda_processes) > 1 and args.cuda_process == "0":
        warnings.append(
            f"CUDA root contains {len(cuda_processes)} processes; defaulted to index 0."
        )
    if len(npu_processes) > 1 and args.npu_process == "0":
        warnings.append(
            f"NPU root contains {len(npu_processes)} processes; defaulted to index 0."
        )

    print(f"CUDA process: {cuda_process_dir}")
    print(
        f"CUDA pass:    {cuda_pass.path} (stage={cuda_pass.stage}, "
        f"fp={cuda_pass.fingerprint[:12]})"
    )
    print(f"NPU process:  {npu_process_dir}")
    print(
        f"NPU pass:     {npu_pass.path} (stage={npu_pass.stage}, "
        f"fp={npu_pass.fingerprint[:12]})"
    )
    print(
        "Thresholds: "
        f"OK if cosine>={args.ok_cosine} and rel_l2<={args.ok_rel_l2}; "
        f"FAIL if cosine<{args.fail_cosine} or rel_l2>{args.fail_rel_l2}; "
        "intermediate values are WARN."
    )
    for warning in warnings:
        print(f"WARNING: {warning}")

    results = compare_passes(cuda_pass.path, npu_pass.path, args)
    print_results(results, args.brief)
    print_summary(results)

    if args.csv:
        write_csv(args.csv, results)
        print(f"CSV report:  {args.csv.resolve()}")
    if args.json:
        write_json(args.json, results)
        print(f"JSON report: {args.json.resolve()}")

    # The comparison result itself is the desired output. Do not return a
    # non-zero status merely because tensors differ; this keeps ad-hoc analysis
    # pipelines simple. Argument/load failures still exit via argparse.error().
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
