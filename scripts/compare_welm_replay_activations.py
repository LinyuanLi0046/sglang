#!/usr/bin/env python3
"""Compare CUDA WeLM dumps with an NPU activation-replay run.

This script extends ``compare_welm_activations.py`` with replay semantics:

* CUDA canonical tensor vs NPU ``*_npu_before_replay`` shows the original
  device difference at the injection boundary.
* CUDA canonical tensor vs NPU canonical tensor verifies that replay actually
  injected the requested CUDA value.
* Canonical tensors produced after the injection boundary show where NPU
  execution first diverges again.

Run one TP rank/pass pair at a time.  Q/K/V dumps are TP-local shards, so the
CUDA and NPU process selectors must refer to the same real TP rank.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import compare_welm_activations as base


BEFORE_SUFFIX = "_npu_before_replay"


def layer_id(name: str) -> Optional[int]:
    match = base.LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def is_expected_injection_name(name: str, point: str) -> bool:
    if point == "embedding":
        return name == "model.layers.0.__input__.0"
    if point == "qkv":
        return name.endswith(
            (
                ".self_attn.q_pre_rope",
                ".self_attn.k_pre_rope",
                ".self_attn.v",
            )
        )
    if point == "attention_input":
        return name.endswith(
            (
                ".self_attn.q_post_rope",
                ".self_attn.k_post_rope",
                ".self_attn.v",
            )
        )
    raise ValueError(f"Unsupported replay point: {point}")


def compare_tensor_pair(
    name: str,
    cuda_path: Optional[Path],
    npu_path: Optional[Path],
    args,
    order: int,
) -> base.ComparisonResult:
    hint = base.diagnostic_hint(name)
    if cuda_path is None or npu_path is None:
        missing_side = "CUDA" if cuda_path is None else "NPU"
        return base.ComparisonResult(
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

    try:
        cuda = base.load_tensor(cuda_path)
        npu = base.load_tensor(npu_path)
    except Exception as exc:
        return base.ComparisonResult(
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

    cuda_shape = tuple(cuda.shape)
    npu_shape = tuple(npu.shape)
    cuda_dtype = str(cuda.dtype).removeprefix("torch.")
    npu_dtype = str(npu.dtype).removeprefix("torch.")
    if cuda_shape != npu_shape:
        return base.ComparisonResult(
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
            cuda_stats=base.collect_stats(cuda, args.chunk_elements),
            npu_stats=base.collect_stats(npu, args.chunk_elements),
            diff=None,
            hint=hint,
        )

    try:
        cuda_stats, npu_stats, diff = base.analyze_same_shape(
            cuda,
            npu,
            args.chunk_elements,
            args.relative_floor,
        )
        status, reasons = base.classify_numeric_result(
            cuda, npu, cuda_stats, npu_stats, diff, args
        )
    except Exception as exc:
        cuda_stats = npu_stats = diff = None
        status = "ERROR"
        reasons = [f"comparison_error: {exc}"]

    return base.ComparisonResult(
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


def exact_replay(result: base.ComparisonResult) -> bool:
    return (
        result.cuda_shape == result.npu_shape
        and result.cuda_dtype == result.npu_dtype
        and result.diff is not None
        and result.diff.equal_fraction == 1.0
        and result.diff.max_abs == 0.0
    )


def metric_text(result: Optional[base.ComparisonResult]) -> str:
    if result is None:
        return "missing"
    if result.diff is None:
        return f"{result.status}({','.join(result.reasons)})"
    return (
        f"{result.status} cos={base.format_float(result.diff.cosine)} "
        f"rel_l2={base.format_float(result.diff.rel_l2)} "
        f"mean_abs={base.format_float(result.diff.mean_abs)} "
        f"max_abs={base.format_float(result.diff.max_abs)} "
        f"equal={base.format_float(result.diff.equal_fraction)}"
    )


def classify_canonical_region(
    name: str,
    point: str,
    injection_names: Sequence[str],
) -> str:
    injection_set = set(injection_names)
    if name in injection_set:
        return "injection_after"

    if point == "embedding":
        anchor = "model.layers.0.__input__.0"
        if base.semantic_order(name) > base.semantic_order(anchor):
            return "post_injection"
        return "upstream"

    current_layer = layer_id(name)
    if current_layer is None:
        return "other"
    same_layer_injections = [
        item for item in injection_names if layer_id(item) == current_layer
    ]
    if not same_layer_injections:
        return "other"
    anchor = max(same_layer_injections, key=base.semantic_order)
    if base.semantic_order(name) > base.semantic_order(anchor):
        return "post_injection_same_layer"
    return "upstream_same_layer"


def replay_semantic_order(
    name: str,
    point: str,
    injection_names: Sequence[str],
) -> Tuple[int, int, int, int, str]:
    """Place all tensors of one replay boundary at the real injection event.

    In particular, attention-input V is captured after RoPE immediately before
    attention, even though its canonical name normally sorts beside raw QKV.
    """
    base_order = base.semantic_order(name)
    if name not in set(injection_names):
        return (*base_order[:3], 1, base_order[3])

    current_layer = layer_id(name)
    same_layer = [item for item in injection_names if layer_id(item) == current_layer]
    event_stage = max(base.semantic_order(item)[2] for item in same_layer)
    suffix_order = {
        "q_pre_rope": 0,
        "k_pre_rope": 1,
        "q_post_rope": 0,
        "k_post_rope": 1,
        ".v": 2,
        "__input__.0": 0,
    }
    within_event = next(
        (order for suffix, order in suffix_order.items() if name.endswith(suffix)),
        9,
    )
    return (base_order[0], base_order[1], event_stage, 0, str(within_event))


def enhanced_rows(
    canonical_results: Sequence[base.ComparisonResult],
    before_results: Dict[str, base.ComparisonResult],
    point: str,
    injection_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    output_order = 0
    for result in canonical_results:
        before = before_results.get(result.name)
        if before is not None:
            row = base.flatten_result(before)
            row["order"] = output_order
            row["comparison_kind"] = "injection_before"
            row["replay_point"] = point
            row["layer"] = layer_id(result.name)
            rows.append(row)
            output_order += 1

        row = base.flatten_result(result)
        row["order"] = output_order
        row["comparison_kind"] = classify_canonical_region(
            result.name, point, injection_names
        )
        row["replay_point"] = point
        row["layer"] = layer_id(result.name)
        rows.append(row)
        output_order += 1
    return rows


def write_enhanced_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "order",
        "comparison_kind",
        "replay_point",
        "layer",
    ] + [
        key
        for key in rows[0]
        if key not in ("order", "comparison_kind", "replay_point", "layer")
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_enhanced_json(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(list(rows), output, indent=2, ensure_ascii=False)


def first_issue(
    results: Sequence[base.ComparisonResult],
    *,
    require_both_tensors: bool = True,
) -> Optional[base.ComparisonResult]:
    return next(
        (
            result
            for result in results
            if result.status in ("WARN", "FAIL", "ERROR")
            and (
                not require_both_tensors
                or (result.cuda_path is not None and result.npu_path is not None)
            )
        ),
        None,
    )


def print_replay_summary(
    point: str,
    injection_names: Sequence[str],
    before_results: Dict[str, base.ComparisonResult],
    canonical_results: Sequence[base.ComparisonResult],
) -> None:
    canonical_map = {result.name: result for result in canonical_results}
    print("\nReplay boundary verification:")
    print("  BEFORE = CUDA canonical vs original NPU value before replacement")
    print("  AFTER  = CUDA canonical vs value actually consumed after replacement")
    all_exact = True
    for name in injection_names:
        before = before_results.get(name)
        after = canonical_map.get(name)
        after_exact = after is not None and exact_replay(after)
        all_exact = all_exact and after_exact
        print(f"\n  {name}")
        print(f"    BEFORE: {metric_text(before)}")
        print(f"    AFTER:  {metric_text(after)} exact={after_exact}")

    print(
        "\nReplay injection check: "
        + ("PASS (all injected tensors are bitwise exact)" if all_exact else "FAIL")
    )

    post_results = [
        result
        for result in canonical_results
        if classify_canonical_region(result.name, point, injection_names)
        in ("post_injection", "post_injection_same_layer")
    ]
    issue = first_issue(post_results)
    if issue is None:
        print(
            "First comparable post-injection divergence: "
            "none crossed configured thresholds"
        )
    else:
        print(
            "First comparable post-injection divergence: "
            f"{issue.name} ({metric_text(issue)})"
        )
        print(f"Inspect: {issue.hint}")

    structural_issue = next(
        (
            result
            for result in post_results
            if result.cuda_path is None or result.npu_path is None
        ),
        None,
    )
    if structural_issue is not None:
        print(
            "First unmatched post-injection dump name (not used as the numeric "
            f"divergence): {structural_issue.name}"
        )

    replay_layers = sorted(
        {
            item
            for item in (layer_id(name) for name in injection_names)
            if item is not None
        }
    )
    if len(replay_layers) > 1:
        print("\nFirst post-injection issue per replayed layer:")
        for current_layer in replay_layers:
            layer_results = [
                result
                for result in post_results
                if layer_id(result.name) == current_layer
            ]
            layer_issue = first_issue(layer_results)
            if layer_issue is None:
                print(f"  layer {current_layer}: none")
            else:
                print(
                    f"  layer {current_layer}: {layer_issue.name} "
                    f"({metric_text(layer_issue)})"
                )


def build_parser():
    parser = base.build_parser()
    parser.description = "Compare CUDA dumps with an NPU WeLM activation-replay run."
    parser.add_argument(
        "--replay-point",
        choices=("embedding", "qkv", "attention_input"),
        help=(
            "Injection point used by SGLANG_WELM_REPLAY_POINT; required unless "
            "--list-passes is used"
        ),
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
        cuda_processes = base.discover_processes(args.cuda_dir)
        npu_processes = base.discover_processes(args.npu_dir)
        if args.list_passes:
            base.print_processes("CUDA", cuda_processes)
            base.print_processes("NPU", npu_processes)
            return 0

        if args.replay_point is None:
            parser.error("--replay-point is required unless --list-passes is used")

        cuda_process_dir, cuda_pass_dirs = base.select_process(
            cuda_processes, args.cuda_process, "CUDA"
        )
        npu_process_dir, npu_pass_dirs = base.select_process(
            npu_processes, args.npu_process, "NPU"
        )
        cuda_passes = [base.describe_pass(path) for path in cuda_pass_dirs]
        npu_passes = [base.describe_pass(path) for path in npu_pass_dirs]
        cuda_pass, npu_pass, warnings = base.select_pass_pair(
            cuda_passes,
            npu_passes,
            args.cuda_pass,
            args.npu_pass,
            args.stage,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"CUDA process: {cuda_process_dir}")
    print(
        f"CUDA pass:    {cuda_pass.path} (stage={cuda_pass.stage}, "
        f"positions=[{cuda_pass.position_min},{cuda_pass.position_max}], "
        f"extend_seq_lens={cuda_pass.extend_seq_lens})"
    )
    print(f"NPU process:  {npu_process_dir}")
    print(
        f"NPU pass:     {npu_pass.path} (stage={npu_pass.stage}, "
        f"positions=[{npu_pass.position_min},{npu_pass.position_max}], "
        f"extend_seq_lens={npu_pass.extend_seq_lens})"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")

    canonical_results = [
        result
        for result in base.compare_passes(cuda_pass.path, npu_pass.path, args)
        if not result.name.endswith(BEFORE_SUFFIX)
    ]
    canonical_map = {result.name: result for result in canonical_results}
    cuda_files = base.tensor_file_map(cuda_pass.path, args.include_weights)
    npu_files = base.tensor_file_map(npu_pass.path, args.include_weights)

    injection_pairs: List[Tuple[str, Path]] = []
    for npu_name, npu_path in npu_files.items():
        if not npu_name.endswith(BEFORE_SUFFIX):
            continue
        canonical_name = npu_name[: -len(BEFORE_SUFFIX)]
        if is_expected_injection_name(canonical_name, args.replay_point):
            if args.name_regex and not re.search(args.name_regex, canonical_name):
                continue
            injection_pairs.append((canonical_name, npu_path))
    injection_pairs.sort(key=lambda item: base.semantic_order(item[0]))
    if not injection_pairs:
        parser.error(
            "No replay boundary tensors were found in the selected NPU pass. "
            "Check --replay-point, SGLANG_DUMP_ACTIVATIONS_LAYER_IDXS, and "
            "pass selection."
        )

    before_results: Dict[str, base.ComparisonResult] = {}
    for order, (canonical_name, npu_before_path) in enumerate(injection_pairs):
        before_results[canonical_name] = compare_tensor_pair(
            canonical_name,
            cuda_files.get(canonical_name),
            npu_before_path,
            args,
            order,
        )
        if canonical_name not in canonical_map:
            canonical_results.append(
                compare_tensor_pair(
                    canonical_name,
                    cuda_files.get(canonical_name),
                    npu_files.get(canonical_name),
                    args,
                    len(canonical_results),
                )
            )

    injection_names = [name for name, _ in injection_pairs]
    canonical_results.sort(
        key=lambda result: replay_semantic_order(
            result.name, args.replay_point, injection_names
        )
    )
    for order, result in enumerate(canonical_results):
        result.order = order
    print_replay_summary(
        args.replay_point,
        injection_names,
        before_results,
        canonical_results,
    )

    base.print_results(canonical_results, args.brief)
    rows = enhanced_rows(
        canonical_results,
        before_results,
        args.replay_point,
        injection_names,
    )
    if args.csv:
        write_enhanced_csv(args.csv, rows)
        print(f"CSV report:  {args.csv.resolve()}")
    if args.json:
        write_enhanced_json(args.json, rows)
        print(f"JSON report: {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
