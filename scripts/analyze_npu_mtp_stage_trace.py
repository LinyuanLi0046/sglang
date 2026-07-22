#!/usr/bin/env python3
"""Compare NPU NEXTN stage traces from normal decode and TARGET_VERIFY.

This is an offline diagnostic.  It never initializes an NPU and only imports
PyTorch when saved ``.pt`` tensors are available for a mismatching checkpoint.

Typical use::

    python scripts/analyze_npu_mtp_stage_trace.py \
      --nomtp-log /path/to/nomtp.log \
      --mtp-log /path/to/mtp.log \
      --nomtp-dump-dir /path/to/nomtp_dump \
      --mtp-dump-dir /path/to/mtp_dump \
      --positions 2876,2877,2878,2938 \
      --report /tmp/mtp_stage_report.txt

The two runs deliberately use different forward modes and token-batch sizes.
Those fields are therefore not part of the alignment key.  Checkpoints are
paired by predicted position, input token, layer, and stage.  ``occurrence`` is
only used to disambiguate retries within one run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


TRACE_TAG = "[SGLANG_NPU_MTP_GREEDY_TRACE]"
STAGE_EVENT = "npu_stage"
SHAPE_EVENT = "npu_stage_shape_mismatch"

TOPK_PRODUCTION_STAGE = "indexer.topk_production"
TOPK_REPEAT_STAGE = "indexer.topk_repeat_same_shape"
TOPK_T1_REPLAY_STAGE = "indexer.topk_t1_replay"
LEGACY_TOPK_PRODUCTION_STAGE = "dsa.indexer_topk"


def _diagnostic_stage_role(stage: str) -> Optional[str]:
    """Map optional shadow-diagnostic stage names to stable semantic roles.

    The exact names above are emitted by the current diagnostic.  Matching the
    final component keeps the analyzer useful if a caller adds a harmless
    namespace prefix.  ``dsa.indexer_topk`` is accepted only as a fallback
    production checkpoint, which lets a newly instrumented MTP log be compared
    with an older no-MTP reference log.
    """

    normalized = stage.strip().lower()
    tail = normalized.rsplit(".", 1)[-1]
    if normalized == LEGACY_TOPK_PRODUCTION_STAGE:
        return "topk_production_legacy"
    if tail == "topk_production":
        return "topk_production"
    if tail == "topk_repeat_same_shape":
        return "topk_repeat_same_shape"
    if tail == "topk_t1_replay":
        return "topk_t1_replay"
    if tail == "history_k_logical":
        return "history_k_logical"
    if tail in {"history_k_scale_logical", "history_scale_logical"}:
        return "history_k_scale_logical"
    return None


def _is_shadow_only_stage(stage: str) -> bool:
    return _diagnostic_stage_role(stage) in {
        "topk_production",
        "topk_repeat_same_shape",
        "topk_t1_replay",
    }


@dataclass(frozen=True)
class TraceEvent:
    source: str
    line_no: int
    order: int
    record: dict[str, Any]

    @property
    def pred_position(self) -> Optional[int]:
        return _optional_int(self.record.get("pred_position"))

    @property
    def input_token(self) -> Optional[int]:
        return _optional_int(self.record.get("input_token"))

    @property
    def layer_id(self) -> Optional[int]:
        return _optional_int(self.record.get("layer_id"))

    @property
    def stage(self) -> str:
        return str(self.record.get("stage", ""))

    @property
    def occurrence(self) -> int:
        return int(self.record.get("occurrence", 0))

    @property
    def fingerprint(self) -> dict[str, Any]:
        value = self.record.get("fingerprint")
        return value if isinstance(value, dict) else {}

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        # forward_mode, tokens_in_forward, physical rank, and row_in_forward
        # intentionally differ between normal decode and TARGET_VERIFY.
        return (
            self.pred_position,
            self.input_token,
            self.layer_id,
            self.stage,
        )

    @property
    def relaxed_key(self) -> tuple[Any, ...]:
        # Used only to explain an unmatched row after token state has diverged.
        return (
            self.pred_position,
            self.layer_id,
            self.stage,
        )


@dataclass
class ParsedLog:
    path: str
    stage_events: list[TraceEvent]
    shape_events: list[TraceEvent]
    event_counts: dict[str, int]
    malformed_trace_lines: list[int]


@dataclass
class TensorMetrics:
    available: bool
    reason: Optional[str] = None
    reference_path: Optional[str] = None
    candidate_path: Optional[str] = None
    shape_equal: Optional[bool] = None
    dtype_reference: Optional[str] = None
    dtype_candidate: Optional[str] = None
    exact_equal: Optional[bool] = None
    allclose: Optional[bool] = None
    changed_count: Optional[int] = None
    changed_fraction: Optional[float] = None
    max_abs: Optional[float] = None
    mean_abs: Optional[float] = None
    rmse: Optional[float] = None
    relative_l2: Optional[float] = None
    cosine: Optional[float] = None
    max_relative: Optional[float] = None
    nonfinite_mismatch_count: Optional[int] = None
    max_abs_flat_index: Optional[int] = None
    max_abs_index: Optional[list[int]] = None
    reference_at_max: Optional[float | int] = None
    candidate_at_max: Optional[float | int] = None
    top_differences: Optional[list[dict[str, Any]]] = None
    same_order: Optional[bool] = None
    same_multiset: Optional[bool] = None
    set_overlap_count: Optional[int] = None
    set_jaccard: Optional[float] = None


@dataclass
class Comparison:
    reference: Optional[TraceEvent]
    candidate: Optional[TraceEvent]
    status: str
    tensor_metrics: Optional[TensorMetrics] = None
    note: Optional[str] = None


@dataclass
class LogicalTensorDifference:
    """First differing logical-token coordinate in a saved history tensor."""

    available: bool
    reason: Optional[str] = None
    reference_path: Optional[str] = None
    candidate_path: Optional[str] = None
    reference_shape: Optional[list[int]] = None
    candidate_shape: Optional[list[int]] = None
    logical_token_axis: int = 0
    common_logical_tokens: Optional[int] = None
    changed_logical_tokens: Optional[int] = None
    first_logical_token: Optional[int] = None
    first_element_flat_index: Optional[int] = None
    first_index: Optional[list[int]] = None
    reference_value: Optional[float | int] = None
    candidate_value: Optional[float | int] = None
    abs_diff: Optional[float] = None


@dataclass
class DiagnosticComparison:
    """A same-run shadow replay or cross-run history/replay comparison."""

    category: str
    left_side: str
    right_side: str
    left: TraceEvent
    right: TraceEvent
    status: str
    tensor_metrics: Optional[TensorMetrics] = None
    logical_difference: Optional[LogicalTensorDifference] = None
    note: Optional[str] = None


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _parse_int_ranges(raw: str) -> Optional[set[int]]:
    if not raw.strip():
        return None
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            values.add(int(part))
            continue
        start_text, end_text = part.split("-", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"descending range is invalid: {part!r}")
        values.update(range(start, end + 1))
    return values


def _extract_trace_json(line: str) -> Optional[dict[str, Any]]:
    marker = line.find(TRACE_TAG)
    if marker < 0:
        return None
    start = line.find("{", marker + len(TRACE_TAG))
    if start < 0:
        raise ValueError("trace marker has no JSON object")
    record, _ = json.JSONDecoder().raw_decode(line[start:])
    if not isinstance(record, dict):
        raise ValueError("trace JSON is not an object")
    return record


def parse_log(
    path: Path,
    *,
    positions: Optional[set[int]],
    layers: Optional[set[int]],
    label: Optional[str],
    rank: Optional[int],
    attention_dp_rank: Optional[int],
    forward_mode: Optional[str],
) -> ParsedLog:
    stage_events: list[TraceEvent] = []
    shape_events: list[TraceEvent] = []
    event_counts: Counter[str] = Counter()
    malformed: list[int] = []
    trace_order = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if TRACE_TAG not in line:
                continue
            try:
                record = _extract_trace_json(line)
            except (ValueError, json.JSONDecodeError):
                malformed.append(line_no)
                continue
            if record is None:
                continue
            event_name = str(record.get("event", "<missing>"))
            event_counts[event_name] += 1
            if label is not None and str(record.get("label", "")) != label:
                continue
            if rank is not None and _optional_int(record.get("rank")) != rank:
                continue
            if (
                attention_dp_rank is not None
                and _optional_int(record.get("attention_dp_rank"))
                != attention_dp_rank
            ):
                continue
            if (
                forward_mode is not None
                and str(record.get("forward_mode")) != forward_mode
            ):
                continue
            event = TraceEvent(str(path), line_no, trace_order, record)
            trace_order += 1
            if event_name == STAGE_EVENT:
                if positions is not None and event.pred_position not in positions:
                    continue
                if layers is not None and event.layer_id not in layers:
                    continue
                stage_events.append(event)
            elif event_name == SHAPE_EVENT:
                if layers is not None and event.layer_id not in layers:
                    continue
                shape_events.append(event)
    return ParsedLog(
        path=str(path),
        stage_events=stage_events,
        shape_events=shape_events,
        event_counts=dict(event_counts),
        malformed_trace_lines=malformed,
    )


def _sha(event: TraceEvent) -> Optional[str]:
    value = event.fingerprint.get("sha256")
    return str(value) if value is not None else None


def _shape(event: TraceEvent) -> Optional[list[int]]:
    value = event.fingerprint.get("shape")
    return list(value) if isinstance(value, list) else None


def _source_dtype(event: TraceEvent) -> Optional[str]:
    value = event.fingerprint.get("source_dtype")
    return str(value) if value is not None else None


def align_events(
    reference: list[TraceEvent], candidate: list[TraceEvent]
) -> tuple[list[Comparison], int, int]:
    reference_groups: dict[tuple[Any, ...], list[TraceEvent]] = defaultdict(list)
    candidate_groups: dict[tuple[Any, ...], list[TraceEvent]] = defaultdict(list)
    for event in reference:
        reference_groups[event.semantic_key].append(event)
    for event in candidate:
        candidate_groups[event.semantic_key].append(event)

    candidate_relaxed: dict[tuple[Any, ...], list[TraceEvent]] = defaultdict(list)
    for event in candidate:
        candidate_relaxed[event.relaxed_key].append(event)

    comparisons: list[Comparison] = []
    paired_candidate_ids: set[int] = set()
    duplicate_reference = sum(
        max(0, len(events) - 1) for events in reference_groups.values()
    )
    duplicate_candidate = sum(
        max(0, len(events) - 1) for events in candidate_groups.values()
    )

    # Reference log order is the logical checkpoint order for each normal
    # decode token.  Occurrence is deliberately not part of the cross-run key:
    # it is local to a mode/rank.  Prefer the same occurrence when there are
    # retries, but surface the ambiguity instead of silently treating it as a
    # different semantic checkpoint.
    for ref_event in reference:
        key = ref_event.semantic_key
        candidates = [
            event
            for event in candidate_groups.get(key, [])
            if id(event) not in paired_candidate_ids
        ]
        if not candidates:
            token_note = None
            near = candidate_relaxed.get(ref_event.relaxed_key, [])
            if near:
                other_tokens = sorted(
                    {event.input_token for event in near},
                    key=lambda value: (value is None, value),
                )
                token_note = (
                    "same position/layer/stage exists with candidate input_token="
                    f"{other_tokens}; model state had already diverged"
                )
            comparisons.append(
                Comparison(ref_event, None, "missing_candidate", note=token_note)
            )
            continue
        matching_occurrence = [
            event for event in candidates if event.occurrence == ref_event.occurrence
        ]
        cand_event = matching_occurrence[0] if matching_occurrence else candidates[0]
        paired_candidate_ids.add(id(cand_event))
        duplicate_key = (
            len(reference_groups[key]) > 1 or len(candidate_groups[key]) > 1
        )
        if duplicate_key:
            status = "ambiguous_occurrence"
        elif _shape(ref_event) != _shape(cand_event):
            status = "shape_mismatch"
        elif _source_dtype(ref_event) != _source_dtype(cand_event):
            status = "dtype_mismatch"
        elif _sha(ref_event) == _sha(cand_event) and _sha(ref_event) is not None:
            status = "exact"
        else:
            status = "value_mismatch"
        note = None
        if duplicate_key:
            ref_occurrences = [event.occurrence for event in reference_groups[key]]
            cand_occurrences = [event.occurrence for event in candidate_groups[key]]
            note = (
                "duplicate semantic checkpoint; no committed-path inference was "
                f"made (reference occurrences={ref_occurrences}, candidate "
                f"occurrences={cand_occurrences})."
            )
        comparisons.append(Comparison(ref_event, cand_event, status, note=note))

    for cand_event in candidate:
        if id(cand_event) not in paired_candidate_ids:
            comparisons.append(Comparison(None, cand_event, "missing_reference"))
    return comparisons, duplicate_reference, duplicate_candidate


def _event_pair_status(left: TraceEvent, right: TraceEvent) -> str:
    if _shape(left) != _shape(right):
        return "shape_mismatch"
    if _source_dtype(left) != _source_dtype(right):
        return "dtype_mismatch"
    if _sha(left) is not None and _sha(left) == _sha(right):
        return "exact"
    return "value_mismatch"


def _local_diagnostic_key(event: TraceEvent) -> tuple[Any, ...]:
    return (
        event.record.get("label", ""),
        event.record.get("rank"),
        event.record.get("attention_dp_rank"),
        event.record.get("forward_mode"),
        event.layer_id,
        event.pred_position,
        event.input_token,
        event.record.get("row_in_forward"),
        event.occurrence,
    )


def _cross_run_diagnostic_key(event: TraceEvent) -> tuple[Any, ...]:
    return (
        event.pred_position,
        event.input_token,
        event.layer_id,
        event.occurrence,
    )


def _role_events(
    events: list[TraceEvent],
) -> dict[tuple[tuple[Any, ...], str], list[TraceEvent]]:
    grouped: dict[tuple[tuple[Any, ...], str], list[TraceEvent]] = defaultdict(list)
    for event in events:
        role = _diagnostic_stage_role(event.stage)
        if role is None:
            continue
        grouped[(_local_diagnostic_key(event), role)].append(event)
    return grouped


def _preferred_production_event(events: list[TraceEvent]) -> Optional[TraceEvent]:
    if not events:
        return None
    nonlegacy = [
        event
        for event in events
        if _diagnostic_stage_role(event.stage) == "topk_production"
    ]
    return (nonlegacy or events)[0]


def _build_within_run_diagnostics(
    events: list[TraceEvent], side: str
) -> list[DiagnosticComparison]:
    grouped = _role_events(events)
    keys = sorted(
        {key for key, _ in grouped},
        key=lambda key: tuple("" if value is None else str(value) for value in key),
    )
    result: list[DiagnosticComparison] = []
    for key in keys:
        production = _preferred_production_event(
            grouped.get((key, "topk_production"), [])
            + grouped.get((key, "topk_production_legacy"), [])
        )
        if production is None:
            continue
        for role, category in (
            ("topk_repeat_same_shape", "within_run_production_vs_repeat"),
            ("topk_t1_replay", "within_run_production_vs_t1_replay"),
        ):
            peers = grouped.get((key, role), [])
            if not peers:
                continue
            peer = peers[0]
            note = None
            if len(peers) > 1:
                note = f"multiple {role} events matched this row; the first was used"
            result.append(
                DiagnosticComparison(
                    category=category,
                    left_side=side,
                    right_side=side,
                    left=production,
                    right=peer,
                    status=_event_pair_status(production, peer),
                    note=note,
                )
            )
    return result


def _cross_role_map(
    events: list[TraceEvent], roles: set[str]
) -> dict[tuple[Any, ...], list[TraceEvent]]:
    grouped: dict[tuple[Any, ...], list[TraceEvent]] = defaultdict(list)
    for event in events:
        if _diagnostic_stage_role(event.stage) in roles:
            grouped[_cross_run_diagnostic_key(event)].append(event)
    return grouped


def build_diagnostic_comparisons(
    reference: list[TraceEvent], candidate: list[TraceEvent]
) -> list[DiagnosticComparison]:
    """Build optional shadow-replay and logical-history comparisons.

    These stages are deliberately analyzed outside normal stage alignment:
    repeat/replay checkpoints commonly exist only in the MTP run and should not
    be reported as ordinary ``missing_reference`` events.
    """

    result = _build_within_run_diagnostics(reference, "reference")
    result.extend(_build_within_run_diagnostics(candidate, "candidate"))

    ref_production = _cross_role_map(
        reference, {"topk_production", "topk_production_legacy"}
    )
    candidate_replay = _cross_role_map(candidate, {"topk_t1_replay"})
    for key in sorted(
        set(ref_production) & set(candidate_replay),
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        left = _preferred_production_event(ref_production[key])
        assert left is not None
        right = candidate_replay[key][0]
        result.append(
            DiagnosticComparison(
                category="cross_run_reference_production_vs_candidate_t1_replay",
                left_side="reference",
                right_side="candidate",
                left=left,
                right=right,
                status=_event_pair_status(left, right),
            )
        )

    for role in ("history_k_logical", "history_k_scale_logical"):
        ref_history = _cross_role_map(reference, {role})
        candidate_history = _cross_role_map(candidate, {role})
        for key in sorted(
            set(ref_history) & set(candidate_history),
            key=lambda item: tuple(
                "" if value is None else str(value) for value in item
            ),
        ):
            left = ref_history[key][0]
            right = candidate_history[key][0]
            result.append(
                DiagnosticComparison(
                    category=f"cross_run_{role}",
                    left_side="reference",
                    right_side="candidate",
                    left=left,
                    right=right,
                    status=_event_pair_status(left, right),
                )
            )

    return sorted(
        result,
        key=lambda item: (
            item.left.pred_position if item.left.pred_position is not None else -1,
            item.left.layer_id if item.left.layer_id is not None else -1,
            item.left.record.get("row_in_forward", -1),
            item.category,
        ),
    )


class DumpResolver:
    def __init__(self, root: Optional[Path]) -> None:
        self.root = root
        self._by_name: Optional[dict[str, list[Path]]] = None

    def _build_name_index(self) -> dict[str, list[Path]]:
        if self._by_name is None:
            index: dict[str, list[Path]] = defaultdict(list)
            if self.root is not None and self.root.exists():
                for path in self.root.rglob("*.pt"):
                    index[path.name].append(path)
            self._by_name = index
        return self._by_name

    def resolve(self, event: TraceEvent) -> tuple[Optional[Path], Optional[str]]:
        logged = event.record.get("tensor_dump")
        if logged:
            direct = Path(str(logged))
            if direct.is_file():
                return direct, None
            matches = self._build_name_index().get(direct.name, [])
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                return None, f"multiple files named {direct.name!r} under {self.root}"

        if self.root is None:
            return None, "no usable tensor_dump path and no dump directory was supplied"
        if not self.root.exists():
            return None, f"dump directory does not exist: {self.root}"

        # Fallback for logs moved away from their original absolute paths.
        pattern = (
            f"*_stage_rank*_mode*_layer{event.layer_id}_"
            f"{_safe_name(event.stage)}_pred{event.pred_position}_"
            f"tok{event.input_token}_occ{event.occurrence}.pt"
        )
        matches = list(self.root.rglob(pattern))
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, f"no dump matched {pattern!r} under {self.root}"
        return None, f"multiple dumps matched {pattern!r} under {self.root}"


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:100]


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "PyTorch is required to compare .pt tensors; run this script in "
            "the same Python environment as SGLang, or omit dump directories"
        ) from exc
    return torch


def _load_saved_tensor(path: Path):
    torch = _import_torch()
    kwargs = {"map_location": "cpu"}
    try:
        payload = torch.load(path, weights_only=True, **kwargs)
    except TypeError:  # torch versions before weights_only was added
        payload = torch.load(path, **kwargs)
    if isinstance(payload, dict):
        tensor = payload.get("tensor")
        if tensor is None:
            tensor = payload.get("logits")
    else:
        tensor = payload
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor or logits tensor")
    return tensor.detach().cpu().contiguous()


def _flat_index_to_index(flat_index: int, shape: Iterable[int]) -> list[int]:
    result: list[int] = []
    remainder = flat_index
    dims = list(shape)
    for dim in reversed(dims):
        result.append(remainder % dim)
        remainder //= dim
    return list(reversed(result))


def compare_saved_tensors(
    reference_path: Path,
    candidate_path: Path,
    *,
    atol: float,
    rtol: float,
    top_elements: int,
) -> TensorMetrics:
    torch = _import_torch()
    try:
        reference = _load_saved_tensor(reference_path)
        candidate = _load_saved_tensor(candidate_path)
    except Exception as exc:
        return TensorMetrics(
            available=False,
            reason=f"failed to load tensor: {type(exc).__name__}: {exc}",
            reference_path=str(reference_path),
            candidate_path=str(candidate_path),
        )

    base = TensorMetrics(
        available=True,
        reference_path=str(reference_path),
        candidate_path=str(candidate_path),
        shape_equal=tuple(reference.shape) == tuple(candidate.shape),
        dtype_reference=str(reference.dtype),
        dtype_candidate=str(candidate.dtype),
    )
    if not base.shape_equal:
        base.reason = "tensor shapes differ: " + (
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
        return base

    base.exact_equal = bool(torch.equal(reference, candidate))
    if reference.numel() == 0:
        base.allclose = base.exact_equal
        base.changed_count = 0
        base.changed_fraction = 0.0
        return base

    is_float = reference.is_floating_point() or candidate.is_floating_point()
    if not is_float:
        reference_i64 = reference.to(torch.int64)
        candidate_i64 = candidate.to(torch.int64)
        changed = reference_i64 != candidate_i64
        changed_count = int(changed.sum().item())
        base.allclose = changed_count == 0
        base.same_order = changed_count == 0
        base.changed_count = changed_count
        base.changed_fraction = changed_count / reference.numel()
        reference_values = reference_i64.reshape(-1).tolist()
        candidate_values = candidate_i64.reshape(-1).tolist()
        base.same_multiset = Counter(reference_values) == Counter(candidate_values)
        reference_set = set(reference_values)
        candidate_set = set(candidate_values)
        union = reference_set | candidate_set
        base.set_overlap_count = len(reference_set & candidate_set)
        base.set_jaccard = (
            len(reference_set & candidate_set) / len(union) if union else 1.0
        )
        if changed_count:
            flat_changed = torch.nonzero(changed.reshape(-1), as_tuple=False).flatten()
            first = int(flat_changed[0].item())
            base.max_abs_flat_index = first
            base.max_abs_index = _flat_index_to_index(first, reference.shape)
            base.reference_at_max = int(reference.reshape(-1)[first].item())
            base.candidate_at_max = int(candidate.reshape(-1)[first].item())
            rows = []
            for flat_index_tensor in flat_changed[:top_elements]:
                flat_index = int(flat_index_tensor.item())
                rows.append(
                    {
                        "flat_index": flat_index,
                        "index": _flat_index_to_index(flat_index, reference.shape),
                        "reference": int(reference.reshape(-1)[flat_index].item()),
                        "candidate": int(candidate.reshape(-1)[flat_index].item()),
                    }
                )
            base.top_differences = rows
        return base

    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    finite_both = torch.isfinite(ref) & torch.isfinite(cand)
    same_nonfinite = (~torch.isfinite(ref)) & (~torch.isfinite(cand)) & (
        (torch.isnan(ref) & torch.isnan(cand)) | (ref == cand)
    )
    nonfinite_mismatch = ~(finite_both | same_nonfinite)
    base.nonfinite_mismatch_count = int(nonfinite_mismatch.sum().item())

    diff = torch.zeros_like(ref)
    diff[finite_both] = cand[finite_both] - ref[finite_both]
    abs_diff = diff.abs()
    tolerance = atol + rtol * ref.abs()
    changed = (finite_both & (abs_diff > tolerance)) | nonfinite_mismatch
    changed_count = int(changed.sum().item())
    base.changed_count = changed_count
    base.changed_fraction = changed_count / ref.numel()
    base.allclose = changed_count == 0

    finite_count = int(finite_both.sum().item())
    if finite_count:
        finite_abs = abs_diff[finite_both]
        base.max_abs = float(finite_abs.max().item())
        base.mean_abs = float(finite_abs.mean().item())
        base.rmse = float(torch.sqrt((diff[finite_both] ** 2).mean()).item())
        ref_l2 = float(torch.linalg.vector_norm(ref[finite_both]).item())
        diff_l2 = float(torch.linalg.vector_norm(diff[finite_both]).item())
        base.relative_l2 = diff_l2 / max(ref_l2, sys.float_info.min)
        denominator = torch.maximum(
            ref[finite_both].abs(),
            torch.full_like(ref[finite_both], sys.float_info.min),
        )
        base.max_relative = float((finite_abs / denominator).max().item())
        ref_vector = ref[finite_both]
        cand_vector = cand[finite_both]
        denom = float(
            (
                torch.linalg.vector_norm(ref_vector)
                * torch.linalg.vector_norm(cand_vector)
            ).item()
        )
        base.cosine = (
            float(torch.dot(ref_vector, cand_vector).item()) / denom
            if denom > 0.0
            else (1.0 if torch.equal(ref_vector, cand_vector) else None)
        )

        ranked_diff = abs_diff.reshape(-1).clone()
        ranked_diff[same_nonfinite.reshape(-1)] = -1.0
        ranked_diff[nonfinite_mismatch.reshape(-1)] = math.inf
        count = min(top_elements, ranked_diff.numel())
        top_values, top_indices = torch.topk(ranked_diff, k=count)
        rows = []
        for difference, flat_index_tensor in zip(top_values, top_indices):
            flat_index = int(flat_index_tensor.item())
            ref_value = ref.reshape(-1)[flat_index]
            cand_value = cand.reshape(-1)[flat_index]
            rows.append(
                {
                    "flat_index": flat_index,
                    "index": _flat_index_to_index(flat_index, reference.shape),
                    "abs_diff": float(difference.item()),
                    "reference": float(ref_value.item()),
                    "candidate": float(cand_value.item()),
                }
            )
        base.top_differences = rows
        max_flat = int(torch.argmax(ranked_diff).item())
        base.max_abs_flat_index = max_flat
        base.max_abs_index = _flat_index_to_index(max_flat, reference.shape)
        base.reference_at_max = float(ref.reshape(-1)[max_flat].item())
        base.candidate_at_max = float(cand.reshape(-1)[max_flat].item())
    return base


def attach_tensor_metrics(
    comparisons: list[Comparison],
    *,
    reference_resolver: DumpResolver,
    candidate_resolver: DumpResolver,
    atol: float,
    rtol: float,
    top_elements: int,
) -> tuple[int, int]:
    compared = 0
    unavailable = 0
    eligible = [
        comparison
        for comparison in comparisons
        if comparison.status
        in {"value_mismatch", "shape_mismatch", "dtype_mismatch"}
        and comparison.reference is not None
        and comparison.candidate is not None
    ]
    attempted = 0
    for comparison in comparisons:
        if comparison.status not in {
            "value_mismatch",
            "shape_mismatch",
            "dtype_mismatch",
        }:
            continue
        if comparison.reference is None or comparison.candidate is None:
            continue
        attempted += 1
        if len(eligible) >= 100 and (attempted == 1 or attempted % 100 == 0):
            print(
                f"loading mismatching tensor pair {attempted}/{len(eligible)}...",
                file=sys.stderr,
            )
        ref_path, ref_error = reference_resolver.resolve(comparison.reference)
        cand_path, cand_error = candidate_resolver.resolve(comparison.candidate)
        if ref_path is None or cand_path is None:
            reasons = [reason for reason in (ref_error, cand_error) if reason]
            comparison.tensor_metrics = TensorMetrics(
                available=False,
                reason="; ".join(reasons),
                reference_path=str(ref_path) if ref_path else None,
                candidate_path=str(cand_path) if cand_path else None,
            )
            unavailable += 1
            continue
        try:
            same_dump = ref_path.resolve() == cand_path.resolve()
        except OSError:
            same_dump = False
        if same_dump:
            comparison.tensor_metrics = TensorMetrics(
                available=False,
                reason=(
                    "reference and candidate resolved to the same .pt file; "
                    "use separate dump directories/labels to avoid overwrite"
                ),
                reference_path=str(ref_path),
                candidate_path=str(cand_path),
            )
            unavailable += 1
            continue
        comparison.tensor_metrics = compare_saved_tensors(
            ref_path,
            cand_path,
            atol=atol,
            rtol=rtol,
            top_elements=top_elements,
        )
        if comparison.tensor_metrics.available:
            compared += 1
        else:
            unavailable += 1
    if len(eligible) >= 100:
        print(
            f"finished tensor-pair pass: compared={compared}, "
            f"unavailable={unavailable}",
            file=sys.stderr,
        )
    return compared, unavailable


def locate_first_logical_tensor_difference(
    reference_path: Path,
    candidate_path: Path,
    *,
    atol: float,
    rtol: float,
    logical_token_axis: int = 0,
) -> LogicalTensorDifference:
    """Locate the first changed logical token and element in two history dumps."""

    torch = _import_torch()
    try:
        reference = _load_saved_tensor(reference_path)
        candidate = _load_saved_tensor(candidate_path)
    except Exception as exc:
        return LogicalTensorDifference(
            available=False,
            reason=f"failed to load history tensor: {type(exc).__name__}: {exc}",
            reference_path=str(reference_path),
            candidate_path=str(candidate_path),
            logical_token_axis=logical_token_axis,
        )

    result = LogicalTensorDifference(
        available=True,
        reference_path=str(reference_path),
        candidate_path=str(candidate_path),
        reference_shape=list(reference.shape),
        candidate_shape=list(candidate.shape),
        logical_token_axis=logical_token_axis,
    )
    if reference.ndim == 0 or candidate.ndim == 0:
        result.available = False
        result.reason = "history tensor is scalar; no logical-token axis exists"
        return result
    axis = logical_token_axis
    if axis < 0:
        axis += reference.ndim
    if (
        axis < 0
        or axis >= reference.ndim
        or reference.ndim != candidate.ndim
        or axis >= candidate.ndim
    ):
        result.available = False
        result.reason = (
            "invalid/incompatible logical-token axis for tensor ranks "
            f"{reference.ndim} and {candidate.ndim}"
        )
        return result

    ref = torch.movedim(reference, axis, 0)
    cand = torch.movedim(candidate, axis, 0)
    ref_tokens, cand_tokens = int(ref.shape[0]), int(cand.shape[0])
    common_tokens = min(ref_tokens, cand_tokens)
    result.common_logical_tokens = common_tokens
    if tuple(ref.shape[1:]) != tuple(cand.shape[1:]):
        result.reason = (
            "per-token tensor shapes differ: "
            f"{tuple(ref.shape[1:])} != {tuple(cand.shape[1:])}"
        )
        result.first_logical_token = 0 if max(ref_tokens, cand_tokens) else None
        return result

    if common_tokens:
        ref_common = ref[:common_tokens]
        cand_common = cand[:common_tokens]
        if ref_common.is_floating_point() or cand_common.is_floating_point():
            ref_values = ref_common.to(torch.float64)
            cand_values = cand_common.to(torch.float64)
            equal = torch.isclose(
                ref_values,
                cand_values,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
        else:
            ref_values = ref_common.to(torch.int64)
            cand_values = cand_common.to(torch.int64)
            equal = ref_values == cand_values
        changed = ~equal
        flat_changed = changed.reshape(common_tokens, -1)
        changed_by_token = flat_changed.any(dim=1)
        result.changed_logical_tokens = int(changed_by_token.sum().item()) + abs(
            ref_tokens - cand_tokens
        )
        changed_tokens = torch.nonzero(changed_by_token, as_tuple=False).flatten()
        if changed_tokens.numel():
            token = int(changed_tokens[0].item())
            element = int(
                torch.nonzero(flat_changed[token], as_tuple=False)[0].item()
            )
            tail_index = _flat_index_to_index(element, ref.shape[1:])
            moved_index = [token, *tail_index]
            original_index: list[int] = []
            tail_iter = iter(tail_index)
            for dim in range(reference.ndim):
                original_index.append(token if dim == axis else next(tail_iter))
            ref_value = ref_values[tuple(moved_index)].item()
            cand_value = cand_values[tuple(moved_index)].item()
            result.first_logical_token = token
            result.first_element_flat_index = element
            result.first_index = original_index
            result.reference_value = ref_value
            result.candidate_value = cand_value
            if isinstance(ref_value, (int, float)) and isinstance(
                cand_value, (int, float)
            ):
                result.abs_diff = abs(float(cand_value) - float(ref_value))
            return result
    else:
        result.changed_logical_tokens = abs(ref_tokens - cand_tokens)

    if ref_tokens != cand_tokens:
        result.first_logical_token = common_tokens
        result.reason = (
            "logical history lengths differ after an equal common prefix: "
            f"{ref_tokens} != {cand_tokens}"
        )
    return result


def attach_diagnostic_tensor_metrics(
    diagnostics: list[DiagnosticComparison],
    *,
    reference_resolver: DumpResolver,
    candidate_resolver: DumpResolver,
    atol: float,
    rtol: float,
    top_elements: int,
) -> tuple[int, int]:
    compared = 0
    unavailable = 0
    resolvers = {
        "reference": reference_resolver,
        "candidate": candidate_resolver,
    }
    for diagnostic in diagnostics:
        if diagnostic.status not in {
            "value_mismatch",
            "shape_mismatch",
            "dtype_mismatch",
        }:
            continue
        left_path, left_error = resolvers[diagnostic.left_side].resolve(
            diagnostic.left
        )
        right_path, right_error = resolvers[diagnostic.right_side].resolve(
            diagnostic.right
        )
        if left_path is None or right_path is None:
            reasons = [
                reason for reason in (left_error, right_error) if reason is not None
            ]
            diagnostic.tensor_metrics = TensorMetrics(
                available=False,
                reason="; ".join(reasons),
                reference_path=str(left_path) if left_path else None,
                candidate_path=str(right_path) if right_path else None,
            )
            unavailable += 1
            continue
        try:
            same_dump = left_path.resolve() == right_path.resolve()
        except OSError:
            same_dump = False
        if same_dump:
            diagnostic.tensor_metrics = TensorMetrics(
                available=False,
                reason="diagnostic stages resolved to the same .pt file",
                reference_path=str(left_path),
                candidate_path=str(right_path),
            )
            unavailable += 1
            continue
        diagnostic.tensor_metrics = compare_saved_tensors(
            left_path,
            right_path,
            atol=atol,
            rtol=rtol,
            top_elements=top_elements,
        )
        if diagnostic.tensor_metrics.available:
            compared += 1
        else:
            unavailable += 1
        if diagnostic.category.startswith("cross_run_history_"):
            axis = _optional_int(
                diagnostic.left.record.get(
                    "history_logical_token_dim",
                    diagnostic.left.record.get("logical_token_dim", 0),
                )
            )
            diagnostic.logical_difference = locate_first_logical_tensor_difference(
                left_path,
                right_path,
                atol=atol,
                rtol=rtol,
                logical_token_axis=axis if axis is not None else 0,
            )
    return compared, unavailable


def _event_name(event: Optional[TraceEvent]) -> str:
    if event is None:
        return "<missing>"
    return (
        f"pos={event.pred_position} tok={event.input_token} "
        f"L{event.layer_id} {event.stage} occ={event.occurrence} "
        f"mode={event.record.get('forward_mode')} line={event.line_no}"
    )


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _metrics_summary(metrics: Optional[TensorMetrics]) -> str:
    if metrics is None:
        return "PT=not requested/found"
    if not metrics.available:
        return f"PT=unavailable ({metrics.reason})"
    if metrics.shape_equal is False:
        return f"PT=shape mismatch ({metrics.reason})"
    if metrics.max_abs is None:
        return (
            f"PT integer changed={metrics.changed_count} "
            f"fraction={_format_number(metrics.changed_fraction)} "
            f"same_multiset={metrics.same_multiset} "
            f"set_jaccard={_format_number(metrics.set_jaccard)} "
            f"first_diff_index={metrics.max_abs_index} "
            f"values={metrics.reference_at_max}->{metrics.candidate_at_max}"
        )
    return (
        f"PT max_abs={_format_number(metrics.max_abs)} "
        f"mean_abs={_format_number(metrics.mean_abs)} "
        f"rel_l2={_format_number(metrics.relative_l2)} "
        f"cos={_format_number(metrics.cosine)} "
        f"changed={metrics.changed_count}/{_format_number(metrics.changed_fraction)} "
        f"max_index={metrics.max_abs_index} "
        f"values={metrics.reference_at_max}->{metrics.candidate_at_max}"
    )


def _logical_difference_summary(
    difference: Optional[LogicalTensorDifference],
) -> str:
    if difference is None:
        return "logical-first-diff=not requested/found"
    if not difference.available:
        return f"logical-first-diff=unavailable ({difference.reason})"
    if difference.first_logical_token is None:
        return (
            "logical history is equal over all saved tokens"
            + (f" ({difference.reason})" if difference.reason else "")
        )
    if difference.first_index is None:
        return (
            f"first differing logical token={difference.first_logical_token}; "
            f"{difference.reason or 'element coordinate unavailable'}"
        )
    return (
        f"first differing logical token={difference.first_logical_token} "
        f"element_flat={difference.first_element_flat_index} "
        f"index={difference.first_index} "
        f"values={difference.reference_value}->{difference.candidate_value} "
        f"abs_diff={_format_number(difference.abs_diff)} "
        f"changed_logical_tokens={difference.changed_logical_tokens}"
    )


def _diagnostic_checkpoint(diagnostic: DiagnosticComparison) -> str:
    return (
        f"pos={diagnostic.left.pred_position} tok={diagnostic.left.input_token} "
        f"L{diagnostic.left.layer_id} row="
        f"{diagnostic.left.record.get('row_in_forward')} "
        f"{diagnostic.left.stage} -> {diagnostic.right.stage}"
    )


def _fingerprint_brief(event: TraceEvent) -> str:
    fingerprint = event.fingerprint
    stats = fingerprint.get("stats", {})
    return (
        f"dtype={fingerprint.get('source_dtype')} "
        f"shape={fingerprint.get('shape')} "
        f"head={fingerprint.get('head_values')} "
        f"sum={stats.get('sum')} abs_sum={stats.get('abs_sum')} "
        f"l2_sq={stats.get('l2_sq')}"
    )


def _execution_context_brief(event: TraceEvent) -> str:
    fields = (
        "rank",
        "attention_dp_rank",
        "forward_mode",
        "row_in_forward",
        "tokens_in_forward",
        "mla_preprocess_used",
        "topk_reused",
        "quant_lightning_indexer",
        "indexer_bs",
        "projected_bs",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
        "block_table_shape",
    )
    return " ".join(
        f"{field}={event.record[field]!r}"
        for field in fields
        if field in event.record
    )


def _stage_hint(stage: str) -> str:
    if stage in {"layer.attn_input", "layer.residual_before_attn"}:
        return (
            "difference already exists on entry; inspect the preceding "
            "layer/checkpoint"
        )
    if stage.startswith("dsa.q_") or stage.startswith("dsa.k_pe"):
        return "attention projection/RoPE path is the first differing region"
    if stage.startswith("indexer.k_cache") or stage.startswith("indexer.k_scale_cache"):
        return "indexer cache scatter/readback or intra-block visibility is implicated"
    if stage.startswith("indexer."):
        return "DSA indexer projection/quantization/operator inputs are implicated"
    if stage == "dsa.indexer_topk":
        return "the two runs select different sparse KV indices at this checkpoint"
    if stage == "dsa.sfa_latent_output":
        return (
            "index selection matched up to here; inspect SFA/KV visibility "
            "and SFA shape path"
        )
    if stage in {"dsa.value_projection_output", "dsa.o_proj_output"}:
        return "post-SFA value/output projection is the first differing region"
    if stage in {"layer.attn_output", "layer.mlp_input", "layer.residual_before_mlp"}:
        return "attention difference is visible before the MoE block"
    if stage in {"layer.mlp_output", "layer.output_hidden", "layer.output_residual"}:
        return "MoE/MXFP4 or residual combination is the first differing region"
    return "inspect this checkpoint and the immediately preceding matching checkpoint"


def _comparison_order(comparison: Comparison) -> tuple[int, int]:
    event = comparison.reference or comparison.candidate
    assert event is not None
    return (event.order, 0 if comparison.reference is not None else 1)


def _material_score(comparison: Comparison) -> float:
    metrics = comparison.tensor_metrics
    if metrics is None or not metrics.available:
        return -1.0
    if metrics.relative_l2 is not None:
        return metrics.relative_l2
    if metrics.changed_fraction is not None:
        return metrics.changed_fraction
    return -1.0


def _stream_summary(events: list[TraceEvent]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            event.record.get("label", ""),
            event.record.get("rank"),
            event.record.get("attention_dp_rank"),
            event.record.get("forward_mode"),
            event.record.get("tokens_in_forward"),
        )
        for event in events
    )


def _format_stream_summary(events: list[TraceEvent]) -> str:
    summary = _stream_summary(events)
    rows = []
    for (label, rank, dp_rank, mode, tokens), count in sorted(
        summary.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        rows.append(
            f"label={label!r}/rank={rank}/dp={dp_rank}/mode={mode}/T={tokens}: {count}"
        )
    return "; ".join(rows) if rows else "<none>"


def _active_stream_count(events: list[TraceEvent]) -> int:
    return len(
        {
            (
                event.record.get("label", ""),
                event.record.get("rank"),
                event.record.get("attention_dp_rank"),
            )
            for event in events
        }
    )


def _functional_metadata_mismatches(
    comparisons: list[Comparison],
) -> list[tuple[Comparison, str, Any, Any]]:
    fields = (
        "is_nextn",
        "is_layer_sparse",
        "mla_preprocess_used",
        "quant_lightning_indexer",
        "topk_reused",
    )
    mismatches: list[tuple[Comparison, str, Any, Any]] = []
    for comparison in comparisons:
        if comparison.reference is None or comparison.candidate is None:
            continue
        for field in fields:
            ref_has = field in comparison.reference.record
            cand_has = field in comparison.candidate.record
            if not ref_has and not cand_has:
                continue
            ref_value = comparison.reference.record.get(field, "<missing>")
            cand_value = comparison.candidate.record.get(field, "<missing>")
            if ref_value != cand_value:
                mismatches.append((comparison, field, ref_value, cand_value))
    return mismatches


def _within_run_cache_checks(
    events: list[TraceEvent],
) -> tuple[int, list[tuple[TraceEvent, TraceEvent, str]]]:
    grouped: dict[tuple[Any, ...], dict[str, list[TraceEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        key = (
            event.record.get("rank"),
            event.pred_position,
            event.input_token,
            event.layer_id,
            event.occurrence,
        )
        grouped[key][event.stage].append(event)

    checked = 0
    violations: list[tuple[TraceEvent, TraceEvent, str]] = []
    stage_pairs = (
        ("indexer.k_to_cache", "indexer.k_cache_readback"),
        ("indexer.k_scale", "indexer.k_scale_cache_readback"),
    )
    for stage_map in grouped.values():
        for source_stage, readback_stage in stage_pairs:
            sources = stage_map.get(source_stage, [])
            readbacks = stage_map.get(readback_stage, [])
            if len(sources) != 1 or len(readbacks) != 1:
                continue
            source, readback = sources[0], readbacks[0]
            checked += 1
            if _shape(source) != _shape(readback):
                violations.append((source, readback, "shape"))
            elif _source_dtype(source) != _source_dtype(readback):
                violations.append((source, readback, "dtype"))
            elif _sha(source) != _sha(readback):
                violations.append((source, readback, "value"))
    return checked, violations


def build_human_report(
    reference: ParsedLog,
    candidate: ParsedLog,
    comparisons: list[Comparison],
    diagnostics: list[DiagnosticComparison],
    *,
    duplicate_reference: int,
    duplicate_candidate: int,
    tensor_compared: int,
    tensor_unavailable: int,
    diagnostic_tensor_compared: int,
    diagnostic_tensor_unavailable: int,
    context: int,
    top: int,
    atol: float,
    rtol: float,
) -> str:
    lines: list[str] = []
    lines.append("NPU MTP stage-trace comparison")
    lines.append("=" * 78)
    lines.append(f"reference : {reference.path}")
    lines.append(f"candidate : {candidate.path}")
    lines.append(
        f"stage events: reference={len(reference.stage_events)} "
        f"candidate={len(candidate.stage_events)}"
    )
    lines.append(f"reference streams: {_format_stream_summary(reference.stage_events)}")
    lines.append(f"candidate streams: {_format_stream_summary(candidate.stage_events)}")
    if _active_stream_count(reference.stage_events) > 1:
        lines.append(
            "WARNING: reference contains multiple label/rank/DP streams; use "
            "--reference-label/--reference-rank/--reference-dp-rank to isolate one."
        )
    if _active_stream_count(candidate.stage_events) > 1:
        lines.append(
            "WARNING: candidate contains multiple label/rank/DP streams; use "
            "--candidate-label/--candidate-rank/--candidate-dp-rank to isolate one."
        )
    reference_modes = {
        event.record.get("forward_mode") for event in reference.stage_events
    }
    candidate_modes = {
        event.record.get("forward_mode") for event in candidate.stage_events
    }
    if len(reference_modes) > 1:
        lines.append(
            f"WARNING: reference has multiple forward modes {reference_modes}; "
            "use --reference-mode to isolate normal decode."
        )
    if len(candidate_modes) > 1:
        lines.append(
            f"WARNING: candidate has multiple forward modes {candidate_modes}; "
            "use --candidate-mode to isolate TARGET_VERIFY."
        )
    null_reference = sum(event.input_token is None for event in reference.stage_events)
    null_candidate = sum(event.input_token is None for event in candidate.stage_events)
    if null_reference or null_candidate:
        lines.append(
            "WARNING: input_token=null events cannot prove trajectory equivalence: "
            f"reference={null_reference}, candidate={null_candidate}"
        )
    lines.append(
        "all trace event counts: "
        f"reference={json.dumps(reference.event_counts, sort_keys=True)} "
        f"candidate={json.dumps(candidate.event_counts, sort_keys=True)}"
    )
    status_counts = Counter(comparison.status for comparison in comparisons)
    lines.append(f"alignment status: {json.dumps(dict(status_counts), sort_keys=True)}")
    lines.append(
        f"duplicate semantic keys: reference={duplicate_reference} "
        f"candidate={duplicate_candidate}"
    )
    if duplicate_reference or duplicate_candidate:
        lines.append(
            "WARNING: duplicate position/token/layer/stage keys were paired by "
            "occurrence when possible. Filter to one run/stream and verify that "
            "the selected occurrence is on the committed trajectory."
        )
    lines.append(
        f"saved tensor comparisons={tensor_compared}, "
        f"unavailable={tensor_unavailable}, allclose tolerance: "
        f"atol={atol:g}, rtol={rtol:g}"
    )
    if diagnostics:
        lines.append(
            f"diagnostic tensor comparisons={diagnostic_tensor_compared}, "
            f"unavailable={diagnostic_tensor_unavailable}"
        )
    if reference.malformed_trace_lines or candidate.malformed_trace_lines:
        lines.append(
            "malformed trace lines: "
            f"reference={reference.malformed_trace_lines[:20]} "
            f"candidate={candidate.malformed_trace_lines[:20]}"
        )
    if reference.shape_events or candidate.shape_events:
        lines.append("")
        lines.append("SHAPE-MISMATCH EVENTS EMITTED BY THE SERVER")
        for label, parsed in (("reference", reference), ("candidate", candidate)):
            for event in parsed.shape_events[:top]:
                lines.append(
                    f"  {label}: L{event.layer_id} {event.stage} line={event.line_no} "
                    f"positions={event.record.get('positions_shape')} "
                    f"tensor={event.record.get('tensor_shape')} "
                    f"token_dim={event.record.get('token_dim')}"
                )
        if len(reference.shape_events) + len(candidate.shape_events) > top * 2:
            lines.append("  ... additional shape events omitted")

    functional_mismatches = _functional_metadata_mismatches(comparisons)
    if functional_mismatches:
        lines.append("")
        lines.append(
            "FUNCTIONAL-METADATA DIFFERENCES "
            "(not normal T=1/T>1 shape metadata)"
        )
        seen: set[tuple[Any, ...]] = set()
        shown = 0
        for comparison, field, ref_value, cand_value in functional_mismatches:
            event = comparison.reference
            key = (
                event.pred_position,
                event.layer_id,
                event.stage,
                field,
                str(ref_value),
                str(cand_value),
            )
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  {_event_name(event)}: {field} reference={ref_value!r} "
                f"candidate={cand_value!r}"
            )
            shown += 1
            if shown >= top:
                break
        if len(seen) < len(functional_mismatches):
            lines.append("  ... additional metadata differences omitted")

    lines.append("")
    lines.append("INDEXER CACHE WRITE/READBACK SELF-CHECK")
    for label, parsed in (("reference", reference), ("candidate", candidate)):
        checked, violations = _within_run_cache_checks(parsed.stage_events)
        lines.append(
            f"  {label}: checked={checked}, mismatched={len(violations)}"
        )
        for source, readback, reason in violations[:top]:
            lines.append(
                f"    {reason}: {_event_name(source)} -> {readback.stage} "
                f"(readback line={readback.line_no})"
            )

    replay_diagnostics = [
        item for item in diagnostics if "topk" in item.left.stage.lower()
        or "topk" in item.right.stage.lower()
    ]
    if replay_diagnostics:
        lines.append("")
        lines.append("INDEXER SAME-SHAPE REPEAT / T=1 SHADOW REPLAY")
        for item in replay_diagnostics:
            lines.append(
                f"  [{item.left_side}->{item.right_side}] "
                f"{_diagnostic_checkpoint(item)}: {item.status}"
            )
            if item.status != "exact":
                lines.append(f"    {_metrics_summary(item.tensor_metrics)}")
            if item.note:
                lines.append(f"    note: {item.note}")

        candidate_repeat: dict[tuple[Any, ...], DiagnosticComparison] = {}
        candidate_replay: dict[tuple[Any, ...], DiagnosticComparison] = {}
        cross_replay: dict[tuple[Any, ...], DiagnosticComparison] = {}
        for item in replay_diagnostics:
            key = _cross_run_diagnostic_key(item.left)
            if (
                item.category == "within_run_production_vs_repeat"
                and item.left_side == "candidate"
            ):
                candidate_repeat[key] = item
            elif (
                item.category == "within_run_production_vs_t1_replay"
                and item.left_side == "candidate"
            ):
                candidate_replay[key] = item
            elif item.category == (
                "cross_run_reference_production_vs_candidate_t1_replay"
            ):
                cross_replay[key] = item
        conclusion_keys = sorted(
            set(candidate_repeat) | set(candidate_replay) | set(cross_replay),
            key=lambda key: tuple(
                "" if value is None else str(value) for value in key
            ),
        )
        for key in conclusion_keys:
            repeat = candidate_repeat.get(key)
            replay = candidate_replay.get(key)
            cross = cross_replay.get(key)
            prefix = f"  AUTO pos={key[0]} tok={key[1]} L{key[2]}: "
            if repeat is not None and repeat.status != "exact":
                lines.append(
                    prefix
                    + "same-shape T>1 repeat changed; prioritize indexer "
                    "non-determinism, stream ordering, or cache visibility."
                )
            elif (
                repeat is not None
                and repeat.status == "exact"
                and replay is not None
                and replay.status != "exact"
                and cross is not None
                and cross.status == "exact"
            ):
                lines.append(
                    prefix
                    + "T>1 production is repeatable, T=1 replay differs from it, "
                    "and replay matches no-MTP: this isolates a T>1 versus T=1 "
                    "LightningIndexer shape-path difference."
                )
            elif (
                repeat is not None
                and repeat.status == "exact"
                and replay is not None
                and replay.status != "exact"
                and cross is not None
                and cross.status != "exact"
            ):
                lines.append(
                    prefix
                    + "T=1 replay does not recover the no-MTP result; inspect "
                    "the saved logical history K/scale before blaming only the "
                    "T>1 kernel shape."
                )
            elif (
                replay is not None
                and replay.status == "exact"
                and cross is not None
                and cross.status != "exact"
            ):
                lines.append(
                    prefix
                    + "candidate T>1 and T=1 agree with each other but differ "
                    "from no-MTP; historical cache/input state is the priority."
                )
            elif all(
                item is None or item.status == "exact"
                for item in (repeat, replay, cross)
            ):
                lines.append(
                    prefix
                    + "all available production/repeat/replay comparisons are exact."
                )

    history_diagnostics = [
        item
        for item in diagnostics
        if item.category.startswith("cross_run_history_")
    ]
    if history_diagnostics:
        lines.append("")
        lines.append("LOGICAL INDEXER HISTORY K/SCALE CROSS-RUN CHECK")
        for item in history_diagnostics:
            lines.append(
                f"  {_diagnostic_checkpoint(item)}: {item.status}"
            )
            if item.status != "exact":
                lines.append(f"    {_metrics_summary(item.tensor_metrics)}")
                lines.append(
                    f"    {_logical_difference_summary(item.logical_difference)}"
                )

    paired = [
        comparison
        for comparison in comparisons
        if comparison.reference is not None and comparison.candidate is not None
    ]
    positions = sorted(
        {
            event.pred_position
            for parsed in (reference, candidate)
            for event in parsed.stage_events
            if event.pred_position is not None
        }
    )
    for position in positions:
        position_comparisons = sorted(
            [
                comparison
                for comparison in comparisons
                if (comparison.reference or comparison.candidate).pred_position
                == position
            ],
            key=_comparison_order,
        )
        position_counts = Counter(item.status for item in position_comparisons)
        lines.append("")
        lines.append(f"POSITION {position}")
        lines.append("-" * 78)
        lines.append(f"status: {json.dumps(dict(position_counts), sort_keys=True)}")

        mismatches = [
            item
            for item in position_comparisons
            if item.status in {"value_mismatch", "shape_mismatch", "dtype_mismatch"}
        ]
        missing = [
            item for item in position_comparisons if item.status.startswith("missing_")
        ]
        ambiguous = [
            item
            for item in position_comparisons
            if item.status == "ambiguous_occurrence"
        ]
        if not mismatches and not missing and not ambiguous:
            lines.append("result: every aligned checkpoint is bitwise identical")
            continue

        if mismatches:
            first = mismatches[0]
            first_index = position_comparisons.index(first)
            preceding = next(
                (
                    item
                    for item in reversed(position_comparisons[:first_index])
                    if item.status == "exact"
                ),
                None,
            )
            lines.append(f"first exact divergence: {_event_name(first.reference)}")
            lines.append(f"  candidate: {_event_name(first.candidate)}")
            lines.append(
                f"  reference context: {_execution_context_brief(first.reference)}"
            )
            lines.append(
                f"  candidate context: {_execution_context_brief(first.candidate)}"
            )
            lines.append(f"  {_metrics_summary(first.tensor_metrics)}")
            if first.tensor_metrics is None or not first.tensor_metrics.available:
                lines.append(
                    f"  reference fingerprint: {_fingerprint_brief(first.reference)}"
                )
                lines.append(
                    f"  candidate fingerprint: {_fingerprint_brief(first.candidate)}"
                )
            lines.append(f"  interpretation: {_stage_hint(first.reference.stage)}")
            if preceding is not None:
                lines.append(
                    "last preceding exact checkpoint: "
                    f"{_event_name(preceding.reference)}"
                )

            discrete = next(
                (
                    item
                    for item in mismatches
                    if (
                        (
                            _source_dtype(item.reference).lower()
                            if _source_dtype(item.reference)
                            else ""
                        )
                        in {
                            "torch.int8",
                            "torch.int16",
                            "torch.int32",
                            "torch.int64",
                            "torch.uint8",
                            "torch.bool",
                        }
                        or "topk" in item.reference.stage
                    )
                ),
                None,
            )
            if discrete is not None:
                lines.append(
                    "first discrete/top-k divergence: "
                    f"{_event_name(discrete.reference)}"
                )
                lines.append(f"  {_metrics_summary(discrete.tensor_metrics)}")

            with_metrics = [item for item in mismatches if _material_score(item) >= 0.0]
            if with_metrics:
                lines.append("largest relative discrepancies with available tensors:")
                for item in sorted(
                    with_metrics, key=_material_score, reverse=True
                )[:top]:
                    lines.append(
                        f"  {_event_name(item.reference)} | "
                        f"{_metrics_summary(item.tensor_metrics)}"
                    )

            lines.append("checkpoint context around the first exact divergence:")
            start = max(0, first_index - context)
            end = min(len(position_comparisons), first_index + context + 1)
            for item in position_comparisons[start:end]:
                marker = "==" if item.status == "exact" else "!!"
                lines.append(
                    f"  {marker} {item.status:18s} "
                    f"{_event_name(item.reference or item.candidate)}"
                )

        if missing:
            lines.append(f"unmatched checkpoints ({len(missing)} total; first {top}):")
            for item in missing[:top]:
                lines.append(
                    f"  {item.status}: {_event_name(item.reference or item.candidate)}"
                    + (f" | {item.note}" if item.note else "")
                )
        if ambiguous:
            lines.append(
                f"ambiguous retry checkpoints ({len(ambiguous)} total; first {top}):"
            )
            for item in ambiguous[:top]:
                lines.append(
                    f"  {_event_name(item.reference)} <-> "
                    f"{_event_name(item.candidate)} | {item.note}"
                )

    if 2876 in positions and 2877 in positions:
        control = [
            item
            for item in comparisons
            if (item.reference or item.candidate).pred_position == 2876
        ]
        next_row = [
            item
            for item in comparisons
            if (item.reference or item.candidate).pred_position == 2877
        ]
        difference_statuses = {
            "value_mismatch",
            "shape_mismatch",
            "dtype_mismatch",
        }
        control_differs = any(item.status in difference_statuses for item in control)
        next_row_differs = any(item.status in difference_statuses for item in next_row)
        control_is_conclusive = control and all(
            item.status == "exact" for item in control
        )
        next_row_is_conclusive = next_row and all(
            item.status == "exact" for item in next_row
        )
        lines.append("")
        lines.append("KNOWN ROW-0 / INTRA-BLOCK CONTROL")
        if control_is_conclusive and next_row_differs:
            lines.append(
                "  pred 2876 (TARGET_VERIFY row 0) is exact, while pred 2877 "
                "differs. This raises the priority of row-dependent intra-block "
                "causal/KV/indexer behavior, but does not exclude a T=1 versus "
                "T=6 kernel whose error depends on row or values."
            )
        elif control_differs:
            lines.append(
                "  pred 2876 already differs. The error is not restricted to "
                "later rows in the verify block. This raises the priority of a "
                "shape-dependent T=1 versus T=6 projection, attention, or MoE "
                "path, but is not proof by itself."
            )
        elif control_is_conclusive and next_row_is_conclusive:
            lines.append("  pred 2876 and 2877 are both exact at every paired stage.")
        else:
            lines.append(
                "  The row-control comparison is inconclusive because checkpoints "
                "are missing or have ambiguous retry occurrences."
            )

    conclusive_mismatches = [
        item
        for item in paired
        if item.status
        in {"value_mismatch", "shape_mismatch", "dtype_mismatch"}
    ]
    has_incomplete_alignment = any(
        item.status.startswith("missing_")
        or item.status == "ambiguous_occurrence"
        for item in comparisons
    )
    if not paired:
        lines.append("")
        lines.append("NO STAGE EVENTS COULD BE ALIGNED.")
        lines.append(
            "Check that both runs used the same prompt and "
            "SGLANG_NPU_MTP_STAGE_TRACE_POSITIONS, and that the reference is "
            "normal decode while the candidate is the MTP run."
        )
    elif not conclusive_mismatches and not has_incomplete_alignment:
        lines.append("")
        lines.append("OVERALL: all paired stage fingerprints are bitwise identical.")
    elif conclusive_mismatches:
        lines.append("")
        lines.append(
            "OVERALL: at least one aligned internal checkpoint differs.  The first "
            "divergence for each position above is the primary localization signal; "
            "later large errors may only be propagation."
        )
    else:
        lines.append("")
        lines.append(
            "OVERALL: no conclusive value mismatch was found among unambiguous "
            "pairs, but missing/retried checkpoints make the comparison incomplete."
        )
    lines.append(
        "CAVEAT: if disabling device graph or adding these synchronizing trace "
        "copies made the output correct, rerun an A/B for graph-specific paths "
        "or multi-stream timing instead of attributing an eager checkpoint."
    )
    return "\n".join(lines) + "\n"


def _event_for_json(event: Optional[TraceEvent]) -> Optional[dict[str, Any]]:
    if event is None:
        return None
    return {
        "source": event.source,
        "line_no": event.line_no,
        "order": event.order,
        "semantic_key": list(event.semantic_key),
        "record": event.record,
    }


def write_json_report(
    path: Path,
    reference: ParsedLog,
    candidate: ParsedLog,
    comparisons: list[Comparison],
    diagnostics: list[DiagnosticComparison],
) -> None:
    payload = {
        "reference": {
            "path": reference.path,
            "event_counts": reference.event_counts,
            "malformed_trace_lines": reference.malformed_trace_lines,
        },
        "candidate": {
            "path": candidate.path,
            "event_counts": candidate.event_counts,
            "malformed_trace_lines": candidate.malformed_trace_lines,
        },
        "comparisons": [
            {
                "status": item.status,
                "note": item.note,
                "reference": _event_for_json(item.reference),
                "candidate": _event_for_json(item.candidate),
                "tensor_metrics": (
                    asdict(item.tensor_metrics) if item.tensor_metrics else None
                ),
            }
            for item in comparisons
        ],
        "diagnostics": [
            {
                "category": item.category,
                "left_side": item.left_side,
                "right_side": item.right_side,
                "status": item.status,
                "note": item.note,
                "left": _event_for_json(item.left),
                "right": _event_for_json(item.right),
                "tensor_metrics": (
                    asdict(item.tensor_metrics) if item.tensor_metrics else None
                ),
                "logical_difference": (
                    asdict(item.logical_difference)
                    if item.logical_difference
                    else None
                ),
            }
            for item in diagnostics
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the first internal tensor divergence between a no-MTP "
            "normal-decode trace and an MTP TARGET_VERIFY trace."
        )
    )
    parser.add_argument(
        "--reference-log",
        "--nomtp-log",
        dest="reference_log",
        type=Path,
        required=True,
        help="server log from the no-MTP reference run",
    )
    parser.add_argument(
        "--candidate-log",
        "--mtp-log",
        dest="candidate_log",
        type=Path,
        required=True,
        help="server log from the MTP run",
    )
    parser.add_argument(
        "--reference-dump-dir",
        "--nomtp-dump-dir",
        dest="reference_dump_dir",
        type=Path,
        help="optional root containing no-MTP .pt dumps",
    )
    parser.add_argument(
        "--candidate-dump-dir",
        "--mtp-dump-dir",
        dest="candidate_dump_dir",
        type=Path,
        help="optional root containing MTP .pt dumps",
    )
    parser.add_argument(
        "--positions",
        default="",
        help="optional predicted positions/ranges, e.g. 2876-2878,2938",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="optional layer ids/ranges, e.g. 0-5,30",
    )
    for side in ("reference", "candidate"):
        alias = "nomtp" if side == "reference" else "mtp"
        parser.add_argument(
            f"--{side}-label",
            f"--{alias}-label",
            dest=f"{side}_label",
            help="optional exact trace label filter (useful for appended logs)",
        )
        parser.add_argument(
            f"--{side}-rank",
            f"--{alias}-rank",
            dest=f"{side}_rank",
            type=int,
            help="optional global-rank filter for a log with multiple active streams",
        )
        parser.add_argument(
            f"--{side}-dp-rank",
            f"--{alias}-dp-rank",
            dest=f"{side}_dp_rank",
            type=int,
            help="optional attention-DP-rank filter",
        )
        parser.add_argument(
            f"--{side}-mode",
            f"--{alias}-mode",
            dest=f"{side}_mode",
            help="optional exact forward_mode filter as printed in JSON",
        )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute tolerance used only for .pt changed-element counts",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="relative tolerance used only for .pt changed-element counts",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="maximum ranked/missing entries shown per position",
    )
    parser.add_argument(
        "--top-elements",
        type=int,
        default=5,
        help="maximum element differences retained in the JSON report",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="checkpoints shown before/after the first divergence",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write the compact text report here (stdout is always printed)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="optionally write all aligned records and tensor metrics as JSON",
    )
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="compare log fingerprints only, even if .pt dumps exist",
    )
    parser.add_argument(
        "--allow-multiple-streams",
        action="store_true",
        help=(
            "allow multiple label/global-rank/DP streams in one side; unsafe "
            "unless their mapping is independently known (for example PP)"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("--atol and --rtol must be non-negative")
    if args.top < 1 or args.top_elements < 1 or args.context < 0:
        raise SystemExit(
            "--top/--top-elements must be positive and --context non-negative"
        )
    for path, label in (
        (args.reference_log, "reference log"),
        (args.candidate_log, "candidate log"),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")

    try:
        positions = _parse_int_ranges(args.positions)
        layers = _parse_int_ranges(args.layers)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    reference = parse_log(
        args.reference_log,
        positions=positions,
        layers=layers,
        label=args.reference_label,
        rank=args.reference_rank,
        attention_dp_rank=args.reference_dp_rank,
        forward_mode=args.reference_mode,
    )
    candidate = parse_log(
        args.candidate_log,
        positions=positions,
        layers=layers,
        label=args.candidate_label,
        rank=args.candidate_rank,
        attention_dp_rank=args.candidate_dp_rank,
        forward_mode=args.candidate_mode,
    )
    mixed_streams = []
    if _active_stream_count(reference.stage_events) > 1:
        mixed_streams.append(
            f"reference: {_format_stream_summary(reference.stage_events)}"
        )
    if _active_stream_count(candidate.stage_events) > 1:
        mixed_streams.append(
            f"candidate: {_format_stream_summary(candidate.stage_events)}"
        )
    if len({event.record.get("forward_mode") for event in reference.stage_events}) > 1:
        mixed_streams.append(
            "reference has multiple forward modes; add --reference-mode"
        )
    if len({event.record.get("forward_mode") for event in candidate.stage_events}) > 1:
        mixed_streams.append(
            "candidate has multiple forward modes; add --candidate-mode"
        )
    if mixed_streams and not args.allow_multiple_streams:
        print(
            "ERROR: stage events contain multiple streams or forward modes. "
            "Automatic cross-context pairing is unsafe. Use the side-specific "
            "--label/--rank/--dp-rank/--mode filters, or pass "
            "--allow-multiple-streams only when the mapping is known.",
            file=sys.stderr,
        )
        for summary in mixed_streams:
            print(f"  {summary}", file=sys.stderr)
        return 3
    diagnostics = build_diagnostic_comparisons(
        reference.stage_events, candidate.stage_events
    )
    # Repeat/replay stages intentionally exist only in a diagnostic run.  Keep
    # them out of normal cross-run alignment so they do not create misleading
    # missing-reference rows in otherwise conclusive reports.
    regular_reference_events = [
        event
        for event in reference.stage_events
        if not _is_shadow_only_stage(event.stage)
    ]
    regular_candidate_events = [
        event
        for event in candidate.stage_events
        if not _is_shadow_only_stage(event.stage)
    ]
    comparisons, duplicate_reference, duplicate_candidate = align_events(
        regular_reference_events, regular_candidate_events
    )

    tensor_compared = 0
    tensor_unavailable = 0
    diagnostic_tensor_compared = 0
    diagnostic_tensor_unavailable = 0
    reference_resolver = DumpResolver(args.reference_dump_dir)
    candidate_resolver = DumpResolver(args.candidate_dump_dir)
    if not args.skip_tensors:
        tensor_compared, tensor_unavailable = attach_tensor_metrics(
            comparisons,
            reference_resolver=reference_resolver,
            candidate_resolver=candidate_resolver,
            atol=args.atol,
            rtol=args.rtol,
            top_elements=args.top_elements,
        )
        (
            diagnostic_tensor_compared,
            diagnostic_tensor_unavailable,
        ) = attach_diagnostic_tensor_metrics(
            diagnostics,
            reference_resolver=reference_resolver,
            candidate_resolver=candidate_resolver,
            atol=args.atol,
            rtol=args.rtol,
            top_elements=args.top_elements,
        )

    report = build_human_report(
        reference,
        candidate,
        comparisons,
        diagnostics,
        duplicate_reference=duplicate_reference,
        duplicate_candidate=duplicate_candidate,
        tensor_compared=tensor_compared,
        tensor_unavailable=tensor_unavailable,
        diagnostic_tensor_compared=diagnostic_tensor_compared,
        diagnostic_tensor_unavailable=diagnostic_tensor_unavailable,
        context=args.context,
        top=args.top,
        atol=args.atol,
        rtol=args.rtol,
    )
    sys.stdout.write(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"compact report written to: {args.report}", file=sys.stderr)
    if args.json_report is not None:
        write_json_report(
            args.json_report,
            reference,
            candidate,
            comparisons,
            diagnostics,
        )
        print(f"JSON report written to: {args.json_report}", file=sys.stderr)

    if not reference.stage_events or not candidate.stage_events:
        print(
            "ERROR: one or both logs contain no selected npu_stage events; see "
            "event counts in the report.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
