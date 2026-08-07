"""One-shot shape logging for WeLMv4 Triton kernels.

The logger intentionally records Python-side tensor metadata only.  It never
reads tensor values and therefore does not synchronize the accelerator.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


_PREFILL_MIN_M = 65536
_LOGGED_KEYS: set[tuple[str, int, str]] = set()
_LOG_LOCK = threading.Lock()


def _get_stage(m: int, stage: Optional[str]) -> str:
    if stage in ("prefill", "decode"):
        return stage

    try:
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch

        if get_is_extend_in_batch():
            return "prefill"
    except Exception:
        # Shape logging must never affect model execution.
        pass

    # Decode cannot realistically have 64K rows.  This fallback also keeps the
    # logger useful in standalone wrapper tests where no ForwardBatch exists.
    return "prefill" if m >= _PREFILL_MIN_M else "decode"


def _tensor_metadata(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _parallel_metadata() -> dict[str, Optional[int]]:
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        return {
            "tp_rank": parallel.tp_rank,
            "pp_rank": parallel.pp_rank,
            "attn_tp_rank": parallel.attn_tp_rank,
            "attn_cp_rank": parallel.attn_cp_rank,
        }
    except Exception:
        return {
            "tp_rank": None,
            "pp_rank": None,
            "attn_tp_rank": None,
            "attn_cp_rank": None,
        }


def log_welmv4_kernel_shapes_once(
    *,
    kernel: str,
    layer_id: Optional[int],
    m: int,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    stage: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one shape record per kernel/layer/prefill-or-decode stage.

    Prefill records are emitted only when ``m >= 65536``.  The output path can
    be overridden with ``SGLANG_WELMV4_KERNEL_SHAPE_LOG``; an empty value
    disables logging.  By default only TP-rank 0 of each PP stage records;
    ``SGLANG_WELMV4_KERNEL_SHAPE_LOG_ALL_RANKS=1`` enables every rank.
    ``os.O_APPEND`` plus one ``os.write`` call keeps each JSON line intact when
    multiple model-worker processes share the file.
    """

    log_path = os.getenv(
        "SGLANG_WELMV4_KERNEL_SHAPE_LOG", "welmv4_kernel_shapes.jsonl"
    )
    if not log_path:
        return

    parallel_metadata = _parallel_metadata()
    log_all_ranks = os.getenv(
        "SGLANG_WELMV4_KERNEL_SHAPE_LOG_ALL_RANKS", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if not log_all_ranks:
        tp_rank = parallel_metadata["tp_rank"]
        if tp_rank is not None and tp_rank != 0:
            return
        if tp_rank is None and _distributed_rank() != 0:
            return

    resolved_stage = _get_stage(m, stage)
    if resolved_stage == "prefill" and m < _PREFILL_MIN_M:
        return

    resolved_layer_id = -1 if layer_id is None else int(layer_id)
    key = (kernel, resolved_layer_id, resolved_stage)
    with _LOG_LOCK:
        if key in _LOGGED_KEYS:
            return
        _LOGGED_KEYS.add(key)

    record = {
        "kernel": kernel,
        "layer_id": resolved_layer_id,
        "stage": resolved_stage,
        "M": int(m),
        "rank": _distributed_rank(),
        "pid": os.getpid(),
        "inputs": {name: _tensor_metadata(value) for name, value in inputs.items()},
        "outputs": {
            name: _tensor_metadata(value) for name, value in outputs.items()
        },
    }
    record.update(parallel_metadata)
    if parameters:
        record["parameters"] = dict(parameters)

    path = Path(log_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        # Debug instrumentation must not make inference fail because the log
        # directory is unavailable or a shared filesystem rejects the write.
        with _LOG_LOCK:
            _LOGGED_KEYS.discard(key)
