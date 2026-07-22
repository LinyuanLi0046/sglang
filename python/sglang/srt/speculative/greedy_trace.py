"""Opt-in diagnostics for deterministic NEXTN/normal-decode divergence.

The helpers in this module are intentionally dormant unless
``SGLANG_NPU_MTP_GREEDY_TRACE=1``.  Trace mode synchronizes small tensors to the
host and performs a TP all-gather, so it is for short, single-request repros
only, not performance measurements.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

TRACE_TAG = "[SGLANG_NPU_MTP_GREEDY_TRACE]"


@dataclass(frozen=True)
class LogitsTopKSnapshot:
    token_ids: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class TopK1Reference:
    """CPU reconstruction of the A5 topk=1 greedy verifier output."""

    accept_lens: list[int]
    accept_indices: list[list[int]]
    accept_tokens: list[list[int]]
    predict: list[int]


def build_topk1_reference(
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    target_predict: torch.Tensor,
) -> TopK1Reference:
    """Reproduce ``verify_tree_greedy_kernel`` for a linear topk=1 tree.

    Inputs must be two-dimensional CPU tensors with shape ``[bs, D]``.  The
    returned ``accept_lens`` includes the final target/bonus token, matching the
    value returned by :func:`eagle_sample`.
    """

    if candidates.device.type != "cpu":
        candidates = candidates.cpu()
    if retrieve_index.device.type != "cpu":
        retrieve_index = retrieve_index.cpu()
    if target_predict.device.type != "cpu":
        target_predict = target_predict.cpu()

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2-D, got {tuple(candidates.shape)}")
    if retrieve_index.shape != candidates.shape:
        raise ValueError(
            "retrieve_index shape must match candidates: "
            f"{tuple(retrieve_index.shape)} != {tuple(candidates.shape)}"
        )
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict shape must match candidates: "
            f"{tuple(target_predict.shape)} != {tuple(candidates.shape)}"
        )

    bs, draft_token_num = candidates.shape
    if draft_token_num < 1:
        raise ValueError("draft_token_num must be at least 1")

    total_tokens = bs * draft_token_num
    reference_predict = [0] * total_tokens
    accept_lens: list[int] = []
    accept_indices: list[list[int]] = []
    accept_tokens: list[list[int]] = []

    for batch_index in range(bs):
        base = batch_index * draft_token_num
        last_accepted_idx = int(retrieve_index[batch_index, 0].item())
        if not base <= last_accepted_idx < base + draft_token_num:
            raise ValueError(
                f"retrieve_index[{batch_index}, 0]={last_accepted_idx} is outside "
                f"request row [{base}, {base + draft_token_num})"
            )
        path = [last_accepted_idx]

        for draft_index in range(1, draft_token_num):
            draft_token = int(candidates[batch_index, draft_index].item())
            target_token = int(target_predict[batch_index, draft_index - 1].item())
            if draft_token != target_token:
                break

            reference_predict[last_accepted_idx] = target_token
            last_accepted_idx = int(
                retrieve_index[batch_index, draft_index].item()
            )
            if not base <= last_accepted_idx < base + draft_token_num:
                raise ValueError(
                    f"retrieve_index[{batch_index}, {draft_index}]="
                    f"{last_accepted_idx} is outside request row "
                    f"[{base}, {base + draft_token_num})"
                )
            path.append(last_accepted_idx)

        final_row = last_accepted_idx - base
        reference_predict[last_accepted_idx] = int(
            target_predict[batch_index, final_row].item()
        )
        accept_lens.append(len(path))
        accept_indices.append(path)
        accept_tokens.append([reference_predict[index] for index in path])

    return TopK1Reference(
        accept_lens=accept_lens,
        accept_indices=accept_indices,
        accept_tokens=accept_tokens,
        predict=reference_predict,
    )


def greedy_trace_enabled() -> bool:
    return envs.SGLANG_NPU_MTP_GREEDY_TRACE.get()


def _trace_label() -> str:
    return envs.SGLANG_NPU_MTP_GREEDY_TRACE_LABEL.get()


def _trace_window() -> tuple[int, int]:
    start = max(0, envs.SGLANG_NPU_MTP_GREEDY_TRACE_START_TOKEN.get())
    max_tokens = envs.SGLANG_NPU_MTP_GREEDY_TRACE_MAX_TOKENS.get()
    end = start + max_tokens if max_tokens >= 0 else 1 << 62
    return start, end


def _request_selected(req: "Req", possible_tokens: int = 1) -> bool:
    rid_filter = envs.SGLANG_NPU_MTP_GREEDY_TRACE_RID.get()
    if rid_filter and str(req.rid) != rid_filter:
        return False
    gen_start = len(req.output_ids)
    trace_start, trace_end = _trace_window()
    return gen_start < trace_end and gen_start + max(1, possible_tokens) > trace_start


def _batch_selected(batch: Optional["ScheduleBatch"], possible_tokens: int) -> bool:
    return batch is not None and any(
        _request_selected(req, possible_tokens) for req in batch.reqs
    )


def _trace_group():
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group,
        is_dp_attention_enabled,
    )

    return get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()


def _rank_fields(group=None) -> dict[str, int]:
    if group is None:
        group = _trace_group()
    fields = {
        "rank": int(group.rank),
        "trace_group_rank": int(group.rank_in_group),
        "trace_group_size": int(group.world_size),
    }
    try:
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            is_dp_attention_enabled,
        )

        fields["attention_dp_rank"] = (
            int(get_attention_dp_rank()) if is_dp_attention_enabled() else 0
        )
    except Exception:
        fields["attention_dp_rank"] = -1
    return fields


def _emit(event: str, payload: dict[str, Any]) -> None:
    record = {"event": event, "label": _trace_label(), **payload}
    logger.warning(
        "%s %s",
        TRACE_TAG,
        json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    )


def _topk_snapshot(
    logits: torch.Tensor, *, copy_to_cpu: bool = True
) -> LogitsTopKSnapshot:
    topn = max(1, envs.SGLANG_NPU_MTP_GREEDY_TRACE_TOPN.get())
    topn = min(topn, logits.shape[-1])
    values, token_ids = torch.topk(logits.detach(), k=topn, dim=-1)
    if copy_to_cpu:
        token_ids = token_ids.cpu()
        values = values.float().cpu()
    return LogitsTopKSnapshot(
        token_ids=token_ids,
        values=values,
    )


def capture_raw_logits(
    batch: Optional["ScheduleBatch"],
    logits: Optional[torch.Tensor],
    *,
    possible_tokens_per_req: int,
) -> Optional[LogitsTopKSnapshot]:
    """Capture pre-sampling top-k only on the trace rank.

    Keeping this separate from the final trace lets callers snapshot logits
    before in-place penalty, grammar, bias, or custom-processor updates.
    """

    if (
        not greedy_trace_enabled()
        or logits is None
        or not _batch_selected(batch, possible_tokens_per_req)
    ):
        return None
    if not batch.sampling_info.is_all_greedy:
        return None
    group = _trace_group()
    if group.rank_in_group != 0:
        return None
    valid_rows = len(batch.reqs) * possible_tokens_per_req
    # Keep the snapshot on device until after sampling/verification.  A host
    # copy here would add a pre-decision synchronization and could hide the
    # stream/lifetime race this diagnostic is meant to find.
    return _topk_snapshot(logits[:valid_rows], copy_to_cpu=False)


def _row_topk(
    snapshot: Optional[LogitsTopKSnapshot], row: int
) -> Optional[dict[str, Any]]:
    if snapshot is None or row >= snapshot.token_ids.shape[0]:
        return None
    token_ids = snapshot.token_ids[row].detach().cpu().tolist()
    values = snapshot.values[row].detach().float().cpu().tolist()
    return {
        "ids": token_ids,
        "values": values,
        "margin": values[0] - values[1] if len(values) > 1 else None,
    }


def _parse_dump_positions() -> set[int]:
    raw = envs.SGLANG_NPU_MTP_GREEDY_TRACE_DUMP_POSITIONS.get().strip()
    if not raw:
        return set()
    positions: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            positions.add(int(item))
    return positions


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:100]


def _maybe_dump_logits(
    *,
    event: str,
    req: "Req",
    pred_position: int,
    gen_index: int,
    logits_row: torch.Tensor,
    metadata: dict[str, Any],
) -> Optional[str]:
    if pred_position not in _parse_dump_positions():
        return None
    dump_dir = envs.SGLANG_NPU_MTP_GREEDY_TRACE_DUMP_DIR.get()
    os.makedirs(dump_dir, exist_ok=True)
    rank = _rank_fields()["rank"]
    filename = (
        f"{_safe_name(_trace_label() or 'run')}_{event}_rank{rank}_"
        f"rid{_safe_name(req.rid)}_gen{gen_index}_pos{pred_position}.pt"
    )
    path = os.path.join(dump_dir, filename)
    torch.save(
        {
            "logits": logits_row.detach().float().cpu(),
            "metadata": metadata,
        },
        path,
    )
    return path


def _gather_consensus(local_values: torch.Tensor, group) -> Optional[torch.Tensor]:
    gathered = group.all_gather(local_values.unsqueeze(0), dim=0)
    return gathered.cpu() if group.rank_in_group == 0 else None


def _consensus_payload(
    gathered: torch.Tensor, sections: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    consensus = torch.all(gathered == gathered[0:1], dim=0)
    payload: dict[str, Any] = {
        "all_equal": bool(consensus.all().item()),
        "mismatch_offsets": torch.nonzero(
            ~consensus, as_tuple=False
        ).flatten().tolist(),
    }
    for name, (start, end) in sections.items():
        section = gathered[:, start:end]
        section_consensus = consensus[start:end]
        payload[name] = {
            "all_equal": bool(section_consensus.all().item()),
            "values_by_rank": section.tolist(),
        }
    return payload


def _tp_normal_consensus(
    logits: torch.Tensor, chosen_token_ids: torch.Tensor
) -> tuple[Any, Any]:
    local_argmax = torch.argmax(logits.detach(), dim=-1).to(torch.int32)
    chosen = chosen_token_ids.detach().reshape(-1).to(torch.int32)
    local_values = torch.cat((local_argmax, chosen))
    group = _trace_group()
    gathered = _gather_consensus(local_values, group)
    if gathered is None:
        return group, None
    rows = local_argmax.numel()
    return group, _consensus_payload(
        gathered,
        {
            "target_argmax": (0, rows),
            "chosen_token": (rows, rows * 2),
        },
    )


def _tp_mtp_consensus(
    *,
    logits: torch.Tensor,
    predict: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    bs: int,
    width: int,
) -> tuple[Any, Any]:
    valid_rows = bs * width
    accept_width = accept_index.shape[1]
    target_argmax = torch.argmax(
        logits[:valid_rows].detach(), dim=-1
    ).to(torch.int32)
    predict_flat = predict[:valid_rows].detach().reshape(-1).to(torch.int32)
    lens_flat = accept_lens[:bs].detach().reshape(-1).to(torch.int32)
    index_flat = (
        accept_index[:bs, :accept_width].detach().reshape(-1).to(torch.int32)
    )
    local_values = torch.cat(
        (target_argmax, predict_flat, lens_flat, index_flat)
    )
    group = _trace_group()
    gathered = _gather_consensus(local_values, group)
    if gathered is None:
        return group, None

    argmax_end = valid_rows
    predict_end = argmax_end + valid_rows
    lens_end = predict_end + bs
    index_end = lens_end + bs * accept_width
    return group, _consensus_payload(
        gathered,
        {
            "target_argmax": (0, argmax_end),
            "predict": (argmax_end, predict_end),
            "accept_lens": (predict_end, lens_end),
            "accept_index": (lens_end, index_end),
        },
    )


def _sampling_values(sampling_info, row: int) -> dict[str, Any]:
    grammars = getattr(sampling_info, "grammars", None)
    return {
        "temperature": float(sampling_info.temperatures[row].item()),
        "top_k": int(sampling_info.top_ks[row].item()),
        "top_p": float(sampling_info.top_ps[row].item()),
        "min_p": float(sampling_info.min_ps[row].item()),
        "is_all_greedy": bool(sampling_info.is_all_greedy),
        "has_penalizer_orchestrator": (
            getattr(sampling_info, "penalizer_orchestrator", None) is not None
        ),
        "has_acc_additive_penalties": (
            getattr(sampling_info, "acc_additive_penalties", None) is not None
        ),
        "has_acc_scaling_penalties": (
            getattr(sampling_info, "acc_scaling_penalties", None) is not None
        ),
        "has_logit_bias": getattr(sampling_info, "logit_bias", None) is not None,
        "has_custom_logit_processor": bool(
            getattr(sampling_info, "has_custom_logit_processor", False)
        ),
        "has_grammar": bool(
            grammars is not None
            and row < len(grammars)
            and grammars[row] is not None
        ),
        "has_vocab_mask": getattr(sampling_info, "vocab_mask", None) is not None,
    }


def trace_normal_greedy_sample(
    *,
    batch: Optional["ScheduleBatch"],
    forward_batch: "ForwardBatch",
    can_run_cuda_graph: bool,
    graph_runner: Optional[Any],
    logits: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    raw_snapshot: Optional[LogitsTopKSnapshot],
) -> None:
    if (
        not greedy_trace_enabled()
        or batch is None
        or not forward_batch.sampling_info.is_all_greedy
    ):
        return

    valid_rows = len(batch.reqs)
    group, consensus = _tp_normal_consensus(
        logits[:valid_rows], chosen_token_ids[:valid_rows]
    )
    if group.rank_in_group != 0:
        return
    if not _batch_selected(batch, 1):
        return

    effective = _topk_snapshot(logits[:valid_rows])
    chosen = chosen_token_ids.detach().cpu().tolist()
    graph_buffers = getattr(graph_runner, "buffers", None)
    uses_static_graph_buffers = can_run_cuda_graph and graph_buffers is not None
    model_inputs = graph_buffers if uses_static_graph_buffers else forward_batch
    seq_lens = model_inputs.seq_lens[:valid_rows].detach().cpu().tolist()
    live_seq_lens = (
        forward_batch.seq_lens[:valid_rows].detach().cpu().tolist()
    )
    if forward_batch.forward_mode.is_decode():
        input_positions = (
            model_inputs.positions[:valid_rows].detach().cpu().tolist()
        )
        input_tokens = (
            model_inputs.input_ids[:valid_rows].detach().cpu().tolist()
        )
        live_input_positions = (
            forward_batch.positions[:valid_rows].detach().cpu().tolist()
        )
        live_input_tokens = (
            forward_batch.input_ids[:valid_rows].detach().cpu().tolist()
        )
    else:
        input_positions = [int(seq_len) - 1 for seq_len in seq_lens]
        input_tokens = [None] * len(batch.reqs)
        live_input_positions = [
            int(seq_len) - 1 for seq_len in live_seq_lens
        ]
        live_input_tokens = [None] * len(batch.reqs)
    out_cache_locs = []
    live_out_cache_locs = []
    if (
        forward_batch.forward_mode.is_decode()
        and forward_batch.out_cache_loc is not None
    ):
        out_cache_locs = (
            model_inputs.out_cache_loc[:valid_rows].detach().cpu().tolist()
        )
        live_out_cache_locs = (
            forward_batch.out_cache_loc[:valid_rows].detach().cpu().tolist()
        )

    for row, req in enumerate(batch.reqs):
        if not _request_selected(req):
            continue
        gen_index = len(req.output_ids)
        pred_position = int(input_positions[row]) + 1
        payload = {
            **_rank_fields(group),
            "rid": str(req.rid),
            "gen_index_guess": gen_index,
            "forward_mode": str(forward_batch.forward_mode),
            "can_run_cuda_graph": bool(can_run_cuda_graph),
            "inspecting_static_graph_buffers": bool(uses_static_graph_buffers),
            "seq_len": int(seq_lens[row]),
            "live_forward_seq_len": int(live_seq_lens[row]),
            "model_seq_len_matches_live": (
                int(seq_lens[row]) == int(live_seq_lens[row])
            ),
            "input_position": int(input_positions[row]),
            "live_forward_input_position": int(live_input_positions[row]),
            "model_position_matches_live": (
                int(input_positions[row]) == int(live_input_positions[row])
            ),
            "pred_position": pred_position,
            "input_token": input_tokens[row],
            "live_forward_input_token": live_input_tokens[row],
            "model_input_token_matches_live": (
                input_tokens[row] == live_input_tokens[row]
            ),
            "out_cache_loc": (
                int(out_cache_locs[row]) if row < len(out_cache_locs) else None
            ),
            "live_forward_out_cache_loc": (
                int(live_out_cache_locs[row])
                if row < len(live_out_cache_locs)
                else None
            ),
            "model_cache_loc_matches_live": (
                int(out_cache_locs[row]) == int(live_out_cache_locs[row])
                if row < len(out_cache_locs)
                and row < len(live_out_cache_locs)
                else None
            ),
            "chosen_token": int(chosen[row]),
            "sampling": _sampling_values(forward_batch.sampling_info, row),
            "raw_top": _row_topk(raw_snapshot, row),
            "effective_top": _row_topk(effective, row),
            "tp_consensus": consensus,
        }
        dump_path = _maybe_dump_logits(
            event="normal",
            req=req,
            pred_position=pred_position,
            gen_index=gen_index,
            logits_row=logits[row],
            metadata=payload,
        )
        if dump_path is not None:
            payload["logits_dump"] = dump_path
        _emit("normal_greedy", payload)


def _as_cpu_2d(tensor: torch.Tensor, bs: int, width: int) -> torch.Tensor:
    return tensor.detach().reshape(-1)[: bs * width].reshape(bs, width).cpu()


def _identity_retrieve_index(bs: int, width: int) -> list[list[int]]:
    return [
        list(range(batch_index * width, (batch_index + 1) * width))
        for batch_index in range(bs)
    ]


def _extract_full_mask_tails(
    custom_mask: torch.Tensor, seq_lens: Sequence[int], width: int
) -> list[list[list[bool]]]:
    """Copy only each verify query's ``D x D`` draft-to-draft mask tail."""

    flat_mask = custom_mask.detach().reshape(-1)
    tails = []
    previous_seq_lens = 0
    for batch_index, seq_len_value in enumerate(seq_lens):
        seq_len = int(seq_len_value)
        batch_token_base = batch_index * width
        seq_tree_index = (batch_token_base + previous_seq_lens) * width
        rows = []
        for token_index in range(width):
            start = (
                seq_tree_index
                + (seq_len + width) * token_index
                + seq_len
            )
            end = start + width
            if end > flat_mask.numel():
                raise ValueError(
                    f"custom_mask is too short for request {batch_index}, row "
                    f"{token_index}: need {end}, have {flat_mask.numel()}"
                )
            rows.append(flat_mask[start:end])
        tails.append(torch.stack(rows).cpu().bool().tolist())
        previous_seq_lens += seq_len
    return tails


def trace_mtp_greedy_verify(
    *,
    batch: "ScheduleBatch",
    forward_batch: "ForwardBatch",
    can_run_cuda_graph: bool,
    graph_runner: Optional[Any],
    verify_input: "EagleVerifyInput",
    logits: torch.Tensor,
    predict: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    raw_snapshot: Optional[LogitsTopKSnapshot],
) -> None:
    if (
        not greedy_trace_enabled()
        or not batch.sampling_info.is_all_greedy
        or batch.forward_mode.is_idle()
    ):
        return

    bs = len(batch.reqs)
    width = int(verify_input.draft_token_num)
    group, consensus = _tp_mtp_consensus(
        logits=logits,
        predict=predict,
        accept_lens=accept_lens,
        accept_index=accept_index,
        bs=bs,
        width=width,
    )
    if group.rank_in_group != 0:
        return
    if not _batch_selected(batch, width):
        return

    valid_rows = bs * width
    effective = _topk_snapshot(logits[:valid_rows])
    target_predict = torch.argmax(logits[:valid_rows].detach(), dim=-1).reshape(
        bs, width
    )

    candidates_cpu = _as_cpu_2d(verify_input.draft_token, bs, width)
    live_input_ids_cpu = _as_cpu_2d(
        forward_batch.input_ids, bs, width
    )
    graph_buffers = getattr(graph_runner, "buffers", None)
    uses_static_graph_buffers = can_run_cuda_graph and graph_buffers is not None
    model_inputs = graph_buffers if uses_static_graph_buffers else forward_batch
    model_input_ids_cpu = _as_cpu_2d(model_inputs.input_ids, bs, width)
    retrieve_cpu = _as_cpu_2d(verify_input.retrieve_index, bs, width)
    retrieve_next_token_cpu = _as_cpu_2d(
        verify_input.retrieve_next_token, bs, width
    ).tolist()
    retrieve_next_sibling_cpu = _as_cpu_2d(
        verify_input.retrieve_next_sibling, bs, width
    ).tolist()
    target_cpu = target_predict.cpu()
    predict_cpu = predict[:valid_rows].detach().cpu().tolist()
    accept_lens_cpu = accept_lens[:bs].detach().cpu().tolist()
    accept_index_cpu = accept_index[:bs].detach().cpu().tolist()
    verify_positions_cpu = _as_cpu_2d(
        verify_input.positions, bs, width
    ).tolist()
    live_positions_cpu = _as_cpu_2d(
        forward_batch.positions, bs, width
    ).tolist()
    positions_cpu = _as_cpu_2d(model_inputs.positions, bs, width).tolist()
    batch_out_cache_cpu = _as_cpu_2d(batch.out_cache_loc, bs, width).tolist()
    live_out_cache_cpu = _as_cpu_2d(
        forward_batch.out_cache_loc, bs, width
    ).tolist()
    out_cache_cpu = _as_cpu_2d(model_inputs.out_cache_loc, bs, width).tolist()
    batch_seq_lens_cpu = batch.seq_lens[:bs].detach().cpu().tolist()
    live_seq_lens_cpu = forward_batch.seq_lens[:bs].detach().cpu().tolist()
    seq_lens_cpu = model_inputs.seq_lens[:bs].detach().cpu().tolist()
    batch_req_pool_indices_cpu = (
        batch.req_pool_indices[:bs].detach().cpu().tolist()
    )
    live_req_pool_indices_cpu = (
        forward_batch.req_pool_indices[:bs].detach().cpu().tolist()
    )
    req_pool_indices_cpu = (
        model_inputs.req_pool_indices[:bs].detach().cpu().tolist()
    )

    mask_tails = None
    mask_error = None
    try:
        mask_tails = _extract_full_mask_tails(
            verify_input.custom_mask, live_seq_lens_cpu, width
        )
    except (AttributeError, RuntimeError, ValueError) as exc:
        mask_error = str(exc)

    expected_cache_rows: list[Optional[list[int]]] = []
    cache_mapping_errors: list[Optional[str]] = []
    req_to_token = batch.req_to_token_pool.req_to_token
    for batch_index in range(bs):
        seq_len = int(seq_lens_cpu[batch_index])
        req_pool_index = int(req_pool_indices_cpu[batch_index])
        if not 0 <= req_pool_index < req_to_token.shape[0]:
            expected_cache_rows.append(None)
            cache_mapping_errors.append(
                f"req_pool_index {req_pool_index} is outside "
                f"[0, {req_to_token.shape[0]})"
            )
        elif seq_len < 0 or seq_len + width > req_to_token.shape[1]:
            expected_cache_rows.append(None)
            cache_mapping_errors.append(
                f"cache slice [{seq_len}, {seq_len + width}) is outside "
                f"[0, {req_to_token.shape[1]})"
            )
        else:
            expected_cache_rows.append(
                req_to_token[
                    req_pool_index, seq_len : seq_len + width
                ].detach().cpu().tolist()
            )
            cache_mapping_errors.append(None)

    reference: Optional[TopK1Reference] = None
    reference_error = None
    if int(verify_input.tree_topk) == 1:
        try:
            reference = build_topk1_reference(
                candidates_cpu, retrieve_cpu, target_cpu
            )
        except ValueError as exc:
            reference_error = str(exc)

    expected_retrieve = _identity_retrieve_index(bs, width)
    is_topk1 = int(verify_input.tree_topk) == 1
    simulate_accept_len = envs.SGLANG_SIMULATE_ACC_LEN.get()
    simulation_active = simulate_accept_len > 0
    predict_global_matches_reference = (
        predict_cpu == reference.predict if reference is not None else None
    )
    for batch_index, req in enumerate(batch.reqs):
        if not _request_selected(req, width):
            continue
        base = batch_index * width
        actual_len = int(accept_lens_cpu[batch_index])
        accept_width = len(accept_index_cpu[batch_index])
        actual_len_in_range = 1 <= actual_len <= min(width, accept_width)
        display_len = min(max(actual_len, 0), width, accept_width)
        actual_indices = [
            int(value)
            for value in accept_index_cpu[batch_index][:display_len]
        ]
        invalid_indices = [
            index
            for index in actual_indices
            if not base <= index < base + width
        ]
        actual_path_tokens = [
            int(predict_cpu[index])
            for index in actual_indices
            if base <= index < base + width
        ]
        actual_front_tokens = [
            int(value) for value in predict_cpu[base : base + display_len]
        ]

        expected_positions = (
            [
                int(seq_lens_cpu[batch_index]) + offset
                for offset in range(width)
            ]
            if is_topk1
            else None
        )
        expected_mask_tail = (
            [
                [column <= row for column in range(width)]
                for row in range(width)
            ]
            if is_topk1
            else None
        )
        expected_retrieve_next_token = (
            list(range(1, width)) + [-1] if is_topk1 else None
        )
        expected_retrieve_next_sibling = [-1] * width if is_topk1 else None
        rows = []
        gen_start = len(req.output_ids)
        for offset in range(width):
            row = base + offset
            pred_position = int(positions_cpu[batch_index][offset]) + 1
            row_payload = {
                "row": row,
                "input_gen_index": gen_start + offset - 1,
                "predicted_gen_index": gen_start + offset,
                "input_token": int(candidates_cpu[batch_index, offset].item()),
                "input_position": int(positions_cpu[batch_index][offset]),
                "pred_position": pred_position,
                "out_cache_loc": int(out_cache_cpu[batch_index][offset]),
                "raw_top": _row_topk(raw_snapshot, row),
                "effective_top": _row_topk(effective, row),
            }
            dump_path = _maybe_dump_logits(
                event="mtp",
                req=req,
                pred_position=pred_position,
                gen_index=gen_start + offset,
                logits_row=logits[row],
                metadata=row_payload,
            )
            if dump_path is not None:
                row_payload["logits_dump"] = dump_path
            rows.append(row_payload)

        reference_payload = None
        if reference is not None:
            ref_len = reference.accept_lens[batch_index]
            ref_indices = reference.accept_indices[batch_index]
            ref_tokens = reference.accept_tokens[batch_index]
            predict_row_matches_reference = (
                predict_cpu[base : base + width]
                == reference.predict[base : base + width]
            )
            reference_payload = {
                "accept_len": ref_len,
                "accept_indices": ref_indices,
                "accept_tokens": ref_tokens,
                "comparison_skipped": (
                    "SGLANG_SIMULATE_ACC_LEN is active"
                    if simulation_active
                    else None
                ),
                "accept_len_matches": (
                    actual_len == ref_len if not simulation_active else None
                ),
                "accept_index_matches": (
                    actual_indices == ref_indices
                    if not simulation_active
                    else None
                ),
                "path_tokens_match": (
                    actual_path_tokens == ref_tokens
                    if not simulation_active
                    else None
                ),
                "front_tokens_match": (
                    actual_front_tokens == ref_tokens
                    if not simulation_active
                    else None
                ),
                "predict_row_matches": (
                    predict_row_matches_reference
                    if not simulation_active
                    else None
                ),
                "predict_global_matches": (
                    predict_global_matches_reference
                    if not simulation_active
                    else None
                ),
            }
            reference_payload["all_match"] = (
                actual_len_in_range
                and not invalid_indices
                and all(
                    reference_payload[key]
                    for key in (
                        "accept_len_matches",
                        "accept_index_matches",
                        "path_tokens_match",
                        "front_tokens_match",
                        "predict_row_matches",
                    )
                )
                if not simulation_active
                else None
            )

        _emit(
            "mtp_verify",
            {
                **_rank_fields(group),
                "rid": str(req.rid),
                "gen_start_guess": gen_start,
                "forward_mode": str(batch.forward_mode),
                "tree_topk": int(verify_input.tree_topk),
                "draft_token_num": width,
                "can_run_cuda_graph": bool(can_run_cuda_graph),
                "inspecting_static_graph_buffers": bool(
                    uses_static_graph_buffers
                ),
                "sampling": _sampling_values(batch.sampling_info, batch_index),
                "simulate_accept_len": simulate_accept_len,
                "seq_len": int(seq_lens_cpu[batch_index]),
                "live_forward_seq_len": int(
                    live_seq_lens_cpu[batch_index]
                ),
                "batch_seq_len": int(batch_seq_lens_cpu[batch_index]),
                "model_seq_len_matches_live": (
                    int(seq_lens_cpu[batch_index])
                    == int(live_seq_lens_cpu[batch_index])
                ),
                "live_seq_len_matches_batch": (
                    int(live_seq_lens_cpu[batch_index])
                    == int(batch_seq_lens_cpu[batch_index])
                ),
                "req_pool_index": int(req_pool_indices_cpu[batch_index]),
                "live_forward_req_pool_index": int(
                    live_req_pool_indices_cpu[batch_index]
                ),
                "batch_req_pool_index": int(
                    batch_req_pool_indices_cpu[batch_index]
                ),
                "model_req_pool_index_matches_live": (
                    int(req_pool_indices_cpu[batch_index])
                    == int(live_req_pool_indices_cpu[batch_index])
                ),
                "live_req_pool_index_matches_batch": (
                    int(live_req_pool_indices_cpu[batch_index])
                    == int(batch_req_pool_indices_cpu[batch_index])
                ),
                "draft_tokens": candidates_cpu[batch_index].tolist(),
                "live_forward_input_tokens": live_input_ids_cpu[
                    batch_index
                ].tolist(),
                "model_input_tokens": model_input_ids_cpu[batch_index].tolist(),
                "live_input_tokens_match_verify_input": torch.equal(
                    candidates_cpu[batch_index],
                    live_input_ids_cpu[batch_index],
                ),
                "model_input_tokens_match_live": torch.equal(
                    model_input_ids_cpu[batch_index],
                    live_input_ids_cpu[batch_index],
                ),
                "positions": positions_cpu[batch_index],
                "live_forward_positions": live_positions_cpu[batch_index],
                "verify_input_positions": verify_positions_cpu[batch_index],
                "model_positions_match_live": (
                    positions_cpu[batch_index] == live_positions_cpu[batch_index]
                ),
                "live_positions_match_verify_input": (
                    live_positions_cpu[batch_index]
                    == verify_positions_cpu[batch_index]
                ),
                "expected_positions": expected_positions,
                "positions_match": (
                    positions_cpu[batch_index] == expected_positions
                    if expected_positions is not None
                    else None
                ),
                "retrieve_index": retrieve_cpu[batch_index].tolist(),
                "expected_retrieve_index": expected_retrieve[batch_index],
                "retrieve_index_matches": (
                    retrieve_cpu[batch_index].tolist()
                    == expected_retrieve[batch_index]
                ),
                "retrieve_next_token": retrieve_next_token_cpu[batch_index],
                "expected_retrieve_next_token": expected_retrieve_next_token,
                "retrieve_next_token_matches": (
                    retrieve_next_token_cpu[batch_index]
                    == expected_retrieve_next_token
                    if is_topk1
                    else None
                ),
                "retrieve_next_sibling": retrieve_next_sibling_cpu[batch_index],
                "expected_retrieve_next_sibling": (
                    expected_retrieve_next_sibling
                ),
                "retrieve_next_sibling_matches": (
                    retrieve_next_sibling_cpu[batch_index]
                    == expected_retrieve_next_sibling
                    if is_topk1
                    else None
                ),
                "tree_mask_tail": (
                    mask_tails[batch_index] if mask_tails is not None else None
                ),
                "tree_mask_note": (
                    "build-tree FULL_MASK output; Ascend target verify may use "
                    "backend causal metadata instead of graph_buffers.custom_mask"
                ),
                "expected_tree_mask_tail": expected_mask_tail,
                "tree_mask_tail_matches": (
                    mask_tails is not None
                    and expected_mask_tail is not None
                    and mask_tails[batch_index] == expected_mask_tail
                    if is_topk1
                    else None
                ),
                "tree_mask_error": mask_error,
                "out_cache_locs": out_cache_cpu[batch_index],
                "live_forward_out_cache_locs": live_out_cache_cpu[batch_index],
                "batch_out_cache_locs": batch_out_cache_cpu[batch_index],
                "model_cache_locs_match_live": (
                    out_cache_cpu[batch_index] == live_out_cache_cpu[batch_index]
                ),
                "live_cache_locs_match_batch": (
                    live_out_cache_cpu[batch_index]
                    == batch_out_cache_cpu[batch_index]
                ),
                "req_to_token_locs": expected_cache_rows[batch_index],
                "cache_mapping_matches": (
                    out_cache_cpu[batch_index] == expected_cache_rows[batch_index]
                    if expected_cache_rows[batch_index] is not None
                    else None
                ),
                "cache_mapping_error": cache_mapping_errors[batch_index],
                "actual_accept_len": actual_len,
                "actual_accept_len_in_range": actual_len_in_range,
                "actual_accept_len_displayed": display_len,
                "actual_accept_indices": actual_indices,
                "invalid_accept_indices": invalid_indices,
                "actual_path_tokens": actual_path_tokens,
                "actual_front_tokens": actual_front_tokens,
                "reference": reference_payload,
                "reference_error": reference_error,
                "tp_consensus": consensus,
                "rows": rows,
            },
        )


def trace_scheduler_commit(
    *,
    req: "Req",
    tokens: Sequence[int],
    phase: str,
    is_spec: bool,
) -> None:
    """Log the authoritative tokens immediately before ``req.output_ids`` update."""

    if not greedy_trace_enabled() or not _request_selected(req, len(tokens)):
        return
    gen_start = len(req.output_ids)
    token_list = [int(token) for token in tokens]
    _emit(
        "scheduler_commit",
        {
            "rid": str(req.rid),
            "phase": phase,
            "is_spec": bool(is_spec),
            "gen_start": gen_start,
            "tokens": token_list,
            "indexed_tokens": [
                {"gen_index": gen_start + offset, "token": token}
                for offset, token in enumerate(token_list)
            ],
        },
    )
