from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.managers.utils import GenerationBatchResult

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.overlap_utils import FutureIndices
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SpecV2OverlapWorkerClient:
    """A thin control-plane wrapper for spec v2 overlap execution.

    This keeps the speculative overlap orchestration out of Scheduler.run_batch()
    without changing the underlying EAGLEWorkerV2 / FutureMap semantics.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def _should_store_real_placeholder_state(
        self, batch_result: GenerationBatchResult
    ) -> bool:
        scheduler = self.scheduler
        draft_input = batch_result.next_draft_input
        if draft_input is None:
            return False
        if getattr(scheduler.server_args, "speculative_eagle_topk", None) != 1:
            return False
        if not draft_input.has_real_future_payload:
            return False
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return False
        if draft_input.last_verified_ids.ndim != 1 or draft_input.token_list.ndim != 2:
            return False
        if draft_input.token_list.shape[0] != draft_input.last_verified_ids.shape[0]:
            return False
        if draft_input.token_list.shape[1] != scheduler.server_args.speculative_num_steps:
            return False
        return True

    def _extract_placeholder_audit_payload(
        self, batch_result: GenerationBatchResult
    ) -> Optional[tuple["torch.Tensor", "torch.Tensor"]]:
        draft_input = batch_result.next_draft_input
        if draft_input is None or not draft_input.has_real_future_payload:
            return None
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return None
        if draft_input.last_verified_ids.ndim != 1 or draft_input.token_list.ndim != 2:
            return None
        if draft_input.token_list.shape[0] != draft_input.last_verified_ids.shape[0]:
            return None
        return draft_input.last_verified_ids, draft_input.token_list

    def _extract_replay_audit_payload(
        self, future_indices: "FutureIndices"
    ) -> Optional[dict[str, "torch.Tensor"]]:
        replay_payload = self.scheduler.future_map.peek_replay_future_state(future_indices)
        verified_id = replay_payload["verified_id"]
        topk_index = replay_payload["topk_index"]
        if verified_id is None or topk_index is None:
            return None
        return {
            "verified_id": verified_id,
            "topk_index": topk_index,
        }

    def _maybe_record_placeholder_audit(
        self,
        batch_result: GenerationBatchResult,
        future_indices: "FutureIndices",
    ) -> None:
        placeholder_payload = self._extract_placeholder_audit_payload(batch_result)
        if placeholder_payload is None:
            return

        spec_last_verified_ids, spec_token_list = placeholder_payload
        replay_payload = self._extract_replay_audit_payload(future_indices)
        if replay_payload is None:
            return

        replay_verified_id = replay_payload["verified_id"]
        replay_topk_index = replay_payload["topk_index"]

        if spec_last_verified_ids.shape != replay_verified_id.shape:
            logger.warning(
                "Phase4 audit shape mismatch for last_verified_ids at interval=%s: "
                "spec=%s replay=%s.",
                future_indices.interval,
                tuple(spec_last_verified_ids.shape),
                tuple(replay_verified_id.shape),
            )
            return

        last_verified_mismatch = (spec_last_verified_ids != replay_verified_id).sum().item()
        if last_verified_mismatch:
            logger.warning(
                "Phase4 audit mismatch at interval=%s: "
                "last_verified_ids mismatch_count=%d batch_size=%d.",
                future_indices.interval,
                last_verified_mismatch,
                spec_last_verified_ids.shape[0],
            )

        if replay_topk_index.ndim != 2:
            logger.warning(
                "Phase4 audit skipped token_list anchor compare at interval=%s because "
                "replay topk_index ndim=%d.",
                future_indices.interval,
                replay_topk_index.ndim,
            )
            return

        if spec_token_list.shape[0] != replay_topk_index.shape[0]:
            logger.warning(
                "Phase4 audit token_list batch mismatch at interval=%s: "
                "spec=%s replay=%s.",
                future_indices.interval,
                tuple(spec_token_list.shape),
                tuple(replay_topk_index.shape),
            )
            return

        anchor_spec = spec_token_list[:, 0]
        anchor_replay = replay_topk_index[:, 0].to(dtype=anchor_spec.dtype)
        token_anchor_mismatch = (anchor_spec != anchor_replay).sum().item()
        if token_anchor_mismatch:
            logger.warning(
                "Phase4 audit mismatch at interval=%s: token_list first-column "
                "mismatch_count=%d batch_size=%d width=%d.",
                future_indices.interval,
                token_anchor_mismatch,
                spec_token_list.shape[0],
                spec_token_list.shape[1],
            )
        elif last_verified_mismatch == 0:
            logger.debug(
                "Phase4 audit passed at interval=%s with batch_size=%d width=%d. "
                "Current audit proves last_verified_ids and token_list first-column "
                "anchor; full token_list equivalence still relies on later phase4/5 "
                "consumer validation.",
                future_indices.interval,
                spec_token_list.shape[0],
                spec_token_list.shape[1],
            )

    def submit(
        self,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
    ) -> Tuple[GenerationBatchResult, "torch.Tensor"]:
        scheduler = self.scheduler

        scheduler.record_batch_in_overlap(model_worker_batch)

        # Sampling info will be modified during forward, so we store a copy.
        model_worker_batch.sampling_info = (
            model_worker_batch.sampling_info.copy_for_forward()
        )

        bs = len(model_worker_batch.seq_lens)
        future_indices = scheduler.future_map.alloc_future_indices(bs)
        batch_result = self.run_with_future_indices(
            batch, model_worker_batch, future_indices
        )
        future_indices_or_next_token_ids = -future_indices.indices
        return batch_result, future_indices_or_next_token_ids

    def run_with_future_indices(
        self,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        future_indices,
    ) -> GenerationBatchResult:
        scheduler = self.scheduler

        with scheduler.forward_stream_ctx, scheduler.record_bubble_metrics(batch):
            scheduler.forward_stream.wait_stream(scheduler.schedule_stream)
            scheduler.future_map.resolve_future(model_worker_batch)
            with scheduler.record_forward_metrics(batch):
                batch_result = scheduler.model_worker.forward_batch_generation(
                    model_worker_batch
                    # here pp is not compatible with overlap
                )
            if self._should_store_real_placeholder_state(batch_result):
                scheduler.future_map.store_spec_future_state(
                    future_indices, batch_result.next_draft_input
                )
            if batch_result.delay_sample_func is None:
                scheduler.future_map.store_to_map(future_indices, batch_result)
                self._maybe_record_placeholder_audit(batch_result, future_indices)
                scheduler._schedule_generation_batch_result_copy(batch, batch_result)
            else:
                if batch_result.copy_done is None:
                    batch_result.copy_done = scheduler.device_module.Event()
                batch_result.future_indices = future_indices

        scheduler._relay_spec_v2_overlap_result(batch, batch_result, future_indices)
        return batch_result
