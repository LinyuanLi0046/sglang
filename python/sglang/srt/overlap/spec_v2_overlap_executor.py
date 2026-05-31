from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.overlap.base_executor import OverlapExecutionResult, PendingOverlapResult
from sglang.srt.overlap.tp_worker_client_v2 import TpWorkerClientV2
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_v2_overlap_client import SpecV2OverlapWorkerClient

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


class SpecV2OverlapExecutor:
    """Behavior-preserving wrapper for the existing spec v2 overlap path."""

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.client = SpecV2OverlapWorkerClient(scheduler)
        self.worker_client: TpWorkerClientV2[object] = TpWorkerClientV2(
            "spec-v2",
            device=scheduler.device,
            gpu_id=scheduler.gpu_id,
        )

    def submit(
        self,
        batch: "ScheduleBatch",
        model_worker_batch: "ModelWorkerBatch",
    ) -> OverlapExecutionResult:
        scheduler = self.scheduler

        scheduler.record_batch_in_overlap(model_worker_batch)

        # Sampling info will be modified during forward, so we store a copy.
        model_worker_batch.sampling_info = (
            model_worker_batch.sampling_info.copy_for_forward()
        )
        bs = len(model_worker_batch.seq_lens)
        future_indices = scheduler.future_map.alloc_future_indices(bs)
        self._install_future_placeholder(batch, future_indices)
        future_indices_or_next_token_ids = -future_indices.indices

        pending_result = PendingOverlapResult(
            async_handle=self.worker_client.submit_async(
                lambda: self._run_in_worker(
                    batch, model_worker_batch, future_indices
                )
            ),
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
            relay_target_batch=batch,
            future_indices=future_indices,
            schedule_safe_without_resolve=self._can_schedule_without_preschedule_resolve(
                batch
            ),
            requires_current_batch_resolve_for_sampling=False,
            requires_resolve_before_next_schedule=True,
        )
        return OverlapExecutionResult(
            batch_result=pending_result,
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
            future_indices=future_indices,
        )

    def _can_schedule_without_preschedule_resolve(self, batch: "ScheduleBatch") -> bool:
        # Phase6-C first cut:
        # only open the overlap window for the common spec-v2 decode path.
        # Keep grammar / structured output and non-decode paths conservative.
        return bool(
            batch.is_spec_v2
            and batch.forward_mode.is_decode()
            and not batch.has_grammar
        )

    def _run_in_worker(
        self,
        batch: "ScheduleBatch",
        model_worker_batch: "ModelWorkerBatch",
        future_indices,
    ) -> object:
        batch_result, batch_relay = self.client.run_with_future_indices(
            batch, model_worker_batch, future_indices
        )
        return batch_result, batch_relay

    def _install_future_placeholder(self, batch: "ScheduleBatch", future_indices) -> None:
        scheduler = self.scheduler
        speculative_num_steps = max(
            int(scheduler.server_args.speculative_num_steps or 0), 1
        )
        future_state_pool = scheduler.future_map.ensure_spec_future_state_pool(
            speculative_num_steps
        )
        future_handle = future_state_pool.alloc_handle(future_indices)
        last_verified_ids, token_list = future_state_pool.create_placeholder_inputs(
            future_handle
        )
        batch.spec_info = EagleDraftInput.create_future_placeholder_input(
            future_handle=future_handle,
            last_verified_ids=last_verified_ids,
            token_list=token_list,
            new_seq_lens=batch.seq_lens.clone(),
        )
