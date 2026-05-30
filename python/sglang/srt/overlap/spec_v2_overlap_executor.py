from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.overlap.base_executor import OverlapExecutionResult, PendingOverlapResult
from sglang.srt.overlap.tp_worker_client_v2 import TpWorkerClientV2
from sglang.srt.speculative.spec_v2_overlap_client import SpecV2OverlapWorkerClient

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


class SpecV2OverlapExecutor:
    """Behavior-preserving wrapper for the existing spec v2 overlap path."""

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.client = SpecV2OverlapWorkerClient(scheduler)
        self.worker_client: TpWorkerClientV2[OverlapExecutionResult] = TpWorkerClientV2(
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
        future_indices_or_next_token_ids = -future_indices.indices

        pending_result = PendingOverlapResult(
            async_handle=self.worker_client.submit_async(
                lambda: self._run_in_worker(
                    batch, model_worker_batch, future_indices
                )
            ),
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
        )
        return OverlapExecutionResult(
            batch_result=pending_result,
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
        )

    def _run_in_worker(
        self,
        batch: "ScheduleBatch",
        model_worker_batch: "ModelWorkerBatch",
        future_indices,
    ) -> OverlapExecutionResult:
        batch_result = self.client.run_with_future_indices(
            batch, model_worker_batch, future_indices
        )
        return batch_result
