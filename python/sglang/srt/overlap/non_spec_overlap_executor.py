from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.overlap.base_executor import OverlapExecutionResult, PendingOverlapResult
from sglang.srt.overlap.tp_worker_client_v2 import TpWorkerClientV2

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


class NonSpecOverlapExecutor:
    """Behavior-preserving wrapper for the existing non-spec overlap path."""

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.worker_client: TpWorkerClientV2[OverlapExecutionResult] = TpWorkerClientV2(
            "non-spec",
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
        requires_current_batch_resolve_for_sampling = (
            model_worker_batch.sampling_info is not None
            and model_worker_batch.sampling_info.grammars is not None
        )

        pending_result = PendingOverlapResult(
            async_handle=self.worker_client.submit_async(
                lambda: self._run_in_worker(batch, model_worker_batch, future_indices)
            ),
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
            requires_current_batch_resolve_for_sampling=(
                requires_current_batch_resolve_for_sampling
            ),
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
        scheduler = self.scheduler

        with scheduler.forward_stream_ctx, scheduler.record_bubble_metrics(batch):
            scheduler.forward_stream.wait_stream(scheduler.schedule_stream)
            scheduler.future_map.resolve_future(model_worker_batch)
            with scheduler.record_forward_metrics(batch):
                batch_result = scheduler.model_worker.forward_batch_generation(
                    model_worker_batch
                    # here pp is not compatible with overlap
                )
            if batch_result.delay_sample_func is None:
                scheduler.future_map.store_to_map(future_indices, batch_result)
                scheduler._schedule_generation_batch_result_copy(batch, batch_result)
            else:
                if batch_result.copy_done is None:
                    batch_result.copy_done = scheduler.device_module.Event()
                batch_result.future_indices = future_indices

        return batch_result
