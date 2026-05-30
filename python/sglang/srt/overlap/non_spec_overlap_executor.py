from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.overlap.base_executor import OverlapExecutionResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


class NonSpecOverlapExecutor:
    """Behavior-preserving wrapper for the existing non-spec overlap path."""

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler

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

        return OverlapExecutionResult(
            batch_result=batch_result,
            future_indices_or_next_token_ids=-future_indices.indices,
        )
