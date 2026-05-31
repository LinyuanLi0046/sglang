from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from sglang.srt.managers.utils import GenerationBatchResult

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


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
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return False
        if draft_input.last_verified_ids.ndim != 1 or draft_input.token_list.ndim != 2:
            return False
        if draft_input.token_list.shape[0] != draft_input.last_verified_ids.shape[0]:
            return False
        if draft_input.token_list.shape[1] != scheduler.server_args.speculative_num_steps:
            return False
        return True

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
                scheduler._schedule_generation_batch_result_copy(batch, batch_result)
            else:
                if batch_result.copy_done is None:
                    batch_result.copy_done = scheduler.device_module.Event()
                batch_result.future_indices = future_indices

        scheduler._relay_spec_v2_overlap_result(batch, batch_result, future_indices)
        return batch_result
