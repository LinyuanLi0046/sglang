from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.overlap.base_executor import OverlapExecutionResult
from sglang.srt.speculative.spec_v2_overlap_client import SpecV2OverlapWorkerClient

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


class SpecV2OverlapExecutor:
    """Behavior-preserving wrapper for the existing spec v2 overlap path."""

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.client = SpecV2OverlapWorkerClient(scheduler)

    def submit(
        self,
        batch: "ScheduleBatch",
        model_worker_batch: "ModelWorkerBatch",
    ) -> OverlapExecutionResult:
        batch_result, future_indices_or_next_token_ids = self.client.submit(
            batch, model_worker_batch
        )
        return OverlapExecutionResult(
            batch_result=batch_result,
            future_indices_or_next_token_ids=future_indices_or_next_token_ids,
        )
