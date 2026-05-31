from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.overlap.tp_worker_client_v2 import AsyncResultHandle


@dataclass
class OverlapExecutionResult:
    batch_result: Union[GenerationBatchResult, "PendingOverlapResult"]
    future_indices_or_next_token_ids: torch.Tensor
    batch_snapshot: Optional[Any] = None
    batch_relay: Optional[Any] = None
    future_indices: Optional[Any] = None
    next_token_ids: Optional[torch.Tensor] = None


class OverlapExecutor(Protocol):
    def submit(self, batch, model_worker_batch) -> OverlapExecutionResult: ...


@dataclass
class PendingOverlapResult:
    async_handle: AsyncResultHandle[object]
    future_indices_or_next_token_ids: torch.Tensor
    batch_snapshot: Optional[Any] = None
    live_batch_ref: Optional[Any] = None
    resolved_live_batch: Optional[Any] = None
    batch_relay: Optional[Any] = None
    relay_applied: bool = False
    future_indices: Optional[Any] = None
    next_token_ids: Optional[torch.Tensor] = None
    requires_resolve_before_mutation: bool = False
    is_resolved_for_mutation: bool = False
    requires_current_batch_resolve_for_sampling: bool = False
    # Deprecated: keep the field for compatibility while scheduler resolve timing
    # is migrated to mutation-time decisions.
    requires_resolve_before_next_schedule: bool = False
    became_mutation_ready: bool = False
    is_window_tail_materialized: bool = False
    is_window_tail_finalized: bool = False
    window_tail_round_created: Optional[int] = None
    extend_input_len_per_req: Optional[list[int]] = None
    extend_logprob_start_len_per_req: Optional[list[int]] = None
    _resolved_batch_result: Optional[GenerationBatchResult] = None

    def set_logprob_metadata(
        self,
        extend_input_len_per_req: Optional[list[int]],
        extend_logprob_start_len_per_req: Optional[list[int]],
    ):
        self.extend_input_len_per_req = extend_input_len_per_req
        self.extend_logprob_start_len_per_req = extend_logprob_start_len_per_req

    def resolve(self) -> GenerationBatchResult:
        if self._resolved_batch_result is None:
            worker_result = self.async_handle.resolve()
            if isinstance(worker_result, tuple):
                batch_result, batch_relay = worker_result
            else:
                batch_result, batch_relay = worker_result, None
            batch_result.extend_input_len_per_req = self.extend_input_len_per_req
            batch_result.extend_logprob_start_len_per_req = (
                self.extend_logprob_start_len_per_req
            )
            self.batch_relay = batch_relay
            self._resolved_batch_result = batch_result
        return self._resolved_batch_result

    def get_mutation_candidate_batch(self) -> Optional[Any]:
        if self.is_resolved_for_mutation and self.resolved_live_batch is not None:
            return self.resolved_live_batch
        return None

    def has_unresolved_live_batch(self) -> bool:
        return self.live_batch_ref is not None and not self.is_resolved_for_mutation

    def get_snapshot_batch(self) -> Optional[Any]:
        return self.batch_snapshot

    def must_resolve_before_scheduler_mutation(self) -> bool:
        return (
            self.requires_resolve_before_mutation and self.has_unresolved_live_batch()
        )

    def mark_window_tail_materialized(self) -> None:
        self.is_window_tail_materialized = True

    def mark_window_tail_finalized(self) -> None:
        self.is_window_tail_finalized = True

    def mark_became_mutation_ready(self) -> None:
        self.became_mutation_ready = True

    def mark_window_tail_round(self, round_id: int) -> None:
        self.window_tail_round_created = round_id
