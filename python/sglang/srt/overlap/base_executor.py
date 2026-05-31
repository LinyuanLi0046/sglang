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
    batch_relay: Optional[Any] = None


class OverlapExecutor(Protocol):
    def submit(self, batch, model_worker_batch) -> OverlapExecutionResult: ...


@dataclass
class PendingOverlapResult:
    async_handle: AsyncResultHandle[object]
    future_indices_or_next_token_ids: torch.Tensor
    batch_relay: Optional[Any] = None
    requires_current_batch_resolve_for_sampling: bool = False
    requires_resolve_before_next_schedule: bool = False
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
