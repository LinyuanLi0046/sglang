from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.overlap.tp_worker_client_v2 import AsyncResultHandle


@dataclass
class OverlapExecutionResult:
    batch_result: Union[GenerationBatchResult, "PendingOverlapResult"]
    future_indices_or_next_token_ids: torch.Tensor


class OverlapExecutor(Protocol):
    def submit(self, batch, model_worker_batch) -> OverlapExecutionResult: ...


@dataclass
class PendingOverlapResult:
    async_handle: AsyncResultHandle[GenerationBatchResult]
    future_indices_or_next_token_ids: torch.Tensor
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
            batch_result = self.async_handle.resolve()
            batch_result.extend_input_len_per_req = self.extend_input_len_per_req
            batch_result.extend_logprob_start_len_per_req = (
                self.extend_logprob_start_len_per_req
            )
            self._resolved_batch_result = batch_result
        return self._resolved_batch_result
