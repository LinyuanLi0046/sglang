from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from sglang.srt.managers.utils import GenerationBatchResult


@dataclass
class OverlapExecutionResult:
    batch_result: GenerationBatchResult
    future_indices_or_next_token_ids: torch.Tensor


class OverlapExecutor(Protocol):
    def submit(self, batch, model_worker_batch) -> OverlapExecutionResult: ...
