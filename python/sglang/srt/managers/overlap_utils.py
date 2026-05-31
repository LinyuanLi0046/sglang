from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


@dataclass
class SpecFutureHandle:
    future_indices: FutureIndices


class SpecFutureStatePool:
    def __init__(
        self,
        future_buffer_len: int,
        speculative_num_steps: int,
        device: torch.device,
    ):
        self.future_buffer_len = future_buffer_len
        self.speculative_num_steps = speculative_num_steps
        self.device = device
        self.future_last_verified_ids = torch.zeros(
            (future_buffer_len,), dtype=torch.int32, device=device
        )
        self.future_token_list = torch.zeros(
            (future_buffer_len, speculative_num_steps),
            dtype=torch.int32,
            device=device,
        )
        self.future_ready = torch.zeros(
            (future_buffer_len,), dtype=torch.bool, device=device
        )

    def alloc_handle(self, future_indices: FutureIndices) -> SpecFutureHandle:
        if future_indices.interval is not None:
            self.future_ready[future_indices.interval] = False
        return SpecFutureHandle(future_indices=future_indices)

    def create_placeholder_inputs(
        self, handle: SpecFutureHandle
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Negative values are future handles only. They are not valid
        # speculative payloads until resolve_placeholder() replaces them
        # with worker-produced data.
        placeholder_ids = -handle.future_indices.indices.to(dtype=torch.int32)
        placeholder_token_list = placeholder_ids.unsqueeze(1).repeat(
            1, self.speculative_num_steps
        )
        return placeholder_ids, placeholder_token_list

    def _validate_resolved_state(
        self,
        handle: SpecFutureHandle,
        last_verified_ids: Optional[torch.Tensor],
        token_list: Optional[torch.Tensor],
    ) -> None:
        if handle.future_indices.interval is None:
            raise ValueError("SpecFutureStatePool requires a concrete future interval.")
        if last_verified_ids is None or token_list is None:
            raise ValueError("Resolved future state requires last_verified_ids and token_list.")
        if last_verified_ids.dtype.is_floating_point or last_verified_ids.dtype == torch.bool:
            raise TypeError(
                f"last_verified_ids must use an integer dtype, got {last_verified_ids.dtype}."
            )
        if token_list.dtype.is_floating_point or token_list.dtype == torch.bool:
            raise TypeError(f"token_list must use an integer dtype, got {token_list.dtype}.")
        if last_verified_ids.ndim != 1:
            raise ValueError(
                f"last_verified_ids must be rank-1, got ndim={last_verified_ids.ndim}."
            )
        if token_list.ndim != 2:
            raise ValueError(f"token_list must be rank-2, got ndim={token_list.ndim}.")
        if token_list.shape[0] != last_verified_ids.shape[0]:
            raise ValueError(
                "Resolved future state has inconsistent batch dimension: "
                f"token_list.shape[0]={token_list.shape[0]} vs "
                f"last_verified_ids.shape[0]={last_verified_ids.shape[0]}."
            )
        if token_list.shape[1] != self.speculative_num_steps:
            raise ValueError(
                "Resolved future state has inconsistent token width: "
                f"token_list.shape[1]={token_list.shape[1]} vs "
                f"speculative_num_steps={self.speculative_num_steps}."
            )
        if last_verified_ids.shape[0] != len(handle.future_indices.indices):
            raise ValueError(
                "Resolved future state batch size does not match future handle: "
                f"{last_verified_ids.shape[0]} vs {len(handle.future_indices.indices)}."
            )

    def resolve_placeholder(self, draft_input: EagleDraftInput) -> None:
        if (
            not draft_input.uses_future_placeholder
            or draft_input.last_verified_ids is None
            or draft_input.token_list is None
        ):
            return

        if draft_input.last_verified_ids.ndim != 1:
            raise ValueError(
                "Future placeholder last_verified_ids must be rank-1, "
                f"got ndim={draft_input.last_verified_ids.ndim}."
            )
        if draft_input.token_list.ndim != 2:
            raise ValueError(
                "Future placeholder token_list must be rank-2, "
                f"got ndim={draft_input.token_list.ndim}."
            )
        if draft_input.token_list.shape[0] != draft_input.last_verified_ids.shape[0]:
            raise ValueError(
                "Future placeholder batch dimensions are inconsistent: "
                f"token_list.shape[0]={draft_input.token_list.shape[0]} vs "
                f"last_verified_ids.shape[0]={draft_input.last_verified_ids.shape[0]}."
            )
        if draft_input.token_list.shape[1] != self.speculative_num_steps:
            raise ValueError(
                "Future placeholder token width is inconsistent: "
                f"token_list.shape[1]={draft_input.token_list.shape[1]} vs "
                f"speculative_num_steps={self.speculative_num_steps}."
            )

        needs_resolve = draft_input.last_verified_ids < 0
        indices = torch.clamp(-draft_input.last_verified_ids.to(dtype=torch.int64), min=0)
        ready_mask = self.future_ready[indices]

        draft_input.last_verified_ids[:] = torch.where(
            ready_mask & needs_resolve,
            self.future_last_verified_ids[indices],
            draft_input.last_verified_ids,
        )
        draft_input.token_list[:] = torch.where(
            ready_mask.unsqueeze(1)
            & needs_resolve.unsqueeze(1)
            & (draft_input.token_list < 0),
            self.future_token_list[indices],
            draft_input.token_list,
        )

    def store_resolved_state(
        self,
        handle: SpecFutureHandle,
        last_verified_ids: Optional[torch.Tensor],
        token_list: Optional[torch.Tensor],
    ) -> None:
        if self.future_buffer_len == 0:
            return
        self._validate_resolved_state(handle, last_verified_ids, token_list)

        intv = handle.future_indices.interval
        self.future_last_verified_ids[intv] = last_verified_ids.to(torch.int32)
        self.future_token_list[intv] = token_list.to(torch.int32)
        self.future_ready[intv] = True

    def is_ready(self, future_indices: FutureIndices) -> bool:
        if self.future_buffer_len == 0:
            return False

        intv = future_indices.interval
        if intv is not None:
            ready = self.future_ready[intv]
        else:
            indices = future_indices.indices
            if indices.numel() == 0:
                return False
            ready = self.future_ready[indices]

        return bool(ready.numel() > 0 and torch.all(ready).item())


@dataclass
class SpecV2FutureRelay:
    future_indices: FutureIndices
    new_seq_lens: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None
    last_verified_ids: Optional[torch.Tensor] = None
    token_list: Optional[torch.Tensor] = None
    has_real_future_payload: Optional[bool] = None
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    verified_id: Optional[torch.Tensor] = None

    def apply_to_draft_input(self, draft_input: EagleDraftInput) -> EagleDraftInput:
        draft_input.attach_future_indices(self.future_indices)
        if self.new_seq_lens is not None:
            draft_input.new_seq_lens = self.new_seq_lens
        if self.verify_done is not None:
            draft_input.verify_done = self.verify_done
        if self.last_verified_ids is not None:
            draft_input.uses_future_placeholder = True
            draft_input.last_verified_ids = self.last_verified_ids
        if self.token_list is not None:
            draft_input.uses_future_placeholder = True
            draft_input.token_list = self.token_list
        if self.has_real_future_payload is not None:
            draft_input.has_real_future_payload = self.has_real_future_payload
        if self.topk_p is not None:
            draft_input.topk_p = self.topk_p
        if self.topk_index is not None:
            draft_input.topk_index = self.topk_index
        if self.hidden_states is not None:
            draft_input.hidden_states = self.hidden_states
        if self.verified_id is not None:
            draft_input.verified_id = self.verified_id
        return draft_input

    def apply_to_batch(
        self, batch: ScheduleBatch, draft_input: EagleDraftInput
    ) -> EagleDraftInput:
        batch.spec_info = self.apply_to_draft_input(draft_input)
        # The future value is consumed during next decode preparation.
        # Keep the current strict seq_lens synchronization semantics unchanged.
        batch.seq_lens = self.new_seq_lens
        return batch.spec_info


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: Optional[SpeculativeAlgorithm] = None,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False
            self.spec_future_state_pool: Optional[SpecFutureStatePool] = None

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        verified_id0 = draft_input.verified_id[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.verified_id_buf = torch.empty(
            (self.future_buffer_len, *verified_id0.shape),
            dtype=verified_id0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)
        else:
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            self.attach_spec_future_buffers(draft_input)
            self.resolve_spec_future_placeholder(draft_input)

    def ensure_spec_future_state_pool(
        self, speculative_num_steps: int
    ) -> SpecFutureStatePool:
        if self.spec_algo.is_none():
            raise RuntimeError(
                "SpecFutureStatePool is only available for speculative decoding."
            )

        if self.spec_future_state_pool is None:
            self.spec_future_state_pool = SpecFutureStatePool(
                future_buffer_len=self.future_buffer_len,
                speculative_num_steps=speculative_num_steps,
                device=self.device,
            )

        return self.spec_future_state_pool

    def resolve_spec_future_placeholder(self, draft_input: EagleDraftInput) -> None:
        if self.spec_future_state_pool is None:
            return
        self.spec_future_state_pool.resolve_placeholder(draft_input)

    def store_spec_future_state(
        self, future_indices: FutureIndices, draft_input: Optional[EagleDraftInput]
    ) -> None:
        if self.spec_future_state_pool is None or draft_input is None:
            return

        self.spec_future_state_pool.store_resolved_state(
            SpecFutureHandle(future_indices=future_indices),
            last_verified_ids=draft_input.last_verified_ids,
            token_list=draft_input.token_list,
        )

    def peek_spec_future_state(
        self, future_indices: FutureIndices
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.spec_future_state_pool is None:
            return None, None, None
        intv = future_indices.interval
        if intv is None or self.is_empty_slice(intv):
            return None, None, None
        return (
            self.spec_future_state_pool.future_last_verified_ids[intv],
            self.spec_future_state_pool.future_token_list[intv],
            self.spec_future_state_pool.future_ready[intv],
        )

    def has_ready_spec_future_state(self, future_indices: FutureIndices) -> bool:
        if self.spec_future_state_pool is None:
            return False
        return self.spec_future_state_pool.is_ready(future_indices)

    def peek_replay_future_state(
        self, future_indices: FutureIndices
    ) -> dict[str, Optional[torch.Tensor]]:
        intv = future_indices.interval
        if intv is None or self.is_empty_slice(intv):
            return {
                "topk_p": None,
                "topk_index": None,
                "verified_id": None,
                "new_seq_lens": None,
                "hidden_states": None,
            }
        hidden_states = None
        if spec_need_hidden_states() and self.hidden_states_buf is not None:
            hidden_states = self.hidden_states_buf[intv]
        return {
            "topk_p": None if self.topk_p_buf is None else self.topk_p_buf[intv],
            "topk_index": None
            if self.topk_index_buf is None
            else self.topk_index_buf[intv],
            "verified_id": None
            if self.verified_id_buf is None
            else self.verified_id_buf[intv],
            "new_seq_lens": None
            if self.new_seq_lens_buf is None
            else self.new_seq_lens_buf[intv],
            "hidden_states": hidden_states,
        }

    def attach_spec_future_buffers(self, draft_input: EagleDraftInput):
        indices = draft_input.future_indices.indices
        # The indices tensor was allocated on the default stream but is
        # used here on the forward stream. Meanwhile, the old spec_info
        # holding this tensor will lose all Python references (replaced at
        # model_worker_batch.spec_info and batch.spec_info), so the
        # caching allocator (torch GC) could reclaim the memory before
        # the GPU finishes reading it.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        draft_input.future_topk_p_buf = self.topk_p_buf
        draft_input.future_topk_index_buf = self.topk_index_buf
        draft_input.future_verified_id_buf = self.verified_id_buf
        draft_input.future_new_seq_lens_buf = self.new_seq_lens_buf
        if spec_need_hidden_states():
            draft_input.future_hidden_states_buf = self.hidden_states_buf

    def build_spec_v2_future_relay(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ) -> SpecV2FutureRelay:
        draft_input: EagleDraftInput = batch_result.next_draft_input
        return SpecV2FutureRelay(
            future_indices=future_indices,
            new_seq_lens=draft_input.new_seq_lens,
            verify_done=draft_input.verify_done,
            last_verified_ids=draft_input.last_verified_ids,
            token_list=draft_input.token_list,
            has_real_future_payload=draft_input.has_real_future_payload,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            verified_id=draft_input.verified_id,
        )

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.verified_id_buf[intv] = draft_input.verified_id
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states
