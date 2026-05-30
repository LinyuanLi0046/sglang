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
        placeholder_ids = -handle.future_indices.indices.to(dtype=torch.int32)
        placeholder_token_list = placeholder_ids.unsqueeze(1).repeat(
            1, self.speculative_num_steps
        )
        return placeholder_ids, placeholder_token_list

    def resolve_placeholder(self, draft_input: EagleDraftInput) -> None:
        if (
            not draft_input.uses_future_placeholder
            or draft_input.last_verified_ids is None
            or draft_input.token_list is None
        ):
            return

        indices = torch.clamp(-draft_input.last_verified_ids.to(dtype=torch.int64), min=0)
        ready_mask = self.future_ready[indices]

        draft_input.last_verified_ids[:] = torch.where(
            ready_mask & (draft_input.last_verified_ids < 0),
            self.future_last_verified_ids[indices],
            draft_input.last_verified_ids,
        )
        draft_input.token_list[:] = torch.where(
            ready_mask.unsqueeze(1) & (draft_input.token_list < 0),
            self.future_token_list[indices],
            draft_input.token_list,
        )

    def store_resolved_state(
        self,
        handle: SpecFutureHandle,
        last_verified_ids: Optional[torch.Tensor],
        token_list: Optional[torch.Tensor],
    ) -> None:
        if (
            last_verified_ids is None
            or token_list is None
            or handle.future_indices.interval is None
        ):
            return

        intv = handle.future_indices.interval
        if self.future_buffer_len == 0:
            return

        self.future_last_verified_ids[intv] = last_verified_ids.to(torch.int32)
        self.future_token_list[intv] = token_list.to(torch.int32)
        self.future_ready[intv] = True


@dataclass
class SpecV2FutureRelay:
    future_indices: FutureIndices
    new_seq_lens: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None

    def apply_to_draft_input(self, draft_input: EagleDraftInput) -> EagleDraftInput:
        draft_input.attach_future_indices(self.future_indices)
        if draft_input.new_seq_lens is None:
            draft_input.new_seq_lens = self.new_seq_lens
        if draft_input.verify_done is None:
            draft_input.verify_done = self.verify_done
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
