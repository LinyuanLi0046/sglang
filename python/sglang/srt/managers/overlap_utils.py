from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleNextStepPayload
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


def build_decode_placeholder_canonical_draft_input(
    future_indices: FutureIndices,
    token_list_width: int,
    template_draft_input: Optional["EagleDraftInput"] = None,
):
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    dtype = torch.int32
    for attr_name in ("last_verified_ids", "verified_id", "real_new_verified_id"):
        template_tensor = (
            getattr(template_draft_input, attr_name, None)
            if template_draft_input is not None
            else None
        )
        if template_tensor is not None:
            dtype = template_tensor.dtype
            break

    placeholder_ids = -future_indices.indices.to(dtype=dtype)
    draft_input_kwargs = {
        "future_indices": future_indices,
        "last_verified_ids": placeholder_ids,
        "token_list": placeholder_ids.unsqueeze(1)
        .expand(-1, token_list_width)
        .clone(),
    }
    if template_draft_input is not None:
        draft_input_kwargs["capture_hidden_mode"] = getattr(
            template_draft_input, "capture_hidden_mode", None
        )
        draft_input_kwargs["num_tokens_per_req"] = getattr(
            template_draft_input, "num_tokens_per_req", -1
        )
        draft_input_kwargs["num_tokens_for_logprob_per_req"] = getattr(
            template_draft_input, "num_tokens_for_logprob_per_req", -1
        )

    return EagleDraftInput(**draft_input_kwargs)


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
            self.topk_p_buf = None
            self.topk_index_buf = None
            self.verified_id_buf = None
            self.new_seq_lens_buf = None
            self.next_step_seq_lens_buf = None
            self.hidden_states_buf = None
            self.last_verified_ids_buf = None
            self.token_list_buf = None
            self.canonical_ready_buf = None

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

        self._lazy_init_next_step_seq_lens_buf(draft_input)
        self._lazy_init_canonical_buf(draft_input)

    def _lazy_init_next_step_seq_lens_buf(
        self, draft_input: "EagleDraftInput | EagleNextStepPayload"
    ):
        next_step_seq_lens0 = getattr(draft_input, "next_step_seq_lens", None)
        if next_step_seq_lens0 is None:
            next_step_seq_lens0 = getattr(draft_input, "new_seq_lens", None)
        if next_step_seq_lens0 is None:
            return
        next_step_seq_lens0 = next_step_seq_lens0[0]
        self.next_step_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *next_step_seq_lens0.shape),
            dtype=next_step_seq_lens0.dtype,
            device=self.device,
        )

    def _lazy_init_canonical_buf(self, draft_input: EagleDraftInput):
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return

        last_verified_ids0 = draft_input.last_verified_ids[0]
        token_list0 = draft_input.token_list[0]
        self.last_verified_ids_buf = torch.empty(
            (self.future_buffer_len, *last_verified_ids0.shape),
            dtype=draft_input.last_verified_ids.dtype,
            device=self.device,
        )
        self.token_list_buf = torch.empty(
            (self.future_buffer_len, *token_list0.shape),
            dtype=draft_input.token_list.dtype,
            device=self.device,
        )
        self.canonical_ready_buf = torch.zeros(
            (self.future_buffer_len,), dtype=torch.bool, device=self.device
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
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # model_worker_batch.spec_info and batch.spec_info), so the
            # caching allocator (torch GC) could reclaim the memory before
            # the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            is_decode_placeholder_path = (
                model_worker_batch.forward_mode.is_decode()
                and not model_worker_batch.is_extend_in_batch
            )
            if is_decode_placeholder_path:
                draft_input.future_topk_p_buf = None
                draft_input.future_topk_index_buf = None
                draft_input.future_verified_id_buf = None
                draft_input.future_new_seq_lens_buf = None
                draft_input.future_next_step_seq_lens_buf = self.next_step_seq_lens_buf
                if spec_need_hidden_states():
                    draft_input.future_hidden_states_buf = None
                draft_input.future_last_verified_ids_buf = self.last_verified_ids_buf
                draft_input.future_token_list_buf = self.token_list_buf
                draft_input.future_canonical_ready_buf = self.canonical_ready_buf
                return
            if self.buf_initialized:
                draft_input.future_topk_p_buf = self.topk_p_buf
                draft_input.future_topk_index_buf = self.topk_index_buf
                draft_input.future_verified_id_buf = self.verified_id_buf
                draft_input.future_new_seq_lens_buf = self.new_seq_lens_buf
                draft_input.future_next_step_seq_lens_buf = self.next_step_seq_lens_buf
                if spec_need_hidden_states():
                    draft_input.future_hidden_states_buf = self.hidden_states_buf
            else:
                draft_input.future_topk_p_buf = None
                draft_input.future_topk_index_buf = None
                draft_input.future_verified_id_buf = None
                draft_input.future_new_seq_lens_buf = None
                draft_input.future_next_step_seq_lens_buf = None
                if spec_need_hidden_states():
                    draft_input.future_hidden_states_buf = None
            draft_input.future_last_verified_ids_buf = self.last_verified_ids_buf
            draft_input.future_token_list_buf = self.token_list_buf
            draft_input.future_canonical_ready_buf = self.canonical_ready_buf

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self,
        future_indices: FutureIndices,
        batch_result: GenerationBatchResult,
        canonical_only_decode: bool = False,
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            payload_source = batch_result.next_step_payload or batch_result.next_draft_input
            self.store_to_map_for_new_batch(
                future_indices,
                payload_source,
                canonical_only_decode=canonical_only_decode,
            )

    def store_to_map_for_new_batch(
        self,
        future_indices: FutureIndices,
        payload_source: "EagleDraftInput | EagleNextStepPayload",
        canonical_only_decode: bool = False,
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        has_canonical_payload = (
            payload_source is not None
            and getattr(payload_source, "last_verified_ids", None) is not None
            and getattr(payload_source, "token_list", None) is not None
        )
        if self.canonical_ready_buf is not None:
            self.canonical_ready_buf[intv] = False
        next_step_seq_lens = getattr(payload_source, "next_step_seq_lens", None)
        if next_step_seq_lens is None:
            next_step_seq_lens = getattr(payload_source, "new_seq_lens", None)
        if canonical_only_decode:
            if not has_canonical_payload:
                raise RuntimeError(
                    "decode steady-state canonical-only store requires last_verified_ids/token_list"
                )
            if next_step_seq_lens is None:
                raise RuntimeError(
                    "decode steady-state canonical-only store requires next_step_seq_lens relay"
                )
        if has_canonical_payload:
            if self.last_verified_ids_buf is None or self.token_list_buf is None:
                self._lazy_init_canonical_buf(payload_source)
            self.last_verified_ids_buf[intv] = payload_source.last_verified_ids
            self.token_list_buf[intv] = payload_source.token_list
            if next_step_seq_lens is not None:
                if self.next_step_seq_lens_buf is None:
                    self._lazy_init_next_step_seq_lens_buf(payload_source)
                self.next_step_seq_lens_buf[intv] = next_step_seq_lens
            self.canonical_ready_buf[intv] = True
            return

        if not self.buf_initialized:
            self._lazy_init_buf(payload_source)

        self.topk_p_buf[intv] = payload_source.topk_p
        self.topk_index_buf[intv] = payload_source.topk_index
        self.verified_id_buf[intv] = payload_source.verified_id
        self.new_seq_lens_buf[intv] = payload_source.new_seq_lens
        self.next_step_seq_lens_buf[intv] = getattr(
            payload_source, "next_step_seq_lens", payload_source.new_seq_lens
        )
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = payload_source.hidden_states

    def replace_canonical_payload_with_future_placeholders(
        self,
        future_indices: FutureIndices,
        draft_input: EagleDraftInput,
    ):
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return

        placeholder_ids = -future_indices.indices.to(draft_input.last_verified_ids.dtype)
        token_width = draft_input.token_list.shape[1]
        draft_input.last_verified_ids = placeholder_ids
        draft_input.token_list = (
            placeholder_ids.unsqueeze(1).expand(-1, token_width).clone()
        )
