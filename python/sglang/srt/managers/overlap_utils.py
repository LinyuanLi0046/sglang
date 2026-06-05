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


@dataclass(frozen=True)
class DecodePlaceholderLaunchSchema:
    token_list_width: int
    placeholder_dtype: torch.dtype = torch.int32
    capture_hidden_mode: Optional[object] = None
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1


def _get_batch_membership_size(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, FutureIndices):
        return int(value.indices.shape[0])
    if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return int(value.shape[0])
    return None


def _get_payload_batch_membership_size(
    payload_source: Optional["EagleDraftInput | EagleNextStepPayload"],
) -> Optional[int]:
    if payload_source is None:
        return None

    for field_name in (
        "next_step_seq_lens",
        "last_verified_ids",
        "token_list",
        "real_new_verified_id",
        "real_token_list",
        "new_seq_lens",
        "verified_id",
        "topk_p",
        "topk_index",
        "hidden_states",
    ):
        batch_size = _get_batch_membership_size(getattr(payload_source, field_name, None))
        if batch_size is not None:
            return batch_size
    return None


def _assert_future_indices_batch_membership_contract(
    future_indices: FutureIndices,
    payload_source: Optional["EagleDraftInput | EagleNextStepPayload"],
    context: str,
) -> None:
    expected_batch_size = _get_batch_membership_size(future_indices)
    actual_batch_size = _get_payload_batch_membership_size(payload_source)
    if expected_batch_size is None or actual_batch_size is None:
        return
    if expected_batch_size != actual_batch_size:
        raise RuntimeError(
            "future_indices batch-membership contract mismatch "
            f"during {context}: expected_batch_size={expected_batch_size}, "
            f"actual_batch_size={actual_batch_size}"
        )


def clone_future_indices_handle(
    future_indices: Optional[FutureIndices],
) -> Optional[FutureIndices]:
    if future_indices is None:
        return None
    return FutureIndices(
        indices=future_indices.indices,
        interval=future_indices.interval,
    )


def sanitize_decode_placeholder_handle_contract(
    draft_input: Optional["EagleDraftInput"],
) -> Optional["EagleDraftInput"]:
    if draft_input is None:
        return None

    # R6-L1: submit-time placeholder must stay a minimal future-handle carrier.
    draft_input.new_seq_lens = None
    draft_input.next_step_seq_lens = None
    draft_input.verify_done = None
    draft_input.verified_id = None
    draft_input.real_new_verified_id = None
    draft_input.real_token_list = None
    draft_input.topk_p = None
    draft_input.topk_index = None
    draft_input.hidden_states = None
    draft_input.future_topk_p_buf = None
    draft_input.future_topk_index_buf = None
    draft_input.future_hidden_states_buf = None
    draft_input.future_verified_id_buf = None
    draft_input.future_new_seq_lens_buf = None
    draft_input.future_next_step_seq_lens_buf = None
    draft_input.future_last_verified_ids_buf = None
    draft_input.future_token_list_buf = None
    draft_input.future_canonical_ready_buf = None
    return draft_input


def clone_decode_placeholder_handle_contract(
    draft_input: Optional["EagleDraftInput"],
    future_indices: Optional[FutureIndices] = None,
) -> Optional["EagleDraftInput"]:
    if draft_input is None:
        return None

    from sglang.srt.speculative.eagle_info import EagleDraftInput

    placeholder_carrier = EagleDraftInput(
        future_indices=clone_future_indices_handle(
            future_indices
            if future_indices is not None
            else getattr(draft_input, "future_indices", None)
        ),
        last_verified_ids=getattr(draft_input, "last_verified_ids", None),
        token_list=getattr(draft_input, "token_list", None),
        verify_token_num=getattr(draft_input, "verify_token_num", -1),
        capture_hidden_mode=getattr(draft_input, "capture_hidden_mode", None),
        num_tokens_per_req=getattr(draft_input, "num_tokens_per_req", -1),
        num_tokens_for_logprob_per_req=getattr(
            draft_input, "num_tokens_for_logprob_per_req", -1
        ),
    )
    return sanitize_decode_placeholder_handle_contract(placeholder_carrier)


def clone_spec_info_for_worker_launch(
    draft_input: Optional["EagleDraftInput"],
    future_indices: Optional[FutureIndices] = None,
    launch_schema: Optional[DecodePlaceholderLaunchSchema] = None,
) -> Optional["EagleDraftInput"]:
    if draft_input is None:
        return None

    from sglang.srt.speculative.eagle_info import EagleDraftInput

    capture_hidden_mode = getattr(draft_input, "capture_hidden_mode", None)
    if capture_hidden_mode is None and launch_schema is not None:
        capture_hidden_mode = launch_schema.capture_hidden_mode

    num_tokens_per_req = int(getattr(draft_input, "num_tokens_per_req", -1) or -1)
    if num_tokens_per_req <= 0 and launch_schema is not None:
        num_tokens_per_req = int(launch_schema.num_tokens_per_req)

    num_tokens_for_logprob_per_req = int(
        getattr(draft_input, "num_tokens_for_logprob_per_req", -1) or -1
    )
    if num_tokens_for_logprob_per_req <= 0 and launch_schema is not None:
        num_tokens_for_logprob_per_req = int(
            launch_schema.num_tokens_for_logprob_per_req
        )

    verify_token_num = int(getattr(draft_input, "verify_token_num", -1) or -1)
    if verify_token_num <= 0 and launch_schema is not None:
        verify_token_num = int(launch_schema.token_list_width) + 1

    worker_private_spec_info = EagleDraftInput(
        future_indices=clone_future_indices_handle(
            future_indices
            if future_indices is not None
            else getattr(draft_input, "future_indices", None)
        ),
        last_verified_ids=getattr(draft_input, "last_verified_ids", None),
        token_list=getattr(draft_input, "token_list", None),
        new_seq_lens=getattr(draft_input, "new_seq_lens", None),
        next_step_seq_lens=getattr(draft_input, "next_step_seq_lens", None),
        verify_done=getattr(draft_input, "verify_done", None),
        capture_hidden_mode=capture_hidden_mode,
        num_tokens_per_req=num_tokens_per_req,
        num_tokens_for_logprob_per_req=num_tokens_for_logprob_per_req,
        verify_token_num=verify_token_num,
    )
    return worker_private_spec_info


def build_decode_placeholder_canonical_draft_input(
    future_indices: FutureIndices,
    launch_schema: DecodePlaceholderLaunchSchema,
):
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    token_list_width = int(launch_schema.token_list_width)
    if token_list_width < 0:
        raise RuntimeError(
            "decode placeholder canonical token_list width must be non-negative: "
            f"actual={token_list_width}"
        )
    dtype = launch_schema.placeholder_dtype or torch.int32

    placeholder_ids = -future_indices.indices.to(dtype=dtype)
    draft_input_kwargs = {
        "future_indices": future_indices,
        "last_verified_ids": placeholder_ids,
        "token_list": placeholder_ids.unsqueeze(1)
        .expand(-1, token_list_width)
        .clone(),
        "capture_hidden_mode": launch_schema.capture_hidden_mode,
        "num_tokens_per_req": launch_schema.num_tokens_per_req,
        "num_tokens_for_logprob_per_req": (
            launch_schema.num_tokens_for_logprob_per_req
        ),
        "verify_token_num": token_list_width + 1,
    }

    return sanitize_decode_placeholder_handle_contract(
        EagleDraftInput(**draft_input_kwargs)
    )


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

    def _check_canonical_buf_width(
        self, draft_input: "EagleDraftInput | EagleNextStepPayload"
    ):
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return
        if self.last_verified_ids_buf is None or self.token_list_buf is None:
            self._lazy_init_canonical_buf(draft_input)
            return

        current_width = self.token_list_buf.shape[1]
        incoming_width = draft_input.token_list.shape[1]
        if incoming_width != current_width:
            raise RuntimeError(
                "canonical token_list width mismatch in FutureMap.store: "
                f"buf_width={current_width}, incoming_width={incoming_width}"
            )

    def _validate_canonical_payload_contract(
        self,
        payload_source: "EagleDraftInput | EagleNextStepPayload",
        *,
        context: str,
        canonical_only_decode: bool,
    ) -> None:
        last_verified_ids = getattr(payload_source, "last_verified_ids", None)
        token_list = getattr(payload_source, "token_list", None)
        if last_verified_ids is None or token_list is None:
            return
        if token_list.ndim != 2:
            raise RuntimeError(
                f"{context}: canonical token_list must be 2D, actual_ndim={token_list.ndim}"
            )
        if last_verified_ids.shape[0] != token_list.shape[0]:
            raise RuntimeError(
                f"{context}: canonical payload batch mismatch between "
                f"last_verified_ids={last_verified_ids.shape[0]} and "
                f"token_list={token_list.shape[0]}"
            )
        verify_token_num = int(getattr(payload_source, "verify_token_num", -1) or -1)
        if verify_token_num > 0 and token_list.shape[1] != verify_token_num - 1:
            raise RuntimeError(
                f"{context}: canonical verify width mismatch, "
                f"verify_token_num={verify_token_num}, "
                f"token_list_width={token_list.shape[1]}"
            )
        if canonical_only_decode and verify_token_num <= 0:
            raise RuntimeError(
                f"{context}: decode canonical-only payload must carry verify_token_num."
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Allocate stable batch-membership handles for the current batch rows."""
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
        _assert_future_indices_batch_membership_contract(
            future_indices,
            payload_source,
            "FutureMap.store_to_map_for_new_batch",
        )
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
        fallback_new_seq_lens = getattr(payload_source, "new_seq_lens", None)
        if next_step_seq_lens is None and not canonical_only_decode:
            next_step_seq_lens = fallback_new_seq_lens
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
            self._validate_canonical_payload_contract(
                payload_source,
                context="FutureMap.store_to_map_for_new_batch",
                canonical_only_decode=canonical_only_decode,
            )
            self._check_canonical_buf_width(payload_source)
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
        _assert_future_indices_batch_membership_contract(
            future_indices,
            draft_input,
            "FutureMap.replace_canonical_payload_with_future_placeholders",
        )
        if draft_input.last_verified_ids is None or draft_input.token_list is None:
            return

        placeholder_ids = -future_indices.indices.to(draft_input.last_verified_ids.dtype)
        token_width = draft_input.token_list.shape[1]
        draft_input.last_verified_ids = placeholder_ids
        draft_input.verify_token_num = token_width + 1
        draft_input.token_list = (
            placeholder_ids.unsqueeze(1).expand(-1, token_width).clone()
        )
