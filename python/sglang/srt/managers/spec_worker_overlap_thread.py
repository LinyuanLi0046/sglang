from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import traceback
from queue import Queue
from typing import Callable, Optional

import torch

from sglang.srt.managers.overlap_utils import (
    DecodePlaceholderLaunchSchema,
    FutureIndices,
    FutureMap,
    build_decode_placeholder_canonical_draft_input,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult

if False:  # pragma: no cover
    from sglang.srt.speculative.eagle_info import EagleDraftInput

logger = logging.getLogger(__name__)


@dataclass
class SpecOverlapApplyState:
    batch_uid: int
    future_indices: FutureIndices
    next_draft_input: Optional["EagleDraftInput"] = None
    future_next_step_seq_lens_buf: Optional[torch.Tensor] = None
    future_canonical_ready_buf: Optional[torch.Tensor] = None
    requires_scheduler_apply: bool = True


class SpecModelWorkerOverlapClient:
    """Spec-v2 overlap client that moves resolve/store/copy off the scheduler thread."""

    def __init__(self, worker, future_map: FutureMap):
        self.worker = worker
        self.future_map = future_map
        self.device = worker.device
        self.gpu_id = getattr(getattr(worker, "target_worker", None), "gpu_id", None)
        self.scheduler_stream = None
        self.thread_exception: Optional[RuntimeError] = None

        num_decode_steps = max(1, self.worker.server_args.num_continuous_decode_steps)
        self.keep_batch_reference_num = 2 * num_decode_steps

        self.input_queue: Queue = Queue()
        self.apply_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            name="spec-overlap-worker",
            daemon=True,
        )
        self.forward_thread.start()

    def _raise_if_thread_failed(self) -> None:
        if self.thread_exception is not None:
            raise self.thread_exception

    def _validate_apply_state_contract(self, apply_state: SpecOverlapApplyState) -> None:
        if apply_state.future_indices is None:
            raise RuntimeError("Spec overlap apply state missing future_indices.")

        if apply_state.requires_scheduler_apply:
            if apply_state.next_draft_input is None:
                raise RuntimeError(
                    "Spec overlap apply state requiring scheduler apply must carry "
                    "next_draft_input."
                )
            return

        if apply_state.future_next_step_seq_lens_buf is None:
            raise RuntimeError(
                "Decode canonical-only apply state must carry "
                "future_next_step_seq_lens_buf."
            )
        if apply_state.future_canonical_ready_buf is None:
            raise RuntimeError(
                "Decode canonical-only apply state must carry "
                "future_canonical_ready_buf."
            )

    def _bind_thread_device(self) -> None:
        if self.gpu_id is None:
            return
        device_module = torch.get_device_module(self.device)
        if hasattr(device_module, "set_device"):
            device_module.set_device(self.gpu_id)

    def _build_decode_placeholder_launch_schema(
        self, model_worker_batch: ModelWorkerBatch
    ) -> DecodePlaceholderLaunchSchema:
        explicit_launch_schema = getattr(
            model_worker_batch, "decode_placeholder_launch_schema", None
        )
        if explicit_launch_schema is not None:
            return explicit_launch_schema

        default_num_tokens_per_req = getattr(self.worker, "speculative_num_steps", 0) + 1
        default_num_tokens_for_logprob_per_req = default_num_tokens_per_req
        existing_spec_info = getattr(model_worker_batch, "spec_info", None)
        get_adjust_token_coefficient = getattr(
            existing_spec_info, "get_spec_adjust_token_coefficient", None
        )
        if callable(get_adjust_token_coefficient):
            num_tokens_per_req, num_tokens_for_logprob_per_req = (
                get_adjust_token_coefficient()
            )
            if num_tokens_per_req is not None and num_tokens_per_req > 0:
                default_num_tokens_per_req = int(num_tokens_per_req)
            if (
                num_tokens_for_logprob_per_req is not None
                and num_tokens_for_logprob_per_req > 0
            ):
                default_num_tokens_for_logprob_per_req = int(
                    num_tokens_for_logprob_per_req
                )

        placeholder_dtype = getattr(
            getattr(model_worker_batch, "input_ids", None), "dtype", torch.int32
        )
        token_list_width = (
            getattr(
                self.worker,
                "speculative_num_draft_tokens",
                self.worker.server_args.speculative_num_draft_tokens,
            )
            - 1
        )
        return DecodePlaceholderLaunchSchema(
            token_list_width=token_list_width,
            placeholder_dtype=placeholder_dtype,
            capture_hidden_mode=getattr(
                model_worker_batch, "capture_hidden_mode", None
            ),
            num_tokens_per_req=default_num_tokens_per_req,
            num_tokens_for_logprob_per_req=default_num_tokens_for_logprob_per_req,
        )

    def forward_thread_func(self) -> None:
        try:
            self._bind_thread_device()
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception as exc:  # pragma: no cover - fatal runtime path
            tb = traceback.format_exc()
            self.thread_exception = RuntimeError(
                f"SpecModelWorkerOverlapClient failed: {exc}\n{tb}"
            )
            logger.error("%s", self.thread_exception)
            self.apply_queue.put(self.thread_exception)
            self.output_queue.put(self.thread_exception)

    @torch.no_grad()
    def forward_thread_func_(self) -> None:
        batch_pt = 0
        batch_lists = [None] * self.keep_batch_reference_num

        while True:
            item = self.input_queue.get()
            if item is None:
                break

            (
                model_worker_batch,
                future_indices,
                return_logprob,
                needs_hidden_states,
                copy_input_token_logprobs,
                should_placeholder_replace,
            ) = item
            if model_worker_batch is None:
                break

            batch_lists[batch_pt % self.keep_batch_reference_num] = model_worker_batch
            batch_pt += 1

            self.future_map.resolve_future(model_worker_batch)
            if (
                model_worker_batch.forward_mode.is_decode()
                and model_worker_batch.spec_algorithm is not None
                and model_worker_batch.spec_algorithm.supports_spec_v2()
                and model_worker_batch.spec_info is not None
            ):
                model_worker_batch.spec_info.prepare_decode_live_view(
                    model_worker_batch,
                    allow_compat_fallback=False,
                )
            batch_result = self.worker.forward_batch_generation(
                model_worker_batch,
                launch_done=model_worker_batch.launch_done,
            )
            if batch_result.delay_sample_func is not None:
                raise RuntimeError(
                    "Spec overlap worker does not support delayed sample in Patch 45."
                )

            batch_result.future_indices = future_indices
            batch_result.launch_done = model_worker_batch.launch_done
            is_decode_placeholder_path = (
                model_worker_batch.forward_mode.is_decode()
                and not model_worker_batch.is_extend_in_batch
            )
            self.future_map.store_to_map(
                future_indices,
                batch_result,
                canonical_only_decode=is_decode_placeholder_path,
            )
            if should_placeholder_replace and batch_result.next_draft_input is not None:
                self.future_map.replace_canonical_payload_with_future_placeholders(
                    future_indices, batch_result.next_draft_input
                )

            if not is_decode_placeholder_path:
                if batch_result.next_draft_input is None:
                    raise RuntimeError(
                        "Spec overlap worker requires next_draft_input before apply-ready."
                    )
                self.apply_queue.put(
                    SpecOverlapApplyState(
                        batch_uid=model_worker_batch.scheduler_batch_uid,
                        future_indices=future_indices,
                        next_draft_input=batch_result.next_draft_input,
                    )
                )
            else:
                future_next_step_seq_lens_buf = self.future_map.next_step_seq_lens_buf
                future_canonical_ready_buf = self.future_map.canonical_ready_buf
                if future_next_step_seq_lens_buf is None:
                    raise RuntimeError(
                        "decode steady-state overlap worker must produce "
                        "future next-step seq-lens companion for canonical "
                        "placeholder apply"
                    )
                if future_canonical_ready_buf is None:
                    raise RuntimeError(
                        "decode steady-state overlap worker must produce "
                        "canonical-ready companion for canonical placeholder apply"
                    )
                self.apply_queue.put(
                    SpecOverlapApplyState(
                        batch_uid=model_worker_batch.scheduler_batch_uid,
                        future_indices=future_indices,
                        future_next_step_seq_lens_buf=future_next_step_seq_lens_buf,
                        future_canonical_ready_buf=future_canonical_ready_buf,
                        requires_scheduler_apply=False,
                    )
                )

            batch_result.copy_done = torch.get_device_module(self.device).Event()
            batch_result.scheduler_batch_uid = model_worker_batch.scheduler_batch_uid
            batch_result.copy_to_cpu_for_spec_overlap(
                return_logprob=return_logprob,
                needs_hidden_states=needs_hidden_states,
                copy_input_token_logprobs=copy_input_token_logprobs,
            )
            self.output_queue.put(batch_result)

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        model_worker_batch.sampling_info = model_worker_batch.sampling_info.copy_for_forward()

        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if getattr(self.device, "type", self.device) == "npu" and hasattr(torch, "npu"):
            torch.npu.set_stream_limit(self.scheduler_stream, 8, 16)
        if hasattr(self.scheduler_stream, "synchronize"):
            self.scheduler_stream.synchronize()

        bs = len(model_worker_batch.seq_lens)
        future_indices = self.future_map.alloc_future_indices(bs)
        needs_hidden_states = bool(getattr(model_worker_batch, "return_hidden_states", False))
        copy_input_token_logprobs = bool(
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        )
        should_placeholder_replace = (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        )
        self.input_queue.put(
            (
                model_worker_batch,
                future_indices,
                model_worker_batch.return_logprob,
                needs_hidden_states,
                copy_input_token_logprobs,
                should_placeholder_replace,
            )
        )

        placeholder_next_draft_input = None
        if model_worker_batch.forward_mode.is_decode() and not model_worker_batch.is_extend_in_batch:
            launch_schema = self._build_decode_placeholder_launch_schema(
                model_worker_batch
            )
            placeholder_next_draft_input = (
                build_decode_placeholder_canonical_draft_input(
                    future_indices=future_indices,
                    launch_schema=launch_schema,
                )
            )

        return GenerationBatchResult(
            next_token_ids=-future_indices.indices,
            future_indices=future_indices,
            next_draft_input=placeholder_next_draft_input,
            scheduler_batch_uid=model_worker_batch.scheduler_batch_uid,
        )

    def ensure_batch_state_ready(
        self,
        batch: ScheduleBatch,
        apply_future_result: Callable[[ScheduleBatch, SpecOverlapApplyState], None],
    ) -> None:
        self._raise_if_thread_failed()

        apply_state = self.apply_queue.get()
        if isinstance(apply_state, RuntimeError):
            raise apply_state

        self._validate_apply_state_contract(apply_state)
        if apply_state.batch_uid is not None and apply_state.batch_uid != id(batch):
            raise RuntimeError(
                "Spec overlap apply state batch_uid mismatch: "
                f"expected={id(batch)}, actual={apply_state.batch_uid}"
            )
        apply_future_result(batch, apply_state)

    def resolve_last_batch_result(
        self,
        placeholder_result: GenerationBatchResult,
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        batch_result = self.output_queue.get()
        if isinstance(batch_result, RuntimeError):
            raise batch_result
        if (
            placeholder_result.scheduler_batch_uid is not None
            and batch_result.scheduler_batch_uid is not None
            and batch_result.scheduler_batch_uid
            != placeholder_result.scheduler_batch_uid
        ):
            raise RuntimeError(
                "Spec overlap output result batch_uid mismatch: "
                f"expected={placeholder_result.scheduler_batch_uid}, "
                f"actual={batch_result.scheduler_batch_uid}"
            )
        if batch_result.launch_done is not None:
            batch_result.launch_done.wait()

        batch_result.extend_input_len_per_req = placeholder_result.extend_input_len_per_req
        batch_result.extend_logprob_start_len_per_req = (
            placeholder_result.extend_logprob_start_len_per_req
        )
        return batch_result
