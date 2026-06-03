from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import traceback
from queue import Queue
from typing import Callable, Optional

import torch

from sglang.srt.managers.overlap_utils import (
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
    future_indices: FutureIndices
    next_draft_input: Optional["EagleDraftInput"] = None


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

    def _bind_thread_device(self) -> None:
        if self.gpu_id is None:
            return
        device_module = torch.get_device_module(self.device)
        if hasattr(device_module, "set_device"):
            device_module.set_device(self.gpu_id)

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
            self.future_map.store_to_map(future_indices, batch_result)
            if should_placeholder_replace and batch_result.next_draft_input is not None:
                self.future_map.replace_canonical_payload_with_future_placeholders(
                    future_indices, batch_result.next_draft_input
                )

            is_decode_placeholder_path = (
                model_worker_batch.forward_mode.is_decode()
                and not model_worker_batch.is_extend_in_batch
            )
            if not is_decode_placeholder_path:
                if batch_result.next_draft_input is None:
                    raise RuntimeError(
                        "Spec overlap worker requires next_draft_input before apply-ready."
                    )
                self.apply_queue.put(
                    SpecOverlapApplyState(
                        future_indices=future_indices,
                        next_draft_input=batch_result.next_draft_input,
                    )
                )

            batch_result.copy_done = torch.get_device_module(self.device).Event()
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
            placeholder_next_draft_input = (
                build_decode_placeholder_canonical_draft_input(
                    future_indices=future_indices,
                    token_list_width=getattr(
                        self.worker,
                        "speculative_num_steps",
                        self.worker.server_args.speculative_num_steps,
                    ),
                    template_draft_input=model_worker_batch.spec_info,
                )
            )

        return GenerationBatchResult(
            next_token_ids=-future_indices.indices,
            future_indices=future_indices,
            next_draft_input=placeholder_next_draft_input,
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

        apply_future_result(batch, apply_state)

    def resolve_last_batch_result(
        self,
        placeholder_result: GenerationBatchResult,
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        batch_result = self.output_queue.get()
        if isinstance(batch_result, RuntimeError):
            raise batch_result
        if batch_result.launch_done is not None:
            batch_result.launch_done.wait()

        batch_result.extend_input_len_per_req = placeholder_result.extend_input_len_per_req
        batch_result.extend_logprob_start_len_per_req = (
            placeholder_result.extend_logprob_start_len_per_req
        )
        return batch_result
