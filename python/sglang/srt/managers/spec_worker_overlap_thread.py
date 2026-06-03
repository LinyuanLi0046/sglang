from __future__ import annotations

import logging
import threading
import traceback
from queue import Queue
from typing import Callable, Deque, Optional

import torch

from sglang.srt.managers.overlap_utils import FutureIndices, FutureMap
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


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
                should_placeholder_replace,
            ) = item
            if model_worker_batch is None:
                break

            batch_lists[batch_pt % self.keep_batch_reference_num] = model_worker_batch
            batch_pt += 1

            self.future_map.resolve_future(model_worker_batch)
            batch_result = self.worker.forward_batch_generation(model_worker_batch)
            if batch_result.delay_sample_func is not None:
                raise RuntimeError(
                    "Spec overlap worker does not support delayed sample in Patch 45."
                )

            batch_result.future_indices = future_indices
            self.future_map.store_to_map(future_indices, batch_result)
            if should_placeholder_replace and batch_result.next_draft_input is not None:
                self.future_map.replace_canonical_payload_with_future_placeholders(
                    future_indices, batch_result.next_draft_input
                )

            # Expose the minimal state needed by scheduler before the full
            # D2H result becomes available.
            self.apply_queue.put(batch_result)

            batch_result.copy_done = torch.get_device_module(self.device).Event()
            batch_result.copy_to_cpu(return_logprob=return_logprob)
            self.output_queue.put(batch_result)

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        model_worker_batch.sampling_info = model_worker_batch.sampling_info.copy_for_forward()

        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if hasattr(self.scheduler_stream, "synchronize"):
            self.scheduler_stream.synchronize()

        bs = len(model_worker_batch.seq_lens)
        future_indices = self.future_map.alloc_future_indices(bs)
        should_placeholder_replace = (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        )
        self.input_queue.put(
            (
                model_worker_batch,
                future_indices,
                model_worker_batch.return_logprob,
                should_placeholder_replace,
            )
        )

        return GenerationBatchResult(
            next_token_ids=-future_indices.indices,
            future_indices=future_indices,
        )

    def ensure_batch_state_ready(
        self,
        batch: ScheduleBatch,
        apply_future_result: Callable[
            [ScheduleBatch, GenerationBatchResult, FutureIndices], None
        ],
    ) -> None:
        self._raise_if_thread_failed()

        batch_result = self.apply_queue.get()
        if isinstance(batch_result, RuntimeError):
            raise batch_result

        apply_future_result(batch, batch_result, batch_result.future_indices)

    def resolve_last_batch_result(
        self,
        placeholder_result: GenerationBatchResult,
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        batch_result = self.output_queue.get()
        if isinstance(batch_result, RuntimeError):
            raise batch_result

        batch_result.extend_input_len_per_req = placeholder_result.extend_input_len_per_req
        batch_result.extend_logprob_start_len_per_req = (
            placeholder_result.extend_logprob_start_len_per_req
        )
        return batch_result
