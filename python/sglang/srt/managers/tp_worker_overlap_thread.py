from __future__ import annotations

import dataclasses
import logging
import math
import threading
import traceback
from queue import Queue
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


def resolve_future_token_ids(
    input_ids: torch.Tensor, future_token_ids_map: torch.Tensor
) -> None:
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


class TpModelWorkerOverlapClient:
    """Non-spec overlap client that moves submit/reclaim ownership to a worker thread."""

    def __init__(self, worker: TpModelWorker):
        self.worker = worker
        self.model_config = worker.model_config
        self.model_runner = worker.model_runner
        self.max_running_requests = worker.max_running_requests
        self.device = worker.device
        self.gpu_id = worker.gpu_id
        self.cur_sampling_info = None
        self.scheduler_stream = None

        context_len = self.worker.model_runner.model_config.context_len
        chunk_size = self.worker.server_args.chunked_prefill_size
        max_chunk_times = (
            1 if chunk_size is None or chunk_size < 0 else math.ceil(context_len / chunk_size)
        )
        future_max_num_tokens = self.max_running_requests * (max_chunk_times + 1)

        self.future_token_ids_ct = 0
        self.future_token_ids_limit = max(1, future_max_num_tokens)
        self.future_token_ids_map = torch.empty(
            (self.future_token_ids_limit + self.max_running_requests,),
            dtype=torch.int64,
            device=self.device,
        )

        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.thread_exception: Optional[RuntimeError] = None

        num_decode_steps = max(1, self.worker.server_args.num_continuous_decode_steps)
        self.keep_batch_reference_num = 2 * num_decode_steps
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            name="non-spec-overlap-worker",
            daemon=True,
        )
        self.forward_thread.start()

    def _raise_if_thread_failed(self) -> None:
        if self.thread_exception is not None:
            raise self.thread_exception

    def _bind_thread_device(self) -> None:
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
                f"TpModelWorkerOverlapClient failed: {exc}\n{tb}"
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

            model_worker_batch, future_token_ids_ct = item
            if model_worker_batch is None:
                break

            batch_lists[batch_pt % self.keep_batch_reference_num] = model_worker_batch
            batch_pt += 1

            resolve_future_token_ids(
                model_worker_batch.input_ids,
                self.future_token_ids_map,
            )

            batch_result = self.worker.forward_batch_generation(
                model_worker_batch,
                launch_done=model_worker_batch.launch_done,
            )

            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = batch_result.next_token_ids

            batch_result.copy_done = torch.get_device_module(self.device).Event()
            batch_result.copy_to_cpu(return_logprob=model_worker_batch.return_logprob)
            self.output_queue.put(batch_result)

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        sampling_info = model_worker_batch.sampling_info.copy_for_forward()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
        )

        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if getattr(self.device, "type", self.device) == "npu" and hasattr(torch, "npu"):
            torch.npu.set_stream_limit(self.scheduler_stream, 8, 16)
        if hasattr(self.scheduler_stream, "synchronize"):
            self.scheduler_stream.synchronize()

        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit

        return GenerationBatchResult(
            next_token_ids=future_next_token_ids,
        )

    def resolve_last_batch_result(
        self,
        placeholder_result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ) -> GenerationBatchResult:
        self._raise_if_thread_failed()

        batch_result = self.output_queue.get()
        if isinstance(batch_result, RuntimeError):
            raise batch_result
        if launch_done is not None:
            launch_done.wait()

        batch_result.extend_input_len_per_req = placeholder_result.extend_input_len_per_req
        batch_result.extend_logprob_start_len_per_req = (
            placeholder_result.extend_logprob_start_len_per_req
        )
        return batch_result
