from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar

import torch

T = TypeVar("T")


@dataclass
class _WorkItem(Generic[T]):
    work_id: int
    fn: Callable[[], T]


@dataclass
class _CompletedWorkItem(Generic[T]):
    work_id: int
    result: object


class AsyncResultHandle(Generic[T]):
    def __init__(self, client: "TpWorkerClientV2[T]", work_id: int):
        self.client = client
        self.work_id = work_id
        self._resolved = False
        self._result: Optional[T] = None

    def resolve(self) -> T:
        if not self._resolved:
            self._result = self.client.resolve(self.work_id)
            self._resolved = True
        return self._result


class TpWorkerClientV2(Generic[T]):
    """A conservative worker-thread client for overlap orchestration.

    This preserves the current scheduler timing semantics while moving overlap
    control-plane work off the scheduler thread into a dedicated worker thread.
    """

    def __init__(
        self,
        name: str,
        *,
        device: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ):
        self.name = name
        self.device = device
        self.gpu_id = gpu_id
        self.input_queue: "queue.Queue[Optional[_WorkItem[T]]]" = queue.Queue()
        self.output_queue: "queue.Queue[_CompletedWorkItem[T]]" = queue.Queue()
        self._next_work_id = 0
        self._completed_results: Dict[int, object] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{name}-overlap-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, fn: Callable[[], T]) -> T:
        return self.submit_async(fn).resolve()

    def submit_async(self, fn: Callable[[], T]) -> AsyncResultHandle[T]:
        with self._lock:
            work_id = self._next_work_id
            self._next_work_id += 1
        self.input_queue.put(_WorkItem(work_id=work_id, fn=fn))
        return AsyncResultHandle(self, work_id)

    def resolve(self, work_id: int) -> T:
        if work_id in self._completed_results:
            result = self._completed_results.pop(work_id)
        else:
            while True:
                completed = self.output_queue.get()
                if completed.work_id == work_id:
                    result = completed.result
                    break
                self._completed_results[completed.work_id] = completed.result
        if isinstance(result, Exception):
            raise result
        return result

    def _worker_loop(self):
        if self.device is not None and self.gpu_id is not None:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        while True:
            item = self.input_queue.get()
            if item is None:
                return
            try:
                result = item.fn()
            except Exception as exc:  # pragma: no cover - passthrough
                self.output_queue.put(_CompletedWorkItem(item.work_id, exc))
            else:
                self.output_queue.put(_CompletedWorkItem(item.work_id, result))
