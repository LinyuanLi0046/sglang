from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

import torch

T = TypeVar("T")


@dataclass
class _WorkItem(Generic[T]):
    fn: Callable[[], T]
    response_queue: "queue.Queue[object]"


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
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{name}-overlap-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, fn: Callable[[], T]) -> T:
        response_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self.input_queue.put(_WorkItem(fn=fn, response_queue=response_queue))
        result = response_queue.get()
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
                item.response_queue.put(exc)
            else:
                item.response_queue.put(result)
