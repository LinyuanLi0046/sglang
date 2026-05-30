from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

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

    def __init__(self, name: str):
        self.name = name
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
