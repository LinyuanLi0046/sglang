"""Publish ready WELM chunks through the existing Mooncake transfer workers.

The community outstanding count owns reserved, waiting, queued and native work.
Only publication ordering lives here; ABORT/ACK and poison stay in CommonKVManager.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque

from sglang.srt.disaggregation.base.conn import KVPoll

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=False)
class LayerwiseTransfer:
    room: int
    pending: deque = dataclasses.field(default_factory=deque)
    groups: list | None = None
    next_group: int = 0
    final: object = None
    final_queued: bool = False
    cancelled: bool = False
    queue_idx: int | None = None
    success_targets: list | None = None


class LayerwisePublisher:
    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.RLock()
        self.wakeup = threading.Event()
        self.rooms = {}

    def bind(self, room):
        with self.lock:
            owner = self.rooms.get(room)
            if owner is not None:
                if not self.drained(owner):
                    raise RuntimeError("Previous WELM transfer still owns this room")
                self.rooms.pop(room)
            owner = LayerwiseTransfer(room)
            self.rooms[room] = owner
            return owner

    def current(self, owner):
        return self.rooms.get(owner.room) is owner

    def drained(self, owner):
        with self.lock:
            return not self.current(owner) or (
                self.manager._staging_outstanding.get(owner.room, 0) == 0
                and owner.room not in self.manager._deferred_ack_poisoned_rooms
            )

    def reserve(self, owner, count):
        with self.lock:
            if not self.current(owner) or owner.cancelled or owner.final_queued:
                return False
            if owner.groups is not None:
                raise RuntimeError("Final Full KV groups were already reserved")
            owner.groups = [None] * count
            self.manager._staging_outstanding[owner.room] += count
            return True

    def add(self, owner, job, queue_idx, group_id=None):
        with self.lock:
            if (
                not self.current(owner)
                or owner.cancelled
                or owner.final_queued
                or self.manager.request_status.get(owner.room, KVPoll.Failed)
                == KVPoll.Failed
            ):
                return False
            if owner.queue_idx is None:
                owner.queue_idx = queue_idx
            job.layerwise_owner = owner
            job.staging_counted = True
            if group_id is not None:
                if (
                    owner.groups is None
                    or not owner.next_group <= group_id < len(owner.groups)
                    or owner.groups[group_id] is not None
                ):
                    raise RuntimeError("Duplicate or unreserved Full KV group")
                owner.groups[group_id] = job
            elif job.is_last_chunk:
                if owner.final is not None:
                    raise RuntimeError("Final WELM tail was already published")
                owner.final = job
                self.manager._staging_outstanding[owner.room] += 1
            else:
                owner.pending.append(job)
                self.manager._staging_outstanding[owner.room] += 1
            self.wakeup.set()
            return True

    def cancel(self, owner):
        with self.lock:
            if not self.current(owner) or owner.cancelled:
                return
            owner.cancelled = True
            # Queued/native jobs retain their count until the worker finishes.
            dropped = len(owner.pending)
            owner.pending.clear()
            if owner.groups is not None:
                dropped += len(owner.groups) - owner.next_group
                owner.groups = None
            if owner.final is not None:
                dropped += 1
                owner.final = None
            self.manager._staging_outstanding[owner.room] -= dropped
            self.wakeup.set()

    def cancel_room(self, room):
        with self.lock:
            owner = self.rooms.get(room)
            if owner is not None:
                self.cancel(owner)

    def complete(self, job):
        with self.lock:
            owner = job.layerwise_owner
            if self.current(owner):
                self.manager._staging_outstanding[owner.room] -= 1
                if self.manager.request_status.get(owner.room) == KVPoll.Failed:
                    self.cancel(owner)
                self.manager._maybe_ack_drained_abort(owner.room)
            self.wakeup.set()

    def clear(self, owner):
        with self.lock:
            if not self.drained(owner):
                return False
            if self.current(owner):
                self.rooms.pop(owner.room)
            return True

    @staticmethod
    def _ready(job):
        return job.ready_fence is None or job.ready_fence.query()

    def _enqueue(self, owner, job):
        # A callable snapshot must not be materialized before its D2H event.
        if callable(job.prefill_kv_indices):
            job.prefill_kv_indices = job.prefill_kv_indices()
        self.manager.transfer_queues[owner.queue_idx].put(job)

    def poll(self, owner):
        with self.lock:
            if not self.current(owner):
                return
            if (
                self.manager.request_status.get(owner.room, KVPoll.Failed)
                == KVPoll.Failed
            ):
                self.cancel(owner)
            if owner.cancelled:
                self.manager._maybe_ack_drained_abort(owner.room)
                return
            # Prefix/middle jobs retain their original per-request order. A ready
            # group must not sit in a worker queue behind an unready predecessor.
            while owner.pending:
                job = owner.pending[0]
                if not self._ready(job):
                    return
                self._enqueue(owner, job)
                owner.pending.popleft()
            if owner.groups is not None:
                while owner.next_group < len(owner.groups):
                    job = owner.groups[owner.next_group]
                    if job is None or not self._ready(job):
                        return
                    self._enqueue(owner, job)
                    owner.groups[owner.next_group] = None
                    owner.next_group += 1
                owner.groups = None
            # A ready final tail is released only after all earlier native writes
            # finished. It carries SWA/state and aux, which authorizes decode.
            if (
                owner.final is not None
                and self.manager._staging_outstanding.get(owner.room, 0) == 1
                and self._ready(owner.final)
            ):
                self._enqueue(owner, owner.final)
                owner.final = None
                owner.final_queued = True

    def run(self):
        import torch

        torch.npu.set_device(self.manager.kv_args.gpu_id)
        while True:
            self.wakeup.wait(0.001)
            self.wakeup.clear()
            with self.lock:
                owners = tuple(self.rooms.values())
            for owner in owners:
                try:
                    self.poll(owner)
                except Exception as exc:
                    logger.exception(
                        "WELM layer readiness failed for room %s", owner.room
                    )
                    with self.lock:
                        # No native write is launched for an unready group.
                        # Existing native work retains its count; P separately
                        # holds the producer fence before freeing source pages.
                        self.cancel(owner)
                    self.manager.conclude_failure(
                        bootstrap_room=owner.room, failure_reason=str(exc)
                    )
