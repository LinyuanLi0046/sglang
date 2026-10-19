"""Full-KV readiness for WELM eager prefill and the Prompt graph.

pyACL is loaded only for the enabled P-side graph adapter. Captured device
signals keep Prompt/Mirror intact; ordinary events expose progress to Host RDMA.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

# The runner has no graph teardown API. Device flags must outlive every graph
# referencing them. Completed per-replay event handles are retired separately.
_graph_lifetime_owners = []


def welm_full_writer_entries(model, pool):
    """Static writer order shared by eager layers and the Prompt/Mirror graphs."""
    writers = []
    for layer in model.layers[model.start_layer : model.end_layer]:
        layer_id = layer.self_attn.attn.layer_id
        entries = pool.get_full_layer_transfer_entries(layer_id)
        if entries is not None:
            writers.append((layer_id, entries))
    if len(writers) != pool.full_layer_nums:
        raise ValueError("WELM layerwise writers must cover the target Full pool")
    return tuple(writers)


def full_writer_groups(writer_entries, group_size, *, tail_writer_ids=()):
    """Return groups in actual writer order, with both registered K/V entries."""
    if group_size < 1:
        raise ValueError("SGLANG_DISAGG_LAYERWISE_GROUP_SIZE must be positive")
    writers = tuple((layer, tuple(entries)) for layer, entries in writer_entries)
    if not writers or len({layer for layer, _ in writers}) != len(writers):
        raise ValueError("WELM Full-KV writer manifest must be nonempty and unique")
    flattened = [entry for _, entries in writers for entry in entries]
    if any(len(entries) != 2 for _, entries in writers) or len(set(flattened)) != len(
        flattened
    ):
        raise ValueError("Each WELM Full writer must own distinct K and V entries")
    tail_writer_ids = tuple(tail_writer_ids)
    main_count = len(writers) - len(tail_writer_ids)
    if tail_writer_ids and (
        main_count < 0
        or tuple(layer for layer, _ in writers[main_count:]) != tail_writer_ids
    ):
        raise ValueError("WELM BCG tail writers must be an ordered Full-writer suffix")
    groups = tuple(
        writers[start : min(start + group_size, main_count)]
        for start in range(0, main_count, group_size)
    )
    if tail_writer_ids:
        groups += (writers[main_count:],)
    entries = tuple(
        tuple(entry for _, pair in group for entry in pair) for group in groups
    )
    ends = {group[-1][0]: index for index, group in enumerate(groups)}
    return tuple(layer for layer, _ in writers), entries, ends


class EagerReadySet:
    """One ordinary event per group, published only after its actual KV write."""

    def __init__(self, plan, manifest, device_module):
        self.plan = plan
        _, self.group_entries, self.group_ends = manifest
        self.device_module = device_module
        self.tickets = []

    def record_writer(self, layer_id):
        group_id = self.group_ends.get(layer_id)
        if group_id is None:
            return
        if group_id != len(self.tickets):
            raise RuntimeError("WELM eager Full-KV groups arrived out of order")
        ticket = self.device_module.Event()
        # Retain even a partially submitted record through the producer fence.
        self.tickets.append(ticket)
        ticket.record()
        self.plan.publish_group(group_id, ticket)

    def finish(self):
        if len(self.tickets) != len(self.group_entries):
            raise RuntimeError("WELM eager forward did not publish every Full-KV group")


@contextmanager
def eager_layerwise_transfer(batch, plan, manifest, device_module):
    """Entered only by the actual eager runner, never by capture or graph replay."""
    active = None
    try:
        if plan.prepare(manifest[1]) and plan.commit():
            active = EagerReadySet(plan, manifest, device_module)
            batch.pd_eager_transfer = active
        yield
        if active is not None:
            active.finish()
    except Exception as exc:
        plan.fail(exc, tickets=active.tickets if active is not None else ())
        raise
    finally:
        batch.pd_eager_transfer = None


class _ACLRuntime:
    """CANN 9.1 pyACL calls; reuse the context already initialized by PTA."""

    def __init__(self, acl, device_module):
        self.rt = acl.rt
        self.device_module = device_module

    @staticmethod
    def check(ret, operation):
        if ret != 0:
            raise RuntimeError(f"WELM layerwise {operation} failed: ACL error {ret}")

    def result(self, result, operation):
        value, ret = result
        self.check(ret, operation)
        return value

    def current_stream(self):
        # This getter drains PTA's Host submission queue, not Device compute.
        return self.device_module.current_stream().npu_stream

    def malloc(self, size):
        # ACL_MEM_MALLOC_NORMAL_ONLY; flags are small persistent allocations.
        return self.result(self.rt.malloc(size, 2), "malloc")

    def create_stream(self):
        return self.result(self.rt.create_stream(), "create_stream")

    def write(self, address, value, stream):
        self.check(self.rt.value_write(address, value, 0, stream), "value_write")

    def wait(self, address, stream):
        self.check(self.rt.value_wait(address, 1, 1, stream), "value_wait")

    def wait_event(self, stream, event):
        self.check(self.rt.stream_wait_event(stream, event), "stream_wait_event")


class _ReadyTicket:
    """Record once per replay; query and owner retirement may use two threads."""

    def __init__(self, runtime, *, gate=False):
        self._runtime = runtime
        # PROGRESS=2, SYNC=1. Only the last ticket also needs stream wait support.
        self.event = runtime.result(
            runtime.rt.create_event_ex_with_flag(3 if gate else 2), "create_event"
        )
        self.armed = False
        self.released = False
        self._complete = False
        self._lock = Lock()

    def record(self, stream):
        with self._lock:
            self._runtime.check(
                self._runtime.rt.record_event(self.event, stream), "record_event"
            )
            self.armed = True

    def query(self):
        with self._lock:
            # CANN reports a new, unrecorded Ex Event as COMPLETE.
            if not self.armed:
                return False
            if not self._complete:
                status = self._runtime.result(
                    self._runtime.rt.query_event_status(self.event),
                    "query_event_status",
                )
                # aclrtEventRecordedStatus, not the legacy aclrtEventStatus.
                if status not in (0, 1):
                    raise RuntimeError(f"Unexpected ACL event status: {status}")
                self._complete = status == 1
            return self._complete

    def release(self):
        # Releasing a request must not cancel an already submitted notification.
        # All ACL destruction remains on the owner/forward thread.
        with self._lock:
            self.released = True

    def retire(self, *, complete=False):
        with self._lock:
            if complete:
                self._complete = True
            if self.event is not None:
                self._runtime.check(
                    self._runtime.rt.destroy_event(self.event), "destroy_event"
                )
                self.event = None


@dataclass
class _Replay:
    previous_gate: _ReadyTicket
    tickets: list | None = None
    gate: _ReadyTicket | None = None


class PromptReadySet:
    """Fixed graph signals plus a fresh Host ticket for each replay/group."""

    _MAX_PENDING = 64

    def __init__(self, runtime, writer_ids, group_entries, group_ends):
        self._runtime = runtime
        self.writer_ids = writer_ids
        self.group_entries = group_entries
        self.group_ends = group_ends
        self._writer_set = frozenset(writer_ids)
        self._capture_writers = []
        self._capturing = False
        self._failed = False
        self._pending = None
        self._last_gate = None
        self._epochs = deque()
        self._reservations = []
        self._flags = runtime.malloc(8 * len(group_entries))
        try:
            self._notify = runtime.create_stream()
        except Exception:
            runtime.check(runtime.rt.free(self._flags), "free")
            raise

    def _check_healthy(self):
        if self._failed:
            raise RuntimeError(
                "WELM layerwise graph signals cannot be reused after failure"
            )

    def begin_capture_forward(self, capturing):
        self._capturing = capturing
        self._capture_writers = []

    def record_writer(self, layer_id):
        if not self._capturing or layer_id not in self._writer_set:
            return
        self._capture_writers.append(layer_id)
        group_id = self.group_ends.get(layer_id)
        if group_id is not None:
            # Capture a device node after the actual KV write. Replay does not
            # execute this Python hook and does not split the Prompt graph.
            self._runtime.write(
                self._flags + 8 * group_id, 1, self._runtime.current_stream()
            )

    def finish_capture(self):
        if tuple(self._capture_writers) != self.writer_ids:
            raise RuntimeError(
                "WELM graph Full-KV writers differ from the transfer manifest: "
                f"expected={self.writer_ids}, captured={self._capture_writers}"
            )
        self._capturing = False
        stream = self._runtime.current_stream()
        self._last_gate = _ReadyTicket(self._runtime, gate=True)
        for group_id in range(len(self.group_entries)):
            self._runtime.write(self._flags + 8 * group_id, 0, stream)
        self._last_gate.record(stream)

    def _reap(self):
        # A completed notify tail proves all this epoch's records finished AND
        # compute crossed its previous gate. The current gate stays alive for
        # the next replay to wait on, independently of request/ticket ownership.
        while self._epochs and self._epochs[0].gate.query():
            epoch = self._epochs[0]
            for ticket in epoch.tickets or ():
                if ticket is not epoch.gate:
                    ticket.retire(complete=True)
            epoch.previous_gate.retire(complete=True)
            self._epochs.popleft()
        remaining = []
        for tickets in self._reservations:
            if all(ticket.released for ticket in tickets):
                for ticket in tickets:
                    ticket.retire()
            else:
                remaining.append(tickets)
        self._reservations = remaining

    def reserve(self):
        self._check_healthy()
        if len(self._epochs) + len(self._reservations) >= self._MAX_PENDING:
            # Reclaim before declining a reservation under capacity pressure.
            # Normally retire only after the next graphs have been submitted.
            self._reap()
            if len(self._epochs) + len(self._reservations) >= self._MAX_PENDING:
                return None
        tickets = []
        try:
            for i in range(len(self.group_entries)):
                tickets.append(
                    _ReadyTicket(self._runtime, gate=i == len(self.group_entries) - 1)
                )
        except Exception:
            for ticket in tickets:
                ticket.retire()
            raise
        self._reservations.append(tickets)
        return tickets

    def before_replay(self):
        self._check_healthy()
        if self._pending is not None or self._last_gate is None:
            raise RuntimeError(
                "WELM graph replay is missing capture or notification binding"
            )
        self._pending = _Replay(self._last_gate)
        self._runtime.wait_event(self._runtime.current_stream(), self._last_gate.event)

    def bind_after_replay(self, tickets=None):
        self._check_healthy()
        epoch = self._pending
        if epoch is None:
            raise RuntimeError("WELM graph notifications require before_replay")
        epoch.tickets = tickets
        if tickets is not None:
            self._reservations.remove(tickets)
            epoch.gate = tickets[-1]
        else:
            epoch.gate = _ReadyTicket(self._runtime, gate=True)
        # Use only raw ACL on this dedicated stream: mixing PTA Event.record
        # with raw waits could delay an early ticket behind later groups.
        for group_id in range(len(self.group_entries)):
            address = self._flags + 8 * group_id
            self._runtime.wait(address, self._notify)
            self._runtime.write(address, 0, self._notify)
            if tickets is not None:
                tickets[group_id].record(self._notify)
        if tickets is None:
            # Even no-plan/middle/empty/cancelled replays write the fixed flags.
            epoch.gate.record(self._notify)
        self._last_gate = epoch.gate
        self._epochs.append(epoch)
        self._pending = None

    def after_replay(self):
        self._check_healthy()
        self._reap()

    def fail(self):
        # Partial submission has unknown Device progress. Never destroy its
        # flags/stream or submit another graph using them; worker failure owns
        # recovery. This method must not mask the original submission error.
        self._failed = True

    def close(self):
        """Startup-test teardown only, after its graph is reset and work drained."""
        self._check_healthy()
        self._runtime.check(
            self._runtime.rt.synchronize_stream(self._notify), "synchronize_stream"
        )
        self._reap()
        for tickets in self._reservations:
            for ticket in tickets:
                ticket.retire()
        self._reservations.clear()
        self._last_gate.retire(complete=True)
        self._runtime.check(
            self._runtime.rt.destroy_stream(self._notify), "destroy_stream"
        )
        self._runtime.check(self._runtime.rt.free(self._flags), "free")


class LayerwiseGraphEvents:
    def __init__(self, runtime, writer_entries, group_size, *, tail_writer_ids=()):
        self._runtime = runtime
        self._manifest = full_writer_groups(
            writer_entries, group_size, tail_writer_ids=tail_writer_ids
        )
        self.ready_sets = {}

    @classmethod
    def create(cls, writer_entries, group_size, *, tail_writer_ids=()):
        try:
            import acl
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Enabled WELM layerwise BCG requires CANN pyACL value_write/value_wait"
            ) from exc
        runtime = _ACLRuntime(acl, torch.npu)
        bridge = cls(
            runtime, writer_entries, group_size, tail_writer_ids=tail_writer_ids
        )
        # Missing/unsupported graph primitives must not silently turn this into
        # whole-prefill transfer. Exercise true mid-graph Host visibility.
        _validate_runtime(runtime)
        logger.info(
            "WELM layerwise pyACL graph signals enabled: %d Full-KV groups, "
            "group_size=%d, main_groups=%d, merged_tail_writers=%d; "
            "startup mid-graph/replay check passed",
            len(bridge._manifest[1]),
            group_size,
            len(bridge._manifest[1]) - bool(tail_writer_ids),
            len(tail_writer_ids),
        )
        return bridge

    def prepare_capture(self, key):
        if key in self.ready_sets:
            raise RuntimeError(
                "WELM layerwise Prompt graph cannot be recaptured in place"
            )
        ready = PromptReadySet(self._runtime, *self._manifest)
        self.ready_sets[key] = ready
        _graph_lifetime_owners.append(ready)
        return ready


def _validate_runtime(runtime):
    """Startup only: Host must see ready while the captured graph tail is held."""
    import torch

    ready = PromptReadySet(runtime, (0,), ((0, 1),), {0: 0})
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    release_stream = runtime.create_stream()
    release_flag = runtime.malloc(8)
    value = torch.zeros(1, device="npu", dtype=torch.int32)
    device = torch.npu.current_device()
    succeeded = False

    def await_ready(ticket):
        # Match the real coordinator's one-time thread initialization.
        torch.npu.set_device(device)
        deadline = time.monotonic() + 30
        while not ticket.query():
            if time.monotonic() >= deadline:
                raise RuntimeError("CANN layerwise mid-graph self-check timed out")
            time.sleep(0.001)

    try:
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            value.add_(1)
            value.zero_()
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            ready.begin_capture_forward(True)
            value.add_(1)
            ready.record_writer(0)
            # Only the Host observing our ready ticket may release this tail.
            runtime.wait(release_flag, runtime.current_stream())
            value.add_(1)
        with torch.npu.stream(stream):
            ready.finish_capture()
            with ThreadPoolExecutor(max_workers=1) as executor:
                for has_work in (True, False, True):
                    runtime.write(release_flag, 0, runtime.current_stream())
                    tickets = ready.reserve() if has_work else None
                    if tickets and tickets[0].query():
                        raise RuntimeError(
                            "Unbound layerwise ticket was reported ready"
                        )
                    ready.before_replay()
                    graph.replay()
                    tail_done = torch.npu.Event()
                    tail_done.record(stream)
                    runtime.current_stream()  # submit graph and tail event
                    ready.bind_after_replay(tickets)
                    ready.after_replay()
                    executor.submit(await_ready, ready._last_gate).result()
                    if tail_done.query():
                        raise RuntimeError(
                            "Layerwise self-check did not hold the graph tail"
                        )
                    runtime.write(release_flag, 1, release_stream)
                    await_ready(tail_done)
        stream.synchronize()
        if value.item() != 6:
            raise RuntimeError(
                "CANN layerwise graph self-check returned incorrect data"
            )
        succeeded = True
    finally:
        if succeeded:
            graph.reset()
            ready.close()
            runtime.check(
                runtime.rt.synchronize_stream(release_stream), "synchronize_stream"
            )
            runtime.check(runtime.rt.destroy_stream(release_stream), "destroy_stream")
            runtime.check(runtime.rt.free(release_flag), "free")
        else:
            # Best effort to unblock the test graph, without waiting on a
            # potentially failed device. Preserve every referenced allocation.
            try:
                runtime.write(release_flag, 1, release_stream)
            except Exception:
                pass
            ready.fail()
            _graph_lifetime_owners.append(
                (ready, graph, stream, value, release_flag, release_stream)
            )
