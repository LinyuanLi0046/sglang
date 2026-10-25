# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bridge one captured NPU read boundary to the scheduler's WAR event.

The fixed device flag is reused only after its previous notification was
consumed. Every replay gets a distinct SYNC | PROGRESS ACL event. Owners and
their flags must outlive the graphs; per-replay events are retired separately.

Adapted from layerwise_graph_events.py at commit
09389213d63fb3ca57e359eff62481fdf4c5980a: retain its CANN flag/gate protocol,
reduce it to one read boundary, and expose a scheduler-compatible wait().
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import torch

# Graph runners do not expose graph teardown. Never free a captured flag via
# Python GC, including after a partially submitted replay raises an exception.
_graph_lifetime_owners = []


class _ACLRuntime:
    def __init__(self):
        try:
            import acl
        except ImportError as exc:
            raise RuntimeError("WeLM graph read-done requires CANN pyACL") from exc
        self.rt = acl.rt
        for name in (
            "value_write",
            "value_wait",
            "create_event_ex_with_flag",
            "query_event_status",
            "stream_wait_event",
        ):
            if not hasattr(self.rt, name):
                raise RuntimeError(f"WeLM graph read-done requires acl.rt.{name}")

    @staticmethod
    def check(ret, operation):
        if ret != 0:
            raise RuntimeError(f"NPU read-done {operation} failed: ACL error {ret}")

    def result(self, result, operation):
        value, ret = result
        self.check(ret, operation)
        return value

    def write(self, address, value, stream):
        self.check(self.rt.value_write(address, value, 0, stream), "value_write")

    def wait(self, address, stream):
        self.check(self.rt.value_wait(address, 1, 1, stream), "value_wait")


class NpuReadDoneTicket:
    """Single-record event compatible with torch_npu Stream.wait_event()."""

    def __init__(self, owner):
        self._owner = owner
        self._runtime = owner._runtime
        # SYNC=1 permits stream waits; PROGRESS=2 permits completion queries.
        self.event = self._runtime.result(
            self._runtime.rt.create_event_ex_with_flag(3), "create_event"
        )
        self.armed = False

    def record(self, stream):
        self._runtime.check(
            self._runtime.rt.record_event(self.event, stream), "record_event"
        )
        self.armed = True

    def wait(self, stream):
        self._owner._check_healthy()
        if not self.armed or self.event is None:
            raise RuntimeError("Unarmed or retired NPU read-done ticket")
        try:
            # Getting the handle flushes PTA's Host submission queue. Do not
            # cache it across calls or replace this with Device synchronize().
            native_stream = stream.npu_stream
            self._runtime.check(
                self._runtime.rt.stream_wait_event(native_stream, self.event),
                "stream_wait_event",
            )
        except BaseException:
            self._owner._failed = True
            raise

    def query(self):
        # An unrecorded ACL ExEvent may report COMPLETE. Never query it early.
        if not self.armed:
            return False
        if self.event is None:
            raise RuntimeError("Cannot query a retired NPU read-done ticket")
        status = self._runtime.result(
            self._runtime.rt.query_event_status(self.event), "query_event_status"
        )
        # aclrtEventRecordedStatus: NOT_READY=0, COMPLETE=1.
        if status not in (0, 1):
            raise RuntimeError(f"Unexpected ACL event status: {status}")
        return status == 1

    def retire(self):
        if self.event is not None:
            self._runtime.check(
                self._runtime.rt.destroy_event(self.event), "destroy_event"
            )
            self.event = None


@dataclass
class _Replay:
    previous_gate: NpuReadDoneTicket
    gate: NpuReadDoneTicket


class NpuGraphReadDone:
    """One owner per complete graph key, driven by the scheduler thread.

    Call begin_capture_forward() for every warmup/capture forward, then
    signal_once() after the last shared read, and after_capture() outside the
    graph capture context. Always wrap a replay in replay(), even when the
    scheduler will use coarse WAR instead of publishing the returned ticket.

    Event retirement requires the existing overlap order: the scheduler
    consumes a published ticket before the next forward, and that forward
    waits for the scheduler stream. Arbitrary delayed consumers are unsupported.
    """

    _MAX_PENDING = 64

    def __init__(self, device):
        self._runtime = _ACLRuntime()
        self._flag = torch.empty(1, dtype=torch.int64, device=device)
        self._address = self._flag.data_ptr()
        if self._address % 8:
            raise RuntimeError("NPU read-done requires an aligned 64-bit flag")
        self._notify = self._runtime.result(
            self._runtime.rt.create_stream(), "create_stream"
        )
        self._failed = False
        self._capturing = False
        self._capture_signals = 0
        self._last_gate = None
        self._pending = None
        self._epochs = deque()
        _graph_lifetime_owners.append(self)

    def _check_healthy(self):
        if self._failed:
            raise RuntimeError("NPU read-done owner cannot be reused after failure")

    def begin_capture_forward(self):
        self._check_healthy()
        if self._last_gate is not None:
            raise RuntimeError("NPU read-done owner cannot be recaptured in place")
        self._capturing = torch.npu.is_current_stream_capturing()
        self._capture_signals = 0

    def signal_once(self):
        if not self._capturing or self._capture_signals:
            return
        self._check_healthy()
        try:
            self._runtime.write(
                self._address, 1, torch.npu.current_stream().npu_stream
            )
            self._capture_signals = 1
        except BaseException:
            self._failed = True
            raise

    def after_capture(self):
        self._check_healthy()
        try:
            if (
                torch.npu.is_current_stream_capturing()
                or not self._capturing
                or self._capture_signals != 1
                or self._last_gate is not None
            ):
                raise RuntimeError("NPU graph must capture exactly one read-done signal")
            self._capturing = False
            self._last_gate = NpuReadDoneTicket(self)
            # Both runners call here inside the outer graph_capture() stream
            # context, after the inner capture has ended. Its stream order and
            # the initial gate suffice; no Device synchronize is necessary.
            stream = torch.npu.current_stream().npu_stream
            self._runtime.write(self._address, 0, stream)
            self._last_gate.record(stream)
        except BaseException:
            self._failed = True
            raise

    def _reap(self):
        # E_n being complete alone does not retire E_n: scheduler may still
        # wait on it. Completion of this owner's next gate proves that forward
        # crossed both its previous gate and the scheduler's previous wait.
        while self._epochs and self._epochs[0].gate.query():
            epoch = self._epochs[0]
            epoch.previous_gate.retire()
            self._epochs.popleft()

    @contextmanager
    def replay(self, stream):
        self._check_healthy()
        try:
            if self._last_gate is None or self._pending is not None:
                raise RuntimeError("NPU read-done replay lacks capture or prior binding")
            if len(self._epochs) >= self._MAX_PENDING:
                self._reap()
            if len(self._epochs) >= self._MAX_PENDING:
                # Check before replay: never submit a signal without capacity
                # to consume it, and never add a per-step CPU completion wait.
                raise RuntimeError("Too many outstanding NPU read-done epochs")
            ticket = NpuReadDoneTicket(self)
            self._pending = _Replay(self._last_gate, ticket)
            self._last_gate.wait(stream)
            yield ticket
            # Flush graph submission before putting raw ACL tasks on notify.
            # This is Host submission ordering, not Device completion waiting.
            _ = stream.npu_stream
            self._runtime.wait(self._address, self._notify)
            self._runtime.write(self._address, 0, self._notify)
            ticket.record(self._notify)
            self._epochs.append(self._pending)
            self._last_gate = ticket
            self._pending = None
            # Query/retire after graph submission so normal retirement can
            # overlap model compute instead of lengthening the replay gap.
            self._reap()
        except BaseException:
            # Keep pending events/flag/stream alive after partial submission;
            # recovery belongs to worker shutdown, never a same-owner retry.
            self._failed = True
            raise
