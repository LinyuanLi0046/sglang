"""Opt-in, CPU-only PD timing. Uses the existing ReqTimeStats logging channel.

No device reads, synchronization, or transport protocol changes belong here.
Monotonic timestamps can only be compared on the same host. MF call durations
include library-side waiting; they are not measurements of wire latency.
"""

from __future__ import annotations

import atexit
import contextvars
import functools
import json
import logging
import os
import queue
import socket
import threading
import time

# Reuse the module selected by existing SGLANG_LOGGING_CONFIG_PATH configs.
_logger = logging.getLogger("sglang.srt.managers.schedule_batch")
_context = contextvars.ContextVar("pd_transfer_timing", default=None)
_writer_lock = threading.Lock()
_writer = None
_gc_diagnostics = None


def _drain_at_exit():
    # Best effort only: abnormal termination can lose the last queued records.
    writer = _writer
    if writer is not None and writer.pid == os.getpid():
        if _gc_diagnostics is not None:
            _gc_diagnostics.uninstall()
            writer.put(None, None)  # Wake the writer to flush GC/status at exit.
        deadline = time.monotonic() + 1.0
        with writer.queue.all_tasks_done:
            while writer.queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                writer.queue.all_tasks_done.wait(remaining)


atexit.register(_drain_at_exit)


class _LogWriter:
    def __init__(self):
        self.pid = os.getpid()
        self.queue = queue.Queue(maxsize=4096)
        self.dropped = 0
        self.gc_status_time = 0.0
        threading.Thread(
            target=self._run, name="pd-time-stats-log", daemon=True
        ).start()

    def _run(self):
        while True:
            try:
                item = self.queue.get(timeout=0.25 if _gc_diagnostics else None)
            except queue.Empty:
                self._flush_gc()
                continue
            try:
                prefix, record = item
                if prefix is not None:
                    # Preserve the ordinary writer path when GC is disabled.
                    if isinstance(record, dict):
                        message = json.dumps(
                            dict(record, dropped_records_total=self.dropped),
                            separators=(",", ":"),
                        )
                    else:
                        message = record
                    _logger.info("%s%s", prefix, message)
                if _gc_diagnostics is not None:
                    self._flush_gc(force=prefix is None)
            except Exception:
                # Diagnostics must not terminate model/transfer workers.
                self.dropped += 1
            finally:
                self.queue.task_done()

    def _write(self, prefix, record):
        if isinstance(record, dict):
            message = json.dumps(
                dict(record, dropped_records_total=self.dropped),
                separators=(",", ":"),
            )
        else:
            message = record
        _logger.info("%s%s", prefix, message)

    def _flush_gc(self, force=False):
        diag = _gc_diagnostics
        if diag is None or diag.pid != self.pid:
            return
        try:
            for record in diag.drain(limit=2048 if force else 128):
                self._write("PDGCStats ", record)
            now = time.monotonic()
            if force or now - self.gc_status_time >= 1.0:
                self._write("PDGCStats ", diag.status())
                self.gc_status_time = now
        except Exception:
            self.dropped += 1

    def put(self, prefix, record):
        try:
            self.queue.put_nowait((prefix, record))
        except queue.Full:
            self.dropped += 1


def _ensure_writer():
    global _writer
    if _writer is None or _writer.pid != os.getpid():
        with _writer_lock:
            if _writer is None or _writer.pid != os.getpid():
                _writer = _LogWriter()
    return _writer


def enable_gc_diagnostics(fields):
    """Initialize output before registering GC; called once during P startup."""
    global _gc_diagnostics
    from sglang.srt.observability.pd_gc_diagnostics import PDGCDiagnostics

    if not _logger.isEnabledFor(logging.INFO):
        raise ValueError(
            "PD GC diagnostics requires INFO for sglang.srt.managers.schedule_batch "
            "in SGLANG_LOGGING_CONFIG_PATH (global INFO is not required)"
        )
    writer = _ensure_writer()
    if _gc_diagnostics is None or _gc_diagnostics.pid != os.getpid():
        _gc_diagnostics = PDGCDiagnostics(fields)
    writer.put("PDGCStats ", _gc_diagnostics.status(record="gc_enabled"))
    _gc_diagnostics.install()
    return _gc_diagnostics.callback


def emit(prefix, record):
    """Bounded asynchronous output; a full diagnostic queue never blocks inference."""
    if not _logger.isEnabledFor(logging.INFO):
        return
    # Keep the ordinary statistics path inline; enabling GC must not add a
    # helper call to every record when the diagnostic switch is off.
    global _writer
    if _writer is None or _writer.pid != os.getpid():
        with _writer_lock:
            if _writer is None or _writer.pid != os.getpid():
                _writer = _LogWriter()
    _writer.put(prefix, record)


def identity(role, room, **fields):
    mono_ns = time.perf_counter_ns()
    return dict(
        schema=1,
        host=socket.gethostname(),
        pid=os.getpid(),
        role=role,
        bootstrap_room=room,
        anchor_mono_ns=mono_ns,
        anchor_wall_ns=time.time_ns(),
        **fields,
    )


def interval_ms(start, end):
    return (end - start) / 1e6 if start is not None and end is not None else None


class RequestTiming:
    """Small per-room state shared by scheduler and notification threads."""

    def __init__(self, role, room, **fields):
        self.fields = identity(role, room, **fields)
        self.events = {}
        self.notices = {}
        self.prepare = {}
        self.chunk_seq = 0
        self.lock = threading.Lock()

    def mark(self, name, **fields):
        if name in self.events:
            return
        with self.lock:
            if name not in self.events:
                self.events[name] = time.perf_counter_ns()
                self.fields.update(fields)

    def notice(self, sender, expected):
        with self.lock:
            self.notices.setdefault(str(sender), time.perf_counter_ns())
            self.fields["expected_senders"] = expected

    def new_chunk(self, **fields):
        with self.lock:
            seq = self.chunk_seq
            self.chunk_seq += 1
            prepare = self.prepare.copy()
        return ChunkTiming(self.fields, seq, prepare, **fields)

    def finish(self, status):
        with self.lock:
            record = dict(
                self.fields,
                record="receiver" if self.fields["role"] == "decode" else "sender",
                status=status,
                events_ns=self.events.copy(),
                notices_ns=self.notices.copy(),
            )
        events = record["events_ns"]
        record["durations_ms"] = {
            name: interval_ms(events.get(start), events.get(end))
            for name, start, end in (
                ("metadata_send", "metadata_send_begin", "metadata_sent"),
                (
                    "metadata_sent_to_all_notices",
                    "metadata_sent",
                    "all_notices_received",
                ),
                (
                    "notice_to_scheduler_poll",
                    "all_notices_received",
                    "local_transfer_success_observed",
                ),
                (
                    "metadata_gate_observed_wait",
                    "metadata_wait_observed",
                    "metadata_gate_pass",
                ),
                ("local_ready_to_tp_ready", "local_ready", "tp_consensus_ready"),
                ("tp_ready_to_commit", "tp_consensus_ready", "commit_begin"),
                ("commit_metadata", "commit_begin", "commit_metadata_done"),
            )
        }
        emit("PDTransferStats ", record)


class ChunkTiming:
    def __init__(self, parent_fields, sequence, prepare, **fields):
        self.fields = dict(
            parent_fields, record="chunk", chunk_seq=sequence, **prepare, **fields
        )
        self.events = {"enqueued": time.perf_counter_ns()}
        self.components = {}
        self.lock = threading.Lock()

    def mark(self, name):
        with self.lock:
            self.events.setdefault(name, time.perf_counter_ns())

    def add_interval(self, component, kind, start, end, **values):
        with self.lock:
            data = self.components.setdefault(component, {})
            group = data.setdefault(kind, {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
            duration = (end - start) / 1e6
            group["count"] += 1
            group["sum_ms"] += duration
            group["max_ms"] = max(group["max_ms"], duration)
            group["first_start_ns"] = min(group.get("first_start_ns", start), start)
            group["last_end_ns"] = max(group.get("last_end_ns", end), end)
            for key, value in values.items():
                group[key] = group.get(key, 0) + value

    def add_gc_preparation(self, record):
        with self.lock:
            if not hasattr(self, "gc_preparations"):
                self.gc_preparations = []
            self.gc_preparations.append(record)

    def finish(self, error=None):
        self.mark("worker_end")
        with self.lock:
            # Futures may finish after a failed/cancelled sibling. Snapshot under
            # the lock, rather than letting the async logger observe mutations.
            components = {
                key: {name: values.copy() for name, values in groups.items()}
                for key, groups in self.components.items()
            }
            record = dict(
                self.fields,
                events_ns=self.events.copy(),
                components=components,
                error=error,
            )
            if _gc_diagnostics is not None and hasattr(self, "gc_preparations"):
                record["gc_preparations"] = self.gc_preparations.copy()
                record["gc_diagnostic_id"] = _gc_diagnostics.identity["diagnostic_id"]
        events = record["events_ns"]
        record["queue_wait_ms"] = interval_ms(
            events.get("enqueued"), events.get("worker_begin")
        )
        record["send_prepare_ms"] = interval_ms(
            record.get("prepare_begin_ns"), events.get("enqueued")
        )
        record["worker_ms"] = interval_ms(
            events.get("worker_begin"), events.get("worker_end")
        )
        for groups in components.values():
            for group in groups.values():
                group["span_ms"] = interval_ms(
                    group["first_start_ns"], group["last_end_ns"]
                )
        emit("PDTransferStats ", record)


def begin_chunk(timing):
    if timing is None:
        return None
    timing.mark("worker_begin")
    return _context.set((timing, "worker"))


def end_chunk(token, timing, error=None, *, deferred=False):
    if token is not None:
        _context.reset(token)
        # A staging retry reuses the chunk; aggregate until it leaves the queue.
        if not deferred:
            timing.finish(error)


def timed_component(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            active = _context.get()
            if active is None:
                return fn(*args, **kwargs)
            timing, _ = active
            token = _context.set((timing, name))
            diag = _gc_diagnostics if name == "kv" else None
            scope = diag.enter(timing.fields) if diag is not None else None
            start = time.perf_counter_ns()
            if scope is not None:
                scope.start(start)
            failed = 0
            try:
                result = fn(*args, **kwargs)
                failed = int(isinstance(result, int) and result != 0)
                return result
            except BaseException:
                failed = 1
                raise
            finally:
                end = time.perf_counter_ns()
                if scope is not None:
                    diag.leave(scope)
                    timing.add_gc_preparation(scope.record())
                _context.reset(token)
                timing.add_interval(name, "host", start, end, failures=failed)

        return wrapped

    return decorate


def timed_transfer_call(fn):
    @functools.wraps(fn)
    def wrapped(self, session_id, buffers, peer_buffer_addresses, lengths):
        active = _context.get()
        if active is None:
            return fn(self, session_id, buffers, peer_buffer_addresses, lengths)
        timing, component = active
        # lengths already lives on the CPU. Count outside the measured call.
        nbytes, segments = int(sum(lengths)), len(lengths)
        start = time.perf_counter_ns()
        scope = _gc_diagnostics.scope() if _gc_diagnostics is not None else None
        if scope is not None:
            scope.enter_engine(start)
        failed = 0
        try:
            result = fn(self, session_id, buffers, peer_buffer_addresses, lengths)
            failed = int(result != 0)
            return result
        except BaseException:
            failed = 1
            raise
        finally:
            end = time.perf_counter_ns()
            if scope is not None:
                scope.phase = "post_engine"
            timing.add_interval(
                component,
                "engine",
                start,
                end,
                bytes=nbytes,
                segments=segments,
                failures=failed,
            )

    return wrapped


class TimingExecutor:
    """Explicit context propagation to per-layer transfer tasks."""

    def __init__(self, executor):
        self.executor = executor

    def submit(self, fn, *args, **kwargs):
        active = _context.get()
        if active is None:
            return self.executor.submit(fn, *args, **kwargs)
        timing, component = active
        submitted = time.perf_counter_ns()

        def run():
            timing.add_interval(
                component, "executor_wait", submitted, time.perf_counter_ns()
            )
            # Propagate only our timing context; leave tracing and other
            # executor-local ContextVars with their original behavior.
            token = _context.set(active)
            try:
                return fn(*args, **kwargs)
            finally:
                _context.reset(token)

        return self.executor.submit(run)

    def __getattr__(self, name):
        return getattr(self.executor, name)


def mark_poll(pollers, polls, event, success):
    for poller, status in zip(pollers, polls):
        timing = getattr(poller, "pd_timing", None)
        if timing is not None and status == success:
            timing.mark(event)


def timed_send_prepare(fn):
    @functools.wraps(fn)
    def wrapped(self, req, last_chunk=False, end_idx=None, **kwargs):
        timing = getattr(req.disagg_kv_sender, "pd_timing", None)
        if timing is not None:
            # Captured by new_chunk before putting work on the transfer queue.
            timing.prepare = dict(
                prepare_begin_ns=time.perf_counter_ns(),
                rid=req.rid,
                attempt=req.prefill_attempt_count,
                input_tokens=len(req.origin_input_ids),
                cached_tokens=req.cached_tokens,
                prefill_start_ns=int(req.time_stats.forward_entry_time * 1e9) or None,
                prefill_result_ready_ns=(
                    int(req.time_stats.prefill_finished_time * 1e9) or None
                ),
                chunk_kind=(
                    "final"
                    if last_chunk
                    else "cached_prefix" if end_idx is not None else "middle"
                ),
            )
        return fn(self, req, last_chunk=last_chunk, end_idx=end_idx, **kwargs)

    return wrapped


def request_record(req, event, **fields):
    stats = req.time_stats
    timestamps = {
        name: int(getattr(stats, name) * 1e9) or None
        for name in (
            "scheduler_recv_time",
            "prefill_bootstrap_queue_entry_time",
            "bootstrap_done_time",
            "wait_queue_entry_time",
            "forward_entry_time",
            "prefill_finished_time",
            "prefill_transfer_queue_entry_time",
            "prefill_kv_transfer_finish_time",
            "decode_prealloc_queue_entry_time",
            "decode_transfer_queue_entry_time",
            "decode_prebuilt_finish_time",
            "completion_time",
        )
    }
    return identity(
        stats.disagg_mode_str(),
        req.bootstrap_room,
        event=event,
        rid=req.rid,
        attempt=req.prefill_attempt_count,
        input_tokens=len(req.origin_input_ids),
        cached_tokens=req.cached_tokens,
        output_tokens=len(req.output_ids),
        timestamps_ns=timestamps,
        **fields,
    )
