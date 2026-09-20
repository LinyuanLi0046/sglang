"""Opt-in, bounded CPU flight recorder for NPU disaggregation.

No device reads, synchronization, logging, or filesystem writes in record().
The file-backed mmap deliberately is NOT multiprocessing shared memory. Its
sequence/checksum protocol is best-effort tear detection, not native atomics.
See scripts/npu_pd_diag.md for coverage, costs, and the independent collector.
"""

import contextlib
import functools
import json
import mmap
import os
import struct
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

REQUESTED = os.getenv("SGLANG_NPU_PD_DIAG", "0") == "1"
VERSION = 1
SLOT = 512
HEADER = 4096
WRITERS = 32
RING = 1024
REQUESTS = 256
CALLS = 32
EVENTS = 2048
JOBS = 1024
STRIDE = 1 + RING + REQUESTS + CALLS
EVENT_BASE = HEADER + WRITERS * STRIDE * SLOT
JOB_BASE = EVENT_BASE + EVENTS * SLOT
SIZE = JOB_BASE + JOBS * SLOT
PAYLOAD = struct.Struct("<QQQQqqqqq40s80s96s16q")
U64 = struct.Struct("<Q")
CRC = struct.Struct("<I")
VALUE_NAMES = (
    "status", "slot", "generation", "seen", "need", "done", "need_done",
    "sent", "need_sent", "source_mask", "available", "needed", "cached",
    "start", "end", "extra",
)
_instance = None


class Context(NamedTuple):
    room: int = -1
    attempt: int = 0
    batch: int = 0
    call: int = 0
    parent: int = 0
    rid: str = ""


def _int(value):
    # Never implicitly convert a Tensor, including a CPU Tensor.
    if not isinstance(value, int):
        return -1
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


def _text(value, limit):
    return str(value).encode("utf-8", errors="replace")[:limit]


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return {"starttime": fields[19], "state": fields[0]}
    except (OSError, IndexError):
        return {"starttime": "unknown", "state": "unknown"}


def pid_namespace():
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown"


def _pack(kind, ctx, reason="", values=None, enter=0, progress=0, phase_enter=0):
    values = values or {}
    return PAYLOAD.pack(
        time.monotonic_ns(), enter, progress, phase_enter,
        _int(ctx.room), _int(ctx.attempt), _int(ctx.batch), _int(ctx.call),
        _int(ctx.parent), _text(kind, 40), _text(ctx.rid, 80), _text(reason, 96),
        *(_int(values.get(name, -1)) for name in VALUE_NAMES),
    )


def write_slot(buf, offset, seq, payload):
    U64.pack_into(buf, offset, 0)
    buf[offset + 8:offset + 8 + len(payload)] = payload
    CRC.pack_into(buf, offset + SLOT - 16, zlib.crc32(payload))
    U64.pack_into(buf, offset + SLOT - 8, seq)
    U64.pack_into(buf, offset, seq)


def read_slot(buf, offset):
    before = U64.unpack_from(buf, offset)[0]
    if not before:
        return None
    raw = bytes(buf[offset:offset + SLOT])
    after = U64.unpack_from(buf, offset)[0]
    seq = U64.unpack_from(raw)[0]
    payload = raw[8:8 + PAYLOAD.size]
    if (before != seq or seq != after or seq != U64.unpack_from(raw, SLOT - 8)[0]
            or zlib.crc32(payload) != CRC.unpack_from(raw, SLOT - 16)[0]):
        return {"torn": True}
    p = PAYLOAD.unpack(payload)
    result = dict(zip(("ts", "enter", "progress", "phase_enter", "room", "attempt",
                       "batch", "call", "parent", "kind", "rid", "reason"), p[:12]))
    for key in ("kind", "rid", "reason"):
        result[key] = result[key].split(b"\0", 1)[0].decode("utf-8", errors="replace")
    result.update(zip(VALUE_NAMES, p[12:]))
    result["seq"] = seq
    return result


class _Writer:
    def __init__(self, recorder, index):
        self.recorder = recorder
        self.base = HEADER + index * STRIDE * SLOT
        self.sequence = 0
        self.requests = OrderedDict()
        self.closed_hints = set()
        self.calls = {}
        self.drops = 0
        self.tid = threading.get_native_id()

    def put(self, kind, ctx, reason="", values=None, **times):
        self.sequence += 1
        payload = _pack(kind, ctx, reason, values, **times)
        write_slot(self.recorder.buf, self.base + (1 + (self.sequence - 1) % RING) * SLOT,
                   self.sequence, payload)
        write_slot(self.recorder.buf, self.base, self.sequence,
                   _pack("WRITER", Context(), values={"status": self.drops,
                         "slot": self.tid, "extra": self.sequence}))
        return payload

    def request(self, ctx, phase, reason, progress, values):
        values = {name: _int(value) for name, value in values.items()}
        key = (ctx.room, ctx.attempt)
        now = time.monotonic_ns()
        old = self.requests.get(key)
        if old is None:
            if len(self.requests) >= REQUESTS:
                terminal = next((k for k, v in self.requests.items()
                                 if v[5] or k in self.closed_hints), None)
                if terminal is None:
                    self.drops += 1
                    self.put("COVERAGE_GAP", ctx, "request table full")
                    return
                index = self.requests.pop(terminal)[0]
                self.closed_hints.discard(terminal)
            else:
                index = len(self.requests)
            old = (index, now, now, now, "", False, {}, "", 0)
        index, enter, last, phase_enter, previous, terminal, data, old_reason, updated = old
        changed = phase != previous or reason != old_reason
        # Poll status/count changes are progress; capacity changes alone are not.
        real = progress or any(
            name in values and values[name] != data.get(name, -1)
            for name in ("status", "seen", "done", "sent")
        )
        if reason.endswith("poll") and previous and not real:
            return  # Do not overwrite a resource blocker with repeated polling.
        if phase != previous:
            phase_enter = now
        if real:
            last = now
        if not changed and not real and now - updated < 1_000_000_000:
            return
        data = dict(data, **values)
        terminal = phase in ("TERMINAL", "CLEARED")
        payload = self.put("PD_" + phase, ctx, reason, data,
                           enter=enter, progress=last, phase_enter=phase_enter)
        write_slot(self.recorder.buf, self.base + (1 + RING + index) * SLOT,
                   self.sequence, payload)
        self.requests[key] = (index, enter, last, phase_enter, phase, terminal,
                              data, reason, now)


class Recorder:
    def __init__(self, directory, role, rank, device, level="cpu"):
        self.pid = os.getpid()
        self.role, self.rank, self.device, self.level = role, rank, device, level
        self.local = threading.local()
        self.lock = threading.Lock()
        self.writers = []
        self.contexts = OrderedDict()
        self.closed_contexts = OrderedDict()
        self.ready = False
        self.device_pool = None
        self.job_lock = threading.Lock()
        self.job_free = list(range(JOBS))
        self.registration_gaps = 0
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Reserve backing pages at startup. posix_fallocate fails cleanly if a
        # container's /dev/shm is too small, instead of a later mmap SIGBUS.
        identity = process_identity(self.pid)
        self.path = directory / f"{role}-{rank}-{self.pid}-{time.time_ns()}.mmap"
        with self.path.open("x+b") as f:
            f.truncate(SIZE)
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(f.fileno(), 0, SIZE)
            self.buf = mmap.mmap(f.fileno(), SIZE, access=mmap.ACCESS_WRITE)
        self.buf[:] = bytes(SIZE)
        self.buf[:16] = b"SGLANG_PD_DIAG1\0"
        manifest = dict(version=VERSION, pid=self.pid, role=role, rank=rank,
                        device=device, level=level, identity=identity,
                        pid_namespace=pid_namespace(),
                        hostname=os.uname().nodename if hasattr(os, "uname") else "unknown",
                        created_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                        size=SIZE, event_coverage="cpu-only until EVENT_POOL_READY",
                        limitations=["best-effort mmap publication", "no MF internal CQ instrumentation",
                                     "no page-allocation generation", "no device synchronization"])
        self.path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
        self.emit("INIT")

    def writer(self):
        writer = getattr(self.local, "writer", None)
        if writer is not None:
            return writer
        if not self.lock.acquire(blocking=False):
            self.registration_gaps += 1
            U64.pack_into(self.buf, 24, self.registration_gaps)
            return None
        try:
            if len(self.writers) >= WRITERS:
                self.registration_gaps += 1
                U64.pack_into(self.buf, 24, self.registration_gaps)
                return None
            writer = _Writer(self, len(self.writers))
            self.writers.append(writer)
            self.local.writer = writer
            return writer
        finally:
            self.lock.release()

    def context(self):
        return getattr(self.local, "context", Context())

    def set_ready(self):
        self.ready = True
        U64.pack_into(self.buf, 32, 1)
        self.emit("WORKER_READY")

    def job_slot(self, ctx):
        if not self.job_lock.acquire(blocking=False):
            self.emit("COVERAGE_GAP", reason="job slot reservation busy")
            return None
        try:
            if not self.job_free:
                self.emit("COVERAGE_GAP", reason="job table full")
                return None
            index = self.job_free.pop()
            write_slot(self.buf, JOB_BASE + index * SLOT, ctx.call,
                       _pack("JOB_QUEUED", ctx, enter=ctx.call, progress=ctx.call))
            return index
        finally:
            self.job_lock.release()

    def job_state(self, index, kind, ctx):
        if index is None:
            return
        # Ownership is handed submitter -> executor -> future callback. A
        # cancelled-before-start future has no executor writer for this slot.
        write_slot(self.buf, JOB_BASE + index * SLOT, ctx.call,
                   _pack(kind, ctx, enter=ctx.call, progress=time.monotonic_ns()))
        if kind in ("JOB_FINISHED", "JOB_CANCELLED", "JOB_SUBMIT_EXCEPTION"):
            if self.job_lock.acquire(blocking=False):
                try:
                    self.job_free.append(index)
                finally:
                    self.job_lock.release()
            else:
                self.emit("COVERAGE_GAP", reason="job slot quarantined")

    def room_context(self, room):
        room = _int(room)
        ctx = self.contexts.get(room)
        if ctx is None:
            ctx = Context(room=room, attempt=time.monotonic_ns())
            if len(self.contexts) >= 4096:
                retired = next((key for key in self.closed_contexts if key in self.contexts), None)
                if retired is not None:
                    self.contexts.pop(retired, None)
                    self.closed_contexts.pop(retired, None)
                else:
                    self.contexts.popitem(last=False)
                    self.emit("COVERAGE_GAP", reason="active room context registry eviction")
            self.contexts[room] = ctx
        return ctx

    def _cache_req_context(self, req, ctx):
        # Input messages (e.g. msgspec.Struct) may reject dynamic attributes.
        # The bounded room registry also carries their context into scheduler Req.
        try:
            req._pd_diag_context = ctx
        except (AttributeError, TypeError):
            pass
        self.contexts[ctx.room] = ctx

    def req_context(self, req):
        rid = getattr(req, "rid", "")
        if not isinstance(rid, str):
            return None
        if rid.startswith("HEALTH_CHECK"):
            room = getattr(req, "bootstrap_room", None)
            if room is not None:
                ctx = self.room_context(room)._replace(rid=rid)
                self.contexts[ctx.room] = ctx
                writer = self.writer()
                if writer is not None:
                    writer.request(ctx, "CLEARED", "health check excluded", False, {})
            return None
        ctx = getattr(req, "_pd_diag_context", None)
        if ctx is None:
            room = getattr(req, "bootstrap_room", None)
            if room is None:
                return None
            ctx = self.room_context(room)
            if ctx.rid and ctx.rid != rid:
                ctx = Context(room=ctx.room, attempt=time.monotonic_ns())
                self.closed_contexts.pop(ctx.room, None)
            ctx = ctx._replace(rid=rid)
            self._cache_req_context(req, ctx)
        return ctx

    def emit(self, kind, ctx=None, reason="", **values):
        try:
            if kind == "OBSERVER_ALIVE":
                U64.pack_into(self.buf, 48, time.monotonic_ns())
            if kind == "EVENT_POOL_READY":
                U64.pack_into(self.buf, 40, U64.unpack_from(self.buf, 40)[0] | 1)
            elif kind == "EVENT_UNSUPPORTED":
                U64.pack_into(self.buf, 40, U64.unpack_from(self.buf, 40)[0] | 4)
            elif kind.startswith("EVENT") and ("EXCEPTION" in kind or "GAP" in kind or "FULL" in kind):
                U64.pack_into(self.buf, 40, U64.unpack_from(self.buf, 40)[0] | 2)
            writer = self.writer()
            if writer is not None:
                if kind == "COVERAGE_GAP":
                    writer.drops += 1
                writer.put(kind, ctx or self.context(), reason, values)
                return True
        except Exception:
            self.registration_gaps += 1
            U64.pack_into(self.buf, 24, self.registration_gaps)
        return False

    def request(self, req_or_room, phase, reason="", progress=False, **values):
        try:
            ctx = (req_or_room if isinstance(req_or_room, Context) else
                   self.room_context(req_or_room) if isinstance(req_or_room, int) else
                   self.req_context(req_or_room))
            if ctx is None:
                return
            if ctx.rid.startswith("HEALTH_CHECK"):
                return
            if phase in ("TERMINAL", "CLEARED"):
                self.closed_contexts[ctx.room] = ctx.attempt
                if len(self.closed_contexts) > 4096:
                    self.closed_contexts.popitem(last=False)
                # Publication remains producer-local. A bounded CPU-only retire
                # hint lets an idle metadata producer reclaim its old fragment
                # when completion was observed by another thread. Never retire
                # an active native call, and always match the exact attempt.
                key = (ctx.room, ctx.attempt)
                for owner in tuple(self.writers):
                    if key in owner.requests:
                        owner.closed_hints.add(key)
            if not isinstance(req_or_room, (int, Context)):
                values["slot"] = _int(getattr(req_or_room, "req_pool_idx", None))
            writer = self.writer()
            if writer is not None:
                writer.request(ctx, phase, reason, progress, values)
        except Exception:
            self.emit("COVERAGE_GAP", reason="request hook failed")

    def begin(self, kind, ctx=None, **values):
        try:
            writer = self.writer()
            if writer is None:
                return None
            parent = ctx or self.context()
            ctx = parent._replace(call=time.monotonic_ns(), parent=parent.call)
            free = next((i for i in range(CALLS) if i not in writer.calls), None)
            payload = writer.put(kind + "_ENTER", ctx, values=values,
                                 enter=ctx.call, progress=ctx.call)
            if free is None:
                writer.drops += 1
                writer.put("COVERAGE_GAP", ctx, "active call table full")
                return None
            writer.calls[free] = (kind, ctx)
            write_slot(self.buf, writer.base + (1 + RING + REQUESTS + free) * SLOT,
                       writer.sequence, payload)
            return writer, free, kind, ctx, parent
        except Exception:
            self.emit("COVERAGE_GAP", reason="begin hook failed")
            return None

    def end(self, token, error=None, **values):
        if token is None:
            return
        try:
            writer, index, kind, ctx, _ = token
            payload = writer.put(kind + ("_EXCEPTION" if error else "_RETURN"), ctx,
                                 type(error).__name__ if error else "", values,
                                 enter=ctx.call, progress=time.monotonic_ns())
            write_slot(self.buf, writer.base + (1 + RING + REQUESTS + index) * SLOT,
                       writer.sequence, payload)
            writer.calls.pop(index, None)
        except Exception:
            self.emit("COVERAGE_GAP", reason="end hook failed")

    def mark(self, kind, layer=-1, stream=None):
        self.emit(kind, extra=layer)
        pool = self.device_pool
        if self.ready and pool is not None and self.level != "cpu":
            try:
                pool.record(kind, self.context(), layer, stream)
            except Exception:
                pool.disabled = True
                self.emit("EVENT_COVERAGE_GAP", reason="record hook failed")


def get():
    return _instance if _instance is not None and _instance.pid == os.getpid() else None


def initialize(server_args, rank, device):
    global _instance
    if not REQUESTED or getattr(server_args, "device", None) != "npu":
        return
    role = getattr(server_args, "disaggregation_mode", "null")
    if role not in ("prefill", "decode") or get() is not None:
        return
    try:
        level = os.getenv("SGLANG_NPU_PD_DIAG_LEVEL", "cpu")
        if level not in ("cpu", "coarse", "stage"):
            raise ValueError("diagnostic level must be cpu, coarse, or stage")
        _instance = Recorder(os.getenv("SGLANG_NPU_PD_DIAG_DIR", "/dev/shm/sglang_pd_diag"),
                             role, rank, device, level)
    except Exception as exc:
        # Startup only. Diagnostics must never prevent serving.
        import logging
        logging.getLogger(__name__).warning("PD diagnostic initialization disabled: %s", exc)


def request(req_or_room, phase, reason="", progress=False, **values):
    recorder = get()
    if recorder is not None:
        recorder.request(req_or_room, phase, reason, progress, **values)


def emit(kind, ctx=None, reason="", **values):
    recorder = get()
    if recorder is not None:
        return recorder.emit(kind, ctx, reason, **values)
    return False


@contextlib.contextmanager
def span(kind, ctx=None, **values):
    recorder = get()
    if recorder is None:
        yield
        return
    token = recorder.begin(kind, ctx, **values)
    previous = recorder.context()
    if token is not None:
        recorder.local.context = token[3]
    try:
        yield
    except BaseException as exc:
        recorder.end(token, exc)
        raise
    else:
        recorder.end(token)
    finally:
        recorder.local.context = previous


def traced(kind, batch=False, room=False, device=False):
    def decorate(fn):
        if not REQUESTED:
            return fn  # Off path keeps the original function, not a wrapper.

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            recorder = get()
            if recorder is None:
                return fn(*args, **kwargs)
            ctx = recorder.context()
            try:
                if room and args:
                    owner = args[0]
                    rid = getattr(owner, "bootstrap_room", None)
                    if rid is not None:
                        ctx = recorder.room_context(rid)
                if batch:
                    batch_arg = args[1] if len(args) > 1 else kwargs.get("batch")
                    batch_id = (time.monotonic_ns() if kind == "RUN_BATCH" else
                                getattr(batch_arg, "_pd_diag_batch", 0) or time.monotonic_ns())
                    if batch_arg is not None:
                        batch_arg._pd_diag_batch = batch_id
                    ctx = ctx._replace(batch=batch_id)
                    for req in getattr(batch_arg, "reqs", ()):
                        req_ctx = recorder.req_context(req)
                        if req_ctx is not None:
                            recorder.emit("BATCH_MEMBER", req_ctx._replace(batch=ctx.batch),
                                          slot=getattr(req, "req_pool_idx", None))
                            if kind == "RUN_BATCH":
                                recorder.request(req, "MODEL", "batch entered", progress=True)
            except Exception:
                recorder.emit("COVERAGE_GAP", reason="call context hook failed")
            with span(kind, ctx):
                if device:
                    recorder.mark(kind + "_DEVICE_BEGIN")
                result = fn(*args, **kwargs)
                if device:
                    recorder.mark(kind + "_DEVICE_END")
                return result
        return wrapped
    return decorate


def submit(executor, fn, *args, **kwargs):
    """Explicit immutable context propagation across executor boundaries."""
    recorder = get()
    if recorder is None:
        return executor.submit(fn, *args, **kwargs)
    ctx = recorder.context()._replace(call=time.monotonic_ns())
    index = recorder.job_slot(ctx)
    recorder.emit("JOB_SUBMIT", ctx)

    def run():
        recorder.job_state(index, "JOB_RUNNING", ctx)
        with span("KV_JOB", ctx):
            return fn(*args, **kwargs)

    try:
        future = executor.submit(run)
    except BaseException:
        recorder.job_state(index, "JOB_SUBMIT_EXCEPTION", ctx)
        raise
    # Completion may run on a different producer; immutable ctx is authoritative.
    def finished(f):
        kind = "JOB_CANCELLED" if f.cancelled() else "JOB_FINISHED"
        recorder.job_state(index, kind, ctx)
        recorder.emit(kind, ctx)
    future.add_done_callback(finished)
    return future


class EventPool:
    """No elapsed timing, wait, or synchronize. False query retains ownership."""
    def __init__(self, recorder, device_api):
        self.recorder, self.api = recorder, device_api
        self.lock = threading.Lock()
        self.events = [device_api.Event(enable_timing=False) for _ in range(EVENTS)]
        self.free = list(range(EVENTS))
        self.pending = {}
        self.disabled = False
        # Event creation can be lazy. Record each once during startup; never
        # recycle its handle until the observer sees completion.
        for _ in range(EVENTS):
            self.record("EVENT_WARM", Context(), -1, None)
        threading.Thread(target=self.observe, name="pd-diag-events", daemon=True).start()

    def record(self, kind, ctx, layer, stream):
        if self.disabled or not self.lock.acquire(blocking=False):
            self.recorder.emit("EVENT_COVERAGE_GAP", reason="event lock or disabled")
            return
        try:
            if not self.free:
                self.recorder.emit("EVENT_POOL_FULL")
                return
            index = self.free.pop()
            marker = ctx._replace(call=time.monotonic_ns())
            # Reserve before leaving the lock; observer cannot query RECORDING.
            self.pending[index] = ("RECORDING", marker, kind, layer)
        finally:
            self.lock.release()
        stream = stream if stream is not None else self.api.current_stream()
        stream_id = getattr(stream, "npu_stream", getattr(stream, "cuda_stream", -1))
        if not self.recorder.emit("EVENT_RECORD_ENTER", marker, reason=kind, slot=index,
                                  extra=layer, source_mask=stream_id):
            self.disabled = True
            return
        write_slot(self.recorder.buf, EVENT_BASE + index * SLOT, marker.call,
                   _pack("EVENT_RECORD_ENTER", marker, kind, {"slot": index, "extra": layer},
                         enter=marker.call, progress=marker.call))
        try:
            self.events[index].record(stream)
        except Exception as exc:
            self.disabled = True  # Quarantine handles; never destroy/recycle here.
            self.recorder.emit("EVENT_RECORD_EXCEPTION", marker, reason=type(exc).__name__)
            return
        # The observer only sees published records after the short registry lock.
        if not self.lock.acquire(blocking=False):
            self.disabled = True
            self.recorder.emit("EVENT_COVERAGE_GAP", reason="record publication busy")
            return
        try:
            write_slot(self.recorder.buf, EVENT_BASE + index * SLOT, marker.call,
                       _pack("EVENT_PENDING", marker, kind, {"slot": index, "extra": layer},
                             enter=marker.call, progress=marker.call))
            self.pending[index] = ("RECORDED", marker, kind, layer)
        finally:
            self.lock.release()

    def observe(self):
        try:
            self.api.set_device(self.recorder.device)
            while not self.disabled:
                self.recorder.emit("OBSERVER_ALIVE")
                with self.lock:
                    pending = tuple(self.pending.items())
                # Bounded native-query work per tick, including startup warming.
                pending = sorted(pending, key=lambda item: item[1][1].call)
                for index, (state, ctx, kind, layer) in pending[:64]:
                    if state != "RECORDED":
                        continue
                    # Native query may hold the GIL: publish evidence BEFORE it.
                    if not self.recorder.emit("EVENT_QUERY_ENTER", ctx, reason=kind, slot=index):
                        continue
                    write_slot(self.recorder.buf, EVENT_BASE + index * SLOT, ctx.call,
                               _pack("EVENT_QUERY_ENTER", ctx, kind, {"slot": index},
                                     enter=ctx.call, progress=time.monotonic_ns()))
                    done = self.events[index].query()
                    self.recorder.emit("EVENT_QUERY_RETURN", ctx, slot=index, status=int(done))
                    write_slot(self.recorder.buf, EVENT_BASE + index * SLOT, ctx.call,
                               _pack("EVENT_DONE" if done else "EVENT_PENDING", ctx, kind,
                                     {"slot": index, "status": int(done), "extra": layer},
                                     enter=ctx.call, progress=time.monotonic_ns()))
                    if done:
                        with self.lock:
                            del self.pending[index]
                            self.free.append(index)
                time.sleep(0.1)
        except Exception as exc:
            self.disabled = True
            self.recorder.emit("EVENT_QUERY_EXCEPTION", reason=type(exc).__name__)


def prepare_events(device_api, server_args):
    recorder = get()
    if recorder is None or recorder.level == "cpu" or recorder.device_pool is not None:
        return
    # This first instrumentation targets eager, non-MTP execution. Never inject
    # Events into graph capture/replay or alter an unsupported execution path.
    if (not getattr(server_args, "disable_cuda_graph", False)
            or getattr(server_args, "speculative_algorithm", None)):
        recorder.emit("EVENT_UNSUPPORTED", reason="Graph/MTP: CPU coverage only")
        return
    try:
        recorder.device_pool = EventPool(recorder, device_api)
        recorder.emit("EVENT_POOL_READY")
    except Exception as exc:
        recorder.emit("EVENT_INIT_EXCEPTION", reason=type(exc).__name__)


def mark(kind, layer=-1, stream=None):
    recorder = get()
    if recorder is not None:
        recorder.mark(kind, layer, stream)


def layer_mark(kind, layer, last=False):
    recorder = get()
    if recorder is None:
        return
    recorder.emit(kind, extra=layer)
    recorder.local.layer = layer
    # Coarse: endpoints of eight-layer groups plus the source/consumer edges.
    selected = layer in (0, 1, 7, 15, 23, 31, 39, 47, 48) or last
    if recorder.level == "stage":
        selected = layer in (0, 1, 24, 47, 48) or last
    if selected and kind == "LAYER_RETURN":
        recorder.mark("LAYER_DEVICE", layer)


def stage_mark(kind, stream=None):
    recorder = get()
    if recorder is None:
        return
    layer = getattr(recorder.local, "layer", -1)
    recorder.emit(kind, extra=layer)
    if recorder.level == "stage" and layer in (0, 1, 24, 47, 48):
        recorder.mark(kind + "_DEVICE", layer, stream)


def release_snapshot(req, tree_cache):
    recorder = get()
    if recorder is None:
        return
    try:
        ctx = recorder.req_context(req)
        if ctx is None:
            return
        slot = getattr(req, "req_pool_idx", None)
        generation = -1
        table = getattr(getattr(tree_cache, "req_to_token_pool", None), "req_generation", None)
        # Explicitly CPU-owned generation table. Never copy a device tensor.
        if slot is not None and table is not None and table.device.type == "cpu":
            generation = int(table[slot])
        recorder.emit("KV_RELEASE_ENTER", ctx, slot=slot, generation=generation,
                      reason="logical release; radix pages may remain resident")
    except Exception:
        recorder.emit("COVERAGE_GAP", reason="release snapshot failed")


def page_snapshot(req, page_indices):
    """Only already materialized CPU numpy page IDs; bounded first/last sample."""
    recorder = get()
    if recorder is None:
        return
    try:
        count = len(page_indices)
        ctx = recorder.req_context(req)
        if count and ctx is not None:
            sample = page_indices[:4].tobytes() + page_indices[-4:].tobytes()
            recorder.emit("KV_PAGE_SAMPLE", ctx, available=count,
                          start=int(page_indices[0]), end=int(page_indices[-1]),
                          extra=zlib.crc32(sample), reason="bounded sample; not page generation")
    except Exception:
        recorder.emit("COVERAGE_GAP", reason="CPU page snapshot failed")


def next_attempt(req, reason):
    recorder = get()
    if recorder is None:
        return
    try:
        previous = recorder.req_context(req)
        if previous is None:
            return
        recorder.request(previous, "TERMINAL", "attempt ended: " + reason, progress=True)
        ctx = previous._replace(attempt=time.monotonic_ns(), batch=0, call=0, parent=0)
        recorder._cache_req_context(req, ctx)
        recorder.closed_contexts.pop(ctx.room, None)
        recorder.request(ctx, "RETRY", reason, progress=True)
    except Exception:
        recorder.emit("COVERAGE_GAP", reason="attempt hook failed")


def emergency_note(kind):
    """Parent crash path only, not the request hot path; best-effort bounded note."""
    if not REQUESTED:
        return
    if get() is not None:
        get().emit(kind)
        return
    try:
        directory = Path(os.getenv("SGLANG_NPU_PD_DIAG_DIR", "/dev/shm/sglang_pd_diag"))
        note = dict(kind=kind, pid=os.getpid(), ts=time.monotonic_ns(), time_ns=time.time_ns(),
                    identity=process_identity(os.getpid()))
        (directory / f"notice-{os.getpid()}.json").write_text(json.dumps(note))
    except Exception:
        pass  # Never delay/replace the original crash handling on failure.
