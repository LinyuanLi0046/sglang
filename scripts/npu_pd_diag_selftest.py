#!/usr/bin/env python3
"""CPU-only recorder/collector contract checks; no torch, NPU, or live attach."""

import concurrent.futures
import asyncio
import importlib.util
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parents[1]
    core = load("diag_test_core", root / "python/sglang/srt/utils/npu_pd_diagnostics.py")
    collect = load("diag_test_collector", root / "scripts/npu_pd_diag_collect.py")
    with tempfile.TemporaryDirectory(prefix="pd-diag-selftest-") as directory:
        recorder = core.Recorder(directory, "prefill", 0, 0)
        core._instance = recorder
        recorder.set_ready()
        # Low-frequency records survive an execution-ring wrap before collection.
        recorder.emit("FRONT_HTTP_ENTER", core.Context(room=42, rid="edge-survives"))
        for _ in range(core.RING * 2):
            recorder.emit("PD_MODEL", core.Context(room=42), reason="batch entered")
        first_reader = collect.Worker(recorder.path)
        first_records, first_state = first_reader.read()
        assert any(r["kind"] == "FRONT_HTTP_ENTER" for r in first_records)
        assert first_state["coverage"]["ring_lost"] > 0
        assert first_state["coverage"]["lifecycle_ring_lost"] == 0
        first_reader.buf.close()
        first_reader.file.close()
        assert not collect.lifecycle_record(dict(kind="PD_MODEL"))

        # procfs streams reject SEEK_END even though ordinary files support it.
        class NoSeek(io.BytesIO):
            def seek(self, *args):
                raise OSError(22, "Invalid argument")
        with patch.object(Path, "open", return_value=NoSeek(b"Name:\tscheduler\n")):
            assert collect.read_proc("/proc/fake/status") == "Name:\tscheduler\n"
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            assert collect.read_proc("/proc/fake/status").startswith("UNAVAILABLE:")

        # ASGI pass-through: no eager body consumption, same messages, request
        # context isolated between concurrent tasks, and exceptions unchanged.
        async def exercise_frontend():
            seen = []
            async def app(scope, receive, send):
                message = await receive()
                assert message["body"] == b"opaque prompt never logged"
                obj = SimpleNamespace(bootstrap_room=int(scope["room"]), rid=scope["rid"])
                core.frontend_request("FRONT_GENERATE", obj)
                await asyncio.sleep(0)
                core.frontend_request("FRONT_IPC_RETURN", obj)
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})
            middleware = core.FrontendMiddleware(app)
            async def one(room):
                messages = []
                receives = 0
                async def receive():
                    nonlocal receives
                    receives += 1
                    return {"type": "http.request", "body": b"opaque prompt never logged"}
                async def send(message):
                    messages.append(message)
                await middleware({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                                  "headers": [(b"x-request-id", f"trace-{room}".encode())],
                                  "room": room, "rid": f"http-{room}"}, receive, send)
                assert receives == 1 and messages[-1]["body"] == b"ok"
                assert core._http_context.get() is None
                seen.extend(messages)
            await asyncio.gather(one(401), one(402))
            failure = asyncio.CancelledError("test cancellation")
            async def failing(scope, receive, send):
                raise failure
            try:
                await core.FrontendMiddleware(failing)(
                    {"type": "http", "method": "POST", "path": "/generate"}, None, None)
            except asyncio.CancelledError as exc:
                assert exc is failure
            else:
                assert False, "middleware swallowed cancellation"
            assert core._http_context.get() is None
        asyncio.run(exercise_frontend())
        front_reader = collect.Worker(recorder.path)
        front_records, _ = front_reader.read()
        contexts = {}
        for entry in front_records:
            if entry.get("rid") in ("http-401", "http-402"):
                contexts.setdefault(entry["rid"], set()).add(entry["call"])
        assert len(contexts) == 2 and all(len(calls) == 1 for calls in contexts.values())
        assert contexts["http-401"] != contexts["http-402"]
        assert not any("opaque prompt" in str(entry) for entry in front_records)
        front_reader.buf.close()
        front_reader.file.close()
        # IPC msgspec.Struct messages reject undeclared attributes, unlike Req.
        class SlottedRequest:
            __slots__ = ("rid", "bootstrap_room")

            def __init__(self):
                self.rid, self.bootstrap_room = "slotted-input", 77

        slotted = SlottedRequest()
        recorder.request(slotted, "ARRIVED", "scheduler received request")
        slot_ctx = recorder.req_context(slotted)
        constructed = SimpleNamespace(rid=slotted.rid, bootstrap_room=77)
        recorder.request(constructed, "ARRIVED", "request constructed")
        assert constructed._pd_diag_context == slot_ctx
        time.sleep(0.02)  # Windows monotonic clock may have a coarse resolution.
        core.next_attempt(slotted, "retry slotted input")
        assert recorder.req_context(slotted).attempt != slot_ctx.attempt
        recorder.request(slotted, "TERMINAL", "complete")
        assert recorder.writer().drops == 0, "slotted input lost its diagnostic context"
        stalled = SimpleNamespace(rid="waiting-request", bootstrap_room=123, req_pool_idx=None)
        recorder.request(stalled, "BOOTSTRAP", "waiting for metadata")
        writer = recorder.writer()
        key = (123, stalled._pd_diag_context.attempt)
        first_progress = writer.requests[key][2]
        for _ in range(10):
            recorder.request(stalled, "BOOTSTRAP", "waiting for metadata")
        assert writer.requests[key][2] == first_progress, "duplicate polls advanced progress"
        # Thousands of normal requests must not evict the stalled active row.
        for i in range(5000):
            req = SimpleNamespace(rid=f"normal-{i}", bootstrap_room=1000 + i, req_pool_idx=i % 64)
            recorder.request(req, "PREALLOC", "queued")
            recorder.request(req, "TERMINAL", "complete", progress=True)
        recorder.request(SimpleNamespace(rid="HEALTH_CHECK_test", bootstrap_room=99), "ARRIVED")
        worker = collect.Worker(recorder.path)
        records, state = worker.read()
        assert any(r["room"] == 123 for r in state["requests"])
        assert all(r["room"] != 99 for r in state["requests"])
        # Bootstrap metadata and completion are different real producer threads.
        # Their fragments must recycle without losing the deliberately stalled row.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(600):
                room = 10000 + i
                recorder.request(room, "METADATA", "received", seen=1, need=1)
                executor.submit(recorder.request, room, "TERMINAL", "done", True).result()
        assert writer.drops == 0, "cross-producer completion failed to retire fragments"
        triggers = worker.triggers([], state, time.monotonic_ns() + 31_000_000_000, 5, 10, 30)
        assert any(t["kind"] == "PD_WAIT_LONG" and not t["stack"] for t in triggers)
        assert state["coverage"]["ring_lost"] > 0
        ctx = stalled._pd_diag_context
        call = recorder.begin("MF_NATIVE", ctx)
        recorder.request(stalled, "TERMINAL", "AbortReq", progress=True)
        _, state = worker.read()
        assert not any(r["room"] == 123 for r in state["requests"])
        assert any(c["kind"] == "MF_NATIVE_ENTER" for c in state["calls"]), "abort drained native call"
        recorder.end(call)
        _, state = worker.read()
        assert not state["calls"]
        # New RID with a reused room is a new attempt, not the terminal old one.
        reused = SimpleNamespace(rid="new-request", bootstrap_room=123, req_pool_idx=0)
        recorder.request(reused, "BOOTSTRAP")
        assert reused._pd_diag_context.attempt != ctx.attempt
        _, state = worker.read()
        assert any(r["rid"] == "new-request" for r in state["requests"])

        # Executor job IDs and room are explicit, including cancellation before start.
        release = threading.Event()
        entered = threading.Event()
        def native():
            entered.set()
            assert recorder.context().room == 123
            release.wait(2)
            return 7
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            recorder.local.context = reused._pd_diag_context
            first = core.submit(executor, native)
            assert entered.wait(1)
            second = core.submit(executor, lambda: 8)
            assert second.cancel()
            _, state = worker.read()
            assert any(j["kind"] == "JOB_RUNNING" for j in state["jobs"])
            assert not any(j["kind"] == "JOB_QUEUED" for j in state["jobs"])
            release.set()
            assert first.result(timeout=2) == 7
        _, state = worker.read()
        assert not state["jobs"]

        # A one-slot fake Event pool cannot record again while query is false.
        done = threading.Event()
        class FakeEvent:
            records = 0
            def record(self, stream):
                self.records += 1
            def query(self):
                return done.is_set()
        event = FakeEvent()
        pool = core.EventPool.__new__(core.EventPool)
        pool.recorder = recorder
        pool.api = SimpleNamespace(set_device=lambda _: None,
                                   current_stream=lambda: SimpleNamespace(npu_stream=123))
        pool.events, pool.free, pool.pending = [event], [0], {}
        pool.lock, pool.disabled = threading.Lock(), False
        pool.record("FAKE_DEVICE", reused._pd_diag_context, 0, None)
        thread = threading.Thread(target=pool.observe, daemon=True)
        thread.start()
        time.sleep(0.15)
        pool.record("CANNOT_REUSE", reused._pd_diag_context, 0, None)
        assert event.records == 1 and not pool.free
        done.set()
        deadline = time.monotonic() + 2
        while not pool.free and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.free == [0]
        pool.disabled = True
        thread.join(1)
        assert not thread.is_alive()

        # Query must be observable while still inside the native binding.
        querying, unblock = threading.Event(), threading.Event()
        class BlockingEvent(FakeEvent):
            def query(self):
                querying.set()
                unblock.wait(2)
                return True
        pool.events = [BlockingEvent()]
        pool.disabled = False
        pool.record("BLOCKING_QUERY", reused._pd_diag_context, 0, None)
        thread = threading.Thread(target=pool.observe, daemon=True)
        thread.start()
        assert querying.wait(1)
        observed = core.read_slot(recorder.buf, core.EVENT_BASE)
        assert observed["kind"] == "EVENT_QUERY_ENTER" and not pool.free
        pool.disabled = True
        unblock.set()
        thread.join(1)
        assert not thread.is_alive()

        _, state = worker.read()
        output = Path(directory) / "artifacts"
        output.mkdir()
        journal = collect.Journal(output, limit=4096)
        journal.append({"check": "selftest"})
        for _ in range(3):
            collect.save_incident(output, [worker], [{"kind": "MANUAL"}], journal, False, [])
        assert len(list(output.glob("incident-*"))) == 3
        for report in output.glob("incident-*/report.json"):
            assert json.loads(report.read_text())["workers"]
        journal.file.close()

        # Tear detection and identity-unavailable (Windows) must fail closed.
        offset = core.HEADER + core.SLOT
        original = recorder.buf[offset + 20]
        recorder.buf[offset + 20] = original ^ 1
        assert core.read_slot(recorder.buf, offset).get("torn")
        recorder.buf[offset + 20] = original
        def original_function():
            return 42
        core.REQUESTED = False
        assert core.traced("OFF")(original_function) is original_function
        core._instance = None
        with core.span("OFF"):
            pass
        worker.buf.close()
        worker.file.close()
        recorder.buf.close()

    # One collector must retain all P4+D4 raw maps, not just the first seven.
    with tempfile.TemporaryDirectory(prefix="pd-diag-paired-") as directory:
        source, output = Path(directory) / "source", Path(directory) / "output"
        output.mkdir()
        old = core.Recorder(source, "prefill", 0, 0)
        manifest = json.loads(old.path.with_suffix(".json").read_text())
        manifest["pid"] = -1
        old.path.with_suffix(".json").write_text(json.dumps(manifest))
        current = [core.Recorder(source, role, rank, rank)
                   for role in ("prefill", "decode") for rank in range(4)]
        workers, retired = {}, set()
        journal = collect.Journal(output, limit=2048)
        lifecycle = collect.Journal(output, limit=4096, prefix="lifecycle")
        with patch.object(collect, "worker_liveness", side_effect=lambda m: False if m["pid"] == -1 else None):
            collect.discover_workers([str(source)], workers, retired, lifecycle)
        assert str(old.path) in retired and len(workers) == 8
        for writer in current:
            writer.set_ready()
        for worker in workers.values():
            worker.read()
        collect.save_incident(output, list(workers.values()), [{"kind": "MANUAL"}],
                              journal, False, [], lifecycle=lifecycle)
        incident = next(output.glob("incident-*"))
        report = json.loads((incident / "report.json").read_text())
        assert report["raw_complete"] and len(report["raw_artifacts"]) == 8
        assert report["raw_expected_bytes"] == 8 * core.SIZE
        assert len(list(incident.glob("*.mmap"))) == 8
        assert len(list(incident.glob("decode-*.json"))) == 4
        assert report["journals"]["lifecycle"]

        # Initial warmup markers cannot fill every observation snapshot.
        warm = core._pack("EVENT_PENDING", core.Context(), reason="EVENT_WARM")
        core.write_slot(current[0].buf, core.EVENT_BASE, 1, warm)
        worker = workers[str(current[0].path)]
        _, snapshot = worker.read()
        assert not snapshot["events"]
        assert "fragments" not in str(collect.compact_observation(snapshot))
        gone = dict(snapshot, alive=False)
        assert [t["kind"] for t in worker.triggers([], gone, time.monotonic_ns(), 5, 10, 30)] == ["PROCESS_GONE"]

        # Protect the first wait while that cohort remains stalled, even when
        # another request stalls later. A later episode can replace it.
        guard = collect.WaitSnapshot()
        first = dict(kind="PD_WAIT_LONG", worker=str(worker.path), key=[123, 1])
        second = dict(first, key=[456, 2])
        assert guard.update([first], workers)
        assert not guard.update([], workers), "temporary progress unlocked the first snapshot"
        assert not guard.update([first], workers), "same pending request replaced first snapshot"
        assert not guard.update([first, second], workers)
        worker.terminals[(123, 1)] = time.monotonic_ns()
        assert not guard.update([second], workers), "overlapping waiter lost the initial snapshot"
        assert not guard.update([], workers)
        worker.terminals[(456, 2)] = time.monotonic_ns()
        third = dict(first, key=[789, 3])
        assert guard.update([third], workers)
        collect.save_incident(output, [worker], [first], journal, False, [],
                              lifecycle=lifecycle, first_wait=True)
        pinned_path = next(output.glob("incident-*-first-wait/report.json"))
        pinned = pinned_path.read_bytes()
        collect.save_incident(output, [worker], [{"kind": "MANUAL"}], journal, False, [])
        collect.save_incident(output, [worker], [{"kind": "MANUAL"}], journal, False, [])
        assert pinned_path.read_bytes() == pinned

        # High-frequency execution history must not evict PD lifecycle evidence.
        entry = dict(kind="PD_TERMINAL", room=123, reason="failed before peer arrived")
        assert collect.lifecycle_record(entry)
        assert not collect.lifecycle_record(dict(kind="EVENT_QUERY_RETURN"))
        assert not collect.lifecycle_record(dict(kind="PD_MODEL"))
        lifecycle.append(dict(event=entry))
        lifecycle.flush()
        for i in range(100):
            journal.append(dict(kind="EVENT_QUERY_RETURN", tick=i))
        assert "failed before peer arrived" in lifecycle.path.read_text()
        # More than four segments, including across a collector restart: no
        # old file may be renamed, overwritten, or deleted.
        archival = collect.Journal(output, limit=64, prefix="archive-test")
        for i in range(20):
            archival.append({"i": i, "data": "x" * 64})
        archival.file.close()
        archival = collect.Journal(output, limit=64, prefix="archive-test")
        archival.append({"i": 20})
        archival.file.close()
        rows = [json.loads(line) for path in output.glob("archive-test-*.jsonl")
                for line in path.read_text().splitlines()]
        assert sorted(row["i"] for row in rows) == list(range(21))

        # A mounted foreign namespace is unknown, not dead: keep reading it
        # and never attach to a coincidentally matching local PID.
        manifest = dict(worker.manifest, identity={"starttime": "42", "state": "S"},
                        pid_namespace="pid:[peer]")
        with patch.object(collect.core, "pid_namespace", return_value="pid:[local]"), \
             patch.object(collect.core, "process_identity", return_value={"starttime": "42", "state": "S"}):
            assert collect.worker_liveness(manifest) is None
            same_namespace = dict(manifest, pid_namespace="pid:[local]")
            assert collect.worker_liveness(same_namespace) is True
            assert collect.worker_liveness(dict(manifest, hostname="other-host")) is None
        with patch.object(collect.core, "pid_namespace", return_value="pid:[local]"), \
             patch.object(collect.core, "process_identity", return_value={"starttime": "unknown", "state": "unknown"}), \
             patch.object(collect.os, "stat", side_effect=PermissionError("cannot inspect peer")):
            assert collect.worker_liveness(same_namespace) is None
        with patch.object(collect.core, "pid_namespace", return_value="pid:[local]"), \
             patch.object(collect.core, "process_identity", return_value={"starttime": "unknown", "state": "unknown"}), \
             patch.object(collect.os, "stat", side_effect=FileNotFoundError("process exited")):
            assert collect.worker_liveness(same_namespace) is False

        # Incomplete copies must be explicit in the saved report, not silent.
        class UnreadableBuffer:
            def __getitem__(self, _):
                raise OSError("simulated raw copy failure")
        original_buffer = worker.buf
        worker.buf = UnreadableBuffer()
        collect.save_incident(output, [worker], [{"kind": "MANUAL"}], journal, False, [])
        worker.buf = original_buffer
        reports = [json.loads(p.read_text()) for p in output.glob("incident-*/report.json")]
        assert any(not r["raw_complete"] and "error" in r["raw_artifacts"][0] for r in reports)
        journal.file.close()
        lifecycle.file.close()
        for reader in workers.values():
            reader.buf.close()
            reader.file.close()
        for writer in [old] + current:
            writer.buf.close()
    print("PASS: retention, duplicate progress, health exclusion, ring wrap, abort/native lifetime,")
    print("      room reuse, executor context/cancel, Event non-reuse/query-inflight, torn reads,")
    print("      disabled identity, append-only journals and incident retention")
    print("      slotted IPC input/retry, P4+D4 complete raw maps, old-run filtering,")
    print("      warmup exclusion, first-wait protection, independent lifecycle retention")
    print("      unknown peer namespace/permissions, explicit raw copy failure")
    print("      procfs without seek, isolated lifecycle ring, concurrent ASGI identity/cancellation")


if __name__ == "__main__":
    main()
