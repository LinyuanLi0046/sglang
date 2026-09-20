#!/usr/bin/env python3
"""CPU-only recorder/collector contract checks; no torch, NPU, or live attach."""

import concurrent.futures
import importlib.util
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


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
        assert len(list(output.glob("incident-*"))) == 2
        for report in output.glob("incident-*/report.json"):
            import json
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
    print("PASS: retention, duplicate progress, health exclusion, ring wrap, abort/native lifetime,")
    print("      room reuse, executor context/cancel, Event non-reuse/query-inflight, torn reads,")
    print("      disabled identity, bounded incident persistence/rotation")


if __name__ == "__main__":
    main()
