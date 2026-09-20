#!/usr/bin/env python3
"""Independent NPU PD collector. No torch import and no device API calls."""

import argparse
import importlib.util
import json
import mmap
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

CORE_PATH = Path(__file__).resolve().parents[1] / "python/sglang/srt/utils/npu_pd_diagnostics.py"
spec = importlib.util.spec_from_file_location("pd_diag_format", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class Journal:
    """Four 64 MiB segments; rotation only touches collector-owned files."""
    def __init__(self, directory, limit=64 * 1024 * 1024):
        self.directory, self.limit = directory, limit
        self.path = directory / "history-0.jsonl"
        self.file = self.path.open("a", encoding="utf-8")
        self.size = self.path.stat().st_size

    def append(self, data):
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        size = len(line.encode("utf-8"))
        if self.size + size > self.limit:
            self.file.close()
            (self.directory / "history-3.jsonl").unlink(missing_ok=True)
            for i in (2, 1, 0):
                src = self.directory / f"history-{i}.jsonl"
                if src.exists():
                    src.replace(self.directory / f"history-{i + 1}.jsonl")
            self.file = self.path.open("a", encoding="utf-8")
            self.size = 0
        self.file.write(line)
        self.size += size

    def flush(self):
        self.file.flush()


class Worker:
    def __init__(self, path):
        self.path = path
        self.manifest = json.loads(path.with_suffix(".json").read_text())
        if self.manifest["version"] != core.VERSION or path.stat().st_size != core.SIZE:
            raise ValueError("unsupported recorder format")
        self.file = path.open("rb")
        self.buf = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.cursors = [0] * core.WRITERS
        self.lost = 0
        self.torn = 0
        self.progress = OrderedDict()
        self.terminals = OrderedDict()
        self.ready = False
        self.last_snapshot = None

    def same_process(self):
        pid = self.manifest["pid"]
        current = core.process_identity(pid)
        expected = self.manifest["identity"]["starttime"]
        host = os.uname().nodename if hasattr(os, "uname") else "unknown"
        return (host == self.manifest["hostname"] and expected != "unknown"
                and current["starttime"] == expected)

    def read(self):
        records, fragments, calls, events, jobs = [], [], [], [], []
        drops = core.U64.unpack_from(self.buf, 24)[0]
        for i in range(core.WRITERS):
            base = core.HEADER + i * core.STRIDE * core.SLOT
            head = core.read_slot(self.buf, base)
            if head and head.get("torn"):
                self.torn += 1
                continue
            if not head:
                continue
            sequence = head["seq"]
            drops += max(0, head["status"])
            if sequence - self.cursors[i] > core.RING:
                self.lost += sequence - self.cursors[i] - core.RING
            begin = max(self.cursors[i] + 1, sequence - core.RING + 1)
            for seq in range(begin, sequence + 1):
                entry = core.read_slot(self.buf, base + (1 + (seq - 1) % core.RING) * core.SLOT)
                if not entry or entry.get("torn") or entry["seq"] != seq:
                    self.torn += 1
                    continue
                entry["writer"] = i
                entry["tid"] = head["slot"]
                records.append(entry)
                key = (i, entry["batch"])
                self.progress[key] = max(entry["ts"], self.progress.get(key, 0))
                self.progress.move_to_end(key)
                if entry["kind"] == "WORKER_READY":
                    self.ready = True
                if entry["kind"] in ("PD_TERMINAL", "PD_CLEARED"):
                    self.terminals[(entry["room"], entry["attempt"])] = entry["ts"]
            self.cursors[i] = sequence
            for n in range(core.REQUESTS + core.CALLS):
                entry = core.read_slot(self.buf, base + (1 + core.RING + n) * core.SLOT)
                if not entry:
                    continue
                if entry.get("torn"):
                    self.torn += 1
                    continue
                entry["writer"], entry["tid"] = i, head["slot"]
                if n < core.REQUESTS:
                    fragments.append(entry)
                    if entry["kind"] in ("PD_TERMINAL", "PD_CLEARED"):
                        self.terminals[(entry["room"], entry["attempt"])] = entry["ts"]
                elif entry["kind"].endswith("_ENTER"):
                    calls.append(entry)
        for n in range(core.EVENTS):
            entry = core.read_slot(self.buf, core.EVENT_BASE + n * core.SLOT)
            if entry and not entry.get("torn") and entry["kind"] != "EVENT_DONE":
                events.append(entry)
        for n in range(core.JOBS):
            entry = core.read_slot(self.buf, core.JOB_BASE + n * core.SLOT)
            if entry and not entry.get("torn") and entry["kind"] in ("JOB_QUEUED", "JOB_RUNNING"):
                jobs.append(entry)
        while len(self.progress) > 8192:
            self.progress.popitem(last=False)
        while len(self.terminals) > 16384:
            self.terminals.popitem(last=False)
            drops += 1
        pending = {}
        for entry in fragments:
            key = (entry["room"], entry["attempt"])
            if key in self.terminals or entry["rid"].startswith("HEALTH_CHECK"):
                continue
            pending.setdefault(key, []).append(entry)
        requests = []
        for key, entries in pending.items():
            newest = max(entries, key=lambda r: r["ts"])
            requests.append(dict(room=key[0], attempt=key[1], rid=newest["rid"],
                                 phase=newest["kind"], reason=newest["reason"],
                                 enter=min(r["enter"] for r in entries),
                                 progress=max(r["progress"] for r in entries),
                                 fragments=entries))
        # Active table is independent of ring history; ready bit is also durable.
        self.ready |= bool(core.U64.unpack_from(self.buf, 32)[0])
        current = core.process_identity(self.manifest["pid"])
        event_flags = core.U64.unpack_from(self.buf, 40)[0]
        observer_ns = core.U64.unpack_from(self.buf, 48)[0]
        observer_stale = bool(event_flags & 1) and time.monotonic_ns() - observer_ns > 2_000_000_000
        identity_known = self.manifest["identity"]["starttime"] != "unknown"
        alive = self.same_process() and current["state"] != "Z"
        snapshot = dict(manifest=self.manifest, ready=self.ready,
                        alive=alive if identity_known else None, process=current,
                        coverage=dict(registration_or_table_drops=drops, ring_lost=self.lost,
                                      torn_reads=self.torn, mf_internal="not instrumented",
                                      event_flags=event_flags, observer_stale=observer_stale,
                                      peer_evidence="only sources supplied to this collector"),
                        requests=requests, calls=calls, events=events, jobs=jobs)
        snapshot["state"] = (
            "UNKNOWN" if not identity_known else
            "PROCESS_GONE" if not alive else
            "INITIALIZING" if not self.ready else
            "EXECUTING_AND_PD_WAITING" if requests and (calls or events or jobs) else
            "PD_WAITING" if requests else
            "EXECUTING" if calls or events or jobs else
            "UNKNOWN" if drops or self.torn or self.lost or observer_stale or event_flags & 2 else
            "NO_ACTIVE_WORK_OBSERVED"
        )
        self.last_snapshot = snapshot
        return records, snapshot

    def triggers(self, records, snapshot, now, snapshot_sec, stack_sec, wait_sec):
        triggers = []
        if snapshot["alive"] is False:
            triggers.append(dict(kind="PROCESS_GONE", key="process", age=0))
        for entry in records:
            if entry["kind"] in ("SCHEDULER_EXCEPTION", "WATCHDOG") or (
                    entry["kind"].endswith("_EXCEPTION") and not entry["kind"].startswith("EVENT")):
                triggers.append(dict(kind="EXCEPTION", key=entry["call"], evidence=entry, age=0))
        if not snapshot["ready"]:
            return triggers
        for entry in snapshot["calls"]:
            since = entry["enter"]
            # Per-writer + batch progress, not unrelated requests or observer ticks.
            if entry["kind"] in ("RUN_BATCH_ENTER", "WORKER_FORWARD_ENTER", "MODEL_ENTER"):
                since = max(since, self.progress.get((entry["writer"], entry["batch"]), 0))
            age = (now - since) / 1e9
            if age >= snapshot_sec:
                triggers.append(dict(kind="EXEC_STALL", key=entry["call"], age=age,
                                     stack=age >= stack_sec, evidence=entry))
        for entry in snapshot["events"]:
            if entry["reason"] == "EVENT_WARM":
                continue
            since = entry["progress"] if entry["kind"] == "EVENT_QUERY_ENTER" else entry["enter"]
            age = (now - since) / 1e9
            if age >= snapshot_sec:
                triggers.append(dict(kind="DEVICE_OBSERVATION_STALL", key=entry["call"], age=age,
                                     stack=age >= stack_sec, evidence=entry))
        for entry in snapshot["requests"]:
            age = (now - entry["progress"]) / 1e9
            if age >= wait_sec:
                triggers.append(dict(kind="PD_WAIT_LONG", key=[entry["room"], entry["attempt"]],
                                     age=age, stack=False, evidence=entry))
        for entry in snapshot["jobs"]:
            age = (now - entry["progress"]) / 1e9
            if age >= snapshot_sec:
                triggers.append(dict(kind="JOB_STALL", key=entry["call"], age=age,
                                     stack=age >= stack_sec, evidence=entry))
        return triggers


def tail(path, limit):
    try:
        with Path(path).open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError as exc:
        return f"UNAVAILABLE: {exc}"


def proc_snapshot(worker):
    if not worker.same_process():
        return {"error": "PID identity unavailable/reused/different namespace; no attach"}
    pid = worker.manifest["pid"]
    result = {name: tail(f"/proc/{pid}/{name}", 64 * 1024)
              for name in ("status", "stat", "limits", "cgroup", "sched", "wchan")}
    tasks = {}
    try:
        for task in list(Path(f"/proc/{pid}/task").iterdir())[:256]:
            tasks[task.name] = {name: tail(task / name, 4096)
                                for name in ("comm", "wchan", "status", "schedstat")}
    except OSError as exc:
        result["tasks_error"] = str(exc)
    result["tasks"] = tasks
    return result


def collect_stacks(workers, directory, budget=20):
    executable = shutil.which("py-spy")
    deadline = time.monotonic() + budget
    outcome = []
    if executable is None:
        return [{"error": "py-spy unavailable; CPU/mmap/proc evidence already saved"}]
    for native in (False, True):
        for worker in workers:
            left = deadline - time.monotonic()
            if left <= 0:
                return outcome + [{"error": "20-second total stack budget exhausted"}]
            if not worker.same_process():
                outcome.append({"pid": worker.manifest["pid"], "error": "identity mismatch; skipped"})
                continue
            pid = worker.manifest["pid"]
            command = [executable, "dump", "--pid", str(pid)] + (["--native"] if native else [])
            name = f"stack-{pid}-{'native' if native else 'python'}.txt"
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                def drain(pipe=process.stdout, path=directory / name):
                    remaining = 128 * 1024
                    with path.open("wb") as output:
                        while chunk := pipe.read(65536):
                            output.write(chunk[:remaining])
                            remaining = max(0, remaining - len(chunk))
                    pipe.close()
                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                try:
                    returncode = process.wait(timeout=min(5, left))
                    outcome.append(dict(pid=pid, native=native, returncode=returncode))
                except subprocess.TimeoutExpired:
                    process.kill()  # Only the collector's py-spy child, never SGLang.
                    process.wait(timeout=1)
                    outcome.append(dict(pid=pid, native=native, error="attach deadline exceeded"))
                reader.join(timeout=min(0.1, max(0, deadline - time.monotonic())))
            except (OSError, subprocess.TimeoutExpired) as exc:
                outcome.append(dict(pid=pid, native=native, error=str(exc)))
    return outcome


def save_incident(output, workers, triggers, journal, stacks, logs):
    # Two owned slots, always save evidence before invoking an external tool.
    slots = [output / f"incident-{i}" for i in range(2)]
    index = next((i for i, p in enumerate(slots) if not p.exists()), None)
    if index is None:
        candidates = list(range(2))
        # Keep the most recent stack evidence while a continuing stall emits
        # further waiting snapshots. It must not rotate away after 30 seconds.
        with_stacks = [i for i in candidates if (slots[i] / "stack-result.json").exists()]
        if with_stacks and not stacks:
            protected = max(with_stacks, key=lambda i: slots[i].stat().st_mtime_ns)
            candidates.remove(protected)
        index = min(candidates, key=lambda i: slots[i].stat().st_mtime_ns)
    directory = slots[index]
    if directory.exists():
        # Only this tool's flat artifact files; never recursively remove a path.
        for p in directory.iterdir():
            if p.is_file() and not p.is_symlink():
                p.unlink()
    directory.mkdir(exist_ok=True)
    journal.flush()
    report = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), triggers=triggers,
                  workers=[w.last_snapshot for w in workers],
                  interpretation="PD_WAIT_LONG is evidence of waiting, not proof of a deadlock. "
                                 "CPU return/clear/ABORT_ACK do not prove NPU or native drain.",
                  missing_peer="Supply both local P/D dirs; remote peer data is not fetched automatically.")
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
        report["artifact_coverage"] = "UNKNOWN: JSON detail exceeded 16 MiB; inspect retained raw maps/history"
        report["trigger_count_before_truncation"] = len(triggers)
        report["triggers"] = [{k: v for k, v in t.items() if k != "evidence"} for t in triggers[:2048]]
        report["workers"] = [{k: v for k, v in w.items() if k not in ("calls", "requests", "events", "jobs")}
                             for w in report["workers"]]
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
    (directory / "report.json").write_text(encoded)
    # Copy a recent bounded journal tail; old rings can wrap before the next incident.
    (directory / "recent-history.jsonl").write_text(tail(journal.path, 16 * 1024 * 1024))
    remaining = 160 * 1024 * 1024
    for worker in workers:
        prefix = worker.path.stem
        data = json.dumps(proc_snapshot(worker), ensure_ascii=False, indent=2)
        if len(data.encode("utf-8")) > 256 * 1024:
            data = json.dumps({"coverage": "truncated proc snapshot", "text": data[:128 * 1024]})
        (directory / f"proc-{prefix}.json").write_text(data)
        if remaining >= core.SIZE:
            (directory / f"{prefix}.mmap").write_bytes(worker.buf[:])
            remaining -= core.SIZE
    for i, log in enumerate(logs[:8]):
        (directory / f"log-{i}.txt").write_text(tail(log, 1024 * 1024))
    if stacks:
        result = collect_stacks(workers, directory)
        (directory / "stack-result.json").write_text(json.dumps(result, indent=2))
    print(f"{report['time']} saved {directory} ({', '.join(sorted({t['kind'] for t in triggers}))})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], help="recorder directory; repeat for P/D")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot-after", type=float, default=5)
    parser.add_argument("--stack-after", type=float, default=10)
    parser.add_argument("--pd-wait-after", type=float, default=30)
    parser.add_argument("--no-stack", action="store_true")
    parser.add_argument("--log-file", action="append", default=[])
    parser.add_argument("--once", action="store_true", help="read and save one snapshot, without attach")
    parser.add_argument("--report", type=Path, help="print an already saved report.json (offline)")
    args = parser.parse_args()
    if args.report:
        data = json.loads(args.report.read_text())
        print(data["time"])
        for trigger in data["triggers"]:
            print(trigger["kind"], "age_s=", round(trigger.get("age", 0), 2), "key=", trigger.get("key"))
        for worker in data["workers"]:
            print(worker["manifest"]["role"], worker["manifest"]["rank"], worker["state"], worker["coverage"])
        print(data["interpretation"])
        return
    if not args.source or args.output is None:
        parser.error("--source and --output required unless using --report")
    if not (0 < args.snapshot_after <= args.stack_after and args.pd_wait_after > 0):
        parser.error("require 0 < snapshot-after <= stack-after and pd-wait-after > 0")
    args.output.mkdir(parents=True, exist_ok=True)
    # One collector per output directory. Crash-stale lock is kept for explicit review.
    lock_path = args.output / "collector.lock"
    with lock_path.open("x") as lock:
        lock.write(str(os.getpid()))
    journal = Journal(args.output)
    workers, seen, notices = {}, {}, {}
    stack_collected = False
    last_flush = last_wait_snapshot = time.monotonic()
    try:
        while True:
            notice_triggers = []
            for source in args.source:
                for path in Path(source).glob("*.mmap"):
                    if str(path) in workers or not path.with_suffix(".json").exists():
                        continue
                    if len(workers) >= 64:
                        raise RuntimeError("64 recorder files exceeded; use a fresh run directory")
                    workers[str(path)] = Worker(path)
                for path in list(Path(source).glob("notice-*.json"))[:128]:
                    try:
                        stamp = path.stat().st_mtime_ns
                        if notices.get(str(path)) != stamp:
                            note = json.loads(path.read_text())
                            notice_triggers.append(dict(kind=note["kind"], key=note["pid"],
                                                        worker=str(path), age=0, evidence=note))
                            notices[str(path)] = stamp
                    except (OSError, ValueError, KeyError):
                        pass  # Partial crash-note publication: retry next tick.
            triggers, fresh = [], []
            triggers.extend(notice_triggers)
            now = time.monotonic_ns()
            for path, worker in workers.items():
                records, snapshot = worker.read()
                for record in records:
                    journal.append(dict(worker=path, event=record))
                journal.append(dict(worker=path, observation=snapshot))
                for t in worker.triggers(records, snapshot, now, args.snapshot_after,
                                         args.stack_after, args.pd_wait_after):
                    t["worker"] = path
                    triggers.append(t)
            need_stack = False
            active_keys = set()
            for t in triggers:
                key = (t["worker"], t["kind"], str(t["key"]))
                active_keys.add(key)
                level = 2 if t.get("stack") else 1
                if t["kind"] == "PD_WAIT_LONG":
                    if time.monotonic() - last_wait_snapshot >= args.pd_wait_after:
                        fresh.append(t)
                elif level > seen.get(key, 0):
                    fresh.append(t)
                    need_stack |= level == 2
                seen[key] = max(level, seen.get(key, 0))
            seen = {key: value for key, value in seen.items() if key in active_keys}
            execution_stalled = any(t["kind"] in ("EXEC_STALL", "DEVICE_OBSERVATION_STALL", "JOB_STALL")
                                    for t in triggers)
            if not execution_stalled:
                stack_collected = False
            if fresh or args.once:
                take_stack = need_stack and not stack_collected and not args.no_stack and not args.once
                save_incident(args.output, list(workers.values()), triggers or [{"kind": "MANUAL"}],
                              journal, take_stack, args.log_file)
                stack_collected |= take_stack
                if any(t["kind"] == "PD_WAIT_LONG" for t in fresh):
                    last_wait_snapshot = time.monotonic()
            if args.once:
                break
            if time.monotonic() - last_flush >= 5:
                journal.flush()
                last_flush = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        journal.flush()
        journal.file.close()
        for worker in workers.values():
            worker.buf.close()
            worker.file.close()
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
