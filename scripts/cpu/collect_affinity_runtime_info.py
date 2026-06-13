#!/usr/bin/env python3
"""Collect runtime evidence for sglang/NPU affinity analysis.

This script is read-only. It does not change affinity, IRQ routing, or
workqueue masks. It snapshots the process/thread/IRQ/topology state into
one JSON file so that a later binding script can be designed with real
runtime evidence instead of assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import psutil


DEFAULT_OUTPUT_PATH = "/tmp/sglang_affinity_runtime_info.json"
DEFAULT_ROOT_KEYWORDS = (
    "sglang.launch_server",
    "python -m sglang.launch_server",
)
DEFAULT_EXTRA_PROCESS_KEYWORDS = (
    "sglang::scheduler",
    "sglang::detoken",
    "data_parallel_controller",
)
DEFAULT_THREAD_KEYWORDS = (
    "acl_thread",
    "release_thread",
    "RT_RECYCLE",
    "CaffeTaskThread",
    "hccp",
    "watchdog",
    "ZMQbg",
)
DEFAULT_INTERRUPT_PATTERNS = (
    r"\bsq\b",
    r"\bcq\b",
    r"trs",
    r"mbox",
    r"npu",
    r"ascend",
    r"hccp",
)
DEFAULT_WORKQUEUE_PATTERNS = (
    r"sq",
    r"cq",
    r"trs",
    r"mbox",
    r"npu",
    r"dev",
)


def parse_cpu_spec(spec: str) -> List[int]:
    cpus: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def safe_read_text(path: str, max_bytes: Optional[int] = None) -> Dict[str, object]:
    result: Dict[str, object] = {"path": path, "ok": False}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes)
        result["ok"] = True
        result["content"] = data
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def run_command(command: Sequence[str], timeout: int = 10) -> Dict[str, object]:
    result: Dict[str, object] = {
        "command": list(command),
        "ok": False,
    }
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        result.update(
            {
                "ok": True,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def get_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def get_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def match_processes_by_keywords(keywords: Sequence[str]) -> List[int]:
    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        haystack = f"{get_name(proc)} {get_cmdline(proc)}"
        if any(keyword in haystack for keyword in keywords):
            matched.append(proc.pid)
    return sorted(set(matched))


def get_descendant_pids(root_pids: Iterable[int]) -> Set[int]:
    pids: Set[int] = set()
    for root_pid in root_pids:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            continue
        pids.add(root.pid)
        try:
            children = root.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            children = []
        pids.update(child.pid for child in children)
    return pids


def get_thread_affinity(tid: int) -> Optional[List[int]]:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(tid))
    except Exception:
        return None


def get_thread_psr(tid: int) -> Optional[int]:
    stat_path = f"/proc/{os.getpid()}/task/{tid}/stat"
    # The thread may belong to another process; caller should pass a full path
    # reader when cross-process lookup is needed.
    try:
        with open(stat_path, "r", encoding="utf-8", errors="replace") as f:
            fields = f.read().split()
        if len(fields) >= 39:
            return int(fields[38])
    except Exception:
        return None
    return None


def get_task_psr(task_stat_path: str) -> Optional[int]:
    try:
        with open(task_stat_path, "r", encoding="utf-8", errors="replace") as f:
            fields = f.read().split()
        if len(fields) >= 39:
            return int(fields[38])
    except Exception:
        return None
    return None


def collect_threads_for_pid(pid: int) -> List[Dict[str, object]]:
    task_dir = Path(f"/proc/{pid}/task")
    threads: List[Dict[str, object]] = []
    if not task_dir.exists():
        return threads

    for entry in sorted(task_dir.iterdir(), key=lambda p: int(p.name)):
        if not entry.is_dir():
            continue
        tid = int(entry.name)
        comm = safe_read_text(str(entry / "comm"), max_bytes=512).get("content", "")
        status = safe_read_text(str(entry / "status"), max_bytes=16 * 1024)
        stat_path = str(entry / "stat")
        threads.append(
            {
                "tid": tid,
                "comm": str(comm).strip(),
                "psr": get_task_psr(stat_path),
                "affinity": get_thread_affinity(tid),
                "status_excerpt": status.get("content", ""),
            }
        )
    return threads


def collect_process_info(pid: int) -> Dict[str, object]:
    proc_info: Dict[str, object] = {"pid": pid}
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        proc_info["missing"] = True
        return proc_info

    proc_info.update(
        {
            "name": get_name(proc),
            "cmdline": get_cmdline(proc),
            "ppid": proc.ppid(),
            "status": str(proc.status()),
            "cpu_affinity": safe_call(lambda: proc.cpu_affinity()),
            "num_threads": safe_call(lambda: proc.num_threads()),
            "create_time": safe_call(lambda: proc.create_time()),
            "uids": safe_call(lambda: list(proc.uids())),
            "cwd": safe_call(lambda: proc.cwd()),
            "exe": safe_call(lambda: proc.exe()),
            "cgroup": safe_read_text(f"/proc/{pid}/cgroup").get("content", ""),
            "sched": safe_read_text(f"/proc/{pid}/sched", max_bytes=64 * 1024).get(
                "content", ""
            ),
            "schedstat": safe_read_text(f"/proc/{pid}/schedstat").get("content", ""),
            "status_file": safe_read_text(
                f"/proc/{pid}/status", max_bytes=64 * 1024
            ).get("content", ""),
            "numa_maps_head": safe_read_text(
                f"/proc/{pid}/numa_maps", max_bytes=64 * 1024
            ).get("content", ""),
            "threads": collect_threads_for_pid(pid),
        }
    )
    return proc_info


def safe_call(fn):
    try:
        return fn()
    except Exception:
        return None


def collect_process_tree_summary(root_pids: Sequence[int], process_pids: Sequence[int]) -> List[Dict[str, object]]:
    by_pid = {}
    for pid in process_pids:
        try:
            proc = psutil.Process(pid)
            by_pid[pid] = {
                "pid": pid,
                "ppid": proc.ppid(),
                "name": get_name(proc),
                "cmdline": get_cmdline(proc),
            }
        except psutil.NoSuchProcess:
            continue
    return [{"root_pid": pid, "descendants": [item for item in by_pid.values() if is_descendant_or_self(item["pid"], pid, by_pid)]} for pid in root_pids]


def is_descendant_or_self(pid: int, root_pid: int, by_pid: Dict[int, Dict[str, object]]) -> bool:
    current = pid
    visited = set()
    while current not in visited:
        visited.add(current)
        if current == root_pid:
            return True
        item = by_pid.get(current)
        if item is None:
            return False
        current = int(item["ppid"])
    return False


def compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    return [re.compile(pat, re.IGNORECASE) for pat in patterns]


def filter_lines(text: str, patterns: Sequence[str]) -> List[str]:
    regexes = compile_patterns(patterns)
    matched = []
    for line in text.splitlines():
        if any(regex.search(line) for regex in regexes):
            matched.append(line)
    return matched


def collect_workqueue_info(patterns: Sequence[str]) -> Dict[str, object]:
    regexes = compile_patterns(patterns)
    roots = [
        Path("/sys/devices/virtual/workqueue"),
        Path("/sys/bus/workqueue/devices"),
    ]
    matched_entries = []
    scanned = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            scanned.append(str(path))
            if not any(regex.search(str(path)) for regex in regexes):
                continue
            content = safe_read_text(str(path), max_bytes=32 * 1024)
            matched_entries.append(
                {
                    "path": str(path),
                    "content": content.get("content", ""),
                    "ok": content.get("ok", False),
                    "error": content.get("error"),
                }
            )
    return {
        "roots": [str(root) for root in roots if root.exists()],
        "matched_entries": matched_entries,
        "scanned_file_count": len(scanned),
    }


def find_thread_matches(process_infos: Sequence[Dict[str, object]], keywords: Sequence[str]) -> List[Dict[str, object]]:
    lower_keywords = [k.lower() for k in keywords]
    matches = []
    for proc in process_infos:
        pid = proc["pid"]
        for thread in proc.get("threads", []):
            comm = str(thread.get("comm", ""))
            if any(keyword in comm.lower() for keyword in lower_keywords):
                matches.append(
                    {
                        "pid": pid,
                        "process_name": proc.get("name"),
                        "tid": thread.get("tid"),
                        "comm": comm,
                        "psr": thread.get("psr"),
                        "affinity": thread.get("affinity"),
                    }
                )
    return matches


def discover_roots(args: argparse.Namespace) -> List[int]:
    if args.root_pid:
        return sorted(set(args.root_pid))
    roots = match_processes_by_keywords(DEFAULT_ROOT_KEYWORDS + tuple(args.root_keyword))
    if roots:
        return roots
    fallback_keywords = DEFAULT_ROOT_KEYWORDS + DEFAULT_EXTRA_PROCESS_KEYWORDS + tuple(
        args.root_keyword
    )
    return match_processes_by_keywords(fallback_keywords)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect runtime process/thread/IRQ/workqueue evidence for sglang affinity analysis."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--root-pid",
        type=int,
        action="append",
        default=[],
        help="Explicit sglang root pid. Can be passed multiple times.",
    )
    parser.add_argument(
        "--root-keyword",
        action="append",
        default=[],
        help="Extra process keyword used when auto-detecting sglang roots.",
    )
    parser.add_argument(
        "--thread-keyword",
        action="append",
        default=[],
        help="Extra thread keyword for highlighting relevant runtime threads.",
    )
    parser.add_argument(
        "--interrupt-pattern",
        action="append",
        default=[],
        help="Extra regex pattern for filtering /proc/interrupts lines.",
    )
    parser.add_argument(
        "--workqueue-pattern",
        action="append",
        default=[],
        help="Extra regex pattern for filtering workqueue/sysfs paths.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    root_pids = discover_roots(args)
    descendant_pids = get_descendant_pids(root_pids)
    extra_pids = match_processes_by_keywords(DEFAULT_EXTRA_PROCESS_KEYWORDS)
    process_pids = sorted(set(root_pids) | set(descendant_pids) | set(extra_pids))

    process_infos = [collect_process_info(pid) for pid in process_pids]
    thread_keywords = DEFAULT_THREAD_KEYWORDS + tuple(args.thread_keyword)
    interrupt_patterns = DEFAULT_INTERRUPT_PATTERNS + tuple(args.interrupt_pattern)
    workqueue_patterns = DEFAULT_WORKQUEUE_PATTERNS + tuple(args.workqueue_pattern)

    proc_interrupts = safe_read_text("/proc/interrupts", max_bytes=1024 * 1024)
    proc_softirqs = safe_read_text("/proc/softirqs", max_bytes=512 * 1024)
    cpu_online = safe_read_text("/sys/devices/system/cpu/online")
    present_cpus = safe_read_text("/sys/devices/system/cpu/present")

    result = {
        "meta": {
            "collected_at_epoch": int(time.time()),
            "collected_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": os.getcwd(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
        },
        "detected": {
            "root_pids": root_pids,
            "root_keywords": list(DEFAULT_ROOT_KEYWORDS + tuple(args.root_keyword)),
            "extra_process_keywords": list(DEFAULT_EXTRA_PROCESS_KEYWORDS),
            "process_pids": process_pids,
            "thread_keywords": list(thread_keywords),
            "interrupt_patterns": list(interrupt_patterns),
            "workqueue_patterns": list(workqueue_patterns),
        },
        "system": {
            "online_cpus": cpu_online.get("content", ""),
            "present_cpus": present_cpus.get("content", ""),
            "lscpu_json": run_command(["lscpu", "-e=CPU,CORE,SOCKET,NODE", "-J"]),
            "lscpu_summary": run_command(["lscpu"]),
            "numactl_H": run_command(["numactl", "-H"]),
            "npu_smi_info": run_command(["npu-smi", "info"]),
            "npu_smi_topo": run_command(["npu-smi", "info", "-t", "topo"]),
            "proc_interrupts": proc_interrupts.get("content", ""),
            "proc_softirqs": proc_softirqs.get("content", ""),
            "matched_interrupt_lines": filter_lines(
                str(proc_interrupts.get("content", "")), interrupt_patterns
            ),
            "workqueue_info": collect_workqueue_info(workqueue_patterns),
        },
        "sglang": {
            "process_tree_summary": collect_process_tree_summary(root_pids, process_pids),
            "processes": process_infos,
            "matched_threads": find_thread_matches(process_infos, thread_keywords),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved runtime affinity evidence to: {output_path}")
    print(f"Detected root pids: {root_pids}")
    print(f"Collected process count: {len(process_infos)}")
    print(f"Matched thread count: {len(result['sglang']['matched_threads'])}")
    print(f"Matched interrupt line count: {len(result['system']['matched_interrupt_lines'])}")
    print(
        f"Matched workqueue entry count: "
        f"{len(result['system']['workqueue_info']['matched_entries'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
