#!/usr/bin/env python3
"""Refine per-scheduler thread affinity inside existing worker CPUs.

This script is designed to run after sglang scheduler processes already have
their coarse worker affinity. For each detected scheduler process, it:

1. Reads the scheduler's current worker CPU set.
2. Splits that worker set into topology-aware local "clusters" using
   `/sys/devices/system/cpu/cpu*/topology/cluster_id` when available.
3. Reserves one local cluster for hot threads.
4. Moves the main scheduler thread, ACL/runtime helper threads, and a small
   number of hottest scheduler threads onto the reserved cluster.
5. Moves all remaining threads of that scheduler onto the rest of the worker
   CPUs, so cold/helper threads no longer contend with hot threads.

This improves isolation inside one scheduler process, but by itself does not
prevent unrelated external processes from running on the same CPUs. That still
requires cgroup/cpuset or a separate background-process isolation mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

import psutil


DEFAULT_BACKUP_PATH = "/tmp/sglang_scheduler_thread_affinity_backup.json"
DEFAULT_SCHEDULER_KEYWORDS = (
    "sglang::scheduler",
    "scheduler",
)
DEFAULT_ALWAYS_HOT_PATTERNS = (
    r"^acl_thread$",
    r"^release_thread$",
    r"^RT_RECYCLE(?:_.+)?$",
)
DEFAULT_SCORING_PATTERNS = (
    r"^sglang::schedul",
)
DEFAULT_CLUSTER_POSITION = "last"
DEFAULT_RESERVE_CLUSTERS = 1
DEFAULT_FALLBACK_CLUSTER_SIZE = 4
DEFAULT_SAMPLE_SECONDS = 1.0
DEFAULT_TOP_K = 2


@dataclass
class ThreadStats:
    sum_exec_runtime: float = 0.0
    nr_switches: int = 0
    nr_voluntary_switches: int = 0
    nr_involuntary_switches: int = 0


@dataclass
class ThreadInfo:
    tid: int
    name: str
    affinity: List[int]
    stats_before: ThreadStats
    stats_after: ThreadStats

    @property
    def runtime_delta(self) -> float:
        return self.stats_after.sum_exec_runtime - self.stats_before.sum_exec_runtime

    @property
    def switch_delta(self) -> int:
        return self.stats_after.nr_switches - self.stats_before.nr_switches

    @property
    def voluntary_switch_delta(self) -> int:
        return (
            self.stats_after.nr_voluntary_switches
            - self.stats_before.nr_voluntary_switches
        )

    @property
    def involuntary_switch_delta(self) -> int:
        return (
            self.stats_after.nr_involuntary_switches
            - self.stats_before.nr_involuntary_switches
        )


def parse_cpu_spec(spec: str) -> List[int]:
    cpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid cpu range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpu_spec(cpus: Sequence[int]) -> str:
    if not cpus:
        return ""
    cpus = sorted(set(cpus))
    start = prev = cpus[0]
    parts: List[str] = []
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def compile_patterns(patterns: Sequence[str]) -> List[Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def matches_any(name: str, patterns: Sequence[Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def get_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return ""


def get_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return ""


def detect_scheduler_pids(
    keywords: Sequence[str], explicit_pids: Sequence[int]
) -> List[int]:
    if explicit_pids:
        return sorted(set(explicit_pids))

    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        haystack = f"{get_name(proc)} {get_cmdline(proc)}"
        if any(keyword in haystack for keyword in keywords):
            matched.append(proc.pid)
    return sorted(set(matched))


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_thread_name(pid: int, tid: int) -> str:
    comm_path = Path(f"/proc/{pid}/task/{tid}/comm")
    text = safe_read_text(comm_path).strip()
    return text or f"tid-{tid}"


def read_sched_stats(pid: int, tid: int) -> ThreadStats:
    text = safe_read_text(Path(f"/proc/{pid}/task/{tid}/sched"))
    stats = ThreadStats()
    if not text:
        return stats

    float_fields = {
        "se.sum_exec_runtime": "sum_exec_runtime",
    }
    int_fields = {
        "nr_switches": "nr_switches",
        "nr_voluntary_switches": "nr_voluntary_switches",
        "nr_involuntary_switches": "nr_involuntary_switches",
    }

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().split()[0]
        if key in float_fields:
            try:
                setattr(stats, float_fields[key], float(value))
            except ValueError:
                pass
        elif key in int_fields:
            try:
                setattr(stats, int_fields[key], int(value))
            except ValueError:
                pass
    return stats


def get_thread_affinity(tid: int) -> List[int]:
    return sorted(os.sched_getaffinity(tid))


def list_thread_ids(pid: int) -> List[int]:
    task_dir = Path(f"/proc/{pid}/task")
    tids = []
    try:
        for entry in task_dir.iterdir():
            if entry.name.isdigit():
                tids.append(int(entry.name))
    except OSError:
        return []
    return sorted(tids)


def collect_thread_infos(pid: int) -> Dict[int, ThreadInfo]:
    infos: Dict[int, ThreadInfo] = {}
    for tid in list_thread_ids(pid):
        try:
            affinity = get_thread_affinity(tid)
        except OSError:
            continue
        infos[tid] = ThreadInfo(
            tid=tid,
            name=read_thread_name(pid, tid),
            affinity=affinity,
            stats_before=read_sched_stats(pid, tid),
            stats_after=ThreadStats(),
        )
    return infos


def refresh_thread_infos(pid: int, infos: Dict[int, ThreadInfo]) -> Dict[int, ThreadInfo]:
    refreshed: Dict[int, ThreadInfo] = {}
    for tid in list_thread_ids(pid):
        base = infos.get(tid)
        if base is None:
            try:
                affinity = get_thread_affinity(tid)
            except OSError:
                continue
            base = ThreadInfo(
                tid=tid,
                name=read_thread_name(pid, tid),
                affinity=affinity,
                stats_before=ThreadStats(),
                stats_after=ThreadStats(),
            )
        base.stats_after = read_sched_stats(pid, tid)
        refreshed[tid] = base
    return refreshed


def chunk_cpus(cpus: Sequence[int], chunk_size: int) -> List[List[int]]:
    ordered = sorted(set(cpus))
    if chunk_size <= 0:
        return [ordered]
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def read_cluster_id(cpu: int) -> Optional[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/cluster_id")
    text = safe_read_text(path).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def build_local_clusters(
    worker_cpus: Sequence[int], fallback_cluster_size: int
) -> List[List[int]]:
    by_cluster: Dict[int, List[int]] = {}
    cluster_supported = False
    for cpu in sorted(set(worker_cpus)):
        cluster_id = read_cluster_id(cpu)
        if cluster_id is None:
            continue
        cluster_supported = True
        by_cluster.setdefault(cluster_id, []).append(cpu)

    if cluster_supported and by_cluster:
        clusters = [sorted(cpus) for _, cpus in sorted(by_cluster.items())]
        covered = sorted({cpu for cpus in clusters for cpu in cpus})
        if covered == sorted(set(worker_cpus)):
            return clusters

    return chunk_cpus(worker_cpus, fallback_cluster_size)


def split_hot_and_normal_cpus(
    worker_cpus: Sequence[int],
    local_clusters: Sequence[Sequence[int]],
    reserve_clusters: int,
    cluster_position: str,
) -> Tuple[List[int], List[int]]:
    if not worker_cpus:
        return [], []

    if local_clusters and reserve_clusters > 0 and len(local_clusters) > reserve_clusters:
        if cluster_position == "first":
            hot_clusters = local_clusters[:reserve_clusters]
            normal_clusters = local_clusters[reserve_clusters:]
        else:
            hot_clusters = local_clusters[-reserve_clusters:]
            normal_clusters = local_clusters[:-reserve_clusters]
        hot_cpus = sorted({cpu for cluster in hot_clusters for cpu in cluster})
        normal_cpus = sorted({cpu for cluster in normal_clusters for cpu in cluster})
        return hot_cpus, normal_cpus

    if len(worker_cpus) <= 1:
        return sorted(set(worker_cpus)), sorted(set(worker_cpus))

    split = max(1, len(worker_cpus) // 2)
    ordered = sorted(set(worker_cpus))
    if cluster_position == "first":
        hot_cpus = ordered[:split]
        normal_cpus = ordered[split:]
    else:
        hot_cpus = ordered[-split:]
        normal_cpus = ordered[:-split]
    if not normal_cpus:
        normal_cpus = hot_cpus
    return hot_cpus, normal_cpus


def classify_hot_threads(
    pid: int,
    thread_infos: Dict[int, ThreadInfo],
    always_hot_patterns: Sequence[Pattern[str]],
    scoring_patterns: Sequence[Pattern[str]],
    top_k: int,
) -> Tuple[List[int], List[ThreadInfo]]:
    hot_tids = {pid}
    scored_candidates: List[ThreadInfo] = []

    for info in thread_infos.values():
        if info.tid == pid:
            hot_tids.add(info.tid)
            continue
        if matches_any(info.name, always_hot_patterns):
            hot_tids.add(info.tid)
            continue
        if matches_any(info.name, scoring_patterns):
            scored_candidates.append(info)

    scored_candidates.sort(
        key=lambda item: (
            item.runtime_delta,
            item.switch_delta,
            item.voluntary_switch_delta,
            -item.tid,
        ),
        reverse=True,
    )
    for info in scored_candidates[: max(0, top_k)]:
        hot_tids.add(info.tid)

    return sorted(hot_tids), scored_candidates


def build_backup_record(
    pid: int,
    thread_infos: Dict[int, ThreadInfo],
    hot_tids: Sequence[int],
    hot_cpus: Sequence[int],
    normal_cpus: Sequence[int],
) -> dict:
    records = []
    hot_tid_set = set(hot_tids)
    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        records.append(
            {
                "tid": info.tid,
                "name": info.name,
                "original_affinity": list(info.affinity),
                "target_group": "hot" if info.tid in hot_tid_set else "normal",
            }
        )
    return {
        "pid": pid,
        "hot_cpus": list(hot_cpus),
        "normal_cpus": list(normal_cpus),
        "threads": records,
    }


def apply_thread_affinity(
    pid: int,
    thread_infos: Dict[int, ThreadInfo],
    hot_tids: Sequence[int],
    hot_cpus: Sequence[int],
    normal_cpus: Sequence[int],
    *,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int]:
    changed = 0
    failed = 0
    hot_tid_set = set(hot_tids)
    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        target_cpus = hot_cpus if info.tid in hot_tid_set else normal_cpus
        if sorted(info.affinity) == sorted(target_cpus):
            if verbose:
                print(
                    f"Keep pid={pid} tid={info.tid} name={info.name} "
                    f"already on {format_cpu_spec(target_cpus)}"
                )
            continue

        action = "Plan" if dry_run else "Move"
        group = "hot" if info.tid in hot_tid_set else "normal"
        print(
            f"{action} pid={pid} tid={info.tid} name={info.name} "
            f"group={group} {format_cpu_spec(info.affinity)} -> "
            f"{format_cpu_spec(target_cpus)}"
        )
        changed += 1
        if dry_run:
            continue
        try:
            os.sched_setaffinity(info.tid, set(target_cpus))
        except OSError as exc:
            failed += 1
            print(
                f"Failed to move pid={pid} tid={info.tid} name={info.name}: {exc}",
                file=sys.stderr,
            )
    return changed, failed


def restore_affinity(backup_path: str) -> int:
    path = Path(backup_path)
    if not path.exists():
        print(f"Backup file not found: {backup_path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    restored = 0
    skipped = 0
    for process in data.get("processes", []):
        for thread in process.get("threads", []):
            try:
                os.sched_setaffinity(thread["tid"], set(thread["original_affinity"]))
                restored += 1
            except OSError as exc:
                skipped += 1
                print(
                    f"Skip restore tid={thread['tid']} name={thread['name']}: {exc}",
                    file=sys.stderr,
                )

    print(
        f"Restore done: restored={restored}, skipped={skipped}, backup={backup_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine thread affinity inside each detected sglang scheduler worker. "
            "Hot threads go to one reserved local cluster; all other threads stay "
            "on the rest of that worker's CPUs."
        )
    )
    parser.add_argument(
        "--scheduler-pid",
        type=int,
        action="append",
        default=[],
        help="Explicit scheduler pid. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scheduler-keyword",
        action="append",
        default=[],
        help=(
            "Extra keyword for auto-detecting scheduler processes from "
            "process name/cmdline."
        ),
    )
    parser.add_argument(
        "--always-hot-pattern",
        action="append",
        default=list(DEFAULT_ALWAYS_HOT_PATTERNS),
        help=(
            "Regex for thread names that must stay on hot CPUs. "
            f"Default: {', '.join(DEFAULT_ALWAYS_HOT_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--scoring-pattern",
        action="append",
        default=list(DEFAULT_SCORING_PATTERNS),
        help=(
            "Regex for thread names that compete for the remaining hot slots. "
            f"Default: {', '.join(DEFAULT_SCORING_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--top-k-hot-threads",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "From threads matching --scoring-pattern, keep the hottest top-k on "
            f"hot CPUs. Default: {DEFAULT_TOP_K}"
        ),
    )
    parser.add_argument(
        "--reserve-clusters",
        type=int,
        default=DEFAULT_RESERVE_CLUSTERS,
        help=(
            "How many local clusters to reserve for hot threads inside one "
            f"scheduler worker. Default: {DEFAULT_RESERVE_CLUSTERS}"
        ),
    )
    parser.add_argument(
        "--cluster-position",
        choices=("first", "last"),
        default=DEFAULT_CLUSTER_POSITION,
        help=(
            "Which side of the worker CPU set should be reserved as hot cluster "
            f"when splitting local clusters. Default: {DEFAULT_CLUSTER_POSITION}"
        ),
    )
    parser.add_argument(
        "--fallback-cluster-size",
        type=int,
        default=DEFAULT_FALLBACK_CLUSTER_SIZE,
        help=(
            "When cluster_id is unavailable, split worker CPUs into contiguous "
            f"chunks of this size. Default: {DEFAULT_FALLBACK_CLUSTER_SIZE}"
        ),
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=DEFAULT_SAMPLE_SECONDS,
        help=(
            "Sampling interval used to estimate hot threads from /proc/<pid>/task/"
            f"<tid>/sched deltas. Default: {DEFAULT_SAMPLE_SECONDS}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without changing affinity.",
    )
    parser.add_argument(
        "--backup-path",
        type=str,
        default=DEFAULT_BACKUP_PATH,
        help=f"Backup file path. Default: {DEFAULT_BACKUP_PATH}",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore affinities from --backup-path and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-thread hotness ranking and decisions.",
    )
    return parser


def ensure_supported_platform() -> None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("This script requires Linux sched affinity support.")


def main() -> int:
    ensure_supported_platform()
    args = build_parser().parse_args()

    if args.restore:
        return restore_affinity(args.backup_path)

    keywords = list(DEFAULT_SCHEDULER_KEYWORDS) + list(args.scheduler_keyword)
    scheduler_pids = detect_scheduler_pids(keywords, args.scheduler_pid)
    if not scheduler_pids:
        print(
            "Failed to detect scheduler pids automatically. Pass --scheduler-pid.",
            file=sys.stderr,
        )
        return 2

    always_hot_patterns = compile_patterns(args.always_hot_pattern)
    scoring_patterns = compile_patterns(args.scoring_pattern)

    backup_payload = {
        "created_at": int(time.time()),
        "processes": [],
    }
    total_changed = 0
    total_failed = 0

    print(f"Detected scheduler pids: {scheduler_pids}")

    for pid in scheduler_pids:
        try:
            proc = psutil.Process(pid)
            worker_cpus = sorted(proc.cpu_affinity())
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            print(f"Skip pid={pid}: failed to inspect scheduler affinity: {exc}")
            total_failed += 1
            continue

        local_clusters = build_local_clusters(
            worker_cpus, args.fallback_cluster_size
        )
        hot_cpus, normal_cpus = split_hot_and_normal_cpus(
            worker_cpus,
            local_clusters,
            args.reserve_clusters,
            args.cluster_position,
        )
        if not hot_cpus or not normal_cpus:
            print(
                f"Skip pid={pid}: failed to derive hot/normal CPU split from "
                f"worker CPUs {format_cpu_spec(worker_cpus)}",
                file=sys.stderr,
            )
            total_failed += 1
            continue

        before = collect_thread_infos(pid)
        if args.sample_seconds > 0:
            time.sleep(args.sample_seconds)
        thread_infos = refresh_thread_infos(pid, before)
        if not thread_infos:
            print(f"Skip pid={pid}: no live threads found.", file=sys.stderr)
            total_failed += 1
            continue

        hot_tids, scored_candidates = classify_hot_threads(
            pid,
            thread_infos,
            always_hot_patterns,
            scoring_patterns,
            args.top_k_hot_threads,
        )

        print(
            f"Scheduler pid={pid} worker={format_cpu_spec(worker_cpus)} "
            f"local_clusters={[format_cpu_spec(cluster) for cluster in local_clusters]} "
            f"hot={format_cpu_spec(hot_cpus)} normal={format_cpu_spec(normal_cpus)} "
            f"hot_threads={len(hot_tids)}/{len(thread_infos)}"
        )

        if args.verbose:
            print("Top scored thread candidates:")
            for info in scored_candidates[:10]:
                print(
                    f"  tid={info.tid} name={info.name} "
                    f"runtime_delta={info.runtime_delta:.3f} "
                    f"switch_delta={info.switch_delta} "
                    f"vol={info.voluntary_switch_delta} "
                    f"invol={info.involuntary_switch_delta}"
                )

        backup_payload["processes"].append(
            build_backup_record(pid, thread_infos, hot_tids, hot_cpus, normal_cpus)
        )

        changed, failed = apply_thread_affinity(
            pid,
            thread_infos,
            hot_tids,
            hot_cpus,
            normal_cpus,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_changed += changed
        total_failed += failed

    if args.dry_run:
        print(
            f"Dry-run done: would_change={total_changed}, failed={total_failed}, "
            f"backup_path={args.backup_path}"
        )
        return 0

    backup_path = Path(args.backup_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(json.dumps(backup_payload, indent=2), encoding="utf-8")
    print(
        f"Apply done: changed={total_changed}, failed={total_failed}, "
        f"backup={args.backup_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
