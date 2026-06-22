#!/usr/bin/env python3
"""Bind sglang scheduler processes with NUMA-local thread groups.

Design goals:

- One scheduler process never crosses NUMA.
- If cpus_per_proc <= 12:
  - logical ranks 0-3 use NUMA 0
  - logical ranks 4-7 use NUMA 2
- If 12 < cpus_per_proc <= 24:
  - logical ranks 0-1 use NUMA 0
  - logical ranks 2-3 use NUMA 1
  - logical ranks 4-5 use NUMA 2
  - logical ranks 6-7 use NUMA 3
- Scheduler-hot threads always stay inside one dedicated hot cluster.
- Binding order follows the old shell script strategy:
  1. set all threads in one scheduler process to scheduler-hot CPUs
  2. override runtime threads
  3. override cold/helper threads
- Optional control/background process isolation only uses CPUs that remain free
  after scheduler/runtime/cold planning.
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
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

import psutil


DEFAULT_BACKUP_PATH = "/tmp/sglang_scheduler_numa_groups_backup.json"
DEFAULT_SCHEDULER_KEYWORDS = (
    "sglang::scheduler",
)
DEFAULT_ROOT_KEYWORDS = (
    "sglang.launch_server",
    "python -m sglang.launch_server",
    "sglang::scheduler",
)
DEFAULT_CONTROL_ROOT_KEYWORDS = (
    "sglang.launch_server",
    "python -m sglang.launch_server",
)
DEFAULT_CONTROL_KEYWORDS = (
    "sglang::detokenizer",
    "sglang::detoken",
)
DEFAULT_MACHINE_UNIQUE_THREAD_PATTERNS = (
    r"^uvb_poll_window_thread$",
)
DEFAULT_RUNTIME_PATTERNS = (
    r"^acl_thread$",
    r"^release_thread$",
    r"^RT_RECYCLE(?:_.+)?$",
)
DEFAULT_SCHEDULER_THREAD_PATTERNS = (
    r"^sglang::schedul",
)
DEFAULT_COLD_PATTERNS = (
    r"^ZMQbg/",
    r"^CaffeTaskThread$",
    r"^WatchdogMonitor$",
    r"^AtraceMonitor$",
    r"^PlogFlush$",
    r"^hccl_watchdog_t$",
)
DEFAULT_CLUSTER_SIZE = 8
DEFAULT_CONTROL_CPUS_PER_HELPER_NUMA = 16
DEFAULT_EXCLUDED_NAMES = {
    "systemd",
    "systemd-journald",
    "systemd-logind",
    "sshd",
    "login",
    "agetty",
    "bash",
    "zsh",
    "fish",
    "tmux",
    "screen",
    "sudo",
    "su",
}
DEFAULT_GROUP_SIZE_PRESETS = {
    8: (8, 0, 0),
    12: (8, 3, 1),
    16: (8, 4, 4),
    18: (8, 6, 4),
    24: (8, 8, 8),
}


@dataclass
class ThreadInfo:
    tid: int
    name: str
    affinity: List[int]


@dataclass
class SchedulerPlan:
    logical_rank: int
    pid: int
    current_worker_cpus: List[int]
    current_numa_id: int
    target_numa_id: int
    hot_cluster_cpus: List[int]
    scheduler_cpus: List[int]
    runtime_cpus: List[int]
    cold_cpus: List[int]


@dataclass
class SpecialThreadMatch:
    pid: int
    tid: int
    process_name: str
    thread_name: str
    affinity: List[int]


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
    ordered = sorted(set(cpus))
    start = prev = ordered[0]
    parts: List[str] = []
    for cpu in ordered[1:]:
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


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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


def get_ancestor_pids(pid: int) -> Set[int]:
    ancestors = set()
    current = psutil.Process(pid)
    while True:
        ancestors.add(current.pid)
        parent = current.parent()
        if parent is None:
            break
        current = parent
    return ancestors


def get_descendant_pids(root_pids: Iterable[int]) -> Set[int]:
    all_pids: Set[int] = set()
    for root_pid in root_pids:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            continue
        all_pids.add(root.pid)
        try:
            children = root.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            children = []
        all_pids.update(child.pid for child in children)
    return all_pids


def auto_detect_sglang_roots(keywords: Sequence[str]) -> List[int]:
    roots = []
    for proc in psutil.process_iter(["pid", "name"]):
        haystack = f"{get_name(proc)} {get_cmdline(proc)}"
        if any(keyword in haystack for keyword in keywords):
            roots.append(proc.pid)
    return sorted(set(roots))


def detect_processes_by_keywords(keywords: Sequence[str]) -> List[int]:
    if not keywords:
        return []
    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        haystack = f"{get_name(proc)} {get_cmdline(proc)}"
        if any(keyword in haystack for keyword in keywords):
            matched.append(proc.pid)
    return sorted(set(matched))


def filter_child_pids(pids: Iterable[int], allowed_pids: Set[int]) -> List[int]:
    return sorted({pid for pid in pids if pid in allowed_pids})


def detect_scheduler_pids(
    keywords: Sequence[str], explicit_pids: Sequence[int]
) -> List[int]:
    if explicit_pids:
        return sorted(set(explicit_pids))

    protected_pids = get_ancestor_pids(os.getpid())
    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.pid in protected_pids:
            continue
        haystack = f"{get_name(proc)} {get_cmdline(proc)}"
        if any(keyword in haystack for keyword in keywords):
            matched.append(proc.pid)
    return sorted(set(matched))


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


def read_thread_name(pid: int, tid: int) -> str:
    text = safe_read_text(Path(f"/proc/{pid}/task/{tid}/comm")).strip()
    return text or f"tid-{tid}"


def get_thread_affinity(tid: int) -> List[int]:
    return sorted(os.sched_getaffinity(tid))


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
        )
    return infos


def read_cluster_id(cpu: int) -> Optional[int]:
    text = safe_read_text(
        Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/cluster_id")
    ).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def read_numa_to_cpus() -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    node_root = Path("/sys/devices/system/node")
    if not node_root.exists():
        raise RuntimeError("NUMA topology not available under /sys/devices/system/node")
    for node_dir in sorted(node_root.iterdir()):
        if not node_dir.name.startswith("node"):
            continue
        suffix = node_dir.name[4:]
        if not suffix.isdigit():
            continue
        cpulist = safe_read_text(node_dir / "cpulist").strip()
        if not cpulist:
            continue
        result[int(suffix)] = parse_cpu_spec(cpulist)
    if not result:
        raise RuntimeError("Failed to read NUMA cpulists")
    return result


def build_numa_clusters(
    numa_to_cpus: Dict[int, List[int]], fallback_cluster_size: int
) -> Dict[int, List[List[int]]]:
    result: Dict[int, List[List[int]]] = {}
    for numa_id, cpus in sorted(numa_to_cpus.items()):
        by_cluster: Dict[int, List[int]] = {}
        cluster_supported = False
        for cpu in cpus:
            cluster_id = read_cluster_id(cpu)
            if cluster_id is None:
                continue
            cluster_supported = True
            by_cluster.setdefault(cluster_id, []).append(cpu)

        if cluster_supported and by_cluster:
            clusters = [sorted(cluster) for _, cluster in sorted(by_cluster.items())]
            covered = sorted({cpu for cluster in clusters for cpu in cluster})
            if covered == sorted(cpus):
                result[numa_id] = clusters
                continue

        ordered = sorted(cpus)
        result[numa_id] = [
            ordered[i : i + fallback_cluster_size]
            for i in range(0, len(ordered), fallback_cluster_size)
        ]
    return result


def build_cpu_to_numa(numa_to_cpus: Dict[int, List[int]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for numa_id, cpus in numa_to_cpus.items():
        for cpu in cpus:
            mapping[cpu] = numa_id
    return mapping


def detect_process_numa(worker_cpus: Sequence[int], cpu_to_numa: Dict[int, int]) -> int:
    counts: Dict[int, int] = {}
    for cpu in worker_cpus:
        numa_id = cpu_to_numa.get(cpu)
        if numa_id is None:
            continue
        counts[numa_id] = counts.get(numa_id, 0) + 1
    if not counts:
        raise RuntimeError(
            f"Failed to map worker CPUs {format_cpu_spec(worker_cpus)} to a NUMA node"
        )
    return max(sorted(counts), key=lambda item: counts[item])


def is_kernel_like_process(proc: psutil.Process) -> bool:
    try:
        if proc.ppid() == 2:
            return True
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return True
    return get_cmdline(proc) == ""


def pick_background_candidates(
    foreground_pids: Set[int],
    protected_pids: Set[int],
    exclude_names: Set[str],
    same_uid_only: bool,
) -> List[psutil.Process]:
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        pid = proc.pid
        if pid in foreground_pids or pid in protected_pids:
            continue
        if pid in (0, 1, 2):
            continue
        if is_kernel_like_process(proc):
            continue
        if get_name(proc) in exclude_names:
            continue
        if same_uid_only and current_uid is not None:
            try:
                if proc.uids().real != current_uid:
                    continue
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        candidates.append(proc)
    return candidates


def build_affinity_record(proc: psutil.Process, original: Sequence[int]) -> dict:
    return {
        "pid": proc.pid,
        "name": get_name(proc),
        "cmdline": get_cmdline(proc),
        "original_affinity": list(original),
    }


def rebind_processes(
    target_pids: Iterable[int],
    target_cpus: Sequence[int],
    *,
    dry_run: bool,
    verbose: bool,
    label: str,
) -> Tuple[List[dict], int]:
    changed = []
    failed = 0
    for pid in sorted(set(target_pids)):
        try:
            proc = psutil.Process(pid)
            original = proc.cpu_affinity()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            failed += 1
            continue

        if sorted(original) == sorted(target_cpus):
            if verbose:
                print(
                    f"Keep {label} pid={proc.pid} name={get_name(proc)} "
                    f"already on {format_cpu_spec(target_cpus)}"
                )
            continue

        record = build_affinity_record(proc, original)
        changed.append(record)

        if verbose or dry_run:
            print(
                f"{'Plan' if dry_run else 'Move'} {label} pid={proc.pid} "
                f"name={record['name']} {format_cpu_spec(original)} -> "
                f"{format_cpu_spec(target_cpus)}"
            )

        if dry_run:
            continue

        try:
            proc.cpu_affinity(list(target_cpus))
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            failed += 1
            print(f"Failed to move {label} pid={proc.pid}: {exc}", file=sys.stderr)

    return changed, failed


def get_online_cpus() -> List[int]:
    online_path = Path("/sys/devices/system/cpu/online")
    if online_path.exists():
        return parse_cpu_spec(online_path.read_text().strip())
    count = os.cpu_count()
    if not count:
        raise RuntimeError("Failed to detect online CPUs")
    return list(range(count))


def derive_group_sizes(
    cpus_per_proc: int,
    cluster_size: int,
    scheduler_cpus: Optional[int],
    runtime_cpus: Optional[int],
    cold_cpus: Optional[int],
) -> Tuple[int, int, int]:
    if scheduler_cpus is None and runtime_cpus is None and cold_cpus is None:
        preset = DEFAULT_GROUP_SIZE_PRESETS.get(cpus_per_proc)
        if preset is None:
            raise RuntimeError(
                "Unsupported cpus_per_proc without explicit group sizes. "
                f"Supported presets: {sorted(DEFAULT_GROUP_SIZE_PRESETS)}"
            )
        scheduler_count, runtime_count, cold_count = preset
    else:
        if None in (scheduler_cpus, runtime_cpus, cold_cpus):
            raise RuntimeError(
                "Either set all of --scheduler-cpus/--runtime-cpus/--cold-cpus "
                "or leave all three unset."
            )
        scheduler_count = int(scheduler_cpus)
        runtime_count = int(runtime_cpus)
        cold_count = int(cold_cpus)

    if scheduler_count <= 0:
        raise RuntimeError("scheduler_cpus must be positive.")
    if scheduler_count > cluster_size:
        raise RuntimeError(
            f"scheduler_cpus={scheduler_count} exceeds cluster size {cluster_size}. "
            "This script keeps scheduler-hot inside one cluster."
        )
    if runtime_count < 0 or cold_count < 0:
        raise RuntimeError("runtime_cpus and cold_cpus must be non-negative.")
    if scheduler_count + runtime_count + cold_count != cpus_per_proc:
        raise RuntimeError(
            f"scheduler({scheduler_count}) + runtime({runtime_count}) + cold({cold_count}) "
            f"!= cpus_per_proc({cpus_per_proc})"
        )
    return scheduler_count, runtime_count, cold_count


def infer_target_numa(logical_rank: int, cpus_per_proc: int) -> int:
    if cpus_per_proc <= 12:
        return 0 if logical_rank <= 3 else 2
    if cpus_per_proc <= 24:
        return logical_rank // 2
    raise RuntimeError(
        f"cpus_per_proc={cpus_per_proc} is unsupported with the no-cross-NUMA rule."
    )


def build_scheduler_entries(
    scheduler_pids: Sequence[int],
    cpu_to_numa: Dict[int, int],
) -> List[Tuple[int, int, List[int], int]]:
    entries = []
    for pid in scheduler_pids:
        proc = psutil.Process(pid)
        worker_cpus = sorted(proc.cpu_affinity())
        current_numa = detect_process_numa(worker_cpus, cpu_to_numa)
        entries.append((pid, worker_cpus, current_numa))

    entries.sort(key=lambda item: (min(item[1]), item[0]))
    ranked = []
    for logical_rank, (pid, worker_cpus, current_numa) in enumerate(entries):
        ranked.append((logical_rank, pid, worker_cpus, current_numa))
    return ranked


def allocate_extra_groups(
    available_cpus: Sequence[int],
    runtime_count: int,
    cold_count: int,
    process_count: int,
) -> List[Tuple[List[int], List[int]]]:
    block_size = runtime_count + cold_count
    if block_size == 0:
        return [([], []) for _ in range(process_count)]
    required = process_count * block_size
    if len(available_cpus) < required:
        raise RuntimeError(
            f"Need {required} non-scheduler CPUs, but only {len(available_cpus)} remain."
        )

    result: List[Tuple[List[int], List[int]]] = []
    offset = 0
    ordered = sorted(available_cpus)
    for _ in range(process_count):
        block = ordered[offset : offset + block_size]
        runtime_cpus = block[:runtime_count]
        cold_cpus = block[runtime_count : runtime_count + cold_count]
        result.append((runtime_cpus, cold_cpus))
        offset += block_size
    return result


def plan_schedulers(
    scheduler_pids: Sequence[int],
    *,
    cpus_per_proc: int,
    cluster_size: int,
    scheduler_count: int,
    runtime_count: int,
    cold_count: int,
) -> List[SchedulerPlan]:
    numa_to_cpus = read_numa_to_cpus()
    numa_to_clusters = build_numa_clusters(numa_to_cpus, cluster_size)
    cpu_to_numa = build_cpu_to_numa(numa_to_cpus)
    ranked_entries = build_scheduler_entries(scheduler_pids, cpu_to_numa)

    entries_by_target_numa: Dict[int, List[Tuple[int, int, List[int], int]]] = {}
    for logical_rank, pid, worker_cpus, current_numa in ranked_entries:
        target_numa = infer_target_numa(logical_rank, cpus_per_proc)
        entries_by_target_numa.setdefault(target_numa, []).append(
            (logical_rank, pid, worker_cpus, current_numa)
        )

    plans: List[SchedulerPlan] = []
    for target_numa, entries in sorted(entries_by_target_numa.items()):
        clusters = [sorted(cluster) for cluster in numa_to_clusters.get(target_numa, [])]
        if not clusters:
            raise RuntimeError(f"NUMA {target_numa} has no cluster information.")
        if len(entries) > len(clusters):
            raise RuntimeError(
                f"NUMA {target_numa} has only {len(clusters)} clusters, "
                f"but needs one scheduler-hot cluster per scheduler for {len(entries)} schedulers."
            )

        hot_clusters = clusters[: len(entries)]
        hot_cluster_cpu_set = {cpu for cluster in hot_clusters for cpu in cluster}
        extra_pool = [
            cpu
            for cpu in sorted(numa_to_cpus[target_numa])
            if cpu not in hot_cluster_cpu_set
        ]
        extra_groups = allocate_extra_groups(
            extra_pool,
            runtime_count=runtime_count,
            cold_count=cold_count,
            process_count=len(entries),
        )

        for index, (logical_rank, pid, worker_cpus, current_numa) in enumerate(entries):
            hot_cluster = hot_clusters[index]
            runtime_cpus, cold_cpus = extra_groups[index]
            scheduler_cpus = sorted(hot_cluster)[:scheduler_count]
            plans.append(
                SchedulerPlan(
                    logical_rank=logical_rank,
                    pid=pid,
                    current_worker_cpus=worker_cpus,
                    current_numa_id=current_numa,
                    target_numa_id=target_numa,
                    hot_cluster_cpus=sorted(hot_cluster),
                    scheduler_cpus=scheduler_cpus,
                    runtime_cpus=runtime_cpus,
                    cold_cpus=cold_cpus,
                )
            )

    plans.sort(key=lambda item: item.logical_rank)
    return plans


def collect_scheduler_service_cpus(plans: Sequence[SchedulerPlan]) -> List[int]:
    cpus: Set[int] = set()
    for plan in plans:
        cpus.update(plan.scheduler_cpus)
        cpus.update(plan.runtime_cpus)
        cpus.update(plan.cold_cpus)
    return sorted(cpus)


def collect_free_cpus_by_numa(
    *,
    plans: Sequence[SchedulerPlan],
    numa_to_cpus: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    service_cpus = set(collect_scheduler_service_cpus(plans))
    result: Dict[int, List[int]] = {}
    for numa_id, cpus in sorted(numa_to_cpus.items()):
        free_cpus = [cpu for cpu in sorted(cpus) if cpu not in service_cpus]
        if free_cpus:
            result[numa_id] = free_cpus
    return result


def infer_helper_cpu_sets(
    *,
    free_cpus_by_numa: Dict[int, List[int]],
    control_cpus_per_helper_numa: int,
) -> Tuple[List[int], List[int], List[int]]:
    helper_numas: List[int] = []
    control_cpus: List[int] = []
    background_cpus: List[int] = []

    for numa_id, free_cpus in sorted(free_cpus_by_numa.items()):
        helper_numas.append(numa_id)
        reserve = min(control_cpus_per_helper_numa, len(free_cpus) // 2)
        if reserve > 0:
            control_cpus.extend(free_cpus[:reserve])
            background_cpus.extend(free_cpus[reserve:])
        else:
            background_cpus.extend(free_cpus)

    return helper_numas, sorted(control_cpus), sorted(background_cpus)


def choose_machine_unique_thread_cpu(
    *,
    plans: Sequence[SchedulerPlan],
    free_cpus_by_numa: Dict[int, List[int]],
) -> Tuple[Optional[int], str]:
    all_free_cpus = sorted(
        {
            cpu
            for cpus in free_cpus_by_numa.values()
            for cpu in cpus
        }
    )
    if all_free_cpus:
        return all_free_cpus[-1], "free"

    for plan in sorted(plans, key=lambda item: item.logical_rank, reverse=True):
        if len(plan.cold_cpus) > 1:
            borrowed_cpu = plan.cold_cpus.pop()
            return borrowed_cpu, f"borrowed rank={plan.logical_rank} cold"

    return None, ""


def remove_cpu_from_list(cpus: Sequence[int], reserved_cpu: Optional[int]) -> List[int]:
    if reserved_cpu is None:
        return sorted(cpus)
    return [cpu for cpu in sorted(cpus) if cpu != reserved_cpu]


def detect_machine_unique_threads(
    patterns: Sequence[Pattern[str]],
) -> List[SpecialThreadMatch]:
    matches: List[SpecialThreadMatch] = []
    for proc in psutil.process_iter(["pid", "name"]):
        process_name = get_name(proc)
        tids = list_thread_ids(proc.pid)
        if not tids:
            continue
        for tid in tids:
            thread_name = read_thread_name(proc.pid, tid)
            if not matches_any(thread_name, patterns):
                continue
            try:
                affinity = get_thread_affinity(tid)
            except OSError:
                continue
            matches.append(
                SpecialThreadMatch(
                    pid=proc.pid,
                    tid=tid,
                    process_name=process_name,
                    thread_name=thread_name,
                    affinity=affinity,
                )
            )
    return matches


def rebind_special_thread(
    match: SpecialThreadMatch,
    target_cpu: int,
    *,
    dry_run: bool,
    verbose: bool,
) -> Tuple[Optional[dict], int]:
    target_cpus = [target_cpu]
    if sorted(match.affinity) == target_cpus:
        if verbose:
            print(
                f"Keep special thread pid={match.pid} tid={match.tid} "
                f"name={match.thread_name} already on {format_cpu_spec(target_cpus)}"
            )
        return None, 0

    print(
        f"{'Plan' if dry_run else 'Move'} special thread pid={match.pid} tid={match.tid} "
        f"proc={match.process_name} name={match.thread_name} "
        f"{format_cpu_spec(match.affinity)} -> {format_cpu_spec(target_cpus)}"
    )
    if dry_run:
        return {
            "pid": match.pid,
            "tid": match.tid,
            "process_name": match.process_name,
            "thread_name": match.thread_name,
            "original_affinity": list(match.affinity),
            "target_cpu": target_cpu,
        }, 0

    try:
        os.sched_setaffinity(match.tid, set(target_cpus))
    except OSError as exc:
        print(
            f"Failed to move special thread pid={match.pid} tid={match.tid}: {exc}",
            file=sys.stderr,
        )
        return None, 1

    return {
        "pid": match.pid,
        "tid": match.tid,
        "process_name": match.process_name,
        "thread_name": match.thread_name,
        "original_affinity": list(match.affinity),
        "target_cpu": target_cpu,
    }, 0


def classify_thread_group(
    pid: int,
    info: ThreadInfo,
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
    cold_patterns: Sequence[Pattern[str]],
) -> str:
    if matches_any(info.name, runtime_patterns):
        return "runtime"
    if info.tid == pid:
        return "scheduler"
    if matches_any(info.name, scheduler_patterns):
        return "scheduler"
    if matches_any(info.name, cold_patterns):
        return "cold"
    return "cold"


def build_backup_record(
    plan: SchedulerPlan,
    thread_infos: Dict[int, ThreadInfo],
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
    cold_patterns: Sequence[Pattern[str]],
) -> dict:
    threads = []
    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        group = classify_thread_group(
            plan.pid,
            info,
            runtime_patterns,
            scheduler_patterns,
            cold_patterns,
        )
        threads.append(
            {
                "tid": info.tid,
                "name": info.name,
                "original_affinity": list(info.affinity),
                "target_group": group,
            }
        )

    return {
        "logical_rank": plan.logical_rank,
        "pid": plan.pid,
        "current_numa_id": plan.current_numa_id,
        "target_numa_id": plan.target_numa_id,
        "current_worker_cpus": list(plan.current_worker_cpus),
        "hot_cluster_cpus": list(plan.hot_cluster_cpus),
        "scheduler_cpus": list(plan.scheduler_cpus),
        "runtime_cpus": list(plan.runtime_cpus),
        "cold_cpus": list(plan.cold_cpus),
        "threads": threads,
    }


def backup_affinity(backup_path: str, payload: dict) -> None:
    path = Path(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_thread_affinity(
    *,
    pid: int,
    tid: int,
    name: str,
    phase: str,
    group: str,
    current_cpus: Sequence[int],
    target_cpus: Sequence[int],
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int]:
    if sorted(current_cpus) == sorted(target_cpus):
        if verbose:
            print(
                f"Keep pid={pid} tid={tid} name={name} phase={phase} group={group} "
                f"already on {format_cpu_spec(target_cpus)}"
            )
        return 0, 0

    action = "Plan" if dry_run else "Move"
    print(
        f"{action} pid={pid} tid={tid} name={name} phase={phase} group={group} "
        f"{format_cpu_spec(current_cpus)} -> {format_cpu_spec(target_cpus)}"
    )
    if dry_run:
        return 1, 0

    try:
        os.sched_setaffinity(tid, set(target_cpus))
    except OSError as exc:
        print(
            f"Failed to move pid={pid} tid={tid} name={name} phase={phase}: {exc}",
            file=sys.stderr,
        )
        return 0, 1
    return 1, 0


def apply_thread_affinity(
    plan: SchedulerPlan,
    thread_infos: Dict[int, ThreadInfo],
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
    cold_patterns: Sequence[Pattern[str]],
    *,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int]:
    changed = 0
    failed = 0

    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        delta_changed, delta_failed = set_thread_affinity(
            pid=plan.pid,
            tid=info.tid,
            name=info.name,
            phase="default",
            group="scheduler",
            current_cpus=info.affinity,
            target_cpus=plan.scheduler_cpus,
            dry_run=dry_run,
            verbose=verbose,
        )
        changed += delta_changed
        failed += delta_failed

    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        group = classify_thread_group(
            plan.pid,
            info,
            runtime_patterns,
            scheduler_patterns,
            cold_patterns,
        )
        if group == "scheduler":
            continue

        target_cpus = plan.runtime_cpus if group == "runtime" else plan.cold_cpus
        delta_changed, delta_failed = set_thread_affinity(
            pid=plan.pid,
            tid=info.tid,
            name=info.name,
            phase="override",
            group=group,
            current_cpus=plan.scheduler_cpus,
            target_cpus=target_cpus,
            dry_run=dry_run,
            verbose=verbose,
        )
        changed += delta_changed
        failed += delta_failed

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

    for thread in data.get("special_threads", []):
        try:
            os.sched_setaffinity(thread["tid"], set(thread["original_affinity"]))
            restored += 1
        except OSError as exc:
            skipped += 1
            print(
                f"Skip restore special tid={thread['tid']} name={thread['thread_name']}: {exc}",
                file=sys.stderr,
            )

    for key in ("control_processes", "moved_processes"):
        for item in data.get(key, []):
            pid = item["pid"]
            original = item["original_affinity"]
            try:
                psutil.Process(pid).cpu_affinity(original)
                restored += 1
            except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
                skipped += 1
                print(f"Skip restore pid={pid}: {exc}", file=sys.stderr)

    print(
        f"Restore done: restored={restored}, skipped={skipped}, backup={backup_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind sglang scheduler threads with no-cross-NUMA planning and "
            "default-then-override thread affinity."
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
        help="Extra keyword for auto-detecting scheduler processes.",
    )
    parser.add_argument(
        "--cpus-per-proc",
        type=int,
        default=16,
        help="Total CPU budget per scheduler process. Default: 16",
    )
    parser.add_argument(
        "--scheduler-cpus",
        type=int,
        default=None,
        help="Override scheduler-hot CPU count. Leave unset to use a preset.",
    )
    parser.add_argument(
        "--runtime-cpus",
        type=int,
        default=None,
        help="Override runtime CPU count. Leave unset to use a preset.",
    )
    parser.add_argument(
        "--cold-cpus",
        type=int,
        default=None,
        help="Override cold/helper CPU count. Leave unset to use a preset.",
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=DEFAULT_CLUSTER_SIZE,
        help=f"Hardware cluster size. Default: {DEFAULT_CLUSTER_SIZE}",
    )
    parser.add_argument(
        "--root-pid",
        type=int,
        action="append",
        default=[],
        help="Explicit sglang root pid for process isolation. Can be passed multiple times.",
    )
    parser.add_argument(
        "--root-keyword",
        action="append",
        default=[],
        help="Extra keyword for auto-detecting sglang roots.",
    )
    parser.add_argument(
        "--control-pid",
        type=int,
        action="append",
        default=[],
        help="Explicit control-plane pid. Can be passed multiple times.",
    )
    parser.add_argument(
        "--control-keyword",
        action="append",
        default=list(DEFAULT_CONTROL_KEYWORDS),
        help=(
            "Keyword for matching control-plane processes in the sglang tree. "
            f"Default: {', '.join(DEFAULT_CONTROL_KEYWORDS)}"
        ),
    )
    parser.add_argument(
        "--runtime-pattern",
        action="append",
        default=list(DEFAULT_RUNTIME_PATTERNS),
        help=(
            "Regex for runtime threads. "
            f"Default: {', '.join(DEFAULT_RUNTIME_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--scheduler-thread-pattern",
        action="append",
        default=list(DEFAULT_SCHEDULER_THREAD_PATTERNS),
        help=(
            "Regex for scheduler-hot threads. "
            f"Default: {', '.join(DEFAULT_SCHEDULER_THREAD_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--cold-thread-pattern",
        action="append",
        default=list(DEFAULT_COLD_PATTERNS),
        help=(
            "Regex for cold/helper threads. "
            f"Default: {', '.join(DEFAULT_COLD_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--control-cpus-per-helper-numa",
        type=int,
        default=DEFAULT_CONTROL_CPUS_PER_HELPER_NUMA,
        help=(
            "When control/background CPU sets are inferred from free CPUs, reserve "
            "up to this many CPUs per helper NUMA for control processes. "
            f"Default: {DEFAULT_CONTROL_CPUS_PER_HELPER_NUMA}"
        ),
    )
    parser.add_argument(
        "--control-cpus",
        type=str,
        default="",
        help="Explicit CPU set for control processes.",
    )
    parser.add_argument(
        "--background-cpus",
        type=str,
        default="",
        help="Explicit CPU set for background processes.",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Background process name to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-pid",
        type=int,
        action="append",
        default=[],
        help="Background process pid to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Attempt to move background processes from all users.",
    )
    parser.add_argument(
        "--skip-process-isolation",
        action="store_true",
        help="Only bind scheduler threads; skip control/background process isolation.",
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
        help="Print detailed per-thread and per-process actions.",
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

    scheduler_count, runtime_count, cold_count = derive_group_sizes(
        cpus_per_proc=args.cpus_per_proc,
        cluster_size=args.cluster_size,
        scheduler_cpus=args.scheduler_cpus,
        runtime_cpus=args.runtime_cpus,
        cold_cpus=args.cold_cpus,
    )

    keywords = list(DEFAULT_SCHEDULER_KEYWORDS) + list(args.scheduler_keyword)
    scheduler_pids = detect_scheduler_pids(keywords, args.scheduler_pid)
    if not scheduler_pids:
        print(
            "Failed to detect scheduler pids automatically. Pass --scheduler-pid.",
            file=sys.stderr,
        )
        return 2

    runtime_patterns = compile_patterns(args.runtime_pattern)
    scheduler_patterns = compile_patterns(args.scheduler_thread_pattern)
    cold_patterns = compile_patterns(args.cold_thread_pattern)
    special_thread_patterns = compile_patterns(DEFAULT_MACHINE_UNIQUE_THREAD_PATTERNS)
    numa_to_cpus = read_numa_to_cpus()

    try:
        plans = plan_schedulers(
            scheduler_pids,
            cpus_per_proc=args.cpus_per_proc,
            cluster_size=args.cluster_size,
            scheduler_count=scheduler_count,
            runtime_count=runtime_count,
            cold_count=cold_count,
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess, RuntimeError) as exc:
        print(f"Failed to build scheduler plans: {exc}", file=sys.stderr)
        return 2

    backup_payload = {
        "created_at": int(time.time()),
        "cpus_per_proc": args.cpus_per_proc,
        "scheduler_cpus": scheduler_count,
        "runtime_cpus": runtime_count,
        "cold_cpus": cold_count,
        "processes": [],
        "control_processes": [],
        "moved_processes": [],
        "root_pids": [],
        "control_pids": [],
        "helper_numas": [],
        "scheduler_service_cpus": [],
        "control_cpus": [],
        "background_cpus": [],
        "special_thread_cpu": None,
        "special_thread_source": "",
        "special_threads": [],
    }

    total_changed = 0
    total_failed = 0

    print(f"Detected scheduler pids: {scheduler_pids}")
    print(
        "Planner preset: "
        f"cpus_per_proc={args.cpus_per_proc} "
        f"scheduler={scheduler_count} runtime={runtime_count} cold={cold_count}"
    )

    free_cpus_by_numa = collect_free_cpus_by_numa(plans=plans, numa_to_cpus=numa_to_cpus)
    special_thread_matches = detect_machine_unique_threads(special_thread_patterns)
    should_bind_special_thread = len(special_thread_matches) == 1
    special_thread_cpu: Optional[int] = None
    special_thread_source = ""
    special_thread_records: List[dict] = []
    helper_numas: List[int] = []

    if should_bind_special_thread:
        special_thread_cpu, special_thread_source = choose_machine_unique_thread_cpu(
            plans=plans,
            free_cpus_by_numa=free_cpus_by_numa,
        )

    for plan in plans:
        try:
            thread_infos = collect_thread_infos(plan.pid)
        except psutil.NoSuchProcess:
            print(f"Skip pid={plan.pid}: process disappeared.", file=sys.stderr)
            total_failed += 1
            continue

        if not thread_infos:
            print(f"Skip pid={plan.pid}: no live threads found.", file=sys.stderr)
            total_failed += 1
            continue

        print(
            f"Scheduler rank={plan.logical_rank} pid={plan.pid} "
            f"current_numa={plan.current_numa_id} target_numa={plan.target_numa_id} "
            f"current_worker={format_cpu_spec(plan.current_worker_cpus)} "
            f"hot_cluster={format_cpu_spec(plan.hot_cluster_cpus)} "
            f"scheduler_hot={format_cpu_spec(plan.scheduler_cpus)} "
            f"runtime={format_cpu_spec(plan.runtime_cpus)} "
            f"cold={format_cpu_spec(plan.cold_cpus)}"
        )

        backup_payload["processes"].append(
            build_backup_record(
                plan,
                thread_infos,
                runtime_patterns,
                scheduler_patterns,
                cold_patterns,
            )
        )

        changed, failed = apply_thread_affinity(
            plan,
            thread_infos,
            runtime_patterns,
            scheduler_patterns,
            cold_patterns,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_changed += changed
        total_failed += failed

    scheduler_service_cpus = collect_scheduler_service_cpus(plans)

    if not args.skip_process_isolation:
        root_keywords = list(DEFAULT_ROOT_KEYWORDS) + list(args.root_keyword)
        root_pids = sorted(set(args.root_pid or auto_detect_sglang_roots(root_keywords)))
        if not root_pids:
            print(
                "Skip process isolation: failed to detect sglang root pids.",
                file=sys.stderr,
            )
        else:
            full_service_tree = get_descendant_pids(root_pids)
            control_pids = filter_child_pids(
                args.control_pid + detect_processes_by_keywords(args.control_keyword),
                full_service_tree,
            )
            launch_server_pids = filter_child_pids(
                auto_detect_sglang_roots(DEFAULT_CONTROL_ROOT_KEYWORDS),
                full_service_tree,
            )
            control_pids = sorted(set(control_pids + launch_server_pids))

            if args.control_cpus:
                control_cpus = parse_cpu_spec(args.control_cpus)
            else:
                helper_numas, inferred_control, inferred_background = infer_helper_cpu_sets(
                    free_cpus_by_numa=free_cpus_by_numa,
                    control_cpus_per_helper_numa=args.control_cpus_per_helper_numa,
                )
                control_cpus = inferred_control

            if args.background_cpus:
                background_cpus = parse_cpu_spec(args.background_cpus)
            else:
                if not helper_numas:
                    helper_numas, _, inferred_background = infer_helper_cpu_sets(
                        free_cpus_by_numa=free_cpus_by_numa,
                        control_cpus_per_helper_numa=args.control_cpus_per_helper_numa,
                    )
                background_cpus = inferred_background

            if should_bind_special_thread:
                control_cpus = remove_cpu_from_list(control_cpus, special_thread_cpu)
                background_cpus = remove_cpu_from_list(background_cpus, special_thread_cpu)

            if control_pids and control_cpus and background_cpus:
                overlap = set(control_cpus) & set(background_cpus)
                if overlap:
                    print(
                        f"Skip process isolation: control/background overlap "
                        f"{format_cpu_spec(sorted(overlap))}",
                        file=sys.stderr,
                    )
                else:
                    control_records, control_failed = rebind_processes(
                        control_pids,
                        control_cpus,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                        label="control",
                    )
                    total_changed += len(control_records)
                    total_failed += control_failed
                    backup_payload["control_processes"] = list(control_records)

                    protected_pids = get_ancestor_pids(os.getpid())
                    protected_pids.update(args.exclude_pid)
                    exclude_names = set(DEFAULT_EXCLUDED_NAMES)
                    exclude_names.update(args.exclude_name)
                    candidates = pick_background_candidates(
                        foreground_pids=set(full_service_tree),
                        protected_pids=protected_pids,
                        exclude_names=exclude_names,
                        same_uid_only=not args.all_users,
                    )

                    moved_background = []
                    for proc in candidates:
                        try:
                            original = proc.cpu_affinity()
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            total_failed += 1
                            continue

                        if sorted(original) == sorted(background_cpus):
                            if args.verbose:
                                print(
                                    f"Keep pid={proc.pid} name={get_name(proc)} already on background CPUs"
                                )
                            continue

                        record = build_affinity_record(proc, original)
                        moved_background.append(record)
                        if args.verbose or args.dry_run:
                            print(
                                f"{'Plan' if args.dry_run else 'Move'} pid={proc.pid} "
                                f"name={record['name']} {format_cpu_spec(original)} -> "
                                f"{format_cpu_spec(background_cpus)}"
                            )

                        if args.dry_run:
                            continue

                        try:
                            proc.cpu_affinity(background_cpus)
                        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
                            total_failed += 1
                            print(f"Failed to move pid={proc.pid}: {exc}", file=sys.stderr)

                    backup_payload["moved_processes"] = list(moved_background)
                    backup_payload["root_pids"] = list(root_pids)
                    backup_payload["control_pids"] = list(control_pids)
                    backup_payload["helper_numas"] = list(helper_numas)
                    backup_payload["scheduler_service_cpus"] = list(scheduler_service_cpus)
                    backup_payload["control_cpus"] = list(control_cpus)
                    backup_payload["background_cpus"] = list(background_cpus)
                    total_changed += len(moved_background)

                    print(f"Detected sglang root pids: {root_pids}")
                    print(f"Detected control pids: {control_pids}")
                    print(f"Helper NUMAs: {helper_numas}")
                    print(f"Scheduler service CPUs: {format_cpu_spec(scheduler_service_cpus)}")
                    print(f"Control CPUs: {format_cpu_spec(control_cpus)}")
                    print(f"Background CPUs: {format_cpu_spec(background_cpus)}")
                    if special_thread_cpu is not None:
                        print(
                            f"Special thread CPU: {special_thread_cpu} "
                            f"({special_thread_source})"
                        )
                    print(f"Candidate background process count: {len(candidates)}")
            else:
                print(
                    "Skip process isolation: no safe free CPU set remained for both "
                    "control and background processes.",
                    file=sys.stderr,
                )

    if len(special_thread_matches) > 1:
        print(
            "Skip special thread binding: detected multiple machine-unique thread matches "
            f"{[(m.pid, m.tid, m.thread_name) for m in special_thread_matches]}",
            file=sys.stderr,
        )
    elif special_thread_matches and special_thread_cpu is None:
        print(
            "Skip special thread binding: neither a free CPU nor a borrowable cold CPU remained.",
            file=sys.stderr,
        )
    elif special_thread_matches and special_thread_cpu is not None:
        record, failed = rebind_special_thread(
            special_thread_matches[0],
            special_thread_cpu,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_failed += failed
        if record is not None:
            special_thread_records.append(record)
            total_changed += 1
            backup_payload["special_threads"] = list(special_thread_records)
            backup_payload["special_thread_cpu"] = special_thread_cpu
            backup_payload["special_thread_source"] = special_thread_source
    elif args.verbose:
        print(
            "No machine-unique external special thread matched "
            f"{', '.join(DEFAULT_MACHINE_UNIQUE_THREAD_PATTERNS)}"
        )

    if args.dry_run:
        print(
            f"Dry-run done: would_change={total_changed}, failed={total_failed}, "
            f"backup_path={args.backup_path}"
        )
        return 0

    backup_affinity(args.backup_path, backup_payload)
    print(
        f"Apply done: changed={total_changed}, failed={total_failed}, "
        f"backup={args.backup_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
