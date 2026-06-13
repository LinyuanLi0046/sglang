#!/usr/bin/env python3
"""Bind scheduler threads with a cluster-aware runtime/hot/cold layout.

This script targets the current Ascend machine layout where each hardware
cluster contains 8 CPUs. Inside one NUMA, scheduler processes are packed as:

- 1 hot cluster per scheduler process.
- The hot cluster is split into 4 runtime CPUs + 4 scheduler-hot CPUs.
- One cold cluster is shared by every adjacent pair of schedulers, each taking
  4 CPUs for helper/cold threads.

For example, if one NUMA has clusters [C0, C1, C2, C3, C4, C5] and four
schedulers on that NUMA, the script plans:

- scheduler0: runtime=C0[0:4], hot=C0[4:8], cold=C1[0:4]
- scheduler1: runtime=C2[0:4], hot=C2[4:8], cold=C1[4:8]
- scheduler2: runtime=C3[0:4], hot=C3[4:8], cold=C4[0:4]
- scheduler3: runtime=C5[0:4], hot=C5[4:8], cold=C4[4:8]

Thread groups are fixed:

- runtime: acl/release/RT_RECYCLE threads
- hot: main thread + all sglang::schedul* threads
- cold: all remaining helper/background threads
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

import psutil


DEFAULT_BACKUP_PATH = "/tmp/sglang_scheduler_thread_affinity_backup.json"
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
DEFAULT_RUNTIME_PATTERNS = (
    r"^acl_thread$",
    r"^release_thread$",
    r"^RT_RECYCLE(?:_.+)?$",
)
DEFAULT_SCHEDULER_THREAD_PATTERNS = (
    r"^sglang::schedul",
)
DEFAULT_CLUSTER_SIZE = 8
DEFAULT_SUBGROUP_SIZE = 4
DEFAULT_CONTROL_CPUS_PER_AUX_NUMA = 16
DEFAULT_EXCLUDED_NAMES = {
    "systemd",
    "systemd-journal",
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


@dataclass
class ThreadInfo:
    tid: int
    name: str
    affinity: List[int]


@dataclass
class SchedulerPlan:
    pid: int
    numa_id: int
    current_worker_cpus: List[int]
    hot_cluster_cpus: List[int]
    cold_cluster_cpus: List[int]
    runtime_cpus: List[int]
    scheduler_cpus: List[int]
    cold_cpus: List[int]


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
    sorted_cpus = sorted(set(cpus))
    start = prev = sorted_cpus[0]
    ranges: List[str] = []
    for cpu in sorted_cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


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


def get_online_cpus() -> List[int]:
    online_path = Path("/sys/devices/system/cpu/online")
    if online_path.exists():
        return parse_cpu_spec(online_path.read_text().strip())
    count = os.cpu_count()
    if not count:
        raise RuntimeError("Failed to detect online CPUs")
    return list(range(count))


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


def split_cluster(cluster_cpus: Sequence[int], subgroup_size: int) -> Tuple[List[int], List[int]]:
    ordered = sorted(cluster_cpus)
    if len(ordered) < subgroup_size * 2:
        raise RuntimeError(
            f"Cluster {format_cpu_spec(cluster_cpus)} has only {len(ordered)} CPUs, "
            f"cannot split into two groups of {subgroup_size}"
        )
    return ordered[:subgroup_size], ordered[subgroup_size : subgroup_size * 2]


def plan_schedulers(
    scheduler_pids: Sequence[int],
    cluster_size: int,
    subgroup_size: int,
) -> List[SchedulerPlan]:
    numa_to_cpus = read_numa_to_cpus()
    numa_to_clusters = build_numa_clusters(numa_to_cpus, cluster_size)
    cpu_to_numa = build_cpu_to_numa(numa_to_cpus)

    schedulers_by_numa: Dict[int, List[Tuple[int, List[int]]]] = {}
    for pid in scheduler_pids:
        proc = psutil.Process(pid)
        worker_cpus = sorted(proc.cpu_affinity())
        numa_id = detect_process_numa(worker_cpus, cpu_to_numa)
        schedulers_by_numa.setdefault(numa_id, []).append((pid, worker_cpus))

    plans: List[SchedulerPlan] = []
    for numa_id, entries in sorted(schedulers_by_numa.items()):
        entries.sort(key=lambda item: (min(item[1]), item[0]))
        clusters = numa_to_clusters.get(numa_id, [])
        required_clusters = len(entries) + math.ceil(len(entries) / 2)
        if len(clusters) < required_clusters:
            raise RuntimeError(
                f"NUMA {numa_id} has only {len(clusters)} clusters "
                f"({[format_cpu_spec(cluster) for cluster in clusters]}), "
                f"but {len(entries)} schedulers need at least {required_clusters} "
                "clusters for the 12-core runtime/hot/cold layout."
            )

        index = 0
        pair_start = 0
        while pair_start < len(entries):
            left_pid, left_worker = entries[pair_start]
            left_hot_cluster = clusters[index]
            index += 1
            shared_cold_cluster = clusters[index]
            index += 1

            left_runtime_cpus, left_scheduler_cpus = split_cluster(
                left_hot_cluster, subgroup_size
            )
            left_cold_cpus = sorted(shared_cold_cluster)[:subgroup_size]
            plans.append(
                SchedulerPlan(
                    pid=left_pid,
                    numa_id=numa_id,
                    current_worker_cpus=left_worker,
                    hot_cluster_cpus=sorted(left_hot_cluster),
                    cold_cluster_cpus=sorted(shared_cold_cluster),
                    runtime_cpus=left_runtime_cpus,
                    scheduler_cpus=left_scheduler_cpus,
                    cold_cpus=left_cold_cpus,
                )
            )

            if pair_start + 1 < len(entries):
                right_pid, right_worker = entries[pair_start + 1]
                right_hot_cluster = clusters[index]
                index += 1
                right_runtime_cpus, right_scheduler_cpus = split_cluster(
                    right_hot_cluster, subgroup_size
                )
                right_cold_cpus = sorted(shared_cold_cluster)[subgroup_size : subgroup_size * 2]
                if len(right_cold_cpus) < subgroup_size:
                    raise RuntimeError(
                        f"Cold cluster {format_cpu_spec(shared_cold_cluster)} does not "
                        f"have enough CPUs for both schedulers on NUMA {numa_id}"
                    )
                plans.append(
                    SchedulerPlan(
                        pid=right_pid,
                        numa_id=numa_id,
                        current_worker_cpus=right_worker,
                        hot_cluster_cpus=sorted(right_hot_cluster),
                        cold_cluster_cpus=sorted(shared_cold_cluster),
                        runtime_cpus=right_runtime_cpus,
                        scheduler_cpus=right_scheduler_cpus,
                        cold_cpus=right_cold_cpus,
                    )
                )

            pair_start += 2

    plans.sort(key=lambda item: item.pid)
    return plans


def collect_scheduler_service_cpus(plans: Sequence[SchedulerPlan]) -> List[int]:
    cpus: Set[int] = set()
    for plan in plans:
        cpus.update(plan.runtime_cpus)
        cpus.update(plan.scheduler_cpus)
        cpus.update(plan.cold_cpus)
    return sorted(cpus)


def infer_aux_process_cpu_sets(
    *,
    plans: Sequence[SchedulerPlan],
    numa_to_cpus: Dict[int, List[int]],
    control_cpus_per_aux_numa: int,
) -> Tuple[List[int], List[int], List[int]]:
    worker_numas = sorted({plan.numa_id for plan in plans})
    auxiliary_numas = sorted(set(numa_to_cpus) - set(worker_numas))
    if not auxiliary_numas:
        raise RuntimeError(
            "Failed to infer auxiliary NUMA nodes for control/background processes."
        )

    control_cpus: List[int] = []
    background_cpus: List[int] = []
    aux_pool: List[int] = []
    for numa_id in auxiliary_numas:
        cpus = sorted(numa_to_cpus[numa_id])
        aux_pool.extend(cpus)
        reserve = min(control_cpus_per_aux_numa, len(cpus))
        control_cpus.extend(cpus[:reserve])
        background_cpus.extend(cpus[reserve:])

    if not control_cpus:
        raise RuntimeError("Derived control CPU set is empty.")
    if not background_cpus:
        raise RuntimeError("Derived background CPU set is empty.")
    return auxiliary_numas, sorted(control_cpus), sorted(background_cpus)


def classify_thread_group(
    pid: int,
    info: ThreadInfo,
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
) -> str:
    if matches_any(info.name, runtime_patterns):
        return "runtime"
    if info.tid == pid:
        return "scheduler"
    if matches_any(info.name, scheduler_patterns):
        return "scheduler"
    return "cold"


def build_backup_record(
    plan: SchedulerPlan,
    thread_infos: Dict[int, ThreadInfo],
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
) -> dict:
    threads = []
    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        group = classify_thread_group(plan.pid, info, runtime_patterns, scheduler_patterns)
        threads.append(
            {
                "tid": info.tid,
                "name": info.name,
                "original_affinity": list(info.affinity),
                "target_group": group,
            }
        )

    return {
        "pid": plan.pid,
        "numa_id": plan.numa_id,
        "current_worker_cpus": list(plan.current_worker_cpus),
        "hot_cluster_cpus": list(plan.hot_cluster_cpus),
        "cold_cluster_cpus": list(plan.cold_cluster_cpus),
        "runtime_cpus": list(plan.runtime_cpus),
        "scheduler_cpus": list(plan.scheduler_cpus),
        "cold_cpus": list(plan.cold_cpus),
        "threads": threads,
    }


def backup_affinity(
    backup_path: str,
    backup_payload: dict,
) -> None:
    path = Path(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(backup_payload, indent=2), encoding="utf-8")


def apply_thread_affinity(
    plan: SchedulerPlan,
    thread_infos: Dict[int, ThreadInfo],
    runtime_patterns: Sequence[Pattern[str]],
    scheduler_patterns: Sequence[Pattern[str]],
    *,
    dry_run: bool,
    verbose: bool,
) -> Tuple[int, int]:
    changed = 0
    failed = 0
    for info in sorted(thread_infos.values(), key=lambda item: item.tid):
        group = classify_thread_group(
            plan.pid, info, runtime_patterns, scheduler_patterns
        )
        if group == "runtime":
            target_cpus = plan.runtime_cpus
        elif group == "scheduler":
            target_cpus = plan.scheduler_cpus
        else:
            target_cpus = plan.cold_cpus

        if sorted(info.affinity) == sorted(target_cpus):
            if verbose:
                print(
                    f"Keep pid={plan.pid} tid={info.tid} name={info.name} "
                    f"group={group} already on {format_cpu_spec(target_cpus)}"
                )
            continue

        action = "Plan" if dry_run else "Move"
        print(
            f"{action} pid={plan.pid} tid={info.tid} name={info.name} "
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
                f"Failed to move pid={plan.pid} tid={info.tid} name={info.name}: {exc}",
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
            "Bind scheduler threads with a fixed 12-core layout per scheduler: "
            "4 runtime CPUs + 4 scheduler-hot CPUs + 4 cold-helper CPUs."
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
        help="Extra keyword for auto-detecting sglang roots from process name/cmdline.",
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
            "Regex for runtime threads that go to the runtime 4-core half cluster. "
            f"Default: {', '.join(DEFAULT_RUNTIME_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--scheduler-thread-pattern",
        action="append",
        default=list(DEFAULT_SCHEDULER_THREAD_PATTERNS),
        help=(
            "Regex for scheduler threads that go to the scheduler-hot 4-core "
            f"half cluster. Default: {', '.join(DEFAULT_SCHEDULER_THREAD_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=DEFAULT_CLUSTER_SIZE,
        help=(
            "Expected hardware cluster size when cluster_id is unavailable. "
            f"Default: {DEFAULT_CLUSTER_SIZE}"
        ),
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=DEFAULT_SUBGROUP_SIZE,
        help=(
            "CPUs used by each sub-group inside one scheduler plan: runtime, "
            f"scheduler-hot, or one half of a shared cold cluster. Default: {DEFAULT_SUBGROUP_SIZE}"
        ),
    )
    parser.add_argument(
        "--control-cpus-per-aux-numa",
        type=int,
        default=DEFAULT_CONTROL_CPUS_PER_AUX_NUMA,
        help=(
            "When control/background CPU sets are inferred from auxiliary NUMA nodes, "
            "reserve this many CPUs per auxiliary NUMA for control processes. "
            f"Default: {DEFAULT_CONTROL_CPUS_PER_AUX_NUMA}"
        ),
    )
    parser.add_argument(
        "--control-cpus",
        type=str,
        default="",
        help=(
            "Explicit CPU set for control processes. Default: inferred from auxiliary "
            "NUMA nodes not used by scheduler workers."
        ),
    )
    parser.add_argument(
        "--background-cpus",
        type=str,
        default="",
        help=(
            "Explicit CPU set for background processes. Default: remaining CPUs in "
            "the auxiliary NUMA nodes."
        ),
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Background process name to exclude from rebinding. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-pid",
        type=int,
        action="append",
        default=[],
        help="Background process pid to exclude from rebinding. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Attempt to move background processes from all users, not only current uid.",
    )
    parser.add_argument(
        "--skip-process-isolation",
        action="store_true",
        help="Only bind scheduler threads; skip the control/background process stage.",
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
        help="Print per-thread decisions.",
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

    runtime_patterns = compile_patterns(args.runtime_pattern)
    scheduler_patterns = compile_patterns(args.scheduler_thread_pattern)
    numa_to_cpus = read_numa_to_cpus()

    try:
        plans = plan_schedulers(
            scheduler_pids,
            cluster_size=args.cluster_size,
            subgroup_size=args.subgroup_size,
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess, RuntimeError) as exc:
        print(f"Failed to build scheduler plans: {exc}", file=sys.stderr)
        return 2

    backup_payload = {
        "created_at": int(time.time()),
        "processes": [],
        "control_processes": [],
        "moved_processes": [],
        "root_pids": [],
        "control_pids": [],
        "worker_numas": [],
        "auxiliary_numas": [],
        "scheduler_service_cpus": [],
        "control_cpus": [],
        "background_cpus": [],
    }
    total_changed = 0
    total_failed = 0

    print(f"Detected scheduler pids: {scheduler_pids}")
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
            f"Scheduler pid={plan.pid} numa={plan.numa_id} "
            f"current_worker={format_cpu_spec(plan.current_worker_cpus)} "
            f"hot_cluster={format_cpu_spec(plan.hot_cluster_cpus)} "
            f"cold_cluster={format_cpu_spec(plan.cold_cluster_cpus)} "
            f"runtime={format_cpu_spec(plan.runtime_cpus)} "
            f"scheduler_hot={format_cpu_spec(plan.scheduler_cpus)} "
            f"cold={format_cpu_spec(plan.cold_cpus)}"
        )

        backup_payload["processes"].append(
            build_backup_record(plan, thread_infos, runtime_patterns, scheduler_patterns)
        )

        changed, failed = apply_thread_affinity(
            plan,
            thread_infos,
            runtime_patterns,
            scheduler_patterns,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_changed += changed
        total_failed += failed

    if not args.skip_process_isolation:
        root_keywords = list(DEFAULT_ROOT_KEYWORDS) + list(args.root_keyword)
        root_pids = sorted(set(args.root_pid or auto_detect_sglang_roots(root_keywords)))
        if not root_pids:
            print(
                "Failed to detect sglang root pids for process isolation. "
                "Pass --root-pid explicitly or use --skip-process-isolation.",
                file=sys.stderr,
            )
            return 2

        full_service_tree = get_descendant_pids(root_pids)
        if not full_service_tree:
            print(
                "Detected sglang root pids have no live process tree for process isolation.",
                file=sys.stderr,
            )
            return 2

        control_pids = filter_child_pids(
            args.control_pid + detect_processes_by_keywords(args.control_keyword),
            full_service_tree,
        )
        launch_server_pids = filter_child_pids(
            auto_detect_sglang_roots(DEFAULT_CONTROL_ROOT_KEYWORDS),
            full_service_tree,
        )
        control_pids = sorted(set(control_pids + launch_server_pids))
        if not control_pids:
            print(
                "Failed to detect control processes for process isolation. "
                "Pass --control-pid/--control-keyword explicitly or use "
                "--skip-process-isolation.",
                file=sys.stderr,
            )
            return 2

        scheduler_service_cpus = collect_scheduler_service_cpus(plans)
        if args.control_cpus:
            control_cpus = parse_cpu_spec(args.control_cpus)
            auxiliary_numas = sorted(set(numa_to_cpus) - {plan.numa_id for plan in plans})
        elif args.background_cpus:
            auxiliary_numas = sorted(set(numa_to_cpus) - {plan.numa_id for plan in plans})
            inferred_control = []
            for numa_id in auxiliary_numas:
                cpus = sorted(numa_to_cpus[numa_id])
                inferred_control.extend(cpus[: min(args.control_cpus_per_aux_numa, len(cpus))])
            control_cpus = sorted(set(inferred_control))
        else:
            auxiliary_numas, control_cpus, inferred_background = infer_aux_process_cpu_sets(
                plans=plans,
                numa_to_cpus=numa_to_cpus,
                control_cpus_per_aux_numa=args.control_cpus_per_aux_numa,
            )

        if args.background_cpus:
            background_cpus = parse_cpu_spec(args.background_cpus)
        else:
            if args.control_cpus:
                online_cpus = get_online_cpus()
                aux_pool = []
                for numa_id in sorted(set(numa_to_cpus) - {plan.numa_id for plan in plans}):
                    aux_pool.extend(numa_to_cpus[numa_id])
                if aux_pool:
                    background_cpus = sorted(set(aux_pool) - set(control_cpus))
                else:
                    background_cpus = sorted(
                        set(online_cpus) - set(scheduler_service_cpus) - set(control_cpus)
                    )
            else:
                background_cpus = inferred_background

        service_cpus = sorted(set(scheduler_service_cpus) | set(control_cpus))
        overlap = set(background_cpus) & set(service_cpus)
        if overlap:
            print(
                f"Background CPUs overlap service CPUs: {format_cpu_spec(sorted(overlap))}",
                file=sys.stderr,
            )
            return 2
        if not background_cpus:
            print("Background CPU set is empty.", file=sys.stderr)
            return 2

        control_records, control_failed = rebind_processes(
            control_pids,
            control_cpus,
            dry_run=args.dry_run,
            verbose=args.verbose,
            label="control",
        )
        backup_payload["control_processes"] = list(control_records)
        backup_payload["root_pids"] = list(root_pids)
        backup_payload["control_pids"] = list(control_pids)
        backup_payload["worker_numas"] = sorted({plan.numa_id for plan in plans})
        backup_payload["auxiliary_numas"] = list(auxiliary_numas)
        backup_payload["scheduler_service_cpus"] = list(scheduler_service_cpus)
        backup_payload["control_cpus"] = list(control_cpus)
        backup_payload["background_cpus"] = list(background_cpus)
        total_changed += len(control_records)
        total_failed += control_failed

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

        print(f"Detected sglang root pids: {root_pids}")
        print(f"Detected control pids: {control_pids}")
        print(f"Process isolation worker NUMAs: {sorted({plan.numa_id for plan in plans})}")
        print(f"Process isolation auxiliary NUMAs: {auxiliary_numas}")
        print(f"Scheduler service CPUs: {format_cpu_spec(scheduler_service_cpus)}")
        print(f"Control CPUs: {format_cpu_spec(control_cpus)}")
        print(f"Background CPUs: {format_cpu_spec(background_cpus)}")
        print(f"Candidate background process count: {len(candidates)}")

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
        total_changed += len(moved_background)

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
