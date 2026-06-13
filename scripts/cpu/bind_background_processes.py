#!/usr/bin/env python3
"""Bind non-sglang background processes to dedicated CPUs.

This script is intentionally conservative:
- It keeps the detected sglang service tree untouched.
- It moves only user-space processes by default.
- It skips itself and its ancestor shell/process tree.

Typical workflow:
1. Start sglang so its worker affinity is already in place.
2. Run this script to move unrelated background processes to the
   complement CPU set.
3. Use --restore to put the moved processes back to their original
   affinity masks.

Foreground detection modes:
- The whole detected sglang process tree is always protected.
- Worker CPUs are inferred from scheduler processes.
- Control processes are moved to the configured control CPU set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import psutil


DEFAULT_BACKUP_PATH = "/tmp/sglang_background_affinity_backup.json"
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
DEFAULT_CONTROL_CPUSET = "32-47,128-143"
DEFAULT_SCHEDULER_KEYWORDS = (
    "sglang::scheduler",
    "scheduler",
)
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
                raise ValueError(f"Invalid cpu range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpu_spec(cpus: Sequence[int]) -> str:
    if not cpus:
        return ""
    sorted_cpus = sorted(set(cpus))
    ranges = []
    start = prev = sorted_cpus[0]
    for cpu in sorted_cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append((start, prev))
        start = prev = cpu
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def get_online_cpus() -> List[int]:
    online_path = Path("/sys/devices/system/cpu/online")
    if online_path.exists():
        return parse_cpu_spec(online_path.read_text().strip())
    count = os.cpu_count()
    if not count:
        raise RuntimeError("Failed to detect online CPUs")
    return list(range(count))


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
        cmdline = get_cmdline(proc)
        name = get_name(proc)
        haystack = f"{name} {cmdline}"
        if any(keyword in haystack for keyword in keywords):
            roots.append(proc.pid)
    return sorted(set(roots))


def detect_processes_by_keywords(keywords: Sequence[str]) -> List[int]:
    if not keywords:
        return []
    matched = []
    for proc in psutil.process_iter(["pid", "name"]):
        cmdline = get_cmdline(proc)
        name = get_name(proc)
        haystack = f"{name} {cmdline}"
        if any(keyword in haystack for keyword in keywords):
            matched.append(proc.pid)
    return sorted(set(matched))


def filter_child_pids(pids: Iterable[int], allowed_pids: Set[int]) -> List[int]:
    return sorted({pid for pid in pids if pid in allowed_pids})


def detect_service_cpus(foreground_pids: Iterable[int]) -> List[int]:
    service_cpus: Set[int] = set()
    for pid in foreground_pids:
        try:
            proc = psutil.Process(pid)
            service_cpus.update(proc.cpu_affinity())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return sorted(service_cpus)


def is_kernel_like_process(proc: psutil.Process) -> bool:
    # Kernel threads often have empty cmdline and/or ppid 2.
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

        name = get_name(proc)
        if name in exclude_names:
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
) -> tuple[List[dict], int]:
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
                f"{'Plan' if dry_run else 'Move'} {label} "
                f"pid={proc.pid} name={record['name']} "
                f"{format_cpu_spec(original)} -> {format_cpu_spec(target_cpus)}"
            )

        if dry_run:
            continue

        try:
            proc.cpu_affinity(list(target_cpus))
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            failed += 1
            print(f"Failed to move {label} pid={proc.pid}: {exc}", file=sys.stderr)

    return changed, failed


def backup_affinity(
    backup_path: str,
    service_cpus: Sequence[int],
    background_cpus: Sequence[int],
    control_cpus: Sequence[int],
    rebound_control: Sequence[dict],
    moved_background: Sequence[dict],
) -> None:
    payload = {
        "created_at": int(time.time()),
        "service_cpus": list(service_cpus),
        "background_cpus": list(background_cpus),
        "control_cpus": list(control_cpus),
        "control_processes": list(rebound_control),
        "moved_processes": list(moved_background),
    }
    path = Path(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def restore_affinity(backup_path: str) -> int:
    path = Path(backup_path)
    if not path.exists():
        print(f"Backup file not found: {backup_path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    restored = 0
    skipped = 0
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
        f"Restore done: restored={restored}, skipped={skipped}, "
        f"backup={backup_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "After sglang starts, move unrelated background processes to the "
            "complement CPU set."
        )
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
        help="Extra keyword for auto-detecting sglang roots from process name/cmdline.",
    )
    parser.add_argument(
        "--scheduler-keyword",
        action="append",
        default=[],
        help=(
            "Extra keyword for matching scheduler processes used to infer "
            "worker CPUs."
        ),
    )
    parser.add_argument(
        "--control-pid",
        type=int,
        action="append",
        default=[],
        help=(
            "PID of a control-plane process to keep in the service CPU set. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--control-keyword",
        action="append",
        default=list(DEFAULT_CONTROL_KEYWORDS),
        help=(
            "Keyword for matching control-plane processes to keep in the "
            "service CPU set. Can be passed multiple times. "
            f"Default: {', '.join(DEFAULT_CONTROL_KEYWORDS)}"
        ),
    )
    parser.add_argument(
        "--service-cpus",
        type=str,
        default="",
        help=(
            "Explicit CPU set reserved for sglang, e.g. 0-31,96-111. "
            "Default: union of current affinity of detected sglang processes."
        ),
    )
    parser.add_argument(
        "--control-cpus",
        type=str,
        default=DEFAULT_CONTROL_CPUSET,
        help=(
            "Explicit CPU set for control-plane processes. When set, the script "
            "first moves control processes to these CPUs, then infers worker "
            "CPUs from scheduler processes only. "
            f"Default: {DEFAULT_CONTROL_CPUSET}"
        ),
    )
    parser.add_argument(
        "--background-cpus",
        type=str,
        default="",
        help=(
            "Explicit CPU set for background processes. "
            "Default: online CPUs minus service CPUs."
        ),
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Process name to exclude from rebinding. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-pid",
        type=int,
        action="append",
        default=[],
        help="PID to exclude from rebinding. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Attempt to move processes from all users, not only current uid.",
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
        help="Print per-process decisions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.restore:
        return restore_affinity(args.backup_path)

    keywords = list(DEFAULT_ROOT_KEYWORDS) + list(args.root_keyword)
    root_pids = sorted(set(args.root_pid or auto_detect_sglang_roots(keywords)))
    if not root_pids:
        print(
            "Failed to detect sglang root pids automatically. "
            "Pass --root-pid explicitly.",
            file=sys.stderr,
        )
        return 2

    full_service_tree = get_descendant_pids(root_pids)
    if not full_service_tree:
        print("Detected root pids have no live process tree.", file=sys.stderr)
        return 2

    scheduler_keywords = list(DEFAULT_SCHEDULER_KEYWORDS) + list(args.scheduler_keyword)
    scheduler_pids = filter_child_pids(
        detect_processes_by_keywords(scheduler_keywords),
        full_service_tree,
    )
    control_pids = filter_child_pids(
        args.control_pid + detect_processes_by_keywords(args.control_keyword),
        full_service_tree,
    )
    if args.control_cpus:
        launch_server_pids = filter_child_pids(
            auto_detect_sglang_roots(DEFAULT_CONTROL_ROOT_KEYWORDS),
            full_service_tree,
        )
        control_pids = sorted(set(control_pids + launch_server_pids))

    foreground_pids = full_service_tree

    control_cpus = parse_cpu_spec(args.control_cpus) if args.control_cpus else []
    rebound_control = []
    control_failed = 0
    if control_cpus:
        if not control_pids:
            print(
                "Control CPU mode found no control processes. "
                "Pass --control-pid/--control-keyword explicitly.",
                file=sys.stderr,
            )
            return 2
        rebound_control, control_failed = rebind_processes(
            control_pids,
            control_cpus,
            dry_run=args.dry_run,
            verbose=args.verbose,
            label="control",
        )

    if args.service_cpus:
        service_cpus = parse_cpu_spec(args.service_cpus)
        if control_cpus:
            service_cpus = sorted(set(service_cpus) | set(control_cpus))
    elif control_cpus:
        worker_cpus = detect_service_cpus(scheduler_pids)
        if not worker_cpus:
            print(
                "Failed to infer worker CPUs from scheduler affinity while "
                "--control-cpus is enabled. Pass --service-cpus explicitly.",
                file=sys.stderr,
            )
            return 2
        service_cpus = sorted(set(worker_cpus) | set(control_cpus))
    else:
        service_cpus = detect_service_cpus(foreground_pids)

    if not service_cpus:
        print(
            "Failed to infer service CPUs from sglang affinity. "
            "Pass --service-cpus explicitly.",
            file=sys.stderr,
        )
        return 2

    online_cpus = get_online_cpus()
    if args.background_cpus:
        background_cpus = parse_cpu_spec(args.background_cpus)
    else:
        background_cpus = sorted(set(online_cpus) - set(service_cpus))

    if not background_cpus:
        hint = ""
        if not args.service_cpus and not control_cpus:
            hint = (
                " Foreground CPUs may already cover all online CPUs; try "
                "--control-cpus or pass --service-cpus explicitly."
            )
        print(f"Background CPU set is empty.{hint}", file=sys.stderr)
        return 2

    protected_pids = get_ancestor_pids(os.getpid())
    protected_pids.update(args.exclude_pid)
    exclude_names = set(DEFAULT_EXCLUDED_NAMES)
    exclude_names.update(args.exclude_name)

    candidates = pick_background_candidates(
        foreground_pids=set(foreground_pids),
        protected_pids=protected_pids,
        exclude_names=exclude_names,
        same_uid_only=not args.all_users,
    )

    print(f"Detected sglang root pids: {root_pids}")
    if scheduler_pids:
        print(f"Detected scheduler pids: {scheduler_pids}")
    if control_pids:
        print(f"Detected control pids: {control_pids}")
    print(f"Foreground process count: {len(foreground_pids)}")
    if control_cpus:
        print(f"Control CPUs: {format_cpu_spec(control_cpus)}")
    print(f"Service CPUs: {format_cpu_spec(service_cpus)}")
    print(f"Background CPUs: {format_cpu_spec(background_cpus)}")
    print(f"Candidate background process count: {len(candidates)}")

    moved = []
    failed = control_failed
    for proc in candidates:
        try:
            original = proc.cpu_affinity()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            failed += 1
            continue

        if sorted(original) == sorted(background_cpus):
            if args.verbose:
                print(f"Keep pid={proc.pid} name={get_name(proc)} already on background CPUs")
            continue

        record = build_affinity_record(proc, original)
        moved.append(record)

        if args.verbose or args.dry_run:
            print(
                f"{'Plan' if args.dry_run else 'Move'} "
                f"pid={proc.pid} name={record['name']} "
                f"{format_cpu_spec(original)} -> {format_cpu_spec(background_cpus)}"
            )

        if args.dry_run:
            continue

        try:
            proc.cpu_affinity(background_cpus)
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            failed += 1
            print(f"Failed to move pid={proc.pid}: {exc}", file=sys.stderr)

    if args.dry_run:
        print(
            f"Dry-run done: would_move={len(moved)}, failed_to_inspect={failed}, "
            f"backup_path={args.backup_path}"
        )
        return 0

    backup_affinity(
        args.backup_path,
        service_cpus,
        background_cpus,
        control_cpus,
        rebound_control,
        moved,
    )
    print(
        f"Apply done: moved={len(moved)}, failed={failed}, "
        f"backup={args.backup_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
