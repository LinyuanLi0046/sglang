"""Topology-aware CPU affinity helpers for Ascend NPU scheduler processes.

This module intentionally does not import ``torch`` or ``torch_npu``.  Early CPU
binding runs before the accelerator runtime is initialized, so the scheduler's
existing ``gpu_id`` argument is the authoritative runtime NPU id.  The physical
NPU id used by ``npu-smi`` is resolved from the Ascend visibility environment.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional

import psutil

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NPU_ROW_RE = re.compile(r"^NPU(?P<id>\d+)$")
_NUMA_NODE_RE = re.compile(r"^node(?P<id>\d+)$")
_VALID_TOPOLOGY_TOKENS = frozenset(
    {"X", "SYS", "PHB", "PIX", "PXB", "SIO", "UB", "NA"}
)
_TOPOLOGY_SOURCE = "npu-smi info -t topo"
_TOPOLOGY_LOG_LOCK = threading.Lock()
_LOGGED_TOPOLOGY_SCOPES: set[str] = set()


class NpuAffinityError(RuntimeError):
    """A recoverable NPU topology or CPU-affinity failure."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        logical_npu_id: Optional[int] = None,
        physical_npu_id: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.logical_npu_id = logical_npu_id
        self.physical_npu_id = physical_npu_id


@dataclass(frozen=True)
class PhysicalCore:
    socket_id: int
    core_id: int
    logical_cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class NpuTopologyEntry:
    physical_npu_id: int
    raw_cpu_affinity: str
    local_cpu_ids: frozenset[int]
    numa_node: int
    physical_cores: tuple[PhysicalCore, ...]


@dataclass(frozen=True)
class NpuTopology:
    source: str
    entries: dict[int, NpuTopologyEntry]
    online_cpu_ids: frozenset[int]


@dataclass(frozen=True)
class NpuAffinityAssignment:
    # This is the runtime device id passed to run_scheduler_process as gpu_id.
    logical_npu_id: int
    physical_npu_id: int
    numa_node: int
    slot_index: int
    slots_on_node: int
    requested_pcores: int
    effective_pcores: int
    physical_core_keys: tuple[tuple[int, int], ...]
    logical_cpu_ids: tuple[int, ...]
    topology_source: str

    @property
    def runtime_npu_id(self) -> int:
        return self.logical_npu_id


@dataclass(frozen=True)
class NpuAffinityApplyResult:
    requested_cpu_ids: tuple[int, ...]
    main_actual_cpu_ids: tuple[int, ...]
    main_matched: bool
    threads_total: int
    threads_bound: int
    threads_exited: int
    threads_failed: int
    threads_mismatched: int


def parse_cpu_list(value: str) -> frozenset[int]:
    """Parse a Linux CPU-list string such as ``0-7,16,32-39``."""

    if value is None or not value.strip():
        raise ValueError("CPU list must not be empty")

    cpu_ids: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Invalid empty token in CPU list {value!r}")
        if "-" in token:
            if token.count("-") != 1:
                raise ValueError(f"Invalid CPU range {token!r}")
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid CPU range {token!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"CPU range end precedes start in {token!r}")
            cpu_ids.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid CPU id {token!r}")
            cpu_ids.add(int(token))

    if not cpu_ids:
        raise ValueError("CPU list must contain at least one CPU")
    return frozenset(cpu_ids)


def format_cpu_list(cpu_ids: Iterable[int]) -> str:
    """Format CPU ids using Linux's compact range notation."""

    values = sorted(set(cpu_ids))
    if not values:
        return ""
    if values[0] < 0:
        raise ValueError("CPU ids must be non-negative")

    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _parse_npu_smi_topology_output(
    output: str,
) -> dict[int, tuple[str, frozenset[int]]]:
    """Parse validated NPU rows from ``npu-smi info -t topo`` output."""

    header_npu_ids: Optional[tuple[int, ...]] = None
    parsed: dict[int, tuple[str, frozenset[int]]] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()

        if len(tokens) >= 3 and tokens[-2:] == ["CPU", "Affinity"]:
            header_ids: list[int] = []
            for token in tokens[:-2]:
                match = _NPU_ROW_RE.fullmatch(token)
                if match is None:
                    raise NpuAffinityError(
                        f"Invalid npu-smi topology header token {token!r}",
                        stage="parse_npu_smi_header",
                    )
                header_ids.append(int(match.group("id")))
            if not header_ids or len(set(header_ids)) != len(header_ids):
                raise NpuAffinityError(
                    "npu-smi topology header has no NPU ids or contains duplicates",
                    stage="parse_npu_smi_header",
                )
            header_npu_ids = tuple(header_ids)
            continue

        first_match = _NPU_ROW_RE.fullmatch(tokens[0])
        if first_match is None:
            continue
        if len(tokens) < 3:
            raise NpuAffinityError(
                f"Malformed npu-smi topology row: {line!r}",
                stage="parse_npu_smi_row",
            )

        physical_npu_id = int(first_match.group("id"))
        if physical_npu_id in parsed:
            raise NpuAffinityError(
                f"Duplicate NPU id NPU{physical_npu_id} in npu-smi topology",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )

        topology_tokens = tokens[1:-1]
        invalid_tokens = [
            token for token in topology_tokens if token not in _VALID_TOPOLOGY_TOKENS
        ]
        if invalid_tokens:
            raise NpuAffinityError(
                f"Unsupported topology token(s) {invalid_tokens!r} in row {line!r}",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )
        if header_npu_ids is not None and len(topology_tokens) != len(header_npu_ids):
            raise NpuAffinityError(
                f"NPU{physical_npu_id} has {len(topology_tokens)} topology fields; "
                f"expected {len(header_npu_ids)}",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )

        raw_cpu_affinity = tokens[-1]
        try:
            local_cpu_ids = parse_cpu_list(raw_cpu_affinity)
        except ValueError as exc:
            raise NpuAffinityError(
                f"Invalid CPU affinity {raw_cpu_affinity!r} for "
                f"NPU{physical_npu_id}: {exc}",
                stage="parse_npu_smi_cpu_affinity",
                physical_npu_id=physical_npu_id,
            ) from exc
        parsed[physical_npu_id] = (raw_cpu_affinity, local_cpu_ids)

    if not parsed:
        raise NpuAffinityError(
            "No NPU topology rows were found in npu-smi output",
            stage="parse_npu_smi_output",
        )
    if header_npu_ids is not None and set(parsed) != set(header_npu_ids):
        raise NpuAffinityError(
            "npu-smi topology rows do not match the NPU ids declared by the header: "
            f"header={list(header_npu_ids)}, rows={sorted(parsed)}",
            stage="parse_npu_smi_output",
        )
    return parsed


def read_online_cpu_ids() -> frozenset[int]:
    path = Path("/sys/devices/system/cpu/online")
    try:
        return parse_cpu_list(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise NpuAffinityError(
            f"Failed to read online CPUs from {path}: {exc}",
            stage="read_online_cpus",
        ) from exc


def read_numa_cpu_sets() -> dict[int, frozenset[int]]:
    result: dict[int, frozenset[int]] = {}
    root = Path("/sys/devices/system/node")
    try:
        node_paths = sorted(root.glob("node[0-9]*"))
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to enumerate NUMA nodes under {root}: {exc}",
            stage="read_numa_cpus",
        ) from exc

    for node_path in node_paths:
        match = _NUMA_NODE_RE.fullmatch(node_path.name)
        if match is None:
            continue
        cpulist_path = node_path / "cpulist"
        try:
            cpu_ids = parse_cpu_list(cpulist_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise NpuAffinityError(
                f"Failed to read NUMA CPU list from {cpulist_path}: {exc}",
                stage="read_numa_cpus",
            ) from exc
        result[int(match.group("id"))] = cpu_ids

    if not result:
        raise NpuAffinityError(
            f"No NUMA node CPU lists were found under {root}",
            stage="read_numa_cpus",
        )
    return result


def read_lscpu_core_topology() -> dict[int, tuple[int, int, int]]:
    """Return logical CPU -> (core id, socket id, NUMA node)."""

    try:
        proc = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NpuAffinityError(
            f"Failed to execute lscpu: {exc}", stage="query_lscpu"
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1000]
        raise NpuAffinityError(
            f"lscpu exited with code {proc.returncode}: {stderr}",
            stage="query_lscpu",
        )

    result: dict[int, tuple[int, int, int]] = {}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or any(not field.isdigit() for field in fields):
            raise NpuAffinityError(
                f"Malformed lscpu topology row {line!r}",
                stage="parse_lscpu",
            )
        cpu_id, core_id, socket_id, numa_node = map(int, fields)
        if cpu_id in result:
            raise NpuAffinityError(
                f"Duplicate logical CPU {cpu_id} in lscpu output",
                stage="parse_lscpu",
            )
        result[cpu_id] = (core_id, socket_id, numa_node)
    if not result:
        raise NpuAffinityError(
            "lscpu returned no CPU topology rows", stage="parse_lscpu"
        )
    return result


def _infer_unique_numa_node(
    local_cpu_ids: frozenset[int], numa_cpu_sets: dict[int, frozenset[int]]
) -> int:
    candidates = [
        node
        for node, node_cpu_ids in numa_cpu_sets.items()
        if local_cpu_ids <= node_cpu_ids
    ]
    if len(candidates) != 1:
        raise NpuAffinityError(
            "NPU CPU affinity does not map to exactly one NUMA node: "
            f"cpus={format_cpu_list(local_cpu_ids)}, candidates={sorted(candidates)}",
            stage="infer_numa_node",
        )
    return candidates[0]


def _build_npu_topology(
    parsed_rows: dict[int, tuple[str, frozenset[int]]],
    *,
    online_cpu_ids: frozenset[int],
    numa_cpu_sets: dict[int, frozenset[int]],
    lscpu_topology: dict[int, tuple[int, int, int]],
) -> NpuTopology:
    entries: dict[int, NpuTopologyEntry] = {}
    for physical_npu_id, (raw_cpu_affinity, local_cpu_ids) in parsed_rows.items():
        offline_cpu_ids = local_cpu_ids - online_cpu_ids
        if offline_cpu_ids:
            raise NpuAffinityError(
                f"NPU{physical_npu_id} CPU affinity contains offline CPUs: "
                f"{format_cpu_list(offline_cpu_ids)}",
                stage="validate_online_cpus",
                physical_npu_id=physical_npu_id,
            )

        try:
            numa_node = _infer_unique_numa_node(local_cpu_ids, numa_cpu_sets)
        except NpuAffinityError as exc:
            exc.physical_npu_id = physical_npu_id
            raise

        core_to_cpus: dict[tuple[int, int], list[int]] = defaultdict(list)
        for cpu_id in sorted(local_cpu_ids):
            cpu_info = lscpu_topology.get(cpu_id)
            if cpu_info is None:
                raise NpuAffinityError(
                    f"Logical CPU {cpu_id} for NPU{physical_npu_id} is missing "
                    "from lscpu",
                    stage="map_cpu_to_core",
                    physical_npu_id=physical_npu_id,
                )
            core_id, socket_id, cpu_numa_node = cpu_info
            if cpu_numa_node != numa_node:
                raise NpuAffinityError(
                    f"Logical CPU {cpu_id} maps to NUMA {cpu_numa_node} in lscpu but "
                    f"NPU{physical_npu_id} affinity maps to NUMA {numa_node}",
                    stage="map_cpu_to_core",
                    physical_npu_id=physical_npu_id,
                )
            core_to_cpus[(socket_id, core_id)].append(cpu_id)

        physical_cores = tuple(
            PhysicalCore(
                socket_id=socket_id,
                core_id=core_id,
                logical_cpu_ids=tuple(sorted(cpu_ids)),
            )
            for (socket_id, core_id), cpu_ids in sorted(core_to_cpus.items())
        )
        entries[physical_npu_id] = NpuTopologyEntry(
            physical_npu_id=physical_npu_id,
            raw_cpu_affinity=raw_cpu_affinity,
            local_cpu_ids=local_cpu_ids,
            numa_node=numa_node,
            physical_cores=physical_cores,
        )

    return NpuTopology(
        source=_TOPOLOGY_SOURCE,
        entries=entries,
        online_cpu_ids=online_cpu_ids,
    )


@lru_cache(maxsize=1)
def query_npu_smi_topology() -> NpuTopology:
    try:
        proc = subprocess.run(
            ["npu-smi", "info", "-t", "topo"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NpuAffinityError(
            "npu-smi topology query timed out after 10 seconds",
            stage="query_npu_smi",
        ) from exc
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to execute npu-smi: {exc}", stage="query_npu_smi"
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1000]
        raise NpuAffinityError(
            f"npu-smi topology query exited with code {proc.returncode}: {stderr}",
            stage="query_npu_smi",
        )

    parsed_rows = _parse_npu_smi_topology_output(proc.stdout)
    online_cpu_ids = read_online_cpu_ids()
    return _build_npu_topology(
        parsed_rows,
        online_cpu_ids=online_cpu_ids,
        numa_cpu_sets=read_numa_cpu_sets(),
        lscpu_topology=read_lscpu_core_topology(),
    )


def _get_visible_devices() -> tuple[Optional[str], Optional[str]]:
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return name, value.strip()
    return None, None


def resolve_physical_npu_id(logical_npu_id: int) -> int:
    if logical_npu_id < 0:
        raise NpuAffinityError(
            f"Runtime NPU id must be non-negative, got {logical_npu_id}",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )

    env_name, raw_devices = _get_visible_devices()
    if raw_devices is None:
        return logical_npu_id

    tokens = [token.strip() for token in raw_devices.split(",")]
    if any(not token or not token.isdigit() for token in tokens):
        raise NpuAffinityError(
            f"Invalid {env_name}={raw_devices!r}; expected comma-separated "
            "physical NPU ids",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    physical_ids = [int(token) for token in tokens]
    if len(set(physical_ids)) != len(physical_ids):
        raise NpuAffinityError(
            f"Invalid {env_name}={raw_devices!r}; physical NPU ids must be unique",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    if logical_npu_id >= len(physical_ids):
        raise NpuAffinityError(
            f"Runtime NPU id {logical_npu_id} is outside {env_name}={raw_devices!r}",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    return physical_ids[logical_npu_id]


def _get_allowed_cpu_ids() -> frozenset[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise NpuAffinityError(
            "os.sched_getaffinity is unavailable; NPU CPU binding requires Linux",
            stage="read_process_cpuset",
        )
    try:
        return frozenset(os.sched_getaffinity(0))
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to read current process CPU affinity: {exc}",
            stage="read_process_cpuset",
        ) from exc


def _log_topology_once(
    topology: NpuTopology, allowed_cpu_ids: frozenset[int], *, scope: str
) -> None:
    with _TOPOLOGY_LOG_LOCK:
        if scope in _LOGGED_TOPOLOGY_SCOPES:
            return
        _LOGGED_TOPOLOGY_SCOPES.add(scope)

    lines = [
        "NPU affinity topology parsed: "
        f"scope={scope} source={topology.source!r} npu_count={len(topology.entries)} "
        f"online_cpus={format_cpu_list(topology.online_cpu_ids)} "
        f"allowed_cpus={format_cpu_list(allowed_cpu_ids)}"
    ]
    for physical_npu_id, entry in sorted(topology.entries.items()):
        lines.append(
            f"NPU{physical_npu_id}: raw_cpu_affinity={entry.raw_cpu_affinity} "
            f"numa={entry.numa_node} physical_cores={len(entry.physical_cores)} "
            f"logical_cpus={len(entry.local_cpu_ids)}"
        )

    groups: dict[tuple[int, frozenset[int]], list[int]] = defaultdict(list)
    for physical_npu_id, entry in topology.entries.items():
        groups[(entry.numa_node, entry.local_cpu_ids)].append(physical_npu_id)
    for (numa_node, local_cpu_ids), physical_ids in sorted(groups.items()):
        lines.append(
            f"NPU locality group: numa={numa_node} npus={sorted(physical_ids)} "
            f"local_cpus={format_cpu_list(local_cpu_ids)}"
        )
    logger.info("\n".join(lines))


def _build_affinity_assignment(
    topology: NpuTopology,
    *,
    logical_npu_id: int,
    physical_npu_id: int,
    allowed_cpu_ids: frozenset[int],
    requested_pcores: int,
) -> NpuAffinityAssignment:
    entry = topology.entries.get(physical_npu_id)
    if entry is None:
        raise NpuAffinityError(
            f"Physical NPU{physical_npu_id} is absent from npu-smi topology; "
            f"available={sorted(topology.entries)}",
            stage="select_npu_topology",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    candidate_cpu_ids = entry.local_cpu_ids & topology.online_cpu_ids & allowed_cpu_ids
    if not candidate_cpu_ids:
        raise NpuAffinityError(
            f"NPU{physical_npu_id} local CPUs have an empty intersection with the "
            f"current cpuset: local={format_cpu_list(entry.local_cpu_ids)}, "
            f"allowed={format_cpu_list(allowed_cpu_ids)}",
            stage="filter_process_cpuset",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    eligible_cores: list[PhysicalCore] = []
    for core in entry.physical_cores:
        sibling_cpu_ids = tuple(
            cpu_id for cpu_id in core.logical_cpu_ids if cpu_id in candidate_cpu_ids
        )
        if sibling_cpu_ids:
            eligible_cores.append(
                PhysicalCore(core.socket_id, core.core_id, sibling_cpu_ids)
            )

    group_entries = sorted(
        (
            candidate
            for candidate in topology.entries.values()
            if candidate.numa_node == entry.numa_node
            and candidate.local_cpu_ids == entry.local_cpu_ids
        ),
        key=lambda candidate: candidate.physical_npu_id,
    )
    group_ids = [candidate.physical_npu_id for candidate in group_entries]
    try:
        slot_index = group_ids.index(physical_npu_id)
    except ValueError as exc:
        raise NpuAffinityError(
            f"NPU{physical_npu_id} is missing from its locality group",
            stage="assign_locality_slot",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        ) from exc

    slots_on_node = len(group_entries)
    max_non_overlapping = len(eligible_cores) // slots_on_node
    if max_non_overlapping <= 0:
        raise NpuAffinityError(
            f"Only {len(eligible_cores)} eligible physical cores are available for "
            f"{slots_on_node} NPUs in NUMA node {entry.numa_node}",
            stage="assign_locality_slot",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    if requested_pcores == 0:
        effective_pcores = max_non_overlapping
    else:
        effective_pcores = min(requested_pcores, max_non_overlapping)
        if requested_pcores > max_non_overlapping:
            logger.warning(
                "SGLANG_NPU_AFFINITY_PCORES_PER_PROC=%s exceeds the non-overlapping "
                "limit %s for NPU%s; clipping to %s",
                requested_pcores,
                max_non_overlapping,
                physical_npu_id,
                effective_pcores,
            )

    start = slot_index * effective_pcores
    selected_cores = eligible_cores[start : start + effective_pcores]
    if len(selected_cores) != effective_pcores:
        raise NpuAffinityError(
            f"Failed to select {effective_pcores} physical cores for "
            f"NPU{physical_npu_id} "
            f"at locality slot {slot_index}/{slots_on_node}",
            stage="assign_physical_cores",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    logical_cpu_ids = tuple(
        sorted(
            cpu_id
            for core in selected_cores
            for cpu_id in core.logical_cpu_ids
        )
    )
    return NpuAffinityAssignment(
        logical_npu_id=logical_npu_id,
        physical_npu_id=physical_npu_id,
        numa_node=entry.numa_node,
        slot_index=slot_index,
        slots_on_node=slots_on_node,
        requested_pcores=requested_pcores,
        effective_pcores=effective_pcores,
        physical_core_keys=tuple(
            (core.socket_id, core.core_id) for core in selected_cores
        ),
        logical_cpu_ids=logical_cpu_ids,
        topology_source=topology.source,
    )


def resolve_npu_affinity_assignment(
    *,
    logical_npu_id: int,
    emit_topology_log: bool,
) -> NpuAffinityAssignment:
    requested_pcores = envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.get()
    if requested_pcores < 0:
        raise ValueError(
            "SGLANG_NPU_AFFINITY_PCORES_PER_PROC must be >= 0"
        )

    physical_npu_id = resolve_physical_npu_id(logical_npu_id)
    try:
        topology = query_npu_smi_topology()
        allowed_cpu_ids = _get_allowed_cpu_ids()
    except NpuAffinityError as exc:
        if exc.logical_npu_id is None:
            exc.logical_npu_id = logical_npu_id
        if exc.physical_npu_id is None:
            exc.physical_npu_id = physical_npu_id
        raise
    if emit_topology_log:
        _log_topology_once(topology, allowed_cpu_ids, scope="launcher")

    assignment = _build_affinity_assignment(
        topology,
        logical_npu_id=logical_npu_id,
        physical_npu_id=physical_npu_id,
        allowed_cpu_ids=allowed_cpu_ids,
        requested_pcores=requested_pcores,
    )
    env_name, visible_devices = _get_visible_devices()
    logger.info(
        "NPU affinity assignment resolved: runtime_npu=%s physical_npu=%s "
        "visible_devices=%s numa=%s slot=%s/%s requested_pcores=%s "
        "effective_pcores=%s physical_cores=%s logical_cpus=%s",
        assignment.logical_npu_id,
        assignment.physical_npu_id,
        f"{env_name}={visible_devices}" if env_name else "unset",
        assignment.numa_node,
        assignment.slot_index,
        assignment.slots_on_node,
        assignment.requested_pcores,
        assignment.effective_pcores,
        assignment.physical_core_keys,
        format_cpu_list(assignment.logical_cpu_ids),
    )
    return assignment


def apply_npu_cpu_affinity(
    assignment: NpuAffinityAssignment,
    *,
    phase: Literal["early", "final"],
    bind_all_threads: bool,
) -> NpuAffinityApplyResult:
    if phase not in ("early", "final"):
        raise ValueError(f"Unsupported NPU affinity phase {phase!r}")
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise NpuAffinityError(
            "Linux sched affinity APIs are unavailable",
            stage="apply_cpu_affinity",
            logical_npu_id=assignment.logical_npu_id,
            physical_npu_id=assignment.physical_npu_id,
        )

    try:
        current_allowed = set(os.sched_getaffinity(0))
        target_cpu_ids = set(assignment.logical_cpu_ids) & current_allowed
        if not target_cpu_ids:
            raise NpuAffinityError(
                "NPU affinity target became empty after intersecting the "
                "current cpuset: "
                f"requested={format_cpu_list(assignment.logical_cpu_ids)}, "
                f"current={format_cpu_list(current_allowed)}",
                stage="apply_cpu_affinity",
                logical_npu_id=assignment.logical_npu_id,
                physical_npu_id=assignment.physical_npu_id,
            )
        os.sched_setaffinity(0, target_cpu_ids)
        main_actual_cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
    except NpuAffinityError:
        raise
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to bind scheduler main thread: {exc}",
            stage="apply_cpu_affinity",
            logical_npu_id=assignment.logical_npu_id,
            physical_npu_id=assignment.physical_npu_id,
        ) from exc

    target_tuple = tuple(sorted(target_cpu_ids))
    main_matched = main_actual_cpu_ids == target_tuple
    if not main_matched:
        logger.warning(
            "NPU affinity main-thread read-back mismatch: requested=%s actual=%s",
            format_cpu_list(target_tuple),
            format_cpu_list(main_actual_cpu_ids),
        )

    threads_total = 0
    threads_bound = 0
    threads_exited = 0
    threads_failed = 0
    threads_mismatched = 0
    if bind_all_threads:
        try:
            threads = psutil.Process(os.getpid()).threads()
        except (psutil.Error, OSError) as exc:
            logger.warning(
                "Failed to enumerate scheduler threads for NPU affinity: %s", exc
            )
            threads = []
            threads_failed += 1
        threads_total = len(threads)
        for thread in threads:
            tid = thread.id
            try:
                os.sched_setaffinity(tid, target_cpu_ids)
                actual_cpu_ids = tuple(sorted(os.sched_getaffinity(tid)))
                threads_bound += 1
                if actual_cpu_ids != target_tuple:
                    threads_mismatched += 1
                    logger.warning(
                        "NPU affinity thread read-back mismatch: tid=%s "
                        "requested=%s actual=%s",
                        tid,
                        format_cpu_list(target_tuple),
                        format_cpu_list(actual_cpu_ids),
                    )
                else:
                    logger.debug(
                        "NPU affinity thread bound: tid=%s cpus=%s",
                        tid,
                        format_cpu_list(actual_cpu_ids),
                    )
            except ProcessLookupError:
                threads_exited += 1
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    threads_exited += 1
                else:
                    threads_failed += 1
                    logger.warning(
                        "Failed to bind scheduler thread tid=%s to CPUs %s: %s",
                        tid,
                        format_cpu_list(target_tuple),
                        exc,
                    )

    result = NpuAffinityApplyResult(
        requested_cpu_ids=target_tuple,
        main_actual_cpu_ids=main_actual_cpu_ids,
        main_matched=main_matched,
        threads_total=threads_total,
        threads_bound=threads_bound,
        threads_exited=threads_exited,
        threads_failed=threads_failed,
        threads_mismatched=threads_mismatched,
    )
    logger.info(
        "NPU affinity applied: pid=%s phase=%s runtime_npu=%s physical_npu=%s "
        "numa=%s slot=%s/%s requested_cpus=%s main_actual_cpus=%s "
        "main_matched=%s bind_all_threads=%s threads_total=%s threads_bound=%s "
        "threads_exited=%s threads_failed=%s threads_mismatched=%s",
        os.getpid(),
        phase,
        assignment.logical_npu_id,
        assignment.physical_npu_id,
        assignment.numa_node,
        assignment.slot_index,
        assignment.slots_on_node,
        format_cpu_list(result.requested_cpu_ids),
        format_cpu_list(result.main_actual_cpu_ids),
        result.main_matched,
        bind_all_threads,
        result.threads_total,
        result.threads_bound,
        result.threads_exited,
        result.threads_failed,
        result.threads_mismatched,
    )
    return result


def format_npu_affinity_error(
    exc: NpuAffinityError,
    *,
    fallback: Literal[
        "defer_to_final", "legacy_gpu_affinity", "skip_memory_bind"
    ],
) -> str:
    return (
        f"topology_source={_TOPOLOGY_SOURCE!r} "
        f"runtime_npu={exc.logical_npu_id} physical_npu={exc.physical_npu_id} "
        f"failure_stage={exc.stage} reason={exc} fallback={fallback}"
    )
