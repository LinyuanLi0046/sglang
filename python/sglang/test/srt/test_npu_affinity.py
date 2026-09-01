import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import call, patch

from sglang.srt.environ import envs
from sglang.srt.utils.npu_affinity import (
    NpuAffinityAssignment,
    NpuAffinityError,
    NpuTopology,
    NpuTopologyEntry,
    PhysicalCore,
    _build_affinity_assignment,
    _build_npu_topology,
    _infer_unique_numa_node,
    _parse_npu_smi_topology_output,
    apply_npu_cpu_affinity,
    format_cpu_list,
    parse_cpu_list,
    resolve_physical_npu_id,
)


NPU_SMI_TOPOLOGY = """\
NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7 CPU Affinity
NPU0 X UB UB UB SYS SYS SYS SYS 0-95,192-287
NPU1 UB X UB UB SYS SYS SYS SYS 0-95,192-287
NPU2 UB UB X UB SYS SYS SYS SYS 0-95,192-287
NPU3 UB UB UB X SYS SYS SYS SYS 0-95,192-287
NPU4 SYS SYS SYS SYS X UB UB UB 96-191,288-383
NPU5 SYS SYS SYS SYS UB X UB UB 96-191,288-383
NPU6 SYS SYS SYS SYS UB UB X UB 96-191,288-383
NPU7 SYS SYS SYS SYS UB UB UB X 96-191,288-383

Legend:
X = Self
SYS = Path traversing PCIe and NUMA nodes.
UB = Connection traversing UB.
"""


def _make_test_topology() -> NpuTopology:
    entries = {}
    online_cpus = set()
    for node in range(2):
        first_core = node * 8
        local_cpus = set()
        physical_cores = []
        for core_id in range(first_core, first_core + 8):
            siblings = (core_id, core_id + 16)
            local_cpus.update(siblings)
            physical_cores.append(
                PhysicalCore(
                    socket_id=node,
                    core_id=core_id,
                    logical_cpu_ids=siblings,
                )
            )
        online_cpus.update(local_cpus)
        for physical_npu_id in range(node * 4, node * 4 + 4):
            entries[physical_npu_id] = NpuTopologyEntry(
                physical_npu_id=physical_npu_id,
                raw_cpu_affinity=format_cpu_list(local_cpus),
                local_cpu_ids=frozenset(local_cpus),
                numa_node=node,
                physical_cores=tuple(physical_cores),
            )
    return NpuTopology(
        source="test",
        entries=entries,
        online_cpu_ids=frozenset(online_cpus),
    )


class TestCpuList(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_cpu_list("0,2,4-7"), frozenset({0, 2, 4, 5, 6, 7}))
        self.assertEqual(format_cpu_list({0, 1, 2, 5, 7, 8}), "0-2,5,7-8")

    def test_invalid_cpu_lists(self):
        for value in ("", "-1", "4-2", "a", "1,,2", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cpu_list(value)


class TestNpuSmiParser(unittest.TestCase):
    def test_parse_current_machine_shape(self):
        parsed = _parse_npu_smi_topology_output(NPU_SMI_TOPOLOGY)
        self.assertEqual(sorted(parsed), list(range(8)))
        self.assertEqual(parsed[0][1], parse_cpu_list("0-95,192-287"))
        self.assertEqual(parsed[7][1], parse_cpu_list("96-191,288-383"))

    def test_reject_unknown_topology_token(self):
        text = "NPU0 CPU Affinity\nNPU0 BAD 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "Unsupported topology"):
            _parse_npu_smi_topology_output(text)

    def test_reject_duplicate_or_missing_npu_rows(self):
        duplicate = "NPU0 CPU Affinity\nNPU0 X 0-3\nNPU0 X 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "Duplicate NPU"):
            _parse_npu_smi_topology_output(duplicate)

        missing = "NPU0 NPU1 CPU Affinity\nNPU0 X UB 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "do not match"):
            _parse_npu_smi_topology_output(missing)


class TestTopologyConstruction(unittest.TestCase):
    def test_current_machine_numa_and_core_mapping(self):
        parsed = _parse_npu_smi_topology_output(NPU_SMI_TOPOLOGY)
        node0 = parse_cpu_list("0-95,192-287")
        node1 = parse_cpu_list("96-191,288-383")
        lscpu_topology = {}
        for cpu_id in range(384):
            if cpu_id < 96:
                core_id, socket_id, node = cpu_id, 0, 0
            elif cpu_id < 192:
                core_id, socket_id, node = cpu_id, 1, 1
            elif cpu_id < 288:
                core_id, socket_id, node = cpu_id - 192, 0, 0
            else:
                core_id, socket_id, node = cpu_id - 192, 1, 1
            lscpu_topology[cpu_id] = (core_id, socket_id, node)

        topology = _build_npu_topology(
            parsed,
            online_cpu_ids=frozenset(range(384)),
            numa_cpu_sets={0: node0, 1: node1},
            lscpu_topology=lscpu_topology,
        )
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=4,
            physical_npu_id=4,
            allowed_cpu_ids=frozenset(range(384)),
            requested_pcores=8,
        )
        self.assertEqual(topology.entries[0].numa_node, 0)
        self.assertEqual(topology.entries[7].numa_node, 1)
        self.assertEqual(
            assignment.physical_core_keys,
            tuple((1, i) for i in range(96, 104)),
        )
        self.assertEqual(
            assignment.logical_cpu_ids,
            tuple(range(96, 104)) + tuple(range(288, 296)),
        )

    def test_npu_cpu_set_must_belong_to_exactly_one_numa_node(self):
        with self.assertRaisesRegex(NpuAffinityError, "exactly one"):
            _infer_unique_numa_node(
                frozenset({0, 4}),
                {0: frozenset(range(4)), 1: frozenset(range(4, 8))},
            )


class TestVisibleDeviceMapping(unittest.TestCase):
    def test_visible_device_reordering(self):
        with patch.dict(
            os.environ,
            {"ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7"},
            clear=False,
        ):
            self.assertEqual(resolve_physical_npu_id(0), 4)
            self.assertEqual(resolve_physical_npu_id(3), 7)
            with self.assertRaisesRegex(NpuAffinityError, "outside"):
                resolve_physical_npu_id(4)

    def test_no_visible_device_filter(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_physical_npu_id(4), 4)


class TestAssignment(unittest.TestCase):
    def test_assignments_are_local_and_non_overlapping(self):
        topology = _make_test_topology()
        assignments = [
            _build_affinity_assignment(
                topology,
                logical_npu_id=npu_id,
                physical_npu_id=npu_id,
                allowed_cpu_ids=topology.online_cpu_ids,
                requested_pcores=2,
            )
            for npu_id in range(8)
        ]
        for assignment in assignments:
            entry = topology.entries[assignment.physical_npu_id]
            self.assertEqual(assignment.effective_pcores, 2)
            self.assertTrue(set(assignment.logical_cpu_ids) <= entry.local_cpu_ids)
        for index, lhs in enumerate(assignments):
            for rhs in assignments[index + 1 :]:
                self.assertFalse(
                    set(lhs.logical_cpu_ids) & set(rhs.logical_cpu_ids),
                    (lhs, rhs),
                )

    def test_zero_means_even_division_and_large_request_is_clipped(self):
        topology = _make_test_topology()
        automatic = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=0,
        )
        clipped = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=99,
        )
        self.assertEqual(automatic.effective_pcores, 2)
        self.assertEqual(clipped.effective_pcores, 2)

    def test_cpuset_keeps_only_available_siblings(self):
        topology = _make_test_topology()
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=frozenset(range(8)),
            requested_pcores=2,
        )
        self.assertEqual(assignment.logical_cpu_ids, (0, 1))

    def test_negative_pcore_setting_is_rejected(self):
        from sglang.srt.utils import npu_affinity

        with envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.override(-1):
            with self.assertRaisesRegex(ValueError, "must be >= 0"):
                npu_affinity.resolve_npu_affinity_assignment(
                    logical_npu_id=0,
                    emit_topology_log=False,
                )


class TestApplyAffinity(unittest.TestCase):
    def test_final_bind_updates_all_threads_and_reads_back(self):
        assignment = NpuAffinityAssignment(
            logical_npu_id=0,
            physical_npu_id=0,
            numa_node=0,
            slot_index=0,
            slots_on_node=1,
            requested_pcores=1,
            effective_pcores=1,
            physical_core_keys=((0, 0),),
            logical_cpu_ids=(0, 16),
            topology_source="test",
        )
        affinity_by_tid = {0: {0, 16}, 101: {0, 16}, 102: {0, 16}}

        def get_affinity(tid):
            return affinity_by_tid[tid]

        def set_affinity(tid, cpus):
            affinity_by_tid[tid] = set(cpus)

        process = SimpleNamespace(
            threads=lambda: [SimpleNamespace(id=101), SimpleNamespace(id=102)]
        )
        with (
            patch.object(
                os, "sched_getaffinity", side_effect=get_affinity, create=True
            ),
            patch.object(
                os, "sched_setaffinity", side_effect=set_affinity, create=True
            ),
            patch(
                "sglang.srt.utils.npu_affinity.psutil.Process",
                return_value=process,
            ),
        ):
            result = apply_npu_cpu_affinity(
                assignment,
                phase="final",
                bind_all_threads=True,
            )

        self.assertTrue(result.main_matched)
        self.assertEqual(result.threads_total, 2)
        self.assertEqual(result.threads_bound, 2)
        self.assertEqual(result.threads_failed, 0)
        self.assertEqual(result.threads_mismatched, 0)


class TestNumaSubprocessPolicy(unittest.TestCase):
    def setUp(self):
        self.assignment = NpuAffinityAssignment(
            logical_npu_id=0,
            physical_npu_id=4,
            numa_node=1,
            slot_index=0,
            slots_on_node=4,
            requested_pcores=8,
            effective_pcores=8,
            physical_core_keys=tuple((1, core) for core in range(96, 104)),
            logical_cpu_ids=tuple(range(96, 104)) + tuple(range(288, 296)),
            topology_source="test",
        )
        self.server_args = SimpleNamespace(device="npu", numa_node=None)

    @contextmanager
    def _common_patches(self):
        from sglang.srt.utils import numa_utils

        with (
            patch.object(
                numa_utils,
                "resolve_npu_affinity_assignment",
                return_value=self.assignment,
            ),
            patch.object(
                numa_utils,
                "_create_numactl_executable",
                return_value=("wrapper", "debug"),
            ),
            patch.object(
                numa_utils,
                "_mp_set_executable",
                side_effect=lambda **_kwargs: nullcontext(),
            ),
        ):
            yield

    def test_v2_without_actual_node_falls_through_to_npu_preferred(self):
        from sglang.srt.utils import numa_utils

        with (
            envs.SGLANG_SET_CPU_AFFINITY.override(False),
            envs.SGLANG_NPU_MEMORY_PREFERRED_BIND.override(True),
            envs.SGLANG_NUMA_BIND_V2.override(True),
            self._common_patches(),
            patch.object(
                numa_utils,
                "get_numa_node_if_available",
                return_value=None,
            ),
            patch.object(
                numa_utils,
                "_probe_numactl_args",
                return_value=("--preferred=1", ""),
            ) as probe,
        ):
            with numa_utils.configure_subprocess(self.server_args, 0):
                pass
        probe.assert_called_once_with("--preferred=1")

    def test_successful_v2_does_not_add_preferred_wrapper(self):
        from sglang.srt.utils import numa_utils

        with (
            envs.SGLANG_SET_CPU_AFFINITY.override(False),
            envs.SGLANG_NPU_MEMORY_PREFERRED_BIND.override(True),
            envs.SGLANG_NUMA_BIND_V2.override(True),
            self._common_patches(),
            patch.object(
                numa_utils,
                "get_numa_node_if_available",
                return_value=1,
            ),
            patch.object(
                numa_utils,
                "_numactl_cpu_mem_args",
                return_value="--cpunodebind=1 --membind=1",
            ),
            patch.object(
                numa_utils,
                "_probe_numactl_args",
                return_value=("--cpunodebind=1 --membind=1", ""),
            ) as probe,
        ):
            with numa_utils.configure_subprocess(self.server_args, 0):
                pass
        probe.assert_called_once_with("--cpunodebind=1 --membind=1")

    def test_failed_v2_can_fall_through_to_npu_preferred(self):
        from sglang.srt.utils import numa_utils

        with (
            envs.SGLANG_SET_CPU_AFFINITY.override(False),
            envs.SGLANG_NPU_MEMORY_PREFERRED_BIND.override(True),
            envs.SGLANG_NUMA_BIND_V2.override(True),
            self._common_patches(),
            patch.object(
                numa_utils,
                "get_numa_node_if_available",
                return_value=1,
            ),
            patch.object(
                numa_utils,
                "_numactl_cpu_mem_args",
                return_value="--cpunodebind=1 --membind=1",
            ),
            patch.object(numa_utils, "_handle_numa_bind_failure"),
            patch.object(
                numa_utils,
                "_probe_numactl_args",
                side_effect=[(None, "denied"), ("--preferred=1", "")],
            ) as probe,
        ):
            with numa_utils.configure_subprocess(self.server_args, 0):
                pass
        self.assertEqual(
            probe.call_args_list,
            [
                call("--cpunodebind=1 --membind=1"),
                call("--preferred=1"),
            ],
        )

    def test_explicit_numa_mismatch_is_rejected_even_when_v2_is_disabled(self):
        from sglang.srt.utils import numa_utils

        server_args = SimpleNamespace(device="npu", numa_node=[0])
        with (
            envs.SGLANG_SET_CPU_AFFINITY.override(True),
            envs.SGLANG_NPU_MEMORY_PREFERRED_BIND.override(False),
            envs.SGLANG_NUMA_BIND_V2.override(False),
            patch.object(
                numa_utils,
                "resolve_npu_affinity_assignment",
                return_value=self.assignment,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "conflicts with NPU topology"):
                with numa_utils.configure_subprocess(server_args, 0):
                    pass


if __name__ == "__main__":
    unittest.main()
