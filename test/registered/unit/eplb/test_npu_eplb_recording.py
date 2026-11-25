"""CPU checks for NPU EPLB statistics; no server or NPU is started."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuEplbRecording(unittest.TestCase):
    def recorder(self):
        recorder = object.__new__(distribution._ExpertDistributionRecorderReal)
        recorder._recording = True
        recorder._disable_all = False
        recorder._on_hook = Mock()
        return recorder

    def test_segmented_padding_does_not_mutate_dispatch_ids(self):
        recorder = self.recorder()
        ids = torch.tensor([[3, 1], [2, 0], [0, 2], [1, 3]], dtype=torch.int32)
        original = ids.clone()
        invalid = torch.tensor([False, True, False, True])
        recorder.on_select_experts(
            ids, invalid_row_mask=invalid, num_token_non_padded=torch.tensor(1)
        )
        recorded = recorder._on_hook.call_args.kwargs["topk_ids"]
        torch.testing.assert_close(ids, original)
        torch.testing.assert_close(
            recorded, torch.tensor([[3, 1], [-1, -1], [0, 2], [-1, -1]], dtype=ids.dtype)
        )

    def test_prefix_and_idle_rows(self):
        recorder = self.recorder()
        ids = torch.tensor([[2, 0], [1, 3], [0, 1]], dtype=torch.int32)
        for count in (0, 2, 3):
            with self.subTest(count=count):
                recorder.on_select_experts(ids, num_token_non_padded=torch.tensor(count))
                recorded = recorder._on_hook.call_args.kwargs["topk_ids"]
                self.assertTrue(torch.equal(recorded[:count], ids[:count]))
                self.assertTrue(torch.all(recorded[count:] == -1))

    def test_draft_pass_does_not_reset_or_publish_target_counts(self):
        recorder = self.recorder()
        recorder._on_forward_pass_start = Mock()
        recorder._on_forward_pass_end = Mock()
        with recorder.disable_this_region():
            with recorder.with_forward_pass(7, SimpleNamespace()) as outputs:
                self.assertEqual(outputs, {})
            recorder.on_select_experts(
                torch.tensor([[0]]), invalid_row_mask=torch.tensor([True])
            )
        recorder._on_forward_pass_start.assert_not_called()
        recorder._on_forward_pass_end.assert_not_called()
        recorder._on_hook.assert_not_called()
        self.assertFalse(recorder._disable_all)

    def test_async_counts_keep_old_mapping_across_ring_wrap(self):
        from sglang.srt.hardware_backend.npu.eplb.manager import _AsyncStatAccumulator

        accumulator = object.__new__(_AsyncStatAccumulator)
        accumulator._expert_location_metadata = SimpleNamespace(
            num_layers=1,
            num_logical_experts=3,
            physical_to_logical_map=torch.tensor([[0, 1, 2]]),
        )
        accumulator.window_samples = 6
        accumulator._previous_layouts = {0: (4, torch.tensor([1, 0, 2]))}
        accumulator._global_physical_count_of_buffered_step = SimpleNamespace(
            _curr_index=2
        )
        # The last two writes wrapped to slots 0/1 under the new placement.
        physical = torch.tensor([[[1, 10, 0]], [[2, 20, 0]], [[3, 30, 0]], [[4, 40, 0]]])
        expected = torch.tensor([[[1, 10, 0]], [[2, 20, 0]], [[30, 3, 0]], [[40, 4, 0]]])
        torch.testing.assert_close(accumulator._logical_counts(physical), expected)

    def test_npu_uses_valid_topk_counts_cuda_ll_keeps_receive_counts(self):
        with (
            patch.object(distribution, "_SelectExpertsSinglePassGatherer") as topk,
            patch.object(distribution, "_DeepepLowLatencySinglePassGatherer") as receive,
        ):
            for device, mode in (
                ("npu", "normal"), ("npu", "low_latency"), ("npu", "auto"),
                ("cuda", "low_latency"),
            ):
                args = SimpleNamespace(
                    device=device,
                    expert_distribution_recorder_mode="stat",
                    moe_a2a_backend="deepep",
                    deepep_mode=mode,
                    elastic_ep_backend=None,
                )
                result = distribution._SinglePassGatherer.init_new(args, Mock(), 0)
                self.assertIs(result, (topk if device == "npu" else receive).return_value)

    def test_async_approximate_counts_preserve_sources(self):
        with (
            patch.object(distribution, "_SelectExpertsSinglePassGatherer") as topk,
            patch.object(distribution, "_DeepepNormalSinglePassGatherer") as receive,
        ):
            for device, backend, mode, asynchronous, expect_topk in (
                ("npu", "deepep", "normal", True, True),
                ("npu", "deepep", "low_latency", True, True),
                ("npu", "deepep", "auto", True, True),
                ("npu", "ascend_fuseep", "auto", False, True),
                ("npu", "deepep", "normal", False, False),
                ("cuda", "deepep", "normal", False, False),
            ):
                with self.subTest(device=device, backend=backend, mode=mode):
                    args = SimpleNamespace(
                        device=device,
                        expert_distribution_recorder_mode="stat_approx",
                        moe_a2a_backend=backend,
                        deepep_mode=mode,
                        eplb_async_rebalance=asynchronous,
                    )
                    result = distribution._SinglePassGatherer.init_new(args, Mock(), 0)
                    expected = topk if expect_topk else receive
                    self.assertIs(result, expected.return_value)

    def test_async_snapshot_checks_threshold_before_reductions(self):
        from sglang.srt.hardware_backend.npu.eplb.manager import AsyncEplbRecorder

        recorder = object.__new__(AsyncEplbRecorder)
        recorder._accumulator = Mock()
        recorder._reset = Mock()
        recorder._accumulator._get_global_average_utilization_rate.return_value = 0.96
        spec = SimpleNamespace()
        decision = Mock(return_value=False)
        self.assertIsNone(recorder.snapshot_for_rebalance(spec, decision))
        decision.assert_called_once_with(0.96)
        recorder._accumulator.snapshot_counts.assert_not_called()
        recorder._reset.assert_called_once_with()

        recorder._reset.reset_mock()
        decision.return_value = True
        snapshot = recorder.snapshot_for_rebalance(spec, decision)
        self.assertIs(snapshot, recorder._accumulator.snapshot_counts.return_value)
        recorder._accumulator.snapshot_counts.assert_called_once_with(spec)
        recorder._reset.assert_called_once_with()

    def test_async_snapshot_owns_counts_and_preserves_source_group(self):
        from sglang.srt.hardware_backend.npu.eplb import manager as eplb_manager

        accumulator = object.__new__(eplb_manager._AsyncStatAccumulator)
        accumulator._previous_layouts = {}
        accumulator._communication_group = object()
        buffered = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
        accumulator._global_physical_count_of_buffered_step = Mock()
        accumulator._global_physical_count_of_buffered_step.get_all.return_value = buffered
        spec = SimpleNamespace(physical_count_mode="static", source_groups=2)
        with (
            patch.object(eplb_manager.dist, "get_world_size", return_value=4),
            patch.object(eplb_manager.dist, "get_rank", return_value=2),
            patch.object(eplb_manager.dist, "reduce") as reduce_sources,
            patch.object(eplb_manager.dist, "all_reduce") as reduce_counts,
        ):
            counts, sources = accumulator.snapshot_counts(spec)
            reduce_sources.assert_called_once_with(
                sources, dst=0, group=accumulator._communication_group
            )
            reduce_counts.assert_called_once_with(
                counts, group=accumulator._communication_group
            )
        buffered.zero_()
        torch.testing.assert_close(
            counts, torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            sources, torch.tensor([[[0, 0]], [[4, 6]]], dtype=torch.int64)
        )

    def test_routing_only_update_does_not_pack_or_transfer_weights(self):
        from sglang.srt.hardware_backend.npu.eplb import runtime as eplb_async

        runtime = object.__new__(eplb_async.NpuEplbRuntime)
        physical = torch.tensor([[0, 1, 0, 1]])
        runtime.metadata = SimpleNamespace(physical_to_logical_map_cpu=physical)
        routing = SimpleNamespace(
            host_rows={0: {"physical_to_logical_map": physical[0].clone()}}
        )
        # No weight views or staging buffers are needed on this path.
        with patch.object(eplb_async, "update_expert_weights_single_layer") as update:
            writeback = runtime._stage_weights(0, routing, [])
            writeback()
            update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
