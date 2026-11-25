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


if __name__ == "__main__":
    unittest.main()
