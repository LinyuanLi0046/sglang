import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.greedy_trace import (
    _extract_full_mask_tails,
    _topk_snapshot,
    build_topk1_reference,
    capture_raw_logits,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEagleMTPGreedyTrace(unittest.TestCase):
    def test_disabled_trace_does_not_resolve_distributed_group(self):
        with envs.SGLANG_NPU_MTP_GREEDY_TRACE.override(False), patch(
            "sglang.srt.speculative.greedy_trace._trace_group"
        ) as trace_group:
            snapshot = capture_raw_logits(
                None, None, possible_tokens_per_req=1
            )

        self.assertIsNone(snapshot)
        trace_group.assert_not_called()

    def test_topk_snapshot_clamps_topn_to_vocab(self):
        with envs.SGLANG_NPU_MTP_GREEDY_TRACE_TOPN.override(8):
            snapshot = _topk_snapshot(torch.tensor([[1.0, 3.0, 2.0]]))

        self.assertEqual(snapshot.token_ids.tolist(), [[1, 2, 0]])
        self.assertEqual(snapshot.values.tolist(), [[3.0, 2.0, 1.0]])

    def test_topk1_reference_covers_reject_partial_and_full_accept(self):
        candidates = torch.tensor(
            [
                [100, 11, 12, 13, 14],
                [200, 21, 22, 23, 24],
                [300, 31, 32, 33, 34],
            ],
            dtype=torch.int64,
        )
        target_predict = torch.tensor(
            [
                [99, 12, 13, 14, 15],
                [21, 22, 999, 24, 25],
                [31, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
        )
        retrieve_index = torch.arange(15, dtype=torch.int64).reshape(3, 5)
        inputs_before = tuple(
            tensor.clone()
            for tensor in (candidates, target_predict, retrieve_index)
        )

        result = build_topk1_reference(
            candidates, retrieve_index, target_predict
        )

        self.assertEqual(result.accept_lens, [1, 3, 5])
        self.assertEqual(
            result.accept_indices,
            [[0], [5, 6, 7], [10, 11, 12, 13, 14]],
        )
        self.assertEqual(
            result.accept_tokens,
            [[99], [21, 22, 999], [31, 32, 33, 34, 35]],
        )
        self.assertEqual(
            result.predict,
            [
                99,
                0,
                0,
                0,
                0,
                21,
                22,
                999,
                0,
                0,
                31,
                32,
                33,
                34,
                35,
            ],
        )
        for tensor, before in zip(
            (candidates, target_predict, retrieve_index), inputs_before
        ):
            self.assertTrue(torch.equal(tensor, before))

    def test_topk1_reference_step_zero_returns_only_bonus(self):
        result = build_topk1_reference(
            candidates=torch.tensor([[123], [456]]),
            retrieve_index=torch.tensor([[0], [1]]),
            target_predict=torch.tensor([[10], [20]]),
        )

        self.assertEqual(result.accept_lens, [1, 1])
        self.assertEqual(result.accept_indices, [[0], [1]])
        self.assertEqual(result.accept_tokens, [[10], [20]])
        self.assertEqual(result.predict, [10, 20])

    def test_topk1_reference_rejects_cross_request_retrieve_index(self):
        with self.assertRaisesRegex(ValueError, "outside request row"):
            build_topk1_reference(
                candidates=torch.tensor([[1, 2], [3, 4]]),
                retrieve_index=torch.tensor([[0, 2], [2, 3]]),
                target_predict=torch.tensor([[2, 5], [4, 6]]),
            )

    def test_extract_full_mask_tails_handles_batch_offsets(self):
        # bs=2, D=2, seq_lens=[2, 3].  The A5 FULL_MASK layout places the
        # four D-wide tails at [2:4], [6:8], [11:13], and [16:18].
        custom_mask = torch.zeros(18, dtype=torch.bool)
        custom_mask[2:4] = torch.tensor([True, False])
        custom_mask[6:8] = torch.tensor([True, True])
        custom_mask[11:13] = torch.tensor([False, True])
        custom_mask[16:18] = torch.tensor([True, False])

        tails = _extract_full_mask_tails(custom_mask, [2, 3], width=2)

        self.assertEqual(
            tails,
            [
                [[True, False], [True, True]],
                [[False, True], [True, False]],
            ],
        )


if __name__ == "__main__":
    unittest.main()

