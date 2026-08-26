import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.models import welmv4
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeModel,
    Qwen2MoeSparseMoeBlock,
)
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestWeLMv4DeepEPLayout(unittest.TestCase):
    def test_prompt_padding_localizes_real_rows_for_tp4(self):
        real_rows = 16489
        padded_rows = ceil_align(real_rows, 4)
        self.assertEqual(padded_rows, 16492)

        local_real_rows = []
        for tp_rank in range(4):
            with (
                patch.object(
                    forward_batch_info,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        attn_tp_size=4, attn_tp_rank=tp_rank
                    ),
                ),
                patch.object(
                    forward_batch_info,
                    "_attn_tp_sequence_sharded_predicate",
                    None,
                ),
            ):
                count = forward_batch_info.compute_local_num_token_non_padded(
                    torch.tensor(real_rows, dtype=torch.int32),
                    num_tokens_per_dp=padded_rows,
                )
            local_real_rows.append(count.item())

        self.assertEqual(local_real_rows, [4123, 4123, 4123, 4120])

    def test_npu_normal_topk_padding_masks_ids_and_weights(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((4, 2), dtype=torch.float32),
            topk_ids=torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
            router_logits=torch.randn((4, 8)),
        )
        masked = Qwen2MoeSparseMoeBlock._mask_npu_padded_topk(
            topk_output, torch.tensor(3, dtype=torch.int32)
        )
        torch.testing.assert_close(masked.topk_weights[:3], torch.ones((3, 2)))
        torch.testing.assert_close(masked.topk_weights[3], torch.zeros(2))
        torch.testing.assert_close(masked.topk_ids[:3], topk_output.topk_ids[:3])
        torch.testing.assert_close(masked.topk_ids[3], torch.full((2,), -1))

    def test_npu_low_latency_topk_padding_preserves_ids_and_masks_weights(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((4, 2), dtype=torch.float32),
            topk_ids=torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
            router_logits=torch.randn((4, 8)),
        )
        masked = Qwen2MoeSparseMoeBlock._mask_npu_padded_topk(
            topk_output,
            torch.tensor(3, dtype=torch.int32),
            preserve_padded_ids=True,
        )
        torch.testing.assert_close(masked.topk_weights[:3], torch.ones((3, 2)))
        torch.testing.assert_close(masked.topk_weights[3], torch.zeros(2))
        torch.testing.assert_close(masked.topk_ids, topk_output.topk_ids)

    def test_npu_topk_padding_policy_resolves_deepep_mode(self):
        with (
            patch.object(welmv4, "get_deepep_mode", return_value=DeepEPMode.AUTO),
            patch.object(
                welmv4,
                "get_forward",
                return_value=SimpleNamespace(deepep_mode_override=None),
            ),
        ):
            self.assertEqual(
                Qwen2MoeSparseMoeBlock._resolve_deepep_mode_for_topk(True),
                DeepEPMode.NORMAL,
            )
            self.assertEqual(
                Qwen2MoeSparseMoeBlock._resolve_deepep_mode_for_topk(False),
                DeepEPMode.LOW_LATENCY,
            )

        with patch.object(
            welmv4,
            "get_forward",
            return_value=SimpleNamespace(
                deepep_mode_override=DeepEPMode.LOW_LATENCY
            ),
        ):
            self.assertEqual(
                Qwen2MoeSparseMoeBlock._resolve_deepep_mode_for_topk(True),
                DeepEPMode.LOW_LATENCY,
            )

    def test_kv_mirror_owner_partials_reconstruct_full_residual(self):
        tp_size = 4
        prompt_rows = 16
        prompt_local_rows = prompt_rows // tp_size
        mirror_rows = 5
        mirror_padded_rows = 8
        hidden_size = 3
        full_residual = torch.arange(
            prompt_rows * hidden_size, dtype=torch.float32
        ).reshape(prompt_rows, hidden_size)
        custom_last_index = torch.tensor([2, 5, 10, 14, 15])

        partials = []
        for tp_rank in range(tp_size):
            local_residual = full_residual.narrow(
                0, tp_rank * prompt_local_rows, prompt_local_rows
            )
            partials.append(
                Qwen2MoeDecoderLayer._build_kv_mirror_residual_partial(
                    local_residual,
                    custom_last_index,
                    prompt_local_rows=prompt_local_rows,
                    mirror_num_real_rows=mirror_rows,
                    mirror_num_padded_rows=mirror_padded_rows,
                    tp_rank=tp_rank,
                )
            )

        reconstructed = torch.stack(partials).sum(dim=0)
        expected = full_residual.new_zeros((mirror_padded_rows, hidden_size))
        expected[:mirror_rows] = full_residual.index_select(
            0, custom_last_index
        )
        torch.testing.assert_close(reconstructed, expected)

    def test_kv_mirror_metadata_uses_contiguous_tp_shards(self):
        cases = (
            (5, 8, [2, 2, 1, 0]),
            (1, 4, [1, 0, 0, 0]),
        )
        for mirror_rows, mirror_padded_rows, expected_counts in cases:
            self._assert_kv_mirror_metadata(
                mirror_rows, mirror_padded_rows, expected_counts
            )

    def _assert_kv_mirror_metadata(
        self, mirror_rows, mirror_padded_rows, expected_counts
    ):
        expected_local_rows = mirror_padded_rows // 4
        for tp_rank, expected_real_rows in enumerate(expected_counts):
            forward_batch = SimpleNamespace(
                kv_mirror_num_real_rows=None,
                kv_mirror_num_padded_rows=None,
                kv_mirror_local_num_tokens=None,
                global_dp_buffer_len=16,
                global_num_tokens_cpu=[16],
                global_num_tokens_gpu=torch.tensor([16], dtype=torch.int32),
                num_token_non_padded=torch.tensor(4, dtype=torch.int32),
                num_token_non_padded_cpu=4,
            )
            with (
                patch.object(
                    welmv4,
                    "get_tensor_model_parallel_world_size",
                    return_value=4,
                ),
                patch.object(
                    welmv4,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_rank=tp_rank),
                ),
            ):
                Qwen2MoeDecoderLayer._update_kv_mirror_scattered_metadata(
                    forward_batch,
                    mirror_num_real_rows=mirror_rows,
                    mirror_num_padded_rows=mirror_padded_rows,
                )

            self.assertEqual(forward_batch.kv_mirror_num_real_rows, mirror_rows)
            self.assertEqual(
                forward_batch.kv_mirror_num_padded_rows, mirror_padded_rows
            )
            self.assertEqual(
                forward_batch.kv_mirror_local_num_tokens, expected_local_rows
            )
            self.assertEqual(
                forward_batch.global_dp_buffer_len, mirror_padded_rows
            )
            self.assertEqual(
                forward_batch.global_num_tokens_cpu, [mirror_padded_rows]
            )
            self.assertEqual(
                forward_batch.global_num_tokens_gpu.item(), mirror_padded_rows
            )
            self.assertEqual(
                forward_batch.num_token_non_padded.item(), expected_real_rows
            )
            self.assertEqual(
                forward_batch.num_token_non_padded_cpu, expected_real_rows
            )

    def test_pad_rows_only_appends_zero_rows(self):
        hidden = torch.arange(10, dtype=torch.bfloat16).reshape(5, 2)
        padded = Qwen2MoeDecoderLayer._pad_rows(hidden, 8)
        torch.testing.assert_close(padded[:5], hidden)
        torch.testing.assert_close(padded[5:], torch.zeros_like(padded[5:]))

    def test_kv_mirror_ll_capacity_can_differ_from_decode(self):
        with (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128),
            envs.SGLANG_WELMV4_MIRROR_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(
                None
            ),
        ):
            self.assertEqual(
                Qwen2MoeDecoderLayer._get_kv_mirror_prefill_ll_capacity(),
                128,
            )

        with (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128),
            envs.SGLANG_WELMV4_MIRROR_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(
                512
            ),
        ):
            self.assertEqual(
                Qwen2MoeDecoderLayer._get_kv_mirror_prefill_ll_capacity(),
                512,
            )

    def test_attention_reduce_scatter_rejects_unaligned_rows(self):
        layer = object.__new__(Qwen2MoeDecoderLayer)
        with patch.object(
            welmv4, "get_tensor_model_parallel_world_size", return_value=4
        ):
            with self.assertRaisesRegex(
                RuntimeError, "padding must be completed before OProj"
            ):
                layer._npu_prefill_deepep_finish_attention(
                    torch.zeros((5, 8), dtype=torch.bfloat16),
                    torch.zeros((5, 8), dtype=torch.float32),
                    use_mmq_norm_after_attn=False,
                )

    def test_final_output_gathers_then_removes_kv_mirror_padding(self):
        model = object.__new__(Qwen2MoeModel)
        local_hidden = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        full_hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        forward_batch = SimpleNamespace(
            welmv4_npu_deepep_scattered=True,
            kv_mirror_num_real_rows=5,
            kv_mirror_num_padded_rows=8,
            global_dp_buffer_len=8,
        )
        with (
            patch.object(
                welmv4, "get_tensor_model_parallel_world_size", return_value=4
            ),
            patch.object(
                Qwen2MoeDecoderLayer,
                "_all_gather_tp_rows",
                return_value=full_hidden,
            ) as gather,
        ):
            restored, aux = model._restore_npu_prefill_deepep_output_layout(
                local_hidden, [], forward_batch
            )

        gather.assert_called_once_with(local_hidden)
        torch.testing.assert_close(restored, full_hidden[:5])
        self.assertEqual(aux, [])


if __name__ == "__main__":
    unittest.main()
