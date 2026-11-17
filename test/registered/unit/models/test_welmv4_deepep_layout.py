import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from sglang.srt.models.welmv4_dp_attention import (
    WelmDpAttentionExecutor,
    WelmRunnerParallelPlan,
    WelmRunnerRole,
)
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestWeLMv4DeepEPLayout(unittest.TestCase):
    def test_nextn_dp_ll_replaces_oproj_allreduce_with_one_reduce_scatter(self):
        attn_group = Mock(world_size=2, rank_in_group=1)
        attn_group.all_reduce.side_effect = lambda x: x * 2
        attn_group.reduce_scatter_tensor.side_effect = (
            lambda out, x: out.copy_(x.chunk(2)[1] * 2)
        )
        ep_group = SimpleNamespace(world_size=4, rank_in_group=3)
        plan = WelmRunnerParallelPlan(
            role=WelmRunnerRole.DRAFT,
            attn_tp_group=attn_group,
            moe_tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
            moe_ep_group=ep_group,
            outer_dp_bridge_group=ep_group,
            outer_target_dp_rank=1,
            outer_target_dp_size=2,
            decode_rs_ag_enabled=True,
            decode_ll_dispatch_capacity=4,
            local_ep_kernel_available=True,
            shared_expert_ep_replicated=True,
            o_proj_returns_partial=False,
            finalized=True,
        )
        executor = WelmDpAttentionExecutor(plan)
        partial = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        layout = forward_batch_info.build_welm_local_ll_row_layout(
            4, 4, torch.tensor([4], dtype=torch.int32)
        )

        def attention(*, hidden_states, skip_o_proj_all_reduce, **kwargs):
            # Physical NextN OProj has reduce_results=True with A>1.
            self.assertFalse(kwargs["use_o_proj_matmul_reduce_scatter"])
            self.assertEqual(kwargs["o_proj_rs_pipeline_chunks"], 0)
            self.assertIsNone(kwargs["o_proj_rs_group"])
            if skip_o_proj_all_reduce:
                return hidden_states
            return attn_group.all_reduce(hidden_states)

        layer = SimpleNamespace(
            layer_id=48,
            is_final_layer=True,
            kv_mirror_layers=(48,),
            ppln=False,
            self_attn=Mock(side_effect=attention, use_o_norm=False),
            input_layernorm=lambda x, residual, **kwargs: (x, residual),
            post_attention_layernorm=lambda x, residual, **kwargs: (x, residual, None),
        )
        modes = forward_batch_info.ForwardMode
        for mode in (modes.EXTEND, modes.DRAFT_EXTEND_V2):
            ll = mode == modes.DRAFT_EXTEND_V2
            batch = SimpleNamespace(
                forward_mode=mode,
                is_extend_in_batch=not ll,
                enable_kv_mirror=False,
                welm_dp_moe_row_layout=layout,
            )
            attn_group.reset_mock()
            with patch.object(
                executor, "_run_moe_transport", side_effect=lambda *, state, **kw: state
            ):
                hidden, residual = executor.forward(
                    layer=layer,
                    positions=torch.arange(4),
                    hidden_states=partial,
                    forward_batch=batch,
                    residual=partial,
                )
            self.assertEqual(attn_group.all_reduce.call_count, int(not ll))
            self.assertEqual(attn_group.reduce_scatter_tensor.call_count, int(ll))
            expected = partial[2:] if ll else partial
            torch.testing.assert_close(hidden, expected * 2)
            torch.testing.assert_close(residual, expected)

    def test_nextn_ll_masks_use_full_local_count_on_pure_tp_and_dp_attention(self):
        # Rank 1 owns [4:8] of the attention group's eight rows. In DP it
        # also belongs to outer shard 1, whose local row view has one entry.
        count = torch.tensor([8], dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=forward_batch_info.ForwardMode.DRAFT_EXTEND_V2,
            is_extend_in_batch=False,
            enable_kv_mirror=False,
            can_run_tbo=False,
            welm_ll_real_rows=count,
            welm_decode_tp_positions=None,
            welm_dp_moe_row_layout=forward_batch_info.build_welm_local_ll_row_layout(
                8, 8, count
            ),
        )
        plan = SimpleNamespace(
            finalized=True,
            role=WelmRunnerRole.DRAFT,
            decode_rs_ag_enabled=True,
            attn_tp_rank=1,
            attn_tp_size=2,
            outer_target_dp_rank=1,
        )
        layer = SimpleNamespace(attn_tp_rank=1, welm_decode_ll_dispatch_capacity=4)
        hidden = torch.empty((8, 2))
        with (
            patch.object(welmv4, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(welmv4, "get_deepep_mode", return_value=DeepEPMode.AUTO),
        ):
            for real_rows in (5, 0, 8):
                count.fill_(real_rows)
                Qwen2MoeDecoderLayer._prepare_npu_decode_rs_ag_scratch(
                    layer, hidden, batch
                )
                WelmDpAttentionExecutor.prepare_forward_scratch(
                    SimpleNamespace(runner_plan=plan),
                    forward_batch=batch,
                    hidden_states=hidden,
                )
                expected = torch.arange(4, 8) < real_rows
                torch.testing.assert_close(batch.welm_decode_tp_valid_mask, expected)
                torch.testing.assert_close(
                    batch.welm_dp_moe_row_layout.current._attn_scattered_valid_mask,
                    expected,
                )

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
            patch.object(
                welmv4, "get_deepep_mode", return_value=DeepEPMode.AUTO
            ),
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
                    mirror_num_padded_rows=mirror_rows,
                    tp_rank=tp_rank,
                )
            )

        reconstructed = torch.stack(partials).sum(dim=0)
        expected = full_residual.index_select(0, custom_last_index)
        torch.testing.assert_close(reconstructed, expected)

    def test_kv_mirror_metadata_switches_to_exact_full_rows(self):
        mirror_rows = 5
        forward_batch = SimpleNamespace(
            kv_mirror_num_real_rows=None,
            kv_mirror_num_padded_rows=None,
            kv_mirror_local_num_tokens=None,
            global_dp_buffer_len=16,
            global_num_tokens_cpu=[16],
            global_num_tokens_gpu=torch.tensor([16], dtype=torch.int32),
            num_token_non_padded=torch.tensor(4, dtype=torch.int32),
            num_token_non_padded_cpu=4,
            welmv4_npu_deepep_scattered=True,
            welmv4_npu_deepep_full_mirror=False,
        )
        Qwen2MoeDecoderLayer._update_kv_mirror_full_metadata(
            forward_batch,
            mirror_num_real_rows=mirror_rows,
        )

        self.assertEqual(forward_batch.kv_mirror_num_real_rows, mirror_rows)
        self.assertEqual(forward_batch.kv_mirror_num_padded_rows, mirror_rows)
        self.assertEqual(forward_batch.kv_mirror_local_num_tokens, mirror_rows)
        self.assertEqual(forward_batch.global_dp_buffer_len, mirror_rows)
        self.assertEqual(forward_batch.global_num_tokens_cpu, [mirror_rows])
        self.assertEqual(forward_batch.global_num_tokens_gpu.item(), mirror_rows)
        self.assertEqual(forward_batch.num_token_non_padded.item(), mirror_rows)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, mirror_rows)
        self.assertFalse(forward_batch.welmv4_npu_deepep_scattered)
        self.assertTrue(forward_batch.welmv4_npu_deepep_full_mirror)

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

    def test_fused_oproj_output_skips_second_reduce_scatter(self):
        local_hidden = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
        full_residual = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        expected_residual = full_residual[4:6]

        def post_attention_layernorm(hidden_states, residual, **kwargs):
            torch.testing.assert_close(hidden_states, local_hidden)
            torch.testing.assert_close(residual, expected_residual)
            self.assertTrue(kwargs["clone_fp32_out"])
            return hidden_states, residual, None

        layer = SimpleNamespace(
            self_attn=SimpleNamespace(o_norm=None),
            post_attention_layernorm=post_attention_layernorm,
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
                return_value=SimpleNamespace(tp_rank=2),
            ),
            patch.object(welmv4, "get_tp_group") as tp_group,
        ):
            hidden, residual, hidden_fp32 = (
                Qwen2MoeDecoderLayer._npu_prefill_deepep_finish_attention(
                    layer,
                    local_hidden,
                    full_residual,
                    use_mmq_norm_after_attn=False,
                    input_is_reduce_scattered=True,
                )
            )

        tp_group.assert_not_called()
        torch.testing.assert_close(hidden, local_hidden)
        torch.testing.assert_close(residual, expected_residual)
        self.assertIsNone(hidden_fp32)

    def test_final_scattered_prompt_output_gathers_full_rows(self):
        model = object.__new__(Qwen2MoeModel)
        local_hidden = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        full_hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        forward_batch = SimpleNamespace(
            welmv4_npu_deepep_scattered=True,
            welmv4_npu_deepep_full_mirror=False,
            kv_mirror_num_real_rows=None,
            kv_mirror_num_padded_rows=None,
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
        torch.testing.assert_close(restored, full_hidden)
        self.assertEqual(aux, [])

    def test_final_full_kv_mirror_output_needs_no_gather_or_trim(self):
        model = object.__new__(Qwen2MoeModel)
        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        forward_batch = SimpleNamespace(
            welmv4_npu_deepep_scattered=False,
            welmv4_npu_deepep_full_mirror=True,
            kv_mirror_num_real_rows=5,
        )
        restored, aux = model._restore_npu_prefill_deepep_output_layout(
            hidden, [], forward_batch
        )
        self.assertIs(restored, hidden)
        self.assertEqual(aux, [])


if __name__ == "__main__":
    unittest.main()
