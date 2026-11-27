"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import sys
import unittest
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    FULL_ATTENTION_WINDOW,
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
    _expand_dsa_sparse_indices,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestExpandDsaSparseIndices(unittest.TestCase):
    def test_2d_input_adds_unsqueeze(self):
        """A [T, K] tensor becomes [T, 1, K]."""
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 1, 3))
        self.assertTrue(torch.equal(result.squeeze(1), topk))

    def test_3d_input_passthrough(self):
        topk = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertTrue(torch.equal(result, topk))

    def test_2d_shape_correctness(self):
        topk = torch.zeros(5, 8)
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.dim(), 3)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 1)
        self.assertEqual(result.shape[2], 8)

    def test_2d_single_row(self):
        topk = torch.tensor([[1, 2, 3, 4]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (1, 1, 4))


class TestReshapeKvForFiaNz(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (-1, 1, num_heads*head_dim//16, page_size, 16)."""
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (1, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_element_preservation(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 2 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.numel(), tensor.numel())
        self.assertTrue(torch.equal(result.flatten(), tensor))

    def test_different_parameters(self):
        num_heads = 4
        head_dim = 128
        page_size = 32
        total = 3 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.randn(total)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (3, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_view_relationship(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())


class TestWelMLayerwiseSinkTritonRouting(unittest.TestCase):
    def setUp(self):
        self.backend = AscendAttnBackend.__new__(AscendAttnBackend)
        self.backend.enable_welm_fia_sink_lse = False

    def test_welm_sink_tensor_uses_triton(self):
        self.backend.is_welm_v4 = True
        self.assertTrue(self.backend._use_welm_sink_triton(torch.ones(6)))
        self.assertFalse(self.backend._use_welm_sink_triton(None))

    def test_other_models_keep_existing_dispatch(self):
        self.backend.is_welm_v4 = False
        self.assertFalse(self.backend._use_welm_sink_triton(torch.ones(6)))

    def test_welm_window_adds_current_token(self):
        self.backend.is_welm_v4 = True
        layer = SimpleNamespace(sliding_window_size=511)
        self.assertEqual(self.backend._get_sink_triton_window_size(layer), 512)

    def test_fia_full_window_uses_explicit_large_pre_tokens(self):
        self.assertEqual(
            self.backend._get_fia_pre_tokens(
                SimpleNamespace(sliding_window_size=-1)
            ),
            FULL_ATTENTION_WINDOW,
        )
        self.assertEqual(
            self.backend._get_fia_pre_tokens(
                SimpleNamespace(sliding_window_size=511)
            ),
            511,
        )

    def test_full_attention_window_stays_disabled(self):
        self.backend.is_welm_v4 = True
        self.assertEqual(
            self.backend._get_sink_triton_window_size(
                SimpleNamespace(sliding_window_size=-1)
            ),
            -1,
        )
        self.assertEqual(
            self.backend._get_sink_triton_window_size(
                SimpleNamespace(sliding_window_size=None)
            ),
            -1,
        )

    def test_other_models_keep_existing_window_value(self):
        self.backend.is_welm_v4 = False
        layer = SimpleNamespace(sliding_window_size=4096)
        self.assertEqual(self.backend._get_sink_triton_window_size(layer), 4096)

    def test_split_pool_mapping_is_authoritative_for_swa_routing(self):
        self.backend.is_hybrid_swa = True
        self.backend.token_to_kv_pool = SWAKVPool.__new__(SWAKVPool)
        self.backend.token_to_kv_pool.layers_mapping = {
            0: (0, False),
            1: (0, True),
        }

        # A positive marker must not route a Full-owned layer to SWA tables.
        self.assertFalse(
            self.backend._is_swa_layer(
                SimpleNamespace(layer_id=0, sliding_window_size=1023)
            )
        )
        self.assertTrue(
            self.backend._is_swa_layer(
                SimpleNamespace(layer_id=1, sliding_window_size=511)
            )
        )

    def test_split_pool_missing_layer_mapping_fails_loudly(self):
        self.backend.is_hybrid_swa = True
        self.backend.token_to_kv_pool = SWAKVPool.__new__(SWAKVPool)
        self.backend.token_to_kv_pool.layers_mapping = {}

        with self.assertRaisesRegex(RuntimeError, "no layer mapping"):
            self.backend._is_swa_layer(
                SimpleNamespace(layer_id=7, sliding_window_size=511)
            )

    def test_non_hybrid_backend_never_uses_swa_mapping(self):
        self.backend.is_hybrid_swa = False
        self.backend.token_to_kv_pool = SWAKVPool.__new__(SWAKVPool)
        self.backend.token_to_kv_pool.layers_mapping = {1: (0, True)}

        self.assertFalse(
            self.backend._is_swa_layer(
                SimpleNamespace(layer_id=1, sliding_window_size=511)
            )
        )

    def test_only_full_welm_sink_layers_use_dedicated_kernel(self):
        self.backend.is_welm_v4 = True
        self.backend.is_hybrid_swa = True
        self.backend.token_to_kv_pool = SWAKVPool.__new__(SWAKVPool)
        self.backend.token_to_kv_pool.layers_mapping = {
            0: (0, False),
            1: (0, True),
        }

        sinks = torch.ones(6)
        self.assertTrue(
            self.backend._is_welm_full_sink_layer(
                SimpleNamespace(layer_id=0, sliding_window_size=-1), sinks
            )
        )
        self.assertFalse(
            self.backend._is_welm_full_sink_layer(
                SimpleNamespace(layer_id=1, sliding_window_size=511), sinks
            )
        )
        self.assertFalse(
            self.backend._is_welm_full_sink_layer(
                SimpleNamespace(layer_id=0, sliding_window_size=-1), None
            )
        )

    def test_only_windowed_welm_sink_layers_use_dedicated_swa_kernel(self):
        self.backend.is_welm_v4 = True
        sinks = torch.ones(6)

        self.assertTrue(
            self.backend._is_welm_swa_sink_layer(
                SimpleNamespace(sliding_window_size=511), sinks
            )
        )
        self.assertFalse(
            self.backend._is_welm_swa_sink_layer(
                SimpleNamespace(sliding_window_size=-1), sinks
            )
        )
        self.assertFalse(
            self.backend._is_welm_swa_sink_layer(
                SimpleNamespace(sliding_window_size=511), None
            )
        )

    def test_swa_sink_kernel_uses_left_token_count_without_plus_one(self):
        self.assertEqual(
            self.backend._get_welm_swa_sink_local_window_size(
                SimpleNamespace(sliding_window_size=511)
            ),
            511,
        )
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            self.backend._get_welm_swa_sink_local_window_size(
                SimpleNamespace(sliding_window_size=-1)
            )

    def test_other_model_does_not_use_dedicated_full_sink_kernel(self):
        self.backend.is_welm_v4 = False
        self.backend.is_hybrid_swa = False
        self.assertFalse(
            self.backend._is_welm_full_sink_layer(
                SimpleNamespace(layer_id=0, sliding_window_size=-1), torch.ones(6)
            )
        )

    def test_swa_semantics_stay_windowed_without_hybrid_pool(self):
        self.backend.is_welm_v4 = True
        self.backend.is_hybrid_swa = False
        self.assertFalse(
            self.backend._is_welm_full_sink_layer(
                SimpleNamespace(layer_id=1, sliding_window_size=511), torch.ones(6)
            )
        )

    def test_full_sink_cache_view_is_zero_copy_head_major(self):
        cache = torch.arange(2 * 4 * 3 * 5).reshape(2 * 4, 3, 5)
        result = self.backend._view_welm_full_sink_kv_cache(
            cache, page_size=4, num_heads=3, head_dim=5
        )

        self.assertEqual(result.shape, (2, 3, 4, 5))
        self.assertEqual(result.data_ptr(), cache.data_ptr())
        self.assertTrue(torch.equal(result[1, 2, 3], cache[7, 2]))

    def test_full_sink_prefill_ignores_attention_dp_padding(self):
        self.backend.page_size = 2
        self.backend.forward_metadata = ForwardMetadata(
            extend_seq_lens_cpu_int=torch.tensor([2, 1], dtype=torch.int32),
            seq_lens_cpu_int=torch.tensor([2, 1], dtype=torch.int32),
            seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            full_sink_prefill_cu_q_lens=torch.tensor([0, 2, 3], dtype=torch.int32),
            full_sink_prefill_schedule=(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
            ),
            full_sink_prefill_schedule_key=(4, 2, 2, False),
        )
        layer = SimpleNamespace(
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=2,
            v_head_dim=2,
            scaling=0.5,
        )
        q = torch.randn(4, 8)
        k_cache = torch.randn(4, 2, 2)
        v_cache = torch.randn(4, 2, 2)
        block_tables = torch.tensor([[0], [1]], dtype=torch.int32)
        fake_prefill = MagicMock(side_effect=lambda **kwargs: torch.ones_like(kwargs["q"]))

        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend."
            "_load_welm_full_sink_attention_ops",
            return_value=(None, fake_prefill, None),
        ):
            output = self.backend._forward_welm_full_sink_prefill(
                q,
                k_cache,
                v_cache,
                layer,
                SimpleNamespace(),
                block_tables,
                torch.ones(4),
            )

        self.assertEqual(output.shape, (4, 8))
        self.assertTrue(torch.equal(output[:3], torch.ones_like(output[:3])))
        self.assertTrue(torch.equal(output[3], torch.zeros_like(output[3])))
        call = fake_prefill.call_args.kwargs
        self.assertEqual(call["q"].shape, (3, 4, 2))
        self.assertEqual(call["key_cache"].shape, (2, 2, 2, 2))

    def test_full_sink_decode_ignores_attention_dp_padding(self):
        self.backend.page_size = 2
        self.backend.graph_mode = False
        self.backend.max_context_len = 32
        self.backend.forward_metadata = ForwardMetadata(
            seq_lens_cpu_int=torch.tensor([3, 4], dtype=torch.int32)
        )
        layer = SimpleNamespace(
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=2,
            v_head_dim=2,
            scaling=0.5,
        )
        q = torch.randn(3, 8)
        k_cache = torch.randn(4, 2, 2)
        v_cache = torch.randn(4, 2, 2)
        seq_lens = torch.tensor([3, 4], dtype=torch.int32)
        block_tables = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        fake_decode = MagicMock(side_effect=lambda **kwargs: torch.ones_like(kwargs["q"]))

        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend."
            "_load_welm_full_sink_attention_ops",
            return_value=(None, None, fake_decode),
        ):
            output = self.backend._forward_welm_full_sink_decode(
                q,
                k_cache,
                v_cache,
                layer,
                block_tables,
                seq_lens,
                torch.ones(4),
            )

        self.assertEqual(output.shape, (3, 8))
        self.assertTrue(torch.equal(output[:2], torch.ones_like(output[:2])))
        self.assertTrue(torch.equal(output[2], torch.zeros_like(output[2])))
        call = fake_decode.call_args.kwargs
        self.assertEqual(call["q"].shape, (2, 4, 2))
        self.assertEqual(call["key_cache"].shape, (2, 2, 2, 2))
        self.assertEqual(call["max_kv_len_hint"], 4)

    def test_swa_sink_prefill_uses_paged_head_major_cache_and_left_window(self):
        self.backend.page_size = 2
        self.backend.forward_metadata = ForwardMetadata(
            extend_seq_lens_cpu_int=torch.tensor([2, 1], dtype=torch.int32),
            seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        )
        layer = SimpleNamespace(
            sliding_window_size=511,
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=2,
            v_head_dim=2,
            scaling=0.5,
        )
        q = torch.randn(4, 8, dtype=torch.float16)
        k_cache = torch.randn(4, 2, 2, dtype=torch.float16)
        v_cache = torch.randn(4, 2, 2, dtype=torch.float16)
        block_tables = torch.tensor([[0], [1]], dtype=torch.int32)
        sinks = torch.ones(4, dtype=torch.float32)
        fake_prefill = MagicMock(side_effect=lambda **kwargs: torch.ones_like(kwargs["q"]))

        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend."
            "_load_welm_swa_sink_attention_ops",
            return_value=(fake_prefill, None),
        ):
            output = self.backend._forward_welm_swa_sink_prefill(
                q,
                k_cache,
                v_cache,
                layer,
                block_tables,
                sinks,
            )

        self.assertEqual(output.shape, (4, 8))
        self.assertTrue(torch.equal(output[:3], torch.ones_like(output[:3])))
        self.assertTrue(torch.equal(output[3], torch.zeros_like(output[3])))
        call = fake_prefill.call_args.kwargs
        self.assertEqual(call["q"].shape, (3, 4, 2))
        self.assertEqual(call["key_cache"].shape, (2, 2, 2, 2))
        self.assertTrue(
            torch.equal(
                call["cu_q_lens"], torch.tensor([0, 2, 3], dtype=torch.int32)
            )
        )
        self.assertEqual(call["local_window_size"], 511)
        self.assertEqual(call["global_window_size"], 0)
        self.assertIs(call["sinks"], sinks)

    def test_swa_sink_decode_ignores_attention_dp_padding(self):
        self.backend.page_size = 2
        self.backend.forward_metadata = ForwardMetadata()
        layer = SimpleNamespace(
            sliding_window_size=511,
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=2,
            v_head_dim=2,
            scaling=0.5,
        )
        q = torch.randn(3, 8, dtype=torch.float16)
        k_cache = torch.randn(4, 2, 2, dtype=torch.float16)
        v_cache = torch.randn(4, 2, 2, dtype=torch.float16)
        seq_lens = torch.tensor([3, 4], dtype=torch.int32)
        block_tables = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        sinks = torch.ones(4, dtype=torch.float32)
        fake_decode = MagicMock(side_effect=lambda **kwargs: torch.ones_like(kwargs["q"]))

        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend."
            "_load_welm_swa_sink_attention_ops",
            return_value=(None, fake_decode),
        ):
            output = self.backend._forward_welm_swa_sink_decode(
                q,
                k_cache,
                v_cache,
                layer,
                block_tables,
                seq_lens,
                sinks,
            )

        self.assertEqual(output.shape, (3, 8))
        self.assertTrue(torch.equal(output[:2], torch.ones_like(output[:2])))
        self.assertTrue(torch.equal(output[2], torch.zeros_like(output[2])))
        call = fake_decode.call_args.kwargs
        self.assertEqual(call["q"].shape, (2, 4, 2))
        self.assertEqual(call["key_cache"].shape, (2, 2, 2, 2))
        self.assertEqual(call["local_window_size"], 511)
        self.assertEqual(call["global_window_size"], 0)
        self.assertIs(call["sinks"], sinks)


class TestForwardMetadata(unittest.TestCase):
    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(ForwardMetadata))

    def test_field_defaults(self):
        metadata = ForwardMetadata()
        non_none_defaults = {
            "welm_flash_max_seqlen_q": -1,
            "welm_decode_metadata_in_graph": False,
            "welm_flash_schedules": {},
        }
        for f in fields(ForwardMetadata):
            if f.name in non_none_defaults:
                self.assertEqual(getattr(metadata, f.name), non_none_defaults[f.name])
            else:
                self.assertIsNone(
                    getattr(metadata, f.name),
                    f"Field '{f.name}' should default to None",
                )

    def test_create_with_values(self):
        block_tables = torch.tensor([[1, 2], [3, 4]])
        seq_lens = torch.tensor([10, 20])
        metadata = ForwardMetadata(
            block_tables=block_tables,
            seq_lens=seq_lens,
            seq_lens_cpu_list=[10, 20],
        )
        self.assertTrue(torch.equal(metadata.block_tables, block_tables))
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertEqual(metadata.seq_lens_cpu_list, [10, 20])

    def test_partial_assignment(self):
        metadata = ForwardMetadata(swa_mask=torch.ones(3, 3))
        self.assertIsNotNone(metadata.swa_mask)
        self.assertIsNone(metadata.block_tables)
        self.assertIsNone(metadata.seq_lens)

    def test_field_names(self):
        names = {f.name for f in fields(ForwardMetadata)}
        expected = {
            "block_tables",
            "block_tables_swa",
            "swa_out_cache_loc",
            "welm_static_decode_full_slots",
            "welm_static_decode_swa_slots",
            "extend_seq_lens_cpu_int",
            "seq_lens_cpu_int",
            "seq_lens_cpu_list",
            "seq_lens_list_cumsum",
            "seq_lens",
            "actual_seq_lengths_q",
            "actual_seq_lengths_q_pa",
            "actual_seq_lengths_kv",
            "swa_mask",
            "prefix_lens",
            "flatten_prefix_block_tables",
            "full_sink_prefill_cu_q_lens",
            "full_sink_prefill_schedule",
            "full_sink_prefill_schedule_key",
            "welm_flash_cu_seqlens_q",
            "welm_flash_seqused_q",
            "welm_flash_seqused_kv",
            "welm_flash_max_seqlen_q",
            "welm_decode_metadata_in_graph",
            "welm_flash_schedules",
            "welm_flash_mirror_q_lengths",
        }
        self.assertEqual(names, expected)


class TestWelmStaticDecodeSlots(unittest.TestCase):
    """CPU checks of the production metadata producers, without kernel launch."""

    class NativePool:
        def __init__(self, dtype=torch.bfloat16):
            self.store_dtype = dtype

    def evaluate_gate(
        self,
        *,
        tp=2,
        pp=1,
        env=None,
        flags=None,
        backend_changes=None,
        runner_changes=None,
        pool=None,
    ):
        backend = self.make_backend(swa=False)
        backend.use_mla = False
        backend.model_dtype = torch.bfloat16
        backend.dllm_config = None
        backend.enable_torch_compile = False
        backend.attn_cp_size = 1
        backend.token_to_kv_pool = self.NativePool() if pool is None else pool
        vars(backend).update(backend_changes or {})
        runner = SimpleNamespace(
            server_args=SimpleNamespace(**(flags or {})),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            is_draft_worker=False,
        )
        vars(runner).update(runner_changes or {})
        env_values = {"SGLANG_NPU_WELMV4_FUSED_QKV": True}
        env_values.update(env or {})
        module = AscendAttnBackend.__module__
        pool_module = "sglang.srt.hardware_backend.npu.memory_pool_npu"
        with (
            patch(
                f"{module}.get_bool_env_var",
                side_effect=lambda name, default: env_values.get(name, False),
            ),
            patch(
                f"{module}.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=tp, pp_size=pp),
            ),
            patch.dict(
                sys.modules,
                {pool_module: SimpleNamespace(NPUMHATokenToKVPool=self.NativePool)},
            ),
        ):
            backend._init_welm_static_decode_qkv(runner)
        return backend.use_welm_static_decode_qkv

    def make_backend(self, *, swa=True):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.device = "cpu"
        backend.use_welm_static_decode_qkv = True
        backend.use_sliding_window_kv_pool = swa
        backend.is_hybrid_swa = swa
        backend.use_welm_flash_attn = False
        backend.capture_welm_decode_metadata = False
        backend.is_welm_v4 = True
        backend.max_context_len = 256
        backend.page_size = 64
        backend.speculative_num_draft_tokens = None
        backend.q_head_num_padding = None
        backend.forward_metadata = ForwardMetadata()
        return backend

    def make_batch(self, locs=(11, 0, -1, 9), lengths=(12, 7, 0, 0)):
        return SimpleNamespace(
            batch_size=len(locs),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            is_extend_in_batch=False,
            out_cache_loc=torch.tensor(locs, dtype=torch.int32),
            seq_lens=torch.tensor(lengths, dtype=torch.int64),
            positions=torch.zeros(len(locs), dtype=torch.int64),
        )

    def test_gate_default_off_does_not_import_npu_pool(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        module = AscendAttnBackend.__module__
        with (
            patch(f"{module}.get_bool_env_var", return_value=False) as get_env,
            patch.dict(
                sys.modules,
                {"sglang.srt.hardware_backend.npu.memory_pool_npu": None},
            ),
        ):
            backend._init_welm_static_decode_qkv(SimpleNamespace())
        self.assertFalse(backend.use_welm_static_decode_qkv)
        get_env.assert_called_once_with(
            "SGLANG_NPU_WELMV4_FUSED_QKV", "False"
        )

    def test_retired_static_decode_flag_does_not_enable_slots(self):
        self.assertFalse(
            self.evaluate_gate(
                env={
                    "SGLANG_NPU_WELMV4_FUSED_QKV": False,
                    "SGLANG_NPU_WELMV4_STATIC_DECODE_QKV": True,
                }
            )
        )

    def test_gate_accepts_native_full_and_swa_pools_for_attention_tp_1_2_4(self):
        swa_pool = SWAKVPool.__new__(SWAKVPool)
        swa_pool.full_kv_pool = self.NativePool()
        swa_pool.swa_kv_pool = self.NativePool()
        for tp in (1, 2, 4):
            for pool in (self.NativePool(), swa_pool):
                with self.subTest(tp=tp, pool_type=type(pool).__name__):
                    self.assertTrue(self.evaluate_gate(tp=tp, pool=pool))

    def test_gate_rejects_unvalidated_execution_modes(self):
        changes = (
            {"tp": 8},
            {"pp": 2},
            {"backend_changes": {"attn_cp_size": 2}},
            {"backend_changes": {"is_welm_v4": False}},
            {"backend_changes": {"use_mla": True}},
            {"backend_changes": {"model_dtype": torch.float16}},
            {"backend_changes": {"dllm_config": object()}},
            {"backend_changes": {"enable_torch_compile": True}},
            {"runner_changes": {"is_draft_worker": True}},
            {
                "runner_changes": {
                    "spec_algorithm": SimpleNamespace(is_none=lambda: False)
                }
            },
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(self.evaluate_gate(**change))
        for name in (
            "enable_two_batch_overlap",
            "enable_single_batch_overlap",
            "enable_pdmux",
            "enable_hierarchical_cache",
            "disaggregation_decode_enable_offload_kvcache",
            "enable_memory_saver",
            "enable_lora",
            "lora_paths",
            "cpu_offload_gb",
        ):
            with self.subTest(flag=name):
                self.assertFalse(self.evaluate_gate(flags={name: True}))
        for name in (
            "SGLANG_NPU_ENABLE_STATIC_KERNEL",
            "SGLANG_NPU_ENABLE_SUPERKERNEL",
        ):
            with self.subTest(env=name):
                self.assertFalse(self.evaluate_gate(env={name: True}))

    def test_gate_rejects_unvalidated_pool_subclasses_and_storage_dtype(self):
        class UnvalidatedPool(self.NativePool):
            pass

        for pool in (
            object(),
            UnvalidatedPool(),
            self.NativePool(dtype=torch.float16),
        ):
            with self.subTest(pool_type=type(pool).__name__):
                self.assertFalse(self.evaluate_gate(pool=pool))
        swa_pool = SWAKVPool.__new__(SWAKVPool)
        swa_pool.full_kv_pool = self.NativePool()
        swa_pool.swa_kv_pool = UnvalidatedPool()
        self.assertFalse(self.evaluate_gate(pool=swa_pool))

    def test_eager_slots_are_private_int64_and_skip_dummy_and_inactive_rows(self):
        backend = self.make_backend()
        batch = self.make_batch()
        # A translated dummy value must not become a write even if nonzero.
        swa = torch.tensor([21, 22, -1, 29], dtype=torch.int64)
        backend.forward_metadata.swa_out_cache_loc = swa
        original_full, original_swa = batch.out_cache_loc.clone(), swa.clone()
        backend._prepare_welm_static_decode_slots(batch, allocate=True)
        metadata = backend.forward_metadata
        torch.testing.assert_close(
            metadata.welm_static_decode_full_slots,
            torch.tensor([11, -1, -1, -1], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.welm_static_decode_swa_slots,
            torch.tensor([21, -1, -1, -1], dtype=torch.int64),
        )
        torch.testing.assert_close(batch.out_cache_loc, original_full)
        torch.testing.assert_close(swa, original_swa)
        self.assertNotEqual(
            metadata.welm_static_decode_swa_slots.data_ptr(), swa.data_ptr()
        )
        self.assertNotEqual(
            metadata.welm_static_decode_full_slots.data_ptr(),
            batch.out_cache_loc.data_ptr(),
        )

    def test_graph_buffers_are_preallocated_and_bucket_views_keep_addresses(self):
        backend = self.make_backend()
        backend.init_cuda_graph_state(max_bs=64, max_num_tokens=64)
        backing = backend.graph_metadata["welm_static_decode_full_slots"]
        swa_backing = backend.graph_metadata["welm_static_decode_swa_slots"]
        self.assertEqual(backing.shape, (64,))
        self.assertEqual(swa_backing.shape, (64,))
        for rows in (1, 2, 4, 8, 16, 32, 64):
            metadata = backend._init_cuda_graph_metadata(
                rows,
                ForwardMode.DECODE,
                torch.zeros(rows, dtype=torch.int64),
                torch.zeros(rows, dtype=torch.int32),
            )
            self.assertEqual(metadata.welm_static_decode_full_slots.shape, (rows,))
            self.assertEqual(metadata.welm_static_decode_full_slots.dtype, torch.int64)
            self.assertEqual(
                metadata.welm_static_decode_full_slots.data_ptr(), backing.data_ptr()
            )
            self.assertEqual(metadata.welm_static_decode_swa_slots.shape, (rows,))
            self.assertEqual(metadata.welm_static_decode_swa_slots.dtype, torch.int64)
            self.assertEqual(
                metadata.welm_static_decode_swa_slots.data_ptr(),
                swa_backing.data_ptr(),
            )

    def test_eager_32_and_64_row_full_and_swa_slots(self):
        for rows in (32, 64):
            with self.subTest(rows=rows):
                backend = self.make_backend()
                locs = list(range(11, 11 + rows))
                locs[0] = 0
                lengths = [20] * (rows - 1) + [0]
                batch = self.make_batch(locs, lengths)
                swa_locs = list(range(1011, 1011 + rows))
                swa = torch.tensor(swa_locs, dtype=torch.int64)
                backend.forward_metadata.swa_out_cache_loc = swa
                original_full = batch.out_cache_loc.clone()
                original_swa = swa.clone()
                backend._prepare_welm_static_decode_slots(batch, allocate=True)
                metadata = backend.forward_metadata
                for slots, source, expected in (
                    (
                        metadata.welm_static_decode_full_slots,
                        batch.out_cache_loc,
                        [-1] + locs[1:-1] + [-1],
                    ),
                    (
                        metadata.welm_static_decode_swa_slots,
                        swa,
                        [-1] + swa_locs[1:-1] + [-1],
                    ),
                ):
                    self.assertEqual(slots.shape, (rows,))
                    self.assertEqual(slots.dtype, torch.int64)
                    self.assertNotEqual(slots.data_ptr(), source.data_ptr())
                    torch.testing.assert_close(
                        slots, torch.tensor(expected, dtype=torch.int64)
                    )
                torch.testing.assert_close(batch.out_cache_loc, original_full)
                torch.testing.assert_close(swa, original_swa)

    def test_unsupported_rows_do_not_offer_eager_or_graph_slots(self):
        for rows in (3, 128):
            with self.subTest(rows=rows):
                backend = self.make_backend()
                batch = self.make_batch(list(range(1, rows + 1)), [20] * rows)
                backend.forward_metadata.swa_out_cache_loc = (
                    batch.out_cache_loc.to(torch.int64) + 1000
                )
                with patch.object(
                    torch,
                    "empty_like",
                    side_effect=AssertionError("unsupported eager allocation"),
                ):
                    backend._prepare_welm_static_decode_slots(batch, allocate=True)
                self.assertIsNone(
                    backend.forward_metadata.welm_static_decode_full_slots
                )
                self.assertIsNone(
                    backend.forward_metadata.welm_static_decode_swa_slots
                )
                # Both shapes fit the backing storage: rejection is the ABI
                # allowlist, rather than an undersized graph allocation.
                backend.init_cuda_graph_state(max_bs=128, max_num_tokens=128)
                metadata = backend._init_cuda_graph_metadata(
                    rows, ForwardMode.DECODE, batch.seq_lens, batch.out_cache_loc
                )
                self.assertIsNone(metadata.welm_static_decode_full_slots)
                self.assertIsNone(metadata.welm_static_decode_swa_slots)

    def test_graph_prologue_updates_after_translation_and_handles_idle_reuse(self):
        for rows, captured_translation in (
            (rows, captured) for rows in (4, 32, 64) for captured in (False, True)
        ):
            with self.subTest(rows=rows, captured_translation=captured_translation):
                backend = self.make_backend()
                backend.init_cuda_graph_state(max_bs=64, max_num_tokens=64)
                batch = self.make_batch([0] * rows, [0] * rows)
                metadata = backend._init_cuda_graph_metadata(
                    rows,
                    ForwardMode.DECODE,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    welm_decode_metadata_in_graph=captured_translation,
                )
                backend.forward_metadata = metadata
                full_slots = metadata.welm_static_decode_full_slots
                swa_slots = metadata.welm_static_decode_swa_slots
                pointers = (full_slots.data_ptr(), swa_slots.data_ptr())

                def translate(_):
                    metadata.swa_out_cache_loc.copy_(batch.out_cache_loc * 2)

                backend._prepare_welm_decode_metadata_device = MagicMock(
                    side_effect=translate
                )
                # The same captured DECODE prologue later sees IDLE's all-zero
                # registry inputs; Python forward_mode remains DECODE throughout.
                for locs, lengths, expected in (
                    (
                        list(range(1, rows + 1)),
                        [10] * rows,
                        list(range(1, rows + 1)),
                    ),
                    (
                        [3, 7] + [0] * (rows - 2),
                        [8, 9] + [0] * (rows - 2),
                        [3, 7] + [-1] * (rows - 2),
                    ),
                    ([0] * rows, [0] * rows, [-1] * rows),
                    (
                        [5] + [0] * (rows - 1),
                        [20] + [0] * (rows - 1),
                        [5] + [-1] * (rows - 1),
                    ),
                ):
                    batch.out_cache_loc.copy_(torch.tensor(locs, dtype=torch.int32))
                    batch.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int64))
                    if not captured_translation:
                        translate(batch)  # legacy out_graph translation
                    with patch.object(
                        torch,
                        "empty_like",
                        side_effect=AssertionError("graph allocation"),
                    ):
                        backend.init_forward_metadata_in_graph(batch)
                    torch.testing.assert_close(
                        full_slots, torch.tensor(expected, dtype=torch.int64)
                    )
                    torch.testing.assert_close(
                        swa_slots,
                        torch.tensor(
                            [2 * x if x > 0 else -1 for x in expected],
                            dtype=torch.int64,
                        ),
                    )
                    self.assertEqual(
                        pointers, (full_slots.data_ptr(), swa_slots.data_ptr())
                    )
                self.assertEqual(
                    backend._prepare_welm_decode_metadata_device.call_count,
                    4 if captured_translation else 0,
                )

    def test_non_decode_speculative_and_misaligned_inputs_do_not_offer_slots(self):
        changes = (
            {"forward_mode": ForwardMode.EXTEND},
            {"forward_mode": ForwardMode.TARGET_VERIFY},
            {"forward_mode": ForwardMode.IDLE},
            {"spec_info": object()},
            {"is_extend_in_batch": True},
            {"batch_size": 2},
            {"positions": torch.zeros(2, dtype=torch.int64)},
            {"positions": torch.zeros(4, dtype=torch.int32)},
            {"positions": torch.zeros(8, dtype=torch.int64)[::2]},
            {"seq_lens": torch.zeros(2, dtype=torch.int64)},
            {"seq_lens": torch.zeros(4, dtype=torch.float32)},
            {"out_cache_loc": torch.zeros(3, dtype=torch.int32)},
        )
        for change in changes:
            with self.subTest(change=change):
                backend = self.make_backend(swa=False)
                batch = self.make_batch()
                vars(batch).update(change)
                backend._prepare_welm_static_decode_slots(batch, allocate=True)
                self.assertIsNone(backend.forward_metadata.welm_static_decode_full_slots)
                self.assertIsNone(backend.forward_metadata.welm_static_decode_swa_slots)

    def test_missing_swa_translation_and_disabled_feature_do_not_offer_slots(self):
        backend = self.make_backend()
        backend._prepare_welm_static_decode_slots(self.make_batch(), allocate=True)
        self.assertIsNone(backend.forward_metadata.welm_static_decode_full_slots)
        backend.use_welm_static_decode_qkv = False
        backend.forward_metadata.swa_out_cache_loc = torch.ones(4, dtype=torch.int64)
        with patch.object(
            torch, "empty_like", side_effect=AssertionError("disabled allocation")
        ):
            backend._prepare_welm_static_decode_slots(self.make_batch(), allocate=True)
        self.assertIsNone(backend.forward_metadata.welm_static_decode_full_slots)


class TestGenerateMaskFlag(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_bool(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertEqual(mask.dtype, torch.bool)

    def test_upper_triangular_pattern(self):
        """generate_mask_flag returns ~tril, i.e. True above the diagonal."""
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[0, 3].item())
        self.assertTrue(mask[2, 3].item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_diagonal_is_false(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(5)
        for i in range(5):
            self.assertFalse(mask[i, i].item())

    def test_1x1(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(1)
        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(mask[0, 0].item())

    def test_symmetric_upper(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_mask_flag(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertTrue(mask[i, j].item())
                self.assertFalse(mask[j, i].item())


class TestGenerateAttnMask(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(8, "norm", torch.float16)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)

    def test_default_dtype_float16(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm")
        self.assertEqual(mask.dtype, torch.float16)

    def test_mix_mode_float16(self):
        """mix + float16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[1, 0].item(), 0.0)

    def test_mix_mode_bfloat16(self):
        """mix + bfloat16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[1, 1].item(), 0.0)

    def test_norm_mode_float16(self):
        """norm + float16 -> upper triangle is -inf (overflow), lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_bfloat16(self):
        """norm + bfloat16 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_float32(self):
        """norm + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_mix_mode_float32(self):
        """mix + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_lower_triangle_all_zero(self):
        """The lower triangle (including diagonal) is always zero."""
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0.0)

    def test_upper_triangle_all_masked(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1.0)

    def test_diagonal_is_zero(self):
        """Diagonal is always zero regardless of mode/dtype."""
        for mode in ("mix", "norm"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                mask = AscendAttnMaskBuilder.generate_attn_mask(4, mode, dtype)
                for i in range(4):
                    self.assertEqual(mask[i, i].item(), 0.0)


class TestGetAttentionMaskId(unittest.TestCase):
    def test_flat_tensor(self):
        """Produces a flat tensor of arange ranges concatenated."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([7, 8, 9, 15, 16, 17, 18, 19])
        self.assertEqual(result.dim(), 1)
        self.assertTrue(torch.equal(result, expected))

    def test_single_sequence(self):
        seq_lens = torch.tensor([5])
        extend_lens = torch.tensor([2])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([3, 4])
        self.assertTrue(torch.equal(result, expected))

    def test_multiple_sequences(self):
        seq_lens = torch.tensor([3, 6, 10])
        extend_lens = torch.tensor([1, 2, 4])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([2, 4, 5, 6, 7, 8, 9])
        self.assertTrue(torch.equal(result, expected))

    def test_total_length(self):
        """Result length equals sum of extend_lens per row."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected_len = 3 + 5
        self.assertEqual(result.numel(), expected_len)

    def test_values_correct(self):
        seq_lens = torch.tensor([4, 8])
        extend_lens = torch.tensor([2, 3])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 3)
        self.assertEqual(result[2].item(), 5)
        self.assertEqual(result[4].item(), 7)


class TestUpdateAttnCache(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_seqlen_greater_than_cached(self):
        """When seqlen > cached, the mask is regenerated and cache updated."""
        old_mask = torch.zeros(4, 4)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.shape, (8, 8))
        self.assertEqual(result_mask.dtype, torch.float16)

    def test_seqlen_less_equal_cached(self):
        """When seqlen <= cached, the existing mask is kept."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_dtype_change(self):
        """When dtype differs, the mask is converted but not regenerated."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.dtype, torch.float32)
        self.assertIsNot(result_mask, old_mask)

    def test_no_change(self):
        """When seqlen <= cached and dtype matches, nothing changes."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_seqlen_greater_and_dtype_change(self):
        """Both regeneration and dtype conversion happen."""
        old_mask = torch.zeros(4, 4, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=16,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 16)
        self.assertEqual(result_mask.shape, (16, 16))
        self.assertEqual(result_mask.dtype, torch.float32)

    def test_seqlen_equal_cached_no_regen(self):
        """seqlen == cached should not trigger regeneration."""
        old_mask = torch.ones(8, 8, dtype=torch.float32)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="mix",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)


class TestGetSplitfuseAttnMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        mask = self.builder.get_splitfuse_attn_mask(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_int8(self):
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask.dtype, torch.int8)

    def test_upper_triangular(self):
        """Upper triangle (excluding diagonal) is 1, rest is 0."""
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask[0, 1].item(), 1)
        self.assertEqual(mask[0, 3].item(), 1)
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertEqual(mask[1, 0].item(), 0)
        self.assertEqual(mask[1, 1].item(), 0)

    def test_lower_triangle_zero(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0)

    def test_upper_triangle_one(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1)


class TestGetSwaMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        """Output shape is (batch, 1, s2)."""
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_1d_input_unsqueezed(self):
        """1-D input of shape (B,) is handled; output still (B, 1, s2)."""
        seq_lens = torch.tensor([3, 6, 9])
        mask = self.builder.get_swa_mask(seq_lens, s2=12, left_context=512)
        self.assertEqual(mask.shape, (3, 1, 12))

    def test_2d_input(self):
        """2-D input of shape (B, 1) is also accepted."""
        seq_lens = torch.tensor([[5], [10]])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_mask_values_large_left_context(self):
        """With left_context >= max seq_len, only indices >= seq_lens are True."""
        seq_lens = torch.tensor([3, 5])
        mask = self.builder.get_swa_mask(seq_lens, s2=8, left_context=512)
        row0 = mask[0, 0]
        self.assertFalse(row0[0].item())
        self.assertFalse(row0[2].item())
        self.assertTrue(row0[3].item())
        self.assertTrue(row0[7].item())
        row1 = mask[1, 0]
        self.assertFalse(row1[4].item())
        self.assertTrue(row1[5].item())
        self.assertTrue(row1[7].item())

    def test_mask_values_small_left_context(self):
        """With a small left_context, earlier positions are also masked."""
        seq_lens = torch.tensor([10, 20])
        mask = self.builder.get_swa_mask(seq_lens, s2=30, left_context=5)
        row0 = mask[0, 0]
        self.assertTrue(row0[0].item())
        self.assertTrue(row0[4].item())
        self.assertFalse(row0[5].item())
        self.assertFalse(row0[9].item())
        self.assertTrue(row0[10].item())
        self.assertTrue(row0[29].item())
        row1 = mask[1, 0]
        self.assertTrue(row1[0].item())
        self.assertTrue(row1[14].item())
        self.assertFalse(row1[15].item())
        self.assertFalse(row1[19].item())
        self.assertTrue(row1[20].item())

    def test_clamp_to_zero(self):
        """When seq_len < left_context, start is clamped to 0."""
        seq_lens = torch.tensor([2])
        mask = self.builder.get_swa_mask(seq_lens, s2=5, left_context=10)
        row = mask[0, 0]
        self.assertFalse(row[0].item())
        self.assertFalse(row[1].item())
        self.assertTrue(row[2].item())
        self.assertTrue(row[4].item())

    def test_default_left_context(self):
        """Default left_context is 512."""
        seq_lens = torch.tensor([10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15)
        self.assertEqual(mask.shape, (1, 1, 15))
        row = mask[0, 0]
        self.assertFalse(row[9].item())
        self.assertTrue(row[10].item())

    def test_dtype_bool(self):
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.dtype, torch.bool)


class TestCanUseTnd(unittest.TestCase):
    def test_128_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=128)
            )
        )

    def test_192_192(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=192)
            )
        )

    def test_256_256(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=256)
            )
        )

    def test_192_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=128)
            )
        )

    def test_64_64(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=64, v_head_dim=64)
            )
        )

    def test_128_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=256)
            )
        )

    def test_256_128(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=128)
            )
        )

    def test_128_192(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=192)
            )
        )

    def test_192_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=256)
            )
        )

    def test_96_96(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=96, v_head_dim=96)
            )
        )


class TestGenerateAlibiBias(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)

    def test_shape(self):
        """Output shape is (num_heads, 1, seq_len)."""
        slopes = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = self.backend._generate_alibi_bias(
            seq_len=8,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (4, 1, 8))

    def test_values(self):
        """Each element is slopes[h] * position."""
        slopes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        seq_len = 5
        result = self.backend._generate_alibi_bias(
            seq_len=seq_len,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(4):
            for p in range(seq_len):
                expected = slopes[h].item() * p
                self.assertAlmostEqual(result[h, 0, p].item(), expected, places=5)

    def test_dtype(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_single_head(self):
        slopes = torch.tensor([1.5])
        result = self.backend._generate_alibi_bias(
            seq_len=3,
            slopes=slopes,
            num_heads=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 0, 1].item(), 1.5)
        self.assertAlmostEqual(result[0, 0, 2].item(), 3.0)

    def test_zero_position_is_zero(self):
        """Position 0 always yields 0 regardless of slopes."""
        slopes = torch.tensor([1.0, 2.0, 3.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(3):
            self.assertAlmostEqual(result[h, 0, 0].item(), 0.0)

    def test_default_dtype_bfloat16(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.dtype, torch.bfloat16)


class TestGetCudaGraphSeqLenFillValue(unittest.TestCase):
    def test_returns_zero(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertEqual(backend.get_cuda_graph_seq_len_fill_value(), 0)


class TestGetVerifyBuffers(unittest.TestCase):
    def test_no_verify_mask(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertIsNone(backend.verify_mask)

    def test_update_is_noop(self):
        backend = object.__new__(AscendAttnBackend)
        backend.update_verify_buffers_to_fill_after_draft(None, None)
        backend.update_verify_buffers_to_fill_after_draft(MagicMock(), 4)
        backend.update_verify_buffers_to_fill_after_draft(None, 16)


class TestCommonTemplate(unittest.TestCase):
    @staticmethod
    def _make_draft_backend(speculative_num_steps):
        backend = object.__new__(AscendAttnMultiStepDraftBackend)
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_calls_fn_for_each_step(self):
        """call_fn is invoked for steps 0..speculative_num_steps-2."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 3)
        for i in range(3):
            call_fn.assert_any_call(i, forward_batch)

    def test_zero_steps(self):
        """speculative_num_steps=1 -> no calls (range(0))."""
        backend = self._make_draft_backend(speculative_num_steps=1)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_not_called()

    def test_two_steps(self):
        """speculative_num_steps=2 -> exactly one call with index 0."""
        backend = self._make_draft_backend(speculative_num_steps=2)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_called_once_with(0, forward_batch)

    def test_call_indices(self):
        backend = self._make_draft_backend(speculative_num_steps=5)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        indices = []
        backend.common_template(forward_batch, lambda i, fb: indices.append(i))
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_assert_spec_info_not_none(self):
        """Raises AssertionError when forward_batch.spec_info is None."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = None
        with self.assertRaises(AssertionError):
            backend.common_template(forward_batch, MagicMock())

    def test_passes_same_forward_batch(self):
        backend = self._make_draft_backend(speculative_num_steps=3)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        for call in call_fn.call_args_list:
            self.assertIs(call.args[1], forward_batch)


if __name__ == "__main__":
    unittest.main()
