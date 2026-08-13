"""Host-side contract tests for the fused NPU paged KV-cache write.

These tests use CPU tensors and replace ``torch_npu`` with a recording stub, so
they validate the memory-pool wiring without requiring an NPU device or CANN.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingTorchNPU:
    def __init__(self):
        self.calls = []

    def npu_scatter_pa_kv_cache(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestNPUMHATokenToKVPoolScatter(unittest.TestCase):
    HEAD_NUM = 2
    HEAD_DIM = 4
    V_HEAD_DIM = 6
    PAGE_SIZE = 4
    NUM_PAGES = 3
    NUM_LAYERS = 2
    START_LAYER = 10

    def _make_bare_pool(self, use_fia: bool):
        pool = object.__new__(NPUMHATokenToKVPool)
        pool.dtype = torch.bfloat16
        pool.store_dtype = torch.bfloat16
        pool.head_num = self.HEAD_NUM
        pool.head_dim = self.HEAD_DIM
        pool.v_head_dim = self.V_HEAD_DIM
        pool.page_size = self.PAGE_SIZE
        pool.start_layer = self.START_LAYER
        pool.use_fia = use_fia

        k_buffer = torch.zeros(
            self.NUM_LAYERS,
            self.NUM_PAGES,
            self.PAGE_SIZE,
            self.HEAD_NUM,
            self.HEAD_DIM,
            dtype=pool.store_dtype,
        )
        v_buffer = torch.zeros(
            self.NUM_LAYERS,
            self.NUM_PAGES,
            self.PAGE_SIZE,
            self.HEAD_NUM,
            self.V_HEAD_DIM,
            dtype=pool.store_dtype,
        )
        if use_fia:
            pool.k_buffer = [
                k_buffer[i].view(-1, 1, self.HEAD_NUM, self.HEAD_DIM)
                for i in range(self.NUM_LAYERS)
            ]
            pool.v_buffer = [
                v_buffer[i].view(-1, 1, self.HEAD_NUM, self.V_HEAD_DIM)
                for i in range(self.NUM_LAYERS)
            ]
        else:
            pool.k_buffer = k_buffer
            pool.v_buffer = v_buffer
        return pool

    def _assert_fused_write(self, use_fia: bool):
        pool = self._make_bare_pool(use_fia)
        recorder = _RecordingTorchNPU()
        loc = torch.tensor([1, 5, 11], dtype=torch.int64)

        # Model QKV splits are commonly non-contiguous along the token axis.
        # Keep that production-like stride and ensure the pool does not force a
        # contiguous copy before invoking ScatterPaKvCache.
        qkv = torch.randn(3, 32, dtype=pool.dtype)
        cache_k = qkv[:, 3 : 3 + self.HEAD_NUM * self.HEAD_DIM]
        cache_v = qkv[:, 18 : 18 + self.HEAD_NUM * self.V_HEAD_DIM]
        self.assertFalse(cache_k.is_contiguous())
        self.assertFalse(cache_v.is_contiguous())

        with mock.patch.object(
            memory_pool_npu, "torch_npu", recorder, create=True
        ):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=self.START_LAYER + 1),
                loc,
                cache_k,
                cache_v,
            )

        self.assertEqual(len(recorder.calls), 1)
        args, kwargs = recorder.calls[0]
        key, value, key_cache, value_cache, slot_mapping = args
        self.assertEqual(tuple(key.shape), (3, self.HEAD_NUM, self.HEAD_DIM))
        self.assertEqual(tuple(value.shape), (3, self.HEAD_NUM, self.V_HEAD_DIM))
        self.assertEqual(
            tuple(key_cache.shape),
            (self.NUM_PAGES, self.PAGE_SIZE, self.HEAD_NUM, self.HEAD_DIM),
        )
        self.assertEqual(
            tuple(value_cache.shape),
            (self.NUM_PAGES, self.PAGE_SIZE, self.HEAD_NUM, self.V_HEAD_DIM),
        )
        self.assertIs(slot_mapping, loc)
        self.assertEqual(slot_mapping.dtype, torch.int64)
        self.assertEqual(kwargs, {"cache_mode": "Norm"})

    def test_fia_layout_uses_one_fused_norm_scatter(self):
        self._assert_fused_write(use_fia=True)

    def test_non_fia_layout_uses_one_fused_norm_scatter(self):
        self._assert_fused_write(use_fia=False)

    def test_layer_override_selects_swa_subpool_layer(self):
        pool = self._make_bare_pool(use_fia=True)
        recorder = _RecordingTorchNPU()
        loc = torch.tensor([2], dtype=torch.int64)
        cache_k = torch.zeros(1, self.HEAD_NUM, self.HEAD_DIM, dtype=pool.dtype)
        cache_v = torch.zeros(1, self.HEAD_NUM, self.V_HEAD_DIM, dtype=pool.dtype)

        with mock.patch.object(
            memory_pool_npu, "torch_npu", recorder, create=True
        ):
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                layer_id_override=self.START_LAYER,
            )

        self.assertEqual(len(recorder.calls), 1)
        args, kwargs = recorder.calls[0]
        self.assertEqual(
            tuple(args[2].shape),
            (self.NUM_PAGES, self.PAGE_SIZE, self.HEAD_NUM, self.HEAD_DIM),
        )
        self.assertIs(args[4], loc)
        self.assertEqual(kwargs, {"cache_mode": "Norm"})

    def test_float8_uint8_storage_is_reinterpreted_as_int8(self):
        pool = self._make_bare_pool(use_fia=True)
        pool.dtype = torch.float8_e4m3fn
        pool.store_dtype = torch.uint8
        pool.k_buffer = [
            torch.zeros(
                self.NUM_PAGES * self.PAGE_SIZE,
                1,
                self.HEAD_NUM,
                self.HEAD_DIM,
                dtype=pool.store_dtype,
            )
            for _ in range(self.NUM_LAYERS)
        ]
        pool.v_buffer = [
            torch.zeros(
                self.NUM_PAGES * self.PAGE_SIZE,
                1,
                self.HEAD_NUM,
                self.V_HEAD_DIM,
                dtype=pool.store_dtype,
            )
            for _ in range(self.NUM_LAYERS)
        ]
        recorder = _RecordingTorchNPU()
        loc = torch.tensor([3], dtype=torch.int32)
        cache_k = torch.zeros(
            1, self.HEAD_NUM, self.HEAD_DIM, dtype=pool.dtype
        )
        cache_v = torch.zeros(
            1, self.HEAD_NUM, self.V_HEAD_DIM, dtype=pool.dtype
        )

        with mock.patch.object(
            memory_pool_npu, "torch_npu", recorder, create=True
        ):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=self.START_LAYER),
                loc,
                cache_k,
                cache_v,
            )

        args, kwargs = recorder.calls[0]
        self.assertEqual(args[0].dtype, torch.int8)
        self.assertEqual(args[1].dtype, torch.int8)
        self.assertEqual(args[2].dtype, torch.int8)
        self.assertEqual(args[3].dtype, torch.int8)
        self.assertIs(args[4], loc)
        self.assertEqual(kwargs, {"cache_mode": "Norm"})


if __name__ == "__main__":
    unittest.main()
