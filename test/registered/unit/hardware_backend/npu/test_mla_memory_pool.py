import unittest

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUMLACompactIndexerPool(unittest.TestCase):
    @staticmethod
    def _make_pool(
        *,
        dtype=torch.bfloat16,
        indexer_layer_ids=(10, 12),
        index_head_dim=128,
        kv_cache_dim=None,
    ):
        return NPUMLATokenToKVPool(
            size=4,
            page_size=2,
            dtype=dtype,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            index_head_dim=index_head_dim,
            layer_num=4,
            device="cpu",
            enable_memory_saver=False,
            kv_cache_dim=kv_cache_dim,
            start_layer=10,
            end_layer=14,
            indexer_layer_ids=indexer_layer_ids,
        )

    def test_allocates_only_real_indexer_layers(self):
        pool = self._make_pool()

        self.assertEqual(pool.indexer_layer_ids, (10, 12))
        self.assertEqual(pool.num_indexer_layers, 2)
        self.assertEqual(pool.index_k_buffer.shape[0], 2)
        self.assertIsNone(pool.index_k_scale_buffer)
        self.assertEqual(
            pool.get_index_k_buffer(10).data_ptr(),
            pool.index_k_buffer[0].data_ptr(),
        )
        self.assertEqual(
            pool.get_index_k_buffer(12).data_ptr(),
            pool.index_k_buffer[1].data_ptr(),
        )
        with self.assertRaisesRegex(ValueError, "not a real Indexer layer"):
            pool.get_index_k_buffer(11)

    def test_none_keeps_legacy_all_layer_layout(self):
        pool = self._make_pool(indexer_layer_ids=None)

        self.assertEqual(pool.indexer_layer_ids, (10, 11, 12, 13))
        self.assertEqual(pool.index_k_buffer.shape[0], 4)

    def test_empty_mapping_allocates_no_indexer_layers(self):
        pool = self._make_pool(indexer_layer_ids=())

        self.assertEqual(pool.num_indexer_layers, 0)
        self.assertEqual(pool.index_k_buffer.shape[0], 0)

    def test_fp8_allocates_one_scale_per_real_indexer_layer(self):
        # 128 FP8 latent bytes + 64 BF16 RoPE values + one FP32 tile scale.
        packed_kv_dim = 128 + 64 * torch.bfloat16.itemsize + torch.float32.itemsize
        pool = self._make_pool(
            dtype=torch.float8_e4m3fn,
            kv_cache_dim=packed_kv_dim,
        )

        self.assertEqual(pool.index_k_buffer.shape[0], 2)
        self.assertEqual(pool.index_k_scale_buffer.shape[0], 2)
        expected_size = sum(
            tensor.nbytes
            for tensor in (
                pool.k_buffer,
                pool.v_buffer,
                pool.index_k_buffer,
                pool.index_k_scale_buffer,
            )
        )
        self.assertEqual(pool.get_kv_size_bytes(), expected_size)

    def test_legacy_fp8_layout_keeps_uniform_pd_buffer_groups(self):
        packed_kv_dim = 128 + 64 * torch.bfloat16.itemsize + torch.float32.itemsize
        pool = self._make_pool(
            dtype=torch.float8_e4m3fn,
            kv_cache_dim=packed_kv_dim,
            indexer_layer_ids=None,
        )

        data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(data_ptrs), 4 * pool.layer_num)
        self.assertEqual(len(data_lens), len(data_ptrs))
        self.assertEqual(len(item_lens), len(data_ptrs))

    def test_rejects_invalid_layer_mappings(self):
        with self.assertRaisesRegex(ValueError, "in increasing"):
            self._make_pool(indexer_layer_ids=(12, 10))
        with self.assertRaisesRegex(ValueError, "local stage range"):
            self._make_pool(indexer_layer_ids=(9,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._make_pool(indexer_layer_ids=(10, 10))


if __name__ == "__main__":
    unittest.main()
