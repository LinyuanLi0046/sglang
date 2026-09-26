"""CPU execution of production Flash metadata/routing (numpy, no torch/NPU).

Checks buffer lifetime, T/B factorization, cache routing, and unchanged default
eager/decode calls. The separate NPU smoke test checks the actual operators.
"""

import copy
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_welmv4_prefill_graph_contracts import ADAPTER, SRT, Adapter, load_method
from test_welmv4_prefill_graph_split_math import Tensor


def tensor(values, *, dtype=np.int32, **kwargs):
    return Tensor(np.array(values, dtype=dtype))


TORCH = types.SimpleNamespace(
    int32=np.int32, int64=np.int64,
    tensor=tensor,
    zeros=lambda shape, *, dtype=np.float32, **kw: Tensor(np.zeros(shape, dtype=dtype)),
    ones=lambda shape, *, dtype=np.float32, **kw: Tensor(np.ones(shape, dtype=dtype)),
    empty=lambda shape, *, dtype=np.float32, **kw: Tensor(np.empty(shape, dtype=dtype)),
    zeros_like=lambda x: Tensor(np.zeros_like(x.data)),
    arange=lambda n, *, dtype=np.int32, **kw: Tensor(np.arange(n, dtype=dtype)),
    cat=lambda xs: Tensor(np.concatenate([x.data for x in xs])),
)


def metadata(**kwargs):
    result = dict(welm_flash_schedules={}, welm_flash_mirror_q_lengths=None,
                  welm_flash_cu_seqlens_q=None, welm_flash_max_seqlen_q=-1)
    result.update(kwargs)
    return types.SimpleNamespace(**result)


def make_backend(hybrid=True):
    path = SRT / "hardware_backend/npu/attention/ascend_backend.py"
    calls, writes = [], []
    npu = types.SimpleNamespace(npu_scatter_pa_kv_cache=lambda *a, **kw: writes.append(a))
    backend = types.SimpleNamespace(
        device="cpu", max_context_len=256, page_size=64, is_hybrid_swa=hybrid,
        use_sliding_window_kv_pool=hybrid,
        forward_metadata=object(), graph_metadata=object(), graph_mode=False,
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=tensor(np.arange(4 * 256).reshape(4, 256))
        ),
        full_to_swa_index_mapping=tensor(np.arange(1024) + 128),
        welm_flash_attn_mask=object(),
        _is_swa_layer=lambda layer: hybrid and layer.layer_id == 1,
        _has_layerwise_sliding_window=lambda layer: layer.sliding_window_size >= 0,
    )
    cache = Tensor(np.zeros((1024, 1, 2, 4), dtype=np.float32))
    backend.token_to_kv_pool = types.SimpleNamespace(
        get_key_buffer=lambda _: cache, get_value_buffer=lambda _: cache,
        translate_loc_from_full_to_swa=lambda x: Tensor(x.data + 128),
    )
    backend._welm_flash_attn_metadata = lambda *a, **kw: calls.append(("schedule", kw)) or object()
    backend._welm_flash_attn = lambda q, *a, **kw: (
        calls.append(("flash", q.shape[0], kw)) or q.clone(), None
    )
    for name in ("create_welm_prefill_graph_metadata", "prepare_welm_prefill_graph_metadata",
                 "write_welm_prefill_graph_kv", "_forward_welm_flash_attention"):
        fn = load_method(path, "AscendAttnBackend", name,
                         {"torch": TORCH, "torch_npu": npu, "ForwardMetadata": metadata})
        setattr(backend, name, types.MethodType(fn, backend))
    return backend, calls, writes


def batch(lengths, kv_lengths, rows=None):
    b = len(lengths)
    return types.SimpleNamespace(
        batch_size=b, seq_lens_cpu=tensor(kv_lengths), seq_lens=tensor(kv_lengths),
        extend_seq_lens_cpu=lengths, extend_seq_lens=tensor(lengths),
        req_pool_indices=tensor(list(range(b)) if rows is None else rows),
    )


def layer(swa=False):
    return types.SimpleNamespace(layer_id=int(swa), tp_q_head_num=4, tp_k_head_num=2,
                                 tp_v_head_num=2, qk_head_dim=4, v_head_dim=4,
                                 sliding_window_size=31 if swa else -1, scaling=0.5)


class TestFlashMetadata(unittest.TestCase):
    def test_page_tables_lengths_and_mirror_views_update_without_global_state(self):
        backend, _, _ = make_backend()
        state = backend.create_welm_prefill_graph_metadata(4)
        mirror_tables = state.block_tables[:2]
        mirror_lens = state.welm_flash_seqused_kv[:2]
        pointers = [x.data.ctypes.data for x in
                    (state.block_tables, state.block_tables_swa, state.welm_flash_seqused_kv)]
        original = (backend.forward_metadata, backend.graph_metadata, backend.graph_mode)
        for lengths, kv, rows in (([5, 7], [133, 199], [2, 1]),
                                 ([1], [33], [3]), ([5, 7], [133, 199], [2, 1])):
            live = batch(lengths, kv, rows)
            backend.prepare_welm_prefill_graph_metadata(state, live)
            b, pages = len(lengths), (max(kv) + 63) // 64
            expected = np.zeros((4, 4), dtype=np.int32)
            expected[:b, :pages] = np.array(rows)[:, None] * 4 + np.arange(pages)
            np.testing.assert_array_equal(state.block_tables.data, expected)
            expected[:b, :pages] += 2
            np.testing.assert_array_equal(state.block_tables_swa.data, expected)
            np.testing.assert_array_equal(state.welm_flash_seqused_q.data, lengths + [0] * (4-b))
            np.testing.assert_array_equal(mirror_lens.data, (kv + [0]*4)[:2])
            np.testing.assert_array_equal(mirror_tables.data, state.block_tables.data[:2])
            self.assertEqual(pointers, [x.data.ctypes.data for x in
                             (state.block_tables, state.block_tables_swa, state.welm_flash_seqused_kv)])
            self.assertEqual(original, (backend.forward_metadata, backend.graph_metadata, backend.graph_mode))

    def test_capture_uses_zero_page_and_capacity_errors_precede_updates(self):
        backend, _, _ = make_backend(hybrid=False)
        state = backend.create_welm_prefill_graph_metadata(2)
        backend.req_to_token_pool = None  # Capture must not dereference it.
        backend.prepare_welm_prefill_graph_metadata(state, batch([3, 7], [3, 7]), capture=True)
        self.assertIsNone(state.block_tables_swa)
        self.assertFalse(np.any(state.block_tables.data))
        for live in (batch([1], [257]), batch([1, 1, 1], [1, 1, 1])):
            with self.assertRaises(ValueError):
                backend.prepare_welm_prefill_graph_metadata(state, live)

    def test_explicit_graph_metadata_and_default_eager_decode_stay_independent(self):
        backend, calls, _ = make_backend()
        state = backend.create_welm_prefill_graph_metadata(4)
        backend.prepare_welm_prefill_graph_metadata(state, batch([3, 2], [70, 130]))
        state.welm_flash_cu_seqlens_q = tensor([0, 3, 5, 5, 8])
        state.extend_seq_lens_cpu_int = None  # Graph must not access CPU lengths.
        q = Tensor(np.ones((8, 4, 4)))
        cache = backend.token_to_kv_pool.get_key_buffer(0)
        original = backend.forward_metadata
        for swa in (False, True, False):
            out = backend._forward_welm_flash_attention(q, cache, cache, layer(swa), None,
                                                       graph_metadata=state)
            self.assertEqual(out.shape, (8, 16))
            call = calls[-1]
            self.assertEqual(call[1], 8)
            self.assertEqual(call[2]["max_seqlen_q"], -1)
            self.assertIs(call[2]["block_table"], state.block_tables_swa if swa else state.block_tables)
            self.assertIs(backend.forward_metadata, original)
            self.assertFalse(backend.graph_mode)
        self.assertEqual(sum(c[0] == "schedule" for c in calls), 2)

        # Default calls preserve eager real-row and decode graph-row selection.
        eager = copy.copy(state)
        eager.welm_flash_schedules = {}
        eager.welm_flash_max_seqlen_q = 3
        eager.extend_seq_lens_cpu_int = tensor([3, 2])
        backend.forward_metadata = eager
        backend._forward_welm_flash_attention(q[:5], cache, cache, layer(), None)
        self.assertEqual(calls[-1][1], 5)
        self.assertEqual(calls[-1][2]["max_seqlen_q"], 3)
        backend.graph_mode = True
        backend._forward_welm_flash_attention(q, cache, cache, layer(), None)
        self.assertEqual(calls[-1][1], 8)
        self.assertIs(backend.forward_metadata, eager)

    def test_writer_selects_full_or_swa_slots_and_preserves_negative_padding(self):
        backend, _, writes = make_backend()
        kv = Tensor(np.ones((8, 2, 4)))
        full, swa = tensor([64, 65, 66, -1, -1, -1, -1, -1]), tensor([192, 193, 194, -1, -1, -1, -1, -1])
        for is_swa in (False, True):
            backend.write_welm_prefill_graph_kv(layer(is_swa), kv, kv, full, swa)
            np.testing.assert_array_equal(writes[-1][4].data, (swa if is_swa else full).data)
            self.assertEqual(writes[-1][0].shape, (8, 2, 4))
            self.assertEqual(writes[-1][2].shape, (16, 64, 2, 4))

    def test_real_capture_preparation_has_t_only_and_exact_b_metadata(self):
        self._check_capture_preparation(pad_mirror=False)

    def test_padded_mirror_metadata_tracks_b_without_changing_addresses(self):
        self._check_capture_preparation(pad_mirror=True)

    def _check_capture_preparation(self, pad_mirror):
        backend, _, _ = make_backend()
        adapter = Adapter.__new__(Adapter)
        adapter.pad_mirror = pad_mirror
        adapter.mirror_q_used = tensor([0] * 4)
        mirror_bs = 4 if pad_mirror else 2
        adapter.backend, adapter.device, adapter.prune = backend, "cpu", True
        adapter.max_requests, adapter.max_tokens = 4, 16
        adapter.flash_inputs = backend.create_welm_prefill_graph_metadata(4)
        adapter.flash_metadata, adapter.rope_tiles, adapter._capture_templates = {}, {}, {}
        adapter.oe_ids = tensor(np.zeros((0, 16)))
        adapter.valid_rows = tensor(np.zeros(16))
        adapter.full_write_locs = tensor(np.zeros(16))
        adapter.swa_write_locs = tensor(np.zeros(16))
        adapter.local_valid_rows = tensor([0])
        adapter.tail_indices = tensor(np.zeros(4))
        with patch.dict(ADAPTER, torch=TORCH):
            for phase, capacity, lengths in (("prompt", 16, [16]), ("prompt", 8, [8]),
                                              ("mirror", 16, [16 // mirror_bs] * mirror_bs)):
                adapter.capture_phase = phase
                fb = batch(lengths, lengths)
                fb.out_cache_loc = tensor(np.zeros(capacity))
                fb.input_ids = tensor(np.zeros(capacity))
                fb.positions = tensor(np.zeros(capacity))
                fb.num_token_non_padded = None
                fb.num_token_non_padded_cpu = capacity
                fb.global_dp_buffer_len = capacity
                fb.global_num_tokens_cpu = [capacity]
                adapter.prepare_capture(fb, capacity)
            p8, p16, m2 = (adapter.flash_metadata[k] for k in
                           (adapter.key(8), adapter.key(16), adapter.mirror_key(mirror_bs)))
            self.assertEqual(len(adapter.flash_metadata), 3)
            self.assertEqual(p8.welm_flash_cu_seqlens_q.shape, (5,))
            self.assertEqual(m2.welm_flash_cu_seqlens_q.shape, (mirror_bs + 1,))
            self.assertEqual(m2.welm_flash_seqused_kv.shape, (mirror_bs,))
            self.assertIs(p8.block_tables, p16.block_tables)
            self.assertIsNot(p8.welm_flash_schedules, m2.welm_flash_schedules)
            before = m2.welm_flash_cu_seqlens_q.data.copy()
            adapter._prepare_flash_offsets(p8, batch([2, 3], [70, 130]), 8)
            np.testing.assert_array_equal(p8.welm_flash_cu_seqlens_q.data, [0, 2, 5, 5, 8])
            np.testing.assert_array_equal(m2.welm_flash_cu_seqlens_q.data, before)
            self.assertTrue(np.all(adapter.full_write_locs.data == -1))
            adapter.model = types.SimpleNamespace(oe_grams=[])
            adapter.capture_phase = "prompt"
            original = backend.forward_metadata
            q_pointer = m2.welm_flash_seqused_q.data.ctypes.data
            cases = [(8, [2, 3], [70, 130]), (16, [10], [202]),
                     (8, [1, 2], [193, 131])]
            if pad_mirror:
                cases += [(8, [1, 1, 1, 1], [63, 64, 65, 200]),
                          (8, [1], [65]), (16, [7, 1, 1], [133, 201, 64])]
            for t, lengths, kv_lengths in cases:
                live = batch(lengths, kv_lengths)
                real = sum(lengths)
                live.out_cache_loc = tensor(np.arange(real) + 200)
                live.ngram_embedding_info = None
                live.global_num_tokens_cpu = [t]
                static = copy.copy(live)
                static.positions = tensor(np.arange(t))
                static.num_token_non_padded = None
                adapter.prepare_replay(live, static, t)
                self.assertIs(adapter.current_flash_metadata, adapter.flash_metadata[adapter.key(t)])
                np.testing.assert_array_equal(adapter.full_write_locs.data[:t],
                                              list(range(200, 200+real)) + [-1]*(t-real))
                np.testing.assert_array_equal(adapter.swa_write_locs.data[:t],
                                              list(range(328, 328+real)) + [-1]*(t-real))
                np.testing.assert_array_equal(m2.welm_flash_seqused_kv.data,
                                              (kv_lengths + [0]*4)[:mirror_bs])
                if pad_mirror:
                    np.testing.assert_array_equal(m2.welm_flash_seqused_q.data,
                                                  [1]*len(lengths) + [0]*(4-len(lengths)))
                    np.testing.assert_array_equal(m2.welm_flash_cu_seqlens_q.data, range(5))
                    self.assertEqual(m2.welm_flash_seqused_q.data.ctypes.data, q_pointer)
                    # Prompt and Mirror lengths share neither values nor storage.
                    np.testing.assert_array_equal(adapter.flash_inputs.welm_flash_seqused_q.data,
                                                  lengths + [0]*(4-len(lengths)))
                self.assertIs(backend.forward_metadata, original)
                self.assertFalse(backend.graph_mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
