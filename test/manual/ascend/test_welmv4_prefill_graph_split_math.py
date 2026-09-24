"""CPU math/lifetime checks of production split helpers (numpy, no torch/NPU).

Device operators are replaced by numpy equivalents; these tests do not prove
kernel numerics, graph capture, or HCCL correctness.
"""

import types
import unittest
from unittest.mock import patch

import numpy as np

from test_welmv4_prefill_graph_contracts import ADAPTER, SRT, Adapter, load_method


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        return self.data.size

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def __sub__(self, other):
        return Tensor(self.data - other)

    def __eq__(self, other):
        return Tensor(self.data == other)

    def to(self, dtype):
        return Tensor(self.data.astype(dtype))

    def index_select(self, dim, indices):
        return Tensor(np.take(self.data, indices.data, axis=dim))

    def copy_(self, source):
        self.data[...] = source.data
        return self

    def clone(self):
        return Tensor(self.data.copy())

    def new_empty(self, shape):
        return Tensor(np.empty(shape, dtype=self.dtype))

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.dtype))

    def narrow(self, dim, start, length):
        assert dim == 0
        return self[start : start + length]


TORCH = types.SimpleNamespace(
    float32=np.dtype("float32"), long=np.int64,
    div=lambda x, y, **kw: Tensor(x.data // y),
    where=lambda mask, x, y: Tensor(np.where(mask.data, x.data, y.data)),
    zeros_like=lambda x: Tensor(np.zeros_like(x.data)),
    empty_like=lambda x: Tensor(np.empty_like(x.data)),
)


class TestSplitMath(unittest.TestCase):
    def test_ep_first_consumer_partials_preserve_exact_selected_fp32_residual(self):
        path = SRT / "models/welmv4.py"
        for tp in (2, 4, 8):
            for capacity, lengths in ((32, [1, 7, 9, 3]), (64, [4, 8]), (128, [113])):
                residual = np.arange(capacity * 3, dtype=np.float32).reshape(capacity, 3) + 0.125
                hidden = residual / 2
                tails = np.cumsum(lengths) - 1
                indices = Tensor(np.pad(tails, (0, 8 - len(tails))).astype(np.int64))
                partials = []
                for rank in range(tp):
                    local = capacity // tp
                    ns = {"torch": TORCH,
                          "get_tensor_model_parallel_world_size": lambda: tp,
                          "get_parallel": lambda: types.SimpleNamespace(tp_rank=rank)}
                    partial = load_method(path, "Qwen2MoeDecoderLayer",
                                          "_build_kv_mirror_residual_partial", ns)
                    prepare = load_method(path, "Qwen2MoeDecoderLayer",
                                          "prepare_graph_mirror_input", ns)
                    first = types.SimpleNamespace(
                        ppln=True, layer_id=4, prenorm_layer_idx=[],
                        _use_npu_prefill_deepep_scattered=lambda *args: True,
                        _npu_prefill_deepep_prepare_attention=lambda h, r, **kw: (Tensor(hidden), r),
                        _build_kv_mirror_residual_partial=partial,
                    )
                    out_h, out_r, is_partial = prepare(
                        first, Tensor(hidden[rank * local : (rank + 1) * local]),
                        Tensor(residual[rank * local : (rank + 1) * local]),
                        object(), indices,
                    )
                    self.assertTrue(is_partial)
                    np.testing.assert_array_equal(out_h.data[:len(tails)], hidden[tails])
                    self.assertEqual(out_r.dtype, np.dtype("float32"))
                    partials.append(out_r.data[:len(tails)])
                np.testing.assert_array_equal(sum(partials), residual[tails])

    def test_pure_tp_norm_runs_once_before_tail_selection(self):
        prepare = load_method(SRT / "models/welmv4.py", "Qwen2MoeDecoderLayer",
                              "prepare_graph_mirror_input", {"torch": TORCH})
        h = Tensor(np.arange(36, dtype=np.float32).reshape(12, 3))
        r = Tensor(h.data + 100)
        indices = Tensor(np.array([3, 9, 0, 0]))
        for after_norm in (False, True):
            calls = []

            def norm(hidden, residual, **kwargs):
                calls.append((hidden.shape, kwargs))
                result = (Tensor(hidden.data + 2), Tensor(residual.data + 3))
                return (result[0], None, result[1]) if after_norm else result

            first = types.SimpleNamespace(
                ppln=after_norm, layer_id=4, prenorm_layer_idx=[],
                input_layernorm=norm,
                _use_npu_prefill_deepep_scattered=lambda *args: False,
            )
            norm.weight = Tensor(np.ones(3, dtype=np.float32))
            out_h, out_r, partial = prepare(first, h, r, object(), indices)
            self.assertFalse(partial)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], (12, 3))
            np.testing.assert_array_equal(out_h.data, (h.data + 2)[indices.data])
            np.testing.assert_array_equal(out_r.data, (r.data + 3)[indices.data])

    def test_prompt_outputs_keep_addresses_and_preserve_raw_mtp_across_t_b_changes(self):
        adapter = Adapter.__new__(Adapter)
        adapter.max_tokens = 16
        adapter.first_mirror = 1
        adapter.mirror_hidden = adapter.mirror_residual = None
        adapter.mirror_positions = Tensor(np.zeros(4, dtype=np.int64))
        adapter.mirror_kv = {7: (Tensor(np.zeros((16, 2))), Tensor(np.zeros((16, 2))))}
        adapter.mtp_kv = {}
        adapter.captured_states = {}
        adapter.capture_keys = set()
        adapter.prune, adapter.ep = True, False
        seen = []

        def prepare_key(positions, key, batch):
            seen.append(key.shape[0])
            key.data[...] += positions.data[:, None]  # stand-in consumer transform

        attn = types.SimpleNamespace(
            kv_mirror_imitated_layers=[0], kv_mirror_layers=[1], kv_mirror_layer_idx=1,
            attn=types.SimpleNamespace(layer_id=7), prepare_graph_mirror_key=prepare_key,
        )
        first = types.SimpleNamespace(
            self_attn=attn,
            prepare_graph_mirror_input=lambda h, r, b, idx: (
                h.index_select(0, idx), r.index_select(0, idx), False
            ),
        )
        adapter.model = types.SimpleNamespace(layers=[None, first], end_layer=2)
        module = types.ModuleType("sglang.srt.models.welmv4")
        module.WELMV4_MTP_MIRROR_STATES_KEY = "mtp"
        addresses = None
        with patch.dict("sys.modules", {module.__name__: module}), patch.dict(ADAPTER, torch=TORCH):
            for capacity, tails in ((8, [3, 7]), (16, [14]), (8, [1, 2, 4, 7])):
                raw = Tensor(np.arange(capacity * 2, dtype=np.float64).reshape(capacity, 2))
                saved = raw.data.copy()
                module.KVMirrorManager = types.SimpleNamespace(get_kv_activation=lambda source: (raw, raw))
                positions = Tensor(np.arange(capacity) + 23)
                adapter.tail_indices = Tensor(np.array(tails + [0] * (4 - len(tails))))
                batch = types.SimpleNamespace(
                    model_specific_states={"mtp": {2: (raw, raw)}},
                    welm_prefill_graph_phase="prompt",
                )
                self.assertIsNone(adapter.finish_prompt(raw, raw, positions, batch))
                adapter.after_capture(adapter.key(capacity), batch)
                np.testing.assert_array_equal(raw.data, saved)
                np.testing.assert_array_equal(adapter.mirror_kv[7][0].data[:capacity], saved + positions.data[:, None])
                np.testing.assert_array_equal(batch.model_specific_states["mtp"][2][0].data, saved)
                current = tuple(t.data.ctypes.data for t in (
                    adapter.mirror_hidden, adapter.mirror_residual,
                    adapter.mirror_kv[7][0], adapter.mtp_kv[2][0],
                ))
                if addresses is not None:
                    self.assertEqual(current, addresses)
                addresses = current
                h, r, p = adapter.begin_mirror(types.SimpleNamespace(batch_size=len(tails)))
                np.testing.assert_array_equal(h.data, saved[tails])
                np.testing.assert_array_equal(p.data, positions.data[tails])
                r.data.fill(-99)
                np.testing.assert_array_equal(adapter.mirror_residual.data[:len(tails)], saved[tails])
        self.assertEqual(seen, [8, 16, 8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
