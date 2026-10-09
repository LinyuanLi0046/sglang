"""Host checks for production MXFP8 AG storage and stream submission contracts.

Run with Python + numpy, without torch/Ascend. Production methods are loaded
from AST; fake tensors preserve byte views and allocation identities. These
tests do not validate FP8 numerics, asynchronous execution, or HCCL capture.
"""

import copy
import logging
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from test_welmv4_prefill_graph_contracts import SRT, ShapeKey, load_method, load_nodes


@dataclass(frozen=True)
class Device:
    type: str = "npu"
    index: int = 0


class Tensor:
    def __init__(self, data, device=Device()):
        self.data = np.asarray(data)
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        return self.data.size

    def data_ptr(self):
        return self.data.ctypes.data

    def __getitem__(self, item):
        return Tensor(self.data[item], self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.device)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), self.device)

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], np.dtype):
            return Tensor(self.data.view(args[0]), self.device)
        return self.reshape(*args)


class Stream:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_stream(self, other):
        self.events.append(("wait", self.name, other.name))


class FakeTorch:
    # float16 is a two-byte host storage surrogate for BF16; no BF16 math here.
    bfloat16 = float16 = np.dtype("float16")
    uint8 = np.dtype("uint8")
    float8_e4m3fn = np.dtype("int8")
    int32, int64, bool = np.dtype("int32"), np.dtype("int64"), np.dtype("bool")

    def __init__(self):
        self.events, self.allocations = [], []
        self.active = Stream("main", self.events)
        self.capturing = False
        self.capture_queries = 0
        self.ops = types.SimpleNamespace(
            npu=types.SimpleNamespace(npu_dynamic_mx_quant=self.quantize)
        )

    def empty(self, shape, *, dtype, device):
        result = Tensor(np.empty(shape, dtype=dtype), device)
        self.allocations.append(result)
        return result

    def zeros(self, shape, *, dtype, device):
        return Tensor(np.zeros(shape, dtype=dtype), device)

    def zeros_like(self, value):
        return Tensor(np.zeros_like(value.data), value.device)

    def get_device_module(self):
        return self

    def current_stream(self):
        return self.active

    def is_current_stream_capturing(self):
        self.capture_queries += 1
        return self.capturing

    @contextmanager
    def stream(self, value):
        previous, self.active = self.active, value
        try:
            yield
        finally:
            self.active = previous

    def quantize(self, hidden, *, dst_type):
        self.events.append(("quantize", self.active.name))
        # Include high-bit FP8 bytes and a packed three-dimensional scale layout.
        data = (np.arange(hidden.numel()) + 128).astype(np.uint8).view(dst_type)
        scale = np.arange(hidden.shape[0] * 6, dtype=np.uint8)
        return Tensor(data.reshape(hidden.shape)), Tensor(scale.reshape(-1, 3, 2))


class Group:
    def __init__(self, torch, name="attention-tp", world_size=4):
        self.torch, self.unique_name, self.world_size = torch, name, world_size

    def all_gather_into_tensor(self, output, send):
        self.torch.events.append(("ag", self.torch.active.name, send.numel()))
        # Distinct rank payloads expose ordering/layout errors.
        output.data[:] = np.concatenate(
            [np.roll(send.data, rank) for rank in range(self.world_size)]
        )


class MXFP8Method:
    pass


class TestMXFP8Workspace(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.group = Group(self.torch)
        ns = {
            "torch": self.torch,
            "copy": copy,
            "logger": logging.getLogger(__name__),
            "ShapeKey": ShapeKey,
            "eager_on_graph": lambda enabled: lambda fn: fn,
            "get_parallel": lambda: types.SimpleNamespace(
                enable_dp_attention=False, pp_size=1, attn_cp_size=1, moe_ep_size=1
            ),
            "envs": types.SimpleNamespace(
                SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES=types.SimpleNamespace(
                    get=lambda: "1,2,4"
                )
            ),
        }
        load_nodes(
            SRT / "model_executor/runner/welm_prefill_graph.py",
            ["parse_capture_batch_sizes", "WelmMXFP8GraphWorkspace", "WelmPrefillGraphAdapter"],
            ns,
        )
        self.Workspace = ns["WelmMXFP8GraphWorkspace"]
        self.Adapter = ns["WelmPrefillGraphAdapter"]
        self.workspace = self.Workspace()
        attn_ns = {"torch": self.torch, "_is_npu": True, "NPUMXFP8LinearMethod": MXFP8Method}
        methods = {
            name: load_method(SRT / "models/welmv4.py", "Qwen2MoeAttention", name, attn_ns)
            for name in (
                "_is_npu_mxfp8_projection",
                "can_reuse_prefill_mxfp8_input",
                "_npu_all_gather_mxfp8_bytes",
                "_npu_all_gather_bf16_gate_input",
                "_npu_prepare_prefill_mxfp8_qkv_input",
                "_npu_project_qkv_with_prefill_mxfp8_input",
            )
        }
        self.attn = type("Attention", (), methods)()
        self.attn.qkv_proj = types.SimpleNamespace(quant_method=MXFP8Method())
        self.attn.alt_stream = Stream("gate", self.torch.events)
        self.attn._npu_mxfp8_mm_from_quantized_input = self.matmul
        self.mm_inputs = []

    def matmul(self, projection, activation, scale, *, output_dtype):
        self.torch.events.append(("mm", self.torch.active.name))
        self.mm_inputs.append((activation.data.copy(), scale.data.copy(), output_dtype))
        return activation

    def hidden(self, rows):
        return Tensor(np.arange(rows * 12, dtype=np.float16).reshape(rows, 12))

    def project(self, rows, workspace):
        return self.attn._npu_project_qkv_with_prefill_mxfp8_input(
            self.hidden(rows), self.group, graph_workspace=workspace
        )

    def test_graph_matches_eager_and_preserves_gate_side_stream(self):
        eager_qkv, eager_gate, eager_alt = self.project(3, None)
        self.torch.events.clear()
        qkv, gate, alt = self.project(3, self.workspace)
        np.testing.assert_array_equal(qkv.data, eager_qkv.data)
        np.testing.assert_array_equal(gate.data, eager_gate.data)
        for actual, expected in zip(self.mm_inputs[-1], self.mm_inputs[-2]):
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(alt and eager_alt)
        self.assertEqual(self.torch.events, [
            ("quantize", "main"), ("ag", "main", 36), ("ag", "main", 18),
            ("wait", "gate", "main"), ("ag", "gate", 36), ("mm", "main"),
        ])

    def test_frozen_graph_storage_survives_larger_eager_and_bucket_changes(self):
        graph_qkv, graph_gate, _ = self.project(8, self.workspace)
        saved_qkv, saved_gate = graph_qkv.data.copy(), graph_gate.data.copy()
        capacities = [value.numel() for value in self.workspace._buffers.values()]
        self.assertEqual(capacities, [8 * 12 * 4, 8 * 6 * 4, 8 * 12 * 4])
        self.workspace.freeze()
        pointers = [value.data_ptr() for value in self.workspace._buffers.values()]
        self.project(8, None)
        eager_pointers = [v.data_ptr() for v in self.group._welmv4_mxfp8_ag_scratch.values()]
        eager_pointers += [v.data_ptr() for v in self.group._welmv4_bf16_gate_ag_scratch.values()]
        self.assertTrue(set(pointers).isdisjoint(eager_pointers))
        self.project(257, None)  # Force all three eager scratch buffers to grow.
        np.testing.assert_array_equal(graph_qkv.data, saved_qkv)
        np.testing.assert_array_equal(graph_gate.data, saved_gate)
        count = len(self.torch.allocations)
        for rows in (1, 3, 8, 2, 8):
            qkv, gate, _ = self.project(rows, self.workspace)
            self.assertEqual(qkv.data_ptr(), graph_qkv.data_ptr())
            self.assertEqual(gate.data_ptr(), graph_gate.data_ptr())
            eager_qkv, eager_gate, _ = self.project(rows, None)
            np.testing.assert_array_equal(qkv.data, eager_qkv.data)
            np.testing.assert_array_equal(gate.data, eager_gate.data)
        self.assertEqual(len(self.torch.allocations), count)
        self.assertEqual([v.data_ptr() for v in self.workspace._buffers.values()], pointers)

    def test_frozen_workspace_rejects_unwarmed_keys_and_overflow(self):
        send = self.hidden(2).reshape(-1)
        original = self.workspace.get_buffer(self.group, "gate-input", send)
        self.workspace.freeze()
        variants = [
            (self.group, "gate-input", self.hidden(3).reshape(-1)),
            (self.group, "unwarmed-role", send),
            (Group(self.torch, "other-tp"), "gate-input", send),
            (Group(self.torch, world_size=8), "gate-input", send),
            (self.group, "gate-input", Tensor(send.data.astype(np.float32))),
            (self.group, "gate-input", Tensor(send.data, Device(index=1))),
        ]
        count = len(self.torch.allocations)
        for group, kind, value in variants:
            with self.subTest(kind=kind, group=group.unique_name, dtype=value.dtype):
                with self.assertRaisesRegex(RuntimeError, "Captured storage cannot grow"):
                    self.workspace.get_buffer(group, kind, value)
        self.assertEqual(len(self.torch.allocations), count)
        self.assertEqual(original.data_ptr(), self.workspace.get_buffer(
            self.group, "gate-input", send
        ).data_ptr())

    def test_overflow_fails_before_collective_without_eager_fallback(self):
        self.project(2, self.workspace)
        self.workspace.freeze()
        self.torch.events.clear()
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            self.project(3, self.workspace)
        self.assertEqual(self.torch.events, [("quantize", "main")])
        self.assertFalse(hasattr(self.group, "_welmv4_mxfp8_ag_scratch"))
        self.assertFalse(hasattr(self.group, "_welmv4_bf16_gate_ag_scratch"))

    def test_single_rank_and_no_gather_do_not_allocate_receive_storage(self):
        single = Group(self.torch, world_size=1)
        hidden = self.hidden(3)
        self.workspace.freeze()
        self.assertIs(self.attn._npu_all_gather_mxfp8_bytes(
            hidden, single, scratch_name="activation", graph_workspace=self.workspace
        ), hidden)
        self.assertIs(self.attn._npu_all_gather_bf16_gate_input(
            hidden, single, graph_workspace=self.workspace
        ), hidden)
        _, gate, alt = self.attn._npu_project_qkv_with_prefill_mxfp8_input(
            hidden, None, graph_workspace=self.workspace
        )
        self.assertIsNone(gate)
        self.assertFalse(alt)
        self.assertEqual(self.torch.allocations, [])

    def test_gate_gather_without_alt_stream_still_uses_graph_storage(self):
        self.attn.alt_stream = None
        _, gate, alt = self.project(3, self.workspace)
        self.assertFalse(alt)
        self.assertEqual(gate.shape, (12, 12))
        self.assertEqual(len(self.workspace._buffers), 3)
        self.assertFalse(any(event[0] == "wait" for event in self.torch.events))
        self.assertTrue(all(event[1] == "main" for event in self.torch.events))

    def adapter(self):
        model = types.SimpleNamespace(
            scale_seq_times=0, oe_grams=[], layers=[types.SimpleNamespace(self_attn=self.attn)]
        )
        args = types.SimpleNamespace(
            dcp_size=1, enable_lora=False, enable_kv_mirror=False, enable_mixed_chunk=False
        )
        backend = types.SimpleNamespace(
            use_welm_flash_attn=True, create_welm_prefill_graph_metadata=lambda b: object()
        )
        runner = types.SimpleNamespace(
            layer_model=model, max_num_tokens=128, device=Device(), max_bs=4,
            model_runner=types.SimpleNamespace(
                attn_backend=backend, server_args=args,
                dtype=self.torch.bfloat16, kv_cache_dtype=self.torch.bfloat16,
            ),
        )
        return self.Adapter(runner)

    def before_capture(self, adapter):
        batch = types.SimpleNamespace(
            welm_prefill_graph_phase="prompt", positions=self.hidden(2).reshape(-1),
            num_token_non_padded=None, global_num_tokens_gpu=None,
        )
        adapter.flash_metadata[adapter.key(batch.positions.numel())] = types.SimpleNamespace(
            welm_flash_schedules={}
        )
        adapter._capture_templates[id(batch)] = (24, 24, None)
        module = types.ModuleType("sglang.srt.models.welmv4")
        module.KVMirrorManager = types.SimpleNamespace(activations_dict_kv={})
        with patch.dict(sys.modules, {module.__name__: module}):
            adapter.before_capture_forward(batch)

    def test_adapter_freezes_only_at_actual_capture_and_never_unfreezes(self):
        adapter = self.adapter()
        workspace = adapter.mxfp8_ag_workspace
        self.assertIsInstance(workspace, self.Workspace)
        for _ in range(2):
            self.before_capture(adapter)
            self.assertFalse(workspace.frozen)
            self.project(8, workspace)
        self.torch.capturing = True
        self.before_capture(adapter)
        self.assertTrue(workspace.frozen)
        self.torch.capturing = False  # Later bucket's warmup.
        self.before_capture(adapter)
        self.assertTrue(workspace.frozen)
        self.assertEqual(self.torch.capture_queries, 3)

    def test_bf16_creates_no_workspace_and_skips_capture_query(self):
        self.attn.qkv_proj = types.SimpleNamespace(quant_method=object())
        self.assertFalse(self.attn.can_reuse_prefill_mxfp8_input())
        adapter = self.adapter()
        self.assertIsNone(adapter.mxfp8_ag_workspace)
        self.torch.capturing = True
        self.before_capture(adapter)
        self.assertEqual(self.torch.capture_queries, 0)
        self.assertEqual(self.torch.allocations, [])

    def test_compressed_tensors_mxfp8_scheme_gets_workspace(self):
        self.attn.qkv_proj = types.SimpleNamespace(
            quant_method=object(), scheme=types.SimpleNamespace(kernel=MXFP8Method())
        )
        self.assertTrue(self.attn.can_reuse_prefill_mxfp8_input())
        self.assertIsInstance(self.adapter().mxfp8_ag_workspace, self.Workspace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
