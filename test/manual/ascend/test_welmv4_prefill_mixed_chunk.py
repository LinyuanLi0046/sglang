"""CPU execution of mixed chunk boundaries; no torch/Ascend required.

Device integration/numerics are covered by the separate NPU smoke tests.
"""

import ast
import logging
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from test_welmv4_prefill_graph_contracts import SRT, Adapter, load_method, load_nodes
from test_welmv4_prefill_graph_split_math import Tensor


NS = types.SimpleNamespace


class TestMixedBuckets(unittest.TestCase):
    def adapter(self, enabled=True):
        adapter = Adapter.__new__(Adapter)
        adapter.mixed_chunk = adapter.pad_mirror = enabled
        adapter.prune = True
        adapter.batch_sizes = (1, 2, 8)
        adapter.max_requests = 8
        adapter._warned = set()
        adapter.backend = NS(max_context_len=4096)
        adapter.model = NS(oe_grams=[], layers_to_capture=[])
        adapter.capture_keys = {adapter.key(4), adapter.key(128)} | {
            adapter.mirror_key(b) for b in adapter.batch_sizes
        }
        adapter.captured_states = {}
        return adapter

    @staticmethod
    def batch(b, mode=3):
        return NS(forward_mode=mode, batch_size=b, enable_kv_mirror=True,
                  extend_seq_lens_cpu=[1] * b, input_ids=[0] * b,
                  seq_lens_cpu=NS(max=lambda: NS(item=lambda: 200)),
                  ngram_embedding_info=None)

    def test_ceil_bucket_and_joint_admission(self):
        adapter = self.adapter()
        for b in range(1, 10):
            with self.subTest(b=b):
                expected = 1 if b == 1 else 2 if b == 2 else 8 if b <= 8 else None
                self.assertEqual(adapter.select_mirror_key(b),
                                 None if expected is None else adapter.mirror_key(expected))
                self.assertEqual(adapter.can_run(self.batch(b), 128), b <= 8)
        # Breal=3,R=3,Tcap=4,Bcap=8: no Bcap<=Tcap restriction.
        self.assertTrue(adapter.can_run(self.batch(3), 4))
        self.assertFalse(adapter.can_run(self.batch(3, mode=2), 4))
        calls = []
        adapter.runner = NS(backend=NS(replay=lambda key, batch, **kw: calls.append(key) or key))
        batch = self.batch(3)
        self.assertEqual(adapter.replay(adapter.key(4), batch), adapter.mirror_key(8))
        self.assertEqual(calls, [adapter.key(4), adapter.mirror_key(8)])
        self.assertEqual(batch.batch_size, 3)
        for missing in (adapter.key(4), adapter.mirror_key(8)):
            calls.clear()
            adapter.capture_keys.remove(missing)
            self.assertFalse(adapter.can_run(batch, 4))
            with self.assertRaises(RuntimeError):
                adapter.replay(adapter.key(4), batch)
            self.assertEqual(calls, [])
            adapter.capture_keys.add(missing)

    def test_mixed_off_retains_exact_prefill_and_rejects_mixed(self):
        adapter = self.adapter(enabled=False)
        self.assertTrue(adapter.can_run(self.batch(2, mode=1), 4))
        self.assertFalse(adapter.can_run(self.batch(3, mode=1), 4))
        self.assertFalse(adapter.can_run(self.batch(2, mode=3), 4))
        # Disabled path must not touch or require any new request-mask buffer.
        adapter._prepare_mirror_requests(2)
        adapter.prune = False
        adapter.capture_keys = {adapter.key(4)}
        batch = self.batch(3, mode=1)
        batch.enable_kv_mirror = False
        self.assertTrue(adapter.can_run(batch, 4))

    def test_request_mask_clears_full_capacity_on_large_small_large(self):
        adapter = self.adapter()
        adapter.mirror_q_used = Tensor(np.empty(8, dtype=np.int32))
        address = adapter.mirror_q_used.data.ctypes.data
        for b in (7, 3, 8, 1, 5, 8):
            adapter._prepare_mirror_requests(b)
            np.testing.assert_array_equal(adapter.mirror_q_used.data, [1]*b + [0]*(8-b))
            self.assertEqual(adapter.mirror_q_used.data.ctypes.data, address)

    def test_body_trims_all_leaves_before_eager_tail_only_for_padded_mirror(self):
        path = SRT / "model_executor/runner/prefill_cuda_graph_runner.py"
        ns = load_nodes(path, ["_slice_output_rows"], {
            "torch": NS(is_tensor=lambda value: isinstance(value, Tensor)),
            "PPProxyTensors": type("Proxy", (), {}),
        })
        execute = load_method(path, "PrefillCudaGraphRunner", "_execute_body_capture", ns)
        for padded, physical in ((True, 8), (False, 3)):
            with self.subTest(padded=padded):
                original = object()
                layer_model = NS(forward=original)
                body = (Tensor(np.arange(physical*2).reshape(physical, 2)),
                        [Tensor(np.ones((physical, 4)))])
                seen = []

                def model_forward(ids, positions, batch):
                    hidden = layer_model.forward()
                    seen.append((hidden[0].shape[0], hidden[1][0].shape[0], batch.batch_size))
                    return hidden

                runner = NS(
                    _is_full_backend=False, _input_embeds_arg_idx=None,
                    buffer_registry=NS(has_slot=lambda name: False),
                    welm_adapter=NS(pad_mirror=padded, replay=lambda *a, **kw: body),
                    layer_model=layer_model, model_runner=NS(model=NS(forward=model_forward)),
                    _prefill_forward_context=lambda *a, **kw: nullcontext(),
                )
                batch = NS(batch_size=3, input_ids=object(), positions=object())
                execute(runner, batch, batch, 128, 113, object())
                self.assertEqual(seen, [(3, 3, 3)])
                self.assertIs(layer_model.forward, original)
                self.assertEqual(body[0].shape[0], physical)


class TestMixedKVPages(unittest.TestCase):
    def test_scheduler_only_counts_live_welm_requests_at_page_boundaries(self):
        required = load_method(SRT / "managers/schedule_batch.py", "ScheduleBatch",
                               "new_tokens_required_next_decode", {})
        reserve = load_method(SRT / "managers/scheduler.py", "Scheduler",
                              "_welm_mixed_decode_kv_tokens", {})
        scheduler = NS(is_mixed_chunk=True,
                       model_config=NS(hf_config=NS(architectures=["WeLMV4MoeForCausalLM"])))
        batch = NS(reqs=[NS(kv_committed_len=n, finished=lambda done=done: done)
                        for n, done in ((63, False), (64, False), (65, False), (128, True))],
                   token_to_kv_pool_allocator=NS(page_size=64),
                   spec_algorithm=NS(is_none=lambda: True))
        batch.new_tokens_required_next_decode = types.MethodType(required, batch)
        self.assertEqual(reserve(scheduler, batch), 64)
        # Empty/all-finished placeholders must not access an allocator.
        self.assertEqual(reserve(scheduler, NS(reqs=[])), 0)
        self.assertEqual(reserve(scheduler, NS(reqs=[NS(finished=lambda: True)])), 0)
        scheduler.is_mixed_chunk = False
        self.assertIsNone(reserve(scheduler, object()))
        scheduler.is_mixed_chunk = True
        scheduler.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
        self.assertIsNone(reserve(scheduler, object()))

    def test_prefill_adder_separates_compute_and_full_swa_page_budgets(self):
        pool = type("SWAPool", (), {})
        other = type("OtherPool", (), {})
        init = load_method(SRT / "managers/schedule_policy.py", "PrefillAdder", "__init__", {
            "SWATokenToKVPoolAllocator": pool, "DeepSeekV4HiSparseTokenToKVPoolAllocator": other,
            "PureSWATokenToKVPoolAllocator": other, "UnifiedMambaTokenToKVPoolAllocator": other,
            "is_dsa_prefill_cp_in_seq_split": lambda: False,
            "is_prefill_context_parallel_enabled": lambda: False,
        })
        for page_cost, full, swa in ((None, 3, 0), (0, 0, 0), (64, 64, 64), (128, 128, 128)):
            adder = NS()
            init(adder, 64, NS(supports_mamba=lambda: False), pool(), None,
                 0.5, 1024, 512, num_mixed_decode_tokens=3,
                 mixed_decode_kv_tokens=page_cost)
            self.assertEqual((adder.rem_input_tokens, adder.rem_chunk_tokens), (1021, 509))
            self.assertEqual((adder.cur_rem_token_offset, adder.rem_total_token_offset,
                              adder.rem_swa_token_offset), (full, full, swa))


class TestMixedDispatchAndScope(unittest.TestCase):
    def test_breakable_live_modes_survive_only_for_npu(self):
        tree = ast.parse((SRT / "model_executor/runner/prefill_cuda_graph_runner.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_batch")
        nodes = [n for n in fn.body if (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in ("pcg_forward_mode", "pcg_global_forward_mode") for t in n.targets)
        ) or (isinstance(n, ast.If) and "is_npu()" in ast.unparse(n.test)
              and "self.prefill_backend_name == Backend.BREAKABLE" in ast.unparse(n.test))]
        self.assertEqual(len(nodes), 3)
        code = compile(ast.Module(body=nodes, type_ignores=[]), "live_mode", "exec")
        for npu in (False, True):
            for backend in ("breakable", "piecewise"):
                for mode in (1, 3):
                    ns = dict(is_npu=lambda: npu, self=NS(prefill_backend_name=backend),
                              Backend=NS(BREAKABLE="breakable"), ForwardMode=NS(EXTEND=1, MIXED=3),
                              forward_batch=NS(forward_mode=mode, global_forward_mode=mode))
                    exec(code, ns)
                    expected = mode if npu and backend == "breakable" else 1
                    self.assertEqual((ns["pcg_forward_mode"], ns["pcg_global_forward_mode"]),
                                     (expected, expected))

    def test_mixed_fused_rope_never_assumes_contiguous_batch_positions(self):
        tree = ast.parse((SRT / "models/welmv4.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_try_npu_fused_qkv")
        node = next(n for n in fn.body if isinstance(n, ast.If) and "mode in" in ast.unparse(n.test))
        wrapper = ast.parse("def policy():\n    pass\n").body[0]
        wrapper.body = [node, ast.Return(value=ast.Name(id="positions_contiguous", ctx=ast.Load()))]
        code = compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "rope_policy", "exec")
        cases = ((1, 1, None, True), (1, 2, None, False), (1, 1, object(), False),
                 (3, 1, None, False), (3, 2, None, False), (2, 1, None, False), (4, 1, None, False))
        for mode, b, graph, expected in cases:
            ns = dict(mode=mode, num_tokens=8, cos_sin_cache=NS(shape=(100, 64)),
                      ForwardMode=NS(EXTEND=1, MIXED=3, DECODE=2, TARGET_VERIFY=4),
                      forward_batch=NS(batch_size=b, welm_prefill_graph=graph, extend_prefix_lens_cpu=[63]))
            exec(code, ns)
            self.assertEqual(ns["policy"](), expected)

    def test_native_mixed_forwards_sinks_and_cache_ownership_before_fia_guards(self):
        mixed = load_method(SRT / "hardware_backend/npu/attention/ascend_backend.py",
                            "AscendAttnBackend", "forward_mixed", {})
        calls = []
        backend = NS(use_welm_flash_attn=True,
                     forward_extend=lambda *a, **kw: calls.append((a, kw)) or "native")
        q, layer, batch, sinks = object(), object(), object(), object()
        self.assertEqual(mixed(backend, q, None, None, layer, batch,
                               save_kv_cache=False, sinks=sinks), "native")
        self.assertEqual(calls[0][0], (q, None, None, layer, batch))
        self.assertIs(calls[0][1]["sinks"], sinks)
        self.assertFalse(calls[0][1]["save_kv_cache"])
        backend.use_welm_flash_attn = False
        backend.use_mla = True
        with self.assertRaises(NotImplementedError):
            mixed(backend, q, None, None, layer, batch)
        self.assertEqual(len(calls), 1)

    def test_mixed_capability_keeps_unsupported_profiles_disabled(self):
        tree = ast.parse((SRT / "server_args.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.BoolOp)
                    and ast.unparse(n.test.values[0]) == "self.enable_mixed_chunk"
                    and "WELM_NPU_USE_FLASH_ATTN" in ast.unparse(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), "mixed_profile", "exec")
        config = NS(dtype="bf16")
        args = NS(enable_mixed_chunk=True, enable_dp_attention=False, pp_size=1,
                  dcp_size=1, kv_cache_dtype="auto", quantization=None, enable_lora=False,
                  _resolved=lambda: NS(attn_cp_size=1), get_model_config=lambda: config)
        env = {"WELM_NPU_USE_FLASH_ATTN": "1"}
        ns = dict(self=args, is_npu=lambda: True, os=NS(environ=env),
                  torch=NS(bfloat16="bf16"), raw_spec_algorithm="", logger=logging.getLogger(__name__))
        with patch.object(ns["logger"], "warning"):
            exec(code, ns)
            self.assertTrue(args.enable_mixed_chunk)
            for name, value in (("enable_dp_attention", True), ("pp_size", 2),
                                ("dcp_size", 2), ("kv_cache_dtype", "fp8_e4m3"),
                                ("quantization", "fp8"), ("enable_lora", True)):
                old = getattr(args, name)
                setattr(args, name, value)
                args.enable_mixed_chunk = True
                exec(code, ns)
                self.assertFalse(args.enable_mixed_chunk, name)
                setattr(args, name, old)
            for key, value in (("raw_spec_algorithm", "EAGLE"), ("is_npu", lambda: False)):
                old, ns[key] = ns[key], value
                args.enable_mixed_chunk = True
                exec(code, ns)
                self.assertFalse(args.enable_mixed_chunk)
                ns[key] = old
            for dtype, flash in (("fp16", "1"), ("bf16", "0")):
                config.dtype, env["WELM_NPU_USE_FLASH_ATTN"] = dtype, flash
                args.enable_mixed_chunk = True
                exec(code, ns)
                self.assertFalse(args.enable_mixed_chunk)

    def test_padding_route_policy_is_graph_mirror_only_for_both_tp_and_ep(self):
        tree = ast.parse((SRT / "models/welmv4.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and "welm_prefill_graph.pad_mirror" in ast.unparse(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), "mirror_routes", "exec")
        for ep in (False, True):
            for phase, pad, expected in (("mirror", True, None), ("prompt", True, "count"),
                                         ("mirror", False, "count")):
                ns = dict(_is_npu=True, num_token_non_padded="count", valid_row_mask="mask",
                          forward_batch=NS(welm_prefill_graph=NS(pad_mirror=pad),
                                           welm_prefill_graph_phase=phase,
                                           welmv4_npu_deepep_full_mirror=ep))
                exec(code, ns)
                self.assertEqual(ns["num_token_non_padded"], expected)
                self.assertEqual(ns["valid_row_mask"], None if expected is None else "mask")
        for batch in (None, NS(welm_prefill_graph=None)):
            ns = dict(_is_npu=True, forward_batch=batch, num_token_non_padded="count")
            exec(code, ns)
            self.assertEqual(ns["num_token_non_padded"], "count")


if __name__ == "__main__":
    unittest.main(verbosity=2)
