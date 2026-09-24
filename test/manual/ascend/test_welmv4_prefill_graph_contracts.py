"""Host-side WeLM graph contract tests, runnable without torch/Ascend.

These execute the production helpers/methods from their AST, injecting only
the unavailable device/runtime dependencies. They do NOT test NPU capture,
kernel numerics, or HCCL. Run: python <this file>.
"""

import ast
import copy
import importlib.util
import itertools
import logging
import random
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


key_spec = importlib.util.spec_from_file_location(
    "_welm_contract_shape_key", SRT / "model_executor/runner/shape_key.py"
)
key_module = importlib.util.module_from_spec(key_spec)
sys.modules[key_spec.name] = key_module
key_spec.loader.exec_module(key_module)
ShapeKey = key_module.ShapeKey

ADAPTER = load_nodes(
    SRT / "model_executor/runner/welm_prefill_graph.py",
    ["parse_capture_batch_sizes", "capture_request_lengths", "padded_rope_tiles", "WelmPrefillGraphAdapter"],
    {
        "copy": copy,
        "logger": logging.getLogger(__name__),
        "ShapeKey": ShapeKey,
        "ForwardMode": types.SimpleNamespace(EXTEND=1),
        "eager_on_graph": lambda enabled: lambda fn: fn,
    },
)
Adapter = ADAPTER["WelmPrefillGraphAdapter"]


class Rows:
    def __init__(self, rows, width=2):
        self.shape = (rows, width)

    def __getitem__(self, item):
        return Rows(len(range(self.shape[0])[item]), self.shape[1])

    def new_zeros(self, shape):
        return Rows(*shape)

    def copy_(self, value):
        assert self.shape == value.shape
        return self


class Scalar:
    def __init__(self, value):
        self.value = value

    def fill_(self, value):
        self.value = value

    def copy_(self, other):
        self.value = other.value


class TestInputContracts(unittest.TestCase):
    def test_batch_sizes_are_exact_and_deduplicated(self):
        parse = ADAPTER["parse_capture_batch_sizes"]
        self.assertEqual(parse("8,2,1,2,4", 4), (1, 2, 4))
        for value in ("", "0,1", "-1,4", "four", "32"):
            with self.assertRaises(ValueError):
                parse(value, 8)

    def test_dummy_requests_are_all_nonempty(self):
        for tokens in range(1, 33):
            for batch in range(1, tokens + 1):
                lengths = ADAPTER["capture_request_lengths"](tokens, batch)
                self.assertEqual(sum(lengths), tokens)
                self.assertEqual(len(lengths), batch)
                self.assertGreater(min(lengths), 0)
        with self.assertRaises(ValueError):
            ADAPTER["capture_request_lengths"](2, 3)

    def test_padded_tiles_cover_real_rows_once_and_never_cross_requests(self):
        rng = random.Random(731)
        for _ in range(250):
            lengths = [rng.randint(1, 513) for _ in range(rng.randint(2, 8))]
            real = sum(lengths)
            capacity = real + rng.randint(0, 255)
            starts = ADAPTER["padded_rope_tiles"](lengths, capacity)
            expected_size = (capacity + 63) // 64 + len(lengths)
            self.assertEqual(len(starts), expected_size)
            boundaries = [sum(lengths[:i]) for i in range(1, len(lengths))]
            visited = []
            for begin, end in zip(starts, starts[1:]):
                self.assertTrue(0 <= begin < capacity)
                self.assertTrue(0 <= end - begin <= 64)
                self.assertFalse(any(begin < boundary < end for boundary in boundaries))
                visited.extend(range(begin, end))
                if begin == end:
                    self.assertEqual(begin, 0)
            self.assertEqual(visited, list(range(real)))

    def test_exact_capacity_empty_tiles_load_position_zero(self):
        # Repeating the terminal sentinel here would read position[128].
        self.assertEqual(ADAPTER["padded_rope_tiles"]([64, 64], 128), [0, 0, 64, 128])

    def test_same_bucket_has_stable_tile_capacity(self):
        sizes = {
            len(ADAPTER["padded_rope_tiles"](lengths, 256))
            for lengths in ([1, 1], [63, 65], [64, 192], [127, 129])
        }
        self.assertEqual(sizes, {6})

    def test_invalid_request_tiles_are_rejected(self):
        for lengths, capacity in (([], 8), ([0, 8], 8), ([-1, 9], 8), ([5, 5], 8)):
            with self.assertRaises(ValueError):
                ADAPTER["padded_rope_tiles"](lengths, capacity)

    def test_key_uses_variant_not_stream_index(self):
        adapter = Adapter.__new__(Adapter)
        adapter.prune = True
        key = adapter.key(128, 4)
        self.assertIsNone(key.stream_idx)
        self.assertEqual(key.variant_label, "welm:prompt:mirror=1")
        self.assertEqual(key, adapter.key(128, 2))
        self.assertNotEqual(key, adapter.mirror_key(128))
        self.assertNotEqual(adapter.mirror_key(4), adapter.mirror_key(2))

    def test_can_run_accepts_already_tp_padded_input(self):
        adapter = Adapter.__new__(Adapter)
        adapter.prune = True
        adapter._warned = set()
        adapter.max_requests = 8
        adapter.capture_keys = {adapter.key(128), adapter.mirror_key(2)}
        adapter.model = types.SimpleNamespace(oe_grams=[2, 2, 3, 3], layers_to_capture=[])
        batch = types.SimpleNamespace(
            forward_mode=1, enable_kv_mirror=True, batch_size=2,
            extend_seq_lens_cpu=[63, 62], input_ids=list(range(128)),
            extend_num_tokens=128, ngram_embedding_info=object(),
        )
        self.assertTrue(adapter.can_run(batch, 128))
        batch.batch_size = 3
        self.assertFalse(adapter.can_run(batch, 128))
        batch.batch_size = 2
        batch.ngram_embedding_info = None
        self.assertFalse(adapter.can_run(batch, 128))


class TestFlashAndStateContracts(unittest.TestCase):
    def setUp(self):
        self.adapter = Adapter.__new__(Adapter)
        self.swa = Rows(128)
        self.adapter.backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(swa_out_cache_loc=self.swa)
        )
        self.batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[63, 62], batch_size=2,
            num_token_non_padded_cpu=125, out_cache_loc=Rows(128),
            custom_last_index=object(), enable_kv_mirror=True,
        )
        self.adapter.current_batch = self.batch
        self.seen = []
        self.radix_module = types.ModuleType("sglang.srt.layers.radix_attention")
        self.radix_module.force_eager_attention = nullcontext
        self.patch = patch.dict(sys.modules, {self.radix_module.__name__: self.radix_module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def layer(self, q, k, v, batch, **kwargs):
        self.seen.append((q.shape[0], k.shape[0], v.shape[0], batch.out_cache_loc.shape[0],
                          hasattr(batch, "custom_last_index"), kwargs["save_kv_cache"],
                          self.adapter.backend.forward_metadata.swa_out_cache_loc.shape[0]))
        return Rows(q.shape[0])

    def test_mirror_keeps_all_new_kv_rows(self):
        output = self.adapter.flash(self.layer, Rows(2), Rows(128), Rows(128),
                                    mirror=True, save_kv_cache=True)
        self.assertEqual(self.seen[-1], (2, 125, 125, 125, True, True, 125))
        self.assertEqual(output.shape[0], 2)
        self.assertIs(self.adapter.backend.forward_metadata.swa_out_cache_loc, self.swa)

    def test_prefix_hides_stale_mirror_flag_and_restores_padding(self):
        output = self.adapter.flash(self.layer, Rows(128), Rows(128), Rows(128),
                                    mirror=False, save_kv_cache=False)
        self.assertEqual(self.seen[-1], (125, 125, 125, 125, False, False, 125))
        self.assertEqual(output.shape[0], 128)
        self.assertTrue(hasattr(self.batch, "custom_last_index"))

    def test_flash_reads_the_new_batch_on_every_call(self):
        for lengths in ([63, 62], [1, 7], [64, 64]):
            self.adapter.current_batch = copy.copy(self.batch)
            self.adapter.current_batch.extend_seq_lens_cpu = lengths
            self.adapter.flash(self.layer, Rows(2), Rows(128), Rows(128),
                               mirror=True, save_kv_cache=True)
            self.assertEqual(self.seen[-1][1], sum(lengths))

    def test_flash_exception_restores_swa_metadata(self):
        def broken(*args, **kwargs):
            raise RuntimeError("flash failed")

        with self.assertRaisesRegex(RuntimeError, "flash failed"):
            self.adapter.flash(broken, Rows(2), Rows(128), Rows(128),
                               mirror=True, save_kv_cache=True)
        self.assertIs(self.adapter.backend.forward_metadata.swa_out_cache_loc, self.swa)

    def test_each_warmup_restores_prefix_state_and_live_scalar(self):
        module = types.ModuleType("sglang.srt.models.welmv4")
        module.KVMirrorManager = types.SimpleNamespace(activations_dict_kv={1: object()})
        batch = self.batch
        batch.num_token_non_padded = Scalar(2)
        batch.global_num_tokens_gpu = Scalar(2)
        self.adapter.local_valid_rows = Scalar(31)
        self.adapter._capture_templates = {id(batch): (125, 128, [128])}
        with patch.dict(sys.modules, {module.__name__: module}):
            for _ in range(3):
                batch.custom_last_index = object()
                batch.welmv4_npu_deepep_full_mirror = True
                batch.welmv4_npu_deepep_scattered = True
                batch.model_specific_states = {"stale": True}
                batch.num_token_non_padded.fill_(2)
                self.adapter.before_capture_forward(batch)
                self.assertFalse(hasattr(batch, "custom_last_index"))
                self.assertFalse(batch.welmv4_npu_deepep_full_mirror)
                self.assertFalse(batch.welmv4_npu_deepep_scattered)
                self.assertIsNone(batch.model_specific_states)
                self.assertEqual(batch.num_token_non_padded.value, 31)
                self.assertEqual(batch.num_token_non_padded_cpu, 125)
                self.assertEqual(batch.global_num_tokens_gpu.value, 128)
                self.assertEqual(module.KVMirrorManager.activations_dict_kv, {})


class TestSplitGraphContracts(unittest.TestCase):
    def test_model_phase_entry_runs_embedding_only_in_prompt_and_stops_at_consumer(self):
        events = []
        forward = load_method(
            SRT / "models/welmv4.py", "Qwen2MoeModel", "forward",
            {"get_global_expert_distribution_recorder": lambda: types.SimpleNamespace(
                with_current_layer=lambda i: nullcontext()
            )},
        )

        class Layer:
            def __init__(self, index):
                self.index = index

            def __call__(self, positions, hidden, batch, residual):
                events.append(("layer", self.index, hidden.shape[0]))
                return hidden, None

        model = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
            layers=[Layer(i) for i in range(4)], start_layer=0, end_layer=4,
            layers_to_capture=[], oe_grams=[], scale_seq_times=0,
            embed_tokens=lambda ids: events.append(("embed",)) or Rows(128),
            norm=lambda h: (h, None),
            _restore_npu_prefill_deepep_output_layout=lambda h, aux, b: (h, aux),
        )
        graph = types.SimpleNamespace(
            prune=True, first_mirror=2,
            finish_prompt=lambda *args: events.append(("handoff",)) or None,
            begin_mirror=lambda b: (Rows(b.batch_size), None, object()),
        )
        batch = types.SimpleNamespace(
            welm_prefill_graph=graph, welm_prefill_graph_phase="prompt",
            batch_size=2, can_run_tbo=False,
            capture_hidden_mode=types.SimpleNamespace(need_capture=lambda: False),
        )
        self.assertIsNone(forward(model, object(), object(), batch))
        self.assertEqual(events, [("embed",), ("layer", 0, 128), ("layer", 1, 128), ("handoff",)])
        events.clear()
        batch.welm_prefill_graph_phase = "mirror"
        output = forward(model, object(), object(), batch)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(events, [("layer", 2, 2), ("layer", 3, 2)])

    def test_capture_loop_is_sum_not_product_and_largest_mirror_first(self):
        capture = load_method(
            SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
            "PrefillCudaGraphRunner", "_capture_one_stream",
            {
                "get_available_gpu_memory": lambda *a, **kw: 100,
                "get_parallel": lambda: types.SimpleNamespace(tp_rank=1),
            },
        )
        calls = []
        adapter = types.SimpleNamespace(prune=True, batch_sizes=(1, 2, 4, 8))
        runner = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(device="npu", gpu_id=0),
            capture_num_tokens=[8, 16, 32], max_num_tokens=32,
            welm_adapter=adapter, _capture_chunked_prefix=False,
        )
        runner.capture_one_shape = lambda t, **kw: calls.append(
            (t, runner._capture_req_slots, kw.get("welm_mirror_bs", 0))
        )
        capture(runner)
        self.assertEqual(calls, [
            (32, 1, 0), (16, 1, 0), (8, 1, 0),
            (32, 8, 8), (32, 4, 4), (32, 2, 2), (32, 1, 1),
        ])
        self.assertEqual(adapter.capture_phase, "prompt")
        self.assertEqual(runner._capture_req_slots, 1)
        calls.clear()
        adapter.prune = False
        capture(runner)
        self.assertEqual(len(calls), 3)

    def test_uncovered_either_family_submits_nothing(self):
        adapter = Adapter.__new__(Adapter)
        adapter.prune = True
        calls = []
        adapter.runner = types.SimpleNamespace(backend=types.SimpleNamespace(
            replay=lambda key, batch, **kw: calls.append(key) or key
        ))
        adapter.captured_states = {}
        batch = types.SimpleNamespace(batch_size=2)
        prompt, mirror = adapter.key(128), adapter.mirror_key(2)
        for keys in ({prompt}, {mirror}, set()):
            adapter.capture_keys = keys
            with self.assertRaises(RuntimeError):
                adapter.replay(prompt, batch)
            self.assertEqual(calls, [])
        adapter.capture_keys = {prompt, mirror, adapter.key(256)}
        for tokens in (128, 256, 128):
            self.assertEqual(adapter.replay(adapter.key(tokens), batch), mirror)
        self.assertEqual(calls, [prompt, mirror, adapter.key(256), mirror, prompt, mirror])

    def test_t_only_tiles_allow_b_changes_and_bmax_larger_than_t(self):
        tiles = ADAPTER["padded_rope_tiles"]
        for capacity in (4, 128, 256):
            sizes = set()
            for bs in (1, 2, 4):
                lengths = ADAPTER["capture_request_lengths"](capacity, bs)
                values = tiles(lengths, capacity, 8)
                sizes.add(len(values))
                actual = [i for start, end in zip(values, values[1:])
                          for i in range(start, end)]
                self.assertEqual(actual, list(range(capacity)))
            self.assertEqual(sizes, {(capacity + 63) // 64 + 8})
        with self.assertRaises(ValueError):
            tiles([1, 1, 1], 8, 2)

    def test_mirror_callback_fetches_live_t_kv_without_captured_t_argument(self):
        adapter = Adapter.__new__(Adapter)
        adapter.mirror_kv = {7: (Rows(256), Rows(256))}
        seen = []

        class Attention:
            layer_id = 7

            def __call__(self, q, k, v, batch, **kw):
                seen.append((q.shape[0], k.shape[0], v.shape[0],
                             batch.out_cache_loc.shape[0]))
                return Rows(q.shape[0])

        module = types.ModuleType("sglang.srt.layers.radix_attention")
        module.force_eager_attention = nullcontext
        with patch.dict(sys.modules, {module.__name__: module}):
            for lengths in ([3, 5], [80, 73], [1, 1]):
                adapter.current_batch = types.SimpleNamespace(
                    batch_size=2, extend_seq_lens_cpu=lengths, out_cache_loc=Rows(256)
                )
                adapter.backend = types.SimpleNamespace(
                    forward_metadata=types.SimpleNamespace(swa_out_cache_loc=Rows(256))
                )
                adapter.flash(Attention(), Rows(2), None, None,
                              mirror=True, save_kv_cache=True)
        self.assertEqual(seen, [(2, 8, 8, 8), (2, 153, 153, 153), (2, 2, 2, 2)])


class TestOverlapPreparation(unittest.TestCase):
    def test_tiles_use_fresh_pinned_sources_and_nonblocking_copy(self):
        pending = []

        class DeviceSlot:
            def copy_(self, source, *, non_blocking=False):
                self_test.assertTrue(non_blocking)
                pending.append(source)

        def tensor(values, *, dtype, device, pin_memory):
            self.assertEqual(device, "cpu")
            self.assertTrue(pin_memory)
            return list(values)

        self_test = self
        adapter = Adapter.__new__(Adapter)
        adapter.device = "npu"
        adapter.rope_tiles = {}
        adapter.max_requests = 8
        adapter.tail_indices = DeviceSlot()
        batch = types.SimpleNamespace(batch_size=2, extend_seq_lens_cpu=[63, 62])
        fake_torch = types.SimpleNamespace(
            int32="int32", int64="int64", tensor=tensor,
            empty=lambda *args, **kwargs: DeviceSlot()
        )
        with patch.dict(ADAPTER, torch=fake_torch):
            adapter._prepare_tiles(batch, 128)
            slot = batch.welmv4_rope_segment_tile_starts
            batch.extend_seq_lens_cpu = [1, 7]
            adapter._prepare_tiles(batch, 128)
        self.assertIs(batch.welmv4_rope_segment_tile_starts, slot)
        self.assertIsNot(pending[0], pending[2])
        self.assertEqual(pending[0], ADAPTER["padded_rope_tiles"]([63, 62], 128, 8))
        self.assertEqual(pending[2], ADAPTER["padded_rope_tiles"]([1, 7], 128, 8))
        self.assertEqual(pending[1], [62, 124, 0, 0, 0, 0, 0, 0])
        self.assertEqual(pending[3], [0, 7, 0, 0, 0, 0, 0, 0])

    def test_welm_prefill_metadata_never_reads_device_lengths_to_host(self):
        class CpuLengths(list):
            def max(self):
                return types.SimpleNamespace(item=lambda: max(self))

            def int(self):
                return self

        class DeviceLengths:
            def max(self):
                raise AssertionError("device max must not become a Python slice bound")

            def cpu(self):
                raise AssertionError("unexpected length D2H")

            def int(self):
                return self

        stops = []

        class BlockTable:
            def __getitem__(self, index):
                bound = index[1]
                if bound.stop is not None:
                    self_test.assertIsInstance(bound.stop, int)
                    stops.append(bound.stop)
                return self

            def __floordiv__(self, value):
                return self

        self_test = self
        mode = types.SimpleNamespace(
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        init = load_method(
            SRT / "hardware_backend/npu/attention/ascend_backend.py",
            "AscendAttnBackend", "init_forward_metadata",
            {
                "ForwardMetadata": types.SimpleNamespace,
                "ForwardMode": types.SimpleNamespace(EXTEND=mode),
                "torch": types.SimpleNamespace(
                    int32="int32", bfloat16="bf16",
                    tensor=lambda values, **kwargs: CpuLengths(values),
                ),
                "get_parallel": lambda: types.SimpleNamespace(enable_dp_attention=False),
                "np": types.SimpleNamespace(cumsum=lambda values: list(itertools.accumulate(values))),
            },
        )
        pinned = []
        backend = types.SimpleNamespace(
            use_welm_flash_attn=True, model_dtype="bf16", attn_cp_size=1,
            is_welm_v4=True, use_mla=False, is_hybrid_swa=False, page_size=4,
            req_to_token_pool=types.SimpleNamespace(req_to_token=BlockTable()),
            use_sliding_window_kv_pool=False,
            _prepare_welm_flash_metadata_inputs=(
                lambda batch, *, pin_memory: pinned.append(pin_memory)
            ),
        )
        batch = types.SimpleNamespace(
            forward_mode=mode, seq_lens=DeviceLengths(),
            seq_lens_cpu=CpuLengths([5, 17]), extend_seq_lens=DeviceLengths(),
            extend_seq_lens_cpu=[2, 7], req_pool_indices=object(),
        )
        init(backend, batch)
        self.assertEqual(stops, [17])
        self.assertEqual(backend.forward_metadata.extend_seq_lens_cpu_int, [2, 7])
        self.assertEqual(pinned, [True])
        self.assertFalse(backend.graph_mode)

    def test_output_trim_preserves_model_state_reference(self):
        trim = load_method(
            SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
            "PrefillCudaGraphRunner", "_trim_logits_output",
            {"LogitsProcessorOutput": types.SimpleNamespace},
        )
        runner = types.SimpleNamespace(
            raw_bs=2, raw_num_tokens=125, _is_full_backend=False,
            model_runner=types.SimpleNamespace(
                spec_algorithm=types.SimpleNamespace(is_speculative=lambda: False)
            ),
        )
        state = {"source0": object()}
        output = types.SimpleNamespace(
            next_token_logits=Rows(2), hidden_states=None, input_token_logprobs=None,
            input_top_logprobs_val=None, input_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None, input_token_ids_logprobs_idx=None,
            model_specific_states=state,
        )
        self.assertIs(trim(runner, output).model_specific_states, state)


class TestNgramHistoryStaging(unittest.TestCase):
    def setUp(self):
        self.pending = []
        self.host_sources = []
        self.kernel_calls = []
        test = self

        class HostTensor:
            def __init__(self, values, dtype):
                self.values = list(values)
                self.dtype = dtype

            def to(self, *, device, non_blocking):
                test.assertEqual(device, "npu")
                test.assertTrue(non_blocking)
                out = types.SimpleNamespace(values=None, dtype=self.dtype)
                # Delay the source read until after later batches are prepared.
                test.pending.append(lambda: setattr(out, "values", list(self.values)))
                return out

        def tensor(values, *, dtype, device, pin_memory):
            self.assertEqual(device, "cpu", "no default NPU H2D in history staging")
            self.assertTrue(pin_memory)
            source = HostTensor(values, dtype)
            self.host_sources.append(source)
            return source

        self.ns = load_nodes(
            SRT / "model_executor/model_runner_components/ngram_embedding_manager.py",
            ["NgramEmbeddingManager", "_history_tensor",
             "update_ngram_token_table_after_sampling"],
            {
                "__name__": __name__,
                "dataclass": dataclass,
                "is_npu": lambda: True,
                "ForwardMode": types.SimpleNamespace(EXTEND=1),
                "torch": types.SimpleNamespace(
                    tensor=tensor, int32="int32", int64="int64", bool="bool",
                    zeros=lambda n, **kw: types.SimpleNamespace(values=[0] * n),
                ),
            },
        )
        self.table = types.SimpleNamespace(
            values=[[-1] * 16 for _ in range(8)],
            dtype="int32", device="npu", shape=(8, 16),
        )
        self.manager = self.ns["NgramEmbeddingManager"](
            enabled=True, table=self.table, n=3, k=0
        )
        module = types.ModuleType("sglang.srt.layers.welmv4_npu_op")

        def ragged(table, tokens, rows, offsets, starts, lengths, **kwargs):
            self.kernel_calls.append((tokens, rows, offsets, starts, lengths))

            def execute():
                self.assertEqual(kwargs["max_req_len"], max(lengths.values))
                for row, offset, start, length in zip(
                    rows.values, offsets.values, starts.values, lengths.values
                ):
                    table.values[row][start:start + length] = (
                        tokens.values[offset:offset + length]
                    )

            self.pending.append(execute)

        module.welmv4_token_table_ragged_update_npu = ragged
        module.welmv4_token_table_decode_update_npu = (
            lambda *args, **kwargs: self.kernel_calls.append((args, kwargs))
        )
        module_patch = patch.dict(sys.modules, {module.__name__: module})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    @staticmethod
    def request(start, length, tokens, **kwargs):
        return types.SimpleNamespace(
            prefix_indices=list(range(start)),
            extend_range=types.SimpleNamespace(length=length),
            origin_input_ids=list(tokens), output_ids=[], **kwargs,
        )

    def test_prefill_prefix_history_and_chunk_mask_survive_queued_batches(self):
        reqs = [
            self.request(0, 3, range(10, 18)),
            self.request(1, 2, range(20, 28)),
            self.request(6, 2, range(30, 38)),
        ]
        batch = types.SimpleNamespace(
            reqs=reqs, forward_mode=1,
            req_pool_indices=types.SimpleNamespace(values=[1, 3, 5]),
        )
        self.manager.prepare_for_forward(batch, chunked_req=reqs[1])
        first_mask = batch.ne_skip_token_table_update
        snapshots = []
        self.pending.append(lambda: snapshots.append(copy.deepcopy(self.table.values)))
        for req in reqs:
            req.origin_input_ids[:] = [x + 100 for x in req.origin_input_ids]
        self.manager.prepare_for_forward(batch, chunked_req=None)
        self.assertIsNone(batch.ne_skip_token_table_update)
        # No work has been executed yet; both batches must retain their own data.
        self.assertTrue(all(x == -1 for row in self.table.values for x in row))
        for operation in self.pending:
            operation()
        self.assertEqual(first_mask.values, [False, True, False])
        for row, start, expected in ((1, 0, [10, 11, 12]),
                                     (3, 0, [20, 21, 22]),
                                     (5, 4, [34, 35, 36, 37])):
            self.assertEqual(snapshots[0][row][start:start + len(expected)], expected)
            self.assertEqual(
                self.table.values[row][start:start + len(expected)],
                [x + 100 for x in expected],
            )
        self.assertEqual(self.table.values[5][:4], [-1] * 4)
        self.assertEqual(len({id(source) for source in self.host_sources}), 9)

    def test_pd_decode_restores_history_once_without_per_step_staging(self):
        req = self.request(
            0, 0, [11, 12], req_pool_idx=4, rid="pd",
            ngram_token_table_needs_init=True,
        )
        req.output_ids = [13]
        batch = types.SimpleNamespace(reqs=[req], forward_mode=2)
        self.manager.prepare_for_forward(batch, chunked_req=None)
        self.assertFalse(req.ngram_token_table_needs_init)
        self.assertEqual(len(self.host_sources), 4)
        pending_count = len(self.pending)
        self.manager.prepare_for_forward(batch, chunked_req=None)
        self.assertEqual(len(self.pending), pending_count)
        self.assertIsNone(batch.ne_skip_token_table_update)
        for operation in self.pending:
            operation()
        self.assertEqual(self.table.values[4], [11, 12, 13] + [-1] * 13)

    def test_decode_sample_update_passes_device_inputs_without_host_reads(self):
        info = types.SimpleNamespace(token_table=self.table, skip_token_table_update=None)
        sampled, rows, lengths = object(), object(), object()
        updated = self.ns["update_ngram_token_table_after_sampling"](
            ngram_embedding_info=info, next_token_ids=sampled,
            req_pool_indices=rows, seq_lens=lengths, batch_size=2,
        )
        self.assertTrue(updated)
        self.assertEqual(self.host_sources, [])
        args, kwargs = self.kernel_calls[0]
        self.assertEqual(args, (self.table, sampled, rows, lengths, None))
        self.assertEqual(kwargs, {"batch_size": 2})

    def test_disabled_and_empty_batches_do_not_stage_history(self):
        self.assertIsNone(self.manager.prepare_for_forward(None, chunked_req=None))
        manager = self.ns["NgramEmbeddingManager"](enabled=False, table=None, n=0, k=0)
        batch = object()
        self.assertIs(manager.prepare_for_forward(batch, chunked_req=None), batch)
        self.assertEqual(self.host_sources, [])

    def test_non_npu_keeps_original_tensor_creation(self):
        calls = []
        self.ns["is_npu"] = lambda: False
        self.ns["torch"] = types.SimpleNamespace(
            tensor=lambda values, **kw: calls.append((list(values), kw)) or "tensor"
        )
        result = self.ns["_history_tensor"]([1, 2], dtype="int32", device="cuda")
        self.assertEqual(result, "tensor")
        self.assertEqual(calls, [([1, 2], {"dtype": "int32", "device": "cuda"})])


class TestEagerBreakContracts(unittest.TestCase):
    def setUp(self):
        self.namespace = load_nodes(
            SRT / "model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
            ["_eager_break_context", "eager_on_graph", "_weak_ref_if_tensor", "_copy_output"],
            {
                "contextmanager": contextmanager,
                "torch": types.SimpleNamespace(is_tensor=lambda value: False),
                "logger": logging.getLogger(__name__),
                "_current_capture_var": ContextVar("test_capture", default=None),
                "_current_stream_var": ContextVar("test_stream", default=None),
                "_forked_streams_var": ContextVar("test_forks", default=None),
            },
        )
        self.events = []
        self.capture = types.SimpleNamespace(
            _end_current_segment=lambda: self.events.append("end"),
            _begin_new_segment=lambda: self.events.append("begin"),
            _barrier_fn=lambda: self.assertIsNone(self.namespace["_current_capture_var"].get()),
            cuda_graph=types.SimpleNamespace(_break_fns=[]),
        )

    def test_nested_break_is_one_break_and_restores_context(self):
        ns = self.namespace
        decorate = ns["eager_on_graph"](True)

        @decorate
        def inner():
            self.assertIsNone(ns["_current_capture_var"].get())
            self.assertIsNone(ns["_forked_streams_var"].get())
            return 7

        @decorate
        def outer():
            return inner()

        token = ns["_current_capture_var"].set(self.capture)
        try:
            self.assertEqual(outer(), 7)
            self.assertIs(ns["_current_capture_var"].get(), self.capture)
        finally:
            ns["_current_capture_var"].reset(token)
        self.assertEqual(self.events, ["end", "begin"])
        self.assertEqual(len(self.capture.cuda_graph._break_fns), 1)
        self.assertEqual(self.capture.cuda_graph._break_fns[0](), 7)

    def test_failed_break_restores_context_without_starting_segment(self):
        ns = self.namespace

        @ns["eager_on_graph"](True)
        def broken():
            raise ValueError("broken callback")

        token = ns["_current_capture_var"].set(self.capture)
        try:
            with self.assertRaisesRegex(ValueError, "broken callback"):
                broken()
            self.assertIs(ns["_current_capture_var"].get(), self.capture)
        finally:
            ns["_current_capture_var"].reset(token)
        self.assertEqual(self.events, ["end"])


class TestGatePolicy(unittest.TestCase):
    def test_graph_prefill_disables_cmo_without_changing_decode(self):
        tree = ast.parse((SRT / "models/welmv4.py").read_text(encoding="utf-8"))
        expression = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "enable_npu_weight_prefetch"
                for target in node.targets
            )
        )
        code = compile(ast.Expression(expression), "cmo_policy", "eval")
        batch = types.SimpleNamespace(welm_prefill_graph=None)
        ns = dict(
            _is_npu=True,
            forward_batch=batch,
            hidden_states=Rows(2),
            is_ordinary_prefill_non_consumer_layer=False,
        )
        self.assertTrue(eval(code, ns))  # Existing decode/mirror prefetch.
        batch.welm_prefill_graph = object()
        self.assertFalse(eval(code, ns))  # Both QKV and router use this gate.
        batch.welm_prefill_graph = None
        self.assertTrue(eval(code, ns))
        ns["is_ordinary_prefill_non_consumer_layer"] = True
        self.assertFalse(eval(code, ns))  # Original non-consumer eager policy.

    def test_only_graph_prefill_disables_gate_side_stream(self):
        tree = ast.parse((SRT / "models/welmv4.py").read_text(encoding="utf-8"))
        expression = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "enable_npu_gate_alt_stream"
                for target in node.targets
            )
        )
        code = compile(ast.Expression(expression), "gate_policy", "eval")
        batch = types.SimpleNamespace(welm_prefill_graph=None)
        ns = dict(
            _is_npu=True,
            forward_batch=batch,
            envs=types.SimpleNamespace(
                SGLANG_NPU_USE_MULTI_STREAM=types.SimpleNamespace(get=lambda: True)
            ),
            self=types.SimpleNamespace(
                alt_stream=object(), gated_self_attention_headwise=True
            ),
            hidden_states=Rows(2),
            use_decode_like_stream_policy=True,
            fused_qkv=None,
        )
        self.assertTrue(eval(code, ns))  # Existing eager mirror/decode policy.
        batch.welm_prefill_graph = object()
        self.assertFalse(eval(code, ns))
        batch.welm_prefill_graph = None
        self.assertTrue(eval(code, ns))  # No global multi-stream mutation.


class TestCaptureFailureCleanup(unittest.TestCase):
    def test_failed_capture_begin_restores_context_and_wait_hook(self):
        graph_type = type("FakeGraph", (), {})
        events = []
        ns = load_nodes(
            SRT / "model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
            ["BreakableCUDAGraphCapture"],
            {
                "BreakableCUDAGraph": graph_type,
                "_current_capture_var": ContextVar("failed_capture", default=None),
                "_current_stream_var": ContextVar("failed_stream", default=None),
                "_forked_streams_var": ContextVar("failed_forks", default=None),
                "_install_wait_stream_hook": lambda: events.append("install"),
                "_uninstall_wait_stream_hook": lambda: events.append("uninstall"),
                "get_device_module": lambda: types.SimpleNamespace(
                    current_stream=lambda: "main"
                ),
            },
        )
        capture = ns["BreakableCUDAGraphCapture"](graph_type())

        def fail():
            raise RuntimeError("begin failed")

        capture._begin_new_segment = fail
        with self.assertRaisesRegex(RuntimeError, "begin failed"):
            with capture:
                self.fail("failed __enter__ must not run body")
        for name in ("_current_capture_var", "_current_stream_var", "_forked_streams_var"):
            self.assertIsNone(ns[name].get())
        self.assertEqual(events, ["install", "uninstall"])


class TestNormalCaptureScope(unittest.TestCase):
    def test_welm_scope_captures_normal_or_raises_never_silently_eager(self):
        path = SRT / "layers/moe/ep_moe/layer.py"
        cls = next(
            node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == "DeepEPMoE"
        )
        method = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        scope = ContextVar("normal_scope", default=False)
        imported = types.ModuleType("sglang.srt.model_executor.runner.welm_prefill_graph")
        imported.welm_normal_graph_scope = scope
        switches = {"ag": True, "a2a": False, "extend": True}
        events = []
        ns = {
            "is_in_breakable_cuda_graph": lambda: True,
            "_is_npu": True,
            "torch": types.SimpleNamespace(
                bfloat16="bf16", empty_like=lambda value: "eager_output"
            ),
            "envs": types.SimpleNamespace(
                SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=types.SimpleNamespace(
                    get=lambda: switches["ag"]
                ),
                SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=types.SimpleNamespace(
                    get=lambda: switches["a2a"]
                ),
            ),
            "get_is_extend_in_batch": lambda: switches["extend"],
            "TopKOutputChecker": types.SimpleNamespace(format_is_standard=lambda _: True),
        }
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        layer = types.SimpleNamespace(
            deprecate_flag=True,
            forward_impl=lambda *args: events.append("captured"),
            a2a_forward_with_output=lambda *args: events.append("eager"),
        )
        hidden = types.SimpleNamespace(dtype="bf16")
        topk = types.SimpleNamespace(topk_ids=None, topk_weights=None, router_logits=None)
        with patch.dict(sys.modules, {imported.__name__: imported}):
            ns["forward"](layer, hidden, topk)
            self.assertEqual(events, ["eager"])  # Other models retain protection.
            token = scope.set(True)
            try:
                ns["forward"](layer, hidden, topk)
                self.assertEqual(events, ["eager", "captured"])
                for name, value in (("ag", False), ("a2a", True), ("extend", False)):
                    old = switches[name]
                    switches[name] = value
                    with self.assertRaisesRegex(RuntimeError, "capability changed"):
                        ns["forward"](layer, hidden, topk)
                    switches[name] = old
                hidden.dtype = "fp8"
                with self.assertRaisesRegex(RuntimeError, "capability changed"):
                    ns["forward"](layer, hidden, topk)
            finally:
                scope.reset(token)
        self.assertEqual(events, ["eager", "captured"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
