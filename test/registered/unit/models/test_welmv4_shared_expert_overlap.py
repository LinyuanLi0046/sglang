"""Exercise WeLM's real Python scheduling with simulated kernels/collectives.

AST loading keeps these CPU tests independent of torch_npu and the model's
optional kernel imports. It does not model device timing or graph capture.
"""

import __future__
import ast
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "python/sglang/srt/models/welmv4.py"
DP_MODEL = ROOT / "python/sglang/srt/models/welmv4_dp_attention.py"
NPU_UTILS = ROOT / "python/sglang/srt/hardware_backend/npu/utils.py"
MODES = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"


def load_definitions(path, names, namespace, methods=None):
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if getattr(node, "name", None) not in names:
            continue
        if methods is not None and isinstance(node, ast.ClassDef):
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
        nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(
        compile(module, str(path), "exec", __future__.annotations.compiler_flag),
        namespace,
    )


class Tensor:
    """Constant-valued tensor carrying stream ownership and a logical shape."""

    def __init__(self, value, shape=(2, 4), events=None):
        self.value, self.shape = value, shape
        self.events = events if events is not None else []
        self.pending = False
        self.device = "npu"

    def view(self, *shape):
        return self

    def to(self, *args, **kwargs):
        return self

    def record_stream(self, stream):
        self.events.append("record:" + stream.name)

    def __add__(self, other):
        assert not other.pending, "Shared output consumed before stream join"
        self.events.append("add")
        return Tensor(self.value + other.value, self.shape, self.events)

    def add_(self, other):
        self.value = (self + other).value
        return self


class Module:
    def register_buffer(self, name, value, **kwargs):
        setattr(self, name, value)


class TestWeLMSharedExpertOverlap(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.pending = []
        self.multi_stream = True
        self.backend = "none"
        self.plan = None
        self.tp_size = 4
        self.rows = 2
        self.collectives = []
        fixture = self

        class Stream:
            def __init__(self, name="shared"):
                self.name = name

            def wait_stream(self, other):
                fixture.events.append("join" if self.name == "main" else "fork")
                if self.name == "main":
                    for tensor in fixture.pending:
                        tensor.pending = False

        class StreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.previous = fixture.current_stream
                fixture.current_stream = self.stream

            def __exit__(self, *args):
                fixture.current_stream = self.previous

        self.current_stream = Stream("main")
        device = SimpleNamespace(
            Stream=Stream,
            stream=StreamContext,
            current_stream=lambda: self.current_stream,
        )
        self.backend_api = SimpleNamespace(
            is_none=lambda: self.backend == "none",
            is_deepep=lambda: self.backend == "deepep",
        )
        self.namespace = {
            "IntEnum": IntEnum,
            "auto": auto,
            "nn": SimpleNamespace(Module=Module),
            "torch": SimpleNamespace(
                float32="float32",
                nn=SimpleNamespace(Parameter=lambda x: x),
                zeros=lambda *args, **kwargs: Tensor(0),
                get_device_module=lambda: device,
                mm=lambda *args, **kwargs: self.event_tensor("router", 0),
            ),
            "_is_npu": True,
            "get_moe_a2a_backend": lambda: self.backend_api,
            "get_tensor_model_parallel_world_size": lambda: self.tp_size,
            "get_welm_runner_build_plan_for_init": lambda: self.plan,
            "get_parallel": lambda: SimpleNamespace(
                moe_ep_size=4 if self.backend == "deepep" else 1,
                moe_tp_size=1 if self.backend == "deepep" else self.tp_size,
                attn_tp_rank=0,
                tp_rank=2,
            ),
            "envs": SimpleNamespace(
                SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(
                    get=lambda: self.multi_stream
                ),
                SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=SimpleNamespace(get=lambda: True),
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: False),
            ),
            "get_bool_env_var": lambda *args: False,
            "TopK": lambda **kwargs: self.topk,
            "ReplicatedLinear": lambda *args, **kwargs: SimpleNamespace(
                weight=Tensor(0)
            ),
            "get_moe_impl_class": lambda _: lambda **kwargs: SimpleNamespace(),
            "Qwen2MoeMLP": self.shared_factory,
            "add_prefix": lambda name, prefix: name,
            "moe_expert_parallel_all_reduce": lambda x: self.all_reduce("ep", x),
            "tensor_model_parallel_all_reduce": lambda x: self.all_reduce("tp", x),
            "DeepEPMode": SimpleNamespace(NORMAL="normal", LOW_LATENCY="ll"),
            "WelmLayerCapabilities": SimpleNamespace,
            "share_stream": None,
        }
        self.topk = type(
            "TopK",
            (),
            {
                "__call__": lambda *args, **kwargs: fixture.events.append("topk"),
                "empty_topk_output": lambda *args, **kwargs: None,
            },
        )()
        load_definitions(MODES, {"ForwardMode"}, self.namespace)
        load_definitions(
            NPU_UTILS,
            {
                "get_share_stream",
                "set_share_stream",
                "wait_share_stream",
                "process_shared_expert",
            },
            self.namespace,
        )
        load_definitions(
            MODEL,
            {"Qwen2MoeSparseMoeBlock"},
            self.namespace,
            methods={
                "__init__",
                "forward",
                "_forward_shared_expert_tp",
                "_use_decode_like_shared_expert",
            },
        )
        load_definitions(
            MODEL,
            {"WeLMV4MoeForCausalLM"},
            self.namespace,
            methods={"load_weights", "_load_parameter_with_shared_tp_copy"},
        )
        load_definitions(DP_MODEL, {"inspect_welm_layer_capabilities"}, self.namespace)
        self.Modes = self.namespace["ForwardMode"]

    def event_tensor(self, event, value):
        self.events.append(event)
        tensor = Tensor(value, (self.rows, 4), self.events)
        tensor.pending = self.current_stream.name == "shared"
        if tensor.pending:
            self.pending.append(tensor)
        return tensor

    def shared_factory(self, **kwargs):
        tp = kwargs.get("tp_size", self.tp_size)
        intermediate = kwargs["intermediate_size"]
        return SimpleNamespace(
            prefix=kwargs.get("prefix"),
            quant_config=kwargs.get("quant_config"),
            gate_up_proj=SimpleNamespace(
                tp_size=tp,
                tp_rank=kwargs.get("tp_rank", 0),
                output_width=2 * intermediate // tp,
            ),
            down_proj=SimpleNamespace(
                tp_size=tp,
                tp_rank=kwargs.get("tp_rank", 0),
                input_width=intermediate // tp,
                reduce_results=kwargs["reduce_results"],
            ),
        )

    def all_reduce(self, group, tensor):
        # A rank contributes 2; the other ranks together contribute 7.
        # Ordinary TP prefill also contributes 5/4 from the shared shard.
        sharded = "shared_tp" in self.events
        self.assertEqual(tensor.value, 3.25 if sharded else 2)
        self.events.append("ar:" + group)
        self.collectives.append(group)
        tensor.value += 10.75 if sharded else 7
        return tensor

    def block(self):
        config = SimpleNamespace(
            num_experts=8,
            num_hidden_layers=2,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            moe_routing_type="default",
            hidden_size=4,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            hidden_act="silu",
            has_shared_expert_gate=False,
        )
        block = self.namespace["Qwen2MoeSparseMoeBlock"](0, config)
        block.get_npu_router_compute_weight_t = lambda: None
        def shared(x, gate_up=None, use_tp_shard=False):
            return self.event_tensor(
                "shared_tp" if use_tp_shard else "shared", 1.25 if use_tp_shard else 5
            )

        block._forward_shared_expert = shared
        block._resolve_deepep_mode_for_topk = lambda _: "normal"
        fixture = self
        block.experts = type(
            "Experts",
            (),
            {
                "__call__": lambda *args: fixture.event_tensor("routed", 2),
                "forward_local_ep_partial": lambda *args: fixture.event_tensor(
                    "routed", 2
                ),
            },
        )()
        return block

    def run_forward(
        self, mode, *, ep=False, mirror=False, scattered=False, block=None, **kwargs
    ):
        self.backend = "deepep" if ep else self.backend
        block = block if block is not None else self.block()
        block.is_kv_mirror_consumer = mirror
        batch = SimpleNamespace(
            forward_mode=mode,
            enable_kv_mirror=mirror,
            num_token_non_padded=None,
            num_token_non_padded_cpu=self.rows,
            welmv4_npu_deepep_scattered=scattered,
            welmv4_npu_deepep_full_mirror=mirror,
        )
        return block.forward(
            Tensor(1, (self.rows, 4), self.events),
            None,
            batch,
            use_welm_local_ep_moe=ep,
            **kwargs,
        )

    def assert_overlap(self, group):
        self.assertEqual(self.collectives, [group])
        self.assertEqual(self.events.count("shared"), 1)
        order = [
            "routed",
            "fork",
            "shared",
            "ar:" + group,
            "join",
            "record:main",
            "add",
        ]
        indices = [self.events.index(event) for event in order]
        self.assertEqual(indices, sorted(indices))

    def test_complete_shared_weights_under_ep_and_tp(self):
        for backend in ("none", "deepep"):
            with self.subTest(backend=backend):
                self.backend = backend
                shared = self.block().shared_expert
                self.assertEqual(shared.gate_up_proj.output_width, 16)
                self.assertEqual(shared.down_proj.input_width, 8)
                self.assertFalse(shared.down_proj.reduce_results)

    def test_only_multi_rank_pure_tp_keeps_both_weight_layouts(self):
        block = self.block()
        shared_tp = block.shared_expert_tp
        self.assertEqual(shared_tp.gate_up_proj.output_width, 4)
        self.assertEqual(shared_tp.down_proj.input_width, 2)
        self.assertEqual(shared_tp.gate_up_proj.tp_rank, 2)
        self.assertFalse(shared_tp.down_proj.reduce_results)
        self.assertEqual(shared_tp.prefix, block.shared_expert.prefix)
        self.backend = "deepep"
        self.assertIsNone(self.block().shared_expert_tp)
        self.backend = "none"
        self.tp_size = 1
        self.assertIsNone(self.block().shared_expert_tp)

    def test_decode_and_verify_overlap_without_override(self):
        for ep in (False, True):
            for mode_name in ("DECODE", "TARGET_VERIFY"):
                with self.subTest(ep=ep, mode=mode_name):
                    self.setUp()
                    result = self.run_forward(getattr(self.Modes, mode_name), ep=ep)
                    self.assertEqual(result.value, 14)
                    self.assert_overlap("ep" if ep else "tp")

    def test_mirror_prefill_overlap(self):
        for ep in (False, True):
            with self.subTest(ep=ep):
                self.setUp()
                result = self.run_forward(self.Modes.EXTEND, ep=ep, mirror=True)
                self.assertEqual(result.value, 14)
                self.assert_overlap("ep" if ep else "tp")

    def test_nextn_extend_override_and_non_dp_wiring(self):
        tree = ast.parse(MODEL.read_text(encoding="utf-8"))
        decoder = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "Qwen2MoeDecoderLayer"
        )
        forward = next(
            n for n in decoder.body if getattr(n, "name", None) == "_forward_non_dp"
        )
        mlp_call = next(
            n
            for n in ast.walk(forward)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "mlp"
        )
        override = next(
            k.value
            for k in mlp_call.keywords
            if k.arg == "use_welm_decode_like_stream_policy"
        )
        for ep in (False, True):
            for mode_name in ("EXTEND", "DRAFT_EXTEND_V2"):
                with self.subTest(ep=ep, mode=mode_name):
                    self.setUp()
                    enabled = eval(
                        compile(ast.Expression(override), str(MODEL), "eval"),
                        {"self": SimpleNamespace(is_nextn=True)},
                    )
                    result = self.run_forward(
                        getattr(self.Modes, mode_name),
                        ep=ep,
                        use_welm_decode_like_stream_policy=enabled,
                    )
                    self.assertEqual(result.value, 14)
                    self.assert_overlap("ep" if ep else "tp")

    def test_resolved_tp_group_overrides_global_deepep_backend(self):
        self.backend = "deepep"
        self.plan = SimpleNamespace(
            has_moe_ep=False,
            moe_ep_size=1,
            moe_tp_size=4,
            moe_tp_group=SimpleNamespace(rank_in_group=1),
        )
        group = SimpleNamespace(all_reduce=lambda x: self.all_reduce("draft-tp", x))
        result = self.run_forward(self.Modes.TARGET_VERIFY, resolved_moe_tp_group=group)
        self.assertEqual(result.value, 14)
        self.assert_overlap("draft-tp")

    def test_resolved_ep_group_is_used_once_with_inplace_merge(self):
        group = SimpleNamespace(all_reduce=lambda x: self.all_reduce("resolved-ep", x))
        result = self.run_forward(
            self.Modes.DECODE,
            ep=True,
            resolved_moe_ep_group=group,
            allow_inplace_expert_shared_merge=True,
        )
        self.assertEqual(result.value, 14)
        self.assert_overlap("resolved-ep")

    def test_serial_fallback_still_adds_shared_after_only_one_reduce(self):
        for ep in (False, True):
            for forced in (False, True):
                with self.subTest(ep=ep, forced=forced):
                    self.setUp()
                    self.multi_stream = forced
                    result = self.run_forward(
                        self.Modes.DECODE, ep=ep, force_serial_shared_expert=forced
                    )
                    self.assertEqual(result.value, 14)
                    self.assertNotIn("fork", self.events)
                    self.assertEqual(len(self.collectives), 1)
                    self.assertEqual(self.events.count("shared"), 1)

    def test_ordinary_tp_prefill_adds_sharded_shared_before_single_reduce(self):
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                self.setUp()
                result = self.run_forward(
                    self.Modes.EXTEND, allow_inplace_expert_shared_merge=inplace
                )
                self.assertEqual(result.value, 14)
                self.assertNotIn("fork", self.events)
                self.assertNotIn("shared", self.events)
                self.assertEqual(self.collectives, ["tp"])
                self.assertLess(
                    self.events.index("shared_tp"), self.events.index("routed")
                )
                self.assertLess(self.events.index("add"), self.events.index("ar:tp"))

    def test_nextn_extend_always_uses_full_copy_even_without_override(self):
        block = self.block()
        block.is_nextn = True
        self.assertEqual(
            self.run_forward(self.Modes.DRAFT_EXTEND_V2, block=block).value, 14
        )
        self.assert_overlap("tp")

    def test_sharded_prefill_can_leave_reduction_to_caller(self):
        result = self.run_forward(self.Modes.EXTEND, use_reduce_scatter=True)
        self.assertEqual(result.value, 3.25)
        self.assertEqual(self.collectives, [])
        self.assertIn("shared_tp", self.events)

    def test_components_are_published_only_after_join(self):
        _, routed, shared = self.run_forward(
            self.Modes.DECODE,
            ep=True,
            return_components=True,
            skip_component_output=True,
        )
        self.assertEqual((routed.value, shared.value), (9, 5))
        self.assertFalse(shared.pending)
        self.assertLess(self.events.index("ar:ep"), self.events.index("join"))
        self.assertNotIn("add", self.events)

    def test_single_rank_keeps_compute_overlap_without_collective(self):
        self.tp_size = 1
        result = self.run_forward(self.Modes.DECODE)
        self.assertEqual(result.value, 7)
        self.assertEqual(self.collectives, [])
        self.assertLess(self.events.index("fork"), self.events.index("routed"))
        self.assertFalse(result.pending)

    def test_no_shared_and_empty_ep_do_not_launch_shared_stream(self):
        block = self.block()
        block.shared_expert = None
        self.assertEqual(self.run_forward(self.Modes.DECODE, block=block).value, 9)
        self.assertNotIn("fork", self.events)
        self.setUp()
        self.rows = 0
        self.run_forward(self.Modes.DECODE, ep=True)
        self.assertEqual(self.collectives, ["ep"])
        self.assertNotIn("fork", self.events)

    def test_replicated_shared_rejects_deferred_reduce_scatter(self):
        with self.assertRaisesRegex(RuntimeError, "deferred MoE ReduceScatter"):
            self.run_forward(self.Modes.DECODE, use_reduce_scatter=True)

    def test_decoder_disables_deferred_reduce_scatter_for_shared_replicas(self):
        tree = ast.parse(MODEL.read_text(encoding="utf-8"))
        decoder = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "Qwen2MoeDecoderLayer"
        )
        communicator = next(
            n
            for n in ast.walk(decoder)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "LayerCommunicator"
        )
        selector = next(
            k.value for k in communicator.keywords if k.arg == "allow_reduce_scatter"
        )
        layer = SimpleNamespace(is_layer_sparse=True, mlp=self.block())
        self.assertTrue(
            eval(
                compile(ast.Expression(selector), str(MODEL), "eval"), {"self": layer}
            )
        )
        forward = next(
            n for n in decoder.body if getattr(n, "name", None) == "_forward_non_dp"
        )
        rs_selector = next(
            n.value
            for n in ast.walk(forward)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "use_reduce_scatter"
                for t in n.targets
            )
        )
        layer.layer_communicator = SimpleNamespace(
            should_use_reduce_scatter=lambda _: True
        )
        for mode, expected in (
            (self.Modes.EXTEND, True), (self.Modes.TARGET_VERIFY, False)
        ):
            batch = SimpleNamespace(forward_mode=mode, enable_kv_mirror=False)
            self.assertEqual(
                eval(
                    compile(ast.Expression(rs_selector), str(MODEL), "eval"),
                    {
                        "self": layer,
                        "forward_batch": batch,
                        "output_hidden_is_scattered": False,
                        "use_full_mirror_layout": False,
                    },
                ),
                expected,
            )

    def test_normal_prefill_keeps_dispatch_overlap_without_extra_allreduce(self):
        self.backend = "deepep"
        result = self.run_forward(self.Modes.EXTEND, scattered=True)
        self.assertEqual(result.value, 7)
        self.assertEqual(self.collectives, [])
        self.assertLess(self.events.index("topk"), self.events.index("fork"))
        self.assertLess(self.events.index("shared"), self.events.index("routed"))
        self.assertEqual(self.events.count("shared"), 1)

    def test_megamoe_keeps_only_gate_up_overlap(self):
        self.backend = "deepep"
        block = self.block()
        block.shared_expert.gate_up_proj = lambda x: (
            self.event_tensor("gate_up", 0),
            None,
        )
        block._forward_shared_expert = lambda x, gate_up: self.event_tensor("shared", 5)
        block.welm_prefill_megamoe = SimpleNamespace(
            local_valid_rows=lambda *args: self.rows,
            forward_layer=lambda *args, **kwargs: self.event_tensor("routed", 2),
        )
        result = self.run_forward(
            self.Modes.EXTEND,
            block=block,
            use_welm_prefill_megamoe=True,
        )
        self.assertEqual(result.value, 7)
        self.assertEqual(self.collectives, [])
        self.assertLess(self.events.index("gate_up"), self.events.index("topk"))
        self.assertLess(self.events.index("join"), self.events.index("shared"))
        self.assertLess(self.events.index("shared"), self.events.index("routed"))

    def test_dp_capabilities_require_complete_shared_in_both_topologies(self):
        inspect = self.namespace["inspect_welm_layer_capabilities"]
        for ep in (False, True):
            with self.subTest(ep=ep):
                ep_size, tp_size = (4, 1) if ep else (1, 4)
                plan = SimpleNamespace(
                    has_moe_ep=ep,
                    moe_ep_size=ep_size,
                    moe_tp_size=tp_size,
                    moe_ep_group=SimpleNamespace(rank_in_group=0) if ep else None,
                    moe_tp_group=SimpleNamespace(rank_in_group=0),
                )
                shared = self.shared_factory(
                    intermediate_size=8, tp_size=1, reduce_results=False
                )
                layer = SimpleNamespace(
                    mlp=SimpleNamespace(
                        tp_size=tp_size,
                        shared_expert=shared,
                        shared_expert_tp=(
                            self.shared_factory(
                                intermediate_size=8,
                                tp_size=tp_size,
                                reduce_results=False,
                            )
                            if not ep else None
                        ),
                        welm_local_ep_kernel_available=True,
                        experts=SimpleNamespace(
                            moe_ep_size=ep_size,
                            moe_ep_rank=0,
                            moe_tp_size=tp_size,
                            moe_tp_rank=0,
                            reduce_results=False,
                            local_ep_dispatcher=object(),
                        ),
                    ),
                    self_attn=SimpleNamespace(
                        o_proj=SimpleNamespace(tp_size=4, reduce_results=False)
                    ),
                )
                self.assertTrue(inspect(layer, plan).shared_expert_ep_replicated)
                if not ep:
                    shard = layer.mlp.shared_expert_tp
                    layer.mlp.shared_expert_tp = None
                    with self.assertRaisesRegex(RuntimeError, "second TP-sharded"):
                        inspect(layer, plan)
                    layer.mlp.shared_expert_tp = shard
                    shard.down_proj.tp_rank = 1
                    with self.assertRaisesRegex(RuntimeError, "resolved MoE-TP group"):
                        inspect(layer, plan)
                    shard.down_proj.tp_rank = 0
                shared.gate_up_proj.tp_size = 4
                shared.down_proj.tp_size = 4
                with self.assertRaisesRegex(
                    RuntimeError, "shared-expert weight coverage"
                ):
                    inspect(layer, plan)

    def test_target_and_nextn_load_weights_and_scales_into_both_copies(self):
        self.namespace["FusedMoE"] = SimpleNamespace(
            make_expert_params_mapping=lambda **_: []
        )
        self.namespace["get_layer_id"] = lambda _: None
        self.namespace["default_weight_loader"] = lambda p, w: setattr(p, "loaded", w)
        for nextn in (False, True):
            with self.subTest(nextn=nextn):
                model = self.namespace["WeLMV4MoeForCausalLM"]()
                model.config = SimpleNamespace(
                    num_experts=8,
                    num_hidden_layers=2,
                    num_target_hidden_layers=2,
                    num_nextn_predict_layers=1,
                )
                model.model = SimpleNamespace()
                finished = []
                model.post_init_after_load_weights = lambda **kw: finished.append(kw)
                prefix = "model.decoder_layers.0" if nextn else "model.layers.0"
                ckpt_prefix = "model.layers.2" if nextn else "model.layers.0"
                params, loads = {}, {}
                for copy in ("shared_expert", "shared_expert_tp"):
                    for proj in ("gate_up_proj", "down_proj"):
                        for attr in ("weight", "weight_scale"):
                            name = f"{prefix}.mlp.{copy}.{proj}.{attr}"
                            loads[name] = []
                            def loader(p, w, *args, key=name):
                                loads[key].append((w, args))

                            params[name] = SimpleNamespace(weight_loader=loader)
                gate_name = f"{prefix}.mlp.shared_expert_gate.weight"
                params[gate_name] = SimpleNamespace()
                model.named_parameters = lambda: params.items()
                weights = [
                    (f"{ckpt_prefix}.mlp.shared_expert.{proj}.{attr}", object())
                    for proj in ("gate_proj", "up_proj", "down_proj")
                    for attr in ("weight", "weight_scale")
                ]
                gate = object()
                weights.append((f"{ckpt_prefix}.mlp.shared_expert_gate.weight", gate))
                model.load_weights(iter(weights), is_nextn=nextn)
                for name, entries in loads.items():
                    if ".shared_expert." not in name:
                        continue
                    tp_name = name.replace(".shared_expert.", ".shared_expert_tp.")
                    self.assertEqual(entries, loads[tp_name])
                    self.assertEqual(
                        [args for _, args in entries],
                        [(0,), (1,)] if "gate_up_proj" in name else [()],
                    )
                self.assertIs(params[gate_name].loaded, gate)
                self.assertEqual(finished, [{"is_nextn": nextn}])


if __name__ == "__main__":
    unittest.main()
