"""CPU contracts for the shared WeLM fused-QKV switch and model dispatch.

Actual model statements run with the metadata doubles from the static Decode
contract tests. No SGLang, torch, NPU or DSL module is imported by these tests.
"""

import ast
import builtins
import copy
import math
from enum import Enum, auto
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Optional, Tuple
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[5]
_MODEL = _ROOT / "python/sglang/srt/models/welmv4.py"
_LAYOUT = _ROOT / "python/sglang/srt/layers/welmv4_qkv_layout.py"
_HELPERS = Path(__file__).with_name("test_welm_static_decode_qkv.py")
_BINDING_MODULE = "sglang.srt.layers.welmv4_static_decode_qkv"


class _Mode(Enum):
    EXTEND = auto()
    MIXED = auto()
    DECODE = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND = auto()


def _attention_ast():
    tree = ast.parse(_MODEL.read_text(), filename=str(_MODEL))
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen2MoeAttention"
    )


def _method(name):
    return copy.deepcopy(next(
        node for node in _attention_ast().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ))


def _exec_nodes(nodes, namespace, filename):
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(tree, str(filename), "exec", dont_inherit=True), namespace)


@pytest.fixture
def doubles():
    """Reuse only existing doubles, without importing its tests or fixtures."""
    source = ast.parse(_HELPERS.read_text(), filename=str(_HELPERS))
    names = {
        "_Tensor", "_RotaryEmbedding", "_UnquantizedLinearMethod", "_load_layout"
    }
    definitions = [
        node for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert len(definitions) == len(names)
    namespace = dict(
        ast=ast, math=math, SimpleNamespace=SimpleNamespace,
        _DEVICE=SimpleNamespace(type="npu", index=0), _LAYOUT=_LAYOUT,
    )
    _exec_nodes(definitions, namespace, _HELPERS)
    base_tensor = namespace["_Tensor"]

    class Tensor(base_tensor):
        def to(self, dtype):
            # All accepted cases supply int64 metadata, so conversion must
            # preserve the tensor object used for physical-slot selection.
            assert dtype == self.dtype
            return self

    return SimpleNamespace(
        Tensor=Tensor,
        BaseTensor=base_tensor,
        Rotary=namespace["_RotaryEmbedding"],
        Unquantized=namespace["_UnquantizedLinearMethod"],
        layout=namespace["_load_layout"](),
    )


def _init_gate(*, tp, global_tp, dp, enabled, nextn=False, consumer=False):
    init = _method("__init__")
    start = next(
        i for i, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "fused_qkv_enabled" for t in node.targets)
    )
    layer = SimpleNamespace(kv_mirror_layer_idx=3, kv_mirror_layers=[3] if consumer else [])
    static_factory = Mock(name="StaticDecodeQKV")
    imports = []

    def environment(name, default):
        assert (name, default) == ("SGLANG_NPU_WELMV4_FUSED_QKV", "0")
        return enabled

    def isolated_import(name, *args, **kwargs):
        imports.append(name)
        assert name == _BINDING_MODULE
        return SimpleNamespace(StaticDecodeQKV=static_factory)

    env_reader = Mock(side_effect=environment)
    namespace = dict(
        __builtins__=dict(vars(builtins), __import__=isolated_import),
        self=layer,
        _is_npu=True,
        attn_tp_size=tp,
        is_nextn=nextn,
        get_bool_env_var=env_reader,
        get_tensor_model_parallel_world_size=lambda: global_tp,
        is_dp_attention_enabled=lambda: dp,
    )
    _exec_nodes(init.body[start:], namespace, _MODEL)
    env_reader.assert_called_once_with("SGLANG_NPU_WELMV4_FUSED_QKV", "0")
    return layer, static_factory, imports


@pytest.mark.parametrize(
    "tp,global_tp,dp,legacy",
    [(1, 1, False, False), (2, 2, False, False), (4, 4, False, True),
     (1, 16, True, False), (2, 16, True, False), (4, 16, True, False),
     (4, 16, False, False), (4, 4, True, False)],
)
def test_single_switch_uses_attention_tp_and_preserves_legacy_guard(tp, global_tp, dp, legacy):
    layer, factory, imports = _init_gate(tp=tp, global_tp=global_tp, dp=dp, enabled=True)
    assert layer.use_npu_fused_qkv is True
    assert layer.fused_qkv_attn_tp_size == tp
    assert layer.use_legacy_npu_fused_qkv is legacy
    factory.assert_called_once_with(tp)
    assert layer.static_decode_qkv is factory.return_value
    assert imports == [_BINDING_MODULE]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_single_switch_off_disables_both_paths_without_import(tp):
    layer, factory, imports = _init_gate(tp=tp, global_tp=16, dp=True, enabled=False)
    assert not layer.use_npu_fused_qkv
    assert not layer.use_legacy_npu_fused_qkv
    assert layer.static_decode_qkv is None
    factory.assert_not_called()
    assert imports == []


@pytest.mark.parametrize("tp,nextn", [(8, False), (1, True), (4, True)])
def test_init_rejects_unsupported_tp_and_nextn(tp, nextn):
    layer, factory, _ = _init_gate(tp=tp, global_tp=tp, dp=False, enabled=True, nextn=nextn)
    assert not layer.use_npu_fused_qkv
    assert layer.static_decode_qkv is None
    factory.assert_not_called()


def test_mirror_consumer_uses_static_query_path_without_full_dynamic_path():
    layer, factory, _ = _init_gate(tp=1, global_tp=16, dp=True, enabled=True, consumer=True)
    assert not layer.use_npu_fused_qkv
    assert not layer.use_legacy_npu_fused_qkv
    factory.assert_called_once_with(1)


def _dispatch_case(doubles, *, tp=1, dp=False, source=False, swa=False, graph=False,
                   mode=_Mode.EXTEND, original_mode=None, m=513):
    Tensor = doubles.Tensor
    qw, kvw = {1: (6144, 512), 2: (3072, 256), 4: (1536, 256)}[tp]
    layer = SimpleNamespace(
        layer_idx=3,
        hidden_size=2048,
        head_dim=256,
        qk_rope_head_dim=64,
        q_size=qw,
        kv_size=kvw,
        q_norm=None,
        k_norm=SimpleNamespace(weight=Tensor((256,)), eps=1e-6),
        rotary_emb=doubles.Rotary(),
        qkv_proj=SimpleNamespace(
            quant_method=doubles.Unquantized(),
            weight=Tensor((qw + 2 * kvw, 2048)), bias=None,
        ),
        qkv_proj_weight=Tensor((qw + 4 * kvw, 2048)),
        qkv_proj_bias=None,
        kv_mirror_layer_idx=3,
        kv_mirror_layers=[7],
        kv_mirror_imitated_layers=[3] if source else [1],
        fused_qkv_attn_tp_size=tp,
        use_legacy_npu_fused_qkv=(tp == 4 and not dp),
        attn=SimpleNamespace(layer_id=3),
    )
    layer.rotary_emb.cos_sin_cache = Tensor((8192, 64), "fp32")
    full, swa_slots, graph_full, graph_swa = [Tensor((m,), "i64") for _ in range(4)]
    batch = SimpleNamespace(
        forward_mode=mode,
        batch_size=1,
        extend_prefix_lens_cpu=[7],
        welm_prefill_graph=object() if graph else None,
        out_cache_loc=full,
        welm_prefill_full_write_locs=graph_full,
        welm_prefill_swa_write_locs=graph_swa,
    )
    if original_mode is not None:
        batch._original_forward_mode = original_mode
    caches = [Tensor((4096, kvw // 256, 256)) for _ in range(2)]
    backend = SimpleNamespace(
        use_welm_static_decode_qkv=True,
        _is_swa_layer=lambda attn: swa,
        forward_metadata=SimpleNamespace(swa_out_cache_loc=swa_slots),
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=Mock(return_value=caches)),
    )
    program = Mock(name="dynamic_program")
    lookup = Mock(name="program_lookup", return_value=program)
    namespace = dict(
        Any=Any, Optional=Optional, Tuple=Tuple, ForwardBatch=object,
        ForwardMode=_Mode,
        torch=SimpleNamespace(Tensor=doubles.BaseTensor, bfloat16="bf16", float32="fp32",
                              int32="i32", int64="i64"),
        torch_npu=SimpleNamespace(get_npu_format=lambda tensor: tensor.npu_format),
        UnquantizedLinearMethod=doubles.Unquantized,
        WelmV4InplaceRotaryEmbedding=doubles.Rotary,
        get_attn_backend=lambda: backend,
        is_dp_attention_enabled=lambda: dp,
        _get_welm_fused_qkv_program=lookup,
    )
    namespace.update(doubles.layout)
    methods = [
        _method(name) for name in (
            "_npu_bf16_fused_qkv_parameters", "_try_npu_fused_qkv", "_try_npu_fused_qkv_mxfp8"
        )
    ]
    _exec_nodes(methods, namespace, _MODEL)
    for method in methods:
        setattr(layer, method.name, MethodType(namespace[method.name], layer))
    return SimpleNamespace(
        layer=layer, batch=batch, backend=backend, caches=caches,
        hidden=Tensor((m, 2048)), positions=Tensor((m,), "i64"),
        selected_slots=(graph_swa if swa else graph_full) if graph else (swa_slots if swa else full),
        lookup=lookup, program=program,
    )


def _run(case):
    return case.layer._try_npu_fused_qkv(case.positions, case.hidden, case.batch)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_inactive_mtp_source_uses_plain_prefill_projection(doubles, tp):
    case = _dispatch_case(doubles, tp=tp, source=True)
    # post_init_after_load_weights established that the configured MTP
    # consumer is absent. Missing derived weights alone must never do this.
    case.layer.kv_mirror_layer_idx = case.layer.layer_idx = 0
    case.layer.attn.layer_id = 0
    case.layer.kv_mirror_layers = [48]
    case.layer.kv_mirror_imitated_layers = [0]
    case.layer._kv_mirror_active_source = False
    del case.layer.qkv_proj_weight
    del case.layer.qkv_proj_bias
    result = _run(case)
    assert result is not None
    assert result[3:] == (None, None)
    assert case.program.call_args.args[1] is case.layer.qkv_proj.weight


@pytest.mark.parametrize("role_known", [False, True])
def test_active_or_uninitialized_source_missing_weights_still_falls_back(doubles, role_known):
    case = _dispatch_case(doubles, source=True)
    if role_known:
        case.layer._kv_mirror_active_source = True
    del case.layer.qkv_proj_weight
    assert _run(case) is None
    case.program.assert_not_called()
    case.lookup.assert_not_called()


def test_active_source_keeps_prefill_mirror_outputs(doubles):
    case = _dispatch_case(doubles, source=True)
    case.layer._kv_mirror_active_source = True
    result = _run(case)
    assert result[3] is not None and result[4] is not None
    assert case.program.call_args.args[1] is case.layer.qkv_proj_weight


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("swa,graph", [(False, False), (True, False), (False, True), (True, True)])
def test_prefill_dispatch_keeps_local_geometry_and_actual_cache_slots(doubles, tp, source, swa, graph):
    case = _dispatch_case(doubles, tp=tp, source=source, swa=swa, graph=graph)
    result = _run(case)
    qw, kvw = {1: (6144, 512), 2: (3072, 256)}[tp]
    n = {1: (7168, 8192), 2: (3584, 4096)}[tp][source]
    assert case.lookup.call_args.args[1:] == (n, 4096, 8192, not graph)
    args = case.program.call_args.args
    assert len(args) == 14
    assert args[1] is (case.layer.qkv_proj_weight if source else case.layer.qkv_proj.weight)
    assert args[3] is case.positions
    assert args[5] is case.selected_slots
    assert result[0].shape == (513, qw)
    assert result[1].shape == result[2].shape == (513, kvw)
    for actual, original in zip(args[11:13], case.caches):
        assert actual.shape == (4096, kvw)
        assert actual.data_ptr() == original.data_ptr()
    if source:
        assert result[3].shape == result[4].shape == (513, kvw)
    else:
        assert result[3:] == (None, None)


@pytest.mark.parametrize("tp,dp", [(1, False), (2, False), (1, True), (2, True), (4, True)])
@pytest.mark.parametrize("m", [3, 8, 32, 64, 128])
def test_static_rejected_new_decode_never_enters_dynamic_kernel(doubles, tp, dp, m):
    # Call the dynamic fallback directly: even a supported static M must not
    # bypass static ABI/graph-owner guards by entering the prefill kernel.
    case = _dispatch_case(doubles, tp=tp, dp=dp, mode=_Mode.DECODE, m=m)
    assert _run(case) is None
    case.lookup.assert_not_called()
    case.program.assert_not_called()
    case.backend.token_to_kv_pool.get_kv_buffer.assert_not_called()
    assert case.hidden.allocations == []


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_dp_synthetic_extend_preserves_decode_rejection(doubles, tp):
    case = _dispatch_case(doubles, tp=tp, dp=True, original_mode=_Mode.DECODE, m=8)
    assert _run(case) is None
    case.lookup.assert_not_called()
    case.program.assert_not_called()


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_dp_prefill_uses_per_row_positions_even_for_local_batch_one(doubles, tp):
    case = _dispatch_case(doubles, tp=tp, dp=True)
    assert _run(case) is not None
    assert case.lookup.call_args.args[-1] is False
    case.program.assert_called_once()


@pytest.mark.parametrize("mode", [_Mode.DECODE, _Mode.TARGET_VERIFY])
def test_original_tp4_decode_and_verify_keep_dynamic_dispatch(doubles, mode):
    case = _dispatch_case(doubles, tp=4, mode=mode, m=8)
    if mode == _Mode.TARGET_VERIFY:
        case.backend.use_welm_static_decode_qkv = False
    result = _run(case)
    assert result[0].shape == (8, 1536)
    assert result[1].shape == result[2].shape == (8, 256)
    assert case.lookup.call_args.args[-1] is False
    case.program.assert_called_once()


@pytest.mark.parametrize("tp", [1, 2])
def test_prefill_preserves_backend_server_and_pool_guard(doubles, tp):
    case = _dispatch_case(doubles, tp=tp)
    case.backend.use_welm_static_decode_qkv = False
    assert _run(case) is None
    case.lookup.assert_not_called()
    case.program.assert_not_called()


@pytest.mark.parametrize(
    "source,change",
    [
        (False, lambda c: setattr(c.layer.qkv_proj, "bias", object())),
        (True, lambda c: setattr(c.layer, "qkv_proj_bias", object())),
        (False, lambda c: setattr(c.layer.qkv_proj, "quant_method", object())),
        (False, lambda c: setattr(c.layer.qkv_proj.weight, "dtype", "fp8")),
        (False, lambda c: setattr(c.layer.qkv_proj.weight, "shape", (6144, 2048))),
        (True, lambda c: setattr(c.layer.qkv_proj_weight, "_contiguous", False)),
        (False, lambda c: setattr(c.layer.qkv_proj.weight, "npu_format", 29)),
    ],
    ids=["plain-bias", "source-bias", "quantization", "weight-dtype", "weight-shape",
         "source-strides", "NZ-weight"],
)
def test_invalid_projection_falls_back_before_program_selection(
    doubles, source, change
):
    case = _dispatch_case(doubles, source=source)
    change(case)
    for _ in range(2):
        assert _run(case) is None
    case.lookup.assert_not_called()
    case.program.assert_not_called()
    assert case.hidden.allocations == []


@pytest.mark.parametrize("tp,dp", [(1, False), (2, False), (1, True), (2, True), (4, True)])
def test_mxfp8_method_refuses_nonlegacy_gate_before_touching_inputs(doubles, tp, dp):
    case = _dispatch_case(doubles, tp=tp, dp=dp)
    assert case.layer._try_npu_fused_qkv_mxfp8(object(), object(), object()) is None
    case.lookup.assert_not_called()


def test_dynamic_program_miss_returns_before_metadata_cast_and_allocation(doubles):
    case = _dispatch_case(doubles)
    case.lookup.return_value = None
    assert _run(case) is None
    case.program.assert_not_called()
    assert case.hidden.allocations == []
