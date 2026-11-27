"""CPU contracts for static Decode QKV admission and model ownership.

These tests execute the actual Python binding and model dispatch statements
with tensor metadata doubles. They never import torch, torch_npu, CANNBotDSL,
or the SGLang package. Numerical kernel coverage stays in the external runner.
"""

import ast
import builtins
import math
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[5]
_BINDING = _ROOT / "python/sglang/srt/layers/welmv4_static_decode_qkv.py"
_LAYOUT = _ROOT / "python/sglang/srt/layers/welmv4_qkv_layout.py"
_MODEL = _ROOT / "python/sglang/srt/models/welmv4.py"
_RUNNER = _ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
_KERNEL_MODULE = "sglang.srt.layers.welmv4_static_decode_qkv_kernel"
_DEVICE = SimpleNamespace(type="npu", index=0)


class _Tensor:
    """Only shape/view/storage behavior; copying APIs are deliberately absent."""

    def __init__(
        self, shape, dtype="bf16", *, storage=None, contiguous=True, npu_format=2,
        storage_offset=0,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _DEVICE
        self.storage = object() if storage is None else storage
        self._contiguous = contiguous
        self.npu_format = npu_format
        self._storage_offset = storage_offset
        self.allocations = []

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def data(self):
        return self

    def is_contiguous(self):
        return self._contiguous

    def stride(self):
        strides = []
        running = 1 if self._contiguous else 2
        for size in reversed(self.shape):
            strides.append(running)
            running *= size
        return tuple(reversed(strides))

    def data_ptr(self):
        return id(self.storage) + 2 * self._storage_offset

    def storage_offset(self):
        return self._storage_offset

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        size = math.prod(self.shape)
        if -1 in shape:
            known = math.prod(dim for dim in shape if dim != -1)
            assert shape.count(-1) == 1 and size % known == 0
            shape = tuple(size // known if dim == -1 else dim for dim in shape)
        assert math.prod(shape) == size
        assert self._contiguous
        return _Tensor(
            shape, self.dtype, storage=self.storage, npu_format=self.npu_format,
            storage_offset=self._storage_offset,
        )

    def contiguous(self):
        assert self._contiguous, "Unexpected copy in the isolated contract"
        return self

    def new_empty(self, shape):
        result = _Tensor(shape, self.dtype)
        self.allocations.append(result)
        return result


class _UnquantizedLinearMethod:
    pass


class _RotaryEmbedding:
    is_neox_style = True

    def __init__(self):
        self.cos_sin_cache = _Tensor((256, 64), "fp32")
        self.single_calls = []
        self.pair_calls = []

    def forward_npu_single(self, positions, tensor):
        self.single_calls.append(tensor)
        return tensor

    def __call__(self, positions, query, key, **kwargs):
        self.pair_calls.append((query, key))
        return query, key


_DECODE = SimpleNamespace(
    is_extend_without_speculative=lambda: False,
    is_extend_or_draft_extend_or_mixed=lambda **kwargs: False,
)
_FORWARD_MODE = SimpleNamespace(DECODE=_DECODE)


def _load_layout():
    """Execute the real layout guards without importing NPU packages."""
    tree = ast.parse(_LAYOUT.read_text(), filename=str(_LAYOUT))
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    namespace = dict(
        __name__="_welm_qkv_layout_contract",
        torch_npu=SimpleNamespace(get_npu_format=lambda tensor: tensor.npu_format),
    )
    exec(compile(tree, str(_LAYOUT), "exec", dont_inherit=True), namespace)
    return {
        name: namespace[name]
        for name in ("qkv_tensor_is_nd", "qkv_cache_view")
    }


@pytest.fixture
def binding(monkeypatch):
    """Keep real binding definitions, replacing only their external imports."""
    tree = ast.parse(_BINDING.read_text(), filename=str(_BINDING))
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    name = "_welm_static_qkv_contract_" + uuid.uuid4().hex
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    program = Mock(name="fused_program")
    compiler = Mock(name="compile_qkv", return_value=program)
    imports = []

    def isolated_import(name, *args, **kwargs):
        imports.append(name)
        if name == _KERNEL_MODULE:
            return SimpleNamespace(compile_qkv=compiler)
        raise AssertionError(f"Unexpected runtime import: {name}")

    npu = SimpleNamespace(
        is_current_stream_capturing=Mock(return_value=False),
        get_device_name=Mock(side_effect=AssertionError("Unexpected device query")),
        current_stream=Mock(side_effect=AssertionError("Unexpected stream query")),
    )
    module.__dict__.update(
        __builtins__=dict(vars(builtins), __import__=isolated_import),
        dataclass=dataclass,
        torch=SimpleNamespace(
            Tensor=_Tensor, bfloat16="bf16", float32="fp32", int64="i64", npu=npu
        ),
        torch_npu=SimpleNamespace(get_npu_format=lambda tensor: tensor.npu_format),
        UnquantizedLinearMethod=_UnquantizedLinearMethod,
        WelmV4InplaceRotaryEmbedding=_RotaryEmbedding,
        ForwardMode=_FORWARD_MODE,
    )
    module.__dict__.update(_load_layout())
    exec(compile(tree, str(_BINDING), "exec", dont_inherit=True), module.__dict__)
    original_platform = module._platform
    module._platform = Mock(return_value=("Ascend950DT", "arch35", 24, "0.8.0"))
    return SimpleNamespace(
        module=module,
        compiler=compiler,
        program=program,
        imports=imports,
        npu=npu,
        original_platform=original_platform,
    )


def _case(binding, *, tp=1, kind="plain", m=4, capacity=128, swa=False):
    qw, kvw, n = binding.module.geometry(tp, kind)
    layer = SimpleNamespace(
        is_nextn=False,
        layer_idx=3,
        head_dim=256,
        qk_rope_head_dim=64,
        q_norm=None,
        k_norm=SimpleNamespace(weight=_Tensor((256,)), eps=1e-6),
        rotary_emb=_RotaryEmbedding(),
        q_size=qw,
        kv_size=kvw,
        kv_mirror_layer_idx=3,
        kv_mirror_layers=[3] if kind == "query" else [7],
        kv_mirror_imitated_layers=[3] if kind == "source" else [1],
        qkv_proj=SimpleNamespace(
            quant_method=_UnquantizedLinearMethod(),
            weight=_Tensor((n, 2048)),
            bias=None,
        ),
        qkv_proj_weight=_Tensor((n, 2048)),
        qkv_proj_bias=None,
        attn=Mock(layer_id=3, return_value=_Tensor((m, qw))),
        need_clear_kv_cache=True,
        gated_self_attention_headwise=False,
    )
    batch = SimpleNamespace(
        forward_mode=_DECODE,
        spec_info=None,
        welm_prefill_graph=None,
        welm_prefill_graph_phase=None,
        batch_size=m,
        enable_kv_mirror=False,
        welm_static_decode_graph_owner=True,
    )
    caches = (_Tensor((capacity, kvw // 256, 256)), _Tensor((capacity, kvw // 256, 256)))
    full_slots = _Tensor((m,), "i64")
    swa_slots = _Tensor((m,), "i64")
    backend = SimpleNamespace(
        use_welm_static_decode_qkv=True,
        graph_mode=True,
        _is_swa_layer=lambda attn: swa,
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=Mock(return_value=caches)),
        forward_metadata=SimpleNamespace(
            welm_static_decode_full_slots=full_slots,
            welm_static_decode_swa_slots=swa_slots,
        ),
    )
    return SimpleNamespace(
        layer=layer,
        hidden=_Tensor((m, 2048)),
        positions=_Tensor((m,), "i64"),
        batch=batch,
        backend=backend,
        caches=caches,
        full_slots=full_slots,
        swa_slots=swa_slots,
        operator=binding.module.StaticDecodeQKV(tp),
    )


def _run(case):
    return case.operator.try_run(
        case.layer, case.positions, case.hidden, case.batch, case.backend
    )


def _post_init_source_roles(attentions, pairs, *, staged_ids=(), is_nextn=False):
    """Exercise the real load-time role selection without loading weights."""
    tree = ast.parse(_MODEL.read_text(), filename=str(_MODEL))
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WeLMV4MoeForCausalLM"
    )
    method = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "post_init_after_load_weights"
    )
    decoder_layers = [SimpleNamespace(self_attn=attention) for attention in attentions]
    staging = {
        str(layer_id): SimpleNamespace(self_attn=SimpleNamespace(
            kv_mirror_layer_idx=layer_id, loaded_qkv_parts={"q", "k", "v"}
        ))
        for layer_id in staged_ids
    }
    owner = SimpleNamespace(model=SimpleNamespace(
        config=SimpleNamespace(
            kv_mirror_layers=[consumer for consumer, _ in pairs],
            kv_mirror_imitated_layers=[source for _, source in pairs],
        ),
        layers=decoder_layers,
        decoder_layers=decoder_layers,
        mtp_mirror_staging_layers=staging,
    ))
    manager = SimpleNamespace(
        prepared_cross_model_pairs=set(),
        decoder_layer={int(layer_id): layer for layer_id, layer in staging.items()},
    )

    def prepare(consumers, sources, *, is_nextn):
        # Model LayerManager's existing record of successfully absorbed MTP
        # projections; all role filtering above still runs the production code.
        manager.prepared_cross_model_pairs.update(
            (source, consumer) for consumer, source in zip(consumers, sources)
            if consumer in staged_ids
        )

    manager.post_init = Mock(side_effect=prepare)
    namespace = dict(
        _is_npu=False,
        LayerManager=manager,
        torch=SimpleNamespace(get_device_module=lambda: SimpleNamespace(empty_cache=Mock())),
        nn=SimpleNamespace(ModuleDict=dict),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(_MODEL),
                 "exec", dont_inherit=True), namespace)

    def initialize():
        namespace[method.name](owner, is_nextn=is_nextn)

    initialize()
    return owner, manager, initialize


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_target_only_mtp_source_uses_plain_static_projection(binding, tp):
    case = _case(binding, tp=tp)
    case.layer.kv_mirror_layer_idx = case.layer.layer_idx = 0
    case.layer.attn.layer_id = 0
    case.layer.kv_mirror_layers = [48]
    case.layer.kv_mirror_imitated_layers = [0]
    del case.layer.qkv_proj_weight
    del case.layer.qkv_proj_bias
    _, manager, _ = _post_init_source_roles([case.layer], [(48, 0)])
    manager.post_init.assert_called_once_with([], [], is_nextn=False)
    assert case.layer._kv_mirror_active_source is False
    result = _run(case)
    assert result is not None
    assert result.mirror_k is result.mirror_v is None
    assert binding.program.call_args.args[1] is case.layer.qkv_proj.weight
    assert binding.compiler.call_args.kwargs["has_mirror"] is False


@pytest.mark.parametrize("prepared", [False, True])
def test_active_source_still_requires_derived_weights(binding, prepared):
    case = _case(binding, kind="source")
    consumer = SimpleNamespace(kv_mirror_layer_idx=7)
    _, manager, _ = _post_init_source_roles([case.layer, consumer], [(7, 3)])
    manager.post_init.assert_called_once_with([7], [3], is_nextn=False)
    assert case.layer._kv_mirror_active_source is True
    if not prepared:
        del case.layer.qkv_proj_weight
        assert _run(case) is None
        binding.program.assert_not_called()
        binding.compiler.assert_not_called()
    else:
        result = _run(case)
        assert result.mirror_k is not None
        assert binding.program.call_args.args[1] is case.layer.qkv_proj_weight
        assert binding.compiler.call_args.kwargs["has_mirror"] is True


def test_mtp_staging_source_role_survives_staging_release_and_reinitialization():
    source = SimpleNamespace(kv_mirror_layer_idx=0)
    owner, manager, initialize = _post_init_source_roles(
        [source], [(48, 0)], staged_ids=(48,)
    )
    assert source._kv_mirror_active_source is True
    assert owner.model.mtp_mirror_staging_layers == {}
    assert manager.prepared_cross_model_pairs == {(0, 48)}
    initialize()
    assert source._kv_mirror_active_source is True
    assert manager.post_init.call_args.args == ([], [])


def test_nextn_consumer_preserves_pair_alignment():
    consumer = SimpleNamespace(kv_mirror_layer_idx=48)
    _, manager, _ = _post_init_source_roles(
        [consumer], [(33, 1), (48, 0)], is_nextn=True
    )
    manager.post_init.assert_called_once_with([48], [0], is_nextn=True)
    assert consumer._kv_mirror_active_source is False


@pytest.mark.parametrize(
    "tp,kind,expected",
    [
        (1, "plain", (6144, 512, 7168)),
        (1, "source", (6144, 512, 8192)),
        (1, "query", (6144, 512, 6144)),
        (2, "plain", (3072, 256, 3584)),
        (2, "source", (3072, 256, 4096)),
        (2, "query", (3072, 256, 3072)),
        (4, "plain", (1536, 256, 2048)),
        (4, "source", (1536, 256, 2560)),
        (4, "query", (1536, 256, 1536)),
    ],
)
def test_local_geometry(binding, tp, kind, expected):
    assert binding.module.geometry(tp, kind) == expected


@pytest.mark.parametrize("tp,kind", [(0, "plain"), (3, "plain"), (8, "query"), (1, "other")])
def test_rejects_unknown_geometry(binding, tp, kind):
    with pytest.raises(ValueError):
        binding.module.geometry(tp, kind)


@pytest.mark.parametrize("swa", [False, True])
def test_tp1_keeps_whole_token_cache_rows_and_physical_slots(binding, swa):
    case = _case(binding, swa=swa)
    result = _run(case)
    args = binding.program.call_args.args
    assert result.q.shape == (4, 6144)
    assert len(args) == 14
    assert args[5] is (case.swa_slots if swa else case.full_slots)
    for actual, original in zip(args[11:13], case.caches):
        assert actual.shape == (128, 512)
        assert actual.data_ptr() == original.data_ptr()
    case.backend.token_to_kv_pool.get_kv_buffer.assert_called_once_with(3)
    assert binding.compiler.call_args.args == (128, 256)
    assert binding.compiler.call_args.kwargs["max_blocks"] == 24


@pytest.mark.parametrize(
    "change",
    [
        lambda c: setattr(c.hidden, "shape", (3, 2048)),
        lambda c: setattr(c.hidden, "shape", (128, 2048)),
        lambda c: setattr(c.hidden, "shape", (4, 1024)),
        lambda c: setattr(c.layer.qkv_proj, "quant_method", object()),
        lambda c: setattr(c.layer.qkv_proj, "bias", object()),
        lambda c: setattr(c.layer.qkv_proj.weight, "dtype", "fp8"),
        lambda c: setattr(c.layer.qkv_proj.weight, "npu_format", 29),
        lambda c: setattr(c.caches[0], "_contiguous", False),
        lambda c: setattr(c.caches[0], "shape", (128, 1, 512)),
        lambda c: setattr(c.full_slots, "dtype", "i32"),
        lambda c: setattr(c.layer, "q_norm", object()),
        lambda c: setattr(c.layer, "is_nextn", True),
        lambda c: setattr(c.batch, "spec_info", object()),
        lambda c: setattr(c.batch, "forward_mode", object()),
        lambda c: setattr(c.backend, "use_welm_static_decode_qkv", False),
        lambda c: setattr(c.batch, "welm_static_decode_graph_owner", False),
    ],
    ids=[
        "M3", "M128", "hidden-width", "quantization", "bias", "weight-dtype",
        "NZ-weight", "strided-cache", "wrong-head-layout", "int32-slots",
        "Q-norm", "NextN", "speculative", "prefill", "unsupported-pool",
        "graph-without-owner",
    ],
)
def test_rejected_inputs_never_compile_or_dispatch(binding, change):
    case = _case(binding)
    change(case)
    # Warmup and capture retries must consistently preserve the native fallback.
    for _ in range(2):
        assert _run(case) is None
    binding.compiler.assert_not_called()
    binding.program.assert_not_called()
    binding.module._platform.assert_not_called()
    assert not case.hidden.allocations


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("m", [3, 128])
def test_unsupported_rows_reject_even_with_matching_tensor_shapes(binding, tp, m):
    case = _case(binding, tp=tp, m=m)
    assert _run(case) is None
    binding.module._platform.assert_not_called()
    binding.compiler.assert_not_called()
    binding.program.assert_not_called()
    assert not case.hidden.allocations


@pytest.mark.parametrize("m", [4, 32, 64])
def test_capture_miss_does_not_import_compile_or_allocate(binding, m):
    case = _case(binding, m=m)
    binding.npu.is_current_stream_capturing.return_value = True
    assert _run(case) is None
    assert binding.imports == []
    binding.compiler.assert_not_called()
    binding.program.assert_not_called()
    assert not case.hidden.allocations


def test_cold_capture_does_not_probe_platform(binding):
    case = _case(binding)
    binding.module._platform = binding.original_platform
    binding.npu.is_current_stream_capturing.return_value = True
    assert _run(case) is None
    binding.npu.get_device_name.assert_not_called()
    binding.npu.current_stream.assert_not_called()
    assert binding.imports == []


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", ["plain", "source", "query"])
@pytest.mark.parametrize("m", [4, 32, 64])
@pytest.mark.parametrize("swa", [False, True], ids=["full", "swa"])
def test_capture_reuses_prepared_program_and_batch_outputs(binding, tp, kind, m, swa):
    case = _case(binding, tp=tp, kind=kind, m=m, swa=swa)
    first = _run(case)
    first_args = binding.program.call_args.args
    qw, kvw, _ = binding.module.geometry(tp, kind)
    assert first.q.shape == (m, qw)
    assert first.k.shape == first.v.shape == (1 if kind == "query" else m, kvw)
    assert first.query_only is (kind == "query")
    if kind == "source":
        assert first.mirror_k.shape == first.mirror_v.shape == (m, kvw)
    else:
        assert first.mirror_k is first.mirror_v is None
    assert first_args[0] is case.hidden
    assert first_args[3] is case.positions
    assert first_args[5] is (case.swa_slots if swa else case.full_slots)
    assert [t.shape for t in first_args[9:11]] == [
        (m if kind == "source" else 1, kvw)
    ] * 2
    for actual, original in zip(first_args[11:13], case.caches):
        assert actual.shape == (128, kvw)
        assert actual.data_ptr() == original.data_ptr()
    assert binding.compiler.call_args.kwargs == {
        "tp_size": tp,
        "has_mirror": kind == "source",
        "q_only": kind == "query",
        "static_m": m,
        "max_blocks": 24,
    }
    binding.npu.is_current_stream_capturing.return_value = True
    second = _run(case)
    binding.compiler.assert_called_once()
    assert first.q is second.q
    assert binding.program.call_count == 2
    assert binding.program.call_args.args[6:11] == first_args[6:11]
    assert len(case.hidden.allocations) == 5


@pytest.mark.parametrize("m", [4, 32, 64])
def test_capture_with_program_but_without_owned_outputs_falls_back(binding, m):
    case = _case(binding, m=m)
    _run(case)
    del case.batch.welm_static_decode_qkv_outputs
    binding.npu.is_current_stream_capturing.return_value = True
    assert _run(case) is None
    binding.compiler.assert_called_once()
    binding.program.assert_called_once()
    assert len(case.hidden.allocations) == 5


@pytest.mark.parametrize("m", [32, 64])
def test_large_capture_without_graph_owner_keeps_native_fallback(binding, m):
    case = _case(binding, m=m)
    case.batch.welm_static_decode_graph_owner = False
    binding.npu.is_current_stream_capturing.return_value = True
    assert _run(case) is None
    binding.module._platform.assert_not_called()
    binding.compiler.assert_not_called()
    binding.program.assert_not_called()
    assert not case.hidden.allocations


def test_m32_and_m64_keep_separate_programs_and_batch_owned_outputs(binding):
    cases = [_case(binding, m=m) for m in (32, 64, 32)]
    results = [_run(case) for case in cases]
    assert [call.kwargs["static_m"] for call in binding.compiler.call_args_list] == [
        32, 64
    ]
    assert len({id(result.q) for result in results}) == 3
    owners = [case.batch.welm_static_decode_qkv_outputs for case in cases]
    assert len({id(owner) for owner in owners}) == 3
    binding.npu.is_current_stream_capturing.return_value = True
    for case, result in zip(cases, results):
        assert _run(case).q is result.q
        assert len(case.hidden.allocations) == 5
    assert binding.compiler.call_count == 2
    assert binding.program.call_count == 6


def test_program_key_reuses_geometry_but_distinguishes_cache_capacity(binding):
    first = _case(binding)
    second = _case(binding)
    first_result = _run(first)
    second_result = _run(second)
    binding.compiler.assert_called_once()
    assert first_result.q is not second_result.q
    _run(_case(binding, capacity=256))
    assert binding.compiler.call_count == 2
    assert binding.compiler.call_args.args[0] == 256


def test_program_key_includes_verified_launch_budget(binding):
    case = _case(binding)
    _run(case)
    binding.module._platform.return_value = ("Ascend950DT", "arch35", 16, "0.8.0")
    _run(case)
    assert binding.compiler.call_count == 2
    assert binding.compiler.call_args.kwargs["max_blocks"] == 16


def test_dispatch_failure_propagates_without_retrying(binding):
    case = _case(binding)
    binding.program.side_effect = RuntimeError("dispatch failed")
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _run(case)
    binding.program.assert_called_once()


def _model_dispatch(case, result, raw_k, raw_v):
    """Execute the model's ownership/postprocessing/attention block unchanged."""
    tree = ast.parse(_MODEL.read_text(), filename=str(_MODEL))
    attention = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen2MoeAttention"
    )
    forward = next(
        node for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )

    start = next(i for i, node in enumerate(forward.body) if assigns(node, "prepared_qkv"))
    end = next(
        i for i in range(start, len(forward.body))
        if isinstance(forward.body[i], ast.If)
        and any(assigns(node, "attn_output") for node in ast.walk(forward.body[i]))
    )
    mirror = SimpleNamespace(
        get_kv_activation=Mock(return_value=(raw_k, raw_v)),
        set_kv_activation=Mock(),
    )
    norm = Mock(side_effect=lambda tensor, weight, eps: tensor)
    namespace = dict(
        self=case.layer,
        static_qkv=result,
        fused_qkv=None,
        frozen_mtp_decode=False,
        external_mirror=None,
        forward_batch=case.batch,
        hidden_states=case.hidden,
        positions=case.positions,
        reuse_prefill_mxfp8_input=False,
        _is_npu=False,
        KVMirrorManager=mirror,
        LayerManager=SimpleNamespace(num_target_layers=48, num_nextn_predict_layer_idx=()),
        mmq_style_k_rms_norm=norm,
    )
    case.layer.attn_sink = None
    statements = ast.Module(body=forward.body[start:end + 1], type_ignores=[])
    exec(compile(statements, str(_MODEL), "exec", dont_inherit=True), namespace)
    return mirror, norm


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("m", [4, 32, 64])
@pytest.mark.parametrize("kind", ["plain", "source", "query"])
def test_model_respects_raw_mirror_and_postprocessing_ownership(binding, tp, m, kind):
    case = _case(binding, tp=tp, kind=kind, m=m)
    result = _run(case)
    kvw = case.layer.kv_size
    raw_k, raw_v = _Tensor((m, kvw)), _Tensor((m, kvw))
    mirror, norm = _model_dispatch(case, result, raw_k, raw_v)
    kwargs = binding.compiler.call_args.kwargs
    assert kwargs["has_mirror"] is (kind == "source")
    assert kwargs["q_only"] is (kind == "query")
    attn_args = case.layer.attn.call_args.args
    assert attn_args[0].data_ptr() == result.q.data_ptr()
    assert case.layer.attn.call_args.kwargs["save_kv_cache"] is (kind == "query")
    assert case.layer.rotary_emb.pair_calls == []
    if kind == "query":
        mirror.get_kv_activation.assert_called_once_with(1, clear=True)
        mirror.set_kv_activation.assert_not_called()
        norm.assert_called_once()
        assert len(case.layer.rotary_emb.single_calls) == 1
        assert case.layer.rotary_emb.single_calls[0].data_ptr() == raw_k.data_ptr()
        assert attn_args[1].data_ptr() == raw_k.data_ptr()
        assert attn_args[2] is raw_v
        assert result.k.shape == (1, kvw)
        with pytest.raises(AssertionError):
            result.full_outputs()
    else:
        mirror.get_kv_activation.assert_not_called()
        norm.assert_not_called()
        assert case.layer.rotary_emb.single_calls == []
        assert attn_args[1] is result.k
        assert attn_args[2] is result.v
        if kind == "source":
            mirror.set_kv_activation.assert_called_once_with(
                3, (result.mirror_k, result.mirror_v)
            )
            assert result.mirror_k is not result.k
            assert result.mirror_v is not result.v
        else:
            mirror.set_kv_activation.assert_not_called()


@pytest.mark.parametrize("static_enabled", [False, True])
@pytest.mark.parametrize("with_dp_layout", [False, True])
def test_graph_retains_whole_batch_for_static_outputs(static_enabled, with_dp_layout):
    tree = ast.parse(_RUNNER.read_text(), filename=str(_RUNNER))
    retain = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_retain_welm_capture_scratch"
    )
    namespace = dict(
        ForwardBatch=object,
        get_attn_backend=lambda: SimpleNamespace(use_welm_static_decode_qkv=static_enabled),
    )
    exec(
        compile(ast.Module(body=[retain], type_ignores=[]), str(_RUNNER), "exec", dont_inherit=True),
        namespace,
    )
    layout = object() if with_dp_layout else None
    batch = SimpleNamespace(
        welm_dp_moe_row_layout=layout,
        welm_static_decode_graph_owner=False,
        welm_static_decode_qkv_outputs={"layer-3": object()},
    )
    runner = SimpleNamespace(
        welm_capture_scratch={},
        welm_local_decode_rows=True,
        model_runner=SimpleNamespace(welm_decode_rs_ag_enabled=False),
    )
    namespace[retain.name](runner, "bucket-4", batch)
    retained = runner.welm_capture_scratch["bucket-4"]
    assert retained is (batch if static_enabled or not with_dp_layout else layout)
    assert batch.welm_static_decode_graph_owner is static_enabled
