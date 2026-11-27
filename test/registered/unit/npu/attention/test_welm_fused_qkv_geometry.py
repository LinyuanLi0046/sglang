"""CPU ABI contracts for the dynamic BF16 QKV geometry expansion.

The geometry module uses only stdlib. Compiler/binding statements and the
head-routing epilogue are taken from production AST and run with metadata
doubles, without importing SGLang, torch, CANNBotDSL or device libraries.
These checks do not validate device math, compilation or graph execution.
"""

import ast
import builtins
import copy
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[5]
_LAYERS = _ROOT / "python/sglang/srt/layers"
_SOURCE = _LAYERS / "fused_qkv_proj_norm_rope_cache.py"
_CASES = (
    # TP, projection width, local Q heads, local KV heads, mirror source.
    (1, 7168, 24, 2, False),
    (1, 8192, 24, 2, True),
    (2, 3584, 12, 1, False),
    (2, 4096, 12, 1, True),
    (4, 2048, 6, 1, False),
    (4, 2560, 6, 1, True),
)
_PARAMS = "tp,width,q_heads,kv_heads,mirror"


@pytest.fixture
def geometry(monkeypatch):
    name = "_welm_dynamic_qkv_geometry_contract"
    spec = importlib.util.spec_from_file_location(
        name, _LAYERS / "welm_fused_qkv_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.FusedQkvGeometry


@dataclass(eq=False)
class _Dim:
    name: str
    min: int


@dataclass
class _TensorSpec:
    shape: tuple
    dtype: str


class _Tensor:
    """Metadata-only tensor; view preserves storage and copying is unavailable."""

    def __init__(self, shape, dtype="bf16", *, storage=None, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "metadata-device"
        self.storage = object() if storage is None else storage
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def view(self, *shape):
        assert self._contiguous
        if -1 in shape:
            assert shape.count(-1) == 1
            extent = math.prod(dim for dim in shape if dim != -1)
            assert math.prod(self.shape) % extent == 0
            shape = tuple(
                math.prod(self.shape) // extent if dim == -1 else dim for dim in shape
            )
        assert math.prod(shape) == math.prod(self.shape)
        return _Tensor(shape, self.dtype, storage=self.storage)

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        return _Tensor(self.shape, dtype)


def _allocate(*shape, dtype, device):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    result = _Tensor(shape, dtype)
    assert device == result.device
    return result


def _reject_runtime_import(name, *args, **kwargs):
    raise AssertionError(f"Unexpected runtime import in metadata contract: {name}")


@pytest.fixture
def contract(geometry):
    tree = ast.parse(_SOURCE.read_text(), filename=str(_SOURCE))
    constants = {"HIDDEN", "HEAD_DIM", "ROPE_DIM", "PAIR_M", "BASE_N", "AVG"}
    functions = {
        "compile_aot",
        "compile_aot_rank_chunk",
        "fused_qkv_proj_norm_rope_cache",
    }
    body = []
    kernel_metadata = None
    epilogue = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            body.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(copy.deepcopy(node))
        elif (
            isinstance(node, ast.ClassDef)
            and node.name == "FusedQkvProjNormRopeCache"
        ):
            host_class = copy.deepcopy(node)
            host_class.decorator_list = []
            for method in host_class.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
            body.append(host_class)
        elif (
            isinstance(node, ast.ClassDef)
            and node.name == "FusedQkvProjNormRopeCacheKernel"
        ):
            kernel_init = next(
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name == "__init__"
            )
            kernel_metadata = copy.deepcopy(node)
            kernel_metadata.name = "KernelMetadata"
            kernel_metadata.decorator_list = []
            kernel_metadata.body = [copy.deepcopy(kernel_init)]
            run = next(
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name == "_run"
            )
            # Select the full production if/elif routing chain, not a rewrite.
            epilogue = next(
                branch
                for branch in ast.walk(run)
                if isinstance(branch, ast.If)
                and isinstance(branch.test, ast.Compare)
                and isinstance(branch.test.left, ast.Name)
                and branch.test.left.id == "u"
                and isinstance(branch.test.comparators[0], ast.Attribute)
                and branch.test.comparators[0].attr == "num_q_heads"
            )
    assert kernel_metadata is not None and epilogue is not None
    # The kernel's real constructor establishes the epilogue's head boundaries.
    body.append(kernel_metadata)
    compiler = Mock(
        side_effect=lambda entry, *specs: SimpleNamespace(entry=entry, specs=specs)
    )
    namespace = dict(
        __name__="_welm_dynamic_qkv_source_contract",
        __builtins__=dict(vars(builtins), __import__=_reject_runtime_import),
        FusedQkvGeometry=geometry,
        Dim=_Dim,
        TensorSpec=_TensorSpec,
        BF16="bf16",
        F32="fp32",
        dtypes=SimpleNamespace(int64="i64", float32="fp32"),
        cannbotdsl=SimpleNamespace(compile=compiler),
        torch=SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            int32="i32",
            int64="i64",
            empty=_allocate,
            zeros=_allocate,
        ),
    )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(_SOURCE), "exec", dont_inherit=True), namespace)
    epilogue = ast.Module(body=[copy.deepcopy(epilogue)], type_ignores=[])
    return SimpleNamespace(
        namespace=namespace,
        compiler=compiler,
        epilogue=compile(epilogue, str(_SOURCE), "exec", dont_inherit=True),
    )


@pytest.mark.parametrize(_PARAMS, _CASES)
def test_geometry_matches_rank_local_projection(
    geometry, tp, width, q_heads, kv_heads, mirror
):
    result = geometry.from_width(width)
    assert result.num_q_heads == q_heads
    assert result.num_kv_heads == kv_heads
    assert result.has_mirror is mirror
    assert result.head_dim == 256
    assert result.q_width == 6144 // tp
    assert result.kv_width == (512 if tp == 1 else 256)
    assert result.qkv_width == width
    assert result.units == {
        "k": q_heads,
        "v": q_heads + kv_heads,
        "mirror_k": q_heads + 2 * kv_heads,
        "mirror_v": q_heads + 3 * kv_heads,
    }


@pytest.mark.parametrize("width", [0, -1, 1536, 2304, 3072, 6144, 7680, 8193])
def test_projection_width_rejection(geometry, contract, width):
    with pytest.raises(ValueError):
        geometry.from_width(width)
    for name in ("compile_aot", "compile_aot_rank_chunk"):
        with pytest.raises(ValueError):
            contract.namespace[name](width, 192, 4096)
    contract.compiler.assert_not_called()


@pytest.mark.parametrize(
    "q_heads,kv_heads,dim",
    [(24, 1, 256), (12, 2, 256), (6, 2, 256), (6, 1, 128)],
)
def test_unsupported_head_geometry_is_rejected(geometry, q_heads, kv_heads, dim):
    with pytest.raises(ValueError):
        geometry(num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=dim)


@pytest.mark.parametrize(_PARAMS, _CASES)
@pytest.mark.parametrize("rank_chunk", [False, True])
@pytest.mark.parametrize("positions_contiguous", [False, True])
def test_compiler_tensor_specs_preserve_token_and_head_axes(
    contract, tp, width, q_heads, kv_heads, mirror, rank_chunk, positions_contiguous
):
    name = "compile_aot_rank_chunk" if rank_chunk else "compile_aot"
    compiled = contract.namespace[name](
        width, 192, 4096, positions_contiguous=positions_contiguous
    )
    specs = compiled.specs
    assert len(specs) == (17 if rank_chunk else 14)
    input_m = specs[0].shape[0]
    full_m = specs[3].shape[0]
    assert input_m.name == ("Mchunk" if rank_chunk else "M")
    assert full_m.name == ("Mfull" if rank_chunk else "M")
    assert input_m.min == full_m.min == 1
    assert (input_m is full_m) is (not rank_chunk)
    qw, kvw = q_heads * 256, kv_heads * 256
    expected = (
        ((input_m, 2048), "bf16"),
        ((width, 2048), "bf16"),
        ((1, 256), "bf16"),
        ((full_m,), "i64"),
        ((4096, 64), "fp32"),
        ((full_m,), "i64"),
        ((full_m, qw), "bf16"),
        ((full_m, kvw), "bf16"),
        ((full_m, kvw), "bf16"),
        (((full_m if mirror else 1), kvw), "bf16"),
        (((full_m if mirror else 1), kvw), "bf16"),
        ((192, kvw), "bf16"),
        ((192, kvw), "bf16"),
    )
    assert [(spec.shape, spec.dtype) for spec in specs[:13]] == list(expected)
    assert specs[13:] == (
        ("fp32", "i64", "i64", "i64") if rank_chunk else ("fp32",)
    )
    op = compiled.entry.__self__
    assert compiled.entry.__name__ == ("run_rank_chunk" if rank_chunk else "run")
    assert (op._num_q_heads, op._num_kv_heads) == (q_heads, kv_heads)
    assert op._has_mirror is mirror
    assert op._positions_contiguous is positions_contiguous
    assert op._return_v is rank_chunk


def test_compile_aot_can_materialize_dense_v(contract):
    compiled = contract.namespace["compile_aot"](7168, 193, 4097, return_v=True)
    assert compiled.entry.__self__._return_v is True
    assert compiled.specs[8].shape[1] == 512
    assert compiled.specs[11].shape == (193, 512)
    assert compiled.specs[4].shape == (4097, 64)


@dataclass
class _Tile:
    target: str
    shape: tuple = ()
    coords: tuple = ()

    def __getitem__(self, indices):
        rows, columns = indices
        assert rows.start is None and rows.step is None
        assert columns.start is None and columns.step is None
        assert 0 <= rows.stop <= 128 and columns.stop == 256
        return _Tile(self.target, (rows.stop, columns.stop), self.coords)

    def partition(self, *, dim, part_id):
        assert (dim, part_id) == (0, 0)
        return self


@pytest.mark.parametrize(_PARAMS, _CASES)
@pytest.mark.parametrize("rows", [0, 1, 3, 16, 32, 64, 127, 128])
def test_kernel_head_routing_keeps_each_kv_head_inside_its_token(
    contract, tp, width, q_heads, kv_heads, mirror, rows
):
    instance = contract.namespace["KernelMetadata"](
        return_v=True, has_mirror=mirror, num_q_heads=q_heads, num_kv_heads=kv_heads
    )
    instance._rope_q = Mock()
    instance._norm_rope_k = Mock()
    instance._cast_rows = Mock()
    routes = (
        [("q", head) for head in range(q_heads)]
        + [("k", head) for head in range(kv_heads)]
        + [("v", head) for head in range(kv_heads)]
        + ([("mk", head) for head in range(kv_heads)] if mirror else [])
        + ([("mv", head) for head in range(kv_heads)] if mirror else [])
    )
    assert len(routes) * 256 == width
    slots = [-1] * rows
    if rows:
        slots[0] = 17
    if rows > 1:
        slots[-1] = 3
    for unit, (target, head) in enumerate(routes):
        writes = []
        context = dict(
            contract.namespace,
            self=instance,
            u=unit,
            pair=0,
            sub=0,
            rows_here=rows,
            row0=0,
            meta_row0=0,
            gm_slot=slots,
            gm_k_cache=_Tile("k_cache"),
            gm_v_cache=_Tile("v_cache"),
            q_span=_Tile("q"),
            k_span=_Tile("k"),
            v_span=_Tile("v"),
            mk_span=_Tile("mk"),
            mv_span=_Tile("mv"),
            cv_r=object(),
            out_w=object(),
            cs_r=object(),
            gamma_w=object(),
            epsilon=1e-6,
            out_ch=SimpleNamespace(next_read=lambda: _Tile("out")),
            const_expr=bool,
            tile_slice=lambda tensor, shape, coords: _Tile(
                tensor.target, shape, coords
            ),
            mem_copy=lambda dst, src: writes.append((dst, src)),
        )
        exec(contract.epilogue, context)
        if not rows:
            assert writes == []  # M=1 has an empty second AIV half.
            continue
        assert writes[0] == (
            _Tile(target, (256, 256), (0, head)),
            _Tile("out", (rows, 256)),
        )
        expected_scatter = (
            [
                _Tile(f"{target}_cache", (1, 256), (slot, head))
                for slot in slots
                if slot != -1
            ]
            if target in ("k", "v")
            else []
        )
        assert [dst for dst, _ in writes[1:]] == expected_scatter
        assert [src for _, src in writes[1:]] == (
            [
                _Tile("out", (1, 256), (row, 0))
                for row, slot in enumerate(slots)
                if slot != -1
            ]
            if target in ("k", "v")
            else []
        )
    assert instance._rope_q.call_count == q_heads
    assert instance._norm_rope_k.call_count == kv_heads
    assert instance._cast_rows.call_count == kv_heads * (3 if mirror else 1)


def _binding_inputs(width, kv_heads, *, layout="heads"):
    kvw = kv_heads * 256
    shapes = {
        "token": (192, kvw),
        "heads": (192, kv_heads, 256),
        "paged": (3, 64, kv_heads, 256),
    }
    return (
        _Tensor((5, 2048)),
        _Tensor((width, 2048)),
        _Tensor((256,)),
        _Tensor((5,), "i64"),
        _Tensor((4096, 64), "fp32"),
        _Tensor((5,), "i64"),
        _Tensor(shapes[layout]),
        _Tensor(shapes[layout]),
    )


@pytest.mark.parametrize(_PARAMS, _CASES)
@pytest.mark.parametrize("layout", ["token", "heads", "paged"])
@pytest.mark.parametrize("return_v", [False, True])
def test_binding_cache_views_retain_token_capacity_and_storage(
    contract, monkeypatch, tp, width, q_heads, kv_heads, mirror, layout, return_v
):
    op = Mock()
    constructor = Mock(return_value=op)
    monkeypatch.setitem(contract.namespace, "FusedQkvProjNormRopeCache", constructor)
    inputs = _binding_inputs(width, kv_heads, layout=layout)
    outputs = contract.namespace["fused_qkv_proj_norm_rope_cache"](
        *inputs, validate=False, positions_contiguous=False, return_v=return_v
    )
    launched = op.run.call_args.args
    assert len(launched) == 14
    for cache, original in zip(launched[11:13], inputs[6:8]):
        assert cache.shape == (192, kv_heads * 256)
        assert cache.storage is original.storage
    assert outputs[0].shape == (5, q_heads * 256)
    assert all(output.shape == (5, kv_heads * 256) for output in outputs[1:])
    assert len(outputs) == 2 + return_v + 2 * mirror
    assert launched[9].shape == ((5 if mirror else 1), kv_heads * 256)
    assert launched[10].shape == launched[9].shape
    assert constructor.call_args.kwargs == dict(
        return_v=return_v,
        has_mirror=mirror,
        positions_contiguous=False,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
    )


@pytest.mark.parametrize(
    "bad_cache",
    [
        _Tensor((384, 256)),  # TP1 must not turn its two heads into token rows.
        _Tensor((192, 4, 128)),
        _Tensor((192, 2, 256), contiguous=False),
        _Tensor((192, 2, 256), dtype="fp32"),
        _Tensor((193, 2, 256)),  # Capacity differs from the other cache.
    ],
)
def test_binding_rejects_unsafe_tp1_cache_views_before_launch(
    contract, monkeypatch, bad_cache
):
    constructor = Mock()
    monkeypatch.setitem(contract.namespace, "FusedQkvProjNormRopeCache", constructor)
    inputs = list(_binding_inputs(7168, 2))
    inputs[6] = bad_cache
    with pytest.raises(AssertionError):
        contract.namespace["fused_qkv_proj_norm_rope_cache"](
            *inputs, validate=False, positions_contiguous=False
        )
    constructor.assert_not_called()
