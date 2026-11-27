"""CPU metadata contracts for the static QKV M32/64 single-launch tiling.

Only selected production AST runs with metadata doubles. No torch, SGLang,
CANNBotDSL, device code, compilation or numerical kernels are imported/run.
These contracts check ABI, launch geometry and row/head/slot routing only.
"""

import ast
import builtins
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[5]
_SOURCE = _ROOT / "python/sglang/srt/layers/welmv4_static_decode_qkv_kernel.py"
_KINDS = ("plain", "source", "query")


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _code(nodes):
    tree = ast.fix_missing_locations(
        ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    )
    return compile(tree, str(_SOURCE), "exec", dont_inherit=True)


def _reject_import(name, *args, **kwargs):
    raise AssertionError(f"Unexpected runtime import in metadata contract: {name}")


@dataclass
class _TensorSpec:
    shape: tuple
    dtype: str


@dataclass
class _Tile:
    target: str
    shape: tuple = ()
    coords: tuple = ()

    def __getitem__(self, indices):
        rows, axis = indices
        assert isinstance(rows, slice) and rows.start is None and rows.step is None
        assert 1 <= rows.stop <= 16 and axis is None
        return self


class _Rows:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def __getitem__(self, row):
        assert 0 <= row < len(self.values)
        self.reads.append(row)
        return self.values[row]


@pytest.fixture
def contract():
    tree = ast.parse(_SOURCE.read_text(), filename=str(_SOURCE))
    body = []
    kernel = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FusedQkvProjNormRopeCacheTP1Kernel"
    )
    run = next(
        node for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    metadata = copy.deepcopy(kernel)
    metadata.name = "KernelMetadata"
    metadata.decorator_list = []
    constructor = next(
        node for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    metadata.body = [copy.deepcopy(constructor)]
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Module constants use only literals/stdlib arithmetic.
            body.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name == "compile_qkv":
            body.append(copy.deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name == "_FusedQkv":
            host = copy.deepcopy(node)
            host.decorator_list = []
            for method in host.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
            body.append(host)
    body.append(metadata)
    compiler = Mock(
        side_effect=lambda entry, *specs: SimpleNamespace(entry=entry, specs=specs)
    )
    namespace = dict(
        __name__="_welm_static_qkv_kernel_contract",
        __builtins__=dict(vars(builtins), __import__=_reject_import),
        TensorSpec=_TensorSpec,
        BF16="bf16",
        F32="fp32",
        dtypes=SimpleNamespace(int64="i64"),
        cannbotdsl=SimpleNamespace(compile=compiler),
    )
    exec(_code(body), namespace)
    # Keep the real scalar setup, grid-stride loop, tile origin and epilogue.
    # Drop allocations and Cube/vector math rather than emulating the DSL.
    allocation_start = next(
        i for i, node in enumerate(run.body) if _assigns(node, "l1_a")
    )
    loop = next(
        node for node in run.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "item"
    )
    compute_start = next(
        i for i, node in enumerate(loop.body) if _assigns(node, "l0c_acc")
    )
    input_tile = next(node for node in ast.walk(loop) if _assigns(node, "a_tile"))
    weight_tile = next(node for node in ast.walk(loop) if _assigns(node, "b_tile"))
    positions_loop = next(
        node for node in ast.walk(loop)
        if isinstance(node, ast.For) and any(_assigns(child, "pos") for child in node.body)
    )
    epilogue = next(
        node for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "u"
        and isinstance(node.test.comparators[0], ast.Attribute)
        and node.test.comparators[0].attr == "q_heads"
    )
    routing_loop = copy.deepcopy(loop)
    record = ast.parse(
        "record_tile(pair, u, row0, cur_tokens, pair_m, m_tiles, a_tile, b_tile)"
    ).body
    routing_loop.body = copy.deepcopy(loop.body[:compute_start]) + [
        copy.deepcopy(input_tile), copy.deepcopy(weight_tile), *record,
        copy.deepcopy(positions_loop), copy.deepcopy(epilogue),
    ]
    layouts = {}
    for name in ("l1_a", "l0a", "l0c", "cv_ub", "out_ch", "cs_ch"):
        assignment = next(node for node in run.body if _assigns(node, name))
        layout = next(
            node for node in ast.walk(assignment)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "make_layout"
        )
        layouts[name] = compile(
            ast.Expression(copy.deepcopy(layout.args[0])), str(_SOURCE), "eval"
        )
    return SimpleNamespace(
        namespace=namespace,
        compiler=compiler,
        routing=_code(run.body[:allocation_start] + [routing_loop]),
        layouts=layouts,
    )


def _compile(contract, tp, kind, m, budget=24):
    return contract.namespace["compile_qkv"](
        257, 4096, tp_size=tp, has_mirror=kind == "source",
        static_m=m, max_blocks=budget, q_only=kind == "query",
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("m", [16, 32, 64])
def test_compile_specs_preserve_full_m(contract, tp, kind, m):
    compiled = _compile(contract, tp, kind, m)
    qw, kvw = 6144 // tp, 512 if tp == 1 else 256
    n = qw + {"plain": 2, "source": 4, "query": 0}[kind] * kvw
    assert compiled.specs == (
        _TensorSpec((m, 2048), "bf16"),
        _TensorSpec((n, 2048), "bf16"),
        _TensorSpec((1, 256), "bf16"),
        _TensorSpec((m,), "i64"),
        _TensorSpec((4096, 64), "fp32"),
        _TensorSpec((m,), "i64"),
        _TensorSpec((m, qw), "bf16"),
        _TensorSpec((1 if kind == "query" else m, kvw), "bf16"),
        _TensorSpec((1 if kind == "query" else m, kvw), "bf16"),
        _TensorSpec((m if kind == "source" else 1, kvw), "bf16"),
        _TensorSpec((m if kind == "source" else 1, kvw), "bf16"),
        _TensorSpec((257, kvw), "bf16"),
        _TensorSpec((257, kvw), "bf16"),
        "fp32",
    )
    contract.compiler.assert_called_once()
    assert compiled.entry.__self__.fixed_m == m


@pytest.mark.parametrize("m", [3, 128])
def test_compile_rejects_unapproved_rows_before_dsl_compile(contract, m):
    with pytest.raises(ValueError, match="static_m"):
        _compile(contract, 1, "plain", m)
    contract.compiler.assert_not_called()


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("m", [4, 32, 64])
@pytest.mark.parametrize("budget", [1, 24, 1000])
def test_host_uses_one_launch_and_budgets_all_row_tiles(contract, tp, kind, m, budget):
    compiled = _compile(contract, tp, kind, m, budget)
    launch = Mock()
    grid = Mock(return_value=launch)

    class Launcher:
        def __getitem__(self, blocks):
            return grid(blocks)

    constructor = Mock(return_value=Launcher())
    contract.namespace["FusedQkvProjNormRopeCacheTP1Kernel"] = constructor
    arguments = tuple(
        SimpleNamespace(shape=spec.shape) if isinstance(spec, _TensorSpec) else 1e-6
        for spec in compiled.specs
    )
    compiled.entry(*arguments)
    units = arguments[1].shape[0] // 256
    grid.assert_called_once_with(min((m // min(m, 16)) * units, budget))
    constructor.assert_called_once_with(
        m, kind == "source", 24 // tp, max(1, 2 // tp), kind == "query"
    )
    launch.assert_called_once_with(*arguments)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("m", [4, 32, 64])
@pytest.mark.parametrize("budget", [1, 7, 24])
def test_kernel_tiles_cover_all_rows_heads_and_physical_slots(
    contract, tp, kind, m, budget
):
    q_heads, kv_heads = 24 // tp, max(1, 2 // tp)
    routes = [("q", h) for h in range(q_heads)]
    if kind != "query":
        routes += [(target, h) for target in ("k", "v") for h in range(kv_heads)]
    if kind == "source":
        routes += [(target, h) for target in ("mk", "mv") for h in range(kv_heads)]
    instance = contract.namespace["KernelMetadata"](
        m, kind == "source", q_heads, kv_heads, kind == "query"
    )
    instance._rope_q = Mock()
    instance._norm_rope_k = Mock()
    instance._cast_rows = Mock()
    # Distinct, nonmonotonic physical slots expose row0/pair mixups. Slot zero
    # remains writable, while invalid entries straddle every 16-row boundary.
    invalid_rows = {row for row in range(m) if row % 16 in (0, 15)} - {0}
    slots = _Rows([
        -1 if row in invalid_rows else (row * 37) % 257 for row in range(m)
    ])
    positions = _Rows([1000 + row * 3 for row in range(m)])
    tiles, writes = [], []
    blocks = min((m // min(m, 16)) * len(routes), budget)
    context = dict(
        contract.namespace,
        self=instance,
        gm_hidden=_Tile("hidden"),
        gm_weight=_Tile("weight", (len(routes) * 256, 2048)),
        gm_positions=positions,
        gm_slot=slots,
        gm_cos_sin=_Tile("cos_sin"),
        **{
            f"gm_{name}": _Tile(name)
            for name in ("q", "k", "v", "mk", "mv", "k_cache", "v_cache")
        },
        get_block_num=lambda: blocks,
        get_subblock_id=lambda: 0,
        const_expr=bool,
        k_l1=0,
        epsilon=1e-6,
        cv_r=object(), out_w=object(), cs_r=object(), gamma_w=object(),
        cs_w=_Tile("cos_sin_tile"),
        out_ch=SimpleNamespace(next_read=lambda: _Tile("out")),
        tile_slice=lambda tensor, shape, coords: _Tile(tensor.target, shape, coords),
        record_tile=lambda *args: tiles.append(args),
        mem_copy=lambda dst, src: writes.append((context["row0"], dst, src)),
    )
    for block in range(blocks):
        context["get_block_idx"] = lambda: block
        exec(contract.routing, context)
    tile_m = min(m, 16)
    assert Counter((pair, unit) for pair, unit, *_ in tiles) == Counter(
        (pair, unit) for pair in range(m // tile_m) for unit in range(len(routes))
    )
    for pair, unit, row0, extent, actual_tile_m, m_tiles, input_tile, weight_tile in tiles:
        assert (row0, extent, actual_tile_m, m_tiles) == (
            pair * tile_m, tile_m, tile_m, m // tile_m
        )
        assert input_tile == _Tile("hidden", (tile_m, 256), (pair, 0))
        # Weight rows use the head unit, never the global row-tile work item.
        assert weight_tile == _Tile("weight", (256, 256), (unit, 0))
    output_rows = Counter()
    cache_rows = Counter()
    for row0, dst, src in writes:
        if dst.target in ("q", "k", "v", "mk", "mv"):
            height, width = dst.shape
            pair, head = dst.coords
            assert (height, width) == (tile_m, 256)
            assert row0 == pair * tile_m
            output_rows.update(
                (dst.target, row, head)
                for row in range(pair * height, (pair + 1) * height)
            )
        elif dst.target in ("k_cache", "v_cache"):
            slot, head = dst.coords
            assert dst.shape == src.shape == (1, 256)
            assert src.target == "out" and src.coords[1] == 0
            cache_rows[(dst.target, slot, head, row0 + src.coords[0])] += 1
        elif dst.target == "cos_sin_tile":
            local_row, column = dst.coords
            assert column == 0
            assert src == _Tile(
                "cos_sin", (1, 64), (positions.values[row0 + local_row], 0)
            )
    assert output_rows == Counter(
        (target, row, head) for target, head in routes for row in range(m)
    )
    assert cache_rows == Counter(
        (f"{target}_cache", slots.values[row], head, row)
        for target, head in routes if target in ("k", "v")
        for row in range(m) if slots.values[row] != -1
    )
    assert Counter(positions.reads) == Counter({row: len(routes) for row in range(m)})
    if kind == "query":
        assert not slots.reads
    else:
        assert Counter(slots.reads) == Counter({row: 2 * kv_heads for row in range(m)})
    # Resource layouts must use the <=16-row tile, not the full M32/64 ABI.
    expected_layouts = {
        "l1_a": (tile_m, 256), "l0a": (tile_m, 64), "l0c": (tile_m, 256),
        "cv_ub": (tile_m + 1, 256), "out_ch": (tile_m, 256),
        "cs_ch": (tile_m + 1, 64),
    }
    assert {
        name: eval(code, context) for name, code in contract.layouts.items()
    } == expected_layouts
