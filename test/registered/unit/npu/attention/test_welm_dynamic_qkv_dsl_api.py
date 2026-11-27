"""CPU source contracts for DSL 0.8 allocation and dynamic split-M tails.

Executes selected production AST with metadata doubles; no DSL imports or NPU
compilation/math. Runtime compilation and precision remain server checks.
"""

import ast
import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[5] / (
    "python/sglang/srt/layers/fused_qkv_proj_norm_rope_cache.py"
)
_ROWS = [1, 2, 3, 16, 32, 64, 127, 128, 129, 255, 256, 257]


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _execute(nodes, namespace):
    tree = ast.fix_missing_locations(
        ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    )
    exec(compile(tree, str(_SOURCE), "exec", dont_inherit=True), namespace)


@pytest.fixture
def source():
    tree = ast.parse(_SOURCE.read_text())
    kernel = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FusedQkvProjNormRopeCacheKernel"
    )
    run = next(
        node for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    constants = {}
    _execute([node for node in tree.body if isinstance(node, ast.Assign)], constants)
    return run, constants


def test_allocations_and_accessors_follow_dsl08_contract(source):
    run, ns = source
    allocations = []

    def ring(location, dtype, layout, *, depth):
        allocations.append((location, dtype, layout, depth))
        return object()

    ns.update(
        RingBuffer=ring,
        MemLoc=SimpleNamespace(**{x: x for x in ("L1", "L0A", "L0B", "L0C", "UB")}),
        BF16="bf16", F32="f32", make_layout=tuple,
    )
    statements = [
        node for node in run.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "RingBuffer"
    ]
    _execute(statements, ns)  # Strict signature rejects the old shape= argument.
    assert len(allocations) == 9
    assert allocations[4] == ("L0C", "f32", (256, 256), 1)
    assert allocations[5] == ("UB", "f32", (129, 256), 1)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("head_next", "tail_next")
        for node in ast.walk(run)
    )
    for node in ast.walk(run):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "copy_options"
        ):
            assert not {kw.arg for kw in node.keywords} & {"split_axis", "actual"}


@pytest.mark.parametrize("rows", _ROWS[:-1])
def test_mmad_and_fixpipe_share_the_compact_tail_view(source, rows):
    run, ns = source
    slots = {}

    class Buffer:
        def __init__(self, name):
            self.name = name

        def next_write(self, *, bare=False):
            assert bare
            return self

        def view(self, shape):
            slot = SimpleNamespace(shape=shape)
            slot.partition = lambda *, dim: (slot, dim)
            slots[self.name] = slot
            return slot

    ns.update(l0c=Buffer("c"), l0a=Buffer("a"), cur_tokens=rows, make_layout=tuple)
    for name in ("l0c_acc", "l0a_slot"):
        _execute([next(node for node in ast.walk(run) if _assigns(node, name))], ns)
    assert slots["c"].shape == (rows, 256)
    assert slots["a"].shape == (rows, 64)
    call = next(
        node for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "matmul"
    )
    received = []
    ns.update(
        matmul=lambda *args, **kwargs: received.append(args),
        gk=0, l0b_slot=object(),
    )
    _execute([ast.Expr(value=call)], ns)
    assert received[0][0] is slots["c"] and received[0][1] is slots["a"]
    fixpipe = next(
        node for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "mem_copy"
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "tile_slice"
        and isinstance(node.args[0].args[0], ast.Name)
        and node.args[0].args[0].id == "cv_w"
    )
    handoff = fixpipe.args[1]
    partition = eval(
        compile(ast.Expression(copy.deepcopy(handoff)), str(_SOURCE), "eval"), ns
    )
    assert partition[0] is slots["c"] and partition[1] == 0


class _Span:
    def __init__(self, rows, columns):
        self.shape = (rows, columns)

    def __getitem__(self, key):
        rows, columns = key
        assert columns is None and 0 <= rows.start <= rows.stop <= self.shape[0]
        return _Span(rows.stop - rows.start, self.shape[1])


@pytest.mark.parametrize("m", _ROWS)
@pytest.mark.parametrize("rank_chunk", [False, True])
def test_actual_split_schedule_covers_odd_tails_and_rank_chunk_offsets(
    source, m, rank_chunk
):
    run, ns = source
    first_alloc = next(i for i, node in enumerate(run.body) if _assigns(node, "l1_a"))
    loop_index = next(
        i for i, node in enumerate(run.body)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "item"
    )
    loop = copy.deepcopy(run.body[loop_index])
    preload = next(i for i, node in enumerate(loop.body) if _assigns(node, "gamma_w"))
    loop.body = loop.body[:preload] + ast.parse(
        "record(u, meta_row0, rows_here, cur_tokens, sub)"
    ).body
    # Keep the production total_items branch immediately before the item loop.
    code = run.body[:first_alloc] + [run.body[loop_index - 1], loop]
    ranks, width, start = (3 if rank_chunk else 1), 7168, 5
    full = m + 11 if rank_chunk else m
    records = []

    def tile(span, shape, coords):
        return _Span(min(shape[0], span.shape[0] - coords[0] * shape[0]), shape[1])

    ns.update(
        self=SimpleNamespace(rank_chunk_layout=rank_chunk, has_mirror=False),
        gm_hidden=_Span(ranks * m, 2048),
        gm_weight=_Span(width, 2048),
        **{f"gm_{name}": _Span(ranks * full, width) for name in ("q", "k", "v")},
        gm_mk=_Span(1, 512), gm_mv=_Span(1, 512),
        full_local_rows=full, chunk_local_rows=m, chunk_start=start,
        const_expr=bool, get_block_idx=lambda: 0, get_block_num=lambda: 1,
        tile_slice=tile,
        cv_valid_extent=lambda tensor, *, axis, part_id: (
            tensor.shape[0] + 1 - part_id
        ) // 2,
        record=lambda *args: records.append(args),
    )
    for sub in (0, 1):
        ns["get_subblock_id"] = lambda: sub
        _execute(code, ns)
    covered = Counter()
    for unit, row0, count, extent, sub in records:
        assert 1 <= extent <= 256 and 0 <= count <= 128
        assert count == (extent + 1 - sub) // 2
        assert 0 <= row0 < ranks * full
        covered.update((unit, row) for row in range(row0, row0 + count))
    assert covered == Counter(
        (unit, rank * full + (start if rank_chunk else 0) + row)
        for unit in range(width // 256)
        for rank in range(ranks)
        for row in range(m)
    )
