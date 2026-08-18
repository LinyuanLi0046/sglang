import ast
import math
from pathlib import Path

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_KERNEL_SOURCE = (
    _REPO_ROOT / "python/sglang/kernels/ops/memory/allocator.py"
)
_NPU_ALLOCATOR_SOURCE = (
    _REPO_ROOT / "python/sglang/srt/hardware_backend/npu/allocator_npu.py"
)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _do_not_specialize(function: ast.FunctionDef) -> set[str]:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "do_not_specialize":
                return {element.value for element in keyword.value.elts}
    return set()


def _reference_alloc(
    prefix_lens: list[int],
    extend_lens: list[int],
    last_locs: list[int],
    free_pages: list[int],
    page_size: int,
) -> tuple[list[int], int]:
    output: list[int] = []
    free_page_cursor = 0
    for prefix_len, extend_len, last_loc in zip(
        prefix_lens, extend_lens, last_locs, strict=True
    ):
        part1 = 0
        if prefix_len % page_size:
            part1 = min(extend_len, page_size - prefix_len % page_size)
        output.extend(last_loc + 1 + offset for offset in range(part1))

        remaining = extend_len - part1
        request_pages = math.ceil(remaining / page_size)
        for offset in range(remaining):
            page_id = free_pages[free_page_cursor + offset // page_size]
            output.append(page_id * page_size + offset % page_size)
        free_page_cursor += request_pages
    return output, free_page_cursor


def _chunk_grid_alloc(
    prefix_lens: list[int],
    extend_lens: list[int],
    last_locs: list[int],
    free_pages: list[int],
    page_size: int,
    block_size: int,
) -> tuple[list[int], int]:
    seq_lens = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(
            prefix_lens, extend_lens, strict=True
        )
    ]
    output = [-1] * sum(extend_lens)
    num_chunks = max(1, math.ceil(max(extend_lens) / block_size))

    for request_id in range(len(prefix_lens)):
        output_start = sum(extend_lens[:request_id])
        new_page_start = sum(
            math.ceil(seq_lens[index] / page_size)
            - math.ceil(prefix_lens[index] / page_size)
            for index in range(request_id)
        )
        prefix_len = prefix_lens[request_id]
        seq_len = seq_lens[request_id]
        pages_before = math.ceil(prefix_len / page_size)
        pages_after = math.ceil(seq_len / page_size)
        num_new_pages = pages_after - pages_before

        part1_end = min(seq_len, pages_before * page_size)
        num_part1 = max(part1_end - prefix_len, 0)
        full_page_start = pages_before * page_size
        full_page_end = seq_len // page_size * page_size
        num_part2 = max(full_page_end - full_page_start, 0)
        num_part3 = seq_len - full_page_end

        for chunk_id in range(num_chunks):
            if chunk_id == 0:
                for offset in range(num_part1):
                    output[output_start + offset] = last_locs[request_id] + 1 + offset

            for offset in range(
                chunk_id * block_size,
                min((chunk_id + 1) * block_size, num_part2),
            ):
                page_id = free_pages[new_page_start + offset // page_size]
                output[output_start + num_part1 + offset] = (
                    page_id * page_size + offset % page_size
                )

            has_part3 = prefix_len + num_part1 < seq_len and num_part3 > 0
            if chunk_id == 0 and has_part3:
                page_id = free_pages[new_page_start + num_new_pages - 1]
                for offset in range(num_part3):
                    output[
                        output_start + num_part1 + num_part2 + offset
                    ] = page_id * page_size + offset

    return output, sum(
        math.ceil(seq_len / page_size)
        - math.ceil(prefix_len / page_size)
        for prefix_len, seq_len in zip(
            prefix_lens, seq_lens, strict=True
        )
    )


def _last_locs(prefix_lens: list[int], page_size: int) -> list[int]:
    return [
        -1
        if prefix_len == 0
        else (100 + request_id) * page_size
        + (prefix_len - 1) % page_size
        for request_id, prefix_len in enumerate(prefix_lens)
    ]


def test_npu_alloc_extend_has_one_request_independent_jit_contract():
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    kernel = _function(module, "alloc_extend_kernel_npu")

    assert _do_not_specialize(kernel) == {
        "pre_lens_ptr",
        "seq_lens_ptr",
        "last_loc_ptr",
        "free_page_ptr",
        "out_indices_ptr",
    }
    constexpr_args = {
        argument.arg
        for argument in kernel.args.args
        if isinstance(argument.annotation, ast.Attribute)
        and argument.annotation.attr == "constexpr"
    }
    assert constexpr_args == {"MAX_BATCH_SIZE", "PAGE_SIZE", "BLOCK_SIZE"}
    argument_names = {argument.arg for argument in kernel.args.args}
    assert "max_num_extend_tokens" not in argument_names
    assert "extend_num_tokens" not in argument_names
    assert "batch_size" not in argument_names
    assert not any(isinstance(node, ast.Return) for node in ast.walk(kernel))
    assert "tl.program_id(1)" in source


def test_npu_allocator_uses_framework_kernel_without_token_threshold():
    source = _NPU_ALLOCATOR_SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "sgl_kernel_npu.mem_cache.allocator" not in source
    assert "next_power_of_2(extend_num_tokens)" not in source
    assert "next_power_of_2(bs)" not in source
    assert "max_num_extend_tokens" not in source
    assert "num_new_pages_item < 200" not in source
    assert "extend_num_tokens <=" not in source
    assert "alloc_extend_kernel_npu[(bs, num_token_chunks)]" in source


def test_chunk_grid_math_matches_real_allocator_ordering():
    page_size = 128
    free_pages = list(range(1000, 20000))
    cases = [
        ([0], [1]),
        ([65], [16384]),
        ([0], [16385]),
        ([17], [20001]),
        (
            [0, 1, 65, 127, 128, 129, 255, 513],
            [1, 127, 128, 129, 2047, 2048, 4097, 20001],
        ),
        (
            [index % 257 for index in range(128)],
            [1 + (index * 193) % 4097 for index in range(128)],
        ),
    ]

    for prefix_lens, extend_lens in cases:
        last_locs = _last_locs(prefix_lens, page_size)
        expected, expected_pages = _reference_alloc(
            prefix_lens,
            extend_lens,
            last_locs,
            free_pages,
            page_size,
        )
        actual, actual_pages = _chunk_grid_alloc(
            prefix_lens,
            extend_lens,
            last_locs,
            free_pages,
            page_size,
            block_size=2048,
        )
        assert actual_pages == expected_pages
        assert actual == expected
