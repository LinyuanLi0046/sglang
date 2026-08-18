import ast
import math
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


_ATTENTION_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/hardware_backend/npu/attention/sink_full_attention.py"
)


def _do_not_specialize_for(function):
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "do_not_specialize":
                return {element.value for element in keyword.value.elts}
    return set()


def _as_bfloat16_then_float32(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _direct_sink_attention(q, k, v, sinks, scale, visible=None):
    logits = np.einsum("hd,td->ht", q, k) * scale
    if visible is not None:
        logits = np.where(visible[None, :], logits, -np.inf)
    sink_logits = sinks[:, None]
    combined = np.concatenate((sink_logits, logits), axis=1)
    combined = combined - np.max(combined, axis=1, keepdims=True)
    weights = np.exp(combined)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights[:, 1:] @ v


def _split_sink_attention(q, k, v, sinks, scale, split_parts, page_size):
    seq_len = k.shape[0]
    raw_chunk = math.ceil(seq_len / split_parts)
    chunk_size = math.ceil(raw_chunk / page_size) * page_size

    partial_acc = []
    partial_lse = []
    for split_idx in range(split_parts):
        start = min(split_idx * chunk_size, seq_len)
        end = min(start + chunk_size, seq_len)
        if start == end:
            partial_acc.append(np.zeros_like(q))
            partial_lse.append(np.full((q.shape[0],), -np.inf, dtype=np.float32))
            continue

        logits = np.einsum("hd,td->ht", q, k[start:end]) * scale
        local_max = np.max(logits, axis=1)
        exp_logits = np.exp(logits - local_max[:, None])
        denom = np.sum(exp_logits, axis=1)
        partial_acc.append((exp_logits @ v[start:end]) / denom[:, None])
        partial_lse.append(local_max + np.log(denom))

    partial_acc = np.stack(partial_acc, axis=0)
    partial_lse = np.stack(partial_lse, axis=0)
    lse_max = np.maximum(np.max(partial_lse, axis=0), sinks)
    split_weights = np.exp(partial_lse - lse_max[None, :])
    sink_weights = np.exp(sinks - lse_max)
    denom = np.sum(split_weights, axis=0) + sink_weights
    return np.einsum("sh,shd->hd", split_weights, partial_acc) / denom[:, None]


def _blocked_sink_attention(q, k, v, sinks, scale, block_size, visible=None):
    """Mirror the non-FD kernels' FP32 online-softmax update order."""
    running_max = sinks.astype(np.float32, copy=True)
    running_sum = np.ones_like(running_max)
    accumulator = np.zeros_like(q, dtype=np.float32)

    for start in range(0, k.shape[0], block_size):
        end = min(start + block_size, k.shape[0])
        logits = np.einsum("hd,td->ht", q, k[start:end]) * scale
        if visible is not None:
            logits = np.where(visible[None, start:end], logits, -np.inf)

        block_max = np.max(logits, axis=1)
        next_max = np.maximum(running_max, block_max)
        probabilities = np.exp(logits - next_max[:, None])
        previous_scale = np.exp(running_max - next_max)
        running_sum = (
            running_sum * previous_scale + np.sum(probabilities, axis=1)
        )
        accumulator = (
            accumulator * previous_scale[:, None]
            + probabilities @ v[start:end]
        )
        running_max = next_max

    return accumulator / running_sum[:, None]


def _swa_kernel_visible(seq_len, block_size, global_window, local_window):
    total_blocks = math.ceil(seq_len / block_size)
    global_blocks = min(math.ceil(global_window / block_size), total_blocks)
    local_start = max(seq_len - 1 - local_window, 0)
    local_start_block = local_start // block_size
    non_global_start = max(global_blocks, local_start_block)

    visible = np.zeros(seq_len, dtype=np.bool_)
    for block_id in range(global_blocks):
        start = block_id * block_size
        end = min(start + block_size, seq_len)
        positions = np.arange(start, end)
        visible[start:end] |= (positions < global_window) | (
            positions + local_window >= seq_len - 1
        )
    for block_id in range(non_global_start, total_blocks):
        start = block_id * block_size
        end = min(start + block_size, seq_len)
        positions = np.arange(start, end)
        visible[start:end] |= positions + local_window >= seq_len - 1
    return visible


def test_mirror_flash_decode_split_merge_matches_direct_attention():
    generator = np.random.default_rng(20260818)
    num_q_heads = 6
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    for seq_len in (1, 63, 64, 65, 129, 257):
        q = _as_bfloat16_then_float32(
            generator.standard_normal((num_q_heads, head_dim), dtype=np.float32)
        )
        k = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        v = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        sinks = generator.standard_normal(num_q_heads, dtype=np.float32)
        expected = _direct_sink_attention(q, k, v, sinks, scale)

        for split_parts in (1, 2, 3, 8):
            actual = _split_sink_attention(
                q,
                k,
                v,
                sinks,
                scale,
                split_parts,
                page_size=64,
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mirror_non_fd_online_softmax_matches_direct_attention():
    generator = np.random.default_rng(20260819)
    num_q_heads = 6
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    for seq_len in (1, 63, 64, 65, 129, 257):
        q = _as_bfloat16_then_float32(
            generator.standard_normal((num_q_heads, head_dim), dtype=np.float32)
        )
        k = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        v = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        sinks = generator.standard_normal(num_q_heads, dtype=np.float32)
        expected = _direct_sink_attention(q, k, v, sinks, scale)
        actual = _blocked_sink_attention(
            q,
            k,
            v,
            sinks,
            scale,
            block_size=64,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mirror_swa_block_partition_matches_visibility_definition():
    for seq_len in (1, 63, 64, 65, 127, 128, 129, 257):
        for global_window, local_window in ((0, 64), (17, 64), (64, 128)):
            positions = np.arange(seq_len)
            expected = (positions < global_window) | (
                positions + local_window >= seq_len - 1
            )
            actual = _swa_kernel_visible(
                seq_len,
                block_size=64,
                global_window=global_window,
                local_window=local_window,
            )
            np.testing.assert_array_equal(actual, expected)


def test_mirror_swa_online_softmax_matches_direct_attention():
    generator = np.random.default_rng(20260820)
    num_q_heads = 6
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    for seq_len in (1, 63, 64, 65, 127, 128, 129, 257):
        q = _as_bfloat16_then_float32(
            generator.standard_normal((num_q_heads, head_dim), dtype=np.float32)
        )
        k = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        v = _as_bfloat16_then_float32(
            generator.standard_normal((seq_len, head_dim), dtype=np.float32)
        )
        sinks = generator.standard_normal(num_q_heads, dtype=np.float32)
        positions = np.arange(seq_len)

        for global_window, local_window in ((0, 64), (17, 64), (64, 128)):
            visible = (positions < global_window) | (
                positions + local_window >= seq_len - 1
            )
            expected = _direct_sink_attention(
                q,
                k,
                v,
                sinks,
                scale,
                visible=visible,
            )
            actual = _blocked_sink_attention(
                q,
                k,
                v,
                sinks,
                scale,
                block_size=64,
                visible=visible,
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_only_request_varying_attention_scalars_skip_specialization():
    module = ast.parse(_ATTENTION_SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "paged_prefill_page_aggregation_kernel": {"stride_bt_batch"},
        "_swa_paged_prefill_aggregation_sink_kernel": {
            "bsz",
            "stride_block_table_b",
        },
        "mirror_paged_decode_fd_kernel": {
            "BATCH_SIZE",
            "KV_SPLIT_PARTS",
            "stride_bt_batch",
        },
        "mirror_paged_decode_fd_reduce_kernel": {
            "BATCH_SIZE",
            "KV_SPLIT_PARTS",
        },
        "mirror_paged_decode_kernel": {"BATCH_SIZE", "stride_bt_batch"},
        "mirror_swa_paged_decode_sink_kernel": {
            "BATCH_SIZE",
            "stride_bt_batch",
        },
    }

    for function_name, expected_runtime_scalars in expected.items():
        function = functions[function_name]
        assert _do_not_specialize_for(function) == expected_runtime_scalars
        constexpr_args = {
            argument.arg
            for argument in function.args.args
            if isinstance(argument.annotation, ast.Attribute)
            and argument.annotation.attr == "constexpr"
        }
        assert expected_runtime_scalars.isdisjoint(constexpr_args)
