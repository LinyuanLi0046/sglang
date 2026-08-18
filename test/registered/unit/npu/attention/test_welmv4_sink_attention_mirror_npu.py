import math

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.sink_full_attention import (
    paged_attention_decode_impl,
    paged_attention_mirror_impl,
    paged_attention_prefill_impl,
    paged_attention_prefill_prepare,
    swa_paged_decode_impl,
    swa_paged_mirror_impl,
    swa_paged_prefill_impl,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="stage-a-unit-test-npu")


def _make_paged_case(seq_lens, *, num_q_heads=6, head_dim=256, page_size=64):
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    num_kv_heads = 1
    pages_per_request = [math.ceil(seq_len / page_size) for seq_len in seq_lens]
    num_pages = sum(pages_per_request)
    max_pages = max(pages_per_request)

    q = torch.randn(
        len(seq_lens),
        num_q_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    key_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    sinks = torch.randn(num_q_heads, device=device, dtype=torch.float32)
    block_tables = torch.full(
        (len(seq_lens), max_pages),
        -1,
        device=device,
        dtype=torch.int32,
    )

    next_page = 0
    for request_id, request_pages in enumerate(pages_per_request):
        block_tables[request_id, :request_pages] = torch.arange(
            next_page,
            next_page + request_pages,
            device=device,
            dtype=torch.int32,
        )
        next_page += request_pages

    return (
        q,
        key_cache,
        value_cache,
        torch.tensor(seq_lens, device=device, dtype=torch.int32),
        block_tables,
        sinks,
    )


def _gather_request_cache(cache, block_table, seq_len):
    page_size = cache.shape[2]
    num_pages = math.ceil(seq_len / page_size)
    page_ids = block_table[:num_pages].to(torch.long)
    return (
        cache.index_select(0, page_ids)
        .permute(0, 2, 1, 3)
        .reshape(-1, cache.shape[1], cache.shape[3])[:seq_len]
    )


def _reference_sink_attention(
    q,
    key_cache,
    value_cache,
    seq_lens,
    block_tables,
    sinks,
    *,
    local_window=None,
    global_window=0,
):
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for request_id, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor.item())
        key = _gather_request_cache(
            key_cache,
            block_tables[request_id],
            seq_len,
        )[:, 0].float()
        value = _gather_request_cache(
            value_cache,
            block_tables[request_id],
            seq_len,
        )[:, 0].float()
        logits = torch.matmul(q[request_id].float(), key.transpose(0, 1)) * scale

        if local_window is not None:
            positions = torch.arange(seq_len, device=q.device)
            visible = (positions < global_window) | (
                positions + local_window >= seq_len - 1
            )
            logits = logits.masked_fill(~visible[None, :], float("-inf"))

        all_logits = torch.cat((sinks[:, None], logits), dim=1)
        weights = torch.softmax(all_logits, dim=1)[:, 1:]
        outputs.append(torch.matmul(weights, value))
    return torch.stack(outputs).to(q.dtype)


def _reference_prefill_sink_attention(
    q,
    key_cache,
    value_cache,
    cu_q_lens,
    seq_lens,
    block_tables,
    sinks,
    *,
    local_window=None,
    global_window=0,
):
    outputs = []
    scale = 1.0 / math.sqrt(q.shape[-1])
    for request_id, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor.item())
        q_start = int(cu_q_lens[request_id].item())
        q_end = int(cu_q_lens[request_id + 1].item())
        q_len = q_end - q_start
        prefix_len = seq_len - q_len
        key = _gather_request_cache(
            key_cache,
            block_tables[request_id],
            seq_len,
        )[:, 0].float()
        value = _gather_request_cache(
            value_cache,
            block_tables[request_id],
            seq_len,
        )[:, 0].float()

        query = q[q_start:q_end].float()
        logits = torch.einsum("qhd,td->qht", query, key) * scale
        positions = torch.arange(seq_len, device=q.device)
        absolute_q_positions = prefix_len + torch.arange(q_len, device=q.device)
        visible = positions[None, :] <= absolute_q_positions[:, None]
        if local_window is not None:
            visible &= (positions[None, :] < global_window) | (
                positions[None, :] + local_window
                >= absolute_q_positions[:, None]
            )
        logits = logits.masked_fill(~visible[:, None, :], float("-inf"))
        sink_logits = sinks[None, :, None].expand(q_len, -1, -1)
        all_logits = torch.cat((sink_logits, logits), dim=2)
        weights = torch.softmax(all_logits, dim=2)[:, :, 1:]
        outputs.append(torch.einsum("qht,td->qhd", weights, value))
    return torch.cat(outputs, dim=0).to(q.dtype)


def _assert_kernel_precision(actual, previous, reference):
    torch.npu.synchronize()
    torch.testing.assert_close(actual, previous, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)


def _make_prefill_case(q_lens=(128,), seq_lens=(256,)):
    assert len(q_lens) == len(seq_lens)
    assert all(q_len <= seq_len for q_len, seq_len in zip(q_lens, seq_lens))
    _, key_cache, value_cache, seq_lens_tensor, block_tables, sinks = (
        _make_paged_case(seq_lens)
    )
    total_q_len = sum(q_lens)
    q = torch.randn(
        total_q_len,
        sinks.shape[0],
        key_cache.shape[-1],
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    cu_q_lens_cpu = [0]
    for q_len in q_lens:
        cu_q_lens_cpu.append(cu_q_lens_cpu[-1] + q_len)
    cu_q_lens = torch.tensor(
        cu_q_lens_cpu,
        device=q.device,
        dtype=torch.int32,
    )
    return (
        q,
        key_cache,
        value_cache,
        cu_q_lens,
        seq_lens_tensor,
        block_tables,
        sinks,
    )


@torch.no_grad()
def test_full_prefill_page_aggregation_compiles_and_matches_reference():
    # The second call changes both batch size and the block-table row stride.
    # It must reuse the first call's JIT binary while still addressing every
    # request's pages correctly.
    for q_lens, seq_lens in (
        ((128,), (256,)),
        ((64, 96), (320, 448)),
    ):
        case = _make_prefill_case(q_lens, seq_lens)
        q, key_cache, value_cache, cu_q_lens, seq_lens_t, block_tables, sinks = case
        task_schedule = paged_attention_prefill_prepare(
            cu_q_lens,
            seq_lens_t,
            q.shape[1],
            key_cache.shape[1],
            False,
            key_cache.shape[2],
            device=q.device,
        )
        actual = paged_attention_prefill_impl(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            cu_q_lens=cu_q_lens,
            seqlens_kv=seq_lens_t,
            block_tables=block_tables,
            gqa_interleave=False,
            task_b=task_schedule[0],
            task_q_block=task_schedule[1],
            task_q_head=task_schedule[2],
            core_task_offsets=task_schedule[3],
            sinks=sinks,
        )
        reference = _reference_prefill_sink_attention(*case)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)


@torch.no_grad()
def test_swa_prefill_aggregation_compiles_and_matches_reference():
    local_window = 128
    for q_lens, seq_lens in (
        ((128,), (256,)),
        ((64, 96), (320, 448)),
    ):
        case = _make_prefill_case(q_lens, seq_lens)
        q, key_cache, value_cache, cu_q_lens, seq_lens_t, block_tables, sinks = case
        actual = swa_paged_prefill_impl(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            cu_q_lens=cu_q_lens,
            kvlens=seq_lens_t,
            block_table=block_tables,
            is_causal=True,
            local_window_size=local_window,
            global_window_size=0,
            gqa_interleave=False,
            sinks=sinks,
        )
        reference = _reference_prefill_sink_attention(
            *case,
            local_window=local_window,
            global_window=0,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)


@torch.no_grad()
def test_full_mirror_non_fd_matches_previous_kernel_and_reference():
    case = _make_paged_case([257, 193])
    q, key_cache, value_cache, seq_lens, block_tables, sinks = case
    kwargs = dict(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        seqlens=seq_lens,
        block_tables=block_tables,
        gqa_interleave=False,
        sinks=sinks,
    )
    previous = paged_attention_decode_impl(**kwargs)
    actual = paged_attention_mirror_impl(**kwargs)
    reference = _reference_sink_attention(*case)
    _assert_kernel_precision(actual, previous, reference)


@torch.no_grad()
def test_full_mirror_fd_matches_previous_kernel_and_reference():
    case = _make_paged_case([2048])
    q, key_cache, value_cache, seq_lens, block_tables, sinks = case
    kwargs = dict(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        seqlens=seq_lens,
        block_tables=block_tables,
        gqa_interleave=False,
        sinks=sinks,
        max_kv_len_hint=2048,
    )
    previous = paged_attention_decode_impl(**kwargs)
    actual = paged_attention_mirror_impl(**kwargs)
    reference = _reference_sink_attention(*case)
    _assert_kernel_precision(actual, previous, reference)


@torch.no_grad()
def test_swa_mirror_matches_previous_kernel_and_reference():
    local_window = 128
    case = _make_paged_case([257, 193])
    q, key_cache, value_cache, seq_lens, block_tables, sinks = case
    kwargs = dict(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        seqlens=seq_lens,
        block_tables=block_tables,
        local_window_size=local_window,
        global_window_size=0,
        gqa_interleave=False,
        sinks=sinks,
    )
    previous = swa_paged_decode_impl(**kwargs)
    actual = swa_paged_mirror_impl(**kwargs)
    reference = _reference_sink_attention(
        *case,
        local_window=local_window,
        global_window=0,
    )
    _assert_kernel_precision(actual, previous, reference)
