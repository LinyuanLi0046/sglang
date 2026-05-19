from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

try:
    import triton.runtime.driver as triton_driver
except ImportError:
    triton_driver = None

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _can_use_npu_triton(device: torch.device) -> bool:
    return (
        device.type == "npu"
        and hasattr(torch, "npu")
        and torch.npu.is_available()
        and triton_driver is not None
    )


def _get_npu_vectorcore_num() -> int:
    device = torch.npu.current_device()
    return triton_driver.active.utils.get_device_properties(device)["num_vectorcore"]


@triton.jit
def _update_token_table_single_token_npu_kernel(
    tokens_ptr,
    token_table_ptr,
    row_indices_ptr,
    column_starts_ptr,
    batch_size,
    max_context_len,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for req_id in range(pid, batch_size, num_programs):
        row = tl.load(row_indices_ptr + req_id).to(tl.int32)
        col = tl.load(column_starts_ptr + req_id)
        value = tl.load(tokens_ptr + req_id)
        out_idx = row * max_context_len + col
        tl.store(token_table_ptr + out_idx, value)


@triton.jit
def _update_token_table_ragged_npu_kernel(
    tokens_ptr,
    token_table_ptr,
    row_indices_ptr,
    column_starts_ptr,
    req_starts_ptr,
    req_lens_ptr,
    batch_size,
    max_context_len,
    BLOCK_TOKENS: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_offsets = tl.arange(0, BLOCK_TOKENS)

    for req_id in range(pid, batch_size, num_programs):
        row = tl.load(row_indices_ptr + req_id).to(tl.int32)
        col_start = tl.load(column_starts_ptr + req_id)
        token_start = tl.load(req_starts_ptr + req_id)
        req_len = tl.load(req_lens_ptr + req_id)
        has_tokens = req_len > 0
        safe_col_start = tl.where(has_tokens, col_start, 0)
        safe_token_start = tl.where(has_tokens, token_start, 0)

        for tile_id in tl.static_range(0, MAX_TILES):
            offs = tile_id * BLOCK_TOKENS + tile_offsets
            mask = offs < req_len
            safe_offs = tl.where(mask, offs, 0)
            values = tl.load(
                tokens_ptr + safe_token_start + safe_offs, mask=mask, other=0
            )
            out_idx = row * max_context_len + safe_col_start + safe_offs
            tl.store(token_table_ptr + out_idx, values, mask=mask)


def _should_use_npu_triton_update(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: torch.Tensor,
) -> bool:
    return (
        _can_use_npu_triton(tokens.device)
        and tokens.dtype == torch.int32
        and ne_token_table.dtype == torch.int32
        and row_indices.dtype == torch.int64
        and column_starts.dtype == torch.int32
        and req_lens.dtype == torch.int32
        and tokens.is_contiguous()
        and ne_token_table.is_contiguous()
        and row_indices.is_contiguous()
        and column_starts.is_contiguous()
        and req_lens.is_contiguous()
        and ignore_tokens.numel() == 0
    )


def _update_token_table_single_token_npu_triton(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
) -> None:
    batch_size = row_indices.numel()
    if batch_size == 0 or tokens.numel() == 0:
        return

    grid = (min(batch_size, _get_npu_vectorcore_num()),)
    _update_token_table_single_token_npu_kernel[grid](
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        batch_size,
        ne_token_table.shape[1],
    )


def _update_token_table_ragged_npu_triton(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
) -> None:
    batch_size = row_indices.numel()
    total_tokens = tokens.numel()
    if batch_size == 0 or total_tokens == 0:
        return

    req_starts = torch.cumsum(req_lens, dim=0, dtype=torch.int32) - req_lens
    block_tokens = 128
    max_tiles = max(1, triton.cdiv(ne_token_table.shape[1], block_tokens))
    grid = (min(batch_size, _get_npu_vectorcore_num()),)
    _update_token_table_ragged_npu_kernel[grid](
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        req_starts,
        req_lens,
        batch_size,
        ne_token_table.shape[1],
        BLOCK_TOKENS=block_tokens,
        MAX_TILES=max_tiles,
    )


@cache_once
def _jit_ngram_embedding_module() -> Module:
    return load_jit(
        "ngram_embedding",
        cuda_files=["ngram_embedding.cuh"],
        cuda_wrappers=[
            ("compute_n_gram_ids", "&NgramEmbeddingKernel::compute_n_gram_ids"),
            ("update_token_table", "&NgramEmbeddingKernel::update_token_table"),
        ],
    )


def _update_token_table_torch_fallback(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: torch.Tensor,
) -> None:
    # Mirror UpdateTokenTableKernel semantics without host scalar reads.
    # Each request consumes a contiguous slice from `tokens` and writes into
    # [row_indices[req_id], column_starts[req_id]:].
    req_starts = torch.cumsum(req_lens, dim=0, dtype=torch.int32) - req_lens
    total_tokens = tokens.numel()
    if total_tokens == 0 or row_indices.numel() == 0:
        return

    req_lens_i64 = req_lens.to(torch.int64)
    req_ids = torch.repeat_interleave(
        torch.arange(row_indices.numel(), device=tokens.device, dtype=torch.int64),
        req_lens_i64,
    )
    req_starts_rep = req_starts.to(torch.int64).repeat_interleave(req_lens_i64)
    token_offsets = torch.arange(total_tokens, device=tokens.device, dtype=torch.int64)
    pos_in_req = token_offsets - req_starts_rep

    rows = row_indices.to(torch.int64).index_select(0, req_ids)
    cols = column_starts.to(torch.int64).index_select(0, req_ids) + pos_in_req

    values = tokens
    if ignore_tokens.numel() > 0:
        ignore_mask = (values[:, None] == ignore_tokens[None, :]).any(dim=1)
        values = torch.where(ignore_mask, -values, values)

    ne_token_table[rows, cols] = values


@debug_kernel_api
def compute_n_gram_ids(
    ne_n: int,
    ne_k: int,
    ne_weights: torch.Tensor,
    ne_mods: torch.Tensor,
    exclusive_ne_embedder_size_sums: torch.Tensor,
    tokens: torch.Tensor,
    exclusive_req_len_sums: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    n_gram_ids: torch.Tensor,
) -> None:
    """
    Compute n-gram IDs for embedding.

    Args:
        ne_n: n value for n-gram
        ne_k: k value for n-gram configurations
        ne_weights: weights tensor with shape [ne_n-1, ne_k, ne_n]
        ne_mods: mods tensor with shape [ne_n-1, ne_k]
        exclusive_ne_embedder_size_sums: exclusive sum of embedder sizes
        tokens: input token ids
        exclusive_req_len_sums: exclusive sum of request lengths
        ne_token_table: token table for all requests
        row_indices: row indices for each request
        column_starts: column start positions for each request
        n_gram_ids: output tensor for n-gram ids
    """
    module = _jit_ngram_embedding_module()
    module.compute_n_gram_ids(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        tokens,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
        n_gram_ids,
    )


@debug_kernel_api
def update_token_table(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: torch.Tensor | None = None,
) -> None:
    """
    Update the token table with new tokens.

    Args:
        tokens: input token ids
        ne_token_table: token table for all requests
        row_indices: row indices for each request
        column_starts: column start positions for each request
        req_lens: request lengths
        ignore_tokens: tokens to be ignored (marked as negative in table)
    """
    if ignore_tokens is None:
        # Create an empty tensor for ignore_tokens
        ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)

    if _should_use_npu_triton_update(
        tokens=tokens,
        ne_token_table=ne_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
        req_lens=req_lens,
        ignore_tokens=ignore_tokens,
    ):
        _update_token_table_ragged_npu_triton(
            tokens=tokens,
            ne_token_table=ne_token_table,
            row_indices=row_indices,
            column_starts=column_starts,
            req_lens=req_lens,
        )
        return

    if tokens.device.type != "cuda":
        _update_token_table_torch_fallback(
            tokens=tokens,
            ne_token_table=ne_token_table,
            row_indices=row_indices,
            column_starts=column_starts,
            req_lens=req_lens,
            ignore_tokens=ignore_tokens,
        )
        return

    module = _jit_ngram_embedding_module()
    module.update_token_table(
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        req_lens,
        ignore_tokens,
    )


@debug_kernel_api
def update_token_table_single_token(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    ignore_tokens: torch.Tensor | None = None,
) -> None:
    if ignore_tokens is None:
        ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)

    if (
        _can_use_npu_triton(tokens.device)
        and tokens.dtype == torch.int32
        and ne_token_table.dtype == torch.int32
        and row_indices.dtype == torch.int64
        and column_starts.dtype == torch.int32
        and tokens.is_contiguous()
        and ne_token_table.is_contiguous()
        and row_indices.is_contiguous()
        and column_starts.is_contiguous()
        and ignore_tokens.numel() == 0
    ):
        _update_token_table_single_token_npu_triton(
            tokens=tokens,
            ne_token_table=ne_token_table,
            row_indices=row_indices,
            column_starts=column_starts,
        )
        return

    update_token_table(
        tokens=tokens,
        ne_token_table=ne_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
        req_lens=torch.ones_like(column_starts),
        ignore_tokens=ignore_tokens,
    )
