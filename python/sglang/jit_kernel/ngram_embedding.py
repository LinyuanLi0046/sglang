from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


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
    # Mirror UpdateTokenTableKernel semantics without scalar .item() reads.
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
def update_token_table_single_token(
    tokens: torch.Tensor,
    ne_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    ignore_tokens: torch.Tensor | None = None,
) -> None:
    """Update one token per request without ragged request expansion."""
    if ignore_tokens is None:
        ignore_tokens = tokens.new_empty(0, dtype=tokens.dtype)
    if tokens.numel() == 0 or row_indices.numel() == 0:
        return

    values = tokens
    if ignore_tokens.numel() > 0:
        ignore_mask = (values[:, None] == ignore_tokens[None, :]).any(dim=1)
        values = torch.where(ignore_mask, -values, values)

    rows = row_indices.to(torch.int64)
    cols = column_starts.to(torch.int64)
    flat_idx = rows * ne_token_table.stride(0) + cols * ne_token_table.stride(1)
    ne_token_table.view(-1).scatter_(0, flat_idx, values.to(ne_token_table.dtype))


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
