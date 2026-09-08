"""Utilities for LongCat/WeLM ngram embedding token tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.speculative.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ForwardMode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class NgramEmbeddingManager:
    enabled: bool
    table: Optional[torch.Tensor]
    n: int
    k: int

    def share_table_from(
        self, owner: "NgramEmbeddingManager"
    ) -> "NgramEmbeddingManager":
        """Return a manager view that uses the canonical owner's token table."""
        if not self.enabled:
            return self
        if not owner.enabled or owner.table is None:
            raise RuntimeError("Cannot share ngram state from a disabled owner")
        if (self.n, self.k) != (owner.n, owner.k):
            raise RuntimeError(
                "Target/draft ngram configurations differ: "
                f"target={(owner.n, owner.k)} vs draft={(self.n, self.k)}."
            )
        if self.table is not None and self.table.shape != owner.table.shape:
            raise RuntimeError(
                "Target/draft ngram token-table shapes differ: "
                f"{owner.table.shape} vs {self.table.shape}."
            )
        if self.table is not None and (
            self.table.dtype != owner.table.dtype
            or self.table.device != owner.table.device
        ):
            raise RuntimeError(
                "Target/draft ngram token tables must share dtype/device: "
                f"target={owner.table.dtype}/{owner.table.device}, "
                f"draft={self.table.dtype}/{self.table.device}."
            )
        return NgramEmbeddingManager(
            enabled=True,
            table=owner.table,
            n=self.n,
            k=self.k,
        )

    def commit_speculative_accepts(
        self,
        *,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        old_seq_lens: torch.Tensor,
    ) -> None:
        """Commit accepted predictions after the incoming bonus/root column."""
        if not self.enabled or self.table is None or accept_index.numel() == 0:
            return
        if is_npu():
            from sglang.srt.layers.welmv4_npu_op import (
                welmv4_token_table_spec_accept_update_npu,
            )

            welmv4_token_table_spec_accept_update_npu(
                self.table,
                predict,
                accept_index,
                accept_lens,
                req_pool_indices,
                old_seq_lens,
            )
            return

        width = accept_index.shape[1]
        step = torch.arange(width, device=accept_index.device).view(1, -1)
        valid = step < accept_lens.view(-1, 1)
        safe = accept_index.to(torch.int64).clamp(min=0)
        values = predict[safe]
        rows = req_pool_indices.to(torch.int64).view(-1, 1).expand_as(safe)
        # old_seq_lens points at the incoming bonus/root b0. Accepted target
        # predictions begin at the following column, hence the required +1.
        columns = old_seq_lens.to(torch.int64).view(-1, 1) + 1 + step
        self.table[rows[valid], columns[valid]] = values[valid].to(self.table.dtype)

    @classmethod
    def from_model(
        cls,
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        req_to_token_pool: ReqToTokenPool,
        server_args: ServerArgs,
        max_running_requests: int,
        device: str,
    ):
        token_table = None
        ngram_embedding_n = 0
        ngram_embedding_k = 0
        use_ngram_embedding = model_config.use_ngram_embedding
        if use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            token_table = torch.empty(
                req_to_token_pool.req_to_token.shape[0],
                model_config.context_len,
                dtype=torch.int32,
                device=device,
            )
            chunked_prefill_size = server_args.chunked_prefill_size
            ngram_modules = [
                module for module in model.modules() if isinstance(module, NgramEmbedding)
            ]
            # LongCat's NgramEmbedding owns per-forward scratch buffers sized
            # from chunked prefill. WeLMv4 computes its hashes directly from
            # the shared token table and therefore does not need those buffers.
            if ngram_modules:
                assert (
                    chunked_prefill_size is not None and chunked_prefill_size > 0
                ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in ngram_modules:
                module.init_buffers(max_running_requests, chunked_prefill_size, device)
            ngram_embedding_n = model_config.ngram_embedding_n
            ngram_embedding_k = model_config.ngram_embedding_k
        return cls(
            enabled=use_ngram_embedding,
            table=token_table,
            n=ngram_embedding_n,
            k=ngram_embedding_k,
        )

    def update_after_decode(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        update_ngram_token_table_after_sampling(
            ngram_embedding_info=ngram_embedding_info,
            next_token_ids=next_token_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            batch_size=forward_batch.batch_size,
        )

    def prepare_for_forward(
        self,
        batch: Optional[ScheduleBatch],
        *,
        chunked_req: Optional[Req],
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.enabled:
            return batch
        self._init_disaggregation_decode_rows(batch)
        batch.ne_token_table = self.table
        # This mask is valid only for the current forward pass. Rebuild it
        # below when the current batch contains an unfinished chunked request.
        batch.ne_skip_token_table_update = None
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            token_offsets = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_range.length
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.n + 1 : end]
                    column_starts.append(start - self.n + 1)
                token_offsets.append(len(all_tokens))
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.table.dtype
            device = self.table.device
            tokens_tensor = torch.tensor(all_tokens, dtype=dtype, device=device)
            column_starts_tensor = torch.tensor(
                column_starts, dtype=torch.int32, device=device
            )
            req_lens_tensor = torch.tensor(
                request_lengths, dtype=torch.int32, device=device
            )
            if is_npu():
                from sglang.srt.layers.welmv4_npu_op import (
                    welmv4_token_table_ragged_update_npu,
                )

                welmv4_token_table_ragged_update_npu(
                    self.table,
                    tokens_tensor,
                    batch.req_pool_indices,
                    torch.tensor(token_offsets, dtype=torch.int32, device=device),
                    column_starts_tensor,
                    req_lens_tensor,
                    max_req_len=max(request_lengths, default=0),
                )
            else:
                _update_token_table(
                    ne_token_table=self.table,
                    tokens=tokens_tensor,
                    row_indices=batch.req_pool_indices,
                    column_starts=column_starts_tensor,
                    req_lens=req_lens_tensor,
                    ignore_tokens=None,
                )
            # Mark the chunked (not-yet-finished) prefill request so sample()
            # skips writing its pseudo next-token into the ngram token table.
            # Use self.chunked_req identity (not req.is_chunked) to avoid
            # overlap-scheduling timing issues.
            if chunked_req is not None:
                skip_token_table_update = [req is chunked_req for req in batch.reqs]
                batch.ne_skip_token_table_update = (
                    torch.tensor(
                        skip_token_table_update, dtype=torch.bool, device=device
                    )
                    if any(skip_token_table_update)
                    else None
                )
        return batch

    def _init_disaggregation_decode_rows(self, batch: ScheduleBatch) -> None:
        """Rebuild newly admitted PD-decode rows from request token history.

        The prefill node transfers KV cache pages and the sampled handoff token.
        The decode request already owns the complete logical history, so copying
        the whole request-scoped token-table row over RDMA is unnecessary.
        """
        reqs = [
            req
            for req in batch.reqs
            if getattr(req, "ngram_token_table_needs_init", False)
        ]
        if not reqs:
            return

        all_tokens = []
        token_offsets = []
        request_lengths = []
        row_indices = []
        table_width = self.table.shape[1]

        for req in reqs:
            if req.req_pool_idx is None:
                raise RuntimeError(
                    "Cannot initialize a PD-decode ngram row before allocating "
                    f"req_pool_idx for request {req.rid}."
                )
            tokens = list(req.origin_input_ids)
            tokens.extend(req.output_ids)
            if len(tokens) > table_width:
                raise RuntimeError(
                    "PD-decode ngram history exceeds the token-table width: "
                    f"request={req.rid}, history={len(tokens)}, width={table_width}."
                )
            token_offsets.append(len(all_tokens))
            request_lengths.append(len(tokens))
            row_indices.append(req.req_pool_idx)
            all_tokens.extend(tokens)

        dtype = self.table.dtype
        device = self.table.device
        tokens_tensor = torch.tensor(all_tokens, dtype=dtype, device=device)
        row_indices_tensor = torch.tensor(
            row_indices, dtype=torch.int64, device=device
        )
        column_starts = torch.zeros(len(reqs), dtype=torch.int32, device=device)
        req_lens = torch.tensor(request_lengths, dtype=torch.int32, device=device)

        if is_npu():
            from sglang.srt.layers.welmv4_npu_op import (
                welmv4_token_table_ragged_update_npu,
            )

            welmv4_token_table_ragged_update_npu(
                self.table,
                tokens_tensor,
                row_indices_tensor,
                torch.tensor(token_offsets, dtype=torch.int32, device=device),
                column_starts,
                req_lens,
                max_req_len=max(request_lengths, default=0),
            )
        else:
            _update_token_table(
                ne_token_table=self.table,
                tokens=tokens_tensor,
                row_indices=row_indices_tensor,
                column_starts=column_starts,
                req_lens=req_lens,
                ignore_tokens=None,
            )

        for req in reqs:
            req.ngram_token_table_needs_init = False


def update_ngram_token_table_after_sampling(
    *,
    ngram_embedding_info,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
) -> bool:
    """Update the ngram token table with sampled tokens.

    Returns whether the token table was updated.
    """
    skip_token_table_update = ngram_embedding_info.skip_token_table_update
    if is_npu():
        from sglang.srt.layers.welmv4_npu_op import (
            welmv4_token_table_decode_update_npu,
        )

        # The kernel consumes the real batch size directly, so CUDA-graph-style
        # padded metadata never creates writes for dummy requests. A present
        # skip mask suppresses unfinished chunked-prefill pseudo samples.
        welmv4_token_table_decode_update_npu(
            ngram_embedding_info.token_table,
            next_token_ids,
            req_pool_indices,
            seq_lens,
            skip_token_table_update,
            batch_size=batch_size,
        )
        # This return value is not used by the runtime. On NPU it denotes that
        # the (possibly all-masked) device update was submitted without adding
        # a synchronizing mask reduction to the hot path.
        return batch_size > 0

    if skip_token_table_update is not None:
        # Skip chunked (not-yet-finished) prefill requests: their sampled token
        # is a pseudo prediction and must not pollute the token table.
        indices = (~skip_token_table_update).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return False
        _update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids[indices].to(torch.int32),
            row_indices=req_pool_indices[indices],
            column_starts=seq_lens[indices].to(torch.int32),
            req_lens=torch.ones(
                indices.numel(), dtype=torch.int32, device=next_token_ids.device
            ),
            ignore_tokens=None,
        )
        return True

    # NGRAM_BS_FIX: seq_lens / next_token_ids / req_pool_indices may be padded to the
    # cuda-graph batch size while batch_size is the real request count. Slice to
    # batch_size so padded rows don't pollute the token table (and shapes match).
    ngram_embedding_info.out_column_starts[:batch_size] = seq_lens[:batch_size]
    ngram_embedding_info.out_req_lens[:batch_size] = 1
    _update_token_table(
        ne_token_table=ngram_embedding_info.token_table,
        tokens=next_token_ids[:batch_size].to(torch.int32),
        row_indices=req_pool_indices[:batch_size],
        column_starts=ngram_embedding_info.out_column_starts[:batch_size],
        req_lens=ngram_embedding_info.out_req_lens[:batch_size],
        ignore_tokens=None,
    )
    return True


def _update_token_table(
    *,
    ne_token_table: torch.Tensor,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    ignore_tokens: Optional[torch.Tensor],
) -> None:
    """Update the request token table with a device-native NPU fallback.

    The upstream fast path is a CUDA JIT extension. Ascend cannot load that
    extension, while the operation itself is only a ragged indexed copy. Keep
    CUDA on the optimized kernel and express the NPU path with ordinary torch
    indexing so torch-npu can lower it to native device operators.
    """
    if not is_npu():
        update_token_table(
            ne_token_table=ne_token_table,
            tokens=tokens,
            row_indices=row_indices,
            column_starts=column_starts,
            req_lens=req_lens,
            ignore_tokens=ignore_tokens,
        )
        return

    num_tokens = tokens.numel()
    if num_tokens == 0:
        return
    req_lens = req_lens.to(dtype=torch.long)
    rows = torch.repeat_interleave(
        row_indices.to(dtype=torch.long), req_lens, output_size=num_tokens
    )
    request_starts = torch.cumsum(req_lens, dim=0) - req_lens
    flat_starts = torch.repeat_interleave(
        request_starts, req_lens, output_size=num_tokens
    )
    offsets = torch.arange(num_tokens, device=tokens.device) - flat_starts
    columns = torch.repeat_interleave(
        column_starts.to(dtype=torch.long), req_lens, output_size=num_tokens
    ) + offsets
    values = tokens.to(dtype=ne_token_table.dtype)
    if ignore_tokens is not None and ignore_tokens.numel() > 0:
        # ``torch.isin`` has varied across torch-npu releases.  The ignored
        # token set is tiny, so express it with basic equality/reduction ops
        # that are available on all supported Ascend stacks.
        ignored = (
            values.unsqueeze(-1)
            == ignore_tokens.to(device=values.device, dtype=values.dtype).reshape(
                1, -1
            )
        ).any(dim=-1)
        values = torch.where(ignored, -values, values)
    ne_token_table[rows, columns] = values
