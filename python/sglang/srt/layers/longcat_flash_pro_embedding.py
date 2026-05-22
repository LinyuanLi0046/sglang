import torch
from torch import nn

try:
    import sgl_kernel_npu  # noqa: F401
except (ImportError, OSError):
    sgl_kernel_npu = None

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.utils import get_bool_env_var
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, NgramEmbeddingInfo

class LongcatFlashProEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        self.over_embedding_m = config.oe_vocab_base
        self.over_embedding_k = config.oe_split_num
        self.over_embedding_n = config.oe_neighbor_num
        self.eos_token_id = config.eos_token_id
        self.n_grams = (self.over_embedding_n - 1) * self.over_embedding_k
        self.oe_hidden_dim = config.oe_hidden_dim
        self.scale = 1 + self.n_grams
        self.use_compute_n_gram_ids = get_bool_env_var(
            "SGLANG_NPU_LONGCATPRO_USE_COMPUTE_N_GRAM_IDS", "true"
        )

        self.word_embeder = VocabParallelEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

        exclusive_sums = torch.zeros(self.n_grams + 1, dtype=torch.int32)
        for i in range(self.n_grams):
            exclusive_sums[i + 1] = exclusive_sums[i] + int(
                self.over_embedding_m + i * 2 + 1
            )
        self.register_buffer(
            "exclusive_oe_embedder_size_sums", exclusive_sums, persistent=False
        )

        self.oe_embeder = VocabParallelEmbedding(
            num_embeddings=int(exclusive_sums[-1].tolist()),
            embedding_dim=self.oe_hidden_dim,
            use_attn_tp_group=is_dp_attention_enabled(),
        )
        self.oe_projection = nn.Parameter(
            torch.empty(self.n_grams, self.oe_hidden_dim, self.embedding_dim),
            requires_grad=False,
        )

        oe_mods = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32
        )
        oe_weights = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n],
            dtype=torch.int32,
        )
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = self.over_embedding_m + 2 * (
                    (n - 2) * self.over_embedding_k + k
                ) + 1
                oe_mods[n - 2][k] = mod
                for delta in range(self.over_embedding_n):
                    oe_weights[n - 2][k][delta] = pow(self.num_embeddings, delta, mod)
        self.register_buffer("oe_mods", oe_mods, persistent=False)
        self.register_buffer("oe_weights", oe_weights, persistent=False)

    def _can_use_compute_n_gram_ids(self) -> bool:
        return hasattr(torch.ops, "npu") and hasattr(
            torch.ops.npu, "compute_n_gram_ids"
        )

    def init_buffers(
        self, max_running_requests: int, chunked_prefill_size: int, device: str
    ):
        return

    def process_weights_after_loading(self):
        self.oe_projection.data = self.oe_projection.data / self.scale
        if is_dp_attention_enabled():
            if (
                not self.word_embeder.enable_tp
                or not self.word_embeder.use_attn_tp_group
            ):
                raise AssertionError(
                    "LongCatPro word embedding must use attention TP group "
                    "under dp-attention."
                )
            if not self.oe_embeder.enable_tp or not self.oe_embeder.use_attn_tp_group:
                raise AssertionError(
                    "LongCatPro OE embedding must use attention TP group "
                    "under dp-attention."
                )
            if self.word_embeder.tp_size <= 1 or self.oe_embeder.tp_size <= 1:
                raise AssertionError("Invalid embedding TP size under dp-attention.")

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        result = torch.zeros_like(tensor)
        mask = tensor == self.eos_token_id
        indices = mask.nonzero(as_tuple=False).flatten()
        prev_idx = 0
        for end_idx in indices.tolist():
            end = int(end_idx) + 1
            if end - prev_idx > n:
                result[prev_idx + n : end] = tensor[prev_idx : end - n]
            prev_idx = end
        if prev_idx < tensor.shape[0] and tensor.shape[0] - prev_idx > n:
            result[prev_idx + n :] = tensor[prev_idx : tensor.shape[0] - n]
        return result

    def _compute_fused_ngram_ids_torch(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        info = forward_batch.ngram_embedding_info
        if info is None:
            raise ValueError("LongcatFlashProEmbedding requires ngram_embedding_info.")

        batch_size, req_pool_indices, column_starts, req_lens = (
            self._build_padded_ngram_metadata(input_ids, forward_batch, info)
        )
        total_tokens = int(input_ids.numel())
        ngram_ids = torch.empty(
            total_tokens, self.n_grams, dtype=torch.int64, device=input_ids.device
        )
        cursor = 0
        req_pool_indices_list = req_pool_indices.tolist()
        column_starts_list = column_starts.tolist()
        req_lens_list = req_lens.tolist()
        for batch_idx in range(batch_size):
            row_idx = int(req_pool_indices_list[batch_idx])
            column_start = int(column_starts_list[batch_idx])
            req_len = int(req_lens_list[batch_idx])
            seq_end = column_start + req_len
            full_seq = info.token_table[row_idx, :seq_end].to(
                device=input_ids.device, dtype=torch.int64
            )
            full_seq[column_start:seq_end] = input_ids[cursor : cursor + req_len].to(
                torch.int64
            )

            shifted = {
                delta: self._shift_right_ignore_eos(full_seq, delta)
                for delta in range(1, self.over_embedding_n)
            }
            branch_idx = 0
            for n in range(2, self.over_embedding_n + 1):
                for k in range(self.over_embedding_k):
                    mod = self.oe_mods[n - 2, k].to(device=input_ids.device)
                    branch_ids = full_seq.clone()
                    for delta in range(1, n):
                        branch_ids += shifted[delta] * self.oe_weights[
                            n - 2, k, delta
                        ].to(device=input_ids.device)
                    branch_ids = branch_ids.remainder(mod)
                    branch_ids += self.exclusive_oe_embedder_size_sums[branch_idx].to(
                        device=input_ids.device, dtype=torch.int64
                    )
                    ngram_ids[cursor : cursor + req_len, branch_idx] = branch_ids[
                        column_start:seq_end
                    ]
                    branch_idx += 1
            cursor += req_len
        return ngram_ids

    def _compute_fused_ngram_ids_npu(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        info = forward_batch.ngram_embedding_info
        if info is None:
            raise ValueError("LongcatFlashProEmbedding requires ngram_embedding_info.")

        batch_size, req_pool_indices, column_starts, req_lens = (
            self._build_padded_ngram_metadata(input_ids, forward_batch, info)
        )

        total_tokens = self._validate_ngram_inputs_npu(
            input_ids,
            batch_size,
            req_pool_indices,
            column_starts,
            req_lens,
            info.token_table,
        )

        return torch.ops.npu.compute_n_gram_ids(
            self.oe_weights.to(device=input_ids.device, dtype=torch.int32),
            self.oe_mods.to(device=input_ids.device, dtype=torch.int32),
            self.exclusive_oe_embedder_size_sums.to(
                device=input_ids.device, dtype=torch.int32
            ),
            input_ids[:total_tokens].to(torch.int32),
            torch.cumsum(req_lens, dim=0, dtype=torch.int32),
            info.token_table.to(device=input_ids.device, dtype=torch.int32),
            req_pool_indices.to(device=input_ids.device, dtype=torch.int64),
            column_starts.to(device=input_ids.device, dtype=torch.int32),
            batch_size=batch_size,
            oe_n=self.over_embedding_n,
            oe_k=self.over_embedding_k,
            max_context_len=info.token_table.shape[1],
        )

    def _build_padded_ngram_metadata(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        info: NgramEmbeddingInfo,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(forward_batch.batch_size)
        if batch_size < 0:
            raise ValueError(f"Invalid batch_size: {batch_size}.")

        total_input_tokens = int(input_ids.numel())
        if batch_size == 0:
            if total_input_tokens != 0:
                raise ValueError(
                    "compute_n_gram_ids contract violated: batch_size is 0 but "
                    f"input_ids has {total_input_tokens} tokens."
                )
            empty_int32 = info.req_lens.new_zeros((0,))
            empty_int64 = forward_batch.req_pool_indices.new_zeros((0,))
            return 0, empty_int64, empty_int32, empty_int32

        real_bs = int(info.req_lens.numel())
        if real_bs > batch_size:
            raise ValueError(
                "compute_n_gram_ids contract violated: metadata batch_size "
                f"{real_bs} exceeds execution batch_size {batch_size}."
            )
        if forward_batch.req_pool_indices.numel() < real_bs:
            raise ValueError(
                "compute_n_gram_ids contract violated: req_pool_indices has "
                f"{forward_batch.req_pool_indices.numel()} entries, expected at least "
                f"{real_bs}."
            )

        req_pool_indices = forward_batch.req_pool_indices[:real_bs].clone()
        column_starts = info.column_starts.clone()
        req_lens = info.req_lens.clone()

        if req_pool_indices.numel() < batch_size:
            req_pool_indices = torch.cat(
                [
                    req_pool_indices,
                    req_pool_indices.new_zeros((batch_size - req_pool_indices.numel(),)),
                ],
                dim=0,
            )
        else:
            req_pool_indices = req_pool_indices[:batch_size]

        if column_starts.numel() < batch_size:
            column_starts = torch.cat(
                [
                    column_starts,
                    column_starts.new_zeros((batch_size - column_starts.numel(),)),
                ],
                dim=0,
            )
        else:
            column_starts = column_starts[:batch_size]

        if req_lens.numel() < batch_size:
            req_lens = torch.cat(
                [req_lens, req_lens.new_zeros((batch_size - req_lens.numel(),))], dim=0
            )
        else:
            req_lens = req_lens[:batch_size]

        total_req_tokens = int(req_lens.sum().tolist())
        if total_req_tokens > total_input_tokens:
            raise ValueError(
                "compute_n_gram_ids contract violated: "
                f"sum(req_lens)={total_req_tokens} exceeds "
                f"input_ids.numel()={total_input_tokens}."
            )

        remaining_tokens = total_input_tokens - total_req_tokens
        if remaining_tokens == 0:
            return batch_size, req_pool_indices, column_starts, req_lens

        max_context_len = info.token_table.shape[1]
        fill_order = list(range(real_bs, batch_size)) + list(range(real_bs))
        for row_idx in fill_order:
            if remaining_tokens == 0:
                break
            capacity = max_context_len - column_starts[row_idx] - req_lens[row_idx]
            if torch.is_nonzero(capacity <= 0):
                continue
            add = torch.minimum(
                capacity, req_lens.new_tensor(remaining_tokens, dtype=req_lens.dtype)
            )
            add_int = int(add.tolist())
            req_lens[row_idx] += add
            remaining_tokens -= add_int

        if remaining_tokens != 0:
            raise ValueError(
                "compute_n_gram_ids contract violated: unable to pad metadata to "
                f"cover {total_input_tokens} input tokens with max_context_len="
                f"{max_context_len}."
            )

        return batch_size, req_pool_indices, column_starts, req_lens

    def _validate_ngram_inputs_npu(
        self,
        input_ids: torch.Tensor,
        batch_size: int,
        req_pool_indices: torch.Tensor,
        column_starts: torch.Tensor,
        req_lens: torch.Tensor,
        token_table: torch.Tensor,
    ) -> int:
        if batch_size < 0:
            raise ValueError(f"Invalid batch_size: {batch_size}.")
        if column_starts.numel() != batch_size:
            raise ValueError(
                "compute_n_gram_ids contract violated: "
                f"column_starts has {column_starts.numel()} entries, "
                f"but execution batch_size is {batch_size}."
            )
        if req_lens.numel() != batch_size:
            raise ValueError(
                "compute_n_gram_ids contract violated: "
                f"req_lens has {req_lens.numel()} entries, "
                f"but execution batch_size is {batch_size}."
            )
        if req_pool_indices.numel() != batch_size:
            raise ValueError(
                "compute_n_gram_ids contract violated: "
                f"req_pool_indices has {req_pool_indices.numel()} entries, "
                f"but execution batch_size is {batch_size}."
            )
        if batch_size == 0:
            if input_ids.numel() != 0:
                raise ValueError(
                    "compute_n_gram_ids contract violated: execution batch_size is 0 but "
                    f"input_ids has {input_ids.numel()} tokens."
                )
            return 0

        total_req_tokens = int(req_lens.sum().tolist())
        total_input_tokens = int(input_ids.numel())
        if total_req_tokens != total_input_tokens:
            raise ValueError(
                "compute_n_gram_ids contract violated: "
                f"sum(req_lens)={total_req_tokens}, "
                f"input_ids.numel()={total_input_tokens}."
            )

        max_context_len = token_table.shape[1]
        if torch.is_nonzero(torch.min(req_lens) < 0):
            raise ValueError("compute_n_gram_ids contract violated: req_lens < 0.")
        if torch.is_nonzero(torch.min(column_starts) < 0):
            raise ValueError(
                "compute_n_gram_ids contract violated: column_starts < 0."
            )
        if torch.is_nonzero(torch.max(column_starts + req_lens) > max_context_len):
            raise ValueError(
                "compute_n_gram_ids contract violated: column_starts + req_lens "
                f"exceeds token_table width {max_context_len}."
            )
        if torch.is_nonzero(torch.min(req_pool_indices) < 0):
            raise ValueError(
                "compute_n_gram_ids contract violated: req_pool_indices < 0."
            )
        if torch.is_nonzero(torch.max(req_pool_indices) >= token_table.shape[0]):
            raise ValueError(
                "compute_n_gram_ids contract violated: req_pool_indices exceeds "
                "token_table row count."
            )
        return total_input_tokens

    def _compute_fused_ngram_ids(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if (
            self.use_compute_n_gram_ids
            and input_ids.device.type == "npu"
            and self._can_use_compute_n_gram_ids()
        ):
            return self._compute_fused_ngram_ids_npu(input_ids, forward_batch)
        return self._compute_fused_ngram_ids_torch(input_ids, forward_batch)

    def _load_oe_embedder_weight(self, index: int, loaded_weight: torch.Tensor):
        oe_weight_start = int(self.exclusive_oe_embedder_size_sums[index].tolist())
        oe_weight_end = int(self.exclusive_oe_embedder_size_sums[index + 1].tolist())
        expected_rows = oe_weight_end - oe_weight_start
        if loaded_weight.shape[0] < expected_rows:
            raise ValueError(
                f"oe_embed_tokens{index} has too few rows: "
                f"expected at least {expected_rows}, got {loaded_weight.shape[0]}"
            )
        if loaded_weight.shape[0] != expected_rows:
            loaded_weight = loaded_weight[:expected_rows]

        tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
        tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
        if tp_end <= tp_start:
            raise AssertionError("Invalid OE shard range.")
        to_load_start = max(oe_weight_start, tp_start)
        to_load_end = min(oe_weight_end, tp_end)
        if to_load_start >= to_load_end:
            return

        src_start = to_load_start - oe_weight_start
        src_end = to_load_end - oe_weight_start
        dest_start = to_load_start - tp_start
        dest_end = to_load_end - tp_start
        if dest_end - dest_start != src_end - src_start:
            raise AssertionError("OE shard copy size mismatch.")
        self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[
            src_start:src_end
        ]

    def load_weight(self, param, weight_name: str, loaded_weight: torch.Tensor):
        if weight_name in (
            "model.embed_tokens.weight",
            "model.embed_tokens.word_embeder.weight",
        ) or ".embed_tokens." in weight_name:
            self.word_embeder.weight_loader(self.word_embeder.weight, loaded_weight)
        elif (
            weight_name.startswith("model.oe_embed_tokens")
            or "model.ngram_embeddings.embedders." in weight_name
        ):
            if weight_name.startswith("model.oe_embed_tokens"):
                index = int(
                    weight_name.replace("model.oe_embed_tokens", "").replace(
                        ".weight", ""
                    )
                )
            else:
                index = int(
                    weight_name.replace("model.ngram_embeddings.embedders.", "").replace(
                        ".weight", ""
                    )
                )
            self._load_oe_embedder_weight(index, loaded_weight)
        elif (
            weight_name.startswith("model.oe_embed_proj")
            or "model.ngram_embeddings.post_projs." in weight_name
        ):
            if weight_name.startswith("model.oe_embed_proj"):
                index = int(
                    weight_name.replace("model.oe_embed_proj", "").replace(
                        ".weight", ""
                    )
                )
            else:
                index = int(
                    weight_name.replace("model.ngram_embeddings.post_projs.", "").replace(
                        ".weight", ""
                    )
                )
            self.oe_projection[index].copy_(loaded_weight.data.t())
        else:
            raise ValueError(f"Unknown LongcatFlashPro embedding weight: {weight_name}")

    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch):
        hidden_states = self.word_embeder(input_ids).to(self.oe_projection.dtype)
        if self.n_grams == 0:
            return hidden_states

        oe_n_gram_ids = self._compute_fused_ngram_ids(input_ids, forward_batch)
        oe_hidden_states = self.oe_embeder(
            oe_n_gram_ids.permute(1, 0).contiguous()
        ).to(self.oe_projection.dtype)
        real_projected = torch.bmm(oe_hidden_states, self.oe_projection).sum(dim=0)
        projected = torch.zeros_like(hidden_states)
        if real_projected.shape[0] > 0:
            projected[: real_projected.shape[0]] = real_projected
        return hidden_states / self.scale + projected
