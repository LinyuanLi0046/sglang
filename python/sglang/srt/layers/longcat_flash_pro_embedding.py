import logging

import torch
from torch import nn

try:
    import sgl_kernel_npu  # noqa: F401
except (ImportError, OSError):
    sgl_kernel_npu = None

from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.utils import get_bool_env_var
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


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
        self.oe_use_global_tp_under_dp = True

        self.word_embeder = VocabParallelEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            enable_tp=not is_dp_attention_enabled(),
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
            num_embeddings=int(exclusive_sums[-1].item()),
            embedding_dim=self.oe_hidden_dim,
            # Keep OE embedding sharded on the global TP group even when
            # dp-attention is enabled, otherwise the fused OE vocabulary would
            # be fully replicated on every rank and OOM during weight loading.
            enable_tp=True,
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
            if not self.oe_use_global_tp_under_dp or not self.oe_embeder.enable_tp:
                raise AssertionError(
                    "LongCatPro OE embedding must remain globally TP-sharded "
                    "when dp-attention is enabled."
                )
            tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
            tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
            if tp_end <= tp_start:
                raise AssertionError(
                    "Invalid LongCatPro OE embedding shard range under dp-attention."
                )
            logger.info(
                "LongCatPro OE embedding keeps global TP sharding under dp-attention: "
                "rows [%d, %d) of %d, shard_rows=%d, tp_size=%d",
                tp_start,
                tp_end,
                self.oe_embeder.org_vocab_size,
                self.oe_embeder.num_embeddings_per_partition,
                self.oe_embeder.tp_size,
            )

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        result = torch.zeros_like(tensor)
        mask = tensor == self.eos_token_id
        indices = mask.nonzero(as_tuple=False).flatten()
        prev_idx = 0
        for end_idx in indices:
            end = int(end_idx.item()) + 1
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

        total_tokens = input_ids.shape[0]
        ngram_ids = torch.empty(
            total_tokens, self.n_grams, dtype=torch.int64, device=input_ids.device
        )
        cursor = 0
        for batch_idx in range(forward_batch.batch_size):
            row_idx = int(forward_batch.req_pool_indices[batch_idx].item())
            column_start = int(info.column_starts[batch_idx].item())
            req_len = int(info.req_lens[batch_idx].item())
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

        return torch.ops.npu.compute_n_gram_ids(
            self.oe_weights.to(device=input_ids.device, dtype=torch.int32),
            self.oe_mods.to(device=input_ids.device, dtype=torch.int32),
            self.exclusive_oe_embedder_size_sums.to(
                device=input_ids.device, dtype=torch.int32
            ),
            input_ids.to(torch.int32),
            torch.cumsum(info.req_lens, dim=0, dtype=torch.int32),
            info.token_table.to(device=input_ids.device, dtype=torch.int32),
            forward_batch.req_pool_indices.to(device=input_ids.device, dtype=torch.int64),
            info.column_starts.to(device=input_ids.device, dtype=torch.int32),
            batch_size=forward_batch.batch_size,
            oe_n=self.over_embedding_n,
            oe_k=self.over_embedding_k,
            max_context_len=info.token_table.shape[1],
        )

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
        oe_weight_start = int(self.exclusive_oe_embedder_size_sums[index].item())
        oe_weight_end = int(self.exclusive_oe_embedder_size_sums[index + 1].item())
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
        to_load_start = max(oe_weight_start, tp_start)
        to_load_end = min(oe_weight_end, tp_end)
        if to_load_start >= to_load_end:
            return

        src_start = to_load_start - oe_weight_start
        src_end = to_load_end - oe_weight_start
        dest_start = to_load_start - tp_start
        dest_end = to_load_end - tp_start
        if tp_end <= tp_start:
            raise AssertionError("Invalid TP shard range for LongCatPro OE embedding.")
        if src_end < src_start or dest_end < dest_start:
            raise AssertionError("Invalid LongCatPro OE embedding shard offsets.")
        if (src_end - src_start) != (dest_end - dest_start):
            raise AssertionError("Mismatched LongCatPro OE embedding shard copy size.")
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
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        if forward_batch.ngram_embedding_info is None:
            raise ValueError(
                "LongcatFlashProEmbedding requires ngram_embedding_info "
                "for non-idle forward."
            )

        oe_n_gram_ids = self._compute_fused_ngram_ids(input_ids, forward_batch)
        if self.oe_hidden_dim * self.n_grams != self.embedding_dim:
            raise AssertionError(
                "LongCatPro OE embedding dimension mismatch: "
                f"{self.oe_hidden_dim} * {self.n_grams} != {self.embedding_dim}"
            )
        if oe_n_gram_ids.shape[0] != input_ids.shape[0]:
            raise AssertionError(
                "LongCatPro OE ngram id rows must match input token count."
            )
        if is_dp_attention_enabled():
            if not self.oe_embeder.enable_tp:
                raise AssertionError(
                    "LongCatPro OE embedding must stay TP-sharded under dp-attention."
                )
            if get_attn_tp_context().input_scattered:
                raise AssertionError(
                    "LongCatPro OE embedding does not support input_scattered "
                    "together with dp-attention."
                )
        oe_hidden_states = self.oe_embeder(
            oe_n_gram_ids.permute(1, 0).contiguous()
        ).to(self.oe_projection.dtype)
        projected = torch.bmm(oe_hidden_states, self.oe_projection).sum(dim=0)
        return hidden_states / self.scale + projected
