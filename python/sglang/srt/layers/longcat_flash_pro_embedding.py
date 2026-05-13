import torch
from torch import nn

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.n_gram_embedding import NgramEmbedding
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LongcatFlashProEmbedding(NgramEmbedding):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        self.over_embedding_m = config.ngram_embedding_m
        self.over_embedding_k = config.ngram_embedding_k
        self.over_embedding_n = config.ngram_embedding_n
        self.eos_token_id = config.eos_token_id
        self.n_grams = (self.over_embedding_n - 1) * self.over_embedding_k
        oe_hidden_dim = self.embedding_dim // self.n_grams

        self.word_embeder = VocabParallelEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            enable_tp=is_dp_attention_enabled(),
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
            embedding_dim=oe_hidden_dim,
            enable_tp=is_dp_attention_enabled(),
        )
        self.oe_projection = nn.Parameter(
            torch.empty(self.n_grams, oe_hidden_dim, self.embedding_dim),
            requires_grad=False,
        )

        oe_mods = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int64
        )
        oe_weights = torch.zeros(
            [self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n],
            dtype=torch.int64,
        )
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = self.over_embedding_m + 2 * ((n - 2) * self.over_embedding_k + k) + 1
                oe_mods[n - 2][k] = mod
                for delta in range(self.over_embedding_n):
                    oe_weights[n - 2][k][delta] = pow(self.num_embeddings, delta, mod)
        self.register_buffer("oe_mods", oe_mods, persistent=False)
        self.register_buffer("oe_weights", oe_weights, persistent=False)

    def init_buffers(self, max_running_requests: int, chunked_prefill_size: int, device: str):
        return

    def reset_decode_cache(self):
        return

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

    def _compute_ngram_ids(
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
            full_seq = info.token_table[row_idx, :seq_end].to(device=input_ids.device, dtype=torch.int64)
            full_seq[column_start:seq_end] = input_ids[cursor : cursor + req_len].to(torch.int64)

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
                        branch_ids += shifted[delta] * self.oe_weights[n - 2, k, delta].to(
                            device=input_ids.device
                        )
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
        self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[src_start:src_end]

    def load_weight(self, param, weight_name: str, loaded_weight: torch.Tensor):
        if weight_name in (
            "model.embed_tokens.weight",
            "model.embed_tokens.word_embeder.weight",
        ) or ".embed_tokens." in weight_name:
            self.word_embeder.weight_loader(self.word_embeder.weight, loaded_weight)
        elif weight_name.startswith("model.oe_embed_tokens"):
            index = int(
                weight_name.replace("model.oe_embed_tokens", "").replace(".weight", "")
            )
            self._load_oe_embedder_weight(index, loaded_weight)
        elif weight_name.startswith("model.oe_embed_proj"):
            index = int(
                weight_name.replace("model.oe_embed_proj", "").replace(".weight", "")
            )
            self.oe_projection[index].copy_(loaded_weight.data.t())
        elif "model.ngram_embeddings.embedders." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.embedders.", "").replace(
                    ".weight", ""
                )
            )
            self._load_oe_embedder_weight(index, loaded_weight)
        elif "model.ngram_embeddings.post_projs." in weight_name:
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
        ngram_ids = self._compute_ngram_ids(input_ids, forward_batch)
        oe_hidden_states = self.oe_embeder(ngram_ids.permute(1, 0).contiguous())
        projected = torch.bmm(
            oe_hidden_states.to(self.oe_projection.dtype), self.oe_projection
        ).sum(dim=0)
        return (hidden_states + projected) / (self.n_grams + 1)
