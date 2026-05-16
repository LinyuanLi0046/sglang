from __future__ import annotations

from typing import Any, Dict, Optional

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    Indexer,
    _use_ag_after_qlora,
    scattered_to_tp_attn_full,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix


class LongcatProNPUIndexer(Indexer):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            layer_id=layer_id,
            scale_fmt=scale_fmt,
            block_size=block_size,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            prefix=prefix,
            quant_config=quant_config,
            alt_stream=alt_stream,
        )
        self.config = config
        self.index_k_norm_type = index_k_norm_type
        self.kv_block_size = getattr(config, "kv_block_size", 1)
        self.q_block_size = getattr(config, "q_block_size", 1)
        self.num_init_tokens = getattr(config, "index_init_tokens", 0)
        self.num_local_tokens = getattr(config, "index_local_tokens", 0)
        self.nsa_enable_prefill_cp = False

        if index_k_norm_type == "rms":
            self.k_norm = RMSNorm(
                self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
            )

        # LongcatPro's reference NPU path keeps indexer routing weights in fp32.
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )

    def _build_token_seq_id(self, actual_seq_lengths_q: torch.Tensor):
        q_lens = actual_seq_lengths_q.to(torch.int64).clone()
        if q_lens.numel() > 1:
            q_lens[1:] = q_lens[1:] - actual_seq_lengths_q[:-1].to(torch.int64)
        q_start = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=actual_seq_lengths_q.device),
                actual_seq_lengths_q[:-1].to(torch.int64),
            ]
        )
        total_tokens = int(actual_seq_lengths_q[-1].item()) if q_lens.numel() > 0 else 0
        seq_id = torch.zeros(
            total_tokens, dtype=torch.int32, device=actual_seq_lengths_q.device
        )
        if total_tokens > 0:
            seq_id.scatter_(0, q_start, torch.ones_like(q_start, dtype=torch.int32))
            seq_id = seq_id.cumsum(0) - 1
        return seq_id, q_lens, q_start

    def _compose_longcat_topk(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        kv_pos: torch.Tensor,
        kv_init_end: torch.Tensor,
        kv_local_start: torch.Tensor,
    ) -> torch.Tensor:
        candidate_slots = torch.arange(
            val.shape[1], dtype=torch.int64, device=val.device
        ).unsqueeze(0)
        val = val.masked_fill(candidate_slots >= kv_pos.unsqueeze(1), float("-inf"))

        valid_idx = (idx >= 0) & (idx < kv_pos.unsqueeze(1))
        val = val.masked_fill(~valid_idx, float("-inf"))

        forced = (
            (idx < kv_init_end.unsqueeze(1)) | (idx >= kv_local_start.unsqueeze(1))
        ) & valid_idx
        val = val.masked_fill(forced, float("-inf"))

        sparse_topk = max(
            self.index_topk - self.num_init_tokens - self.num_local_tokens,
            0,
        )
        if sparse_topk > 0:
            sparse_val, sparse_sel = val.topk(sparse_topk, dim=1)
            sparse_idx = torch.gather(idx, dim=1, index=sparse_sel)
            sparse_idx = sparse_idx.masked_fill(torch.isneginf(sparse_val), -1)
        else:
            sparse_idx = torch.empty(
                idx.shape[0], 0, dtype=idx.dtype, device=idx.device
            )

        base_init = torch.arange(
            self.num_init_tokens, dtype=idx.dtype, device=idx.device
        ).unsqueeze(0)
        base_init = base_init.expand(idx.shape[0], -1)
        init_res = torch.where(
            base_init < kv_init_end.unsqueeze(1),
            base_init,
            torch.full_like(base_init, -1),
        )

        offset = torch.arange(
            self.num_local_tokens, dtype=idx.dtype, device=idx.device
        ).unsqueeze(0)
        offset = offset.expand(idx.shape[0], -1)
        local_vals = kv_local_start.unsqueeze(1) + offset
        local_lens = kv_pos - kv_local_start
        local_res = torch.where(
            offset < local_lens.unsqueeze(1),
            local_vals,
            torch.full_like(local_vals, -1),
        )

        topk_indices = torch.cat([init_res, local_res, sparse_idx], dim=1)
        mask_res = (topk_indices == -1).to(torch.float32)
        _, order = torch.sort(mask_res, dim=1)
        topk_indices = topk_indices.gather(1, order)
        return topk_indices.to(torch.int32).unsqueeze(1)

    def _get_full_candidate_count(self, actual_seq_lengths_kv: torch.Tensor) -> int:
        if actual_seq_lengths_kv.numel() == 0:
            return self.index_topk
        return max(int(actual_seq_lengths_kv.max().item()), self.index_topk)

    def _logical_indices_to_pa_slots(
        self,
        topk_indices_local: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = topk_indices_local.squeeze(1).to(torch.int64)
        if idx.numel() == 0:
            return idx, torch.zeros_like(idx, dtype=torch.bool)

        seq_id, _, _ = self._build_token_seq_id(actual_seq_lengths_q)
        kv_lens = actual_seq_lengths_kv.to(torch.int64)
        valid_mask = (idx >= 0) & (idx < kv_lens[seq_id].unsqueeze(1))

        safe_idx = idx.clamp(min=0)
        page_idx = torch.div(safe_idx, page_size, rounding_mode="floor")
        page_offset = safe_idx.remainder(page_size)
        req_block_table = block_table.index_select(0, seq_id.to(torch.int64))
        physical_page = req_block_table.gather(1, page_idx)
        slots = physical_page.to(torch.int64) * page_size + page_offset
        return slots, valid_mask

    def _gather_candidate_keys_pa_bsnd(
        self,
        past_key_states: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        flat_key_states = past_key_states.reshape(-1, self.head_dim)
        candidate_k = flat_key_states.index_select(0, slots.reshape(-1).to(torch.int64))
        return candidate_k.view(slots.shape[0], slots.shape[1], self.head_dim)

    def _recompute_candidate_values_pa_bsnd(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        past_key_states: torch.Tensor,
        topk_indices_local: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        page_size: int,
        candidate_chunk_size: int = 128,
    ) -> torch.Tensor:
        idx = topk_indices_local.squeeze(1)
        token_num = idx.shape[0]
        candidate_num = idx.shape[1] if idx.dim() == 2 else 0
        if token_num == 0 or candidate_num == 0:
            return torch.empty(
                token_num, 1, candidate_num, dtype=torch.float32, device=q.device
            )

        slots, valid_mask = self._logical_indices_to_pa_slots(
            topk_indices_local=topk_indices_local,
            block_table=block_table,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            page_size=page_size,
        )

        q_fp32 = q.to(torch.float32)
        weights_fp32 = weights.view(token_num, self.n_heads).to(torch.float32)
        topk_values = torch.empty(
            token_num, candidate_num, dtype=torch.float32, device=q.device
        )

        for start in range(0, candidate_num, candidate_chunk_size):
            end = min(start + candidate_chunk_size, candidate_num)
            candidate_k = self._gather_candidate_keys_pa_bsnd(
                past_key_states=past_key_states,
                slots=slots[:, start:end],
            ).to(torch.float32)
            logits = torch.einsum("thd,tcd->thc", q_fp32, candidate_k)
            logits = torch.relu(logits)
            topk_values[:, start:end] = (
                logits * weights_fp32.unsqueeze(-1)
            ).sum(dim=1)

        topk_values.masked_fill_(~valid_mask, float("-inf"))
        return topk_values.unsqueeze(1)

    def _postprocess_longcat_topk_single_rank(
        self,
        topk_indices_local: torch.Tensor,
        topk_values: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        idx = topk_indices_local.squeeze(1).to(torch.int64)
        val = topk_values.squeeze(1)
        seq_id, q_lens, q_start = self._build_token_seq_id(actual_seq_lengths_q)
        kv_lens = actual_seq_lengths_kv.to(torch.int64)

        if idx.shape[0] == 0:
            return idx.to(torch.int32).unsqueeze(1)

        if is_prefill:
            local_pos = torch.arange(idx.shape[0], device=idx.device) - q_start[seq_id]
            kv_pos = kv_lens[seq_id] - q_lens[seq_id] + local_pos + 1
        else:
            kv_pos = kv_lens[seq_id]

        kv_init_end = torch.minimum(
            kv_pos,
            torch.full_like(kv_pos, self.num_init_tokens),
        )
        local_keep = torch.minimum(
            torch.clamp(kv_pos - kv_init_end, min=0),
            torch.full_like(kv_pos, self.num_local_tokens),
        )
        kv_local_start = kv_pos - local_keep

        return self._compose_longcat_topk(
            idx=idx,
            val=val,
            kv_pos=kv_pos,
            kv_init_end=kv_init_end,
            kv_local_start=kv_local_start,
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if torch_npu is None:
            raise RuntimeError("LongcatProNPUIndexer requires torch_npu")

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )
        bs = q_lora.shape[0]

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[0]
                    q = q.view(bs, self.n_heads, self.head_dim)
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                q = self.wq_b(q_lora)[0]
                q = q.view(bs, self.n_heads, self.head_dim)
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )
                q = torch.cat([q_pe, q_nope], dim=-1)
                q_rope_event = None

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0]
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0]
                weights_event = None

            k_proj = self.wk(x)[0]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)
        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0]
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0]
                weights_event = None

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]
            q = q.view(bs, self.n_heads, self.head_dim)
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )

            k_proj = self.wk(x)[0]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            k_pe = k_pe.unsqueeze(1)

            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)
            q_rope_event = None

        forward_batch.token_to_kv_pool.set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )

        if is_prefill:
            actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0).to(
                device=k.device, dtype=torch.int32
            )
        else:
            actual_seq_lengths_q = torch.arange(
                1,
                bs + 1,
                dtype=torch.int32,
                device=k.device,
            )
        actual_seq_lengths_kv = forward_batch.seq_lens.to(
            device=k.device, dtype=torch.int32
        )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and q_rope_event is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if weights_event is not None:
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)

        block_table = forward_batch.attn_backend.forward_metadata.block_tables
        if is_prefill:
            block_table = block_table[: actual_seq_lengths_q.size(0)]
        candidate_count = self._get_full_candidate_count(actual_seq_lengths_kv)

        # DeepSeek V3.2 路径只要 topk_indices,不需要 topk_values
        # LongCat Pro 路径要 topk_indices + topk_values 所以传了 return_value=True
        # 但 npu_lightning_indexer 算子明确不支持 layout_key="PA_BSND" 同时 return_value=True
        topk_indices_local = torch_npu.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights.to(torch.bfloat16),
            actual_seq_lengths_query=actual_seq_lengths_q,
            actual_seq_lengths_key=actual_seq_lengths_kv,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=candidate_count,
            sparse_mode=3,
        )[0]
        topk_values = self._recompute_candidate_values_pa_bsnd(
            q=q.view(-1, self.n_heads, self.head_dim),
            weights=weights,
            past_key_states=past_key_states,
            topk_indices_local=topk_indices_local,
            block_table=block_table,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            page_size=forward_batch.token_to_kv_pool.page_size,
        )
        return self._postprocess_longcat_topk_single_rank(
            topk_indices_local=topk_indices_local,
            topk_values=topk_values,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            is_prefill=is_prefill,
        )
