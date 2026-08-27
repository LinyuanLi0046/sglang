# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Inference-only WeLMV4 NextN model for Spec V2 MTP."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import get_welmv4_layerwise_sliding_windows
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeModel,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
)
from sglang.srt.utils import add_prefix, is_npu


class WeLMV4ModelNextN(nn.Module):
    """One physical MTP layer backed by checkpoint/config logical layer 48."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.is_nextn_model = True
        self.start_layer = 0
        self.end_layer = 1
        self.total_num_layers = 1
        self.vocab_size = config.vocab_size
        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None

        num_physical_layers = int(config.num_hidden_layers)
        num_target_layers = int(config.num_target_hidden_layers)
        if num_physical_layers != 1:
            raise ValueError(
                "WeLMV4 NextN currently requires exactly one physical MTP "
                f"layer, got {num_physical_layers}."
            )
        logical_layer_id = num_target_layers
        mirror_layers = list(getattr(config, "kv_mirror_layers", []) or [])
        mirror_sources = list(
            getattr(config, "kv_mirror_imitated_layers", []) or []
        )
        cross_pairs = [
            (source, consumer)
            for consumer, source in zip(mirror_layers, mirror_sources)
            if int(consumer) >= num_target_layers
        ]
        if cross_pairs != [(0, logical_layer_id)]:
            raise ValueError(
                "WeLMV4 NextN requires the single cross-model KV-mirror pair "
                f"source0->consumer{logical_layer_id}, got {cross_pairs}."
            )

        # Checkpoint layer 48 owns these four modules. The base loader maps
        # model.layers.48.{enorm,hnorm,eh_proj,shared_head.norm} to model.*.
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=True
        )
        self.shared_head = nn.Module()
        # The checkpoint head norm uses WeLM's FP32-reduction implementation;
        # this also preserves the recurrent pre-norm residual for the next step.
        self.shared_head.norm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        alt_stream = torch.get_device_module().Stream() if is_npu() else None
        self.decoder_layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    config=config,
                    layer_id=0,
                    config_layer_id=logical_layer_id,
                    quant_config=quant_config,
                    is_nextn=True,
                    # Quantization rules must be selected with the checkpoint
                    # name, even though the runtime module/cache slot is local 0.
                    prefix=add_prefix(f"layers.{logical_layer_id}", prefix),
                    alt_stream=alt_stream,
                )
            ]
        )

    def _compute_oe_embedding(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return Qwen2MoeModel._compute_oe_embedding(
            self, input_ids, forward_batch, hidden_states
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.embed_tokens is None:
            raise RuntimeError(
                "WeLMV4 NextN embeddings are not bound to the target model."
            )
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if self.oe_grams and forward_batch.ngram_embedding_info is not None:
            hidden_states = self._compute_oe_embedding(
                input_ids, forward_batch, hidden_states
            )

        main_hidden_states = forward_batch.spec_info.hidden_states
        if main_hidden_states is None:
            raise RuntimeError("WeLMV4 NextN requires target hidden states.")

        # With KV-mirror query pruning, source0 still provides K/V for every
        # shifted prompt token, but the physical MTP layer evaluates only the
        # final query of each request.  Build the full token + OE embedding
        # first so the final row keeps the correct n-gram history, then contract
        # both inputs to the same B-row query layout.  ``positions`` deliberately
        # stays in the full T-row layout: RoPE uses ``custom_last_index`` for Q
        # while rotating the external source0 K rows at all T positions.
        is_kv_mirror_prefill = (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
        if is_kv_mirror_prefill:
            if forward_batch.extend_seq_lens is None:
                raise RuntimeError(
                    "WeLMV4 NextN KV-mirror prefill requires extend_seq_lens."
                )
            custom_last_index = getattr(forward_batch, "custom_last_index", None)
            if custom_last_index is None:
                custom_last_index = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                forward_batch.custom_last_index = custom_last_index

            query_rows = custom_last_index.numel()
            full_rows = input_ids.numel()
            real_batch_rows = getattr(
                forward_batch, "_original_batch_size", None
            )
            if real_batch_rows is None:
                real_batch_rows = forward_batch.batch_size
            if query_rows != real_batch_rows:
                raise RuntimeError(
                    "WeLMV4 NextN KV-mirror tail index count must match the "
                    f"real request count, got {query_rows} vs {real_batch_rows}."
                )
            last_index = custom_last_index.to(
                device=hidden_states.device, dtype=torch.long
            )

            if hidden_states.shape[0] == full_rows:
                hidden_states = hidden_states.index_select(0, last_index)
            elif hidden_states.shape[0] != query_rows:
                raise RuntimeError(
                    "WeLMV4 NextN KV-mirror token embedding must have either "
                    f"full-token or request rows, got {hidden_states.shape[0]} "
                    f"for T={full_rows}, B={query_rows}."
                )

            # MLP-sync padding saves the original draft hidden tensor before
            # expanding it to the padded token count.  For a mirror-pruned
            # target that original tensor is already B rows; selecting the
            # prompt-tail indices from its padded copy would pick dummy zeros.
            unpadded_main_hidden_states = getattr(
                forward_batch, "hidden_states_backup", None
            )
            if unpadded_main_hidden_states is not None:
                main_hidden_states = unpadded_main_hidden_states

            original_token_rows = getattr(
                forward_batch, "_original_num_tokens", None
            )
            main_hidden_has_token_rows = main_hidden_states.shape[0] == full_rows or (
                original_token_rows is not None
                and main_hidden_states.shape[0] == original_token_rows
            )
            if main_hidden_has_token_rows:
                main_hidden_states = main_hidden_states.index_select(
                    0,
                    custom_last_index.to(
                        device=main_hidden_states.device, dtype=torch.long
                    ),
                )
            elif main_hidden_states.shape[0] != query_rows:
                raise RuntimeError(
                    "WeLMV4 NextN KV-mirror target hidden states must have either "
                    f"full-token or request rows, got {main_hidden_states.shape[0]} "
                    f"for T={full_rows}, B={query_rows}."
                )

            # ``prepare_mlp_sync_batch`` localized this scalar for the old
            # T-row token layout.  The NextN MoE now receives a replicated
            # B-row last-query layout, so every one of those rows is real on
            # every EP rank.  Leaving the T-local count in place would mask
            # valid requests (and may mask all rows on a high TP/EP rank).
            if forward_batch.num_token_non_padded is not None:
                forward_batch.num_token_non_padded.fill_(query_rows)
            forward_batch.num_token_non_padded_cpu = query_rows

        # Eager MLP-sync localizes this scalar as if DRAFT_EXTEND_V2 rows were
        # sequence-sharded across attention-TP ranks.  The NextN decoder does
        # not enter WeLM's target-only scattered-prefill path, so every rank in
        # fact owns the same complete row set.  Restore the global real count
        # before routing; unlike disabling the mask entirely, eager execution
        # still keeps suffix rows added for TP alignment inactive. Graph replay
        # uses its fixed bucket width and slices dummy results afterwards.
        if (
            is_npu()
            and forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.num_token_non_padded is not None
        ):
            num_real_rows = forward_batch.num_token_non_padded_cpu
            if num_real_rows is None:
                num_real_rows = getattr(
                    forward_batch, "_original_num_tokens", None
                )
            if num_real_rows is None:
                num_real_rows = hidden_states.shape[0]
            num_real_rows = int(num_real_rows)
            if not 0 <= num_real_rows <= hidden_states.shape[0]:
                raise RuntimeError(
                    "WeLMV4 NextN DRAFT_EXTEND_V2 real row count is outside "
                    f"the full input layout: {num_real_rows} vs "
                    f"{hidden_states.shape[0]}."
                )
            forward_batch.num_token_non_padded.fill_(num_real_rows)

        if hidden_states.shape[0] != main_hidden_states.shape[0]:
            raise RuntimeError(
                "WeLMV4 NextN token/target-hidden row mismatch: "
                f"{hidden_states.shape[0]} vs {main_hidden_states.shape[0]}."
            )
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states
        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (self.enorm(hidden_states), self.hnorm(main_hidden_states)),
                    dim=-1,
                )
            )

        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.decoder_layers[0](
                positions, hidden_states, forward_batch, residual
            )

        if residual is None:
            hidden_states_for_next_mtp = hidden_states.to(
                self.shared_head.norm.weight.dtype
            )
            norm_output, _ = self.shared_head.norm(
                hidden_states_for_next_mtp,
                output_dtype=self.shared_head.norm.weight.dtype,
            )
        else:
            # Reuse the fused add+RMSNorm result so recurrent MTP receives the
            # exact pre-head-norm residual sum (and dtype) used for logits.
            norm_output, residual_out = self.shared_head.norm(
                hidden_states,
                residual,
                output_dtype=self.shared_head.norm.weight.dtype,
            )
            hidden_states_for_next_mtp = residual_out.to(
                self.shared_head.norm.weight.dtype
            )
        return norm_output, hidden_states_for_next_mtp


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Do not run the target constructor: it clears the process-global
        # LayerManager that must contain target logical layers 0..47 while the
        # draft logical layer 48 performs its load-time KV-mirror fixup.
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.model = WeLMV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def get_attention_sliding_window_size(self) -> Optional[int]:
        windows = get_welmv4_layerwise_sliding_windows(
            self.config,
            context_len=getattr(self.config, "context_len", None),
            num_layers=int(self.config.num_hidden_layers),
            layer_offset=int(self.config.num_target_hidden_layers),
        )
        swa_windows = [window for window in windows if window >= 0]
        return max(swa_windows, default=None)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        hidden_states, hidden_states_for_next_mtp = self.model(
            input_ids, positions, forward_batch
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            [hidden_states_for_next_mtp],
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed[0]
        self.model.oe_embed = embed[1]
        self.model.oe_gate_up_proj = embed[2]
        self.lm_head = head


EntryClass = WeLMV4MoeForCausalLMNextN
