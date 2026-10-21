# SPDX-License-Identifier: Apache-2.0
"""ModelSlim dynamic grouped Q/K with static per-channel MXFP8 V."""

import logging

import torch

from .modelslim_kv_cache import ModelSlimKVCacheScheme, ModelSlimKVCacheSpec

logger = logging.getLogger(__name__)


class ModelSlimQKMXFP8DynamicVMXFP8PerChannelScheme(ModelSlimKVCacheScheme):
    def create_weights(self, layer):
        if self.binding.shared_latent:
            raise ValueError(f"{self.prefix}: MXFP8 KV cache requires independent K/V")
        self._create_parameter(
            layer, "fa_v_scale", self.binding.value_head_dim, torch.uint8, 127
        )
        self._create_parameter(
            layer, "fa_v_offset", self.binding.value_head_dim, torch.uint8, 0
        )

    def process_weights_after_loading(self, layer):
        state = layer.modelslim_kv_state
        scale = layer.fa_v_scale
        if not scale.modelslim_loaded:
            logger.warning(
                "%s.fa_v.scale is missing; using scale=1 (E8M0 byte 127; "
                "raw=%s, effective=%s, source=%s)",
                self.prefix,
                state.raw_quant_type,
                state.effective_quant_type,
                state.quant_type_source,
            )
            scale.data.fill_(127)
        if (scale == 255).any():
            raise ValueError(f"{self.prefix}.fa_v.scale contains E8M0 NaN")
        if layer.fa_v_offset.modelslim_loaded and torch.count_nonzero(
            layer.fa_v_offset
        ):
            raise ValueError(f"{self.prefix}.fa_v.offset must be zero for MXFP8")
        # ModelSlim minmax uses zero for a channel with zero calibration absmax.
        # Match the Ascend consumer: normalize both the writer and FIA to one.
        self._set_runtime_tensor(
            state,
            "v_scale_encoded",
            torch.where(scale == 0, 127, scale).to(torch.uint8),
        )
        self._set_runtime_tensor(
            state,
            "v_inv_scale",
            torch.exp2(state.v_scale_encoded.float() - 127).reciprocal(),
        )
        self._set_runtime_tensor(
            state, "p_scale", torch.ones(1, dtype=torch.float32, device=scale.device)
        )

    def get_cache_spec(self):
        binding = self.binding
        return ModelSlimKVCacheSpec(
            layout="gqa_mxfp8",
            key_head_dim=binding.head_dim,
            value_head_dim=binding.value_head_dim,
            local_kv_heads=binding.num_local_kv_heads,
            scale_dtype=torch.uint8,
            group_size=32,
        )

    def quantize_query(self, state, q):
        import torch_npu

        quantized, scale = torch_npu.npu_dynamic_mx_quant(
            q, dst_type=torch.float8_e4m3fn
        )
        return quantized, scale.view(torch.uint8).reshape(
            *q.shape[:-1], q.shape[-1] // 32
        )

    def quantize_cache(self, state, k, v):
        import torch_npu

        k, k_scale = self.quantize_query(state, k)
        # Flatten heads/channels to the single static quantization axis.
        v = torch_npu.npu_quantize(
            v.float().flatten(-2),
            state.v_inv_scale.reshape(-1),
            None,
            torch.float8_e4m3fn,
            axis=-1,
            div_mode=False,
        ).view(v.shape)
        return k, v, k_scale
