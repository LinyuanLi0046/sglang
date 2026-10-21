# SPDX-License-Identifier: Apache-2.0
"""Dynamic per-token/head Q and static per-head K/V in ModelSlim FP8."""

import logging

import torch

from .modelslim_kv_cache import ModelSlimKVCacheScheme, ModelSlimKVCacheSpec

logger = logging.getLogger(__name__)


class ModelSlimQFP8DynamicKVFP8Scheme(ModelSlimKVCacheScheme):
    def create_weights(self, layer):
        for component in ("k", "v"):
            self._create_parameter(
                layer, f"fa_{component}_scale", 1, torch.float32, 1.0
            )
            self._create_parameter(
                layer, f"fa_{component}_offset", 1, torch.float32, 0.0
            )

    def process_weights_after_loading(self, layer):
        state = layer.modelslim_kv_state
        components = ("k",) if self.binding.shared_latent else ("k", "v")
        for component in ("k", "v"):
            scale = getattr(layer, f"fa_{component}_scale")
            offset = getattr(layer, f"fa_{component}_offset")
            if component in components and not scale.modelslim_loaded:
                logger.warning(
                    "%s.fa_%s.scale is missing; using scale=1 "
                    "(raw=%s, effective=%s, source=%s)",
                    self.prefix,
                    component,
                    state.raw_quant_type,
                    state.effective_quant_type,
                    state.quant_type_source,
                )
                scale.data.fill_(1)
            if scale.modelslim_loaded and (
                not torch.isfinite(scale).all() or not (scale > 0).all()
            ):
                raise ValueError(
                    f"{self.prefix}.fa_{component}.scale must be finite and positive"
                )
            if offset.modelslim_loaded and torch.count_nonzero(offset):
                raise ValueError(
                    f"{self.prefix}.fa_{component}.offset must be zero for FP8"
                )

        # MLA's fa_v is export compatibility data. QK and PV share fa_k's C.
        self._set_runtime_tensor(state, "k_scale", layer.fa_k_scale)
        self._set_runtime_tensor(
            state,
            "v_scale",
            layer.fa_k_scale if self.binding.shared_latent else layer.fa_v_scale,
        )
        self._set_runtime_tensor(state, "k_inv_scale", state.k_scale.reciprocal())
        self._set_runtime_tensor(state, "v_inv_scale", state.v_scale.reciprocal())
        if not self.binding.shared_latent:
            dtype = self.binding.compute_dtype
            self._set_runtime_tensor(
                state,
                "k_antiquant_scale",
                state.k_scale.expand(-1, self.binding.head_dim).to(dtype),
            )
            self._set_runtime_tensor(
                state,
                "v_antiquant_scale",
                state.v_scale.expand(-1, self.binding.value_head_dim).to(dtype),
            )

    def get_cache_spec(self):
        binding = self.binding
        return ModelSlimKVCacheSpec(
            layout="mla_fp8" if binding.shared_latent else "gqa_fp8",
            key_head_dim=binding.head_dim,
            value_head_dim=(
                binding.side_dim if binding.shared_latent else binding.value_head_dim
            ),
            local_kv_heads=binding.num_local_kv_heads,
            value_dtype=(
                torch.bfloat16 if binding.shared_latent else torch.float8_e4m3fn
            ),
        )

    def quantize_query(self, state, q):
        import torch_npu

        shape = q.shape
        quantized, scale = torch_npu.npu_dynamic_quant(
            q.reshape(-1, shape[-1]), dst_type=torch.float8_e4m3fn
        )
        return quantized.view(shape), scale.view(*shape[:-1], 1).float()

    @staticmethod
    def _quantize(x, inverse_scale):
        import torch_npu

        # Out of place: MLA prefill still uses the original BF16 latent.
        if x.dtype == torch.float8_e4m3fn:
            return x
        # AscendQuant requires matching input/scale dtypes. Keep the checkpoint
        # scale in FP32; axis -2 applies one scale to each KV head.
        return torch_npu.npu_quantize(
            x.float(),
            inverse_scale.reshape(-1),
            None,
            torch.float8_e4m3fn,
            axis=-2,
            div_mode=False,
        )

    def quantize_cache(self, state, k, v):
        k = self._quantize(k, state.k_inv_scale)
        if self.binding.shared_latent:
            v = v.to(torch.bfloat16)
        else:
            v = self._quantize(v, state.v_inv_scale)
        return k, v, None, None

    def dequantize_cache(self, state, k, v=None, k_scale=None, v_scale=None):
        import torch_npu

        # Explicit singleton axes avoid ambiguity when head count equals D.
        k = torch_npu.npu_anti_quant(
            k,
            state.k_scale.view(*([1] * (k.ndim - 2)), -1, 1),
            dst_dtype=self.binding.compute_dtype,
        )
        if v is not None and not self.binding.shared_latent:
            v = torch_npu.npu_anti_quant(
                v,
                state.v_scale.view(*([1] * (v.ndim - 2)), -1, 1),
                dst_dtype=self.binding.compute_dtype,
            )
        return k, v
