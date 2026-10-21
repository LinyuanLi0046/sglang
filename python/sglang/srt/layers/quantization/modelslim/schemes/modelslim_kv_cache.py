# SPDX-License-Identifier: Apache-2.0
"""ModelSlim KV cache contracts and shared parameter lifecycle."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class ModelSlimKVCacheBinding:
    """Physical cache geometry and the model's existing KV head ownership."""

    num_kv_heads: int
    num_local_kv_heads: int
    head_dim: int
    value_head_dim: int
    shared_latent: bool = False
    side_dim: int = 0
    kv_tp_rank: int = 0
    num_kv_head_replicas: int = 1
    canonical_prefix: Optional[str] = None
    compute_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_qkv(cls, qkv_proj, prefix):
        from sglang.srt.distributed.utils import get_group_rank_size

        kv_tp_rank, _ = get_group_rank_size(qkv_proj.kv_tp_group)
        return cls(
            num_kv_heads=qkv_proj.total_num_kv_heads,
            num_local_kv_heads=qkv_proj.num_kv_heads,
            head_dim=qkv_proj.head_size,
            value_head_dim=qkv_proj.v_head_size,
            kv_tp_rank=kv_tp_rank,
            num_kv_head_replicas=qkv_proj.num_kv_head_replicas,
            canonical_prefix=prefix,
            compute_dtype=torch.get_default_dtype(),
        )


@dataclass(frozen=True)
class ModelSlimKVCacheSpec:
    """Allocation contract; independent of checkpoint scale values."""

    layout: str
    key_head_dim: int
    value_head_dim: int
    local_kv_heads: int
    key_dtype: torch.dtype = torch.float8_e4m3fn
    value_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: Optional[torch.dtype] = None
    group_size: int = 0


@dataclass
class ModelSlimKVCacheState:
    """Shared by MLA execution views without registering duplicate parameters."""

    scheme: "ModelSlimKVCacheScheme"
    binding: ModelSlimKVCacheBinding
    cache_spec: ModelSlimKVCacheSpec
    raw_quant_type: Optional[str] = None
    effective_quant_type: Optional[str] = None
    quant_type_source: str = "checkpoint"
    k_scale: Optional[torch.Tensor] = None
    k_inv_scale: Optional[torch.Tensor] = None
    v_inv_scale: Optional[torch.Tensor] = None
    v_scale_encoded: Optional[torch.Tensor] = None
    p_scale: Optional[torch.Tensor] = None
    refresh_cache_scales: Optional[Callable[[], None]] = None


class ModelSlimKVCacheScheme(ABC):
    """ModelSlim parameter lifecycle and cache encoding, without slot ownership."""

    def __init__(self, prefix, binding):
        self.prefix = prefix
        self.binding = binding

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def get_cache_spec(self) -> ModelSlimKVCacheSpec:
        raise NotImplementedError

    @abstractmethod
    def quantize_query(self, state: ModelSlimKVCacheState, q: torch.Tensor):
        """Return encoded Q and per-token/head (or group) dequantization scales."""
        raise NotImplementedError

    @abstractmethod
    def quantize_cache(self, state: ModelSlimKVCacheState, k, v):
        """Return encoded K, V and optional dynamic K scale."""
        raise NotImplementedError

    def _create_parameter(self, layer, name, channels, dtype, default):
        """Load whole KV heads using the same shard/replica mapping as QKV."""
        binding = self.binding
        parameter = torch.nn.Parameter(
            torch.full((binding.num_local_kv_heads, channels), default, dtype=dtype),
            requires_grad=False,
        )
        parameter.modelslim_loaded = False

        def weight_loader(param, loaded_weight):
            if loaded_weight.dtype != dtype:
                raise ValueError(
                    f"{self.prefix}.{name}: expected {dtype}, got {loaded_weight.dtype}"
                )
            # A local checkpoint shard is already in this rank's head order.
            if loaded_weight.numel() == param.numel():
                local_weight = loaded_weight.reshape_as(param)
            elif loaded_weight.numel() == binding.num_kv_heads * channels:
                first_head = (
                    binding.kv_tp_rank // binding.num_kv_head_replicas
                ) * binding.num_local_kv_heads
                local_weight = loaded_weight.reshape(
                    binding.num_kv_heads, channels
                ).narrow(0, first_head, binding.num_local_kv_heads)
            else:
                raise ValueError(
                    f"{self.prefix}.{name}: cannot map shape "
                    f"{tuple(loaded_weight.shape)} to "
                    f"{binding.num_local_kv_heads} local KV heads "
                    f"with {channels} scale channels"
                )
            param.data.copy_(local_weight)
            param.modelslim_loaded = True

        parameter.weight_loader = weight_loader
        layer.register_parameter(name, parameter)
        return parameter

    @staticmethod
    def _set_runtime_tensor(state, name, value):
        # Preserve graph-captured addresses and aliases when weights are reloaded.
        current = getattr(state, name)
        if current is None:
            setattr(state, name, value.detach().clone())
        else:
            current.copy_(value)
