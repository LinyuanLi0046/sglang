"""Opt-in BF16 static decode QKV binding for WeLM attention TP=1/2/4.

The loader owns weight sharding. The attention backend owns physical slots.
This module performs no collective and never copies a weight or KV cache.
"""

from dataclasses import dataclass

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.welmv4_op import WelmV4InplaceRotaryEmbedding
from sglang.srt.layers.welmv4_qkv_layout import (
    qkv_cache_view,
    qkv_tensor_is_nd,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_PROGRAMS = {}
_PLATFORMS = {}
STATIC_ROWS = (1, 2, 4, 8, 16, 32, 64)


def geometry(tp_size, kind):
    """Return local Q width, KV width and projection rows; no tensor slicing."""
    if tp_size not in (1, 2, 4) or kind not in ("plain", "source", "query"):
        raise ValueError("Unsupported WeLM static decode QKV geometry")
    qw, kvw = (24 // tp_size) * 256, max(1, 2 // tp_size) * 256
    return qw, kvw, qw + {"plain": 2, "source": 4, "query": 0}[kind] * kvw


def _platform(device, capturing):
    key = str(device)
    if key not in _PLATFORMS:
        if capturing:
            return None
        name = torch.npu.get_device_name(device)
        if "950DT" not in name:
            _PLATFORMS[key] = None
            return None
        try:
            import cannbotdsl
        except ModuleNotFoundError as exc:
            if exc.name != "cannbotdsl":
                raise
            _PLATFORMS[key] = None
            return None
        version = getattr(cannbotdsl, "__version__", "")
        if version != "0.8.0":
            _PLATFORMS[key] = None
            return None
        from cannbotdsl.ops.info import get_platform_info

        info = get_platform_info(
            stream=torch.npu.current_stream(device), force_refresh=True
        )
        if not info.available or info.cube_core_num < 1:
            _PLATFORMS[key] = None
            return None
        _PLATFORMS[key] = (
            name, str(info.npu_arch), int(info.cube_core_num), version
        )
    return _PLATFORMS[key]


def _tensor(tensor, shape, dtype, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == tuple(shape)
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
    )


@dataclass
class StaticQKVResult:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    mirror_k: torch.Tensor
    mirror_v: torch.Tensor
    query_only: bool

    def full_outputs(self):
        assert not self.query_only
        return self.q, self.k, self.v, self.mirror_k, self.mirror_v


class StaticDecodeQKV:
    def __init__(self, tp_size):
        self.tp_size = tp_size

    def try_run(self, layer, positions, hidden, batch, backend):
        if not getattr(backend, "use_welm_static_decode_qkv", False):
            return None
        if backend.graph_mode and not batch.welm_static_decode_graph_owner:
            return None
        if (
            batch.forward_mode != ForwardMode.DECODE
            or batch.spec_info is not None
            or layer.is_nextn
            or batch.welm_prefill_graph is not None
            or batch.welm_prefill_graph_phase is not None
        ):
            return None
        m = hidden.shape[0]
        if m not in STATIC_ROWS or self.tp_size not in (1, 2, 4):
            return None
        # Check the actual projection method, including per-layer mixed precision.
        if type(layer.qkv_proj.quant_method) is not UnquantizedLinearMethod:
            return None
        if (
            layer.head_dim != 256
            or layer.qk_rope_head_dim != 64
            or layer.q_norm is not None
            or layer.k_norm is None
            or not isinstance(layer.rotary_emb, WelmV4InplaceRotaryEmbedding)
            or not layer.rotary_emb.is_neox_style
        ):
            return None
        consumer = layer.kv_mirror_layer_idx in layer.kv_mirror_layers
        source = getattr(
            layer,
            "_kv_mirror_active_source",
            layer.kv_mirror_layer_idx in layer.kv_mirror_imitated_layers,
        )
        if consumer and source:
            return None
        kind = "query" if consumer else "source" if source else "plain"
        if kind != "plain" and not hasattr(layer, "qkv_proj_weight"):
            return None
        weight = (
            layer.qkv_proj.weight if kind == "plain" else layer.qkv_proj_weight
        )
        bias = layer.qkv_proj.bias if kind == "plain" else layer.qkv_proj_bias
        if bias is not None:
            return None
        qw, kvw, n = geometry(self.tp_size, kind)
        if (layer.q_size, layer.kv_size) != (qw, kvw):
            return None
        device = hidden.device
        table = layer.rotary_emb.cos_sin_cache
        gamma = layer.k_norm.weight
        slots = (
            backend.forward_metadata.welm_static_decode_swa_slots
            if backend._is_swa_layer(layer.attn)
            else backend.forward_metadata.welm_static_decode_full_slots
        )
        if device.type != "npu":
            return None
        tensors = (
            (hidden, (m, 2048), torch.bfloat16),
            (weight, (n, 2048), torch.bfloat16),
            (gamma, (256,), torch.bfloat16),
        )
        if (
            any(
                not _tensor(t, shape, dtype, device)
                for t, shape, dtype in tensors
            )
            or table.ndim != 2
            or table.shape[0] < 1
        ):
            return None
        if any(
            not _tensor(t, shape, dtype, device)
            for t, shape, dtype in (
                (table, (table.shape[0], 64), torch.float32),
                (positions, (m,), torch.int64),
                (slots, (m,), torch.int64),
            )
        ):
            return None
        # Backend admits only exact legacy NPUMHA pools, including SWA children.
        # Never reshape/copy a cache: the kernel must write the attention storage.
        k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.attn.layer_id)
        if any(
            t.dtype != torch.bfloat16
            or t.device != device
            or not t.is_contiguous()
            or t.ndim < 3
            or tuple(t.shape[-2:]) != (kvw // 256, 256)
            for t in (k_cache, v_cache)
        ) or k_cache.shape != v_cache.shape:
            return None
        if any(
            not qkv_tensor_is_nd(t)
            for t in (hidden, weight, gamma, table)
        ):
            return None
        k_cache = qkv_cache_view(k_cache, kvw)
        if k_cache is None:
            return None
        v_cache = qkv_cache_view(v_cache, kvw)
        if v_cache is None:
            return None
        if k_cache.shape[0] < 1 or k_cache.data_ptr() == v_cache.data_ptr():
            return None
        capturing = torch.npu.is_current_stream_capturing()
        platform = _platform(device, capturing)
        if platform is None:
            return None
        key = (
            str(device), platform, self.tp_size, kind, m,
            k_cache.shape[0], table.shape[0], "bf16-i64-token-major-NK",
        )
        program = _PROGRAMS.get(key)
        if program is None:
            if capturing:
                return None
            from sglang.srt.layers.welmv4_static_decode_qkv_kernel import compile_qkv

            program = compile_qkv(
                k_cache.shape[0],
                table.shape[0],
                tp_size=self.tp_size,
                has_mirror=source,
                q_only=consumer,
                static_m=m,
                max_blocks=platform[2],
            )
            _PROGRAMS[key] = program
        # A capture ForwardBatch is retained by its graph owner. Reuse its own
        # warmup outputs during capture, never one process-global output arena.
        owners = getattr(batch, "welm_static_decode_qkv_outputs", None)
        if owners is None:
            if capturing:
                return None
            owners = batch.welm_static_decode_qkv_outputs = {}
        output_key = (layer.attn.layer_id, key)
        outputs = owners.get(output_key)
        if outputs is None:
            if capturing:
                return None
            kv_rows, mirror_rows = (1 if consumer else m), (m if source else 1)
            outputs = (
                hidden.new_empty((m, qw)),
                hidden.new_empty((kv_rows, kvw)),
                hidden.new_empty((kv_rows, kvw)),
                hidden.new_empty((mirror_rows, kvw)),
                hidden.new_empty((mirror_rows, kvw)),
            )
            owners[output_key] = outputs
        # No fallback after dispatch: errors propagate, without a duplicate KV write.
        program(
            hidden, weight.data, gamma.data.view(1, 256), positions, table, slots,
            *outputs, k_cache, v_cache, float(layer.k_norm.eps),
        )
        q, k, v, mk, mv = outputs
        return StaticQKVResult(
            q, k, v, mk if source else None, mv if source else None, consumer
        )
