# SPDX-License-Identifier: Apache-2.0
"""FIA V2 adapters for the ModelSlim NPU cache layouts."""

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz


def _mla_cache_view(cache, page_size, head_dim):
    if is_fia_nz():
        d0 = 32 // cache.element_size()
        return cache.view(-1, 1, head_dim // d0, page_size, d0)
    return cache.view(-1, page_size, head_dim)


def forward_modelslim_attention(
    backend, q, layer, forward_batch, *, q_rope=None, q_descale=None
):
    """Read the cache after the model/pool has written the current tokens."""
    state = layer.modelslim_kv_state
    metadata = backend.forward_metadata
    mode = forward_batch.forward_mode
    q = q.reshape(-1, layer.tp_q_head_num, state.binding.head_dim)
    padded_tokens = q.shape[0]
    if not backend.graph_mode:
        valid_tokens = forward_batch.global_num_token_non_padded_cpu
        if valid_tokens is not None:
            q = q[:valid_tokens]
            if q_rope is not None:
                q_rope = q_rope[:valid_tokens]
                q_descale = q_descale[:valid_tokens]

    # Graph.update rebinds this Host IntArray; do not capture a CPU tensor.
    kv_lens = metadata.seq_lens_cpu_list
    if metadata.seq_lens_cpu_int is not None:
        kv_lens = metadata.seq_lens_cpu_int.tolist()
    if mode.is_decode_or_idle():
        q_lens = [1] * q.shape[0]
    elif mode.is_target_verify():
        width = backend.speculative_num_draft_tokens
        q_lens = [width] * (q.shape[0] // width)
    else:
        q_lens = forward_batch.extend_seq_lens_cpu
        if not backend.graph_mode:
            q_lens = q_lens[: forward_batch._original_batch_size]
    if q.shape[0] == 0:
        return q.new_zeros(
            (padded_tokens, layer.tp_q_head_num * layer.v_head_dim),
            dtype=state.binding.compute_dtype,
        )
    batch_size = len(q_lens)
    kv_lens = kv_lens[:batch_size]
    block_table = metadata.block_tables[:batch_size]
    q_ends = []
    for length in q_lens:
        q_ends.append((q_ends[-1] if q_ends else 0) + length)
    causal = not mode.is_decode_or_idle()
    kwargs = dict(
        block_table=block_table,
        block_size=backend.page_size,
        num_query_heads=layer.tp_q_head_num,
        num_key_value_heads=state.binding.num_local_kv_heads,
        input_layout="TND",
        actual_seq_qlen=q_ends,
        actual_seq_kvlen=kv_lens,
        atten_mask=backend.fia_mask if causal else None,
        sparse_mode=3 if causal else 0,
        softmax_scale=layer.scaling,
        out_dtype=state.binding.compute_dtype,
    )
    pool = backend.token_to_kv_pool
    k_cache = pool.get_key_buffer(layer.layer_id)
    v_cache = pool.get_value_buffer(layer.layer_id)
    if state.cache_spec.layout == "mla_fp8":
        # The model has absorbed kv_b into Q and compensated the BF16 side by
        # sq*sc. FIA multiplies both QK terms by sq*sc, and PV by sc.
        query_heads = layer.tp_q_head_num
        padded_heads = 1 << (query_heads - 1).bit_length()
        q_rope = q_rope.reshape(-1, query_heads, state.binding.side_dim)
        q_descale = q_descale.reshape(-1, query_heads)
        if padded_heads != query_heads:
            # Byte padding avoids unsupported FP8 cat kernels.
            q_bytes = q.view(torch.uint8)
            q = torch.cat(
                (
                    q_bytes,
                    q_bytes.new_zeros(
                        q.shape[0], padded_heads - query_heads, q.shape[-1]
                    ),
                ),
                dim=1,
            ).view(torch.float8_e4m3fn)
            q_rope = torch.cat(
                (
                    q_rope,
                    q_rope.new_zeros(
                        q.shape[0], padded_heads - query_heads, q_rope.shape[-1]
                    ),
                ),
                dim=1,
            )
            q_descale = torch.cat(
                (
                    q_descale,
                    q_descale.new_ones(q.shape[0], padded_heads - query_heads),
                ),
                dim=1,
            )
        k_cache = _mla_cache_view(k_cache, backend.page_size, state.binding.head_dim)
        side_cache = _mla_cache_view(v_cache, backend.page_size, state.binding.side_dim)
        kwargs.update(
            num_query_heads=padded_heads,
            query_rope=q_rope,
            key_rope=side_cache,
            dequant_scale_query=q_descale.contiguous(),
            dequant_scale_key=state.k_scale.reshape(1),
            dequant_scale_value=state.k_scale.reshape(1),
            query_quant_mode=3,
            key_quant_mode=0,
            value_quant_mode=0,
        )
        output = torch_npu.npu_fused_infer_attention_score_v2(
            q, k_cache, k_cache, **kwargs
        )[0]
        output = output[:, :query_heads]
    elif state.cache_spec.layout == "gqa_mxfp8":
        q, q_scale = state.scheme.quantize_query(state, q)
        kv_heads = state.binding.num_local_kv_heads
        groups = layer.tp_q_head_num // kv_heads
        q_scale = q_scale.reshape(q.shape[0], layer.tp_q_head_num, -1, 2)
        # FIA's decode-sized MX query uses N2TGD; larger Q uses TND.
        if groups * max(q_lens, default=0) <= 80:
            q_scale = (
                q_scale.view(q.shape[0], kv_heads, groups, -1, 2)
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )
        k_scale, v_scale = pool.get_kv_scale_buffer(layer.layer_id)
        kwargs.update(
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
            query_dtype=torch.float8_e4m3fn,
            key_dtype=torch.float8_e4m3fn,
            value_dtype=torch.float8_e4m3fn,
            query_quant_mode=6,
            key_quant_mode=6,
            value_quant_mode=8,
            quant_scale_p=state.p_scale,
        )
        output = torch_npu.npu_fused_infer_attention_score_v2(
            q, k_cache, v_cache, **kwargs
        )[0]
    else:
        # Paged GQA FP8 in CANN 9.1 uses high-precision Q + KV antiquant.
        # Preserve ModelSlim's dynamic-Q rounding before that computation.
        q8, q_scale = state.scheme.quantize_query(state, q)
        q = torch_npu.npu_anti_quant(
            q8.reshape(-1, q.shape[-1]),
            q_scale.reshape(-1, 1),
            dst_dtype=state.binding.compute_dtype,
        ).view(q.shape)
        k_cache = k_cache.view(
            -1, backend.page_size, layer.tp_k_head_num * layer.qk_head_dim
        )
        v_cache = v_cache.view(
            -1, backend.page_size, layer.tp_v_head_num * layer.v_head_dim
        )
        kwargs.update(
            dequant_scale_key=state.k_antiquant_scale,
            dequant_scale_value=state.v_antiquant_scale,
            key_quant_mode=0,
            value_quant_mode=0,
        )
        output = torch_npu.npu_fused_infer_attention_score_v2(
            q, k_cache, v_cache, **kwargs
        )[0]
    output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
    if output.shape[0] < padded_tokens:
        output = torch.cat(
            (output, output.new_zeros(padded_tokens - output.shape[0], output.shape[1]))
        )
    return output
