from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.layers.welmv4_shape_logger import (
    WELMV4_KERNEL_SHAPE_LOG_ENABLED,
    log_welmv4_kernel_shapes_once,
)
from sglang.srt.utils import is_npu


def _get_num_sms(multiplier: int = 1) -> int:
    if is_npu():
        device = torch.npu.current_device()
        device_properties = (
            triton.runtime.driver.active.utils.get_device_properties(device)
        )
        num_cores = device_properties.get(
            "num_vectorcore", device_properties.get("num_aicore", -1)
        )
        assert num_cores > 0, "Failed to detect NPU core count."
        return num_cores * multiplier

    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        * multiplier
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"GROUP_SIZE_M": group_size_m, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k},
        )
        for group_size_m in [4, 8, 16, 32, 64]
        for block_size_n in [512]
        for block_size_k in [32, 64, 128, 256, 512, 1024, 2048]
    ],
    key=["N", "K"],
)
@triton.jit
def mmq_style_router_linear_kernel_npu(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * GROUP_SIZE_M + tl.arange(0, GROUP_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((GROUP_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def mmq_style_router_linear_npu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and weight.dim() == 2
    assert x.shape[1] == weight.shape[1], "hidden_size mismatch: x.shape[1] must equal weight.shape[1]"

    M, K = x.shape
    N = weight.shape[0]

    x = x.contiguous()
    weight = weight.to(dtype=x.dtype).contiguous()

    c = torch.empty((M, N), dtype=torch.float32, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["GROUP_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    mmq_style_router_linear_kernel_npu[grid](
        x,
        weight,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


@triton.jit
def _rope_npu(
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    """单 token 多 head 的 RoPE 辅助 kernel (就地修改)。

    [优化] 消除 trans/split/join/reshape 等 layout 变换,改为分两半
    load → 计算 → 分两半 store,所有访存行优先连续,避免 NPU store
    转置退化。计算逻辑 (GPT-NeoX rotate-half) 完全等价。
    """
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_head_offset = tl.arange(0, num_heads_blocked)
    half_rope_offset = tl.arange(0, half_rope_dim)
    mask = num_head_offset[:, None] < num_heads
    base = data_ptr + num_head_offset[:, None] * head_dim
    # 分两半加载: 前半 x1 与后半 x2,每半沿 rope 维度连续
    x1 = tl.load(
        base + half_rope_offset[None, :], mask=mask, care_padding=False
    )
    x2 = tl.load(
        base + (half_rope_dim + half_rope_offset)[None, :],
        mask=mask,
        care_padding=False,
    )
    # GPT-NeoX rotate-half
    x_out1 = x1 * cos - x2 * sin
    x_out2 = x1 * sin + x2 * cos
    # 分两半存储: 行优先连续写回,无转置退化
    tl.store(base + half_rope_offset[None, :], x_out1, mask=mask)
    tl.store(
        base + (half_rope_dim + half_rope_offset)[None, :], x_out2, mask=mask
    )


@triton.jit
def _welmv4_inplace_rope_kernel_npu(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    N: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    num_sms: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads_blocked: tl.constexpr,
):
    """WeLMv4 尾部 RoPE 主 kernel (就地修改 Q/K)。

    与原始 kernel 的差异:
      - 所有 tl.load 添加 care_padding=False
      - 调用的 _rope_npu 已重写 (消除 trans/split/join, 分半 load/store)
    Grid (1D, 跨步循环)、计算逻辑、条件分支均保持不变。
    """
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_off = tl.arange(0, half_rope_dim)
    sin_off = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(tl.program_id(0), N, num_sms, num_stages=num_stages):
        position_id = tl.load(position_ptr + token_id)
        # [NPU 迁移] 添加 care_padding=False
        cos_sin_cache = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + cos_off, care_padding=False
        )
        sin_sin_cache = tl.load(
            cos_sin_cache_ptr + position_id * rope_dim + sin_off, care_padding=False
        )
        q_data_ptr = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data_ptr = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _rope_npu(
            k_data_ptr, cos_sin_cache, sin_sin_cache,
            num_k_heads, num_k_heads_blocked, head_dim, rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                position_id = tl.load(last_index_ptr + token_id)
                position_id = tl.load(position_ptr + position_id)
                # [NPU 迁移] 添加 care_padding=False
                cos_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + cos_off,
                    care_padding=False,
                )
                sin_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + sin_off,
                    care_padding=False,
                )
                _rope_npu(
                    q_data_ptr, cos_sin_cache, sin_sin_cache,
                    num_q_heads, num_q_heads_blocked, head_dim, rope_dim,
                )
        else:
            _rope_npu(
                q_data_ptr, cos_sin_cache, sin_sin_cache,
                num_q_heads, num_q_heads_blocked, head_dim, rope_dim,
            )


def welmv4_inplace_rope_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    last_index: torch.Tensor = None,
    head_dim: int = 128,
    rope_dim: int = 64,
    num_stages: int = 4,
    layer_id: Optional[int] = None,
    stage: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 就地应用尾部 RoPE。

    Args:
        query: (N, num_q_heads, head_dim)
        key:   (N, num_k_heads, head_dim)
        positions: (N,) int32
        cos_sin_cache: (max_pos, rope_dim) float32, 前半 cos 后半 sin
        last_index: (BS,) int32 或 None; KV-mirror 模式 Q 的源 token 索引
        head_dim, rope_dim: 维度参数
        num_stages: tl.range 软件流水线阶段数

    Returns:
        (query, key) 就地修改后返回
    """
    N = positions.shape[0]
    num_q_heads = query.shape[1]
    num_k_heads = key.shape[1]
    num_sms = min(N, _get_num_sms(multiplier=8))
    BS = last_index.numel() if last_index is not None else 0

    _welmv4_inplace_rope_kernel_npu[(num_sms,)](
        query, key, positions, cos_sin_cache, last_index,
        N, BS,
        query.stride(0), key.stride(0),
        head_dim, rope_dim,
        num_sms, num_stages,
        num_q_heads, num_k_heads,
        triton.next_power_of_2(num_q_heads),
        triton.next_power_of_2(num_k_heads),
    )
    if WELMV4_KERNEL_SHAPE_LOG_ENABLED:
        log_welmv4_kernel_shapes_once(
            kernel="_welmv4_inplace_rope_kernel_npu",
            layer_id=layer_id,
            stage=stage,
            m=N,
            inputs={
                "query": query,
                "key": key,
                "positions": positions,
                "cos_sin_cache": cos_sin_cache,
                "last_index": last_index,
            },
            outputs={"query": query, "key": key},
            parameters={
                "BS": BS,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "num_q_heads": num_q_heads,
                "num_k_heads": num_k_heads,
                "num_sms": num_sms,
            },
        )
    return query, key


# -----------------------------------------------------------------------------
# WeLMv4 over-encoding helpers for Ascend A5.
#
# These kernels intentionally stop at hashed OE ids and token-table updates.
# OE embedding lookup/concat remains on the framework's native PyTorch path.
# -----------------------------------------------------------------------------

_WELMV4_OE_HASH_MULTIPLIER = 2654435761
_WELMV4_OE_BRANCHES = 4
_WELMV4_VECTOR_CORE_CACHE: dict[tuple[str, int], int] = {}


@triton.jit
def _welmv4_u32_remainder(value, divisor: tl.constexpr):
    """Exact uint32 remainder supported efficiently by Triton Ascend."""
    quotient = value // divisor
    return value - quotient * divisor


@triton.jit
def _welmv4_oe_hash_prefill_4way_kernel(
    input_ids_ptr,
    token_table_ptr,
    req_rows_ptr,
    token_offsets_ptr,
    req_lens_ptr,
    column_starts_ptr,
    hashed_out_ptr,
    num_tokens,
    context_len: tl.constexpr,
    num_token_tiles,
    num_tasks,
    VOCAB_SIZE: tl.constexpr,
    OE_V0: tl.constexpr,
    OE_V1: tl.constexpr,
    OE_V2: tl.constexpr,
    OE_V3: tl.constexpr,
    HASH_MUL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fuse request mapping, history gather, pack, hash and four mods."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_T)

    for task_idx in range(task_start, task_end):
        req_idx = task_idx // num_token_tiles
        tile_idx = task_idx - req_idx * num_token_tiles
        local_offsets = tile_idx * BLOCK_T + base_offsets

        req_len = tl.load(req_lens_ptr + req_idx).to(tl.int32)
        token_start = tl.load(token_offsets_ptr + req_idx).to(tl.int32)
        request_row = tl.load(req_rows_ptr + req_idx).to(tl.int32)
        column_start = tl.load(column_starts_ptr + req_idx).to(tl.int32)

        flat_indices = token_start + local_offsets
        token_mask = (local_offsets.to(tl.float32) < req_len.to(tl.float32)) & (
            flat_indices.to(tl.float32) < num_tokens
        )
        logical_positions = column_start + local_offsets
        current = tl.load(
            input_ids_ptr + flat_indices, mask=token_mask, other=0
        ).to(tl.uint32)

        # Current-chunk history is contiguous in input_ids. Only the first two
        # boundary tokens need an indexed request-token-table read.
        local_offsets_fp32 = local_offsets.to(tl.float32)
        logical_positions_fp32 = logical_positions.to(tl.float32)
        prev1_from_chunk_mask = token_mask & (local_offsets_fp32 >= 1.0)
        prev1_from_table_mask = (
            token_mask & (local_offsets_fp32 < 1.0) & (logical_positions_fp32 >= 1.0)
        )
        prev1_from_chunk = tl.load(
            input_ids_ptr + flat_indices - 1,
            mask=prev1_from_chunk_mask,
            other=0,
        ).to(tl.uint32)
        prev1_from_table = tl.load(
            token_table_ptr
            + request_row * context_len
            + logical_positions
            - 1,
            mask=prev1_from_table_mask,
            other=0,
        ).to(tl.uint32)
        previous1 = prev1_from_chunk + prev1_from_table

        prev2_from_chunk_mask = token_mask & (local_offsets_fp32 >= 2.0)
        prev2_from_table_mask = (
            token_mask & (local_offsets_fp32 < 2.0) & (logical_positions_fp32 >= 2.0)
        )
        prev2_from_chunk = tl.load(
            input_ids_ptr + flat_indices - 2,
            mask=prev2_from_chunk_mask,
            other=0,
        ).to(tl.uint32)
        prev2_from_table = tl.load(
            token_table_ptr
            + request_row * context_len
            + logical_positions
            - 2,
            mask=prev2_from_table_mask,
            other=0,
        ).to(tl.uint32)
        previous2 = prev2_from_chunk + prev2_from_table

        packed2 = current + previous1 * VOCAB_SIZE
        packed3 = packed2 + previous2 * VOCAB_SIZE * VOCAB_SIZE
        hash2 = (packed2 * HASH_MUL).to(tl.uint32)
        hash3 = (packed3 * HASH_MUL).to(tl.uint32)

        tl.store(
            hashed_out_ptr + flat_indices,
            _welmv4_u32_remainder(hash2, OE_V0).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + num_tokens + flat_indices,
            _welmv4_u32_remainder(hash2, OE_V1).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + 2 * num_tokens + flat_indices,
            _welmv4_u32_remainder(hash3, OE_V2).to(tl.int32),
            mask=token_mask,
        )
        tl.store(
            hashed_out_ptr + 3 * num_tokens + flat_indices,
            _welmv4_u32_remainder(hash3, OE_V3).to(tl.int32),
            mask=token_mask,
        )


@triton.jit
def _welmv4_oe_hash_decode_4way_kernel(
    input_ids_ptr,
    token_table_ptr,
    req_rows_ptr,
    column_starts_ptr,
    hashed_out_ptr,
    batch_size,
    context_len: tl.constexpr,
    num_tasks,
    VOCAB_SIZE: tl.constexpr,
    OE_V0: tl.constexpr,
    OE_V1: tl.constexpr,
    OE_V2: tl.constexpr,
    OE_V3: tl.constexpr,
    HASH_MUL: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Decode specialization: one current token per real request."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_B)

    for task_idx in range(task_start, task_end):
        req_indices = task_idx * BLOCK_B + base_offsets
        request_mask = req_indices.to(tl.float32) < batch_size
        request_rows = tl.load(
            req_rows_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.int32)
        positions = tl.load(
            column_starts_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.int32)
        current = tl.load(
            input_ids_ptr + req_indices, mask=request_mask, other=0
        ).to(tl.uint32)
        previous1 = tl.load(
            token_table_ptr + request_rows * context_len + positions - 1,
            mask=request_mask & (positions.to(tl.float32) >= 1.0),
            other=0,
        ).to(tl.uint32)
        previous2 = tl.load(
            token_table_ptr + request_rows * context_len + positions - 2,
            mask=request_mask & (positions.to(tl.float32) >= 2.0),
            other=0,
        ).to(tl.uint32)

        packed2 = current + previous1 * VOCAB_SIZE
        packed3 = packed2 + previous2 * VOCAB_SIZE * VOCAB_SIZE
        hash2 = (packed2 * HASH_MUL).to(tl.uint32)
        hash3 = (packed3 * HASH_MUL).to(tl.uint32)
        tl.store(
            hashed_out_ptr + req_indices,
            _welmv4_u32_remainder(hash2, OE_V0).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + batch_size + req_indices,
            _welmv4_u32_remainder(hash2, OE_V1).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + 2 * batch_size + req_indices,
            _welmv4_u32_remainder(hash3, OE_V2).to(tl.int32),
            mask=request_mask,
        )
        tl.store(
            hashed_out_ptr + 3 * batch_size + req_indices,
            _welmv4_u32_remainder(hash3, OE_V3).to(tl.int32),
            mask=request_mask,
        )


@triton.jit
def _welmv4_token_table_ragged_update_kernel(
    token_table_ptr,
    tokens_ptr,
    row_indices_ptr,
    token_offsets_ptr,
    column_starts_ptr,
    req_lens_ptr,
    num_tokens,
    context_len: tl.constexpr,
    num_token_tiles,
    num_tasks,
    BLOCK_T: tl.constexpr,
):
    """Copy request-segmented tokens without materializing flat row/col ids."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_T)

    for task_idx in range(task_start, task_end):
        req_idx = task_idx // num_token_tiles
        tile_idx = task_idx - req_idx * num_token_tiles
        local_offsets = tile_idx * BLOCK_T + base_offsets
        req_len = tl.load(req_lens_ptr + req_idx).to(tl.int32)
        token_start = tl.load(token_offsets_ptr + req_idx).to(tl.int32)
        row = tl.load(row_indices_ptr + req_idx).to(tl.int32)
        column_start = tl.load(column_starts_ptr + req_idx).to(tl.int32)
        token_indices = token_start + local_offsets
        columns = column_start + local_offsets
        mask = (
            (local_offsets.to(tl.float32) < req_len.to(tl.float32))
            & (token_indices.to(tl.float32) < num_tokens)
            & (columns.to(tl.float32) < context_len)
        )
        values = tl.load(tokens_ptr + token_indices, mask=mask, other=0)
        tl.store(token_table_ptr + row * context_len + columns, values, mask=mask)


@triton.jit
def _welmv4_token_table_decode_update_kernel(
    token_table_ptr,
    next_token_ids_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    skip_mask_ptr,
    batch_size,
    context_len: tl.constexpr,
    num_tasks,
    HAS_SKIP_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Write one sampled token per real request, honoring chunk skip state."""
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)
    tasks_per_program = (num_tasks + program_count - 1) // program_count
    task_start = pid * tasks_per_program
    task_end = tl.minimum(task_start + tasks_per_program, num_tasks)
    base_offsets = tl.arange(0, BLOCK_B)

    for task_idx in range(task_start, task_end):
        req_indices = task_idx * BLOCK_B + base_offsets
        real_mask = req_indices.to(tl.float32) < batch_size
        if HAS_SKIP_MASK:
            skip = tl.load(
                skip_mask_ptr + req_indices, mask=real_mask, other=1
            ).to(tl.int1)
        else:
            skip = tl.zeros((BLOCK_B,), dtype=tl.int1)
        rows = tl.load(
            req_pool_indices_ptr + req_indices, mask=real_mask, other=0
        ).to(tl.int32)
        columns = tl.load(
            seq_lens_ptr + req_indices, mask=real_mask, other=0
        ).to(tl.int32)
        values = tl.load(next_token_ids_ptr + req_indices, mask=real_mask, other=0)
        update_mask = (
            real_mask & (~skip) & (columns.to(tl.float32) < context_len)
        )
        tl.store(
            token_table_ptr + rows * context_len + columns,
            values,
            mask=update_mask,
        )


def _welmv4_vector_core_count(device: torch.device) -> int:
    index = torch.npu.current_device() if device.index is None else int(device.index)
    key = (device.type, index)
    cached = _WELMV4_VECTOR_CORE_CACHE.get(key)
    if cached is not None:
        return cached
    properties = triton.runtime.driver.active.utils.get_device_properties(index)
    count = int(properties.get("num_vectorcore", properties.get("num_aicore", -1)))
    if count <= 0:
        raise RuntimeError("Failed to detect the Ascend Vector Core count")
    _WELMV4_VECTOR_CORE_CACHE[key] = count
    return count


def _welmv4_1d_grid(num_tasks: int, device: torch.device) -> tuple[int]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    return (min(num_tasks, _welmv4_vector_core_count(device)),)


def _validate_welmv4_oe_vocab_sizes(
    oe_vocab_sizes: Sequence[int],
) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in oe_vocab_sizes)
    if len(values) != _WELMV4_OE_BRANCHES or any(v <= 0 for v in values):
        raise ValueError("oe_vocab_sizes must contain four positive integers")
    if any(v > (1 << 31) for v in values):
        raise ValueError("int32 OE hash output requires every vocab size <= 2^31")
    return values


def welmv4_oe_hash_prefill_4way_npu(
    input_ids: torch.Tensor,
    token_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    req_lens: torch.Tensor,
    column_starts: torch.Tensor,
    *,
    max_req_len: int,
    vocab_size: int,
    oe_vocab_sizes: Sequence[int],
    block_t: int = 512,
) -> torch.Tensor:
    """Return int32 hashed ids with layout [4, num_tokens] for prefill."""
    oe_v0, oe_v1, oe_v2, oe_v3 = _validate_welmv4_oe_vocab_sizes(
        oe_vocab_sizes
    )
    num_tokens = input_ids.numel()
    output = torch.empty(
        (_WELMV4_OE_BRANCHES, num_tokens),
        dtype=torch.int32,
        device=input_ids.device,
    )
    if num_tokens == 0:
        return output
    num_token_tiles = triton.cdiv(max_req_len, block_t)
    num_tasks = req_lens.numel() * num_token_tiles
    _welmv4_oe_hash_prefill_4way_kernel[
        _welmv4_1d_grid(num_tasks, input_ids.device)
    ](
        input_ids,
        token_table,
        req_pool_indices,
        token_offsets,
        req_lens,
        column_starts,
        output,
        num_tokens,
        token_table.shape[1],
        num_token_tiles,
        num_tasks,
        VOCAB_SIZE=int(vocab_size),
        OE_V0=oe_v0,
        OE_V1=oe_v1,
        OE_V2=oe_v2,
        OE_V3=oe_v3,
        HASH_MUL=_WELMV4_OE_HASH_MULTIPLIER,
        BLOCK_T=block_t,
    )
    return output


def welmv4_oe_hash_decode_4way_npu(
    input_ids: torch.Tensor,
    token_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    column_starts: torch.Tensor,
    *,
    vocab_size: int,
    oe_vocab_sizes: Sequence[int],
    block_b: int = 128,
) -> torch.Tensor:
    """Return int32 hashed ids with layout [4, batch_size] for decode."""
    oe_v0, oe_v1, oe_v2, oe_v3 = _validate_welmv4_oe_vocab_sizes(
        oe_vocab_sizes
    )
    batch_size = input_ids.numel()
    if (
        req_pool_indices.numel() < batch_size
        or column_starts.numel() < batch_size
    ):
        raise ValueError("decode metadata must contain at least batch_size entries")
    output = torch.empty(
        (_WELMV4_OE_BRANCHES, batch_size),
        dtype=torch.int32,
        device=input_ids.device,
    )
    if batch_size == 0:
        return output
    num_tasks = triton.cdiv(batch_size, block_b)
    _welmv4_oe_hash_decode_4way_kernel[
        _welmv4_1d_grid(num_tasks, input_ids.device)
    ](
        input_ids,
        token_table,
        req_pool_indices,
        column_starts,
        output,
        batch_size,
        token_table.shape[1],
        num_tasks,
        VOCAB_SIZE=int(vocab_size),
        OE_V0=oe_v0,
        OE_V1=oe_v1,
        OE_V2=oe_v2,
        OE_V3=oe_v3,
        HASH_MUL=_WELMV4_OE_HASH_MULTIPLIER,
        BLOCK_B=block_b,
    )
    return output


def welmv4_token_table_ragged_update_npu(
    token_table: torch.Tensor,
    tokens: torch.Tensor,
    row_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    *,
    max_req_len: int,
    block_t: int = 256,
) -> None:
    """Update prefill request segments directly in the request token table."""
    num_tokens = tokens.numel()
    if num_tokens == 0:
        return
    num_token_tiles = triton.cdiv(max_req_len, block_t)
    num_tasks = req_lens.numel() * num_token_tiles
    _welmv4_token_table_ragged_update_kernel[
        _welmv4_1d_grid(num_tasks, tokens.device)
    ](
        token_table,
        tokens,
        row_indices,
        token_offsets,
        column_starts,
        req_lens,
        num_tokens,
        token_table.shape[1],
        num_token_tiles,
        num_tasks,
        BLOCK_T=block_t,
    )


def welmv4_token_table_decode_update_npu(
    token_table: torch.Tensor,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    skip_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    block_b: int = 16,
) -> None:
    """Update one sampled token per real request; ignore graph padding rows."""
    if batch_size == 0:
        return
    num_tasks = triton.cdiv(batch_size, block_b)
    # A compile-time flag removes the skip load for normal decode. Triton still
    # needs a pointer argument, so next_token_ids is an unused safe placeholder.
    skip_mask_ptr = next_token_ids if skip_mask is None else skip_mask
    _welmv4_token_table_decode_update_kernel[
        _welmv4_1d_grid(num_tasks, next_token_ids.device)
    ](
        token_table,
        next_token_ids,
        req_pool_indices,
        seq_lens,
        skip_mask_ptr,
        batch_size,
        token_table.shape[1],
        num_tasks,
        HAS_SKIP_MASK=skip_mask is not None,
        BLOCK_B=block_b,
    )
