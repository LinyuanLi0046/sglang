from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.welmv4_shape_logger import log_welmv4_kernel_shapes_once
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
        for group_size_m in [4, 8, 16]
        for block_size_n in [512]
        for block_size_k in [32, 64, 128, 256, 512, 1024, 2048]
    ],
    key=["M", "N", "K"],
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
