from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import WelmDpRowView
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None


def world_dp_gather_enabled() -> bool:
    """Whether DP gathers should use expanded WORLD after joiner admission."""
    dp = get_flags().dp
    return dp.use_world_group_for_gather and not dp.joiner_skip_all_gather


def enable_joiner_all_gather():
    get_flags().dp.joiner_skip_all_gather = False


def update_dp_attention_post_scale(new_dp_size: int, new_dp_rank: int):
    global _ATTN_DP_SIZE, _ATTN_DP_RANK
    _ATTN_DP_SIZE = new_dp_size
    _ATTN_DP_RANK = new_dp_rank
    get_flags().dp.use_world_group_for_gather = True
    logger.debug(
        "[Elastic EP] dp_attention switched to WORLD: dp_size=%d dp_rank=%d",
        new_dp_size,
        new_dp_rank,
    )


_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")


class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        dp_size = get_attention_dp_size()

        # (trangdough) pplx-kernels a2a is a symmetric collective: every EP rank
        # must dispatch the same number of tokens or the device-side handshake
        # deadlocks (idle DP ranks with 0 tokens never signal their peers).
        # Force MAX_LEN so all ranks are padded to equal token counts.
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if get_moe_a2a_backend().is_pplx():
            return DpPaddingMode.MAX_LEN

        # When is_extend_in_batch and dp_size > 1, use SUM_LEN to avoid padding
        # overhead from uneven token distribution.
        # For dp_size=1, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        if is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        # prefer MAX_LEN when communication cost is equal to enable symmetric memory
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:
    """Facade for the DP gathered-buffer state: allocation metadata lives on
    ``flags.dp`` (set once at initialize_dp_attention). The per-forward
    sizing quartet stays as class attributes: the values are read inside
    torch.compile-traced model code, and attribute-source ints get dynamo's
    automatic-dynamic treatment, while contextvars are untraceable and dict
    slots value-guard into the recompile limit (one recompile per distinct
    size)."""

    # Real defaults (not bare annotations): the sizing quartet is overwritten
    # per-forward by set_dp_buffer_len, but callers that run before the first
    # forward — notably the load-time mhc_pre prewarm, which has no ForwardBatch
    # yet — read _dp_max_padding via is_allocation_symmetric(). A bare
    # annotation creates no class attribute, so those reads raised
    # AttributeError. Defaulting _dp_max_padding to False (non-symmetric) is
    # safe for prewarm: it only JIT-compiles kernels and never enters a real
    # all-reduce, so the symmetric pool is not needed there.
    _global_dp_buffer_len: int = 0
    _local_dp_buffer_len: int = 0
    _dp_max_padding: bool = False
    _global_num_tokens: Optional[List[int]] = None
    _global_num_tokens_gpu: Optional[torch.Tensor] = None

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        dp.buffer_hidden_size = hidden_size
        dp.buffer_dtype = dtype
        dp.buffer_device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
        global_num_tokens_gpu: Optional[torch.Tensor] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens
        cls._global_num_tokens_gpu = global_num_tokens_gpu

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def set_local_dp_buffer_len(cls, local_dp_buffer_len: int) -> None:
        cls._local_dp_buffer_len = local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_global_num_tokens_gpu(cls) -> Optional[torch.Tensor]:
        return cls._global_num_tokens_gpu

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_device

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
    global_num_tokens_gpu: Optional[torch.Tensor] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len,
        local_dp_buffer_len,
        dp_max_padding,
        global_num_tokens,
        global_num_tokens_gpu,
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer(group=group)


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def set_local_dp_buffer_len(local_dp_buffer_len: int) -> None:
    _DpGatheredBufferWrapper.set_local_dp_buffer_len(local_dp_buffer_len)


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    # Sticky within the thread: every ForwardBatch construction writes it,
    # graph runners force False around capture; readers are the EP
    # dispatchers on the same (single) forward thread.
    from sglang.srt.runtime_context import get_forward

    get_forward().set("is_extend_in_batch", is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    from sglang.srt.runtime_context import get_forward

    return get_forward().is_extend_in_batch


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp = get_flags().dp
    dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    enable_dp_attention = server_args.enable_dp_attention
    dp_size = server_args.dp_size
    attn_cp_size = server_args.attn_cp_size

    dp.enabled = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _ATTN_DP_SIZE = dp_size if enable_dp_attention else 1

    if server_args.elastic_ep_backend is not None and server_args.max_ep_size:
        _ATTN_DP_RANK = tp_rank + server_args.ep_join_rank_offset
        if server_args.is_ep_scale_joiner:
            dp.joiner_skip_all_gather = True

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    return get_flags().dp.enabled


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_dp_size


@contextmanager
def draft_dp_attention_context(
    *, enabled: bool = False, rank: int = 0, size: int = 1
):
    """Temporarily publish the draft model's DP-attention identity.

    The canonical getters stay unchanged.  This scoped mutation is used only
    around draft construction/launch and is always restored, including when a
    draft call raises.
    """

    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp = get_flags().dp
    old_enabled = dp.enabled
    old_rank = _ATTN_DP_RANK
    old_size = _ATTN_DP_SIZE
    dp.enabled = bool(enabled)
    _ATTN_DP_RANK = int(rank)
    _ATTN_DP_SIZE = int(size)
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_size
        _ATTN_DP_RANK = old_rank
        dp.enabled = old_enabled


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_attention_dp_rank()
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (
        is_partial or get_attn_tensor_model_parallel_rank() == 0
    ):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    else:
        NUM_GPUS_PER_NODE = 8
        if (
            not local_tokens.dtype.is_floating_point
            and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
        ):
            from sglang.srt.distributed.parallel_state import inplace_all_reduce

            inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

        else:
            global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    use_world = world_dp_gather_enabled()

    if get_attn_tensor_model_parallel_world_size() == 1:
        if use_world:
            torch.distributed.all_gather_into_tensor(
                global_tokens,
                local_tokens,
                group=torch.distributed.group.WORLD,
            )
        else:
            get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attn_tensor_model_parallel_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(
        get_attn_tensor_model_parallel_world_size()
    )[get_attn_tensor_model_parallel_rank()]
    get_attn_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    if use_world:
        torch.distributed.all_gather_into_tensor(
            global_tokens,
            scattered_local_tokens,
            group=torch.distributed.group.WORLD,
        )
    else:
        get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


# Variable-length DP-MoE gather (reference https://github.com/ROCm/ATOM/pull/930): instead of padding every
# rank to max_len (all_gather) or all-reducing a sum_len zero-buffer (all_reduce),
# gather exactly sum(per-rank tokens) via all_gatherv. Env-gated; only the simple
# tp_size==dp_size (attn_tp_size==1) case is supported for now (e.g. tp8dp8).
_USE_DP_GATHERV = get_bool_env_var("SGLANG_DP_USE_GATHERV")

_DP_GATHER_FP8_GROUP = 128
# Grow-only gathered fp8 payload / scales buffers, keyed by device.
_dp_gather_fp8_bufs: dict = {}


@functools.lru_cache(maxsize=1)
def _use_dp_gather_fp8() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_DP_GATHER_FP8.get()


def _get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
    key = str(device)
    bufs = _dp_gather_fp8_bufs.get(key)
    if bufs is None or bufs[0].shape[0] < rows:
        bufs = (
            torch.empty((rows, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                (rows, hidden // _DP_GATHER_FP8_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        _dp_gather_fp8_bufs[key] = bufs
    return bufs[0][:rows], bufs[1][:rows]


@triton.jit
def _dequant_per_token_group_fp8_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # HIDDEN may not be a multiple of BLOCK (e.g. DeepSeek 7168 vs BLOCK
    # 2048): the tail iteration must be masked or it reads/writes up to
    # BLOCK-1 elements past the row (cross-row corruption + OOB on the last
    # row).  HIDDEN is constexpr, so the mask folds away when it divides.
    for start in tl.static_range(0, HIDDEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * NGROUPS + offs // GROUP, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)


@triton.jit
def _mask_dp_pad_topk_ids_kernel(
    topk_ids_ptr,
    counts_ptr,
    max_len,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = row // max_len
    pos = row % max_len
    valid = pos < tl.load(counts_ptr + rank)
    if valid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(topk_ids_ptr + row * TOPK + offs, -1, mask=offs < TOPK)


def mask_dp_pad_moe_topk_ids(topk_ids: torch.Tensor) -> None:
    """Set MAX_LEN pad rows' (post-translation, local) topk_ids to -1 in place.

    Under dp-attention MAX_LEN padding the gathered MoE buffer is
    [dp_size * max_len, hidden] with rank r's real rows at
    [r*max_len, r*max_len + global_num_tokens[r]); the pad rows carry stale
    hidden values, run the router, and get dispatched into experts whose
    outputs are then discarded by the post-reorder scatter — pure wasted
    compute, and a masked-grouped-GEMM workspace blow-up when they collide
    on the same top-k.  -1 is the drop sentinel both the triton fused_moe
    (filter_expert) and the DeepGEMM EP preprocess honor; it must be applied
    AFTER the local_expert_mapping gather (a pre-translation -1 aliases to
    the mapping table's last entry).  Capture-safe: per-batch state is read
    only from the replay-updated global_num_tokens_gpu tensor.
    """
    counts = _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    max_len = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
    rows, topk = topk_ids.shape
    if max_len <= 0 or rows != counts.shape[0] * max_len:
        # Layout mismatch (e.g. non-DP or logits-path caller): do nothing.
        return
    _mask_dp_pad_topk_ids_kernel[(rows,)](
        topk_ids,
        counts,
        max_len,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
    )


def _dp_gather_via_all_gatherv_fp8(
    global_tokens: torch.Tensor,
    local_real: torch.Tensor,
    sizes: List[int],
):
    """fp8 wire format for the variable-length DP gather: quantize the local
    rows per-token-group (the SAME group-128 quantization the MoE expert GEMMs
    apply to their input downstream), gather payload (as uint8 — NCCL has no
    fp8 dtype; the gatherv leg is broadcast-only so a byte view is safe) and
    scales in two output-buffered gatherv calls, then dequantize into the
    bf16 global buffer.  Zero pad rows quantize to (q=0, s=eps) and so
    dequantize back to exact zeros — the MoE-tail invariant is preserved.
    The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rows = global_tokens.shape[0]
    hidden = global_tokens.shape[-1]
    q, s = sglang_per_token_group_quant_fp8(
        local_real.contiguous(), _DP_GATHER_FP8_GROUP
    )
    gq, gs = _get_dp_gather_fp8_bufs(rows, hidden, global_tokens.device)
    tp_group = get_tp_group()
    tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp_group.all_gatherv(s, sizes=sizes, output=gs)
    _dequant_per_token_group_fp8_kernel[(rows,)](
        gq.view(torch.float8_e4m3fn),
        gs,
        global_tokens,
        HIDDEN=hidden,
        NGROUPS=hidden // _DP_GATHER_FP8_GROUP,
        GROUP=_DP_GATHER_FP8_GROUP,
        BLOCK=2048,
    )


def is_dp_gatherv_active() -> bool:
    """Variable-length DP-MoE gather/scatter (all_gatherv + reduce_scatterv) is
    enabled and applicable to the CURRENT forward. Requires:
      - env SGLANG_DP_USE_GATHERV (default off),
      - supported layout (attn_tp_size==1, tp_size==dp_size),
      - SUM_LEN padding mode. The gatherv pair (all_gatherv + reduce_scatterv) is
        only valid under SUM_LEN; under MAX_LEN the buffer is equal-padded and the
        gather/combine use all_gather / (aiter) reduce_scatter instead. Reading the
        per-forward padding via _DpGatheredBufferWrapper.is_dp_max_padding() (set by
        set_dp_buffer_len) keeps callers that lack a ForwardBatch (e.g.
        dp_reduce_scatter_tensor) consistent."""
    return (
        _USE_DP_GATHERV
        and not world_dp_gather_enabled()
        and get_attn_tensor_model_parallel_world_size() == 1
        and get_tensor_model_parallel_world_size() == get_attention_dp_size()
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:
    """Per-rank CPU token counts for the buffer being gathered. The MoE gather
    passes a ForwardBatch (global_num_tokens_cpu); the logits gather passes a
    LogitsMetadata (global_num_tokens_for_logprob_cpu). Return the sizes that
    match the LOCAL tensor for this context, or None to fall back."""
    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):
    # attn_tp_size == 1: each DP rank contributes exactly `sizes[rank]` rows.
    # CRITICAL: the MoE downstream runs on the WHOLE `global_tokens` buffer
    # (M = global_tokens.shape[0]), so the gather MUST fill every row. We pad
    # each rank's local tensor up to sizes[rank] with zeros (matching the
    # buffer's reserved per-rank slot) so sum(sizes) == buffer rows and there
    # is no uninitialized tail for the MoE to read.
    rank = get_attention_dp_rank()
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)
    # sum(sizes) == global_tokens.shape[0] is guaranteed by the caller (else it
    # falls back to all_reduce). Pass global_tokens as the NCCL output buffer so
    # the gather writes directly into it -- avoids the previous extra full-buffer
    # torch.cat + copy_ (two ~sum(sizes)*hidden DtoD copies, ~700us/layer at c512).
    # NOTE: the fp8 branch condition must be identical on EVERY DP rank (all
    # ranks must issue the same NCCL op sequence) — env/dtype/hidden are
    # rank-uniform; never gate on per-rank state like forward_mode (ranks can
    # be extend/idle-mixed within one global forward).  Prefill-only is
    # already structural: the gatherv path runs only under SUM_LEN padding,
    # which decode-only steps and CUDA-graph capture never select.
    if (
        _use_dp_gather_fp8()
        and global_tokens.dtype == torch.bfloat16
        and global_tokens.shape[-1] % _DP_GATHER_FP8_GROUP == 0
    ):
        _dp_gather_via_all_gatherv_fp8(global_tokens, local_real, sizes)
        return
    get_tp_group().all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):
        # The gatherv per-rank sizes MUST sum to the pre-allocated global buffer
        # (the MoE runs on the whole buffer, so any unfilled tail = garbage).
        # The buffer was sized from the ceil_align'd global_num_tokens stored via
        # set_dp_buffer_len (forward_batch_info), so the authoritative sizes are
        # get_dp_global_num_tokens() — the SAME source the reduce_scatterv combine
        # uses (symmetric). _dp_gatherv_sizes() reads the raw (un-aligned, and for
        # the MoE-gather context the logprob-token) counts, which do NOT match the
        # buffer for prefill steps -> would force an all_reduce fallback.
        # Prefer the buffer-aligned sizes; fall back to the per-batch sizes only
        # if they happen to match (e.g. the logits gather path).
        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if (
        forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"

        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if is_dp_gatherv_active():
        # Variable-length combine matching all_gatherv dispatch: scatter the
        # global (sum_len) tensor back to per-rank token counts. Fall through to
        # the default reduce-scatter path if per-rank sizes are unavailable.
        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_tp_group().reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attn_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


# ---------------------------------------------------------------------------
# Two-batch-overlap (non-EP / DP TP-MoE) async gather + combine.
#
# The DP TP-MoE path (deepseek_v4) gathers local hidden -> a global buffer
# before the experts and reduce-scatters back after. For TBO we run those two
# collectives on a single shared comm stream (mirroring the mori dispatcher's
# _comm_stream) and return a CUDA event, so the op engine can yield and let the
# OTHER ubatch's attn+MoE compute run on the compute stream while this ubatch's
# gather/combine proceeds on the comm stream. Both ubatches share ONE comm
# stream -> their collectives serialize in-order (no concurrent-collective
# deadlock on the RCCL communicator), each overlapping the other's compute.
# ---------------------------------------------------------------------------
def get_dp_tbo_comm_stream() -> torch.cuda.Stream:
    from sglang.srt.runtime_context import get_stream

    return get_stream("dp_tbo_comm")


# Persistent reusable CUDA events for non-EP DP TBO, keyed by (kind, subbatch).
# CRITICAL: do NOT create a fresh event per gather/combine -- that is ~244 new
# torch.cuda.Event per forward (61 layers x 2 ubatches x 2), and the HSA signal
# pool is exhausted after a few hundred forwards -> HSA_STATUS_ERROR_OUT_OF_RESOURCES
# ("...create internal OS-specific events"). Reuse one event per (kind, subbatch)
# and just re-record it (mirrors the mori CommStreamPool event reuse).
def _tbo_event(key) -> torch.cuda.Event:
    from sglang.srt.runtime_context import get_resources

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:
    """Launch `dp_gather_partial` (all_gatherv) on the shared DP TBO comm stream;
    re-record + return a PERSISTENT event (keyed by `event_key`) that fires when
    the gather completes. Caller yields, then `compute_stream.wait_event(ev)`
    before reading `global_tokens`."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    # Keep buffers alive across streams (caching allocator).
    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)  # inputs were produced on the compute stream
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


# Persistent grow-only buffers for non-EP DP TBO, keyed by (kind, tbo_subbatch).
# Reused across ALL layers (and forwards) so the caching allocator does not churn
# a fresh per-layer `torch.empty` for the 8x DP-gather / combine buffers. That
# churn (different sizes per forward x 2 ubatches x 61 layers, kept alive by the
# comm-stream record_stream) ballooned `reserved` to ~270GB and tripped
# HSA_STATUS_ERROR_OUT_OF_RESOURCES at large prefill chunks, even though the live
# (allocated) working set was only ~10GB.
_TBO_PERSIST_BUF: dict = {}


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return a [rows, hidden] view of a grow-only persistent buffer for `key`.
    Reallocates only when the request exceeds the cached capacity / changes
    dtype|hidden. Caller must treat the returned view as scratch (overwritten)."""
    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:
    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
    Matches the gatherv (SUM_LEN) path."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_tp_group().reduce_scatterv(global_tokens, output=output_local, sizes=sizes)
        ev.record(comm)
    return ev


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attn_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE."""
    from sglang.srt.runtime_context import get_server_args

    sa = get_server_args()
    return sa.attn_cp_size > sa.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attn_tp_group().all_gather(input, output_tensor_list=output_list)


# ---------------------------------------------------------------------------
# WeLMv4 explicit row-layout collectives.
#
# These APIs intentionally do not read _DpGatheredBufferWrapper.  Their row
# metadata belongs to one ForwardBatch (or one graph runner's private static
# buffers), so scheduler overlap cannot make a later batch overwrite it.
# ---------------------------------------------------------------------------


def _welm_validate_scratch_tensor(
    tensor: torch.Tensor,
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if (
        tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
    ):
        raise RuntimeError(
            f"WeLMv4 {name} scratch changed after allocation: "
            f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}; expected shape={tuple(shape)}, "
            f"dtype={dtype}, device={device}"
        )


def _welm_prepare_idle_buffers(
    row_view: "WelmDpRowView",
    like: torch.Tensor,
    *,
    rows: int,
) -> None:
    """Allocate the eager-idle tensors once per ForwardBatch row view."""

    shape = (int(rows), *like.shape[1:])
    if row_view._idle_hidden_buffer is None:
        row_view._idle_hidden_buffer = like.new_zeros(shape)
        row_view._idle_residual_buffer = like.new_zeros(
            shape, dtype=torch.float32
        )
        return
    _welm_validate_scratch_tensor(
        row_view._idle_hidden_buffer,
        shape=shape,
        dtype=like.dtype,
        device=like.device,
        name="idle hidden",
    )
    assert row_view._idle_residual_buffer is not None
    _welm_validate_scratch_tensor(
        row_view._idle_residual_buffer,
        shape=shape,
        dtype=torch.float32,
        device=like.device,
        name="idle residual",
    )


def welm_dp_prepare_full_transport_scratch(
    row_view: "WelmDpRowView",
    like: torch.Tensor,
) -> None:
    """Prepare and refresh FULL/local-EP transport scratch before layers.

    This function must run once on every model forward.  Graph warmup may have
    allocated the buffers already, but the comparisons below still execute in
    the real capture so replay recomputes masks from ``real_rows_gpu``.
    """

    device = like.device
    counts = row_view.real_rows_gpu
    if counts.device != device:
        raise RuntimeError(
            "WeLMv4 real-row metadata and hidden states are on different devices"
        )
    global_rows = int(row_view.global_slot_rows)
    gather_shape = (global_rows, *like.shape[1:])
    if row_view._replicate_gather_buffer is None:
        row_view._replicate_gather_buffer = like.new_empty(gather_shape)
    else:
        _welm_validate_scratch_tensor(
            row_view._replicate_gather_buffer,
            shape=gather_shape,
            dtype=like.dtype,
            device=device,
            name="replicate-gather",
        )

    if row_view._segmented_positions is None:
        row_view._segmented_positions = torch.empty(
            (global_rows,), dtype=torch.int32, device=device
        )
        for start, slot_rows in zip(
            row_view.slot_offsets_cpu, row_view.slot_rows_cpu
        ):
            slot_rows = int(slot_rows)
            if slot_rows > 0:
                torch.arange(
                    slot_rows,
                    dtype=torch.int32,
                    device=device,
                    out=row_view._segmented_positions.narrow(
                        0, int(start), slot_rows
                    ),
                )
        row_view._segmented_valid_mask = torch.empty(
            (global_rows,), dtype=torch.bool, device=device
        )
        row_view._segmented_invalid_mask = torch.empty(
            (global_rows,), dtype=torch.bool, device=device
        )
    else:
        _welm_validate_scratch_tensor(
            row_view._segmented_positions,
            shape=(global_rows,),
            dtype=torch.int32,
            device=device,
            name="segmented positions",
        )
    assert row_view._segmented_valid_mask is not None
    assert row_view._segmented_invalid_mask is not None
    _welm_validate_scratch_tensor(
        row_view._segmented_valid_mask,
        shape=(global_rows,),
        dtype=torch.bool,
        device=device,
        name="segmented valid mask",
    )
    _welm_validate_scratch_tensor(
        row_view._segmented_invalid_mask,
        shape=(global_rows,),
        dtype=torch.bool,
        device=device,
        name="segmented invalid mask",
    )
    for dp_id, (start, slot_rows) in enumerate(
        zip(row_view.slot_offsets_cpu, row_view.slot_rows_cpu)
    ):
        slot_rows = int(slot_rows)
        if slot_rows > 0:
            torch.lt(
                row_view._segmented_positions.narrow(
                    0, int(start), slot_rows
                ),
                counts[dp_id],
                out=row_view._segmented_valid_mask.narrow(
                    0, int(start), slot_rows
                ),
            )
    torch.logical_not(
        row_view._segmented_valid_mask,
        out=row_view._segmented_invalid_mask,
    )
    if int(row_view.local_real_rows) == 0:
        _welm_prepare_idle_buffers(
            row_view, like, rows=int(row_view.local_slot_rows)
        )


def welm_dp_prepare_attn_scattered_transport_scratch(
    row_view: "WelmDpRowView",
    like: torch.Tensor,
    *,
    dp_rank: int,
    attn_tp_rank: int,
    attn_tp_size: int,
) -> None:
    """Prepare and refresh NORMAL-AG scattered masks before the layer loop."""

    full_slot_rows = int(row_view.local_slot_rows)
    if attn_tp_size < 1 or full_slot_rows % int(attn_tp_size) != 0:
        raise RuntimeError(
            "WeLMv4 DP attention slot must be divisible by attention-TP: "
            f"slot={full_slot_rows}, attn_tp={attn_tp_size}"
        )
    shard_rows = full_slot_rows // int(attn_tp_size)
    device = like.device
    counts = row_view.real_rows_gpu
    if counts.device != device:
        raise RuntimeError(
            "WeLMv4 real-row metadata and hidden states are on different devices"
        )
    key = (
        int(dp_rank),
        int(attn_tp_rank),
        int(attn_tp_size),
        int(shard_rows),
        device,
    )
    if row_view._attn_scattered_positions is None:
        row_view._attn_scattered_key = key
        row_view._attn_scattered_positions = torch.empty(
            (shard_rows,), dtype=torch.int32, device=device
        )
        if shard_rows > 0:
            start = int(attn_tp_rank) * shard_rows
            torch.arange(
                start,
                start + shard_rows,
                dtype=torch.int32,
                device=device,
                out=row_view._attn_scattered_positions,
            )
        row_view._attn_scattered_valid_mask = torch.empty(
            (shard_rows,), dtype=torch.bool, device=device
        )
        row_view._attn_scattered_invalid_mask = torch.empty(
            (shard_rows,), dtype=torch.bool, device=device
        )
    elif row_view._attn_scattered_key != key:
        raise RuntimeError(
            "WeLMv4 attention-scattered scratch topology changed after allocation"
        )
    assert row_view._attn_scattered_valid_mask is not None
    assert row_view._attn_scattered_invalid_mask is not None
    torch.lt(
        row_view._attn_scattered_positions,
        counts[int(dp_rank)],
        out=row_view._attn_scattered_valid_mask,
    )
    torch.logical_not(
        row_view._attn_scattered_valid_mask,
        out=row_view._attn_scattered_invalid_mask,
    )
    if int(row_view.local_real_rows) == 0:
        _welm_prepare_idle_buffers(row_view, like, rows=shard_rows)


def welm_dp_idle_buffers(
    row_view: "WelmDpRowView",
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = row_view._idle_hidden_buffer
    residual = row_view._idle_residual_buffer
    if hidden is None or residual is None:
        raise RuntimeError(
            "WeLMv4 eager-idle transport scratch was not prepared before layers"
        )
    return hidden, residual


def welm_dp_segmented_valid_mask(
    row_view: "WelmDpRowView",
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return validity for ``[dp0 slot, dp1 slot, ...]`` gathered rows."""

    mask = row_view._segmented_valid_mask
    if mask is None:
        raise RuntimeError(
            "WeLMv4 FULL transport scratch was not prepared before layers"
        )
    if device is not None and mask.device != device:
        raise RuntimeError("WeLMv4 segmented mask is on the wrong device")
    return mask


def welm_dp_segmented_invalid_mask(
    row_view: "WelmDpRowView",
) -> torch.Tensor:
    mask = row_view._segmented_invalid_mask
    if mask is None:
        raise RuntimeError(
            "WeLMv4 FULL transport scratch was not prepared before layers"
        )
    return mask


def welm_dp_attn_scattered_valid_mask(
    row_view: "WelmDpRowView",
    *,
    dp_rank: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    local_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Validity for the contiguous attn-TP shard produced by ReduceScatter."""

    full_slot_rows = int(row_view.local_slot_rows)
    if attn_tp_size < 1 or full_slot_rows % attn_tp_size != 0:
        raise RuntimeError(
            "WeLMv4 DP attention slot must be divisible by attention-TP: "
            f"slot={full_slot_rows}, attn_tp={attn_tp_size}"
        )
    shard_rows = full_slot_rows // attn_tp_size
    if int(local_rows) != shard_rows:
        raise RuntimeError(
            "WeLMv4 scattered tensor does not match its row-layout shard: "
            f"tensor={local_rows}, expected={shard_rows}"
        )
    key = (
        int(dp_rank),
        int(attn_tp_rank),
        int(attn_tp_size),
        int(shard_rows),
        device,
    )
    if row_view._attn_scattered_key != key:
        raise RuntimeError(
            "WeLMv4 NORMAL transport scratch was not prepared for this shard"
        )
    mask = row_view._attn_scattered_valid_mask
    if mask is None:
        raise RuntimeError(
            "WeLMv4 NORMAL transport scratch was not prepared before layers"
        )
    return mask


def welm_dp_attn_scattered_invalid_mask(
    row_view: "WelmDpRowView",
) -> torch.Tensor:
    mask = row_view._attn_scattered_invalid_mask
    if mask is None:
        raise RuntimeError(
            "WeLMv4 NORMAL transport scratch was not prepared before layers"
        )
    return mask


def welm_dp_replicate_gather(
    local_tokens: torch.Tensor,
    row_view: "WelmDpRowView",
    *,
    group: GroupCoordinator,
    dp_rank: int,
    contribute: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather one replica per DP shard into fixed segmented slots.

    A zero staging tensor plus SUM is used intentionally: it supports unequal
    eager rows, fixed graph slots, and attention-TP replica de-duplication with
    one rank-uniform collective sequence.
    """

    if local_tokens.dim() < 1:
        raise RuntimeError("WeLMv4 DP gather requires a row-major tensor")
    dp_rank = int(dp_rank)
    if not 0 <= dp_rank < len(row_view.slot_rows_cpu):
        raise RuntimeError("WeLMv4 DP rank is outside the row-layout")
    global_rows = int(row_view.global_slot_rows)
    expected_shape = (global_rows, *local_tokens.shape[1:])
    if output is None:
        output = row_view._replicate_gather_buffer
        if output is None:
            raise RuntimeError(
                "WeLMv4 replicate-gather scratch was not prepared before layers"
            )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "WeLMv4 cached gather output shape mismatch: "
                f"{tuple(output.shape)} != {expected_shape}"
            )
        if output.dtype != local_tokens.dtype or output.device != local_tokens.device:
            raise RuntimeError(
                "WeLMv4 cached gather output dtype/device does not match input"
            )
        output.zero_()
    else:
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "WeLMv4 DP gather output shape mismatch: "
                f"{tuple(output.shape)} != {expected_shape}"
            )
        output.zero_()

    local_slot_rows = int(row_view.local_slot_rows)
    if local_tokens.shape[0] > local_slot_rows:
        raise RuntimeError(
            "WeLMv4 local tensor exceeds its DP slot: "
            f"{local_tokens.shape[0]} > {local_slot_rows}"
        )
    if contribute and local_tokens.shape[0] > 0:
        start = int(row_view.slot_offsets_cpu[dp_rank])
        output.narrow(0, start, local_tokens.shape[0]).copy_(local_tokens)
    if global_rows == 0:
        # Every rank sees the same empty layout, so there is no peer payload
        # to exchange.  Avoid issuing a zero-element HCCL collective.
        return output
    return group.all_reduce(output)


def welm_dp_local_slot_slice(
    global_tokens: torch.Tensor,
    row_view: "WelmDpRowView",
    *,
    dp_rank: int,
) -> torch.Tensor:
    """Restore this rank's complete fixed DP slot after MoE combine."""

    dp_rank = int(dp_rank)
    start = int(row_view.slot_offsets_cpu[dp_rank])
    rows = int(row_view.local_slot_rows)
    if start < 0 or start + rows > global_tokens.shape[0]:
        raise RuntimeError(
            "WeLMv4 local DP slot is outside the gathered tensor: "
            f"[{start}, {start + rows}) vs {global_tokens.shape[0]}"
        )
    # A row-prefix/slice of the contiguous [rows, hidden] MoE output is already
    # contiguous.  Return the view so the DP combine path does not allocate a
    # fresh local tensor on every decoder layer.
    return global_tokens.narrow(0, start, rows)


def welm_reconstruct_attn_scattered_residual_rows(
    residual: torch.Tensor,
    custom_last_index: torch.Tensor,
    prompt_view: "WelmDpRowView",
    request_view: "WelmDpRowView",
    *,
    group: GroupCoordinator,
    dp_rank: int,
    attn_tp_rank: int,
    attn_tp_size: int,
) -> torch.Tensor:
    """Rebuild FP32 T->B residual rows from their unique attn-TP owners."""

    if residual.dtype != torch.float32:
        raise RuntimeError(
            f"WeLMv4 mirror residual must be FP32, got {residual.dtype}"
        )
    prompt_slot_rows = int(prompt_view.local_slot_rows)
    if prompt_slot_rows % int(attn_tp_size) != 0:
        raise RuntimeError(
            "WeLMv4 prompt slot is not divisible by attention-TP"
        )
    shard_rows = prompt_slot_rows // int(attn_tp_size)
    if residual.shape[0] != shard_rows:
        raise RuntimeError(
            "WeLMv4 mirror residual shard shape mismatch: "
            f"{residual.shape[0]} != {shard_rows}"
        )

    request_real_rows = int(request_view.local_real_rows)
    request_slot_rows = int(request_view.local_slot_rows)
    if custom_last_index.numel() != request_real_rows:
        raise RuntimeError(
            "WeLMv4 mirror request rows do not match custom_last_index: "
            f"{request_real_rows} != {custom_last_index.numel()}"
        )
    indices = custom_last_index.to(torch.int64)
    owner = torch.div(indices, shard_rows, rounding_mode="floor")
    local_offset = indices - int(attn_tp_rank) * shard_rows
    owned = owner == int(attn_tp_rank)
    safe_offset = torch.where(owned, local_offset, torch.zeros_like(local_offset))
    selected = residual.index_select(0, safe_offset.to(torch.long))
    selected = torch.where(owned[:, None], selected, torch.zeros_like(selected))
    partial = residual.new_zeros((request_slot_rows, *residual.shape[1:]))
    if request_real_rows > 0:
        partial.narrow(0, 0, request_real_rows).copy_(selected)
    combined = group.all_reduce(partial)
    # The fixed slot is required only for the rank-uniform collective.  Keep
    # subsequent mirror attention/norm on the real B-row prefix; the FULL
    # transport stages it back into its slot without a per-layer pad tensor.
    return combined.narrow(0, 0, request_real_rows)
