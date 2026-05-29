from enum import IntEnum
from typing import Tuple, Optional
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2

def _get_num_vectorcore() -> int:
    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, num_vectorcore = get_device_properties()
        return int(num_vectorcore)
    except Exception:
        device = torch.npu.current_device()
        device_properties = triton.runtime.driver.active.utils.get_device_properties(
            device
        )
        num_vectorcore = int(device_properties.get("num_vectorcore", -1))
        if num_vectorcore <= 0:
            raise RuntimeError("Failed to detect Ascend vector core count.")
        return num_vectorcore

def _get_npu_aicore_num() -> int:
    device = torch.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    aicore_num = int(props.get("num_aicore", 0))
    if aicore_num <= 0:
        raise RuntimeError(f"Failed to query num_aicore from device properties: {props}")
    return aicore_num


NUM_VECTOR_CORES = _get_num_vectorcore()
NUM_CUBE_CORES = _get_npu_aicore_num()

@triton.jit(do_not_specialize=["batch_size", "topk", "parent_stride"])
def _build_tree_efficient_kernel(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    batch_size,
    topk,
    parent_stride,
    TREE_MASK_MODE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_DRAFT)
    offsets_i64 = offsets.to(tl.int64)
    row_mask = offsets < DRAFT_TOKEN_NUM
    selected_stride = DRAFT_TOKEN_NUM - 1

    for bs in tl.range(pid, batch_size, num_programs):
        bs_i64 = bs.to(tl.int64)
        batch_token_base = bs_i64 * DRAFT_TOKEN_NUM
        batch_parent_base = bs_i64 * parent_stride
        batch_selected_base = bs_i64 * selected_stride
        seq_len = tl.load(verified_seq_len_ptr + bs_i64)

        seq_tree_idx = batch_token_base * DRAFT_TOKEN_NUM
        for prev_bs in tl.range(0, bs, 1):
            prev_bs_i64 = prev_bs.to(tl.int64)
            prev_seq_len = tl.load(verified_seq_len_ptr + prev_bs_i64)
            seq_tree_idx += prev_seq_len * DRAFT_TOKEN_NUM

        tl.store(
            retrive_index_ptr + batch_token_base + offsets_i64,
            batch_token_base + offsets_i64,
            mask=row_mask,
        )
        tl.store(
            retrive_next_token_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(
            retrive_next_sibling_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(positions_ptr + batch_token_base, seq_len)

        for i in range(DRAFT_TOKEN_NUM - 1, 0, -1):
            current_selected_offset = batch_selected_base + (i - 1)
            parent_tb_idx = tl.load(selected_index_ptr + current_selected_offset) // topk
            parent_position = 0

            if parent_tb_idx > 0:
                parent_token_idx = tl.load(parent_list_ptr + batch_parent_base + parent_tb_idx)
                parent_position = DRAFT_TOKEN_NUM
                for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                    candidate_token_idx = tl.load(
                        selected_index_ptr + batch_selected_base + candidate_pos
                    )
                    if parent_position == DRAFT_TOKEN_NUM and candidate_token_idx == parent_token_idx:
                        parent_position = candidate_pos + 1

            if parent_position != DRAFT_TOKEN_NUM:
                existing_next = tl.load(
                    retrive_next_token_ptr + batch_token_base + parent_position
                )
                if existing_next == -1:
                    tl.store(
                        retrive_next_token_ptr + batch_token_base + parent_position, i
                    )
                else:
                    tl.store(
                        retrive_next_token_ptr + batch_token_base + parent_position, i
                    )
                    tl.store(
                        retrive_next_sibling_ptr + batch_token_base + i, existing_next
                    )

        for tid in range(DRAFT_TOKEN_NUM):
            if TREE_MASK_MODE == 0:
                token_tree_idx = seq_tree_idx + (seq_len + DRAFT_TOKEN_NUM) * tid + seq_len
            else:
                token_tree_idx = batch_token_base * DRAFT_TOKEN_NUM + batch_token_base

            tl.store(
                tree_mask_ptr + token_tree_idx + offsets_i64,
                offsets == 0,
                mask=row_mask,
            )

            if tid > 0:
                position = 0
                cur_position = tid - 1
                active = 1

                for _ in range(DRAFT_TOKEN_NUM):
                    if active == 1:
                        position += 1
                        tl.store(
                            tree_mask_ptr + token_tree_idx + 1 + cur_position,
                            True,
                        )
                        parent_tb_idx = (
                            tl.load(selected_index_ptr + batch_selected_base + cur_position)
                            // topk
                        )
                        if parent_tb_idx == 0:
                            active = 0
                        else:
                            token_idx = tl.load(
                                parent_list_ptr + batch_parent_base + parent_tb_idx
                            )
                            next_position = DRAFT_TOKEN_NUM - 1
                            for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                                candidate_token_idx = tl.load(
                                    selected_index_ptr + batch_selected_base + candidate_pos
                                )
                                if (
                                    next_position == DRAFT_TOKEN_NUM - 1
                                    and candidate_token_idx == token_idx
                                ):
                                    next_position = candidate_pos
                            cur_position = next_position

                tl.store(positions_ptr + batch_token_base + tid, seq_len + position)


def build_tree_kernel_efficient_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    batch_size = int(verified_seq_len.numel())
    parent_stride = topk * (depth - 1) + 1
    num_cores = NUM_VECTOR_CORES
    block_draft = triton.next_power_of_2(draft_token_num)

    _build_tree_efficient_kernel[(num_cores,)](
        parent_list_ptr=parent_list.reshape(-1),
        selected_index_ptr=selected_index.reshape(-1),
        verified_seq_len_ptr=verified_seq_len.reshape(-1),
        tree_mask_ptr=tree_mask.reshape(-1),
        positions_ptr=positions.reshape(-1),
        retrive_index_ptr=retrive_index.reshape(-1),
        retrive_next_token_ptr=retrive_next_token.reshape(-1),
        retrive_next_sibling_ptr=retrive_next_sibling.reshape(-1),
        batch_size=batch_size,
        topk=topk,
        parent_stride=parent_stride,
        TREE_MASK_MODE=int(tree_mask_mode),
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK_DRAFT=block_draft,
    )


ASSIGN_TO_POOL = 0
RETRIEVE_FROM_POOL = 1
MAX_STEP = 6

@triton.jit(do_not_specialize=["batch_size", "pool_len"])
def _cache_location_assigns_kernel(
    req_pool_indices_ptr,
    token_pool_ptr,
    start_offset_ptr,
    end_offset_ptr,
    out_cache_loc_ptr,
    batch_size,
    pool_len,
    ASSIGN_MODE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BS_UPPER: tl.constexpr,
    MAX_STEP_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    for row_idx in tl.range(pid, batch_size, NUM_CORES):
        req_idx = tl.load(req_pool_indices_ptr + row_idx)
        kv_start = tl.load(start_offset_ptr + row_idx)
        kv_end = tl.load(end_offset_ptr + row_idx)
        step = kv_end - kv_start

        prefix_idx = tl.arange(0, BS_UPPER)
        prefix_start = tl.load(start_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0)
        prefix_end = tl.load(end_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0)
        cache_idx_start = tl.sum(prefix_end - prefix_start, axis=0)

        token_ptr = token_pool_ptr + req_idx * pool_len + kv_start
        cache_ptr = out_cache_loc_ptr + cache_idx_start
        elem = tl.arange(0, MAX_STEP_CONST)
        mask = elem < step

        if ASSIGN_MODE == 0:
            data = tl.load(cache_ptr + elem, mask=mask, other=0)
            tl.store(token_ptr + elem, data, mask=mask)
        else:
            data = tl.load(token_ptr + elem, mask=mask, other=0)
            tl.store(cache_ptr + elem, data, mask=mask)


def _cache_location_assigns_impl(
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    assign_mode: int = ASSIGN_TO_POOL,
    num_cores: int | None = None,
) -> torch.Tensor:
    if assign_mode not in (ASSIGN_TO_POOL, RETRIEVE_FROM_POOL):
        raise ValueError("assign_mode must be 0 or 1.")
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return token_pool if assign_mode == ASSIGN_TO_POOL else out_cache_loc
    if num_cores is None:
        num_cores = NUM_VECTOR_CORES
    num_cores = int(max(1, num_cores))
    bs_upper = int(triton.next_power_of_2(batch_size))
    _cache_location_assigns_kernel[(num_cores,)](
        req_pool_indices,
        token_pool,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
        int(token_pool.shape[1]),
        ASSIGN_MODE=assign_mode,
        NUM_CORES=num_cores,
        BS_UPPER=bs_upper,
        MAX_STEP_CONST=MAX_STEP,
    )
    return token_pool if assign_mode == ASSIGN_TO_POOL else out_cache_loc


def cache_loc_assign(
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> torch.Tensor:
    return _cache_location_assigns_impl(
        req_pool_indices,
        token_pool,
        start_offset,
        end_offset,
        out_cache_loc,
        ASSIGN_TO_POOL,
    )


def cache_loc_update(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc_copy: torch.Tensor,
) -> torch.Tensor:
    return _cache_location_assigns_impl(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc_copy,
        RETRIEVE_FROM_POOL,
    )

@triton.jit
def verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    accept_index_stride,
    num_draft_tokens: tl.constexpr,
):
    req_idx = tl.program_id(0)
    base = req_idx * num_draft_tokens

    last_accepted_idx = tl.load(retrive_index + base).to(tl.int32)
    tl.store(accept_index + req_idx * accept_index_stride, last_accepted_idx)

    num_accepted = 0
    rejected = False

    for i in range(1, num_draft_tokens):
        if not rejected:
            draft_token = tl.load(candidates + base + i).to(tl.int32)
            target_token = tl.load(target_predict + base + i - 1).to(tl.int32)

            if draft_token == target_token:
                draft_idx = tl.load(retrive_index + base + i).to(tl.int32)
                tl.store(predicts + last_accepted_idx, target_token)
                num_accepted += 1
                tl.store(
                    accept_index + req_idx * accept_index_stride + num_accepted,
                    draft_idx,
                )
                last_accepted_idx = draft_idx
            else:
                rejected = True

    final_pos = last_accepted_idx - base
    final_token = tl.load(target_predict + base + final_pos).to(tl.int32)

    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, final_token)


def verify_tree_greedy_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    bs, num_draft_tokens = candidates.shape

    verify_tree_greedy_kernel[(bs,)](
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        accept_index_stride=accept_index.shape[1],
        num_draft_tokens=num_draft_tokens,
    )

@triton.jit
def alloc_extend_kernel_triton(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens,
    BLOCK_SIZE: tl.constexpr = 2048,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid).to(tl.int64)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    blk_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(num_loop):
        offset_many_page = blk_offset + i * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )

    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS = 128


@triton.jit(do_not_specialize=["batch_size", "pool_len"])
def _alloc_extend_assign_req_to_token_pool_kernel(
    req_pool_indices_ptr,
    req_to_token_ptr,
    pre_lens_ptr,
    seq_lens_ptr,
    free_page_ptr,
    batch_size,
    pool_len,
    page_size: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BS_PAD: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = tl.arange(0, BS_PAD)
    blk_offsets = tl.arange(0, BLOCK_TOKENS)
    page_offsets = tl.arange(0, page_size)

    for row_idx in tl.range(pid, batch_size, NUM_CORES):
        req_idx = tl.load(req_pool_indices_ptr + row_idx).to(tl.int64)
        pre_len = tl.load(pre_lens_ptr + row_idx).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + row_idx).to(tl.int64)
        token_row_ptr = req_to_token_ptr + req_idx * pool_len

        prev_mask = (row_offsets < row_idx) & (row_offsets < batch_size)
        prev_seq = tl.load(seq_lens_ptr + row_offsets, mask=prev_mask, other=0).to(tl.int64)
        prev_pre = tl.load(pre_lens_ptr + row_offsets, mask=prev_mask, other=0).to(tl.int64)

        prev_pages_after = (prev_seq + page_size - 1) // page_size
        prev_pages_before = (prev_pre + page_size - 1) // page_size
        new_page_start_loc = tl.sum(prev_pages_after - prev_pages_before, axis=0)

        last_loc = tl.full((), -1, tl.int64)
        if pre_len > 0:
            last_loc = tl.load(token_row_ptr + pre_len - 1).to(tl.int64)

        aligned_pre_end = ((pre_len + page_size - 1) // page_size) * page_size
        num_part1 = tl.maximum(tl.minimum(seq_len, aligned_pre_end) - pre_len, 0)
        if num_part1 > 0:
            tl.store(
                token_row_ptr + pre_len + page_offsets,
                last_loc + 1 + page_offsets,
                mask=page_offsets < num_part1,
            )

        if pre_len + num_part1 != seq_len:
            full_page_end = (seq_len // page_size) * page_size
            num_part2 = full_page_end - aligned_pre_end
            if num_part2 > 0:
                num_loops = tl.cdiv(num_part2, BLOCK_TOKENS)
                for i in range(num_loops):
                    offset_many_page = blk_offsets + i * BLOCK_TOKENS
                    valid = offset_many_page < num_part2
                    page_start = tl.load(
                        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
                        mask=valid,
                        other=0,
                    ).to(tl.int64)
                    tl.store(
                        token_row_ptr + pre_len + num_part1 + offset_many_page,
                        page_start * page_size + offset_many_page % page_size,
                        mask=valid,
                    )

            if pre_len + num_part1 + num_part2 != seq_len:
                num_pages_before = (pre_len + page_size - 1) // page_size
                num_pages_after = (seq_len + page_size - 1) // page_size
                num_page_start_loc_self = num_pages_after - num_pages_before
                num_part3 = seq_len - full_page_end
                last_page = tl.load(
                    free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
                ).to(tl.int64)
                tl.store(
                    token_row_ptr + pre_len + num_part1 + num_part2 + page_offsets,
                    last_page * page_size + page_offsets,
                    mask=page_offsets < num_part3,
                )


def alloc_extend_assign_req_to_token_pool_triton(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    free_pages: torch.Tensor,
    page_size: int,
    num_cores: int | None = None,
) -> torch.Tensor:
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return req_to_token
    if batch_size > FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS:
        raise ValueError(
            f"batch_size={batch_size} exceeds fused kernel limit "
            f"{FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS}"
        )
    if num_cores is None:
        num_cores = NUM_VECTOR_CORES
    _alloc_extend_assign_req_to_token_pool_kernel[(int(max(1, num_cores)),)](
        req_pool_indices,
        req_to_token,
        prefix_lens,
        seq_lens,
        free_pages,
        batch_size,
        int(req_to_token.shape[1]),
        page_size=page_size,
        NUM_CORES=int(max(1, num_cores)),
        BS_PAD=FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS,
        BLOCK_TOKENS=256,
    )
    return req_to_token

@triton.jit(
    do_not_specialize=[
        "raw_bs",
        "bs",
        "topk",
        "raw_num_token",
        "num_tokens",
        "speculative_num_steps",
    ]
)
def _draft_replay_pack_fused_kernel(
    dst_seq_lens_ptr,
    src_seq_lens_ptr,
    dst_topk_p_ptr,
    src_topk_p_ptr,
    dst_topk_index_ptr,
    src_topk_index_ptr,
    dst_req_pool_indices_ptr,
    src_req_pool_indices_ptr,
    dst_out_cache_loc_ptr,
    src_out_cache_loc_ptr,
    dst_positions_ptr,
    src_positions_ptr,
    raw_bs,
    bs,
    topk,
    seq_len_fill_value,
    raw_num_token,
    num_tokens,
    speculative_num_steps,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Phase 1: seq-domain pack and padding
    for row in tl.range(pid, bs, num_programs):
        row_i64 = row.to(tl.int64)
        if row < raw_bs:
            seq_len = tl.load(src_seq_lens_ptr + row_i64)
            tl.store(dst_seq_lens_ptr + row_i64, seq_len)

            req_idx = tl.load(src_req_pool_indices_ptr + row_i64)
            tl.store(dst_req_pool_indices_ptr + row_i64, req_idx)

            topk_base = row_i64 * topk
            num_loops = tl.cdiv(topk, BLOCK_TOPK)
            for i in range(num_loops):
                cur = topk_offsets + i * BLOCK_TOPK
                mask = cur < topk
                src_topk_p = tl.load(src_topk_p_ptr + topk_base + cur, mask=mask, other=0.0)
                src_topk_index = tl.load(
                    src_topk_index_ptr + topk_base + cur, mask=mask, other=0
                )
                tl.store(dst_topk_p_ptr + topk_base + cur, src_topk_p, mask=mask)
                tl.store(dst_topk_index_ptr + topk_base + cur, src_topk_index, mask=mask)
        else:
            tl.store(dst_seq_lens_ptr + row_i64, seq_len_fill_value)

    # Phase 2: token-domain pack and tail zero padding
    for block_idx in tl.range(pid, tl.cdiv(num_tokens, BLOCK_SIZE), num_programs):
        token_offset = block_idx * BLOCK_SIZE + offsets
        token_mask = token_offset < num_tokens
        copy_mask = token_offset < raw_num_token
        src_pos = tl.load(src_positions_ptr + token_offset, mask=copy_mask, other=0)
        tl.store(dst_positions_ptr + token_offset, src_pos, mask=token_mask)

    raw_out_len = raw_num_token * speculative_num_steps
    out_len = num_tokens * speculative_num_steps
    for block_idx in tl.range(pid, tl.cdiv(out_len, BLOCK_SIZE), num_programs):
        out_offset = block_idx * BLOCK_SIZE + offsets
        out_mask = out_offset < out_len
        copy_mask = out_offset < raw_out_len
        src_loc = tl.load(src_out_cache_loc_ptr + out_offset, mask=copy_mask, other=0)
        tl.store(dst_out_cache_loc_ptr + out_offset, src_loc, mask=out_mask)


def draft_replay_pack_npu_triton_fused(
    dst_seq_lens: torch.Tensor,
    src_seq_lens: torch.Tensor,
    dst_out_cache_loc: torch.Tensor,
    src_out_cache_loc: torch.Tensor,
    dst_positions: torch.Tensor,
    src_positions: torch.Tensor,
    dst_topk_p: torch.Tensor,
    src_topk_p: torch.Tensor,
    dst_topk_index: torch.Tensor,
    src_topk_index: torch.Tensor,
    dst_req_pool_indices: torch.Tensor,
    src_req_pool_indices: torch.Tensor,
    raw_bs: int,
    bs: int,
    topk: int,
    speculative_num_steps: int,
    seq_len_fill_value: int,
    num_cores: int | None = None,
) -> None:
    if bs <= 0:
        return
    if num_cores is None:
        num_cores = NUM_VECTOR_CORES
    num_cores = int(max(1, num_cores))
    raw_num_token = raw_bs * topk
    num_tokens = bs * topk

    _draft_replay_pack_fused_kernel[(num_cores,)](
        dst_seq_lens,
        src_seq_lens,
        dst_topk_p,
        src_topk_p,
        dst_topk_index,
        src_topk_index,
        dst_req_pool_indices,
        src_req_pool_indices,
        dst_out_cache_loc,
        src_out_cache_loc,
        dst_positions,
        src_positions,
        raw_bs,
        bs,
        topk,
        seq_len_fill_value,
        raw_num_token,
        num_tokens,
        speculative_num_steps,
        BLOCK_TOPK=16,
        BLOCK_SIZE=256,
    )

@triton.jit(
    do_not_specialize=["raw_bs", "topk", "hidden_size", "copy_len", "pool_len"]
)
def _draft_future_replay_prepare_kernel(
    indices_ptr,
    src_topk_p_ptr,
    src_topk_index_ptr,
    src_hidden_states_ptr,
    src_verified_id_ptr,
    src_new_seq_lens_ptr,
    src_seq_lens_ptr,
    src_req_pool_indices_ptr,
    req_to_token_ptr,
    dst_topk_p_ptr,
    dst_topk_index_ptr,
    dst_hidden_states_ptr,
    dst_verified_id_ptr,
    dst_new_seq_lens_ptr,
    dst_positions_ptr,
    dst_out_cache_loc_ptr,
    raw_bs,
    topk,
    hidden_size,
    copy_len,
    pool_len,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_COPY: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    copy_offsets = tl.arange(0, BLOCK_COPY)

    topk_loops = tl.cdiv(topk, BLOCK_TOPK)
    hidden_loops = tl.cdiv(hidden_size, BLOCK_HIDDEN)
    copy_loops = tl.cdiv(copy_len, BLOCK_COPY)

    for row in tl.range(pid, raw_bs, num_programs):
        row_i64 = row.to(tl.int64)
        src_row = tl.load(indices_ptr + row_i64).to(tl.int64)
        seq_len = tl.load(src_seq_lens_ptr + row_i64)
        req_idx = tl.load(src_req_pool_indices_ptr + row_i64).to(tl.int64)

        tl.store(dst_verified_id_ptr + row_i64, tl.load(src_verified_id_ptr + src_row))
        tl.store(
            dst_new_seq_lens_ptr + row_i64,
            tl.load(src_new_seq_lens_ptr + src_row),
        )

        src_topk_base = src_row * topk
        dst_topk_base = row_i64 * topk
        for i in range(topk_loops):
            cur = topk_offsets + i * BLOCK_TOPK
            mask = cur < topk
            src_topk_p = tl.load(src_topk_p_ptr + src_topk_base + cur, mask=mask, other=0.0)
            src_topk_index = tl.load(
                src_topk_index_ptr + src_topk_base + cur, mask=mask, other=0
            )
            tl.store(dst_topk_p_ptr + dst_topk_base + cur, src_topk_p, mask=mask)
            tl.store(
                dst_topk_index_ptr + dst_topk_base + cur, src_topk_index, mask=mask
            )
            tl.store(
                dst_positions_ptr + dst_topk_base + cur,
                seq_len.to(tl.int64),
                mask=mask,
            )

        src_hidden_base = src_row * hidden_size
        dst_hidden_base = row_i64 * hidden_size
        for i in range(hidden_loops):
            cur = hidden_offsets + i * BLOCK_HIDDEN
            mask = cur < hidden_size
            src_hidden = tl.load(
                src_hidden_states_ptr + src_hidden_base + cur,
                mask=mask,
                other=0,
            )
            tl.store(
                dst_hidden_states_ptr + dst_hidden_base + cur,
                src_hidden,
                mask=mask,
            )

        token_pool_ptr = req_to_token_ptr + req_idx * pool_len + seq_len.to(tl.int64)
        dst_out_base = row_i64 * copy_len
        for i in range(copy_loops):
            cur = copy_offsets + i * BLOCK_COPY
            mask = cur < copy_len
            cache_loc = tl.load(token_pool_ptr + cur, mask=mask, other=0)
            tl.store(
                dst_out_cache_loc_ptr + dst_out_base + cur,
                cache_loc.to(tl.int64),
                mask=mask,
            )


def draft_future_replay_prepare_npu_triton(
    future_indices: torch.Tensor,
    src_topk_p: torch.Tensor,
    src_topk_index: torch.Tensor,
    src_hidden_states: torch.Tensor,
    src_verified_id: torch.Tensor,
    src_new_seq_lens: torch.Tensor,
    src_seq_lens: torch.Tensor,
    src_req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    dst_topk_p: torch.Tensor,
    dst_topk_index: torch.Tensor,
    dst_hidden_states: torch.Tensor,
    dst_verified_id: torch.Tensor,
    dst_new_seq_lens: torch.Tensor,
    dst_positions: torch.Tensor,
    dst_out_cache_loc: torch.Tensor,
    topk: int,
    speculative_num_steps: int,
    num_cores: int | None = None,
) -> None:
    raw_bs = int(src_seq_lens.shape[0])
    if raw_bs <= 0:
        return
    if num_cores is None:
        num_cores = NUM_VECTOR_CORES
    num_cores = int(max(1, min(num_cores, raw_bs)))
    hidden_size = int(dst_hidden_states.shape[1])
    copy_len = int(topk * speculative_num_steps)

    _draft_future_replay_prepare_kernel[(num_cores,)](
        future_indices,
        src_topk_p,
        src_topk_index,
        src_hidden_states,
        src_verified_id,
        src_new_seq_lens,
        src_seq_lens,
        src_req_pool_indices,
        req_to_token,
        dst_topk_p,
        dst_topk_index,
        dst_hidden_states,
        dst_verified_id,
        dst_new_seq_lens,
        dst_positions,
        dst_out_cache_loc,
        raw_bs,
        topk,
        hidden_size,
        copy_len,
        req_to_token.shape[1],
        BLOCK_TOPK=16,
        BLOCK_HIDDEN=256,
        BLOCK_COPY=256,
    )

# def _is_fast_path_candidate(
#     tensor_a: torch.Tensor,
#     tensor_b: torch.Tensor,
#     tensor_c: torch.Tensor,
# ) -> bool:
#     return (
#         tensor_a.is_contiguous()
#         and tensor_b.is_contiguous()
#         and tensor_c.is_contiguous()
#         and tensor_a.shape[0] % 16 == 0
#         and tensor_a.shape[-1] % 128 == 0
#         and tensor_b.shape[-1] % 128 == 0
#     )


# if triton is not None:

#     _STRICT32_SMALL_HEAD_CONFIGS = [
#         triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
#     ]
#     _STRICT32_CONFIGS = [
#         triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
#         triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}),
#     ]
#     _STRICT16_CONFIGS = [
#         triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
#         triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}),
#     ]
#     _GENERIC_CONFIGS = [
#         triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
#         triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
#         triton.Config({"BLOCK_B": 8, "BLOCK_N": 256, "BLOCK_K": 64}),
#         triton.Config({"BLOCK_B": 4, "BLOCK_N": 128, "BLOCK_K": 64}),
#     ]

#     @triton.autotune(
#         configs=_STRICT32_SMALL_HEAD_CONFIGS, key=["batch_size", "num_heads", "k", "n"]
#     )
#     @triton.jit
#     def _batch_matmul_transpose_strict32_small_head_kernel(
#         a_ptr,
#         b_ptr,
#         c_ptr,
#         batch_size,
#         num_heads,
#         k,
#         n,
#         stride_ab,
#         stride_ah,
#         stride_ak,
#         stride_bh,
#         stride_bk,
#         stride_bn,
#         stride_cb,
#         stride_ch,
#         stride_cn,
#         BLOCK_N: tl.constexpr,
#         BLOCK_K: tl.constexpr,
#     ):
#         pid = tl.program_id(0)
#         num_core = tl.num_programs(0)
#         BLOCK_B: tl.constexpr = 32

#         num_n_blocks = tl.cdiv(n, BLOCK_N)
#         total_tasks = num_heads * num_n_blocks

#         offs_b = tl.arange(0, BLOCK_B)
#         offs_n = tl.arange(0, BLOCK_N)
#         offs_k = tl.arange(0, BLOCK_K)

#         for task_id in range(pid, total_tasks, num_core):
#             head_id = task_id // num_n_blocks
#             n_block_id = task_id % num_n_blocks
#             n_start = n_block_id * BLOCK_N

#             head_offset = head_id.to(tl.int64)
#             n_offsets = (n_start + offs_n).to(tl.int64)

#             for b_start in range(0, batch_size, BLOCK_B):
#                 b_offsets = (b_start + offs_b).to(tl.int64)
#                 acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

#                 for k_start in range(0, k, BLOCK_K):
#                     k_offsets = (k_start + offs_k).to(tl.int64)
#                     a_ptrs = (
#                         a_ptr
#                         + b_offsets[:, None] * stride_ab
#                         + head_offset * stride_ah
#                         + k_offsets[None, :] * stride_ak
#                     )
#                     b_ptrs = (
#                         b_ptr
#                         + head_offset * stride_bh
#                         + k_offsets[:, None] * stride_bk
#                         + n_offsets[None, :] * stride_bn
#                     )

#                     a = tl.load(a_ptrs)
#                     b = tl.load(b_ptrs)
#                     acc += tl.dot(a, b)

#                 c_ptrs = (
#                     c_ptr
#                     + b_offsets[:, None] * stride_cb
#                     + head_offset * stride_ch
#                     + n_offsets[None, :] * stride_cn
#                 )
#                 tl.store(
#                     c_ptrs,
#                     acc.to(c_ptr.dtype.element_ty),
#                 )

#     @triton.autotune(configs=_STRICT32_CONFIGS, key=["batch_size", "num_heads", "k", "n"])
#     @triton.jit
#     def _batch_matmul_transpose_strict32_kernel(
#         a_ptr,
#         b_ptr,
#         c_ptr,
#         batch_size,
#         num_heads,
#         k,
#         n,
#         stride_ab,
#         stride_ah,
#         stride_ak,
#         stride_bh,
#         stride_bk,
#         stride_bn,
#         stride_cb,
#         stride_ch,
#         stride_cn,
#         BLOCK_N: tl.constexpr,
#         BLOCK_K: tl.constexpr,
#     ):
#         pid = tl.program_id(0)
#         num_core = tl.num_programs(0)
#         BLOCK_B: tl.constexpr = 32

#         num_n_blocks = tl.cdiv(n, BLOCK_N)
#         total_tasks = num_heads * num_n_blocks

#         offs_b = tl.arange(0, BLOCK_B)
#         offs_n = tl.arange(0, BLOCK_N)
#         offs_k = tl.arange(0, BLOCK_K)

#         for task_id in range(pid, total_tasks, num_core):
#             head_id = task_id // num_n_blocks
#             n_block_id = task_id % num_n_blocks
#             n_start = n_block_id * BLOCK_N

#             head_offset = head_id.to(tl.int64)
#             n_offsets = (n_start + offs_n).to(tl.int64)

#             for b_start in range(0, batch_size, BLOCK_B):
#                 b_offsets = (b_start + offs_b).to(tl.int64)
#                 acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

#                 for k_start in range(0, k, BLOCK_K):
#                     k_offsets = (k_start + offs_k).to(tl.int64)
#                     a_ptrs = (
#                         a_ptr
#                         + b_offsets[:, None] * stride_ab
#                         + head_offset * stride_ah
#                         + k_offsets[None, :] * stride_ak
#                     )
#                     b_ptrs = (
#                         b_ptr
#                         + head_offset * stride_bh
#                         + k_offsets[:, None] * stride_bk
#                         + n_offsets[None, :] * stride_bn
#                     )

#                     a = tl.load(a_ptrs)
#                     b = tl.load(b_ptrs)
#                     acc += tl.dot(a, b)

#                 c_ptrs = (
#                     c_ptr
#                     + b_offsets[:, None] * stride_cb
#                     + head_offset * stride_ch
#                     + n_offsets[None, :] * stride_cn
#                 )
#                 tl.store(
#                     c_ptrs,
#                     acc.to(c_ptr.dtype.element_ty),
#                 )

#     @triton.autotune(configs=_STRICT16_CONFIGS, key=["batch_size", "num_heads", "k", "n"])
#     @triton.jit
#     def _batch_matmul_transpose_strict16_kernel(
#         a_ptr,
#         b_ptr,
#         c_ptr,
#         batch_size,
#         num_heads,
#         k,
#         n,
#         stride_ab,
#         stride_ah,
#         stride_ak,
#         stride_bh,
#         stride_bk,
#         stride_bn,
#         stride_cb,
#         stride_ch,
#         stride_cn,
#         BLOCK_N: tl.constexpr,
#         BLOCK_K: tl.constexpr,
#     ):
#         pid = tl.program_id(0)
#         num_core = tl.num_programs(0)
#         BLOCK_B: tl.constexpr = 16

#         num_n_blocks = tl.cdiv(n, BLOCK_N)
#         total_tasks = num_heads * num_n_blocks

#         offs_b = tl.arange(0, BLOCK_B)
#         offs_n = tl.arange(0, BLOCK_N)
#         offs_k = tl.arange(0, BLOCK_K)

#         for task_id in range(pid, total_tasks, num_core):
#             head_id = task_id // num_n_blocks
#             n_block_id = task_id % num_n_blocks
#             n_start = n_block_id * BLOCK_N

#             head_offset = head_id.to(tl.int64)
#             n_offsets = (n_start + offs_n).to(tl.int64)

#             for b_start in range(0, batch_size, BLOCK_B):
#                 b_offsets = (b_start + offs_b).to(tl.int64)
#                 acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

#                 for k_start in range(0, k, BLOCK_K):
#                     k_offsets = (k_start + offs_k).to(tl.int64)
#                     a_ptrs = (
#                         a_ptr
#                         + b_offsets[:, None] * stride_ab
#                         + head_offset * stride_ah
#                         + k_offsets[None, :] * stride_ak
#                     )
#                     b_ptrs = (
#                         b_ptr
#                         + head_offset * stride_bh
#                         + k_offsets[:, None] * stride_bk
#                         + n_offsets[None, :] * stride_bn
#                     )

#                     a = tl.load(a_ptrs)
#                     b = tl.load(b_ptrs)
#                     acc += tl.dot(a, b)

#                 c_ptrs = (
#                     c_ptr
#                     + b_offsets[:, None] * stride_cb
#                     + head_offset * stride_ch
#                     + n_offsets[None, :] * stride_cn
#                 )
#                 tl.store(
#                     c_ptrs,
#                     acc.to(c_ptr.dtype.element_ty),
#                 )

#     @triton.autotune(configs=_GENERIC_CONFIGS, key=["batch_size", "num_heads", "k", "n"])
#     @triton.jit
#     def _batch_matmul_transpose_generic_kernel(
#         a_ptr,
#         b_ptr,
#         c_ptr,
#         batch_size,
#         num_heads,
#         k,
#         n,
#         stride_ab,
#         stride_ah,
#         stride_ak,
#         stride_bh,
#         stride_bk,
#         stride_bn,
#         stride_cb,
#         stride_ch,
#         stride_cn,
#         BLOCK_B: tl.constexpr,
#         BLOCK_N: tl.constexpr,
#         BLOCK_K: tl.constexpr,
#     ):
#         pid = tl.program_id(0)
#         num_core = tl.num_programs(0)

#         num_n_blocks = tl.cdiv(n, BLOCK_N)
#         total_tasks = num_heads * num_n_blocks

#         offs_b = tl.arange(0, BLOCK_B)
#         offs_n = tl.arange(0, BLOCK_N)
#         offs_k = tl.arange(0, BLOCK_K)

#         for task_id in range(pid, total_tasks, num_core):
#             head_id = task_id // num_n_blocks
#             n_block_id = task_id % num_n_blocks
#             n_start = n_block_id * BLOCK_N

#             head_offset = head_id.to(tl.int64)
#             n_offsets = (n_start + offs_n).to(tl.int64)
#             n_mask = n_offsets < n

#             for b_start in range(0, batch_size, BLOCK_B):
#                 b_offsets = (b_start + offs_b).to(tl.int64)
#                 b_mask = b_offsets < batch_size
#                 acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

#                 for k_start in range(0, k, BLOCK_K):
#                     k_offsets = (k_start + offs_k).to(tl.int64)
#                     k_mask = k_offsets < k

#                     a_ptrs = (
#                         a_ptr
#                         + b_offsets[:, None] * stride_ab
#                         + head_offset * stride_ah
#                         + k_offsets[None, :] * stride_ak
#                     )
#                     b_ptrs = (
#                         b_ptr
#                         + head_offset * stride_bh
#                         + k_offsets[:, None] * stride_bk
#                         + n_offsets[None, :] * stride_bn
#                     )

#                     a = tl.load(a_ptrs, mask=b_mask[:, None] & k_mask[None, :], other=0.0)
#                     b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
#                     acc += tl.dot(a, b)

#                 c_ptrs = (
#                     c_ptr
#                     + b_offsets[:, None] * stride_cb
#                     + head_offset * stride_ch
#                     + n_offsets[None, :] * stride_cn
#                 )
#                 tl.store(
#                     c_ptrs,
#                     acc.to(c_ptr.dtype.element_ty),
#                     mask=b_mask[:, None] & n_mask[None, :],
#                 )


# def batch_matmul_transpose(
#     tensor_a: torch.Tensor,
#     tensor_b: torch.Tensor,
#     tensor_c: torch.Tensor,
# ) -> torch.Tensor:
#     batch_size, num_heads, k = tensor_a.shape
#     n = tensor_b.shape[2]

#     grid = (NUM_CUBE_CORES,)

#     if _is_fast_path_candidate(tensor_a, tensor_b, tensor_c):
#         if batch_size % 32 == 0:
#             kernel = (
#                 _batch_matmul_transpose_strict32_small_head_kernel
#                 if num_heads <= 8
#                 else _batch_matmul_transpose_strict32_kernel
#             )
#         elif batch_size % 16 == 0:
#             kernel = _batch_matmul_transpose_strict16_kernel
#         else:
#             kernel = _batch_matmul_transpose_generic_kernel
#     else:
#         kernel = _batch_matmul_transpose_generic_kernel

#     kernel[grid](
#         tensor_a,
#         tensor_b,
#         tensor_c,
#         batch_size,
#         num_heads,
#         k,
#         n,
#         tensor_a.stride(0),
#         tensor_a.stride(1),
#         tensor_a.stride(2),
#         tensor_b.stride(0),
#         tensor_b.stride(1),
#         tensor_b.stride(2),
#         tensor_c.stride(0),
#         tensor_c.stride(1),
#         tensor_c.stride(2),
#     )
#     return tensor_c

_GENERIC_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 64}),
]

_FAST_CONFIGS = [
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 128}),
]

_B1_FAST_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
]


@triton.autotune(configs=_GENERIC_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_generic_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # One physical core handles multiple tiles in a cyclic manner.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = tl.cdiv(B, BLOCK_B)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_m_groups = tl.cdiv(M, GROUP_M)
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            if m_idx < M:
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)

                    a_ptrs = (
                        a_ptr
                        + offs_b[:, None] * stride_ab
                        + m_idx * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    w_ptrs = (
                        w_ptr
                        + m_idx * stride_wm
                        + offs_k[:, None] * stride_wk
                        + offs_n[None, :] * stride_wn
                    )

                    mask_a = (offs_b[:, None] < B) & (offs_k[None, :] < K)
                    mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                    w = tl.load(w_ptrs, mask=mask_w, other=0.0)

                    acc += tl.dot(a, w)

                c_ptrs = (
                    c_ptr
                    + offs_b[:, None] * stride_cb
                    + m_idx * stride_cm
                    + offs_n[None, :] * stride_cn
                )
                mask_c = (offs_b[:, None] < B) & (offs_n[None, :] < N)
                out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
                tl.store(c_ptrs, out, mask=mask_c)


@triton.autotune(configs=_FAST_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Fast kernel assumes no tail blocks in B/K/N and M divisible by GROUP_M.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = B // BLOCK_B
    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)

                a_ptrs = (
                    a_ptr
                    + offs_b[:, None] * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )

                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + offs_b[:, None] * stride_cb
                + m_idx * stride_cm
                + offs_n[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


@triton.autotune(configs=_B1_FAST_CONFIGS, key=["M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_b1_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Specialized no-mask kernel for B==1 to avoid masked lanes in tl.dot.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_n_tiles

    offs_n = tl.arange(0, BLOCK_N)

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // num_n_tiles
        n_tile_idx = block_idx % num_n_tiles
        n_base = n_tile_idx * BLOCK_N
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a_ptrs = (
                    a_ptr
                    + 0 * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + (n_base + offs_n)[None, :] * stride_wn
                )
                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + 0 * stride_cb
                + m_idx * stride_cm
                + (n_base + offs_n)[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


def _to_nd_from_nz(tensor_b: torch.Tensor) -> torch.Tensor:
    # NZ input: [M, N/16, K, 16] -> ND [M, K, N]
    if tensor_b.ndim != 4:
        raise ValueError("NZ format expects tensor_b with 4 dimensions [M, N/16, K, 16].")
    if tensor_b.shape[-1] != 16:
        raise ValueError("NZ format expects tensor_b.shape[-1] == 16.")
    m, n16, k, inner = tensor_b.shape
    n = n16 * inner
    return tensor_b.permute(0, 2, 1, 3).contiguous().view(m, k, n)


def batch_matmul_transpose(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    format_mode: Optional[str] = None,
    quant_mode: Optional[str] = None,
) -> None:
    """
    Triton NPU implementation with the same public signature as:
      torch.ops.npu.batch_matmul_transpose(tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None)

    Contract:
      tensor_a: [B, M, K]
      tensor_b (ND): [M, K, N]
      tensor_b (NZ): [M, N/16, K, 16]
      tensor_c: [B, M, N] (written in-place)
    """
    del quant_mode  # Kept for signature compatibility.

    mode = "ND" if format_mode is None else str(format_mode).upper()
    # Permanent fast path for decode hot loop:
    # - assume caller provides valid tensors
    # - avoid Python-side validation overhead
    if mode != "ND" and mode != "NZ":
        mode = "ND"

    if mode == "NZ":
        tensor_b_nd = _to_nd_from_nz(tensor_b)
    else:
        tensor_b_nd = tensor_b

    b, m, k = tensor_a.shape
    n = tensor_b_nd.shape[2]

    # Keep all pointers contiguous for predictable memory access on NPU.
    a = tensor_a.contiguous()
    w = tensor_b_nd.contiguous()

    # Small-M strategy:
    # - M < 16: keep GROUP_M=1 to avoid reducing block count too aggressively.
    # - otherwise allow GROUP_M=2 when divisible to improve data reuse.
    group_m = 1 if m < 16 else (2 if (m % 2 == 0) else 1)

    # Fast path (mask-free): all fast configs currently assume BLOCK_B=8, BLOCK_N=128.
    # Keep divisibility guard strict to ensure no tail blocks.
    use_fast = (b % 8 == 0) and (n % 128 == 0) and (k % 128 == 0) and (m % group_m == 0)
    use_b1_fast = (b == 1) and (n % 128 == 0) and (k % 64 == 0) and (m % group_m == 0)

    grid = (NUM_CUBE_CORES,)
    if use_b1_fast:
        _batch_matmul_transpose_b1_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    elif use_fast:
        _batch_matmul_transpose_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    else:
        _batch_matmul_transpose_generic_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )


@triton.jit(do_not_specialize=["num_tokens", "page_size", "roundup"])
def _num_new_pages_sum_kernel(
    seq_lens_ptr,
    prefix_lens_ptr,
    out_sum_ptr,
    num_tokens,
    page_size,
    roundup,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_sum = 0
    num_full_blocks = num_tokens // BLOCK_SIZE

    # Keep the hot path fully unmasked and assign blocks round-robin to vector cores.
    for block_idx in tl.range(pid, num_full_blocks, num_programs):
        block_start = block_idx * BLOCK_SIZE
        block_offsets = block_start + offsets
        seq_lens = tl.load(seq_lens_ptr + block_offsets)
        prefix_lens = tl.load(prefix_lens_ptr + block_offsets)
        new_pages = ((seq_lens + roundup) // page_size) - (
            (prefix_lens + roundup) // page_size
        )
        partial_sum += tl.sum(new_pages, axis=0)

    tail_size = num_tokens - num_full_blocks * BLOCK_SIZE
    if pid == num_full_blocks % num_programs and tail_size > 0:
        block_start = num_full_blocks * BLOCK_SIZE
        block_offsets = block_start + offsets
        tail_mask = offsets < tail_size
        seq_lens = tl.load(seq_lens_ptr + block_offsets, mask=tail_mask, other=0)
        prefix_lens = tl.load(
            prefix_lens_ptr + block_offsets, mask=tail_mask, other=0
        )
        new_pages = ((seq_lens + roundup) // page_size) - (
            (prefix_lens + roundup) // page_size
        )
        partial_sum += tl.sum(new_pages, axis=0)

    tl.atomic_add(out_sum_ptr, partial_sum)


def get_num_new_pages_triton(
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_size: int,
    out_sum: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    num_tokens = seq_lens.shape[0]
    if num_tokens == 0:
        if out_sum is None:
            return torch.zeros((1,), dtype=torch.int32, device=seq_lens.device)
        out_sum.zero_()
        return out_sum

    seq_lens = seq_lens.contiguous()
    prefix_lens = prefix_lens.contiguous()

    if out_sum is None:
        out_sum = torch.zeros((1,), dtype=torch.int32, device=seq_lens.device)
    else:
        out_sum.zero_()

    _num_new_pages_sum_kernel[(NUM_VECTOR_CORES,)](
        seq_lens_ptr=seq_lens,
        prefix_lens_ptr=prefix_lens,
        out_sum_ptr=out_sum,
        num_tokens=num_tokens,
        page_size=page_size,
        roundup=page_size - 1,
        BLOCK_SIZE=4096,
    )
    return out_sum