import triton
import triton.language as tl


NPU_ALLOC_EXTEND_MAX_BATCH_SIZE = 128
NPU_ALLOC_EXTEND_BLOCK_SIZE = 2048


# free_page_ptr aliases self.free_pages, which the paged allocator re-slices
# after every allocation (self.free_pages = self.free_pages[num_new_pages:]).
# Slicing only advances data_ptr() by num_new_pages * 8 bytes, so the pointer
# flips between 16-byte-aligned and unaligned across calls. Triton specializes
# on pointer alignment by default and bakes it into the cache key, compiling two
# kernel variants (one with tt.divisibility=16 on free_page_ptr, one without)
# so the second prefill on a fresh DCP server hits the alternate alignment and
# pays an extra ~100ms JIT for that kernel variant. do_not_specialize skips
# that specialization so only one kernel is ever compiled; the perf cost is
# negligible (this kernel runs in ~10us and only loads ~4KB through this ptr).
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
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
    last_loc = tl.load(last_loc_ptr + pid)
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

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling -- no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
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


@triton.jit(
    do_not_specialize=[
        "pre_lens_ptr",
        "seq_lens_ptr",
        "last_loc_ptr",
        "free_page_ptr",
        "out_indices_ptr",
    ]
)
def alloc_extend_kernel_npu(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices_ptr,
    MAX_BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Allocate arbitrary-length NPU extend indices with one JIT variant.

    Grid axis 0 is the request id and axis 1 is a fixed-size token chunk.
    Request lengths therefore only change the launch grid and runtime masks;
    they never enter the JIT signature.  ``MAX_BATCH_SIZE`` is a fixed envelope
    used for the small prefix reductions, not the live request batch size.
    """
    request_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    # Prefix sums locate this request in the concatenated output and in the
    # free-page slice.  All masked lanes explicitly contribute zero.
    batch_offsets = tl.arange(0, MAX_BATCH_SIZE)
    preceding_request_mask = batch_offsets < request_id
    preceding_seq_lens = tl.load(
        seq_lens_ptr + batch_offsets,
        mask=preceding_request_mask,
        other=0,
    )
    preceding_pre_lens = tl.load(
        pre_lens_ptr + batch_offsets,
        mask=preceding_request_mask,
        other=0,
    )
    output_start = tl.sum(preceding_seq_lens - preceding_pre_lens)

    preceding_pages_after = (
        preceding_seq_lens + PAGE_SIZE - 1
    ) // PAGE_SIZE
    preceding_pages_before = (
        preceding_pre_lens + PAGE_SIZE - 1
    ) // PAGE_SIZE
    new_page_start = tl.sum(preceding_pages_after - preceding_pages_before)

    seq_len = tl.load(seq_lens_ptr + request_id)
    pre_len = tl.load(pre_lens_ptr + request_id)
    pages_after = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pages_before = (pre_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_new_pages = pages_after - pages_before

    # Part 1: finish the request's existing partial page.  Only chunk zero
    # performs the small boundary writes; all token chunks remain independent.
    next_page_boundary = pages_before * PAGE_SIZE
    part1_end = tl.minimum(seq_len, next_page_boundary)
    num_part1 = tl.maximum(part1_end - pre_len, 0)
    page_offsets = tl.arange(0, PAGE_SIZE)
    is_first_chunk = chunk_id == 0
    part1_mask = page_offsets < num_part1
    part1_mask = part1_mask & is_first_chunk
    last_loc = tl.load(
        last_loc_ptr + request_id,
        mask=is_first_chunk,
        other=0,
    )
    tl.store(
        out_indices_ptr + output_start + page_offsets,
        last_loc + 1 + page_offsets,
        mask=part1_mask,
    )

    # Part 2: each program writes one independent 2K slice of the new full
    # pages.  Increasing the request length only increases grid axis 1.
    full_page_start = pages_before * PAGE_SIZE
    full_page_end = (seq_len // PAGE_SIZE) * PAGE_SIZE
    num_part2 = tl.maximum(full_page_end - full_page_start, 0)
    block_offsets = tl.arange(0, BLOCK_SIZE)
    part2_offsets = chunk_id * BLOCK_SIZE + block_offsets
    part2_mask = part2_offsets < num_part2
    page_ids = tl.load(
        free_page_ptr
        + new_page_start
        + part2_offsets // PAGE_SIZE,
        mask=part2_mask,
        other=0,
    )
    tl.store(
        out_indices_ptr + output_start + num_part1 + part2_offsets,
        page_ids * PAGE_SIZE + part2_offsets % PAGE_SIZE,
        mask=part2_mask,
    )

    # Part 3: write the final new partial page, again only from chunk zero.
    num_part3 = seq_len - full_page_end
    has_remaining_after_part1 = pre_len + num_part1 < seq_len
    has_part3 = has_remaining_after_part1 & (num_part3 > 0)
    has_part3 = has_part3 & is_first_chunk
    last_new_page_offset = tl.maximum(num_new_pages - 1, 0)
    last_new_page = tl.load(
        free_page_ptr + new_page_start + last_new_page_offset,
        mask=has_part3,
        other=0,
    )
    part3_mask = page_offsets < num_part3
    part3_mask = part3_mask & has_part3
    tl.store(
        out_indices_ptr
        + output_start
        + num_part1
        + num_part2
        + page_offsets,
        last_new_page * PAGE_SIZE + page_offsets,
        mask=part3_mask,
    )


# Same free_page_ptr alignment rationale as alloc_extend_kernel above.
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)
