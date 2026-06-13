from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages, next_power_of_2, is_npu_before_atlas_a5
_is_npu_before_atlas_a5 = is_npu_before_atlas_a5()
import triton.backends.ascend.runtime
if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.roundup = page_size - 1
        self._num_new_pages_out_sum = None

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        num_new_pages = (
            (seq_lens + self.roundup) // self.page_size
            - (prefix_lens + self.roundup) // self.page_size
        ).sum()
        num_new_pages_item = num_new_pages.item()
        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        if num_new_pages_item < 200:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            max_num_extend_tokens = next_power_of_2(extend_num_tokens)
            bs = prefix_lens.shape[0]
            if _is_npu_before_atlas_a5:
                from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
                alloc_extend_kernel[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.free_pages,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                    max_num_extend_tokens,
                )
            else:
                from sglang.srt.hardware_backend.npu.triton import alloc_extend_kernel_triton
                alloc_extend_kernel_triton[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.free_pages,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                    max_num_extend_tokens,
                )
        else:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
            alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.page_size,
                self.device,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_extend_and_assign(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        extend_num_tokens: int,
    ) -> bool:
        if _is_npu_before_atlas_a5:
            return False

        from sglang.srt.hardware_backend.npu.triton import (
            FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS,
            alloc_extend_assign_req_to_token_pool_triton,
            get_num_new_pages_triton,
        )

        batch_size = int(prefix_lens.shape[0])
        if batch_size == 0:
            return True

        num_new_pages = (
            (seq_lens + self.roundup) // self.page_size
            - (prefix_lens + self.roundup) // self.page_size
        ).sum()
        num_new_pages_item = num_new_pages.item()

        # if self._num_new_pages_out_sum is None:
        #     self._num_new_pages_out_sum = torch.empty(
        #         (1,), dtype=torch.int32, device=self.device
        #     )
        # num_new_pages = get_num_new_pages_triton(
        #     seq_lens=seq_lens,
        #     prefix_lens=prefix_lens,
        #     page_size=self.page_size,
        #     out_sum=self._num_new_pages_out_sum,
        # )
        # num_new_pages_item = int(num_new_pages.cpu().item())

        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return False

        if num_new_pages_item >= 200:
            return False

        if batch_size > FUSED_ALLOC_EXTEND_ASSIGN_MAX_BS:
            return False

        alloc_extend_assign_req_to_token_pool_triton(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            free_pages=self.free_pages,
            page_size=self.page_size,
        )
        self.free_pages = self.free_pages[num_new_pages_item:]
        return True

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages > len(self.free_pages):
            return None

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            device = free_index.device
            free_page_indices = torch.unique(free_index.cpu() // self.page_size)
            free_page_indices = free_page_indices.to(device)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)
