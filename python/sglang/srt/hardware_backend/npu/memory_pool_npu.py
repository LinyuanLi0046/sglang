from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _init_npu_conv_state(
    conv_state_in, conv_state_shape, speculative_num_draft_tokens: Optional[int] = None
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # conv_state shape (layers, pool_size, conv_wind + draft_step, dim) for conv1d ascendc ops require dim as last dim
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                conv_shape[1] + extra_conv_len,
                conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self.v_buffer[i].view(-1, 1, self.head_num, self.v_head_dim)
                    for i in range(self.layer_num)
                ]

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]

            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer,
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer,
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.head_num, self.v_head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMLATokenToKVPool(MLATokenToKVPool):

    dsa_kv_quant_tile_size = 128

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        kv_cache_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        indexer_layer_ids: Optional[Sequence[int]] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        if self.index_head_dim is None:
            if indexer_layer_ids is not None and len(indexer_layer_ids) > 0:
                raise ValueError(
                    "indexer_layer_ids must be empty when index_head_dim is None"
                )
            resolved_indexer_layer_ids = ()
        elif indexer_layer_ids is None:
            # Backward compatibility: callers that do not describe the real
            # Indexer layers retain the legacy one-Indexer-cache-per-layer
            # layout.
            resolved_indexer_layer_ids = tuple(
                range(self.start_layer, self.start_layer + self.layer_num)
            )
        else:
            resolved_indexer_layer_ids = tuple(indexer_layer_ids)
            if len(set(resolved_indexer_layer_ids)) != len(
                resolved_indexer_layer_ids
            ):
                raise ValueError(
                    "indexer_layer_ids must not contain duplicate layer ids: "
                    f"{resolved_indexer_layer_ids}"
                )
            if resolved_indexer_layer_ids != tuple(
                sorted(resolved_indexer_layer_ids)
            ):
                raise ValueError(
                    "indexer_layer_ids must be in increasing logical-layer order: "
                    f"{resolved_indexer_layer_ids}"
                )
            layer_end = self.start_layer + self.layer_num
            invalid_layer_ids = [
                layer_id
                for layer_id in resolved_indexer_layer_ids
                if layer_id < self.start_layer or layer_id >= layer_end
            ]
            if invalid_layer_ids:
                raise ValueError(
                    "indexer_layer_ids must be absolute layer ids in the local "
                    f"stage range [{self.start_layer}, {layer_end}), got "
                    f"{invalid_layer_ids}"
                )

        self.indexer_layer_ids = resolved_indexer_layer_ids
        self.num_indexer_layers = len(self.indexer_layer_ids)
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        self.has_index_k_scale = (
            self.index_head_dim is not None and dtype == torch.float8_e4m3fn
        )
        self.dsa_kv_cache_store_fp8 = (
            index_head_dim is not None
            and dtype == torch.float8_e4m3fn
            and kv_cache_dim is not None
            and kv_cache_dim != kv_lora_rank + qk_rope_head_dim
        )
        self.kv_cache_dim = (
            kv_cache_dim if self.dsa_kv_cache_store_fp8 else kv_lora_rank
        )
        self.kr_cache_dim = (
            0 if self.dsa_kv_cache_store_fp8 else qk_rope_head_dim
        )
        self.k_store_dtype = self.store_dtype
        self.v_store_dtype = self.store_dtype
        if self.dsa_kv_cache_store_fp8:
            self.k_store_dtype = torch.float8_e4m3fn
            self.v_store_dtype = torch.bfloat16

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_cache_dim,
                ),
                dtype=self.k_store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kr_cache_dim,
                ),
                dtype=self.v_store_dtype,
                device=self.device,
            )
            self.index_k_buffer = None
            self.index_k_scale_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        self.num_indexer_layers,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.k_store_dtype,
                    device=self.device,
                )
                if self.has_index_k_scale:
                    self.index_k_scale_buffer = torch.zeros(
                        (
                            self.num_indexer_layers,
                            self.size // self.page_size + 1,
                            self.page_size,
                            1,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )

        self._finalize_allocation_log(size)

    def _get_indexer_slot(self, layer_id: int) -> int:
        if self.index_head_dim is None:
            raise RuntimeError("This KV pool does not have an Indexer cache")
        try:
            return self.indexer_layer_id_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"Layer {layer_id} is not a real Indexer layer in this KV pool; "
                f"configured Indexer layers: {self.indexer_layer_ids}"
            ) from exc

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
            if self.index_k_scale_buffer is not None:
                for index_k_scale_cache in self.index_k_scale_buffer:
                    kv_size_bytes += get_tensor_size_bytes(index_k_scale_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        data_ptrs = [
            self.index_k_buffer[i].data_ptr()
            for i in range(self.num_indexer_layers)
        ]
        data_lens = [
            self.index_k_buffer[i].nbytes for i in range(self.num_indexer_layers)
        ]
        item_lens = [
            self.index_k_buffer[i][0].nbytes
            for i in range(self.num_indexer_layers)
        ]
        if self.index_k_scale_buffer is not None:
            data_ptrs += [
                self.index_k_scale_buffer[i].data_ptr()
                for i in range(self.num_indexer_layers)
            ]
            data_lens += [
                self.index_k_scale_buffer[i].nbytes
                for i in range(self.num_indexer_layers)
            ]
            item_lens += [
                self.index_k_scale_buffer[i][0].nbytes
                for i in range(self.num_indexer_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.k_store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.v_store_dtype == self.store_dtype and self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.k_store_dtype != self.dtype:
            return self.index_k_buffer[indexer_slot].view(self.dtype)
        return self.index_k_buffer[indexer_slot]

    def get_index_k_scale_buffer(self, layer_id: int):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.index_k_scale_buffer is None:
            raise RuntimeError(
                "Indexer scale cache is only available for an FP8 Indexer cache"
            )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        return self.index_k_scale_buffer[indexer_slot]

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr()
                for i in range(self.num_indexer_layers)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes
                for i in range(self.num_indexer_layers)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes
                for i in range(self.num_indexer_layers)
            ]
            if self.index_k_scale_buffer is not None:
                kv_data_ptrs += [
                    self.index_k_scale_buffer[i].data_ptr()
                    for i in range(self.num_indexer_layers)
                ]
                kv_data_lens += [
                    self.index_k_scale_buffer[i].nbytes
                    for i in range(self.num_indexer_layers)
                ]
                kv_item_lens += [
                    self.index_k_scale_buffer[i][0].nbytes
                    for i in range(self.num_indexer_layers)
                ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _pack_dsa_fp8_kv_cache(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        tile_size = self.dsa_kv_quant_tile_size
        if self.kv_lora_rank % tile_size != 0:
            raise RuntimeError(
                f"kv_lora_rank {self.kv_lora_rank} must be divisible by "
                f"DSA FP8 tile size {tile_size}"
            )

        num_tiles = self.kv_lora_rank // tile_size
        expected_cache_dim = (
            self.kv_lora_rank
            + self.qk_rope_head_dim * torch.bfloat16.itemsize
            + num_tiles * torch.float32.itemsize
        )
        if self.kv_cache_dim != expected_cache_dim:
            raise RuntimeError(
                f"Unexpected DSA FP8 cache dim {self.kv_cache_dim}, "
                f"expected {expected_cache_dim}"
            )
        if cache_k.numel() != num_tokens * self.kv_lora_rank:
            raise RuntimeError(
                f"Unexpected DSA k_nope shape {tuple(cache_k.shape)} for "
                f"{num_tokens} cache locations"
            )
        if cache_v.numel() != num_tokens * self.qk_rope_head_dim:
            raise RuntimeError(
                f"Unexpected DSA k_rope shape {tuple(cache_v.shape)} for "
                f"{num_tokens} cache locations"
            )

        # Quantize each 128-element no-PE tile independently.  The scale
        # returned by npu_dynamic_quant is the FP32 dequant scale consumed by
        # npu_kv_quant_sparse_flash_attention.
        k_nope_tiles = (
            cache_k.to(torch.bfloat16)
            .reshape(num_tokens * num_tiles, tile_size)
            .contiguous()
        )
        k_nope_fp8, k_nope_scale = torch_npu.npu_dynamic_quant(
            k_nope_tiles,
            dst_type=torch.float8_e4m3fn,
        )
        k_nope_fp8 = k_nope_fp8.reshape(
            num_tokens, 1, self.kv_lora_rank
        ).contiguous()
        k_rope_bf16 = (
            cache_v.to(torch.bfloat16)
            .reshape(num_tokens, 1, self.qk_rope_head_dim)
            .contiguous()
        )
        k_nope_scale = (
            k_nope_scale.to(torch.float32)
            .reshape(num_tokens, 1, num_tiles)
            .contiguous()
        )

        # Packed byte layout expected by the NPU quant sparse attention op:
        # [nope_fp8(512) | rope_bf16_bytes(128) | fp32_scales(16)].
        packed = torch.empty(
            (num_tokens, 1, self.kv_cache_dim),
            dtype=torch.float8_e4m3fn,
            device=cache_k.device,
        )
        packed_bytes = packed.view(torch.uint8)
        rope_begin = self.kv_lora_rank
        rope_end = rope_begin + self.qk_rope_head_dim * torch.bfloat16.itemsize
        packed_bytes[..., :rope_begin].copy_(k_nope_fp8.view(torch.uint8))
        packed_bytes[..., rope_begin:rope_end].copy_(
            k_rope_bf16.view(torch.uint8)
        )
        packed_bytes[..., rope_end:].copy_(k_nope_scale.view(torch.uint8))
        return packed

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if self.dsa_kv_cache_store_fp8:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            packed_cache = self._pack_dsa_fp8_kv_cache(
                cache_k,
                cache_v,
                loc.numel(),
            )
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.kv_cache_dim
                ),
                loc.view(-1, 1),
                packed_cache,
            )
            return

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        indexer_slot = self._get_indexer_slot(layer_id)
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.k_store_dtype != self.dtype:
            index_k = index_k.view(self.k_store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[indexer_slot].view(-1, 1, self.index_head_dim),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        indexer_slot = self._get_indexer_slot(layer_id)
        if self.index_k_scale_buffer is None:
            raise RuntimeError(
                "Indexer scale cache is only available for an FP8 Indexer cache"
            )
        torch_npu.npu_scatter_nd_update_(
            self.index_k_scale_buffer[indexer_slot].view(-1, 1),
            loc.view(-1, 1),
            index_k_scale.view(-1, 1),
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k, v, optional index_k/scale]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.npu.synchronize()
        buf_of_layers = []
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            num_slots = (
                self.k_buffer[local_layer_id].shape[0]
                * self.k_buffer[local_layer_id].shape[1]
            )
            k_layer = self.k_buffer[local_layer_id].reshape(
                num_slots, 1, self.kv_cache_dim
            )
            v_layer = self.v_buffer[local_layer_id].reshape(
                num_slots, 1, self.kr_cache_dim
            )
            tensors_per_layer = [k_layer, v_layer]
            if has_ik:
                layer_id = self.start_layer + local_layer_id
                indexer_slot = self.indexer_layer_id_to_slot.get(layer_id)
                if indexer_slot is not None:
                    tensors_per_layer.append(
                        self.index_k_buffer[indexer_slot].reshape(
                            num_slots, 1, self.index_head_dim
                        )
                    )
                    if self.index_k_scale_buffer is not None:
                        tensors_per_layer.append(
                            self.index_k_scale_buffer[indexer_slot].reshape(
                                num_slots, 1
                            )
                        )
            buf_of_layers.append(tensors_per_layer)

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            num_slots = (
                self.k_buffer[local_layer_id].shape[0]
                * self.k_buffer[local_layer_id].shape[1]
            )
            k_layer = self.k_buffer[local_layer_id].reshape(
                num_slots, 1, self.kv_cache_dim
            )
            v_layer = self.v_buffer[local_layer_id].reshape(
                num_slots, 1, self.kr_cache_dim
            )
            indexer_slot = None
            if has_ik:
                layer_id = self.start_layer + local_layer_id
                indexer_slot = self.indexer_layer_id_to_slot.get(layer_id)
            if indexer_slot is not None:
                ik_layer = self.index_k_buffer[indexer_slot].reshape(
                    num_slots, 1, self.index_head_dim
                )
                if self.index_k_scale_buffer is not None:
                    ik_scale_layer = self.index_k_scale_buffer[
                        indexer_slot
                    ].reshape(num_slots, 1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if indexer_slot is not None:
                    ik_cpu = chunk[2]
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
                    if self.index_k_scale_buffer is not None:
                        ik_scale_cpu = chunk[3]
                        ik_scale_layer[chunk_indices] = ik_scale_cpu.to(
                            ik_scale_layer.device, non_blocking=True
                        )
        torch.npu.synchronize()
