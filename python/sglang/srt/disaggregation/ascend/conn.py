import concurrent.futures
import enum
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.observability.pd_time_stats import timed_component
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.network import get_local_ip_auto
from sglang.srt.utils.npu_pd_affinity import get_pd_thread_affinity

logger = logging.getLogger(__name__)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU per-pool PD components, kept out of the cross-hardware
    StateType enum. Sent via the same page-indexed path as SWA."""

    DSV4_SWA = "dsv4_swa"
    DSV4_C4 = "dsv4_c4"
    DSV4_C128 = "dsv4_c128"
    DSV4_INDEXER = "dsv4_indexer"
    DSV4_C4_STATE = "dsv4_c4_state"
    DSV4_C128_STATE = "dsv4_c128_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


class AscendKVManager(MooncakeKVManager):
    def _start_transfer_thread(self, target, *, role, args=(), name=None, daemon=None):
        affinity = get_pd_thread_affinity()
        start = (
            affinity.start_thread
            if affinity is not None
            else super()._start_transfer_thread
        )
        return start(target, role=role, args=args, name=name, daemon=daemon)

    def _get_transfer_pool_initializer(self):
        affinity = get_pd_thread_affinity()
        return affinity.initialize_pool_thread if affinity is not None else None

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        writers = envs.SGLANG_ASCEND_PD_KV_WRITERS.get()
        if writers is not None:
            self._configure_kv_writers(writers)
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def _configure_kv_writers(self, value: str):
        writers = [int(rank) for rank in value.split(",")]
        if sorted(writers) != list(range(4)) or any(
            p // 2 != d % 2 for d, p in enumerate(writers)
        ):
            raise ValueError(
                f"SGLANG_ASCEND_PD_KV_WRITERS={value!r} must assign each P rank "
                "once to a D rank with the same KV head"
            )
        prefill = self.disaggregation_mode == DisaggregationMode.PREFILL
        args = self.server_args
        model = args.get_model_config()
        kv = self.kv_args
        if (
            args.tp_size != 4
            or args.dp_size != self.attn_dp_size
            or (self.attn_dp_size, self.attn_tp_size) != ((1, 4) if prefill else (2, 2))
            or (self.pp_size, self.attn_cp_size, self.dcp_size) != (1, 1, 1)
            or AscendTransferEngine._get_transfer_protocol() != "host_rdma"
            or envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            or model.hf_config.architectures[0] != "WeLMV4MoeForCausalLM"
            or model.get_total_num_kv_heads() != 2
            or model.get_swa_num_kv_heads(1) != 2
            or kv.state_types not in ([], [StateType.SWA])
        ):
            raise ValueError(
                "Ascend KV writers require two WeLM KV heads, P TP4, D attention "
                "DP2 x TP2, PP/CP/DCP=1 and direct host_rdma"
            )
        # Each P accepts real writes for exactly one global D rank. The other
        # replica still registers and consumes dummy room metadata as usual.
        d_rank = writers.index(self.attn_tp_rank) if prefill else get_parallel().tp_rank
        self.kv_writer = writers[d_rank]
        layout = [
            writers,
            d_rank,
            get_bool_env_var("ASCEND_USE_FIA", "False"),
            getattr(model, "full_attention_layer_ids", []),
            getattr(model, "swa_attention_layer_ids", []),
            getattr(model.hf_config, "sliding_window_size_layerwise", None),
            getattr(model.hf_config, "kv_mirror_layers", None),
            getattr(model.hf_config, "kv_mirror_imitated_layers", None),
            args.speculative_algorithm,
            kv.kv_item_lens,
            kv.aux_item_lens,
        ]
        self.kv_writer_registration = (json.dumps(layout).encode("ascii"),)

    def _resolve_rank_mapping(self, info):
        super()._resolve_rank_mapping(info)
        if self.kv_writer_registration:
            sizes = (info.attn_tp_size, info.dp_size, info.pp_size, info.attn_cp_size)
            if sizes != (4, 1, 1, 1) or self.kv_writer not in info.target_tp_ranks:
                raise ValueError(f"P{self.kv_writer} is ineligible for {sizes=}")
            info.target_tp_rank = self.kv_writer
            info.required_prefill_response_num = 1

    def _is_replica_writer(self, peer):
        kv = self.kv_args
        return (
            self.kv_writer_registration == (peer.writer_layout,)
            and peer.dst_attn_tp_size == 2
            and peer.dst_tp_rank == self.attn_tp_rank // 2
            and (peer.dst_dcp_size, peer.dst_dcp_rank) == (1, 0)
            and len(peer.dst_kv_ptrs) == len(kv.kv_data_ptrs)
            and len(peer.dst_aux_ptrs) == len(kv.aux_data_ptrs)
            and list(map(len, peer.dst_state_data_ptrs))
            == list(map(len, kv.state_data_ptrs))
            and peer.dst_state_item_lens == kv.state_item_lens
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if ptrs:
            self.engine.batch_register(ptrs, lens)

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        start_layer = self.kv_args.prefill_start_layer
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        total_kv_layers = getattr(self.kv_args, "total_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        # When only speculative-algorithm is enabled for decode
        # the KV has one more layer than prefill.
        # The draft layer needs to be skipped.
        dst_total_layers = (
            min(len(dst_kv_ptrs) // kv_buf_groups, total_kv_layers)
            if total_kv_layers
            else len(dst_kv_ptrs) // kv_buf_groups
        )
        end_layer = start_layer + src_layers
        if src_layers == dst_total_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            for i in range(kv_buf_groups):
                layer_offset = i * dst_total_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    @timed_component("kv")
    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend direct KV transfer does not support separate destination "
                "device KV indices"
            )

        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def append_transfer_addresses(
            src_ptr: int,
            dst_ptr: int,
            item_len: int,
            src_addrs: List[int],
            dst_addrs: List[int],
            lengths: List[int],
        ) -> None:
            # Build the final engine arguments directly. Per-segment tuples and
            # the subsequent _transfer_data zip(*) transpose create many GC
            # containers on fragmented KV transfers, causing host-side pauses.
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addrs.append(src_ptr + int(prefill_index[0]) * item_len)
                dst_addrs.append(dst_ptr + int(decode_index[0]) * item_len)
                lengths.append(item_len * len(prefill_index))

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            # Each concurrent worker owns its lists until its sync call returns.
            src_addrs, dst_addrs, lengths = [], [], []
            append_transfer_addresses(
                src_ptr, dst_ptr, item_len, src_addrs, dst_addrs, lengths
            )
            if not src_addrs:
                return 0
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addrs, dst_addrs, lengths
            )

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            src_addrs, dst_addrs, lengths = [], [], []
            for src_ptr, dst_ptr, item_len in layers_params:
                append_transfer_addresses(
                    src_ptr, dst_ptr, item_len, src_addrs, dst_addrs, lengths
                )
            if not src_addrs:
                return 0
            # Keep the engine wrapper, including transfer timing/error handling.
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addrs, dst_addrs, lengths
            )

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
