import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs_new/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        hcom_url = os.getenv("ASCEND_MF_HCOM_URL", "")

        if transfer_protocol == "host_rdma":
            trans_op_type = TransferEngine.TransDataOpType.HOST_RDMA
            hcom_url = self._get_worker_hcom_url(
                hcom_url, get_world_group().rank_in_group
            )
        elif transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url,
            self.session_id,
            self.role,
            self.npu_id,
            trans_op_type,
            self.role,
            hcom_url,
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "sdma", "host_rdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None

    def _get_worker_hcom_url(self, hcom_url: str, world_rank: int) -> str:
        if not hcom_url:
            return hcom_url

        address, separator, port_str = hcom_url.rpartition(":")
        if not separator or not address.startswith("tcp://"):
            raise ValueError(
                "ASCEND_MF_HCOM_URL must use tcp://<IPv4>:<port> or "
                "tcp://<IPv4>/<mask>:<port>"
            )

        try:
            base_port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}"
            ) from exc

        # Each SGLang worker creates an independent MemFabric store and is rank
        # zero in that store. Reserve one port for each role at every SGLang
        # world rank so that colocated Prefill and Decode workers never bind the
        # same HCOM service endpoint. DP-attention does not change world_rank.
        role_slot = 0 if self.role == "Decode" else 1
        worker_port = base_port + world_rank * 2 + role_slot
        if not 1024 <= worker_port <= 65535:
            raise ValueError(
                "Resolved ASCEND_MF_HCOM_URL port is out of range: "
                f"base_port={base_port}, world_rank={world_rank}, "
                f"role={self.role}, resolved_port={worker_port}"
            )

        worker_hcom_url = f"{address}:{worker_port}"
        logger.info(
            "Resolved Ascend Host RDMA endpoint: role=%s, world_rank=%d, "
            "base=%s, endpoint=%s",
            self.role,
            world_rank,
            hcom_url,
            worker_hcom_url,
        )
        return worker_hcom_url
