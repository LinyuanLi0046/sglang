import pytest
import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.host_mapped_embedding import (
    allocate_host_mapped_npu_tensor,
)


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_host_mapped_embedding_matches_cpu(dtype):
    device_id = torch.npu.current_device()
    allocation = allocate_host_mapped_npu_tensor((32, 64), dtype, device_id)
    try:
        reference = torch.arange(
            32 * 64, dtype=torch.float32, device="cpu"
        ).reshape(32, 64)
        reference = (reference / 257.0).to(dtype)
        allocation.cpu_view.copy_(reference)

        ids_cpu = torch.tensor(
            [0, 1, 7, 31, 7], dtype=torch.long, device="cpu"
        )
        ids_npu = ids_cpu.to(f"npu:{device_id}")
        actual = F.embedding(ids_npu, allocation.npu_view)
        torch.npu.synchronize()

        torch.testing.assert_close(
            actual.cpu(),
            F.embedding(ids_cpu, reference),
            rtol=0,
            atol=0,
        )
        assert allocation.cpu_view.data_ptr() == allocation.host_ptr
        assert allocation.npu_view.data_ptr() == allocation.dev_ptr
    finally:
        allocation.close(synchronize=True)


def test_npu_host_mapped_embedding_graph_replay():
    device_id = torch.npu.current_device()
    dtype = torch.bfloat16
    allocation = allocate_host_mapped_npu_tensor((16, 32), dtype, device_id)
    try:
        reference = torch.arange(
            16 * 32, dtype=torch.float32, device="cpu"
        ).reshape(16, 32)
        reference = (reference / 97.0).to(dtype)
        allocation.cpu_view.copy_(reference)
        static_ids = torch.tensor(
            [0, 5, 15, 5],
            dtype=torch.long,
            device=f"npu:{device_id}",
        )

        for _ in range(2):
            F.embedding(static_ids, allocation.npu_view)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(
            graph,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            actual = F.embedding(static_ids, allocation.npu_view)

        graph.replay()
        torch.npu.synchronize()
        expected = F.embedding(static_ids.cpu(), reference)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    finally:
        allocation.close(synchronize=True)
