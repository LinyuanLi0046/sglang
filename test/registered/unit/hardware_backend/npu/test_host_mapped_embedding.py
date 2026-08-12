from types import SimpleNamespace

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu import host_mapped_embedding as host_mapped


class _FakeACLRuntime:
    def __init__(self):
        self.register_args = None
        self.unregister_args = []
        self.free_args = []

    def malloc_host(self, size):
        self.malloc_size = size
        return 0x1003, 0

    def host_register(self, pointer, size, register_type):
        self.register_args = (pointer, size, register_type)
        return 0x9000, 0

    def host_unregister(self, pointer):
        self.unregister_args.append(pointer)
        return 0

    def free_host(self, pointer):
        self.free_args.append(pointer)
        return 0


def test_alignment_and_contiguous_strides():
    assert host_mapped._align_up(0x1000) == 0x1000
    assert host_mapped._align_up(0x1003) == 0x2000
    assert host_mapped._contiguous_strides(()) == ()
    assert host_mapped._contiguous_strides((3,)) == (1,)
    assert host_mapped._contiguous_strides((2, 3, 5)) == (15, 5, 1)


def test_allocation_aligns_registers_and_releases_base_pointer(monkeypatch):
    fake_runtime = _FakeACLRuntime()
    fake_acl = SimpleNamespace(rt=fake_runtime)

    def fake_storage(pointer, device, nbytes):
        return SimpleNamespace(pointer=pointer, device=device, nbytes=nbytes)

    def fake_tensor(storage, shape, dtype, device):
        return SimpleNamespace(
            storage=storage,
            shape=tuple(shape),
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(host_mapped, "_load_acl", lambda: fake_acl)
    monkeypatch.setattr(host_mapped, "_storage_from_pointer", fake_storage)
    monkeypatch.setattr(host_mapped, "_tensor_from_storage", fake_tensor)
    monkeypatch.setattr(host_mapped, "_npu_device", lambda device_id: f"npu:{device_id}")

    allocation = host_mapped._allocate_without_probe((2, 3), torch.float16, 0)

    assert allocation.host_base_ptr == 0x1003
    assert allocation.host_ptr == 0x2000
    assert allocation.dev_ptr == 0x9000
    assert allocation.tensor_bytes == 12
    assert fake_runtime.malloc_size == 12 + 4095
    assert fake_runtime.register_args == (0x2000, 12, 0)

    allocation.close(synchronize=False)
    allocation.close(synchronize=False)
    assert fake_runtime.unregister_args == [0x2000]
    assert fake_runtime.free_args == [0x1003]


def test_vocab_loader_writes_host_alias_after_tp_slice():
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    target = torch.full((4, 3), -1, dtype=torch.float16)
    npu_placeholder = Parameter(torch.full((4, 3), 99, dtype=torch.float16))
    npu_placeholder.output_dim = 0
    npu_placeholder._host_mapped_npu_allocation = SimpleNamespace(cpu_view=target)

    layer = SimpleNamespace(
        tp_size=2,
        use_presharded_weights=False,
        org_vocab_size=6,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=3,
            org_vocab_end_index=6,
        ),
    )
    checkpoint = torch.arange(18, dtype=torch.float16).reshape(6, 3)

    VocabParallelEmbedding.weight_loader(layer, npu_placeholder, checkpoint)

    torch.testing.assert_close(target[:3], checkpoint[3:6], rtol=0, atol=0)
    assert torch.count_nonzero(target[3]).item() == 0
    # The mapped NPU Parameter must not be the target of copy_/fill_.
    assert torch.all(npu_placeholder == 99)
