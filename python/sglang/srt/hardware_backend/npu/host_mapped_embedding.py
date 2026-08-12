"""Host-backed NPU tensors for the large WeLMv4 embedding tables.

The allocation has two aliases over the same physical host memory:

* ``cpu_view`` uses the original host pointer and is the only view used while
  loading or inspecting weights.
* ``npu_view`` uses the device pointer returned by ``acl.rt.host_register`` and
  is read by NPU embedding kernels.

CANN explicitly forbids memory-copy operations on the mapped device pointer,
so callers must never load a checkpoint through ``npu_view.copy_``.  Keeping
the two views in one owner also makes the register/unregister lifetime explicit.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HOST_PAGE_SIZE = 4096
# aclrtHostRegisterType and the similarly named V2 flags intentionally use
# different values.  This is ACL_HOST_REGISTER_MAPPED for host_register, not
# ACL_HOST_REG_MAPPED (0x2) for host_register_v2.
_ACL_HOST_REGISTER_MAPPED = 0

_PROBE_LOCK = threading.Lock()
_PROBED_CONFIGS: set[tuple[int, torch.dtype]] = set()


def _synchronize_npu(device_id: int) -> None:
    # torch_npu versions differ on whether synchronize accepts an explicit
    # device. The model worker has already selected its current device, so the
    # no-argument form is sufficient and is the most portable one.
    del device_id
    torch.npu.synchronize()


def _load_acl():
    try:
        import acl
    except ImportError as exc:  # pragma: no cover - exercised on an NPU host
        raise RuntimeError(
            "NPU host-backed WeLMv4 embeddings require the pyACL `acl` module "
            "from the active CANN installation."
        ) from exc
    return acl


def _check_acl(ret: int, operation: str) -> None:
    if int(ret) != 0:
        raise RuntimeError(f"{operation} failed with ACL error code {int(ret)}")


def _align_up(value: int, alignment: int = _HOST_PAGE_SIZE) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a positive power of two, got {alignment}")
    return (value + alignment - 1) & -alignment


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        return ()
    strides = [0] * len(shape)
    strides[-1] = 1
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def _tensor_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"negative tensor dimension in {shape}")
    numel = math.prod(shape)
    if numel == 0:
        raise ValueError("NPU host-backed embedding tensors must be non-empty")
    return numel * torch.empty((), dtype=dtype, device="cpu").element_size()


def _storage_from_pointer(pointer: int, device: torch.device, nbytes: int):
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "This PyTorch build does not expose "
            "torch._C._construct_storage_from_data_pointer."
        )
    return constructor(pointer, device, nbytes)


def _npu_device(device_id: int) -> torch.device:
    return torch.device("npu", int(device_id))


def _tensor_from_storage(
    storage: Any,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in shape)
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage,
        0,
        shape,
        _contiguous_strides(shape),
    )


class HostMappedNPUAllocation:
    """Own an ACL-registered host allocation and its CPU/NPU tensor aliases."""

    def __init__(
        self,
        *,
        acl_module: Any,
        host_base_ptr: int,
        host_ptr: int,
        dev_ptr: int,
        allocated_bytes: int,
        tensor_bytes: int,
        device_id: int,
        cpu_storage: Any,
        npu_storage: Any,
        cpu_view: torch.Tensor,
        npu_view: torch.Tensor,
    ) -> None:
        self._acl = acl_module
        self.host_base_ptr = int(host_base_ptr)
        self.host_ptr = int(host_ptr)
        self.dev_ptr = int(dev_ptr)
        self.allocated_bytes = int(allocated_bytes)
        self.tensor_bytes = int(tensor_bytes)
        self.device_id = int(device_id)
        self.cpu_storage = cpu_storage
        self.npu_storage = npu_storage
        self.cpu_view = cpu_view
        self.npu_view = npu_view
        self.registered = True
        self.freed = False
        self.closed = False

    def close(self, *, synchronize: bool = True) -> None:
        if self.closed:
            return
        if synchronize:
            _synchronize_npu(self.device_id)

        if self.registered:
            ret = self._acl.rt.host_unregister(self.host_ptr)
            _check_acl(ret, "acl.rt.host_unregister")
            self.registered = False
        if not self.freed:
            ret = self._acl.rt.free_host(self.host_base_ptr)
            _check_acl(ret, "acl.rt.free_host")
            self.freed = True
        self.closed = not self.registered and self.freed

    def __del__(self) -> None:
        if getattr(self, "closed", True):
            return
        try:
            self.close(synchronize=True)
        except Exception:
            # The ACL runtime may already be torn down during interpreter exit.
            # Leaking here is safer than freeing memory still referenced by a
            # device task; the operating system reclaims it with the process.
            pass


def _best_effort_release_partial(
    acl_module: Any,
    host_base_ptr: int | None,
    host_ptr: int | None,
    registered: bool,
) -> None:
    if registered and host_ptr is not None:
        try:
            acl_module.rt.host_unregister(host_ptr)
        except Exception:
            pass
    if host_base_ptr is not None:
        try:
            acl_module.rt.free_host(host_base_ptr)
        except Exception:
            pass


def _allocate_without_probe(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int,
) -> HostMappedNPUAllocation:
    acl_module = _load_acl()
    shape = tuple(int(dim) for dim in shape)
    tensor_bytes = _tensor_nbytes(shape, dtype)
    # acl.rt.malloc_host guarantees only 64-byte alignment while
    # acl.rt.host_register requires a 4 KiB page-aligned host address.
    allocated_bytes = tensor_bytes + _HOST_PAGE_SIZE - 1
    host_base_ptr = None
    host_ptr = None
    registered = False

    try:
        host_base_ptr, ret = acl_module.rt.malloc_host(allocated_bytes)
        _check_acl(ret, "acl.rt.malloc_host")
        host_base_ptr = int(host_base_ptr)
        host_ptr = _align_up(host_base_ptr)
        if host_ptr + tensor_bytes > host_base_ptr + allocated_bytes:
            raise RuntimeError("internal error while aligning ACL host allocation")

        dev_ptr, ret = acl_module.rt.host_register(
            host_ptr,
            tensor_bytes,
            _ACL_HOST_REGISTER_MAPPED,
        )
        _check_acl(ret, "acl.rt.host_register")
        dev_ptr = int(dev_ptr)
        registered = True

        cpu_device = torch.device("cpu")
        npu_device = _npu_device(device_id)
        cpu_storage = _storage_from_pointer(host_ptr, cpu_device, tensor_bytes)
        npu_storage = _storage_from_pointer(dev_ptr, npu_device, tensor_bytes)
        cpu_view = _tensor_from_storage(cpu_storage, shape, dtype, cpu_device)
        npu_view = _tensor_from_storage(npu_storage, shape, dtype, npu_device)

        allocation = HostMappedNPUAllocation(
            acl_module=acl_module,
            host_base_ptr=host_base_ptr,
            host_ptr=host_ptr,
            dev_ptr=dev_ptr,
            allocated_bytes=allocated_bytes,
            tensor_bytes=tensor_bytes,
            device_id=device_id,
            cpu_storage=cpu_storage,
            npu_storage=npu_storage,
            cpu_view=cpu_view,
            npu_view=npu_view,
        )
        return allocation
    except Exception:
        _best_effort_release_partial(
            acl_module,
            host_base_ptr,
            host_ptr,
            registered,
        )
        raise


def _probe_host_mapped_embedding(dtype: torch.dtype, device_id: int) -> None:
    allocation = None
    probe_error = None
    try:
        allocation = _allocate_without_probe((8, 16), dtype, device_id)
        reference = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
        reference = (reference / 127.0).to(dtype=dtype)
        allocation.cpu_view.copy_(reference)

        ids_cpu = torch.tensor([0, 3, 7, 3], dtype=torch.long)
        ids_npu = ids_cpu.to(device=torch.device("npu", device_id))
        actual = F.embedding(ids_npu, allocation.npu_view)
        _synchronize_npu(device_id)
        expected = F.embedding(ids_cpu, reference)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        if allocation.npu_view.data_ptr() != allocation.dev_ptr:
            raise RuntimeError(
                "the external NPU tensor does not retain the ACL mapped device pointer"
            )
        if allocation.cpu_view.data_ptr() != allocation.host_ptr:
            raise RuntimeError(
                "the external CPU tensor does not retain the ACL host pointer"
            )
    except Exception as exc:
        probe_error = exc
    finally:
        if allocation is not None:
            try:
                allocation.close(synchronize=True)
            except Exception as exc:
                if probe_error is None:
                    probe_error = exc
                else:
                    logger.exception(
                        "Failed to release the temporary NPU host-backed "
                        "embedding probe allocation"
                    )

    if probe_error is not None:
        raise RuntimeError(
            "NPU host-backed WeLMv4 embedding probe failed. The active "
            "CANN/PyTorch/TorchNPU stack could not use an acl.rt.host_register "
            "address as an F.embedding weight; no silent HBM fallback was made."
        ) from probe_error


def ensure_host_mapped_embedding_supported(
    dtype: torch.dtype,
    device_id: int,
) -> None:
    """Run the external-storage embedding probe once per device and dtype."""

    key = (int(device_id), dtype)
    if key in _PROBED_CONFIGS:
        return
    with _PROBE_LOCK:
        if key in _PROBED_CONFIGS:
            return
        _probe_host_mapped_embedding(dtype, int(device_id))
        _PROBED_CONFIGS.add(key)
        logger.info(
            "Validated NPU host-backed embedding storage on npu:%d with dtype=%s",
            int(device_id),
            dtype,
        )


def allocate_host_mapped_npu_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int,
) -> HostMappedNPUAllocation:
    """Allocate a probed host-backed NPU tensor and its CPU loading alias."""

    ensure_host_mapped_embedding_supported(dtype, int(device_id))
    allocation = _allocate_without_probe(shape, dtype, int(device_id))
    logger.info(
        "Allocated NPU host-backed embedding: device=npu:%d shape=%s "
        "dtype=%s tensor_bytes=%d host_ptr=%#x dev_ptr=%#x",
        int(device_id),
        tuple(int(dim) for dim in shape),
        dtype,
        allocation.tensor_bytes,
        allocation.host_ptr,
        allocation.dev_ptr,
    )
    return allocation


def get_host_mapped_cpu_view(tensor: torch.Tensor) -> torch.Tensor | None:
    allocation = getattr(tensor, "_host_mapped_npu_allocation", None)
    if allocation is None:
        return None
    if allocation.closed:
        raise RuntimeError("attempted to access a released host-backed NPU tensor")
    return allocation.cpu_view
