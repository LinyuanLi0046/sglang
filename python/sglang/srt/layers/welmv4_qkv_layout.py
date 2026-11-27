"""Metadata-only layout guards for fused WeLM QKV tensors and cache aliases."""

import torch_npu


def qkv_tensor_is_nd(tensor):
    return torch_npu.get_npu_format(tensor) == 2


def qkv_cache_view(tensor, kv_width):
    """Return a token-major alias of an already validated BF16 NPUMHA cache.

    NPUMHA allocates [layer, page, token, head, dim] in one five-dimensional
    storage. Its contiguous per-layer views can retain ACL_FORMAT_NCDHW (30).
    Like ND (2), this is an unpadded base format with the original dimension
    order. The kernel uses the view's data pointer, including its layer offset.
    Do not apply this allowance to weights, or to padded NZ/NC1HWC0 caches.
    Callers still own dtype, device, head geometry and pool-type validation.
    """
    if not tensor.is_contiguous():
        return None
    if torch_npu.get_npu_format(tensor) not in (2, 30):
        return None
    # view cannot copy a cache. Check the alias explicitly: writing a temporary
    # would silently leave the Attention/HiCache/PD storage unchanged.
    view = tensor.view(-1, kv_width)
    if (
        view.data_ptr() != tensor.data_ptr()
        or view.storage_offset() != tensor.storage_offset()
        or view.stride() != (kv_width, 1)
    ):
        return None
    return view
