"""Metadata contracts for NPUMHA cache views; no NPU kernels are executed."""

import pytest

from test_welm_fused_qkv_dispatch import (
    _dispatch_case,
    _run as _run_prefill,
    doubles as doubles,
)
from test_welm_static_decode_qkv import (
    _case,
    _load_layout,
    _run as _run_decode,
    _Tensor,
    binding as binding,
)


def _layer_caches(tensor_type, *, tp, paged, npu_format=30):
    """A nonzero layer offset in separate five-dimensional K/V allocations."""
    heads = 2 if tp == 1 else 1
    slots, kv_width = 256, heads * 256
    shape = (2, 128, heads, 256) if paged else (slots, 1, heads, 256)
    return tuple(
        tensor_type(shape, npu_format=npu_format, storage_offset=2 * slots * kv_width)
        for _ in range(2)
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", ["plain", "source", "query"])
@pytest.mark.parametrize("m", [1, 4, 32, 64])
@pytest.mark.parametrize("swa,paged", [(False, False), (True, True)])
def test_decode_ncdhw_layer_views_keep_offsets_and_full_token_rows(
    binding, tp, kind, m, swa, paged
):
    case = _case(binding, tp=tp, kind=kind, m=m, swa=swa)
    caches = _layer_caches(_Tensor, tp=tp, paged=paged)
    case.backend.token_to_kv_pool.get_kv_buffer.return_value = caches
    assert _run_decode(case) is not None
    args = binding.program.call_args.args
    kv_width = 512 if tp == 1 else 256
    for actual, original in zip(args[11:13], caches):
        assert actual.shape == (256, kv_width)
        assert actual.stride() == (kv_width, 1)
        assert actual.data_ptr() == original.data_ptr()
        assert actual.storage_offset() == original.storage_offset() > 0
    assert args[5] is (case.swa_slots if swa else case.full_slots)
    assert binding.compiler.call_args.args[0] == 256
    allocations = len(case.hidden.allocations)
    binding.npu.is_current_stream_capturing.return_value = True
    assert _run_decode(case) is not None
    binding.compiler.assert_called_once()
    captured_args = binding.program.call_args.args
    assert captured_args[6:11] == args[6:11]
    assert len(case.hidden.allocations) == allocations
    for actual, original in zip(captured_args[11:13], caches):
        assert actual.data_ptr() == original.data_ptr()
        assert actual.storage_offset() == original.storage_offset()


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("swa,paged", [(False, False), (True, True)])
def test_prefill_ncdhw_layer_views_keep_offsets_and_physical_slots(
    doubles, tp, source, swa, paged
):
    case = _dispatch_case(doubles, tp=tp, dp=True, source=source, swa=swa, m=8)
    caches = _layer_caches(doubles.Tensor, tp=tp, paged=paged)
    case.backend.token_to_kv_pool.get_kv_buffer.return_value = caches
    assert _run_prefill(case) is not None
    args = case.program.call_args.args
    kv_width = 512 if tp == 1 else 256
    for actual, original in zip(args[11:13], caches):
        assert actual.shape == (256, kv_width)
        assert actual.data_ptr() == original.data_ptr()
        assert actual.storage_offset() == original.storage_offset() > 0
    assert args[5] is case.selected_slots


@pytest.mark.parametrize("index", [0, 1], ids=["K", "V"])
@pytest.mark.parametrize("npu_format", [-1, 0, 1, 3, 27, 29, 32, 33, "NCDHW"])
def test_decode_rejects_unvalidated_cache_formats_before_launch(
    binding, index, npu_format
):
    case = _case(binding)
    case.caches[index].npu_format = npu_format
    assert _run_decode(case) is None
    binding.compiler.assert_not_called()
    binding.program.assert_not_called()


@pytest.mark.parametrize("index", [0, 1], ids=["K", "V"])
@pytest.mark.parametrize("npu_format", [3, 29, 32, 33])
def test_prefill_rejects_padded_cache_formats_before_launch(doubles, index, npu_format):
    case = _dispatch_case(doubles, dp=True)
    case.caches[index].npu_format = npu_format
    assert _run_prefill(case) is None
    case.lookup.assert_not_called()
    case.program.assert_not_called()


@pytest.mark.parametrize("field", ["hidden", "weight", "gamma", "rope"])
def test_ncdhw_allowance_is_limited_to_cache(binding, field):
    case = _case(binding)
    tensor = {
        "hidden": case.hidden,
        "weight": case.layer.qkv_proj.weight,
        "gamma": case.layer.k_norm.weight,
        "rope": case.layer.rotary_emb.cos_sin_cache,
    }[field]
    tensor.npu_format = 30
    assert _run_decode(case) is None
    binding.program.assert_not_called()


@pytest.mark.parametrize("defect", ["noncontiguous", "copied", "offset", "stride"])
def test_cache_alias_contract_rejects_bad_views(defect):
    class BadView(_Tensor):
        def view(self, *shape):
            result = super().view(*shape)
            if defect == "copied":
                result.storage = object()
            elif defect == "offset":
                result._storage_offset += 1
                result.data_ptr = self.data_ptr
            elif defect == "stride":
                result.stride = lambda: (1024, 1)
            return result

    tensor = BadView(
        (128, 1, 2, 256), npu_format=30, storage_offset=1024,
        contiguous=defect != "noncontiguous",
    )
    check = _load_layout()["qkv_cache_view"]
    for _ in range(2):
        assert check(tensor, 512) is None
