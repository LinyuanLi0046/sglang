import pytest
import torch

from sglang.srt.layers.welmv4_npu_op import (
    welmv4_oe_hash_explicit_history_4way_npu,
    welmv4_token_table_spec_accept_update_npu,
)
from sglang.test.ci.ci_register import register_npu_ci


register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")


_HASH_MULTIPLIER = 2654435761
_UINT32_MASK = (1 << 32) - 1
_VOCAB_SIZE = 155648
_OE_VOCAB_SIZES = (16000008, 16000016, 16000024, 16000032)


def _explicit_history_hash_reference(
    current: torch.Tensor,
    previous1: torch.Tensor,
    previous2: torch.Tensor,
) -> torch.Tensor:
    """Independent CPU reference with the kernel's uint32 wrap semantics."""
    branches = [[] for _ in _OE_VOCAB_SIZES]
    for current_id, previous1_id, previous2_id in zip(
        current.tolist(), previous1.tolist(), previous2.tolist()
    ):
        packed2 = (current_id + previous1_id * _VOCAB_SIZE) & _UINT32_MASK
        packed3 = (
            packed2 + previous2_id * _VOCAB_SIZE * _VOCAB_SIZE
        ) & _UINT32_MASK
        hash2 = (packed2 * _HASH_MULTIPLIER) & _UINT32_MASK
        hash3 = (packed3 * _HASH_MULTIPLIER) & _UINT32_MASK
        hashes = (hash2, hash2, hash3, hash3)
        for branch, (value, divisor) in enumerate(
            zip(hashes, _OE_VOCAB_SIZES)
        ):
            branches[branch].append(value % divisor)
    return torch.tensor(branches, dtype=torch.int32)


@pytest.mark.parametrize("batch_size", [1, 127, 129, 257])
@torch.no_grad()
def test_explicit_history_hash_matches_uint32_reference(batch_size):
    indices = torch.arange(batch_size, dtype=torch.int64)
    current = (indices * 7919 + 17) % _VOCAB_SIZE
    previous1 = (indices * 104729 + 65537) % _VOCAB_SIZE
    previous2 = (indices * 13007 + 131071) % _VOCAB_SIZE
    expected = _explicit_history_hash_reference(current, previous1, previous2)

    actual = welmv4_oe_hash_explicit_history_4way_npu(
        current.to("npu"),
        previous1.to("npu"),
        previous2.to("npu"),
        vocab_size=_VOCAB_SIZE,
        oe_vocab_sizes=_OE_VOCAB_SIZES,
    )

    torch.npu.synchronize()
    assert actual.dtype == torch.int32
    assert actual.shape == (4, batch_size)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _commit_accepts_reference(
    table: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    old_seq_lens: torch.Tensor,
) -> torch.Tensor:
    expected = table.clone()
    width = accept_index.shape[1]
    context_len = table.shape[1]
    for batch_idx in range(accept_index.shape[0]):
        row = int(req_pool_indices[batch_idx])
        for step in range(min(int(accept_lens[batch_idx]), width)):
            column = int(old_seq_lens[batch_idx]) + 1 + step
            if column >= context_len:
                continue
            source = max(int(accept_index[batch_idx, step]), 0)
            expected[row, column] = predict[source].to(expected.dtype)
    return expected


@torch.no_grad()
def test_spec_accept_commit_matches_reference_and_preserves_incoming_root():
    table = torch.full((6, 18), -1, dtype=torch.int32)
    predict = torch.arange(32, dtype=torch.int32) * 13 + 100
    accept_index = torch.tensor(
        [
            [-1, -1, -1, -1, -1],
            [2, -1, -1, -1, -1],
            [4, 6, 8, -1, -1],
            [1, 3, 5, 7, 9],
        ],
        dtype=torch.int32,
    )
    accept_lens = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
    req_pool_indices = torch.tensor([4, 1, 5, 2], dtype=torch.int64)
    old_seq_lens = torch.tensor([3, 7, 12, 16], dtype=torch.int64)

    # old_seq_lens points at the incoming bonus/root and must remain untouched.
    for batch_idx, row in enumerate(req_pool_indices.tolist()):
        table[row, int(old_seq_lens[batch_idx])] = 9000 + batch_idx
    expected = _commit_accepts_reference(
        table,
        predict,
        accept_index,
        accept_lens,
        req_pool_indices,
        old_seq_lens,
    )

    actual = table.to("npu")
    welmv4_token_table_spec_accept_update_npu(
        actual,
        predict.to("npu"),
        accept_index.to("npu"),
        accept_lens.to("npu"),
        req_pool_indices.to("npu"),
        old_seq_lens.to("npu"),
    )

    torch.npu.synchronize()
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for batch_idx, row in enumerate(req_pool_indices.tolist()):
        root_column = int(old_seq_lens[batch_idx])
        assert actual[row, root_column].item() == 9000 + batch_idx

