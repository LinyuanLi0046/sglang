from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Batch:
    def __init__(self, frequencies, device="cpu"):
        self.device = device
        self.reqs = [
            SimpleNamespace(sampling_params=SimpleNamespace(frequency_penalty=freq))
            for freq in frequencies
        ]


@pytest.fixture(autouse=True)
def clear_settings(monkeypatch):
    for name in (
        "FREQUENCY_PENALTY_EXCLUDE_TOKENS",
        "FREQUENCY_PENALTY_FACTOR_START",
        "FREQUENCY_PENALTY_FACTOR_OFFSET",
    ):
        monkeypatch.delenv(name, raising=False)


def make_penalizer(frequencies, device="cpu"):
    batch = Batch(frequencies, device)
    orch = BatchedPenalizerOrchestrator(8, batch, {BatchedFrequencyPenalizer})
    return batch, orch, orch.penalizers[BatchedFrequencyPenalizer]


def test_default_frequency_penalty_unchanged():
    batch, orch, pen = make_penalizer([0.5, -0.5])
    for _ in range(3):
        orch.cumulate_output_tokens(torch.tensor([2, 3]))
    logits = torch.zeros(2, 8)
    orch.apply(logits)
    expected = torch.zeros(2, 8)
    expected[0, 2], expected[1, 3] = -1.5, 1.5
    torch.testing.assert_close(logits, expected)


def test_excluded_tokens_on_batch_device(monkeypatch):
    monkeypatch.setenv("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "[2, 99]")

    # Module/penalizer initialization must never enumerate CUDA devices.
    def fail_cuda():
        raise AssertionError("unexpected CUDA initialization")

    monkeypatch.setattr(torch.cuda, "device_count", fail_cuda)
    batch, orch, pen = make_penalizer([0.5, 1.0])
    orch.cumulate_output_tokens(torch.tensor([2, 3]))
    orch.cumulate_output_tokens(torch.tensor([4, 2]))
    expected = torch.zeros(2, 8)
    expected[0, 4], expected[1, 3] = 0.5, 1.0
    torch.testing.assert_close(pen.cumulated_frequency_penalties, expected)
    assert pen.token_penalty_mask.device == pen.frequency_penalties.device


def test_length_scaling_boundaries(monkeypatch):
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_START", "2")
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_OFFSET", "2")
    batch, orch, pen = make_penalizer([0.5])
    for step, expected in enumerate([0.5, 1.0, 1.5, 2.5, 3.5, 5.0]):
        orch.cumulate_output_tokens(torch.tensor([3]))
        assert pen.cumulated_frequency_penalties[0, 3].item() == expected


def test_scaling_and_exclusion_together(monkeypatch):
    monkeypatch.setenv("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "[2]")
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_START", "0")
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_OFFSET", "2")
    batch, orch, pen = make_penalizer([1.0])
    orch.cumulate_output_tokens(torch.tensor([2]))
    orch.cumulate_output_tokens(torch.tensor([3]))
    assert pen.cumulated_frequency_penalties[0, 2].item() == 0
    assert pen.cumulated_frequency_penalties[0, 3].item() == 2


def test_request_lengths_survive_merge_and_filter(monkeypatch):
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_START", "0")
    monkeypatch.setenv("FREQUENCY_PENALTY_FACTOR_OFFSET", "2")
    old_batch, old_orch, old = make_penalizer([1.0])
    for _ in range(3):
        old_orch.cumulate_output_tokens(torch.tensor([3]))
    new_batch, new_orch, new = make_penalizer([1.0, 0.5])
    old_orch.merge(new_orch)
    old_batch.reqs.extend(new_batch.reqs)
    old_orch.cumulate_output_tokens(torch.tensor([3, 3, 3]))
    torch.testing.assert_close(old.generation_lengths, torch.tensor([[4], [1], [1]]))
    torch.testing.assert_close(
        old.cumulated_frequency_penalties[:, 3], torch.tensor([8.0, 1.0, 0.5])
    )
    old_batch.reqs = [old_batch.reqs[2], old_batch.reqs[0]]
    old_orch.filter(torch.tensor([2, 0]))
    old_orch.cumulate_output_tokens(torch.tensor([3, 3]))
    torch.testing.assert_close(old.generation_lengths, torch.tensor([[2], [5]]))
    torch.testing.assert_close(
        old.cumulated_frequency_penalties[:, 3], torch.tensor([1.5, 11.0])
    )
    old_orch.release()
    assert not hasattr(old, "generation_lengths")
    assert not hasattr(old, "token_penalty_mask")


@pytest.mark.parametrize(
    "name,value",
    [
        ("FREQUENCY_PENALTY_FACTOR_OFFSET", "0"),
        ("FREQUENCY_PENALTY_FACTOR_OFFSET", "-1"),
        ("FREQUENCY_PENALTY_FACTOR_START", "bad"),
        ("FREQUENCY_PENALTY_FACTOR_START", "-1"),
        ("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "{}"),
        ("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "[1.5]"),
        ("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "[-1]"),
    ],
)
def test_invalid_config_has_actionable_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        make_penalizer([1.0])
