from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


def _fused_expert_bias_topk_npu(*args, **kwargs):
    from sglang.srt.hardware_backend.npu.moe.topk import (
        fused_expert_bias_topk_npu,
    )

    return fused_expert_bias_topk_npu(*args, **kwargs)


def _install_fake_npu_op(monkeypatch, *, force_normalized_weights=False):
    calls = []

    def fake_op(x, **kwargs):
        scores = torch.softmax(x, dim=-1) if kwargs["norm_type"] == 0 else x.sigmoid()
        routing_scores = scores + kwargs["bias"]
        _, ids = torch.topk(routing_scores, k=kwargs["k"], dim=-1)
        weights = scores.gather(1, ids)
        if force_normalized_weights or kwargs["norm_type"] == 1 or kwargs["renorm"]:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        norm_scores = scores if kwargs["out_flag"] else torch.empty(0)
        calls.append(kwargs)
        return weights, ids.to(torch.int32), norm_scores

    monkeypatch.setattr(torch.ops, "npu", SimpleNamespace(npu_moe_gating_top_k=fake_op))
    return calls


def test_sigmoid_without_renorm_gathers_unbiased_scores(monkeypatch):
    calls = _install_fake_npu_op(monkeypatch, force_normalized_weights=True)
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    bias = torch.tensor([0.0, -0.5, 0.8, 0.0], dtype=torch.float32)

    weights, ids = _fused_expert_bias_topk_npu(
        logits,
        bias,
        top_k=2,
        scoring_func="sigmoid",
        renormalize=False,
    )

    scores = logits.sigmoid()
    expected_ids = torch.topk(scores + bias, k=2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    assert torch.equal(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    assert calls[0]["out_flag"] is False


def test_softmax_without_renorm_keeps_unbiased_scores(monkeypatch):
    calls = _install_fake_npu_op(monkeypatch)
    logits = torch.tensor([[1.0, -0.5, 0.2, 2.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.7, 0.0, -0.4], dtype=torch.float32)

    weights, ids = _fused_expert_bias_topk_npu(
        logits,
        bias,
        top_k=2,
        scoring_func="softmax",
        renormalize=False,
    )

    scores = torch.softmax(logits, dim=-1)
    expected_ids = torch.topk(scores + bias, k=2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    assert torch.equal(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    assert calls[0]["out_flag"] is False


def test_rejects_unsupported_scoring_function():
    logits = torch.zeros((1, 4), dtype=torch.float32)
    bias = torch.zeros((4,), dtype=torch.float32)

    try:
        _fused_expert_bias_topk_npu(
            logits,
            bias,
            top_k=2,
            scoring_func="sqrtsoftplus",
            renormalize=False,
        )
    except ValueError as error:
        assert "softmax or sigmoid" in str(error)
    else:
        raise AssertionError("unsupported scoring function did not raise")


def test_softmax_with_renorm_uses_fused_output(monkeypatch):
    calls = _install_fake_npu_op(monkeypatch)
    logits = torch.tensor([[1.0, -0.5, 0.2, 2.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.7, 0.0, -0.4], dtype=torch.float32)

    weights, ids = _fused_expert_bias_topk_npu(
        logits,
        bias,
        top_k=2,
        scoring_func="softmax",
        renormalize=True,
    )

    scores = torch.softmax(logits, dim=-1)
    expected_ids = torch.topk(scores + bias, k=2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    assert calls[0]["out_flag"] is False
