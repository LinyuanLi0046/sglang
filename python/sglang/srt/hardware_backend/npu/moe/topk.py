from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _apply_routed_scaling_after_renorm(
    topk_weights: torch.Tensor,
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Mirror GPU post-renorm scaling when apply_routed_scaling_factor_on_output is set."""
    if (
        topk_config.renormalize
        and topk_config.apply_routed_scaling_factor_on_output
        and topk_config.routed_scaling_factor is not None
    ):
        return topk_weights * topk_config.routed_scaling_factor
    return topk_weights


def fused_expert_bias_topk_npu(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ungrouped expert-bias routing with Ascend MoeGatingTopK.

    ``expert_bias`` participates only in expert selection. Returned weights are
    gathered from the un-biased normalized scores, matching WeLM's custom
    routing callback. Ascend's native TopK tie order is intentionally used.
    """
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(
            "fused_expert_bias_topk_npu only supports softmax or sigmoid, "
            f"got {scoring_func!r}"
        )
    if router_logits.ndim != 2:
        raise ValueError(
            "router_logits must be 2D for fused NPU TopK, "
            f"got shape {tuple(router_logits.shape)}"
        )
    if expert_bias.ndim != 1 or expert_bias.shape[0] != router_logits.shape[1]:
        raise ValueError(
            "expert_bias must be 1D and match the expert dimension, "
            f"got {tuple(expert_bias.shape)} for logits {tuple(router_logits.shape)}"
        )

    # MoeGatingTopK requires x and bias to have the same dtype. WeLM's router
    # and expert bias are FP32, so the casts are normally no-ops.
    logits_fp32 = router_logits.to(torch.float32)
    bias_fp32 = expert_bias.to(device=router_logits.device, dtype=torch.float32)
    op_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
        logits_fp32,
        k=top_k,
        bias=bias_fp32,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=1 if renormalize else 0,
        norm_type=0 if scoring_func == "softmax" else 1,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=float(1e-20),
    )

    if renormalize or scoring_func == "softmax":
        # For softmax, renorm=1 makes the op normalize the selected scores. For
        # sigmoid, MoeGatingTopK always normalizes selected scores. Both match
        # WeLM when renormalize=True. Softmax + renormalize=False already
        # returns the selected un-biased softmax scores.
        topk_weights = op_weights
    else:
        # Sigmoid + renormalize=False cannot use op_weights: MoeGatingTopK
        # normalizes sigmoid weights regardless of the renorm attribute. Avoid
        # materializing the op's full [M, E] normOut; gather only [M, K] raw
        # logits and apply the same elementwise sigmoid used by WeLM.
        selected_logits = torch.gather(
            logits_fp32,
            dim=1,
            index=topk_ids.to(torch.int64),
        )
        topk_weights = torch.sigmoid(selected_logits)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
    expert_bias: Optional[torch.Tensor] = None,
    scoring_func_override: Optional[str] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    # WeLM opt-in path: top-k over (scores + expert_bias); weights from the
    # un-biased scores. This bypasses its custom callback without changing it.
    if expert_bias is not None:
        topk_weights, topk_ids = fused_expert_bias_topk_npu(
            router_logits,
            expert_bias,
            top_k=topk_config.top_k,
            scoring_func=scoring_func_override or topk_config.scoring_func,
            renormalize=renormalize,
        )

    # sqrtsoftplus (DSV4 noaux_tc): top-k over (scores + bias); weights from
    # un-biased scores. The custom op fuses softplus/sqrt/topk/gather/norm/cast.
    elif topk_config.scoring_func == "sqrtsoftplus":
        routed_scaling_factor = (
            topk_config.routed_scaling_factor
            if topk_config.apply_routed_scaling_factor_on_output
            else 1.0
        )
        topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(
            x=router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            input_ids=None,
            tid2eid=None,
            routed_scaling_factor=float(routed_scaling_factor),
            norm_type=2,
        )
        topk_weights = topk_weights.to(torch.float32)

    # Fast path: simple top-k without grouped routing and bias
    elif not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )

        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            # 1 for sigmoid, 0 for softmax
            norm_type=(0 if topk_config.scoring_func == "softmax" else 1),
            routed_scaling_factor=(
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
