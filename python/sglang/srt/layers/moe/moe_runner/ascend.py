"""Ascend MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.activation import (
    AllGatherActivationWrapper,
    NPUGeluAndMul,
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluStepAndMul,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUUnquantMoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
        AscendLocalEPCombineInput,
        AscendLocalEPDispatchOutput,
        AscendTPDispatchOutput,
        AscendTPCombineInput,
    )

from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
)


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AscendRunnerInput(RunnerInput):
    """Input bundle for the NPU runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # None for unquant
    expert_tokens: torch.Tensor
    group_list_type: int  # 0 or 1 (passed to NPU ops)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


@dataclass
class AscendRunnerOutput(RunnerOutput):
    """Output bundle from the NPU runner."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


# ---------------------------------------------------------------------------
# Main runner core
# ---------------------------------------------------------------------------
class AscendRunnerCore(MoeRunnerCore):
    runner_backend = MoeRunnerBackend.ASCEND

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

        self._welm_compile_region = None

        kernel = config.layer.w2_kernel

        if isinstance(kernel, NPUMXFP8MoEMethod):
            # MXFP8 fuses gate/up + swiglu + requant into gmm1, so there is no
            # separate activation step — run() skips it. Left None on purpose so
            # that reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds for
            # both dispatchers: ascend_tp gets its activation quant fused into
            # routing, DeepEP dispatches bf16 and gmm1 quantises it itself.
            self.activation = None
        elif get_moe_a2a_backend().is_deepep():
            # DeepEP path: use a unified kernel that decides quantisation
            is_quant_kernel = isinstance(
                kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
            )
            self.activation = NPUSwigluDeepEPKernel(need_quant=is_quant_kernel)
        else:
            # Non‑DeepEP (ascend_tp) path
            # 1. Choose the base activation according to the quant method
            if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
                inner = NPUSwigluQuant()
            else:
                if config.activation == "npu_swiglu_oai":
                    # NPUSwigluOAI requires the runner config to pass
                    # gemm1_alpha and gemm1_clamp_limit to the triton kernel.
                    inner = NPUSwigluOAI(moe_runner_config=config)
                elif config.activation == "silu":
                    if config.gemm1_clamp_limit is not None:
                        inner = NPUSwigluStepAndMul(
                            clamp_limit=config.gemm1_clamp_limit
                        )
                    else:
                        inner = NPUSwiglu()
                else:
                    inner = NPUGeluAndMul()

            # 2. If the quant method (GGUF) needs TP all‑gather, wrap the activation
            if getattr(config, "use_tp_all_gather_activation", False):
                self.activation = AllGatherActivationWrapper(inner, dim=-1)
            else:
                self.activation = inner

    def enable_welm_compile_region(self, name: str):
        """Bind the WeLM FFN region without changing the ordinary runner path."""
        if self._welm_compile_region is not None:
            return self._welm_compile_region

        w13_kernel = self.config.layer.w13_kernel
        w2_kernel = self.config.layer.w2_kernel
        use_swiglu_group = isinstance(w13_kernel, NPUUnquantMoEMethod) and isinstance(
            w2_kernel, NPUUnquantMoEMethod
        )
        is_mxfp8 = isinstance(w13_kernel, NPUMXFP8MoEMethod) and isinstance(
            w2_kernel, NPUMXFP8MoEMethod
        )
        if not (use_swiglu_group or is_mxfp8):
            raise ValueError(
                "WeLM routed FFN regional compilation supports unquantized "
                "and MXFP8 Ascend GMMs only"
            )
        if is_mxfp8:
            from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
                _require_e8m0_dtype,
            )

            # Resolve the lazy dtype before entering a fullgraph FX trace.
            _require_e8m0_dtype()
        if use_swiglu_group:
            # DeepEP's existing activation does not apply the configured clamp.
            # A TP post-SiLU clamp cannot be replaced by SwigluGroup's pre-clamp.
            if not isinstance(self.activation, (NPUSwiglu, NPUSwigluDeepEPKernel)):
                raise ValueError(
                    "WeLM SwigluGroup compilation requires an unclamped SwiGLU "
                    "without activation AllGather"
                )
            import torch_npu

            if not torch_npu.npu.get_device_name().startswith("Ascend950"):
                raise ValueError("WeLM SwigluGroup compilation requires Ascend 950")
            if not hasattr(torch.ops.cann_ops_nn, "swiglu_group"):
                package = os.environ.get(
                    "WELM_SWIGLU_TORCH_PACKAGE", "cann_ops_nn_welm_swiglu91"
                )
                try:
                    module = import_module(
                        package + ".ops.activation.swiglu_group.swiglu_group"
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "WeLM routed FFN compilation requires the Ascend 950 "
                        f"SwigluGroup Torch package {package!r}; source the "
                        "SwigluGroup installation's env.sh before launching"
                    ) from exc
                # Load the original Torch bridge outside FX tracing/capture.
                module.builder.load()
            if not hasattr(torch.ops.cann_ops_nn, "swiglu_group"):
                raise RuntimeError("cann_ops_nn.swiglu_group is not registered")

        from sglang.srt.hardware_backend.npu.compile_backend import NpuCompileRegion

        self._welm_compile_region = NpuCompileRegion(
            name,
            self._run_tensor_inputs,
            compile_fn=partial(
                self._run_tensor_inputs, use_swiglu_group=use_swiglu_group
            ),
        )
        return self._welm_compile_region

    def run(
        self,
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AscendRunnerOutput:
        """
        Execute the MoE layer using NPU‑specific grouped matmul ops.
        """
        region = self._welm_compile_region
        if region is not None and region.is_active:
            # Unquantized callers pass the layer itself, which may not define
            # the optional quantization or bias attributes.
            hidden_states = region(
                runner_input.hidden_states,
                runner_input.expert_tokens,
                runner_input.hidden_states_scale,
                runner_input.group_list_type,
                quant_info.w13_weight,
                quant_info.w2_weight,
                getattr(quant_info, "w13_weight_scale", None),
                getattr(quant_info, "w2_weight_scale", None),
                getattr(quant_info, "w13_weight_offset", None),
                getattr(quant_info, "w2_weight_offset", None),
                getattr(quant_info, "w13_weight_bias", None),
                getattr(quant_info, "w2_weight_bias", None),
                getattr(quant_info, "w13_scale_bias", None),
                getattr(quant_info, "w2_scale_bias", None),
            )
        else:
            hidden_states = self._run_experts(
                runner_input.hidden_states,
                runner_input.expert_tokens,
                runner_input.hidden_states_scale,
                runner_input.group_list_type,
                quant_info,
            )
        return AscendRunnerOutput(hidden_states=hidden_states)

    def _run_tensor_inputs(
        self,
        x,
        expert_tokens,
        hidden_states_scale,
        group_list_type,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_weight_offset,
        w2_weight_offset,
        w13_weight_bias,
        w2_weight_bias,
        w13_scale_bias,
        w2_scale_bias,
        *,
        use_swiglu_group=False,
    ):
        # Only tensors and the static group-list format cross the FX boundary.
        # Rebuild the lightweight payload here to reuse the existing GMM methods.
        quant_info = AscendQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            w13_weight_offset=w13_weight_offset,
            w2_weight_offset=w2_weight_offset,
            w13_weight_bias=w13_weight_bias,
            w2_weight_bias=w2_weight_bias,
            w13_scale_bias=w13_scale_bias,
            w2_scale_bias=w2_scale_bias,
        )
        return self._run_experts(
            x,
            expert_tokens,
            hidden_states_scale,
            group_list_type,
            quant_info,
            use_swiglu_group=use_swiglu_group,
        )

    def _run_experts(
        self,
        x,
        expert_tokens,
        hidden_states_scale,
        group_list_type,
        quant_info,
        *,
        use_swiglu_group=False,
    ):
        original_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16

        w13_kernel = self.config.layer.w13_kernel

        if isinstance(w13_kernel, NPUMXFP8MoEMethod):
            # --- w13 projection + activation, fused into one kernel ---
            # MXFP8 gmm1 returns activations already requantised for gmm2, so
            # there is no separate activation step to run.
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_swiglu(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=hidden_states_scale,
                group_list_type=group_list_type,
            )
        else:
            # --- w13 (gate & up) projection ---
            hidden_states = w13_kernel.apply(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=hidden_states_scale,
                output_dtype=original_dtype,
                weight_prefix="w13",
                group_list_type=group_list_type,
            )

            # --- Activation ---
            # The DeepEP kernel expects extra dispatch metadata
            if use_swiglu_group:
                if group_list_type == 0:
                    counts = torch.cat(
                        (expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1])
                    )
                elif group_list_type == 1:
                    counts = expert_tokens
                elif group_list_type == 2:
                    # ascend_tp may use [expert index, count] metadata. GMMs
                    # retain that format; SwigluGroup only needs its counts.
                    counts = expert_tokens[:, 1].contiguous()
                else:
                    raise ValueError("Unsupported SwigluGroup group-list format")
                hidden_states = torch.ops.cann_ops_nn.swiglu_group(
                    hidden_states,
                    weight=None,
                    group_index=counts.to(torch.int64),
                    clamp_limit=-1.0,
                )
                pertoken_scale = None
            elif isinstance(self.activation, NPUSwigluDeepEPKernel):
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=group_list_type,
                )
            else:
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states
                )

        # --- w2 (down) projection ---
        hidden_states = self.config.layer.w2_kernel.apply(
            quant_info,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=original_dtype,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# QuantInfo
# ---------------------------------------------------------------------------
@dataclass
class AscendQuantInfo(MoeQuantInfo):
    """Quantization payload for Ascend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    w13_weight_offset: Optional[torch.Tensor] = None
    w2_weight_offset: Optional[torch.Tensor] = None
    w13_weight_bias: Optional[torch.Tensor] = None
    w2_weight_bias: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pre/Post permute hooks
# ---------------------------------------------------------------------------


@register_pre_permute("ascend_tp", "ascend")
def pre_permute_ascend_tp_to_ascend(
    dispatch_output: AscendTPDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    return AscendRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens,
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("ascend_local_ep", "ascend")
def pre_permute_ascend_local_ep_to_ascend(
    dispatch_output: AscendLocalEPDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    running_state["local_ep_topk_weights"] = dispatch_output.topk_weights
    running_state["local_ep_expanded_row_idx"] = dispatch_output.expanded_row_idx
    return AscendRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens,
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("deepep_normal", "ascend")
def pre_permute_deepep_normal_to_ascend(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    group_list = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int64,
        device=hidden_states.device,
    )
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_pre_permute("deepep_ll", "ascend")
def pre_permute_deepep_ll_to_ascend(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        group_list,
        _,
    ) = dispatch_output
    group_list = group_list.to(torch.int64)
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_post_permute("ascend", "ascend_tp")
def post_permute_ascend_to_ascend_tp(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendTPCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPCombineInput

    return AscendTPCombineInput(hidden_states=runner_output.hidden_states)


@register_post_permute("ascend", "ascend_local_ep")
def post_permute_ascend_to_ascend_local_ep(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendLocalEPCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
        AscendLocalEPCombineInput,
    )

    return AscendLocalEPCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_weights=running_state["local_ep_topk_weights"],
        expanded_row_idx=running_state["local_ep_expanded_row_idx"],
    )


@register_post_permute("ascend", "deepep_normal")
def post_permute_ascend_to_deepep_normal(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_post_permute("ascend", "deepep_ll")
def post_permute_ascend_to_deepep_ll(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
