from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    NPUFinalizeRouting,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    get_ascend_dispatcher_output_dtype,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var


class AscendTPDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_TP


class AscendTPCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_TP


class AscendLocalEPDispatchOutput(NamedTuple):
    """Locally routed portion of a replicated-token EP batch.

    The tensor shape remains static (including graph padding), but init-routing
    only places routes for the experts owned by this rank in the valid prefix.
    No collective is issued by this dispatcher.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_LOCAL_EP


class AscendLocalEPCombineInput(NamedTuple):
    """Expert output plus all state needed for communication-free unpermute."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_LOCAL_EP


class AscendTPDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self._dispatch_output: Optional[AscendTPDispatchOutput] = None

        self.quant_config: Optional[dict] = None

        # Initialise routing kernels with default (no quant config yet)
        self.set_ascend_dispatcher_output_dtype()

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()

        # If the quantisation is GGUF and TP is active, wrap the finalizer
        # with an all‑gather so that the dispatcher stays completely clean.
        if (
            isinstance(self.quant_config, dict)
            and self.quant_config.get("quant_type") == "gguf"
            and get_parallel().tp_size > 1
        ):
            self.finalize = AllGatherFinalizeRoutingWrapper(self.finalize, dim=-1)

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """Choose init & finalize routing kernels based on quant config."""
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            self.group_list_type = (
                2
                if get_bool_env_var("SGLANG_NPU_MOE_USE_GROUP_LIST_TYPE_2")
                else 1
            )
            self.init = NPUMoEInitRouting_v2(
                quant_mode=-1,
                expert_tokens_num_type=self.group_list_type,
            )
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            self.init = NPUMoEInitRouting_v2(quant_mode=1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.MXFP8:
            self.init = NPUMoEInitRouting_v2(quant_mode=MXFP8_QUANT_MODE)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        else:
            raise ValueError(
                f"Unsupported ascend_dispatcher_output_dtype: {self.ascend_dispatcher_output_dtype}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> AscendTPDispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        # BF16 finalize routing supports FP32 scales and accumulates in FP32.
        # Preserve router precision instead of truncating the weights to BF16.
        if not (
            self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16
            and hidden_states.dtype == torch.bfloat16
            and topk_weights.dtype == torch.float32
        ):
            topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        top_k = topk_weights.shape[-1]

        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = self.init._init_routing(
            hidden_states,
            topk_ids,
            self.num_experts,
            top_k,
        )

        self._dispatch_output = AscendTPDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
        )
        return self._dispatch_output

    def combine(self, combine_input: AscendTPCombineInput) -> torch.Tensor:
        if self._dispatch_output is None:
            raise RuntimeError("combine() called before dispatch()")

        dispatch_out = self._dispatch_output

        # The finalizer (possibly wrapped with TP all‑gather) does all the work.
        final_hidden_states = self.finalize._finalize_routing(
            combine_input.hidden_states,
            topk_weights=dispatch_out.topk_weights,
            expanded_row_idx=dispatch_out.expanded_row_idx,
            topk_ids=dispatch_out.topk_ids,
        )

        self._dispatch_output = None
        return final_hidden_states


class AscendLocalEPDispatcher(BaseDispatcher):
    """Route a replicated token batch to this EP rank's local experts only.

    Unlike :class:`AscendTPDispatcher`, this path deliberately performs no
    communication and carries dispatch metadata explicitly in the combine
    input.  Its caller is responsible for summing the rank-local routed
    outputs over the MoE EP group.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        first_expert_idx: int,
        last_expert_idx: int,
    ):
        super().__init__()
        if moe_runner_config.num_experts is None:
            raise ValueError("num_experts is required for local EP routing")
        if moe_runner_config.top_k is None:
            raise ValueError("top_k is required for local EP routing")

        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self.first_expert_idx = first_expert_idx
        self.last_expert_idx = last_expert_idx

        if not 0 <= first_expert_idx < last_expert_idx <= self.num_experts:
            raise ValueError(
                "Local EP expert range must be a non-empty sub-range of "
                f"[0, {self.num_experts}], got "
                f"[{first_expert_idx}, {last_expert_idx})"
            )
        num_local_experts = last_expert_idx - first_expert_idx
        if (
            moe_runner_config.num_local_experts is not None
            and moe_runner_config.num_local_experts != num_local_experts
        ):
            raise ValueError(
                "Local EP expert range does not match the runner's local "
                f"weights: range has {num_local_experts} experts but "
                f"num_local_experts={moe_runner_config.num_local_experts}"
            )

        self.quant_config: Optional[dict] = None
        self.set_ascend_dispatcher_output_dtype()

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """Select the same routing quantization modes as the Ascend TP path."""
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        # active_expert_range with type 1 returns one count per local expert,
        # exactly matching the rank-local GMM weight order. Type 2 carries
        # global expert ids and is therefore intentionally not used here.
        self.group_list_type = 1
        active_expert_range = (self.first_expert_idx, self.last_expert_idx)
        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            self.init = NPUMoEInitRouting_v2(
                quant_mode=-1,
                expert_tokens_num_type=1,
                active_expert_range=active_expert_range,
            )
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            self.init = NPUMoEInitRouting_v2(
                quant_mode=1,
                expert_tokens_num_type=1,
                active_expert_range=active_expert_range,
            )
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.MXFP8:
            self.init = NPUMoEInitRouting_v2(
                quant_mode=MXFP8_QUANT_MODE,
                expert_tokens_num_type=1,
                active_expert_range=active_expert_range,
            )
        else:
            raise ValueError(
                "Unsupported local EP ascend_dispatcher_output_dtype: "
                f"{self.ascend_dispatcher_output_dtype}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> AscendLocalEPDispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if not (
            self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16
            and hidden_states.dtype == torch.bfloat16
            and topk_weights.dtype == torch.float32
        ):
            topk_weights = topk_weights.to(hidden_states.dtype)

        # A direct range comparison is safe for graph padding ids such as -1;
        # indexing a global-to-local map with those ids would not be.
        owned = (topk_ids >= self.first_expert_idx) & (
            topk_ids < self.last_expert_idx
        )
        local_topk_weights = torch.where(
            owned, topk_weights, torch.zeros_like(topk_weights)
        )
        topk_ids = topk_ids.to(torch.int32)

        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = self.init._init_routing(
            hidden_states,
            topk_ids,
            self.num_experts,
            local_topk_weights.shape[-1],
        )

        return AscendLocalEPDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=local_topk_weights,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
        )

    def combine(self, combine_input: AscendLocalEPCombineInput) -> torch.Tensor:
        # Keep the raw -1 entries emitted by row_idx_type=0. They mark routes
        # outside this rank's active expert range for token_unpermute. This is
        # safe only because dispatch() gives every such route an exact zero
        # probability; do not remove that mask or replace this with abs().
        return torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=combine_input.hidden_states,
            sorted_indices=combine_input.expanded_row_idx,
            probs=combine_input.topk_weights,
        )
