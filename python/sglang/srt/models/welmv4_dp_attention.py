"""WeLMv4-specific DP-attention execution contracts.

This module deliberately owns only WeLMv4's model-side semantics.  Generic
``ForwardBatch`` row metadata lives in ``forward_batch_info.py`` and process
group creation remains the responsibility of the runner.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import torch

from sglang.srt.layers.communicator import (
    finish_welm_attn_partial_full,
    reduce_attn_partial_to_scattered,
    welm_attn_all_gather_rows,
)
from sglang.srt.layers.dp_attention import (
    welm_dp_attn_scattered_invalid_mask,
    welm_dp_attn_scattered_valid_mask,
    welm_dp_idle_buffers,
    welm_dp_local_slot_slice,
    welm_dp_prepare_attn_scattered_transport_scratch,
    welm_dp_prepare_full_transport_scratch,
    welm_dp_replicate_gather,
    welm_dp_segmented_invalid_mask,
    welm_dp_segmented_valid_mask,
    welm_reconstruct_attn_scattered_residual_rows,
)
from sglang.srt.layers.welmv4_op import mmq_style_norm_after_attn

if TYPE_CHECKING:
    from sglang.srt.distributed import GroupCoordinator
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        WelmDpMoeRowLayout,
        WelmDpRowView,
    )


class WelmRunnerRole(Enum):
    TARGET_DP = auto()
    DRAFT = auto()


class WelmDpLayout(Enum):
    DP_LOCAL_TP_ATTN_FULL = auto()
    EP_SCATTERED = auto()


class WelmMoeTransport(Enum):
    DP_GATHER_TP_MOE = auto()
    DEEPEP_NORMAL_AG_SCATTERED = auto()
    EXPLICIT_AG_LOCAL_SORT_EP_AR = auto()


class AttentionFinishKind(Enum):
    IDENTITY = auto()
    ATTN_TP_ALL_REDUCE = auto()
    ATTN_TP_REDUCE_SCATTER = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class WelmRunnerParallelPlan:
    """Immutable runner topology/capability snapshot.

    Group objects are supplied by the runner after distributed initialization;
    this class never creates or discovers process groups.
    """

    role: WelmRunnerRole
    attn_tp_group: "GroupCoordinator"
    moe_tp_group: "GroupCoordinator"
    moe_ep_group: Optional["GroupCoordinator"] = None
    moe_ep_normal_group: Optional["GroupCoordinator"] = None
    outer_dp_bridge_group: Optional["GroupCoordinator"] = None
    outer_target_dp_rank: int
    outer_target_dp_size: int
    normal_allgather_enabled: bool = False
    local_ep_kernel_available: bool = False
    shared_expert_ep_replicated: bool = False
    o_proj_returns_partial: bool = True
    first_mirror_consumer: Optional[int] = None
    finalized: bool = False

    @property
    def has_moe_ep(self) -> bool:
        return self.moe_ep_group is not None and self.moe_ep_group.world_size > 1

    @property
    def attn_tp_size(self) -> int:
        return int(self.attn_tp_group.world_size)

    @property
    def attn_tp_rank(self) -> int:
        return int(self.attn_tp_group.rank_in_group)

    @property
    def moe_tp_size(self) -> int:
        return int(self.moe_tp_group.world_size)

    @property
    def moe_ep_size(self) -> int:
        return 1 if self.moe_ep_group is None else int(self.moe_ep_group.world_size)

    @property
    def row_collective_group(self) -> "GroupCoordinator":
        if self.role is WelmRunnerRole.DRAFT and self.has_moe_ep:
            if self.outer_dp_bridge_group is None:
                raise RuntimeError(
                    "WeLMv4 DRAFT+EP requires an explicit outer DP bridge group"
                )
            return self.outer_dp_bridge_group
        if self.has_moe_ep:
            assert self.moe_ep_group is not None
            return self.moe_ep_group
        return self.moe_tp_group


def _validate_group_rank_size(
    *, name: str, group: "GroupCoordinator", rank: int, size: int
) -> None:
    if int(group.world_size) != int(size) or int(group.rank_in_group) != int(rank):
        raise RuntimeError(
            f"WeLMv4 {name} group does not match resolved parallel state: "
            f"group={group.rank_in_group}/{group.world_size}, state={rank}/{size}"
        )


def build_welm_target_runner_plan(
    *,
    parallel_state: "ParallelState",
    attn_tp_group: "GroupCoordinator",
    moe_tp_group: "GroupCoordinator",
    moe_ep_group: Optional["GroupCoordinator"] = None,
    moe_ep_normal_group: Optional["GroupCoordinator"] = None,
    normal_allgather_enabled: bool = False,
    local_ep_kernel_available: bool = False,
    shared_expert_ep_replicated: bool = False,
    o_proj_returns_partial: bool = True,
    first_mirror_consumer: Optional[int] = None,
) -> WelmRunnerParallelPlan:
    """Build the target plan from already-created, resolved group handles."""

    _validate_group_rank_size(
        name="attention-TP",
        group=attn_tp_group,
        rank=parallel_state.attn_tp_rank,
        size=parallel_state.attn_tp_size,
    )
    if moe_ep_group is not None and int(moe_ep_group.world_size) <= 1:
        moe_ep_group = None
    return WelmRunnerParallelPlan(
        role=WelmRunnerRole.TARGET_DP,
        attn_tp_group=attn_tp_group,
        moe_tp_group=moe_tp_group,
        moe_ep_group=moe_ep_group,
        moe_ep_normal_group=moe_ep_normal_group,
        outer_dp_bridge_group=None,
        outer_target_dp_rank=int(parallel_state.attn_dp_rank),
        outer_target_dp_size=int(parallel_state.attn_dp_size),
        normal_allgather_enabled=normal_allgather_enabled,
        local_ep_kernel_available=local_ep_kernel_available,
        shared_expert_ep_replicated=shared_expert_ep_replicated,
        o_proj_returns_partial=o_proj_returns_partial,
        first_mirror_consumer=first_mirror_consumer,
    )


def build_provisional_welm_draft_runner_plan(
    *,
    draft_parallel_state: "ParallelState",
    draft_attn_tp_group: "GroupCoordinator",
    draft_moe_tp_group: "GroupCoordinator",
    draft_moe_ep_group: Optional["GroupCoordinator"] = None,
    outer_dp_bridge_group: Optional["GroupCoordinator"] = None,
    outer_target_dp_rank: int,
    outer_target_dp_size: int,
    local_ep_kernel_available: bool = False,
    shared_expert_ep_replicated: bool = False,
    o_proj_returns_partial: bool = False,
) -> WelmRunnerParallelPlan:
    """Build the pre-model-construction plan for the physical NextN layer."""

    _validate_group_rank_size(
        name="draft attention-TP",
        group=draft_attn_tp_group,
        rank=draft_parallel_state.attn_tp_rank,
        size=draft_parallel_state.attn_tp_size,
    )
    if draft_moe_ep_group is not None and int(draft_moe_ep_group.world_size) <= 1:
        draft_moe_ep_group = None
    if draft_moe_ep_group is not None and outer_dp_bridge_group is None:
        raise RuntimeError(
            "WeLMv4 physical MTP EP plan requires outer_dp_bridge_group"
        )
    return WelmRunnerParallelPlan(
        role=WelmRunnerRole.DRAFT,
        attn_tp_group=draft_attn_tp_group,
        moe_tp_group=draft_moe_tp_group,
        moe_ep_group=draft_moe_ep_group,
        outer_dp_bridge_group=outer_dp_bridge_group,
        outer_target_dp_rank=int(outer_target_dp_rank),
        outer_target_dp_size=int(outer_target_dp_size),
        local_ep_kernel_available=local_ep_kernel_available,
        shared_expert_ep_replicated=shared_expert_ep_replicated,
        o_proj_returns_partial=o_proj_returns_partial,
    )


def finalize_welm_runner_plan(
    plan: WelmRunnerParallelPlan,
    *,
    local_ep_kernel_available: Optional[bool] = None,
    shared_expert_ep_replicated: Optional[bool] = None,
    o_proj_returns_partial: Optional[bool] = None,
) -> WelmRunnerParallelPlan:
    """Finalize post-construction capabilities without mutating the plan."""

    finalized = replace(
        plan,
        local_ep_kernel_available=(
            plan.local_ep_kernel_available
            if local_ep_kernel_available is None
            else bool(local_ep_kernel_available)
        ),
        shared_expert_ep_replicated=(
            plan.shared_expert_ep_replicated
            if shared_expert_ep_replicated is None
            else bool(shared_expert_ep_replicated)
        ),
        o_proj_returns_partial=(
            plan.o_proj_returns_partial
            if o_proj_returns_partial is None
            else bool(o_proj_returns_partial)
        ),
        finalized=True,
    )
    validate_welm_runner_plan(finalized, require_finalized=True)
    return finalized


def validate_welm_runner_plan(
    plan: WelmRunnerParallelPlan, *, require_finalized: bool = False
) -> None:
    """Validate only capabilities whose absence would silently corrupt output."""

    if plan.outer_target_dp_size < 1:
        raise RuntimeError("WeLMv4 outer target DP size must be positive")
    if not 0 <= plan.outer_target_dp_rank < plan.outer_target_dp_size:
        raise RuntimeError(
            "WeLMv4 outer target DP rank is outside its collective domain"
        )
    if require_finalized and not plan.finalized:
        raise RuntimeError("WeLMv4 runner plan has not been finalized")
    uses_dp_row_executor = (
        plan.role is WelmRunnerRole.TARGET_DP
        or (plan.role is WelmRunnerRole.DRAFT and plan.has_moe_ep)
    )
    if require_finalized and uses_dp_row_executor:
        row_group = plan.row_collective_group
        expected_row_group_size = (
            plan.outer_target_dp_size * plan.attn_tp_size
        )
        if int(row_group.world_size) != expected_row_group_size:
            raise RuntimeError(
                "WeLMv4 row collective does not cover the complete outer-DP "
                "x attention-TP domain: "
                f"group={row_group.world_size}, expected={expected_row_group_size}"
            )
        expected_row_group_rank = (
            plan.outer_target_dp_rank * plan.attn_tp_size + plan.attn_tp_rank
        )
        if int(row_group.rank_in_group) != expected_row_group_rank:
            raise RuntimeError(
                "WeLMv4 row collective rank order is incompatible with the "
                "segmented DP-slot owner mapping: "
                f"group_rank={row_group.rank_in_group}, "
                f"expected={expected_row_group_rank}"
            )
    if plan.has_moe_ep and plan.role is WelmRunnerRole.TARGET_DP:
        if plan.moe_tp_size != 1:
            raise RuntimeError(
                "WeLMv4 explicit local-EP transport requires resolved MoE-TP size 1"
            )
        if not plan.normal_allgather_enabled:
            raise RuntimeError(
                "WeLMv4 target DP+EP ordinary prefill requires DeepEP "
                "NORMAL AllGather"
            )
        if plan.moe_ep_normal_group is None:
            raise RuntimeError(
                "WeLMv4 target DP+EP requires the resolved NORMAL comm group"
            )
        normal_ranks = getattr(plan.moe_ep_normal_group, "ranks", None)
        ep_ranks = getattr(plan.moe_ep_group, "ranks", None)
        if (
            normal_ranks is not None
            and ep_ranks is not None
            and set(normal_ranks) != set(ep_ranks)
        ):
            raise RuntimeError(
                "WeLMv4 DeepEP NORMAL group does not cover the resolved MoE-EP group"
            )
        if plan.attn_tp_size > 1 and plan.finalized and not plan.o_proj_returns_partial:
            raise RuntimeError(
                "WeLMv4 target DP+EP requires OProj partial rows for the "
                "attention-TP ReduceScatter contract"
            )
    if plan.has_moe_ep and plan.role is WelmRunnerRole.DRAFT:
        if plan.moe_tp_size != 1:
            raise RuntimeError(
                "WeLMv4 physical MTP local-EP transport requires resolved "
                "MoE-TP size 1"
            )
        if plan.outer_dp_bridge_group is None:
            raise RuntimeError("WeLMv4 DRAFT+EP requires an outer DP bridge group")
    if plan.has_moe_ep and plan.finalized:
        if not plan.local_ep_kernel_available:
            raise RuntimeError("WeLMv4 DP+EP requires the Ascend local-EP kernel")
        if not plan.shared_expert_ep_replicated:
            raise RuntimeError(
                "WeLMv4 DP+EP requires a complete shared-expert replica per EP rank"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class WelmLayerCapabilities:
    """Post-load facts sampled from one structurally representative layer."""

    local_ep_kernel_available: bool
    shared_expert_ep_replicated: bool
    o_proj_returns_partial: bool


def inspect_welm_layer_capabilities(
    layer: Any, plan: WelmRunnerParallelPlan
) -> WelmLayerCapabilities:
    """Validate the weight/collective contract once after model loading.

    WeLM decoder layers are constructed from one uniform topology.  Release
    startup therefore checks one representative target layer (or the sole
    physical NextN layer) instead of repeating the same identity and coverage
    checks over every layer.  Model binding still visits every layer, but only
    to publish the finalized immutable plan.
    """

    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    self_attn = getattr(layer, "self_attn", None)
    o_proj = getattr(self_attn, "o_proj", None)
    if mlp is None or experts is None or o_proj is None:
        raise RuntimeError(
            "WeLMv4 runner plan requires a representative sparse decoder layer"
        )

    actual_expert_topology = (
        int(getattr(experts, "moe_ep_size", -1)),
        int(getattr(experts, "moe_ep_rank", -1)),
        int(getattr(experts, "moe_tp_size", -1)),
        int(getattr(experts, "moe_tp_rank", -1)),
    )
    expected_expert_topology = (
        plan.moe_ep_size,
        0 if plan.moe_ep_group is None else int(plan.moe_ep_group.rank_in_group),
        plan.moe_tp_size,
        int(plan.moe_tp_group.rank_in_group),
    )
    if actual_expert_topology != expected_expert_topology:
        raise RuntimeError(
            "WeLMv4 routed-expert weight coverage does not match the resolved "
            "MoE groups: "
            f"actual={actual_expert_topology}, expected={expected_expert_topology}"
        )
    if bool(getattr(experts, "reduce_results", False)):
        raise RuntimeError(
            "WeLMv4 DP execution requires routed experts to return partial "
            "outputs before the single transport-owned combine"
        )
    if not plan.has_moe_ep and int(getattr(mlp, "tp_size", -1)) != plan.moe_tp_size:
        raise RuntimeError(
            "WeLMv4 pure-TP MoE block and resolved MoE-TP group have "
            f"different sizes: block={getattr(mlp, 'tp_size', None)}, "
            f"group={plan.moe_tp_size}"
        )

    shared = getattr(mlp, "shared_expert", None)
    shared_expert_ep_replicated = True
    if shared is not None:
        expected_shared_tp_size = 1 if plan.has_moe_ep else plan.moe_tp_size
        shared_tp_sizes = (
            int(getattr(shared.gate_up_proj, "tp_size", -1)),
            int(getattr(shared.down_proj, "tp_size", -1)),
        )
        expected_shared_tp_sizes = (
            expected_shared_tp_size,
            expected_shared_tp_size,
        )
        if shared_tp_sizes != expected_shared_tp_sizes:
            raise RuntimeError(
                "WeLMv4 shared-expert weight coverage does not match the "
                f"resolved transport: actual={shared_tp_sizes}, "
                f"expected={expected_shared_tp_sizes}"
            )
        if bool(getattr(shared.down_proj, "reduce_results", False)):
            raise RuntimeError(
                "WeLMv4 DP execution requires shared down-projection to "
                "return a partial output"
            )
        shared_expert_ep_replicated = shared_tp_sizes == (1, 1)

    local_ep_kernel_available = bool(
        getattr(mlp, "welm_local_ep_kernel_available", False)
    )
    if plan.has_moe_ep and getattr(experts, "local_ep_dispatcher", None) is None:
        raise RuntimeError(
            "WeLMv4 DP+EP requires the initialized Ascend local-EP dispatcher"
        )

    return WelmLayerCapabilities(
        local_ep_kernel_available=local_ep_kernel_available,
        shared_expert_ep_replicated=shared_expert_ep_replicated,
        o_proj_returns_partial=(
            int(getattr(o_proj, "tp_size", 1)) > 1
            and not bool(getattr(o_proj, "reduce_results", True))
        ),
    )


@dataclass(frozen=True, slots=True)
class WelmBatchExecutionPlan:
    attention_finish: AttentionFinishKind
    moe_transport: WelmMoeTransport


@dataclass(slots=True)
class WelmDpLayerState:
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]
    hidden_layout: WelmDpLayout
    residual_layout: Optional[WelmDpLayout]
    row_layout: "WelmDpMoeRowLayout"


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftSharedModulePlan:
    embed_tokens: Any
    oe_embed: Any
    oe_gate_up_proj: Any
    lm_head: Any

    @classmethod
    def build(
        cls,
        embed: Sequence[Any],
        head: Any,
        *,
        vocab_group: "GroupCoordinator",
        logits_processor: Optional[Any] = None,
    ) -> "DraftSharedModulePlan":
        if len(embed) != 3:
            raise RuntimeError(
                "WeLMv4 NextN sharing requires base embedding, OE embedding, "
                "and replicated OE gate/up projection"
            )
        embed_tokens, oe_embed, oe_gate_up_proj = embed
        group_size = int(vocab_group.world_size)
        group_rank = int(vocab_group.rank_in_group)

        def validate_vocab_shard(name: str, module: Any) -> Any:
            if not bool(getattr(module, "use_attn_tp_group", False)):
                raise RuntimeError(
                    f"WeLMv4 shared {name} is not sharded on the attention-TP "
                    "vocab group"
                )
            if int(getattr(module, "tp_size", -1)) != group_size:
                raise RuntimeError(
                    f"WeLMv4 shared {name} TP size does not match the vocab "
                    f"group: module={getattr(module, 'tp_size', None)}, "
                    f"group={group_size}"
                )
            shard_indices = getattr(module, "shard_indices", None)
            get_indices = getattr(type(module), "_get_indices", None)
            required_fields = (
                "num_embeddings_padded",
                "org_vocab_size_padded",
                "num_embeddings",
                "org_vocab_size",
            )
            if shard_indices is None or get_indices is None or any(
                not hasattr(module, field) for field in required_fields
            ):
                raise RuntimeError(
                    f"WeLMv4 shared {name} does not expose a verifiable vocab "
                    "shard contract"
                )
            expected = get_indices(
                int(module.num_embeddings_padded),
                int(module.org_vocab_size_padded),
                int(module.num_embeddings),
                int(module.org_vocab_size),
                group_rank,
                group_size,
            )
            if shard_indices != expected:
                raise RuntimeError(
                    f"WeLMv4 shared {name} shard indices do not match vocab "
                    f"group rank {group_rank}/{group_size}"
                )
            return shard_indices

        embed_indices = validate_vocab_shard("embed_tokens", embed_tokens)
        head_indices = validate_vocab_shard("lm_head", head)
        if (
            embed_indices != head_indices
            or int(embed_tokens.num_embeddings) != int(head.num_embeddings)
            or int(embed_tokens.embedding_dim) != int(head.embedding_dim)
        ):
            raise RuntimeError(
                "WeLMv4 shared embedding and LM-head use different vocab shards"
            )

        try:
            oe_modules = tuple(oe_embed)
        except TypeError as exc:
            raise RuntimeError(
                "WeLMv4 OE embeddings must be an iterable module container"
            ) from exc
        if not oe_modules:
            raise RuntimeError("WeLMv4 shared OE embedding list is empty")
        for index, module in enumerate(oe_modules):
            validate_vocab_shard(f"oe_embed[{index}]", module)

        if int(getattr(oe_gate_up_proj, "tp_size", 1)) != 1 or bool(
            getattr(oe_gate_up_proj, "use_attn_tp_group", False)
        ):
            raise RuntimeError(
                "WeLMv4 shared OE gate/up projection must be a complete replica"
            )

        if logits_processor is not None:
            if not bool(getattr(logits_processor, "use_attn_tp_group", False)):
                raise RuntimeError(
                    "WeLMv4 draft LogitsProcessor is not configured for the "
                    "attention-TP vocab group"
                )
            if int(getattr(logits_processor, "attn_tp_size", -1)) != group_size:
                raise RuntimeError(
                    "WeLMv4 draft LogitsProcessor attention-TP size does not "
                    "match the shared vocab group"
                )
        return cls(
            embed_tokens=embed_tokens,
            oe_embed=oe_embed,
            oe_gate_up_proj=oe_gate_up_proj,
            lm_head=head,
        )

    @property
    def embed_tuple(self) -> tuple[Any, Any, Any]:
        return self.embed_tokens, self.oe_embed, self.oe_gate_up_proj


_WELM_RUNNER_BUILD_PLAN: ContextVar[Optional[WelmRunnerParallelPlan]] = ContextVar(
    "welm_runner_build_plan", default=None
)


@contextmanager
def welm_runner_build_plan_context(
    plan: Optional[WelmRunnerParallelPlan],
) -> Iterator[None]:
    token = _WELM_RUNNER_BUILD_PLAN.set(plan)
    try:
        yield
    finally:
        _WELM_RUNNER_BUILD_PLAN.reset(token)


def get_welm_runner_build_plan_for_init() -> Optional[WelmRunnerParallelPlan]:
    return _WELM_RUNNER_BUILD_PLAN.get()


class WelmDpExecutionPlanner:
    def __init__(self, runner_plan: WelmRunnerParallelPlan):
        self.runner_plan = runner_plan

    def resolve(
        self,
        *,
        has_ordinary_prefill: bool,
        layer_id: int,
        enable_kv_mirror: bool,
    ) -> WelmBatchExecutionPlan:
        plan = self.runner_plan
        if plan.role is WelmRunnerRole.DRAFT:
            if not plan.has_moe_ep:
                raise RuntimeError("A no-EP WeLMv4 draft must use _forward_non_dp()")
            transport = WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR
            finish = (
                AttentionFinishKind.ATTN_TP_ALL_REDUCE
                if plan.o_proj_returns_partial and plan.attn_tp_size > 1
                else AttentionFinishKind.IDENTITY
            )
            return WelmBatchExecutionPlan(finish, transport)

        if not plan.has_moe_ep:
            transport = WelmMoeTransport.DP_GATHER_TP_MOE
            finish = (
                AttentionFinishKind.ATTN_TP_ALL_REDUCE
                if plan.o_proj_returns_partial and plan.attn_tp_size > 1
                else AttentionFinishKind.IDENTITY
            )
            return WelmBatchExecutionPlan(finish, transport)

        in_scattered_prefill_prefix = (
            has_ordinary_prefill
            and (
                not enable_kv_mirror
                or plan.first_mirror_consumer is None
                or layer_id < plan.first_mirror_consumer
            )
        )
        if in_scattered_prefill_prefix:
            finish = (
                AttentionFinishKind.ATTN_TP_REDUCE_SCATTER
                if plan.o_proj_returns_partial and plan.attn_tp_size > 1
                else AttentionFinishKind.IDENTITY
            )
            return WelmBatchExecutionPlan(
                finish, WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED
            )
        finish = (
            AttentionFinishKind.ATTN_TP_ALL_REDUCE
            if plan.o_proj_returns_partial and plan.attn_tp_size > 1
            else AttentionFinishKind.IDENTITY
        )
        return WelmBatchExecutionPlan(
            finish, WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR
        )


class WelmDpAttentionExecutor:
    """WeLMv4 DP-attention layer executor with explicit tensor layouts."""

    def __init__(self, runner_plan: WelmRunnerParallelPlan):
        validate_welm_runner_plan(runner_plan)
        self.runner_plan = runner_plan
        self.planner = WelmDpExecutionPlanner(runner_plan)

    def bind_finalized_runner_plan(self, runner_plan: WelmRunnerParallelPlan) -> None:
        """Atomically replace the constructor-time provisional plan."""

        validate_welm_runner_plan(runner_plan, require_finalized=True)
        if runner_plan.role is not self.runner_plan.role:
            raise RuntimeError("WeLMv4 runner role changed while finalizing the plan")
        self.runner_plan = runner_plan
        self.planner = WelmDpExecutionPlanner(runner_plan)

    def prepare_forward_scratch(
        self,
        *,
        forward_batch: "ForwardBatch",
        hidden_states: torch.Tensor,
    ) -> None:
        """Allocate once and refresh dynamic masks before the layer loop."""

        plan = self.runner_plan
        if not plan.finalized:
            raise RuntimeError(
                "WeLMv4 DP runner plan must be finalized before scratch setup"
            )
        row_layout = getattr(forward_batch, "welm_dp_moe_row_layout", None)
        if row_layout is None:
            raise RuntimeError(
                "WeLMv4 DP execution requires ForwardBatch.welm_dp_moe_row_layout"
            )
        if hidden_states.dim() < 2:
            raise RuntimeError("WeLMv4 DP scratch requires row-major hidden states")

        has_ordinary_prefill = bool(
            forward_batch.is_extend_in_batch
            and not forward_batch.forward_mode.is_target_verify()
        )
        enable_kv_mirror = bool(
            forward_batch.enable_kv_mirror
            and plan.first_mirror_consumer is not None
        )
        requirements = []
        if plan.role is WelmRunnerRole.DRAFT:
            active_view = (
                row_layout.request
                if forward_batch.is_extend_in_batch
                and row_layout.request is not None
                else row_layout.current
            )
            requirements.append(
                (active_view, WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR)
            )
        elif has_ordinary_prefill:
            requirements.append(
                (
                    row_layout.current,
                    (
                        WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED
                        if plan.has_moe_ep
                        else WelmMoeTransport.DP_GATHER_TP_MOE
                    ),
                )
            )
            if enable_kv_mirror:
                if row_layout.request is None:
                    raise RuntimeError(
                        "WeLMv4 mirror prefill requires request-row metadata"
                    )
                requirements.append(
                    (
                        row_layout.request,
                        WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR,
                    )
                )
        else:
            requirements.append(
                (
                    row_layout.current,
                    (
                        WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR
                        if plan.has_moe_ep
                        else WelmMoeTransport.DP_GATHER_TP_MOE
                    ),
                )
            )

        prepared = set()
        for row_view, transport in requirements:
            key = (id(row_view), transport)
            if key in prepared:
                continue
            prepared.add(key)
            if transport is WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED:
                welm_dp_prepare_attn_scattered_transport_scratch(
                    row_view,
                    hidden_states,
                    dp_rank=plan.outer_target_dp_rank,
                    attn_tp_rank=plan.attn_tp_rank,
                    attn_tp_size=plan.attn_tp_size,
                )
            else:
                welm_dp_prepare_full_transport_scratch(row_view, hidden_states)

    def forward(
        self,
        *,
        layer: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        plan = self.runner_plan
        if not plan.finalized:
            raise RuntimeError(
                "WeLMv4 DP runner plan must be finalized before forward/Graph capture"
            )
        row_layout = getattr(forward_batch, "welm_dp_moe_row_layout", None)
        if row_layout is None:
            raise RuntimeError(
                "WeLMv4 DP execution requires ForwardBatch.welm_dp_moe_row_layout"
            )

        has_ordinary_prefill = bool(
            forward_batch.is_extend_in_batch
            and not forward_batch.forward_mode.is_target_verify()
        )
        enable_kv_mirror = bool(
            forward_batch.enable_kv_mirror
            and plan.first_mirror_consumer is not None
        )
        batch_plan = self.planner.resolve(
            has_ordinary_prefill=has_ordinary_prefill,
            layer_id=layer.layer_id,
            enable_kv_mirror=enable_kv_mirror,
        )
        active_view = self._active_row_view(
            layer=layer,
            forward_batch=forward_batch,
            row_layout=row_layout,
            has_ordinary_prefill=has_ordinary_prefill,
            enable_kv_mirror=enable_kv_mirror,
        )
        input_is_scattered = self._input_is_scattered(
            layer_id=layer.layer_id,
            has_ordinary_prefill=has_ordinary_prefill,
            enable_kv_mirror=enable_kv_mirror,
        )

        mirror_transition = self._is_first_target_mirror_transition(
            layer_id=layer.layer_id,
            has_ordinary_prefill=has_ordinary_prefill,
            enable_kv_mirror=enable_kv_mirror,
        )
        later_mirror_prefill = bool(
            plan.role is WelmRunnerRole.TARGET_DP
            and has_ordinary_prefill
            and enable_kv_mirror
            and layer.layer_id > plan.first_mirror_consumer
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
        mirror_row_indices = None
        if active_view.local_real_rows > 0:
            if later_mirror_prefill:
                # Trim before input norm: after the first mirror transition the
                # MoE hidden output occupies the fixed request slot, while the
                # FP32 residual deliberately stays at real B rows.
                pre_norm_state = WelmDpLayerState(
                    hidden_states=hidden_states,
                    residual=residual,
                    hidden_layout=WelmDpLayout.DP_LOCAL_TP_ATTN_FULL,
                    residual_layout=WelmDpLayout.DP_LOCAL_TP_ATTN_FULL,
                    row_layout=row_layout,
                )
                pre_norm_state = self._trim_mirror_prefill_rows(
                    pre_norm_state, active_view
                )
                hidden_states = pre_norm_state.hidden_states
                residual = pre_norm_state.residual
            hidden_states, residual = self._prepare_attention_input(
                layer=layer,
                hidden_states=hidden_states,
                residual=residual,
                input_is_scattered=input_is_scattered,
            )
            state = WelmDpLayerState(
                hidden_states=hidden_states,
                residual=residual,
                hidden_layout=WelmDpLayout.DP_LOCAL_TP_ATTN_FULL,
                residual_layout=(
                    WelmDpLayout.EP_SCATTERED
                    if input_is_scattered
                    else WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
                ),
                row_layout=row_layout,
            )
            if mirror_transition:
                state, mirror_row_indices = self._prepare_first_mirror_query(
                    state=state,
                    forward_batch=forward_batch,
                    input_is_scattered=input_is_scattered,
                )

            attn_output = layer.self_attn(
                positions=positions,
                hidden_states=state.hidden_states,
                forward_batch=forward_batch,
                skip_o_norm=True,
                skip_o_proj_all_reduce=plan.o_proj_returns_partial,
            )
            state = self._finish_attention_and_norm(
                layer=layer,
                state=state,
                attn_output=attn_output,
                forward_batch=forward_batch,
                batch_plan=batch_plan,
                active_view=active_view,
                mirror_transition=mirror_transition,
                mirror_row_indices=mirror_row_indices,
            )
        else:
            # Eager idle only.  Skip norm/attention entirely and reuse the
            # batch-owned typed scratch prepared before the layer loop. Graph
            # capture uses a positive capture width and never records this
            # Python branch.
            idle_hidden, idle_residual = welm_dp_idle_buffers(active_view)
            idle_layout = (
                WelmDpLayout.EP_SCATTERED
                if batch_plan.moe_transport
                is WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED
                else WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
            )
            state = WelmDpLayerState(
                hidden_states=idle_hidden,
                residual=idle_residual,
                hidden_layout=idle_layout,
                residual_layout=idle_layout,
                row_layout=row_layout,
            )

        state = self._run_moe_transport(
            layer=layer,
            state=state,
            forward_batch=forward_batch,
            batch_plan=batch_plan,
            active_view=active_view,
        )
        if (
            layer.is_final_layer
            and forward_batch.forward_mode.is_extend_without_speculative()
            and active_view is row_layout.request
        ):
            # Mirror attention already consumes one real row per request.  The
            # effective EXTEND path covers both true prefill and a mixed-round
            # decode shard converted to the common MAX slot geometry; remove
            # the request-slot suffix before either publishes hidden/logit rows.
            # An unconverted DECODE shard does not enter this branch and keeps
            # the existing generic decode trimming behavior.
            state = self._trim_mirror_prefill_rows(state, active_view)
        self._assert_fp32_residual(state.residual)
        return state.hidden_states, state.residual

    def _active_row_view(
        self,
        *,
        layer: Any,
        forward_batch: "ForwardBatch",
        row_layout: "WelmDpMoeRowLayout",
        has_ordinary_prefill: bool,
        enable_kv_mirror: bool,
    ) -> "WelmDpRowView":
        if self.runner_plan.role is WelmRunnerRole.DRAFT:
            # NextN prompt seeding has already contracted token/OE embeddings
            # with custom_last_index before entering physical layer48.
            # Select the request domain from the group-wide round predicate:
            # a locally idle rank must enter the same bridge geometry as an
            # active prefill rank.  Draft decode/extend graphs have no request
            # view and therefore stay on current.
            if forward_batch.is_extend_in_batch and row_layout.request is not None:
                return row_layout.request
            return row_layout.current
        if not enable_kv_mirror:
            return row_layout.current
        return row_layout.for_layer(
            layer.layer_id,
            first_mirror_consumer=self.runner_plan.first_mirror_consumer,
            is_extend_in_batch=has_ordinary_prefill,
        )

    def _input_is_scattered(
        self,
        *,
        layer_id: int,
        has_ordinary_prefill: bool,
        enable_kv_mirror: bool,
    ) -> bool:
        plan = self.runner_plan
        if (
            plan.role is not WelmRunnerRole.TARGET_DP
            or not plan.has_moe_ep
            or not has_ordinary_prefill
            or layer_id <= 0
        ):
            return False
        return (
            not enable_kv_mirror
            or plan.first_mirror_consumer is None
            or layer_id <= plan.first_mirror_consumer
        )

    def _is_first_target_mirror_transition(
        self,
        *,
        layer_id: int,
        has_ordinary_prefill: bool,
        enable_kv_mirror: bool,
    ) -> bool:
        return bool(
            self.runner_plan.role is WelmRunnerRole.TARGET_DP
            and has_ordinary_prefill
            and enable_kv_mirror
            and layer_id == self.runner_plan.first_mirror_consumer
        )

    @staticmethod
    def _assert_fp32_residual(residual: Optional[torch.Tensor]) -> None:
        if residual is not None and residual.dtype != torch.float32:
            raise RuntimeError(
                f"WeLMv4 DP residual must remain FP32, got {residual.dtype}"
            )

    def _normalize_input(
        self,
        *,
        layer: Any,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            if residual is not None and residual.dtype != torch.float32:
                raise RuntimeError(
                    f"WeLMv4 DP residual must remain FP32, got {residual.dtype}"
                )
            return hidden_states, (
                residual
                if residual is not None
                else hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
            )

        residual_after_layernorm = (
            layer.ppln and layer.layer_id not in layer.prenorm_layer_idx
        )
        if residual_after_layernorm:
            hidden_states, _, residual = layer.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=True,
                clone_fp32_out=True,
                output_dtype=(
                    layer.input_layernorm.weight.dtype
                    if hidden_states.dtype == torch.float32
                    else hidden_states.dtype
                ),
            )
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=False,
            )
            if residual.dtype != torch.float32:
                residual = residual.to(torch.float32)
        self._assert_fp32_residual(residual)
        return hidden_states, residual

    def _prepare_attention_input(
        self,
        *,
        layer: Any,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_is_scattered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._normalize_input(
            layer=layer, hidden_states=hidden_states, residual=residual
        )
        if input_is_scattered:
            hidden_states = welm_attn_all_gather_rows(
                hidden_states, attn_tp_group=self.runner_plan.attn_tp_group
            )
        return hidden_states, residual

    def _prepare_first_mirror_query(
        self,
        *,
        state: WelmDpLayerState,
        forward_batch: "ForwardBatch",
        input_is_scattered: bool,
    ) -> tuple[WelmDpLayerState, Optional[torch.Tensor]]:
        prompt_view = state.row_layout.current
        request_view = state.row_layout.request
        if request_view is None:
            raise RuntimeError(
                "WeLMv4 first mirror consumer requires the request row view"
            )
        contracts = (
            int(prompt_view.local_real_rows) != int(request_view.local_real_rows)
            or int(prompt_view.local_slot_rows)
            != int(request_view.local_slot_rows)
        )
        # MAX_LEN can temporarily represent a decode shard as EXTEND so it can
        # share the ordinary-prefill collective/attention geometry with its DP
        # peers.  Use the original local phase only to choose the T->B mapping;
        # keep the effective EXTEND mode for the physical mirror-attention path.
        local_forward_mode = (
            getattr(forward_batch, "_original_forward_mode", None)
            or forward_batch.forward_mode
        )
        is_local_prefill = local_forward_mode.is_extend_without_speculative()
        uses_mirror_prefill_attention = (
            forward_batch.forward_mode.is_extend_without_speculative()
        )
        if not contracts and not uses_mirror_prefill_attention:
            return state, None
        if is_local_prefill:
            row_indices = getattr(forward_batch, "custom_last_index", None)
            if row_indices is None:
                # Match the established non-DP WeLM path: ordinary prefill
                # does not publish this model-private T->B map when building
                # ForwardBatch, so derive it lazily at the first mirror
                # consumer.  extend_seq_lens contains only real request
                # lengths; any DP/Graph slot padding is a suffix and must not
                # participate in the cumulative tail positions.
                row_indices = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                forward_batch.custom_last_index = row_indices
        else:
            # In a non-Spec mixed round, another DP shard may contain
            # ordinary prefill while this shard contains decode.  Transport
            # still changes group-wide at the first mirror layer, but this
            # shard is already in B-row order; only its attn-TP alignment
            # suffix must be removed.
            row_indices = torch.arange(
                int(request_view.local_real_rows),
                device=state.hidden_states.device,
                dtype=torch.long,
            )
            if uses_mirror_prefill_attention:
                # The attention/RoPE mirror path is selected by the effective
                # EXTEND mode and therefore still needs a Q->source-K position
                # map.  For a logical decode shard its real rows are already in
                # request order, so the exact map is the identity [0..B).
                forward_batch.custom_last_index = row_indices
        if row_indices.numel() != int(request_view.local_real_rows):
            raise RuntimeError(
                "WeLMv4 request row count does not match mirror selection"
            )
        state.hidden_states = state.hidden_states.index_select(
            0, row_indices.to(torch.long)
        )
        if not input_is_scattered:
            assert state.residual is not None
            state.residual = state.residual.index_select(
                0, row_indices.to(torch.long)
            )
        return state, row_indices

    @staticmethod
    def _trim_mirror_prefill_rows(
        state: WelmDpLayerState,
        request_view: "WelmDpRowView",
    ) -> WelmDpLayerState:
        """Expose only real request rows to a prefill mirror attention call."""

        real_rows = int(request_view.local_real_rows)
        if state.hidden_states.shape[0] < real_rows:
            raise RuntimeError(
                "WeLMv4 mirror prefill hidden rows are smaller than the real "
                f"request count: {state.hidden_states.shape[0]} < {real_rows}"
            )
        state.hidden_states = state.hidden_states.narrow(0, 0, real_rows)
        if state.residual is not None:
            if state.residual.shape[0] < real_rows:
                raise RuntimeError(
                    "WeLMv4 mirror prefill residual rows are smaller than the "
                    f"real request count: {state.residual.shape[0]} < {real_rows}"
                )
            state.residual = state.residual.narrow(0, 0, real_rows)
        return state

    def _finish_attention_and_norm(
        self,
        *,
        layer: Any,
        state: WelmDpLayerState,
        attn_output: torch.Tensor,
        forward_batch: "ForwardBatch",
        batch_plan: WelmBatchExecutionPlan,
        active_view: "WelmDpRowView",
        mirror_transition: bool,
        mirror_row_indices: Optional[torch.Tensor],
    ) -> WelmDpLayerState:
        plan = self.runner_plan
        if (
            batch_plan.moe_transport
            is WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED
        ):
            if (
                batch_plan.attention_finish
                is AttentionFinishKind.ATTN_TP_REDUCE_SCATTER
            ):
                attn_output, residual = reduce_attn_partial_to_scattered(
                    attn_output,
                    state.residual,
                    attn_tp_group=plan.attn_tp_group,
                )
            elif plan.attn_tp_size == 1:
                residual = state.residual
            else:
                raise RuntimeError(
                    "WeLMv4 NORMAL transport requires ReduceScatter when "
                    "attention-TP is greater than one"
                )
            state.hidden_layout = WelmDpLayout.EP_SCATTERED
            state.residual_layout = WelmDpLayout.EP_SCATTERED
        else:
            if batch_plan.attention_finish is AttentionFinishKind.ATTN_TP_ALL_REDUCE:
                attn_output = finish_welm_attn_partial_full(
                    attn_output, attn_tp_group=plan.attn_tp_group
                )
            residual = state.residual
            if mirror_transition and state.residual_layout is WelmDpLayout.EP_SCATTERED:
                assert residual is not None
                if mirror_row_indices is not None:
                    residual = welm_reconstruct_attn_scattered_residual_rows(
                        residual,
                        mirror_row_indices,
                        state.row_layout.current,
                        state.row_layout.request,
                        group=plan.attn_tp_group,
                        dp_rank=plan.outer_target_dp_rank,
                        attn_tp_rank=plan.attn_tp_rank,
                        attn_tp_size=plan.attn_tp_size,
                    )
                else:
                    residual = welm_attn_all_gather_rows(
                        residual, attn_tp_group=plan.attn_tp_group
                    )
            assert residual is not None
            if (
                attn_output.shape[0] != residual.shape[0]
                or attn_output.shape[0] > int(active_view.local_slot_rows)
            ):
                raise RuntimeError(
                    "WeLMv4 FULL attention/residual rows must match and fit "
                    f"the DP slot: attention={attn_output.shape[0]}, "
                    f"residual={residual.shape[0]}, "
                    f"slot={active_view.local_slot_rows}"
                )
            state.hidden_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
            state.residual_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL

        assert residual is not None
        self._assert_fp32_residual(residual)
        if layer.self_attn.use_o_norm:
            attn_output, residual, _ = mmq_style_norm_after_attn(
                attn_output,
                residual,
                layer.self_attn.o_norm.weight,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
                return_fp32_out=False,
            )
        else:
            attn_output, residual, _ = layer.post_attention_layernorm(
                attn_output, residual, clone_fp32_out=True
            )
        self._assert_fp32_residual(residual)
        state.hidden_states = attn_output
        state.residual = residual
        return state

    @staticmethod
    def _store_final_components(layer: Any, mlp_output: Any) -> torch.Tensor:
        layer.final_mlp_experts_output = None
        layer.final_mlp_shared_output = None
        if not isinstance(mlp_output, tuple):
            return mlp_output
        hidden_states, experts_output, shared_output = mlp_output
        layer.final_mlp_experts_output = experts_output
        layer.final_mlp_shared_output = shared_output
        return hidden_states

    def _run_moe_transport(
        self,
        *,
        layer: Any,
        state: WelmDpLayerState,
        forward_batch: "ForwardBatch",
        batch_plan: WelmBatchExecutionPlan,
        active_view: "WelmDpRowView",
    ) -> WelmDpLayerState:
        plan = self.runner_plan
        transport = batch_plan.moe_transport
        layer.final_mlp_experts_output = None
        layer.final_mlp_shared_output = None

        if transport is WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED:
            if state.hidden_layout is not WelmDpLayout.EP_SCATTERED:
                raise RuntimeError(
                    "WeLMv4 DeepEP NORMAL received a non-scattered hidden layout"
                )
            valid_mask = welm_dp_attn_scattered_valid_mask(
                active_view,
                dp_rank=plan.outer_target_dp_rank,
                attn_tp_rank=plan.attn_tp_rank,
                attn_tp_size=plan.attn_tp_size,
                local_rows=state.hidden_states.shape[0],
                device=state.hidden_states.device,
            )
            invalid_mask = welm_dp_attn_scattered_invalid_mask(active_view)
            state.hidden_states.masked_fill_(invalid_mask[:, None], 0)
            assert state.residual is not None
            state.residual.masked_fill_(invalid_mask[:, None], 0)
            mlp_output = layer.mlp(
                state.hidden_states,
                None,
                forward_batch,
                False,
                return_components=False,
                use_welm_prefill_normal_stream_policy=True,
                valid_row_mask=valid_mask,
                invalid_row_mask=invalid_mask,
                invalid_topk_id=-1,
                allow_inplace_expert_shared_merge=True,
            )
            state.hidden_states = self._store_final_components(layer, mlp_output)
            state.hidden_states.masked_fill_(invalid_mask[:, None], 0)
            if layer.is_final_layer:
                # A mirror-disabled ordinary prefill ends while still in the
                # NORMAL scattered layout. Restore this DP shard before model
                # output/norm publication.
                state.hidden_states = welm_attn_all_gather_rows(
                    state.hidden_states, attn_tp_group=plan.attn_tp_group
                )
                assert state.residual is not None
                state.residual = welm_attn_all_gather_rows(
                    state.residual, attn_tp_group=plan.attn_tp_group
                )
                state.hidden_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
                state.residual_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
            return state

        if state.hidden_layout is not WelmDpLayout.DP_LOCAL_TP_ATTN_FULL:
            raise RuntimeError(
                "WeLMv4 FULL MoE transport received a non-local hidden layout"
            )
        local_output_rows = int(state.hidden_states.shape[0])
        full_hidden = welm_dp_replicate_gather(
            state.hidden_states,
            active_view,
            group=plan.row_collective_group,
            dp_rank=plan.outer_target_dp_rank,
            contribute=plan.attn_tp_rank == 0,
        )
        if full_hidden.shape[0] == 0:
            state.hidden_states = welm_dp_local_slot_slice(
                full_hidden,
                active_view,
                dp_rank=plan.outer_target_dp_rank,
            )
            state.hidden_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
            state.residual_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
            return state
        valid_mask = welm_dp_segmented_valid_mask(
            active_view, device=full_hidden.device
        )
        invalid_mask = welm_dp_segmented_invalid_mask(active_view)
        if valid_mask.shape[0] != full_hidden.shape[0]:
            raise RuntimeError(
                "WeLMv4 segmented MoE mask does not match gathered rows"
            )
        full_hidden.masked_fill_(invalid_mask[:, None], 0)
        use_local_ep = (
            transport is WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR
        )
        mlp_output = layer.mlp(
            full_hidden,
            None,
            forward_batch,
            False,
            return_components=False,
            use_welm_local_ep_moe=use_local_ep,
            valid_row_mask=valid_mask,
            invalid_row_mask=invalid_mask,
            invalid_topk_id=-1 if use_local_ep else 0,
            allow_inplace_expert_shared_merge=True,
            resolved_moe_tp_group=(
                None if use_local_ep else plan.moe_tp_group
            ),
            resolved_moe_ep_group=(
                plan.moe_ep_group if use_local_ep else None
            ),
            use_welm_decode_like_stream_policy=use_local_ep,
        )
        full_hidden = self._store_final_components(layer, mlp_output)
        full_hidden.masked_fill_(invalid_mask[:, None], 0)
        local_hidden_slot = welm_dp_local_slot_slice(
            full_hidden,
            active_view,
            dp_rank=plan.outer_target_dp_rank,
        )
        if local_output_rows > local_hidden_slot.shape[0]:
            raise RuntimeError(
                "WeLMv4 local MoE output exceeds its restored DP slot"
            )
        state.hidden_states = local_hidden_slot.narrow(
            0, 0, local_output_rows
        )
        local_start = int(active_view.slot_offsets_cpu[plan.outer_target_dp_rank])
        local_invalid_mask = invalid_mask.narrow(
            0, local_start, int(active_view.local_slot_rows)
        )
        assert state.residual is not None
        if state.residual.shape[0] > int(active_view.local_slot_rows):
            raise RuntimeError(
                "WeLMv4 local residual exceeds its fixed DP slot: "
                f"{state.residual.shape[0]} > {active_view.local_slot_rows}"
            )
        state.residual.masked_fill_(
            local_invalid_mask.narrow(0, 0, state.residual.shape[0])[:, None],
            0,
        )
        state.hidden_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
        state.residual_layout = WelmDpLayout.DP_LOCAL_TP_ATTN_FULL
        return state
