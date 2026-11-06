# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DeepEP capture/replay adapter — records the dispatch mode used during
capture and re-applies it during replay so DeepEP all-to-all has
consistent expert routing across the captured graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def _welm_draft_graph_uses_local_ep_only(model_runner: ModelRunner) -> bool:
    """Resolve the physical NextN graph's transport once before capture.

    A local-EP graph must not mark the process-wide DeepEP buffer NORMAL:
    the following real LL graph would then clean a buffer that NextN never
    used. Keep other models and genuine DeepEP draft graphs unchanged.
    """
    if (
        not envs.SGLANG_WELMV4_DECODE_RS_AG.get()
        or model_runner.device != "npu"
        or not model_runner.is_draft_worker
        or model_runner.server_args.disaggregation_mode not in ("null", "decode")
        or not get_moe_a2a_backend().is_deepep()
        or not get_deepep_mode().is_auto()
    ):
        return False

    architectures = model_runner.model_config.hf_config.architectures or []
    if "WeLMV4MoeForCausalLMNextN" not in architectures:
        return False
    inner_model = getattr(model_runner.model, "model", None)
    layers = getattr(inner_model, "decoder_layers", None)
    if layers is None or len(layers) != 1:
        return False
    layer = layers[0]
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    if (
        not getattr(layer, "is_nextn", False)
        or not getattr(mlp, "welm_local_ep_kernel_available", False)
        or getattr(experts, "local_ep_dispatcher", None) is None
    ):
        return False

    executor = getattr(layer, "_welm_dp_executor", None)
    plan = getattr(model_runner, "runner_parallel_plan", None)
    if executor is not None:
        # The finalized DRAFT+EP executor always selects explicit local-EP,
        # including padded/idle participants. Kernel availability alone is
        # insufficient: the target also has a local-EP fallback dispatcher.
        return bool(
            plan is not None
            and getattr(plan.role, "name", None) == "DRAFT"
            and plan.finalized
            and plan.has_moe_ep
            and plan.local_ep_kernel_available
            and executor.runner_plan is plan
        )

    # Pure-TP NextN uses _forward_non_dp. Its DECODE/TARGET_VERIFY and
    # DRAFT_EXTEND_V2 graph modes all select this legacy full-row selector.
    return bool(getattr(mlp, "supports_welm_local_ep_moe", False))


class DeepEPCudaGraphRunnerAdapter:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        # Record DeepEP mode used during capture to ensure replay consistency.
        self._captured_deepep_mode = None

    @classmethod
    def for_draft_runner(
        cls, model_runner: ModelRunner
    ) -> DeepEPCudaGraphRunnerAdapter:
        return cls(enabled=not _welm_draft_graph_uses_local_ep_only(model_runner))

    def capture(self, is_extend_in_batch: bool) -> None:
        if not self._enabled or not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def replay(self) -> None:
        if not self._enabled or not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)
