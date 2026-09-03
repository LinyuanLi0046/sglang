# Copyright 2023-2024 SGLang Team
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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.configs.model_config import get_welmv4_layerwise_sliding_windows
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.communication_op import (
    moe_expert_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)
from sglang.srt.layers.rotary_embedding.yarn import yarn_get_mscale
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
    mmq_style_expert_bias_topk,
    mmq_style_k_rms_norm,
    mmq_style_norm_after_attn,
    mmq_style_router_linear,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.welmv4_dp_attention import (
    WelmDpAttentionExecutor,
    WelmRunnerParallelPlan,
    WelmRunnerRole,
    get_welm_runner_build_plan_for_init,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.server_args import get_global_server_args

# from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_npu, make_layers

if is_npu():
    import torch_npu

    from sglang.srt.hardware_backend.npu.cmo import (
        prepare_weight_cache,
        wait_cmo_stream,
    )
    from sglang.srt.hardware_backend.npu.utils import (
        process_shared_expert,
        wait_share_stream,
    )
    from sglang.srt.layers.welmv4_npu_op import (
        build_welmv4_rope_segment_tile_starts,
        inplace_sigmoid_mul_npu,
        mmq_style_router_linear_npu,
        welmv4_oe_hash_decode_4way_npu,
        welmv4_oe_hash_explicit_history_4way_npu,
        welmv4_oe_hash_prefill_4way_npu,
    )

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()


class WelmV4CommunicatorRMSNorm(nn.Module):
    """Adapt WeLM fused RMSNorm to LayerCommunicator's return-value contract."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = WelmV4FusedRMSNorm(hidden_size, eps=eps)
        self.weight = self.norm.weight
        self.eps = self.norm.eps
        self.variance_epsilon = self.norm.eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if post_residual_addition is not None:
            residual = (
                post_residual_addition
                if residual is None
                else residual + post_residual_addition
            )
        output = self.norm(x, residual, **kwargs)
        if not kwargs and residual is None and isinstance(output, tuple):
            return output[0]
        return output


def hash_input_ids_vectorized(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.to(torch.int64)
    result = ids * 2654435761
    result = result & 0xFFFFFFFF
    return result.to(input_ids.dtype)


class KVMirrorManager:
    """
    Manager for kv mirror algorithm
    """

    activations_dict_kv = dict()

    @staticmethod
    def set_kv_activation(layer_number, kv_activation):
        KVMirrorManager.activations_dict_kv[layer_number] = kv_activation

    @staticmethod
    def get_kv_activation(layer_number, clear=False):
        assert (
            layer_number in KVMirrorManager.activations_dict_kv
        ), f"layer {layer_number} not in activations_dict_kv, only layers {KVMirrorManager.activations_dict_kv.keys()} are existing"
        kv_activation = KVMirrorManager.activations_dict_kv.pop(layer_number)
        if clear:
            KVMirrorManager.activations_dict_kv.clear()
        return kv_activation


WELMV4_MTP_MIRROR_STATES_KEY = "welmv4_mtp_mirror_states"


def _set_welm_mtp_mirror_state(
    forward_batch: ForwardBatch,
    consumer_layer_id: int,
    mirror_k: torch.Tensor,
    mirror_v: torch.Tensor,
) -> None:
    states = forward_batch.model_specific_states
    if states is None:
        states = {}
        forward_batch.model_specific_states = states
    mirror_states = states.setdefault(WELMV4_MTP_MIRROR_STATES_KEY, {})
    mirror_states[int(consumer_layer_id)] = (mirror_k, mirror_v)


def _get_welm_mtp_mirror_state(
    forward_batch: ForwardBatch, consumer_layer_id: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    spec_info = getattr(forward_batch, "spec_info", None)
    states = getattr(spec_info, "model_specific_states", None)
    if states is None:
        states = forward_batch.model_specific_states
    if not states:
        return None
    return states.get(WELMV4_MTP_MIRROR_STATES_KEY, {}).get(
        int(consumer_layer_id)
    )


class _WelmDerivedMXFP8Linear(nn.Module):
    """A load-time derived projection that reuses an existing MXFP8 method.

    Target-model source and mirror layers can rewrite their existing qkv_proj in
    place. A separately loaded NextN consumer instead keeps its loader-owned
    full module and exposes this derived Q-only projection to both mirror-fill
    and frozen-draft runtime paths, without introducing another quantization
    implementation.
    """

    def __init__(
        self,
        source_proj: nn.Module,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.quant_method = source_proj.quant_method
        self.scheme = source_proj.scheme
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            weight_scale.clone(), requires_grad=False
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.quant_method.apply(self, hidden_states, self.bias), None


class LayerManager:
    decoder_layer = dict()
    num_nextn_predict_layers: int = 0
    num_target_layers: int = 0
    num_nextn_predict_layer_idx: List[int] = []
    # Cross-model pairs are first prepared while the target loader still owns
    # raw checkpoint-layout weights. The later NextN load then only has to
    # reduce its consumer projection to Q; the target source may already have
    # undergone ModelSlim runtime relayout and must not be rewritten again.
    prepared_cross_model_pairs = set()

    @staticmethod
    def set_decoder_layer(layer_idx, decoder_layer):
        LayerManager.decoder_layer[layer_idx] = decoder_layer

    @staticmethod
    def _qkv_quant_kind(qkv_proj: nn.Module) -> str:
        quant_method_name = type(qkv_proj.quant_method).__name__
        if quant_method_name == "UnquantizedLinearMethod":
            return "unquantized"
        if (
            quant_method_name == "ModelSlimLinearMethod"
            and type(getattr(qkv_proj, "scheme", None)).__name__
            == "ModelSlimMXFP8Scheme"
        ):
            return "modelslim_mxfp8"
        scheme_name = type(getattr(qkv_proj, "scheme", None)).__name__
        return f"unsupported:{quant_method_name}/{scheme_name}"

    @staticmethod
    def _replace_qkv_with_raw_mxfp8(
        qkv_proj: nn.Module,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_partition_sizes: List[int],
    ) -> None:
        """Replace a loaded QKV payload before ModelSlim runtime relayout."""
        qkv_proj.weight = nn.Parameter(weight.clone(), requires_grad=False)
        qkv_proj.weight_scale = nn.Parameter(
            weight_scale.clone(), requires_grad=False
        )
        if bias is None:
            qkv_proj.bias = None
        else:
            qkv_proj.bias = nn.Parameter(bias.clone(), requires_grad=False)

        # ColumnParallelLinear.forward uses the actual weight shape.  Keep its
        # metadata consistent as well for prefetch, debug output and optional
        # consumers such as LoRA that inspect partition widths after loading.
        local_output_size = int(weight.shape[0])
        qkv_proj.output_size_per_partition = local_output_size
        qkv_proj.output_partition_sizes = list(output_partition_sizes)
        qkv_proj.output_size = local_output_size * qkv_proj.tp_size
        qkv_proj.output_sizes = [
            int(size) * qkv_proj.tp_size for size in output_partition_sizes
        ]

    @staticmethod
    def _prepare_already_paired_nextn_consumer(mirror_layer_attn) -> None:
        """Make a freshly loaded NextN consumer Q-only without touching source."""
        mirror_quant_kind = LayerManager._qkv_quant_kind(
            mirror_layer_attn.qkv_proj
        )
        if mirror_quant_kind == "modelslim_mxfp8":
            mirror_qkv_proj = mirror_layer_attn.qkv_proj
            if not hasattr(mirror_qkv_proj, "weight_scale"):
                raise RuntimeError(
                    "WeLMv4 ModelSlim NextN Q extraction must run before "
                    "consumer weight post-processing."
                )
            q_size = mirror_layer_attn.q_size
            bias = getattr(mirror_qkv_proj, "bias", None)
            mirror_layer_attn.kv_mirror_query_proj = _WelmDerivedMXFP8Linear(
                mirror_qkv_proj,
                mirror_qkv_proj.weight[:q_size, :],
                mirror_qkv_proj.weight_scale[:q_size, :],
                bias[:q_size] if bias is not None else None,
            )
            return
        if mirror_quant_kind != "unquantized":
            raise NotImplementedError(
                "WeLMv4 cross-model NextN projection rewriting supports only "
                "unquantized or ModelSlim W8A8_MXFP8 QKV weights."
            )

        q_size = mirror_layer_attn.q_size
        qkv_proj = mirror_layer_attn.qkv_proj
        mirror_layer_attn.qkv_proj_weight = qkv_proj.weight[:q_size, :].clone()
        bias = getattr(qkv_proj, "bias", None)
        mirror_layer_attn.qkv_proj_bias = (
            bias[:q_size].clone() if bias is not None else None
        )

    @staticmethod
    def post_init(kv_mirror_layers, kv_mirror_imitated_layers, is_nextn=False):

        if is_nextn:
            LayerManager.num_nextn_predict_layer_idx = kv_mirror_layers
        for mirror_layer_id in kv_mirror_layers:
            if mirror_layer_id >= len(LayerManager.decoder_layer):
                continue
            imitated_layer_id = kv_mirror_imitated_layers[
                kv_mirror_layers.index(mirror_layer_id)
            ]
            mirror_layer_attn = LayerManager.decoder_layer[mirror_layer_id].self_attn
            imitated_layer_attn = LayerManager.decoder_layer[
                imitated_layer_id
            ].self_attn

            pair = (int(imitated_layer_id), int(mirror_layer_id))
            if is_nextn and pair in LayerManager.prepared_cross_model_pairs:
                LayerManager._prepare_already_paired_nextn_consumer(
                    mirror_layer_attn
                )
                continue

            mirror_quant_kind = LayerManager._qkv_quant_kind(
                mirror_layer_attn.qkv_proj
            )
            imitated_quant_kind = LayerManager._qkv_quant_kind(
                imitated_layer_attn.qkv_proj
            )
            if mirror_quant_kind != imitated_quant_kind:
                raise ValueError(
                    "WeLMv4 KV-mirror source and consumer QKV projections must "
                    "use the same precision, but pair "
                    f"{imitated_layer_id}->{mirror_layer_id} uses "
                    f"{imitated_quant_kind} and {mirror_quant_kind}."
                )
            if mirror_quant_kind.startswith("unsupported:"):
                raise NotImplementedError(
                    "WeLMv4 KV-mirror projection rewriting currently supports "
                    "only unquantized or ModelSlim W8A8_MXFP8 QKV weights; pair "
                    f"{imitated_layer_id}->{mirror_layer_id} uses "
                    f"{mirror_quant_kind.removeprefix('unsupported:')}."
                )

            mirror_qkv_proj_weight = mirror_layer_attn.qkv_proj.weight
            mirror_qkv_proj_bias = getattr(mirror_layer_attn.qkv_proj, "bias", None)
            imitated_qkv_proj_weight = imitated_layer_attn.qkv_proj.weight
            imitated_qkv_proj_bias = getattr(imitated_layer_attn.qkv_proj, "bias", None)
            assert (mirror_qkv_proj_bias is not None) == (
                imitated_qkv_proj_bias is not None
            )

            if mirror_quant_kind == "modelslim_mxfp8":
                mirror_qkv_proj = mirror_layer_attn.qkv_proj
                imitated_qkv_proj = imitated_layer_attn.qkv_proj
                if not hasattr(mirror_qkv_proj, "weight_scale") or not hasattr(
                    imitated_qkv_proj, "weight_scale"
                ):
                    raise RuntimeError(
                        "WeLMv4 ModelSlim KV-mirror QKV rewriting must run on "
                        "the raw [N, K] checkpoint layout before MXFP8 weight "
                        "post-processing."
                    )

                mirror_q_size = mirror_layer_attn.q_size
                mirror_weight_data = mirror_qkv_proj_weight[:mirror_q_size, :]
                mirror_scale_data = mirror_qkv_proj.weight_scale[
                    :mirror_q_size, :
                ]
                imitated_weight_data = torch.concat(
                    [
                        imitated_qkv_proj_weight,
                        mirror_qkv_proj_weight[mirror_q_size:, :],
                    ],
                    dim=0,
                )
                imitated_scale_data = torch.concat(
                    [
                        imitated_qkv_proj.weight_scale,
                        mirror_qkv_proj.weight_scale[mirror_q_size:, :],
                    ],
                    dim=0,
                )

                mirror_bias_data = (
                    mirror_qkv_proj_bias[:mirror_q_size]
                    if mirror_qkv_proj_bias is not None
                    else None
                )
                imitated_bias_data = (
                    torch.concat(
                        [
                            imitated_qkv_proj_bias,
                            mirror_qkv_proj_bias[mirror_q_size:],
                        ],
                        dim=0,
                    )
                    if mirror_qkv_proj_bias is not None
                    else None
                )

                # Build both payloads before replacing either projection: the
                # source needs the consumer's original K/V rows.
                LayerManager._replace_qkv_with_raw_mxfp8(
                    imitated_qkv_proj,
                    imitated_weight_data,
                    imitated_scale_data,
                    imitated_bias_data,
                    [
                        imitated_layer_attn.q_size,
                        imitated_layer_attn.kv_size,
                        imitated_layer_attn.kv_size,
                        mirror_layer_attn.kv_size,
                        mirror_layer_attn.kv_size,
                    ],
                )
                imitated_layer_attn._kv_mirror_mxfp8_source_projection = True

                if mirror_layer_attn.is_nextn:
                    # NextN active draft and mirror-fill forwards are Q-only.
                    # Keep the original module for loader ownership and expose
                    # a derived Q projection for every runtime path.
                    mirror_layer_attn.kv_mirror_query_proj = (
                        _WelmDerivedMXFP8Linear(
                            mirror_qkv_proj,
                            mirror_weight_data,
                            mirror_scale_data,
                            mirror_bias_data,
                        )
                    )
                else:
                    LayerManager._replace_qkv_with_raw_mxfp8(
                        mirror_qkv_proj,
                        mirror_weight_data,
                        mirror_scale_data,
                        mirror_bias_data,
                        [mirror_layer_attn.q_size],
                    )
                    mirror_layer_attn._kv_mirror_mxfp8_query_projection = True
                if mirror_layer_id >= LayerManager.num_target_layers:
                    LayerManager.prepared_cross_model_pairs.add(pair)
                continue

            mirror_weight_data = mirror_qkv_proj_weight[
                : mirror_layer_attn.q_size, :
            ]
            imitated_weight_data = torch.concat(
                [
                    imitated_qkv_proj_weight,
                    mirror_qkv_proj_weight[mirror_layer_attn.q_size :, :],
                ],
                dim=0,
            )

            # Use in-place copy to preserve tensor addresses for CUDA graph
            # compatibility. Creating new tensors would invalidate captured
            # CUDA graphs that reference the old memory addresses.
            if hasattr(mirror_layer_attn, "qkv_proj_weight"):
                mirror_layer_attn.qkv_proj_weight.copy_(mirror_weight_data)
            else:
                mirror_layer_attn.qkv_proj_weight = mirror_weight_data.clone()

            if hasattr(imitated_layer_attn, "qkv_proj_weight"):
                imitated_layer_attn.qkv_proj_weight.copy_(imitated_weight_data)
            else:
                imitated_layer_attn.qkv_proj_weight = imitated_weight_data.clone()

            if mirror_qkv_proj_bias is not None:
                mirror_bias_data = mirror_qkv_proj_bias[
                    : mirror_layer_attn.q_size
                ]
                imitated_bias_data = torch.concat(
                    [
                        imitated_qkv_proj_bias,
                        mirror_qkv_proj_bias[mirror_layer_attn.q_size :],
                    ],
                    dim=0,
                )
                if hasattr(mirror_layer_attn, "qkv_proj_bias") and mirror_layer_attn.qkv_proj_bias is not None:
                    mirror_layer_attn.qkv_proj_bias.copy_(mirror_bias_data)
                else:
                    mirror_layer_attn.qkv_proj_bias = mirror_bias_data.clone()

                if hasattr(imitated_layer_attn, "qkv_proj_bias") and imitated_layer_attn.qkv_proj_bias is not None:
                    imitated_layer_attn.qkv_proj_bias.copy_(imitated_bias_data)
                else:
                    imitated_layer_attn.qkv_proj_bias = imitated_bias_data.clone()
            else:
                imitated_layer_attn.qkv_proj_bias = None
                mirror_layer_attn.qkv_proj_bias = None

            if mirror_layer_id >= LayerManager.num_target_layers:
                LayerManager.prepared_cross_model_pairs.add(pair)

        torch.get_device_module().empty_cache()


class _WelmMTPMirrorStagingAttention(nn.Module):
    """Load a cross-model consumer QKV while target weights are still raw.

    ModelSlim rewrites QKV storage immediately after ``model.load_weights``.
    A small target-owned staging projection lets source0 absorb layer48 K/V
    before that rewrite. It is released before serving and the real draft layer
    later loads its own layer48 weights normally.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        logical_layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        attn_tp_size = get_parallel().attn_tp_size
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // num_heads)
        )
        self.q_size = (num_heads // attn_tp_size) * head_dim
        self.kv_size = max(1, num_kv_heads // attn_tp_size) * head_dim
        self.is_nextn = True
        self.loaded_qkv_parts = set()
        if getattr(config, "qkv_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_bias")
        elif getattr(config, "qkv_proj_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_proj_bias")
        else:
            qkv_bias = True
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=get_parallel().attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix(
                f"layers.{logical_layer_id}.self_attn.qkv_proj", prefix
            ),
        )


class _WelmMTPMirrorStagingLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        logical_layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = _WelmMTPMirrorStagingAttention(
            config, logical_layer_id, quant_config, prefix
        )


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_clamp_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.swiglu_clamp_limit = swiglu_clamp_limit

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        if self.swiglu_clamp_limit is not None and self.swiglu_clamp_limit > 0:
            d = gate_up.shape[-1] // 2
            gate = F.silu(gate_up[..., :d]).clamp_(max=self.swiglu_clamp_limit)
            up = gate_up[..., d:].clamp(min=-self.swiglu_clamp_limit, max=self.swiglu_clamp_limit)
            x = gate * up
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


def expert_bias_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    renormalize: bool = False,
    score_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if score_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1).type_as(gating_output)
    else:
        scores = torch.sigmoid(gating_output).type_as(gating_output)

    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and expert_bias.dtype == torch.float32
    ):
        topk_scores, indices = mmq_style_expert_bias_topk(scores, expert_bias, topk)
    else:
        scores_for_routing = scores + expert_bias
        _, indices = torch.topk(scores_for_routing, topk, dim=-1)
        topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)

    if renormalize:
        topk_scores = (
            topk_scores.float()
            / topk_scores.float().sum(dim=-1, keepdim=True).clamp_min(1e-20)
        ).type_as(topk_scores)
    return topk_scores, indices


def sigmoid_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if softmax, then use qwen3 moe's routing function
    scores = torch.sigmoid(gating_output).type_as(gating_output)
    scores_for_routing = scores
    if correction_bias is not None:
        # Bias changes expert selection only; the dispatched mixture weights
        # are gathered from the original sigmoid scores.
        scores_for_routing = scores + correction_bias
    _, indices = torch.topk(scores_for_routing, topk, dim=-1)
    topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)
    if renormalize:
        topk_scores = (
            topk_scores.float()
            / topk_scores.float().sum(dim=-1, keepdim=True).clamp_min(1e-20)
        ).type_as(topk_scores)
    return topk_scores, indices


class Qwen2MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[Any] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.expert_bias = torch.nn.Parameter(
            torch.zeros((config.num_experts), dtype=torch.float32)
        )
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.num_hidden_layers = int(
            getattr(config, "num_target_hidden_layers", 0)
        ) + int(config.num_hidden_layers)
        self.last_final_experts_output: Optional[torch.Tensor] = None
        self.last_final_shared_output: Optional[torch.Tensor] = None
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        moe_clamp_limits = getattr(config, "moe_expert_swiglu_clamp_limit_layerwise", [])
        moe_clamp_limit = (
            moe_clamp_limits[layer_id]
            if layer_id < len(moe_clamp_limits) and moe_clamp_limits[layer_id] > 0
            else None
        )
        shared_clamp_limits = getattr(config, "shared_expert_swiglu_clamp_limit_layerwise", [])
        shared_clamp_limit = (
            shared_clamp_limits[layer_id]
            if layer_id < len(shared_clamp_limits) and shared_clamp_limits[layer_id] > 0
            else None
        )

        self.router_score_func = (
            config.router_score_func
            if hasattr(config, "router_score_func")
            else "softmax"
        )
        if config.moe_routing_type == "expert_bias":
            from functools import partial

            custom_routing_function = partial(
                expert_bias_routing,
                expert_bias=self.expert_bias,
                score_func=self.router_score_func,
            )
            self.custom_routing_function = custom_routing_function
        else:
            if self.router_score_func == "softmax":
                self.custom_routing_function = None
            elif self.router_score_func == "sigmoid":
                self.custom_routing_function = sigmoid_routing_function
            else:
                raise ValueError(f"Unknown router_score_func: {self.router_score_func}")

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self.custom_routing_function,
        )

        runner_plan = get_welm_runner_build_plan_for_init()
        self.welm_runner_plan = runner_plan
        legacy_local_ep_kernel_available = (
            _is_npu
            and get_moe_a2a_backend().is_deepep()
            and get_parallel().moe_ep_size > 1
            and get_parallel().moe_ep_size == self.tp_size
            and get_parallel().moe_tp_size == 1
        )
        planned_local_ep_kernel_available = (
            _is_npu
            and runner_plan is not None
            and runner_plan.has_moe_ep
            and get_moe_a2a_backend().is_deepep()
            and runner_plan.moe_ep_size > 1
            and runner_plan.moe_tp_size == 1
        )
        self.welm_local_ep_kernel_available = (
            legacy_local_ep_kernel_available
            or planned_local_ep_kernel_available
        )
        # The non-DP path consumes this selector.  DP execution selects the
        # separately named kernel capability through its immutable plan.
        self.supports_welm_local_ep_moe = legacy_local_ep_kernel_available

        if get_bool_env_var("SGLANG_WELMV4_MMQ_SCORE_ON_SWIGLU", "false"):
            logger.warning(
                "SGLANG_WELMV4_MMQ_SCORE_ON_SWIGLU is not supported by the "
                "latest MoE runner API and will be ignored."
            )
        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            # WeLM applies silu(gate).clamp(max=L) * up.clamp(-L, L).
            # In latest main this is the cross-platform gemm1 clamp contract;
            # swiglu_limit is a different, DSV4-specific CUDA/HIP path.
            gemm1_clamp_limit=moe_clamp_limit,
            enable_local_ep_dispatcher=self.welm_local_ep_kernel_available,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.weight.data = self.gate.weight.to(torch.float32)
        self.register_buffer("_npu_router_compute_weight", None, persistent=False)
        self.register_buffer("_npu_router_compute_weight_t", None, persistent=False)
        if config.shared_expert_intermediate_size > 0:
            use_ep_replicated_shared_expert = (
                get_moe_a2a_backend().is_deepep()
                and (runner_plan is None or runner_plan.has_moe_ep)
            )
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                swiglu_clamp_limit=shared_clamp_limit,
                **(
                    dict(tp_rank=0, tp_size=1)
                    if use_ep_replicated_shared_expert
                    else {}
                ),
            )
        else:
            self.shared_expert = None

        self.shared_expert_gate = None
        has_shared_expert_gate = getattr(
            config, "has_shared_expert_gate", True
        )  # default to true since qwen2_moe always has it
        if has_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self.is_kv_mirror_consumer = self.layer_id in set(
            getattr(config, "kv_mirror_layers", []) or []
        )

    def _forward_shared_expert(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.shared_expert is None:
            return None

        shared_output = self.shared_expert(hidden_states)
        if self.shared_expert_gate is not None:
            shared_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
            )
        return shared_output

    def get_npu_router_compute_weight(
        self, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._npu_router_compute_weight is None:
            self._npu_router_compute_weight = self.gate.weight.to(
                dtype=dtype
            ).contiguous()
        return self._npu_router_compute_weight

    def prepare_npu_router_compute_weight_t(self, dtype: torch.dtype) -> None:
        self._npu_router_compute_weight_t = (
            self.gate.weight.to(dtype=dtype).t().contiguous()
        )

    def get_npu_router_compute_weight_t(self) -> torch.Tensor:
        assert self._npu_router_compute_weight_t is not None
        return self._npu_router_compute_weight_t

    @staticmethod
    def _mask_npu_padded_topk(
        topk_output: StandardTopKOutput,
        num_token_non_padded: Optional[torch.Tensor] = None,
        preserve_padded_ids: bool = False,
        valid_row_mask: Optional[torch.Tensor] = None,
        invalid_row_mask: Optional[torch.Tensor] = None,
        invalid_topk_id: Optional[int] = None,
    ) -> StandardTopKOutput:
        if valid_row_mask is not None:
            if (
                valid_row_mask.dim() != 1
                or valid_row_mask.shape[0] != topk_output.topk_ids.shape[0]
            ):
                raise RuntimeError(
                    "WeLMv4 segmented MoE mask must have one entry per row: "
                    f"{tuple(valid_row_mask.shape)} vs {topk_output.topk_ids.shape[0]}"
                )
            if invalid_row_mask is None:
                raise RuntimeError(
                    "WeLMv4 segmented MoE masking requires its preallocated "
                    "invalid-row mask"
                )
            if (
                invalid_row_mask.dim() != 1
                or invalid_row_mask.shape != valid_row_mask.shape
            ):
                raise RuntimeError(
                    "WeLMv4 invalid-row mask must match the valid-row mask"
                )
            padded_rows = invalid_row_mask
            if invalid_topk_id is not None:
                topk_output.topk_ids.masked_fill_(
                    padded_rows[:, None], int(invalid_topk_id)
                )
            elif not preserve_padded_ids:
                topk_output.topk_ids.masked_fill_(padded_rows[:, None], -1)
            topk_output.topk_weights.masked_fill_(padded_rows[:, None], 0)
            return topk_output
        else:
            if invalid_row_mask is not None:
                raise RuntimeError(
                    "WeLMv4 invalid-row mask requires a valid-row mask"
                )
            if num_token_non_padded is None:
                raise RuntimeError("WeLMv4 padded TopK requires row validity metadata")
            padded_rows = torch.arange(
                topk_output.topk_ids.shape[0], device=topk_output.topk_ids.device
            ) >= num_token_non_padded
        topk_ids = topk_output.topk_ids
        if invalid_topk_id is not None:
            topk_ids = torch.where(
                padded_rows[:, None],
                torch.full_like(topk_output.topk_ids, int(invalid_topk_id)),
                topk_output.topk_ids,
            )
        elif not preserve_padded_ids:
            topk_ids = torch.where(
                padded_rows[:, None],
                torch.full_like(topk_output.topk_ids, -1),
                topk_output.topk_ids,
            )
        return StandardTopKOutput(
            torch.where(
                padded_rows[:, None],
                torch.zeros_like(topk_output.topk_weights),
                topk_output.topk_weights,
            ),
            topk_ids,
            topk_output.router_logits,
        )

    @staticmethod
    def _resolve_deepep_mode_for_topk(is_prefill_batch: bool) -> DeepEPMode:
        mode_override = get_forward().deepep_mode_override
        if mode_override is not None and not isinstance(mode_override, DeepEPMode):
            raise TypeError(
                "deepep_mode_override must be a DeepEPMode or None, got "
                f"{type(mode_override).__name__}"
            )
        return (
            mode_override
            if mode_override is not None
            else get_deepep_mode().resolve(is_prefill_batch)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        return_components: bool = False,
        skip_component_output: bool = False,
        use_welm_local_ep_moe: bool = False,
        use_welm_decode_like_stream_policy: bool = False,
        use_welm_prefill_normal_stream_policy: bool = False,
        valid_row_mask: Optional[torch.Tensor] = None,
        invalid_row_mask: Optional[torch.Tensor] = None,
        invalid_topk_id: Optional[int] = None,
        allow_inplace_expert_shared_merge: bool = False,
        resolved_moe_tp_group=None,
        resolved_moe_ep_group=None,
    ) -> torch.Tensor:
        if allow_inplace_expert_shared_merge and return_components:
            raise RuntimeError(
                "WeLMv4 in-place expert/shared merge cannot publish separate "
                "MLP components"
            )
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        moe_a2a_backend = get_moe_a2a_backend()
        is_prefill_batch = (
            forward_batch is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
                include_draft_extend_v2=True
            )
        )
        is_kv_mirror_prefill = (
            forward_batch is not None
            and forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.is_kv_mirror_consumer
        )
        # Preserve the existing dual-stream policy for non-prefill modes and
        # additionally let mirror prefill use that same policy.
        use_decode_like_stream_policy = (
            not is_prefill_batch
            or is_kv_mirror_prefill
            or use_welm_decode_like_stream_policy
        )
        enable_npu_decode_like_dual_stream = (
            _is_npu
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and self.shared_expert is not None
            and hidden_states.shape[0] > 0
            and (moe_a2a_backend.is_none() or moe_a2a_backend.is_deepep())
            and use_decode_like_stream_policy
        )
        enable_npu_prefill_normal_shared_overlap = (
            _is_npu
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and envs.SGLANG_DEEPEP_NORMAL_USE_ALLGATHER.get()
            and self.shared_expert is not None
            and hidden_states.shape[0] > 0
            and forward_batch is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
            and not is_kv_mirror_prefill
            and moe_a2a_backend.is_deepep()
            and get_parallel().moe_ep_size > 1
            and (
                use_welm_prefill_normal_stream_policy
                or (
                    forward_batch.welmv4_npu_deepep_scattered
                    and not forward_batch.welmv4_npu_deepep_full_mirror
                )
            )
            and self._resolve_deepep_mode_for_topk(True) == DeepEPMode.NORMAL
        )
        enable_npu_shared_alt_stream = (
            enable_npu_decode_like_dual_stream
            or enable_npu_prefill_normal_shared_overlap
        )
        num_token_non_padded = (
            getattr(forward_batch, "num_token_non_padded", None)
            if forward_batch is not None
            else None
        )
        if valid_row_mask is not None:
            # A scalar cannot describe holes between fixed DP graph slots.
            num_token_non_padded = None
        # Decode and Spec-V2 target verify keep the legacy FULL replicated
        # token layout when the WeLMv4 DeepEP scattered path is disabled.  The
        # generic gathered-buffer metadata above is TP-localized, so applying
        # it to these FULL rows would incorrectly mask real requests after the
        # first local shard.  Leave padding unmasked in this layout. Ordinary
        # scattered prefill keeps the localized count; the explicitly NORMAL
        # NextN DRAFT_EXTEND_V2 path restores its full-layout real count before
        # entering this block, so it can still mask only the padded suffix.
        is_full_deepep_decode_like = (
            _is_npu
            and moe_a2a_backend.is_deepep()
            and forward_batch is not None
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
            and not forward_batch.welmv4_npu_deepep_scattered
        )
        if is_full_deepep_decode_like:
            num_token_non_padded = None
        if moe_a2a_backend.is_deepep() and hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        else:
            if self.shared_expert is not None and not enable_npu_shared_alt_stream:
                shared_output = self._forward_shared_expert(hidden_states)
            if _is_npu:
                router_logits = torch.mm(
                    hidden_states,
                    self.get_npu_router_compute_weight_t(),
                    out_dtype=torch.float32,
                )
            else:
                router_logits = mmq_style_router_linear(
                    hidden_states, self.gate.weight
                )
            if enable_npu_decode_like_dual_stream:
                # Start after the router Cube GEMM. The shared expert overlaps
                # routing, dispatch, and the routed-expert path until final add;
                # both paths only read the original hidden_states storage.
                shared_output = process_shared_expert(
                    hidden_states, self._forward_shared_expert
                )
            # Ascend's generic fused TopK dispatch ignores custom routing callbacks.
            # Route WeLM's expert-bias callback through MoeGatingTopK explicitly;
            # it uses the native TopK tie order rather than MMQ's tie_rank order.
            if _is_npu and self.custom_routing_function is not None:
                if (
                    getattr(self.custom_routing_function, "func", None)
                    is expert_bias_routing
                ):
                    topk_output = self.topk.forward_npu(
                        hidden_states,
                        router_logits,
                        num_token_non_padded=num_token_non_padded,
                        expert_location_dispatch_info=(
                            ExpertLocationDispatchInfo.init_new(
                                layer_id=self.layer_id
                            )
                            if not self.is_nextn
                            else None
                        ),
                        expert_bias=self.expert_bias,
                        scoring_func_override=self.router_score_func,
                    )
                else:
                    topk_output = self.topk.forward_native(
                        hidden_states,
                        router_logits,
                        num_token_non_padded=num_token_non_padded,
                    )
            else:
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    num_token_non_padded=num_token_non_padded,
                )
        if _is_npu and (num_token_non_padded is not None or valid_row_mask is not None):
            if not isinstance(topk_output, StandardTopKOutput):
                raise RuntimeError(
                    "WeLMv4 NPU padding requires StandardTopKOutput, got "
                    f"{type(topk_output).__name__}"
                )
            # NPU DeepEP normal AllGather consumes -1 as an inactive route. The
            # legacy low-latency kernels do not unless their optional negative-ID
            # mode is enabled, so keep the router's valid, distinct expert IDs
            # there and make dummy routes numerically inert via zero weights.
            preserve_padded_ids = (
                moe_a2a_backend.is_deepep()
                and self._resolve_deepep_mode_for_topk(is_prefill_batch)
                == DeepEPMode.LOW_LATENCY
            )
            topk_output = self._mask_npu_padded_topk(
                topk_output,
                num_token_non_padded,
                preserve_padded_ids=preserve_padded_ids,
                valid_row_mask=valid_row_mask,
                invalid_row_mask=invalid_row_mask,
                invalid_topk_id=invalid_topk_id,
            )
        if enable_npu_prefill_normal_shared_overlap:
            # Ordinary non-mirror prefill keeps LOCAL token rows and DeepEP
            # NORMAL+AllGather starts with communication. Launch only after
            # router/TopK work so the shared expert's primary overlap window
            # begins at dispatch. The existing final-add wait remains the
            # consumption boundary, matching decode/mirror behavior.
            shared_output = process_shared_expert(
                hidden_states, self._forward_shared_expert
            )
        if use_welm_local_ep_moe:
            if not self.welm_local_ep_kernel_available:
                raise RuntimeError(
                    "WeLMv4 local EP MoE was selected for an unsupported "
                    "parallel configuration."
                )
            if use_reduce_scatter:
                raise RuntimeError(
                    "WeLMv4 local EP MoE requires a FULL replicated token "
                    "layout, but use_reduce_scatter is enabled."
                )
            experts_output = self.experts.forward_local_ep_partial(
                hidden_states, topk_output
            )
            # Only routed experts are partial across EP ranks. The shared
            # expert below is fully replicated under DeepEP and must be added
            # after this sum, otherwise it would be multiplied by EP size.
            experts_output = (
                resolved_moe_ep_group.all_reduce(experts_output)
                if resolved_moe_ep_group is not None
                else moe_expert_parallel_all_reduce(experts_output)
            )
        else:
            experts_output = self.experts(hidden_states, topk_output)
        if return_components and skip_component_output:
            if enable_npu_shared_alt_stream:
                # This early return hands shared_output to the caller, so it is
                # the consumption boundary.
                wait_share_stream()
            return (
                experts_output.view(num_tokens, hidden_dim),
                experts_output.view(num_tokens, hidden_dim),
                shared_output.view(num_tokens, hidden_dim)
                if shared_output is not None
                else None,
            )
        final_hidden_states = experts_output
        self.last_final_experts_output = None
        self.last_final_shared_output = None
        if shared_output is not None:
            if (
                self.layer_id == self.num_hidden_layers - 1
                and self.tp_size == 1
                and not use_reduce_scatter
                and not allow_inplace_expert_shared_merge
            ):
                self.last_final_experts_output = experts_output
                self.last_final_shared_output = shared_output
            if enable_npu_shared_alt_stream:
                # Normal inference first consumes shared_output in this add.
                wait_share_stream()
            if allow_inplace_expert_shared_merge:
                final_hidden_states.add_(shared_output)
            else:
                final_hidden_states = final_hidden_states + shared_output
        if (
            self.tp_size > 1
            and not use_reduce_scatter
            and (
                (
                    self.welm_runner_plan is not None
                    and not self.welm_runner_plan.has_moe_ep
                )
                or not get_moe_a2a_backend().is_deepep()
            )
        ):
            final_hidden_states = (
                resolved_moe_tp_group.all_reduce(final_hidden_states)
                if resolved_moe_tp_group is not None
                else tensor_model_parallel_all_reduce(final_hidden_states)
            )

        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        if return_components:
            return (
                final_hidden_states,
                experts_output.view(num_tokens, hidden_dim),
                shared_output.view(num_tokens, hidden_dim)
                if shared_output is not None
                else None,
            )
        return final_hidden_states


class LinearScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


# WelmV4InplaceRotaryEmbedding
class Qwen2MoeYarnScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        compress: float = 0,
        max_position: int = 40 * 4096,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.compress = compress
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.max_position = max_position
        inv_freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq_extra", inv_freq_extra, persistent=False)
        self.register_buffer("inv_freq_inter", inv_freq_inter, persistent=False)

        self.cos_sin_cache = self._update_cos_sin_cache(self.max_position)

    def _update_cos_sin_cache(self, seqlen: int):
        """Update cos/sin cache with YaRN scaling"""
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float32
        ).to(device=self.inv_freq_inter.device)

        inv_freq = (
            self.inv_freq_inter * (1 - inv_freq_mask)
            + self.inv_freq_extra * inv_freq_mask
        )

        seq = (
            torch.arange(seqlen, device=self.inv_freq_extra.device, dtype=torch.float32)
            * self.compress
        )

        freqs = torch.outer(seq, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        _cos_cached = (torch.cos(freqs) * _mscale).to(torch.float32)
        _sin_cached = (torch.sin(freqs) * _mscale).to(torch.float32)
        cache = torch.cat((_cos_cached, _sin_cached), dim=-1)
        return cache


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    compress: float = 1.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        raise ValueError(f"Please set RoPE scaling")
    else:
        scaling_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        if scaling_type is None:
            raise ValueError("RoPE scaling must define either 'rope_type' or 'type'")

        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )

        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            base_max_position = int(original_max_position * scaling_factor)
            if max_position < base_max_position:
                raise ValueError(
                    f"max_position ({max_position}) < original_max_position "
                    f"({original_max_position}) * scaling_factor ({scaling_factor})"
                )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = Qwen2MoeYarnScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
                compress=compress,
                max_position=max_position,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


class Qwen2MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        compress: float = 1.0,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        out_bias: int = False,
        qk_norm: bool = False,
        k_norm: bool = False,
        qk_rope_head_dim: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
        kv_mirror_layers=[],
        kv_mirror_imitated_layers=[],
        sliding_window_size: int = -1,
        enable_attn_sink_layerwise=[],
        layer_idx: Optional[int] = None,
        o_norm=False,
        rms_norm_eps: float = 1e-5,
        total_layer_num: int = 1,
        is_nextn: bool = False,
        alt_stream: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.alt_stream = alt_stream

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.compress = compress
        self.max_position_embeddings = max_position_embeddings
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_norm = qk_norm
        self.only_k_norm = k_norm
        self.use_o_norm = o_norm
        self.total_layer_num = total_layer_num
        self.o_norm = (
            WelmV4FusedRMSNorm(self.hidden_size, eps=rms_norm_eps)
            if self.use_o_norm
            else None
        )

        self.q_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=rms_norm_eps)
            if self.qk_norm
            else None
        )
        self.k_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=rms_norm_eps)
            if self.qk_norm or self.only_k_norm
            else None
        )

        self.kv_mirror_layers = kv_mirror_layers
        self.kv_mirror_imitated_layers = kv_mirror_imitated_layers
        self.layer_idx = layer_idx
        logger.debug(
            "WeLMv4 layer %s: kv_mirror_layers=%s, kv_mirror_imitated_layers=%s",
            layer_idx,
            self.kv_mirror_layers,
            self.kv_mirror_imitated_layers,
        )
        self.sliding_window_size = int(sliding_window_size)
        logger.debug(
            "WeLMv4 layer %s: sliding_window_size=%s",
            layer_idx,
            self.sliding_window_size,
        )
        if len(enable_attn_sink_layerwise) > layer_idx:
            self.enable_attention_sink = enable_attn_sink_layerwise[layer_idx]
        else:
            self.enable_attention_sink = False
        logger.debug(
            "WeLMv4 layer %s: enable_attention_sink=%s",
            layer_idx,
            self.enable_attention_sink,
        )
        if self.enable_attention_sink:
            self.attn_sink = nn.Parameter(
                torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
            )
        else:
            self.attn_sink = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=not is_dp_attention_enabled(),
            prefix=add_prefix("o_proj", prefix),
        )
        self._welm_npu_o_proj_hcom_name = None
        if rope_scaling is None:
            rope_scaling = {"type": "linear", "factor": 1 / self.compress}
        else:
            assert self.compress == 1.0, "Compress must be 1.0 for custom rope scaling."
            scaling_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
            if scaling_type == "yarn":
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                apply_softmax_scale = rope_scaling.get("apply_softmax_scale", False)
                scaling_factor = rope_scaling["factor"]
                if apply_softmax_scale and mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                    self.scaling = self.scaling * mscale * mscale

        self.rotary_emb = get_rope(
            # self.qk_rope_head_dim,
            self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.rotary_emb_orig = get_rope(
            self.qk_rope_head_dim,
            # self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=self.sliding_window_size,
        )
        self.gated_self_attention_headwise = True
        if self.gated_self_attention_headwise:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        self.attn.is_kv_mirror = self.layer_idx in self.kv_mirror_layers
        # `layer_id` is the physical/cache slot. `layer_idx` is the checkpoint
        # and layerwise-config id. They differ for NextN: physical 0, logical 48.
        self.kv_mirror_layer_idx = layer_idx
        if get_global_server_args().speculative_algorithm is not None:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_nextn_predict_layers - 1
            )
        else:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_target_layers - 1
            )
        self.is_nextn = is_nextn

    @staticmethod
    def _linear_prefetch_tensors(
        projection: nn.Module,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        tensors = [projection.weight]
        weight_scale = getattr(projection, "weight_scale_inv", None)
        if weight_scale is None:
            weight_scale = getattr(projection, "weight_scale", None)
        if weight_scale is not None:
            tensors.append(weight_scale)
        return tensors if len(tensors) > 1 else tensors[0]

    def get_qkv_prefetch_weight(
        self, forward_batch: ForwardBatch
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        external_mirror = (
            _get_welm_mtp_mirror_state(forward_batch, self.kv_mirror_layer_idx)
            if self.is_nextn
            else None
        )
        frozen_mtp_decode = (
            self.is_nextn
            and forward_batch.forward_mode.is_decode()
            and bool(getattr(forward_batch.spec_info, "welmv4_mtp_frozen_kv", False))
        )
        if self.is_nextn and (external_mirror is not None or frozen_mtp_decode):
            if hasattr(self, "kv_mirror_query_proj"):
                return self._linear_prefetch_tensors(self.kv_mirror_query_proj)
            if hasattr(self, "qkv_proj_weight"):
                return self.qkv_proj_weight
        use_full_qkv = (
            self.kv_mirror_layer_idx in self.kv_mirror_layers
            and self.kv_mirror_layer_idx
            in LayerManager.num_nextn_predict_layer_idx
            and not forward_batch.forward_mode.is_extend_without_speculative()
        )
        if use_full_qkv:
            return self._linear_prefetch_tensors(self.qkv_proj)
        if hasattr(self, "kv_mirror_query_proj"):
            return self._linear_prefetch_tensors(self.kv_mirror_query_proj)
        if hasattr(self, "qkv_proj_weight"):
            return self.qkv_proj_weight
        return self._linear_prefetch_tensors(self.qkv_proj)

    def _get_welm_npu_o_proj_hcom_name(self) -> str:
        if self._welm_npu_o_proj_hcom_name is None:
            process_group = get_tp_group().device_group
            backend = process_group._get_backend(torch.device("npu"))
            self._welm_npu_o_proj_hcom_name = backend.get_hccl_comm_name(
                process_group.rank()
            )
        return self._welm_npu_o_proj_hcom_name

    def _npu_o_proj_matmul_reduce_scatter(
        self, attn_output: torch.Tensor
    ) -> torch.Tensor:
        bias = (
            self.o_proj.bias
            if self.o_proj.tp_rank == 0 and not self.o_proj.skip_bias_add
            else None
        )
        return torch_npu.npu_mm_reduce_scatter_base(
            attn_output.contiguous(),
            self.o_proj.weight.transpose(0, 1),
            self._get_welm_npu_o_proj_hcom_name(),
            self.o_proj.tp_size,
            reduce_op="sum",
            bias=bias,
            comm_turn=0,
            comm_mode="ccu",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_o_norm: bool = False,
        skip_o_proj_all_reduce: bool = False,
        use_o_proj_matmul_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        external_mirror = (
            _get_welm_mtp_mirror_state(forward_batch, self.kv_mirror_layer_idx)
            if self.is_nextn
            else None
        )
        frozen_mtp_decode = (
            self.is_nextn
            and forward_batch.forward_mode.is_decode()
            and bool(getattr(forward_batch.spec_info, "welmv4_mtp_frozen_kv", False))
        )
        if (
            self.is_nextn
            and hidden_states.shape[0] > 0
            and external_mirror is None
            and not frozen_mtp_decode
        ):
            raise RuntimeError(
                "WeLMV4 NextN requires external source0 mirror K/V during "
                "draft-extend, or an explicitly frozen KV snapshot during "
                "active draft decode."
            )

        if frozen_mtp_decode:
            if hasattr(self, "kv_mirror_query_proj"):
                q, _ = self.kv_mirror_query_proj(hidden_states)
            elif getattr(self, "_kv_mirror_mxfp8_query_projection", False):
                q, _ = self.qkv_proj(hidden_states)
            else:
                q = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
            k = v = None
        elif external_mirror is not None:
            k, v = external_mirror
            mirror_indices = getattr(
                forward_batch.spec_info, "mirrored_kv_indices", None
            )
            if mirror_indices is not None:
                safe_indices = mirror_indices.to(torch.int64).clamp(min=0)
                k = k[safe_indices]
                v = v[safe_indices]
            elif forward_batch.forward_mode.is_extend_without_speculative():
                # Target DP+EP ordinary prefill uses one group-wide MAX token
                # slot for DeepEP NORMAL, so source0's exported mirror K/V may
                # include a padding suffix that is not present in this draft
                # replica's local prefill layout.  ``positions`` is the
                # physical T-row contract for the draft attention/KV-cache
                # write; discard only that target-owned suffix before RoPE.
                # Decode/verify/draft-extend use mirrored_kv_indices (or a
                # frozen snapshot) and must retain their existing geometry.
                mirror_rows = int(k.shape[0])
                position_rows = int(positions.numel())
                if int(v.shape[0]) != mirror_rows:
                    raise RuntimeError(
                        "WeLMv4 external mirror K/V row mismatch: "
                        f"K={mirror_rows}, V={v.shape[0]}."
                    )
                if mirror_rows < position_rows:
                    raise RuntimeError(
                        "WeLMv4 external mirror K/V has fewer rows than draft "
                        f"prefill positions: K/V={mirror_rows}, "
                        f"positions={position_rows}."
                    )
                if mirror_rows > position_rows:
                    k = k.narrow(0, 0, position_rows)
                    v = v.narrow(0, 0, position_rows)
            if hasattr(self, "kv_mirror_query_proj"):
                q, _ = self.kv_mirror_query_proj(hidden_states)
            elif getattr(self, "_kv_mirror_mxfp8_query_projection", False):
                q, _ = self.qkv_proj(hidden_states)
            else:
                q = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
        elif self.kv_mirror_layer_idx in self.kv_mirror_imitated_layers:
            if getattr(self, "_kv_mirror_mxfp8_source_projection", False):
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v, mirror_k, mirror_v = qkv.split(
                    [
                        self.q_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                    ],
                    dim=-1,
                )
                consumer_layer_id = self.kv_mirror_layers[
                    self.kv_mirror_imitated_layers.index(self.kv_mirror_layer_idx)
                ]
                if consumer_layer_id >= LayerManager.num_target_layers:
                    _set_welm_mtp_mirror_state(
                        forward_batch, consumer_layer_id, mirror_k, mirror_v
                    )
                else:
                    KVMirrorManager.set_kv_activation(
                        self.kv_mirror_layer_idx, (mirror_k, mirror_v)
                    )
            elif hasattr(self, "qkv_proj_weight"):
                qkv = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
                q, k, v, mirror_k, mirror_v = qkv.split(
                    [
                        self.q_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                    ],
                    dim=-1,
                )
                consumer_layer_id = self.kv_mirror_layers[
                    self.kv_mirror_imitated_layers.index(self.kv_mirror_layer_idx)
                ]
                if consumer_layer_id >= LayerManager.num_target_layers:
                    _set_welm_mtp_mirror_state(
                        forward_batch, consumer_layer_id, mirror_k, mirror_v
                    )
                else:
                    KVMirrorManager.set_kv_activation(
                        self.kv_mirror_layer_idx, (mirror_k, mirror_v)
                    )
            else:
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        elif self.kv_mirror_layer_idx in self.kv_mirror_layers:
            if (
                self.kv_mirror_layer_idx in LayerManager.num_nextn_predict_layer_idx
                and not forward_batch.forward_mode.is_extend_without_speculative()
            ):
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            else:
                mirror_layer_number = self.kv_mirror_imitated_layers[
                    self.kv_mirror_layers.index(self.kv_mirror_layer_idx)
                ]
                k, v = KVMirrorManager.get_kv_activation(
                    mirror_layer_number, clear=self.need_clear_kv_cache
                )
                if hasattr(self, "kv_mirror_query_proj"):
                    q, _ = self.kv_mirror_query_proj(hidden_states)
                elif getattr(self, "_kv_mirror_mxfp8_query_projection", False):
                    q, _ = self.qkv_proj(hidden_states)
                else:
                    q = F.linear(
                        hidden_states, self.qkv_proj_weight, self.qkv_proj_bias
                    )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        gate = None
        is_prefill_batch = (
            forward_batch is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
                include_draft_extend_v2=True
            )
        )
        is_kv_mirror_prefill = (
            forward_batch is not None
            and forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.kv_mirror_layer_idx in self.kv_mirror_layers
        )
        # Preserve the existing gate stream policy for non-prefill modes and
        # additionally let mirror prefill use that same policy.
        use_decode_like_stream_policy = (
            not is_prefill_batch or is_kv_mirror_prefill
        )
        enable_npu_gate_alt_stream = (
            _is_npu
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and self.alt_stream is not None
            and self.gated_self_attention_headwise
            and hidden_states.shape[0] > 0
            and use_decode_like_stream_policy
        )
        if enable_npu_gate_alt_stream:
            device_module = torch.get_device_module()
            current_stream = device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with device_module.stream(self.alt_stream):
                gate = self.gate_proj(hidden_states)[0].unsqueeze(-1)

        q_shape = q.shape
        k_shape = None if k is None else k.shape

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if self.q_norm is not None:
            q_by_head, _ = self.q_norm(q_by_head)
        # Q is a strided view of the fused QKV projection when q_norm is
        # disabled.  Attention requires packed Q later, so materialize that
        # layout before RoPE and let downstream contiguous() calls be no-ops.
        q = q_by_head.view(q.shape).contiguous()

        if k is not None:
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            if self.k_norm is not None:
                k_by_head = mmq_style_k_rms_norm(
                    k_by_head.contiguous(),
                    self.k_norm.weight,
                    self.k_norm.eps,
                )
            k = k_by_head.view(k.shape)

        # A single non-speculative extend request is globally contiguous;
        # ordinary multi-request prefill carries independently contiguous
        # segment metadata.  Speculative, decode, and mirror-consumer paths
        # retain the generic kernel.
        positions_are_contiguous = (
            forward_batch.batch_size == 1
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
        segment_tile_starts = getattr(
            forward_batch, "welmv4_rope_segment_tile_starts", None
        )

        qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if k is None:
            # WeLM RoPE updates Q and K in place. They must not alias: passing
            # Q as both operands would rotate the same storage twice on NPU.
            unused_key = q.clone()
            q, _ = self.rotary_emb(
                positions,
                q,
                unused_key,
                positions_are_contiguous=positions_are_contiguous,
                segment_tile_starts=segment_tile_starts,
            )
            q = q.view(q_shape)
        elif qk_nope_head_dim > 0:
            is_kv_mirror_last_query = (
                forward_batch.enable_kv_mirror
                and forward_batch.forward_mode.is_extend_without_speculative()
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            )
            if is_kv_mirror_last_query:
                custom_last_index = getattr(
                    forward_batch, "custom_last_index", None
                )
                if custom_last_index is None:
                    raise RuntimeError(
                        "WeLMv4 KV-mirror last-query RoPE requires "
                        "custom_last_index."
                    )
                if (
                    q.shape[0] != custom_last_index.numel()
                    or k.shape[0] != positions.numel()
                ):
                    raise RuntimeError(
                        "WeLMv4 KV-mirror RoPE requires Q=B and K/positions=T, "
                        f"got Q={q.shape[0]}, B={custom_last_index.numel()}, "
                        f"K={k.shape[0]}, T={positions.numel()}."
                    )
                # Both target mirror consumers and the physical NextN layer
                # keep full prompt K/V but contract Q to one row per request.
                # Rotate Q at the request-tail positions and K at every
                # original prompt position.
                q, k = self.rotary_emb(
                    positions,
                    q,
                    k,
                    last_index=custom_last_index,
                    positions_are_contiguous=positions_are_contiguous,
                    segment_tile_starts=segment_tile_starts,
                )
            else:
                q, k = self.rotary_emb(
                    positions,
                    q,
                    k,
                    positions_are_contiguous=positions_are_contiguous,
                    segment_tile_starts=segment_tile_starts,
                )
            q = q.view(q_shape)
            k = k.view(k_shape)
        else:
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                positions_are_contiguous=positions_are_contiguous,
                segment_tile_starts=segment_tile_starts,
            )

        attn_kwargs = {}
        if self.attn_sink is not None:
            attn_kwargs["sinks"] = self.attn_sink
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not frozen_mtp_decode,
            **attn_kwargs,
        )
        if self.gated_self_attention_headwise:
            attn_shape = attn_output.shape
            if gate is None:
                gate = self.gate_proj(hidden_states)[0].unsqueeze(-1)
            # gate: (bs * seq_len, num_heads, 1)
            attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
            if enable_npu_gate_alt_stream:
                # Normal inference first consumes gate in sigmoid_mul. Joining
                # here lets gate projection overlap the complete attention op.
                current_stream.wait_stream(self.alt_stream)
            if _is_npu:
                inplace_sigmoid_mul_npu(gate, attn_output)
            else:
                inplace_sigmoid_mul(gate, attn_output)
            attn_output = attn_output.view(attn_shape)

        if use_o_proj_matmul_reduce_scatter:
            output = self._npu_o_proj_matmul_reduce_scatter(attn_output)
        else:
            output, _ = self.o_proj(
                attn_output,
                skip_all_reduce=skip_o_proj_all_reduce,
            )
        if self.o_norm is not None and not skip_o_norm:
            output, _ = self.o_norm(output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[Any] = None,
        is_nextn: bool = False,
        config_layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.physical_layer_id = int(layer_id)
        self.config_layer_id = int(
            layer_id if config_layer_id is None else config_layer_id
        )
        self.is_nextn = is_nextn
        self.hidden_size = config.hidden_size
        # Transformers v5 standardizes legacy rope_scaling into
        # rope_parameters and renames ``type`` to ``rope_type``. Keep the old
        # field as a fallback for remote configs based on Transformers v4.
        rope_scaling = getattr(config, "rope_parameters", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = (
            rope_scaling.get("rope_theta", getattr(config, "rope_theta", 10000))
            if isinstance(rope_scaling, dict)
            else getattr(config, "rope_theta", 10000)
        )
        if isinstance(rope_scaling, dict) and (
            rope_scaling.get("rope_type") or rope_scaling.get("type")
        ) == "default":
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        scale_seq_times = getattr(config, "scale_seq_times", 0)
        if scale_seq_times > 0:
            max_position_embeddings = max_position_embeddings * (scale_seq_times + 1)
        if getattr(config, "qkv_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_bias")
        elif getattr(config, "qkv_proj_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_proj_bias")
        else:
            qkv_bias = True
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        qk_norm = getattr(config, "qk_norm", False)
        k_norm = getattr(config, "k_norm", False)
        out_bias = getattr(config, "out_proj_bias", False)
        head_dim = getattr(
            config, "head_dim", self.hidden_size // config.num_attention_heads
        )
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", head_dim)

        self.kv_mirror_layers = getattr(config, "kv_mirror_layers", [])
        self.kv_mirror_imitated_layers = getattr(
            config, "kv_mirror_imitated_layers", []
        )
        # Query pruning begins at the first mirror consumer encountered by the
        # ascending decoder loop.  Valid target consumers form a contiguous
        # suffix, so every later layer also obtains full K/V from its source.
        self.first_target_kv_mirror_layer = min(
            (
                int(mirror_layer_id)
                for mirror_layer_id in self.kv_mirror_layers
                if 0 <= int(mirror_layer_id) < int(config.num_hidden_layers)
            ),
            default=None,
        )
        self.sliding_window_size = get_welmv4_layerwise_sliding_windows(
            config,
            context_len=getattr(config, "context_len", None),
            num_layers=1,
            layer_offset=self.config_layer_id,
        )[0]
        self.enable_attn_sink_layerwise = getattr(
            config, "enable_attn_sink_layerwise", []
        )
        self.ppln = getattr(config, "ppln", False)
        o_norm = getattr(config, "o_norm", False)
        self.prenorm_layer_idx = getattr(config, "prenorm_layer_idx", [])
        logger.debug(
            "WeLMv4 layer %s: ppln=%s, o_norm=%s, prenorm_layer_idx=%s",
            self.config_layer_id,
            self.ppln,
            o_norm,
            self.prenorm_layer_idx,
        )
        total_layer_num = config.num_hidden_layers

        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=self.physical_layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            qk_norm=qk_norm,
            k_norm=k_norm,
            qk_rope_head_dim=qk_rope_head_dim,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            prefix=add_prefix("self_attn", prefix),
            kv_mirror_layers=self.kv_mirror_layers,
            kv_mirror_imitated_layers=self.kv_mirror_imitated_layers,
            sliding_window_size=self.sliding_window_size,
            enable_attn_sink_layerwise=self.enable_attn_sink_layerwise,
            layer_idx=self.config_layer_id,
            o_norm=o_norm and self.config_layer_id not in self.prenorm_layer_idx,
            rms_norm_eps=config.rms_norm_eps,
            total_layer_num=total_layer_num,
            is_nextn=is_nextn,
            alt_stream=alt_stream,
        )
        LayerManager.num_nextn_predict_layers = getattr(
            config, "num_nextn_predict_layers", 0
        )
        self.layer_id = self.config_layer_id
        self.is_final_layer = self.physical_layer_id == total_layer_num - 1 or is_nextn

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        # Qwen2MoE all layers are sparse (include nextn layers)
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=self.physical_layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=True,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=self.config_layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
                is_nextn=self.is_nextn,
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = WelmV4CommunicatorRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = WelmV4CommunicatorRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.final_mlp_experts_output: Optional[torch.Tensor] = None
        self.final_mlp_shared_output: Optional[torch.Tensor] = None

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=self.is_final_layer,
        )
        self._welm_runner_plan = get_welm_runner_build_plan_for_init()
        self._welm_dp_executor = None
        if self._welm_runner_plan is not None:
            if self._welm_runner_plan.role is WelmRunnerRole.TARGET_DP or (
                self._welm_runner_plan.role is WelmRunnerRole.DRAFT
                and self._welm_runner_plan.has_moe_ep
            ):
                self._welm_dp_executor = WelmDpAttentionExecutor(
                    self._welm_runner_plan
                )
        LayerManager.set_decoder_layer(self.self_attn.kv_mirror_layer_idx, self)

    def bind_finalized_runner_plan(
        self, runner_plan: WelmRunnerParallelPlan
    ) -> None:
        """Publish the finalized immutable plan before forward/Graph capture."""

        if self._welm_runner_plan is None:
            raise RuntimeError(
                "Cannot bind a WeLMv4 DP plan to a layer constructed without one"
            )
        if self._welm_dp_executor is None:
            # DRAFT without physical MTP EP intentionally stays on the exact
            # non-DP implementation.
            if runner_plan.role is not WelmRunnerRole.DRAFT or runner_plan.has_moe_ep:
                raise RuntimeError("WeLMv4 finalized plan requires a missing executor")
        else:
            self._welm_dp_executor.bind_finalized_runner_plan(runner_plan)
        self.mlp.welm_runner_plan = runner_plan
        self._welm_runner_plan = runner_plan

    def _use_npu_prefill_deepep_scattered(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> bool:
        # This target-only fast path keeps prompt rows LOCAL between
        # physical decoder layers. NextN has one physical layer and its
        # input has already been restored to FULL by the target model;
        # its first-layer residual is therefore legitimately None.
        tp_size = get_tensor_model_parallel_world_size()
        return (
            _is_npu
            and not self.is_nextn
            and get_moe_a2a_backend().is_deepep()
            and forward_batch.forward_mode.is_extend_without_speculative()
            and tp_size > 1
            and get_parallel().moe_ep_size == tp_size
            and self.attn_tp_size == tp_size
            and getattr(self.self_attn.o_proj, "tp_size", tp_size) == tp_size
            and hidden_states.shape[0] > 0
        )

    @staticmethod
    def _all_gather_tp_rows(local_tensor: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        gathered = local_tensor.new_empty(
            (local_tensor.shape[0] * tp_size, *local_tensor.shape[1:])
        )
        get_tp_group().all_gather_into_tensor(gathered, local_tensor.contiguous())
        return gathered

    def _npu_prefill_deepep_prepare_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        residual_after_layernorm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None or hidden_states.shape != residual.shape:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP scattered input requires matching LOCAL "
                f"hidden/residual tensors, got {tuple(hidden_states.shape)} and "
                f"{None if residual is None else tuple(residual.shape)}"
            )
        if residual_after_layernorm:
            local_hidden_states, _, local_residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=True,
                clone_fp32_out=True,
            )
        else:
            local_hidden_states, local_residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=False,
            )
        if local_residual.dtype != torch.float32:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP scattered residual must remain FP32, got "
                f"{local_residual.dtype}"
            )
        return self._all_gather_tp_rows(local_hidden_states), local_residual

    def _npu_prefill_deepep_finish_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        use_mmq_norm_after_attn: bool,
        input_is_reduce_scattered: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() != 2 or residual is None or residual.dim() != 2:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP attention output requires 2D hidden/residual"
            )
        tp_size = get_tensor_model_parallel_world_size()
        if input_is_reduce_scattered:
            local_tokens, hidden_size = hidden_states.shape
            num_tokens = local_tokens * tp_size
            local_hidden_states = hidden_states.contiguous()
        else:
            num_tokens, hidden_size = hidden_states.shape
            if num_tokens % tp_size != 0:
                raise RuntimeError(
                    "Prefill token padding must be completed before OProj "
                    f"ReduceScatter: {num_tokens} tokens are not divisible by "
                    f"TP{tp_size}"
                )
            local_tokens = num_tokens // tp_size
            local_hidden_states = hidden_states.new_empty(
                (local_tokens, hidden_size)
            )
            get_tp_group().reduce_scatter_tensor(
                local_hidden_states, hidden_states.contiguous()
            )

        global_shape = (num_tokens, hidden_size)
        local_shape = (local_tokens, hidden_size)
        if residual.shape == global_shape:
            tp_rank = get_parallel().tp_rank
            local_residual = residual.narrow(
                0, tp_rank * local_tokens, local_tokens
            ).contiguous()
        elif residual.shape == local_shape:
            local_residual = residual.contiguous()
        else:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP expected FULL or LOCAL residual, got "
                f"{tuple(residual.shape)} for FULL/LOCAL shapes "
                f"{global_shape}/{local_shape}"
            )
        if local_residual.dtype != torch.float32:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP scattered residual must be FP32 before "
                f"post-attention norm, got {local_residual.dtype}"
            )

        if use_mmq_norm_after_attn:
            return mmq_style_norm_after_attn(
                local_hidden_states,
                local_residual,
                self.self_attn.o_norm.weight,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
                return_fp32_out=False,
            )

        if self.self_attn.o_norm is not None:
            local_hidden_states, _ = self.self_attn.o_norm(local_hidden_states)
        return self.post_attention_layernorm(
            local_hidden_states,
            local_residual,
            clone_fp32_out=True,
        )

    def _npu_prefill_replicate_kv_mirror_residual(
        self,
        residual: torch.Tensor,
        custom_last_index: torch.Tensor,
        *,
        prompt_num_padded_rows: int,
        mirror_num_real_rows: int,
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_parallel().tp_rank
        if prompt_num_padded_rows % tp_size != 0:
            raise RuntimeError("KV Mirror prompt layout is not divisible by TP")
        prompt_local_rows = prompt_num_padded_rows // tp_size
        if residual.shape[0] != prompt_local_rows or residual.dtype != torch.float32:
            raise RuntimeError(
                "KV Mirror owner redistribution requires LOCAL FP32 residual "
                f"[{prompt_local_rows}, D], got {tuple(residual.shape)} "
                f"{residual.dtype}"
            )

        residual_partial = self._build_kv_mirror_residual_partial(
            residual,
            custom_last_index,
            prompt_local_rows=prompt_local_rows,
            mirror_num_real_rows=mirror_num_real_rows,
            mirror_num_padded_rows=mirror_num_real_rows,
            tp_rank=tp_rank,
        )
        # Each request's last prompt token is owned by exactly one prompt
        # shard.  Sum the disjoint owner rows to reconstruct the exact [B, D]
        # FP32 residual on every TP/EP rank.  Do not index a LOCAL residual
        # with a global prompt position and do not pad B for this FULL layout.
        return get_tp_group().all_reduce(residual_partial.contiguous())

    @staticmethod
    def _build_kv_mirror_residual_partial(
        residual: torch.Tensor,
        custom_last_index: torch.Tensor,
        *,
        prompt_local_rows: int,
        mirror_num_real_rows: int,
        mirror_num_padded_rows: int,
        tp_rank: int,
    ) -> torch.Tensor:
        owner_rank = torch.div(
            custom_last_index, prompt_local_rows, rounding_mode="floor"
        )
        local_offset = custom_last_index - tp_rank * prompt_local_rows
        owned = owner_rank == tp_rank

        # Keep every intermediate shape fixed at [B] or [B, D].  Boolean
        # indexing such as local_offset[owned] has a data-dependent output
        # shape and forces the host to wait for the NPU to count the selected
        # rows, draining the queued non-mirror prefill work at the first mirror
        # layer.  Non-owner ranks use a valid dummy offset and are masked back
        # to exact FP32 zero before the owner-partial all-reduce.
        safe_local_offset = torch.where(
            owned,
            local_offset,
            torch.zeros_like(local_offset),
        ).to(torch.long)
        selected = residual.index_select(0, safe_local_offset)
        selected = torch.where(
            owned[:, None],
            selected,
            torch.zeros_like(selected),
        )

        if mirror_num_padded_rows == mirror_num_real_rows:
            return selected

        residual_partial = residual.new_zeros(
            (mirror_num_padded_rows, residual.shape[1])
        )
        residual_partial.narrow(0, 0, mirror_num_real_rows).copy_(selected)
        return residual_partial

    @staticmethod
    def _update_pure_tp_kv_mirror_full_metadata(
        forward_batch: ForwardBatch,
        *,
        mirror_num_real_rows: int,
    ) -> None:
        forward_batch.kv_mirror_num_real_rows = mirror_num_real_rows
        forward_batch.kv_mirror_num_padded_rows = mirror_num_real_rows
        forward_batch.kv_mirror_local_num_tokens = mirror_num_real_rows
        forward_batch.global_dp_buffer_len = mirror_num_real_rows
        if forward_batch.global_num_tokens_cpu is not None:
            if len(forward_batch.global_num_tokens_cpu) != 1:
                raise RuntimeError(
                    "WeLMv4 FULL KV Mirror currently supports pure TP only"
                )
            forward_batch.global_num_tokens_cpu = [mirror_num_real_rows]
        if forward_batch.global_num_tokens_gpu is not None:
            if forward_batch.global_num_tokens_gpu.numel() != 1:
                raise RuntimeError(
                    "WeLMv4 FULL KV Mirror currently supports pure TP only"
                )
            forward_batch.global_num_tokens_gpu.fill_(mirror_num_real_rows)
        if forward_batch.num_token_non_padded is None:
            raise RuntimeError(
                "WeLMv4 NPU DeepEP requires num_token_non_padded metadata"
            )
        forward_batch.num_token_non_padded.fill_(mirror_num_real_rows)
        forward_batch.num_token_non_padded_cpu = mirror_num_real_rows
        forward_batch.welmv4_npu_deepep_scattered = False
        forward_batch.welmv4_npu_deepep_full_mirror = True

    @staticmethod
    def _get_kv_mirror_prefill_ll_capacity() -> int:
        mirror_capacity = (
            envs.SGLANG_WELMV4_MIRROR_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        if mirror_capacity is not None:
            return mirror_capacity
        return envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._welm_dp_executor is None:
            return self._forward_non_dp(
                positions, hidden_states, forward_batch, residual
            )
        return self._welm_dp_executor.forward(
            layer=self,
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
        )

    def _forward_non_dp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the DP-disabled WeLM path.

        Enabling DP attention requires a ``WelmDpAttentionExecutor`` and is
        handled by the static dispatch in ``forward``.  Keep this function free
        of DP layout inference, collectives, and metadata mutation.
        """

        residual_after_layernorm = (
            self.ppln and self.layer_id not in self.prenorm_layer_idx
        )
        use_npu_prefill_deepep_scattered = (
            self._use_npu_prefill_deepep_scattered(hidden_states, forward_batch)
        )
        is_kv_mirror_prefill = (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.first_target_kv_mirror_layer is not None
        )
        is_first_kv_mirror_consumer = (
            is_kv_mirror_prefill
            and self.layer_id == self.first_target_kv_mirror_layer
        )
        already_full_mirror = (
            is_kv_mirror_prefill
            and forward_batch.welmv4_npu_deepep_full_mirror
        )
        # The prompt prefix stays LOCAL between layers.  The first mirror
        # consumer receives a LOCAL input, selects one row per request after
        # attention, and changes the persistent layout to FULL.  Every later
        # mirror layer consumes and produces the same exact [B, D] FULL rows.
        input_hidden_is_scattered = (
            use_npu_prefill_deepep_scattered
            and self.layer_id > 0
            and not already_full_mirror
        )
        output_hidden_is_scattered = (
            use_npu_prefill_deepep_scattered
            and not is_first_kv_mirror_consumer
            and not already_full_mirror
        )
        use_full_mirror_layout = (
            use_npu_prefill_deepep_scattered
            and is_kv_mirror_prefill
            and (is_first_kv_mirror_consumer or already_full_mirror)
        )
        if use_npu_prefill_deepep_scattered and not already_full_mirror:
            if forward_batch.num_token_non_padded is None:
                raise RuntimeError(
                    "WeLMv4 NPU DeepEP scattered prefill requires localized "
                    "num_token_non_padded metadata"
                )
            if output_hidden_is_scattered:
                forward_batch.welmv4_npu_deepep_scattered = True
        enable_npu_weight_prefetch = (
            _is_npu and hidden_states.shape[0] > 0
        )
        qkv_prefetch_started = False
        if enable_npu_weight_prefetch:
            qkv_prefetch_weight = self.self_attn.get_qkv_prefetch_weight(
                forward_batch
            )
            prepare_weight_cache(hidden_states, qkv_prefetch_weight)
            qkv_prefetch_started = True
        if input_hidden_is_scattered:
            hidden_states, residual = self._npu_prefill_deepep_prepare_attention(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
            )
        elif residual_after_layernorm:
            # 纯TP layer0 layer2-47
            hidden_states, _, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
                clone_fp32_out=True,
                output_dtype=self.input_layernorm.weight.dtype
                if hidden_states.dtype == torch.float32
                else hidden_states.dtype,
            )
        else:
            # 纯TP layer1
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
            )
        if qkv_prefetch_started:
            # Keep CMO overlap bounded to input RMSNorm; QKV starts afterwards.
            wait_cmo_stream()
        # use_mmq_norm_after_attn 纯TP 0+2-47 layer 为True
        use_mmq_norm_after_attn = residual_after_layernorm and self.self_attn.use_o_norm

        prompt_num_padded_rows = None
        if is_first_kv_mirror_consumer:
            custom_last_index = getattr(forward_batch, "custom_last_index", None)
            if custom_last_index is None:
                custom_last_index = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                forward_batch.custom_last_index = custom_last_index
            prompt_num_padded_rows = hidden_states.shape[0]
            hidden_states = hidden_states.index_select(
                0, custom_last_index.to(torch.long)
            )
        use_npu_prefill_oproj_matmul_reduce_scatter = (
            envs.SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER.get()
            and output_hidden_is_scattered
            and not use_full_mirror_layout
            and hidden_states.shape[0] % get_tensor_model_parallel_world_size()
            == 0
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                skip_o_norm=(
                    use_mmq_norm_after_attn
                    or output_hidden_is_scattered
                ),
                skip_o_proj_all_reduce=output_hidden_is_scattered,
                use_o_proj_matmul_reduce_scatter=(
                    use_npu_prefill_oproj_matmul_reduce_scatter
                ),
            )
        if is_first_kv_mirror_consumer:
            mirror_num_real_rows = int(forward_batch.batch_size)
            if forward_batch.custom_last_index.numel() != mirror_num_real_rows:
                raise RuntimeError(
                    "KV Mirror request count does not match custom_last_index: "
                    f"{mirror_num_real_rows} != "
                    f"{forward_batch.custom_last_index.numel()}"
                )
            if use_full_mirror_layout:
                if input_hidden_is_scattered:
                    residual = self._npu_prefill_replicate_kv_mirror_residual(
                        residual,
                        forward_batch.custom_last_index,
                        prompt_num_padded_rows=prompt_num_padded_rows,
                        mirror_num_real_rows=mirror_num_real_rows,
                    )
                else:
                    residual = residual.index_select(
                        0, forward_batch.custom_last_index.to(torch.long)
                    )
                if hidden_states.shape[0] != mirror_num_real_rows:
                    raise RuntimeError(
                        "KV Mirror FULL attention rows do not match request count: "
                        f"{hidden_states.shape[0]} != {mirror_num_real_rows}"
                    )
                self._update_pure_tp_kv_mirror_full_metadata(
                    forward_batch,
                    mirror_num_real_rows=mirror_num_real_rows,
                )
            else:
                residual = residual.index_select(
                    0, forward_batch.custom_last_index.to(torch.long)
                )

        router_prefetch_started = False
        if enable_npu_weight_prefetch and hidden_states.shape[0] > 0:
            router_compute_weight = self.mlp.get_npu_router_compute_weight_t()
            prepare_weight_cache(hidden_states, router_compute_weight)
            router_prefetch_started = True
        if output_hidden_is_scattered:
            hidden_states, residual, hidden_states_fp32 = (
                self._npu_prefill_deepep_finish_attention(
                    hidden_states,
                    residual,
                    use_mmq_norm_after_attn=use_mmq_norm_after_attn,
                    input_is_reduce_scattered=(
                        use_npu_prefill_oproj_matmul_reduce_scatter
                    ),
                )
            )
        elif use_mmq_norm_after_attn:
            hidden_states, residual, hidden_states_fp32 = mmq_style_norm_after_attn(
                hidden_states,
                residual,
                self.self_attn.o_norm.weight,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
                return_fp32_out=False,
            )
        else:
            (
                hidden_states,
                residual,
                hidden_states_fp32,
            ) = self.post_attention_layernorm(
                hidden_states, residual, clone_fp32_out=True
            )
        if router_prefetch_started:
            # Keep CMO overlap bounded to post-attention RMSNorm.
            wait_cmo_stream()
        # The prefix's normal-AllGather DeepEP returns a complete LOCAL token
        # shard.  The mirror suffix's LL DeepEP returns one complete FULL
        # request-row set.  Neither layout needs a second framework RS.
        use_reduce_scatter = (
            False
            if output_hidden_is_scattered or use_full_mirror_layout
            else self.layer_communicator.should_use_reduce_scatter(forward_batch)
        )
        self.final_mlp_experts_output = None
        self.final_mlp_shared_output = None
        has_full_replicated_moe_input = (
            forward_batch.welmv4_npu_deepep_full_mirror
            or forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
            or (
                self.is_nextn
                and (
                    forward_batch.forward_mode.is_extend_without_speculative()
                    or forward_batch.forward_mode.is_draft_extend_v2()
                )
            )
        )
        use_welm_local_ep_moe = (
            self.mlp.supports_welm_local_ep_moe
            and has_full_replicated_moe_input
        )
        if use_welm_local_ep_moe and output_hidden_is_scattered:
            raise RuntimeError(
                "WeLMv4 local EP MoE cannot consume a scattered token layout."
            )
        use_kv_mirror_prefill_ll = (
            use_full_mirror_layout
            and forward_batch.welmv4_npu_deepep_full_mirror
            and not use_welm_local_ep_moe
        )
        # Spec-V2 DRAFT_EXTEND_V2 is a multi-token extend even though it is
        # launched from a decode ScheduleBatch.  In the eager path the copied
        # ``is_extend_in_batch`` flag therefore still selects DeepEP LL.  That
        # disagrees with both the padded-row routing above (normal mode uses
        # expert id -1 for dummy rows) and the draft-extend graph adapter,
        # which captures DeepEP as an extend batch.  Keep the correction local
        # to the WeLM NextN physical MTP layer. The local-EP path does not
        # dispatch through DeepEP, but retains this override because TopK uses
        # it to select the correct graph-padding id rule.
        use_nextn_draft_extend_normal = (
            _is_npu
            and self.is_nextn
            and forward_batch.forward_mode.is_draft_extend_v2()
        )
        mirror_ll_capacity = None
        if use_kv_mirror_prefill_ll:
            mirror_ll_capacity = self._get_kv_mirror_prefill_ll_capacity()
            if hidden_states.shape[0] > mirror_ll_capacity:
                raise RuntimeError(
                    "WeLMv4 mirror prefill LL capacity is smaller than the "
                    f"FULL request-row count: {mirror_ll_capacity} < "
                    f"{hidden_states.shape[0]}"
                )
        with get_forward().scoped(
            deepep_mode_override=(
                DeepEPMode.LOW_LATENCY
                if use_kv_mirror_prefill_ll
                else (
                    DeepEPMode.NORMAL if use_nextn_draft_extend_normal else None
                )
            ),
            deepep_num_max_dispatch_tokens_override=mirror_ll_capacity,
        ):
            mlp_output = self.mlp(
                hidden_states,
                hidden_states_fp32,
                forward_batch,
                use_reduce_scatter,
                use_welm_local_ep_moe=use_welm_local_ep_moe,
                return_components=self.is_final_layer,
                skip_component_output=(
                    self.is_final_layer
                    and residual is not None
                    and getattr(self.mlp, "tp_size", 1) == 1
                ),
            )
        experts_output = None
        shared_output = None
        if isinstance(mlp_output, tuple):
            hidden_states, experts_output, shared_output = mlp_output
        else:
            hidden_states = mlp_output

        if self.is_final_layer:
            self.final_mlp_experts_output = experts_output
            self.final_mlp_shared_output = shared_output
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config
        # Some WeLMv4 remote configs do not define pad_token_id.  Padding is
        # handled by SGLang's request metadata, so keep this optional just as
        # the mainline Qwen2 implementation does.
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        kv_mirror_layers = list(getattr(config, "kv_mirror_layers", []) or [])
        kv_mirror_imitated_layers = list(
            getattr(config, "kv_mirror_imitated_layers", []) or []
        )
        if len(kv_mirror_layers) != len(kv_mirror_imitated_layers):
            raise ValueError(
                "WeLMv4 kv_mirror_layers and kv_mirror_imitated_layers must "
                "have the same length."
            )
        if len(set(kv_mirror_layers)) != len(kv_mirror_layers) or len(
            set(kv_mirror_imitated_layers)
        ) != len(kv_mirror_imitated_layers):
            raise ValueError("WeLMv4 KV-mirror layer lists must contain unique IDs.")
        if set(kv_mirror_layers) & set(kv_mirror_imitated_layers):
            raise ValueError(
                "A WeLMv4 layer cannot be both a KV-mirror source and consumer."
            )
        num_nextn_predict_layers = int(
            getattr(config, "num_nextn_predict_layers", 0) or 0
        )
        if num_nextn_predict_layers < 0:
            raise ValueError("WeLMv4 num_nextn_predict_layers must be non-negative.")
        # KV-mirror IDs use one logical namespace.  Target layers occupy
        # [0, num_hidden_layers), followed by optional NextN/MTP layers.  A
        # target-only server therefore legitimately sees (and later ignores)
        # pairs whose consumer is the first MTP layer, e.g. 0 -> 48 for a
        # 48-layer target model.
        num_logical_layers = config.num_hidden_layers + num_nextn_predict_layers
        for mirror_layer, imitated_layer in zip(
            kv_mirror_layers, kv_mirror_imitated_layers
        ):
            if not (
                0 <= imitated_layer < mirror_layer < num_logical_layers
            ):
                raise ValueError(
                    "Each WeLMv4 KV-mirror pair must satisfy "
                    f"0 <= source ({imitated_layer}) < consumer "
                    f"({mirror_layer}) < num_logical_layers "
                    f"({num_logical_layers} = {config.num_hidden_layers} target + "
                    f"{num_nextn_predict_layers} NextN/MTP)."
                )
        # Keep the potentially very large base/OE tables in mapped pinned host
        # memory when explicitly requested. CUDA uses UVA; Ascend registers a
        # page-aligned ACL host allocation and presents its dev_ptr as an NPU
        # tensor after a one-time F.embedding compatibility probe.
        self.use_host_embeddings = (
            (_is_cuda or _is_npu)
            and get_global_server_args().enable_over_encoding
        )
        self.scale_seq_times = getattr(config, "scale_seq_times", 0)
        if self.scale_seq_times > 0:
            raise NotImplementedError(
                "The rebased WeLMv4 path does not yet support scale_seq_times > 0. "
                "The current WeLMv4 checkpoint uses scale_seq_times=0; silently "
                "running an expanded-sequence checkpoint would corrupt KV-cache "
                "allocation and attention metadata."
            )

        if len(self.oe_vocab_sizes) > 0:
            self.oe_embed = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        self.oe_vocab_sizes[i],
                        self.oe_dim,
                        use_attn_tp_group=is_dp_attention_enabled(),
                        host_tensor=self.use_host_embeddings,
                    )
                    for i in range(len(self.oe_vocab_sizes))
                ]
            )
            self.oe_gate_up_proj = ReplicatedLinear(
                self.oe_dim * len(self.oe_vocab_sizes),
                config.hidden_size,
                bias=False,
                quant_config=None,
            )

        # Scale sequence length embeddings: N additional embedding groups
        if self.scale_seq_times > 0:
            self.scale_seq_embed_tokens_list = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        config.hidden_size,
                        use_attn_tp_group=is_dp_attention_enabled(),
                        host_tensor=self.use_host_embeddings,
                    )
                    for _ in range(self.scale_seq_times)
                ]
            )
            if len(self.oe_vocab_sizes) > 0:
                self.scale_seq_oe_embed_list = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                VocabParallelEmbedding(
                                    self.oe_vocab_sizes[j],
                                    self.oe_dim,
                                    use_attn_tp_group=is_dp_attention_enabled(),
                                    host_tensor=self.use_host_embeddings,
                                )
                                for j in range(len(self.oe_vocab_sizes))
                            ]
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )
                self.scale_seq_oe_up_proj_list = nn.ModuleList(
                    [
                        ReplicatedLinear(
                            self.oe_dim * len(self.oe_vocab_sizes),
                            config.hidden_size,
                            bias=False,
                            quant_config=None,
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                host_tensor=self.use_host_embeddings,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        LayerManager.num_target_layers = config.num_hidden_layers
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.mtp_mirror_staging_layers = nn.ModuleDict()
        if get_global_server_args().speculative_algorithm is not None:
            for consumer_layer_id, _source_layer_id in zip(
                getattr(config, "kv_mirror_layers", []) or [],
                getattr(config, "kv_mirror_imitated_layers", []) or [],
            ):
                consumer_layer_id = int(consumer_layer_id)
                if consumer_layer_id < int(config.num_hidden_layers):
                    continue
                staging_layer = _WelmMTPMirrorStagingLayer(
                    config,
                    consumer_layer_id,
                    quant_config,
                    prefix,
                )
                self.mtp_mirror_staging_layers[str(consumer_layer_id)] = (
                    staging_layer
                )
                LayerManager.set_decoder_layer(
                    consumer_layer_id, staging_layer
                )
        if self.pp_group.is_last_rank:
            self.norm = WelmV4FusedRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def bind_finalized_runner_plan(
        self, runner_plan: WelmRunnerParallelPlan
    ) -> None:
        for layer in self.layers:
            bind = getattr(layer, "bind_finalized_runner_plan", None)
            if bind is not None:
                bind(runner_plan)

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def _compute_oe_embedding(
        self,
        input_ids,
        forward_batch,
        base_hidden_states,
        oe_embed_modules=None,
        oe_up_proj_module=None,
    ):
        """Compute over-encoding embedding and combine with base hidden states.
        If oe_embed_modules/oe_up_proj_module are None, use the main OE modules."""
        if oe_embed_modules is None:
            oe_embed_modules = self.oe_embed
        if oe_up_proj_module is None:
            oe_up_proj_module = self.oe_gate_up_proj

        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            raise RuntimeError(
                "WeLMv4 over-encoding requires ngram_embedding_info. Ensure the "
                "model config contains non-empty oe_grams so ModelConfig enables "
                "the request token table."
            )

        num_tokens = input_ids.numel()
        # Attention-DP may fabricate collective-only tokens on a locally idle
        # rank after ngram metadata has been constructed.  Those rows do not
        # belong to a request and therefore must not index the request token
        # table (whose metadata is intentionally empty on that rank).
        if getattr(forward_batch, "_original_batch_size", None) == 0:
            return base_hidden_states

        is_nextn_model = bool(getattr(self, "is_nextn_model", False))
        is_frozen_mtp_decode = (
            is_nextn_model
            and forward_batch.forward_mode.is_decode()
            and bool(
                getattr(forward_batch.spec_info, "welmv4_mtp_frozen_kv", False)
            )
        )
        num_token_non_padded = getattr(
            forward_batch, "num_token_non_padded_cpu", None
        )
        if is_frozen_mtp_decode:
            # prepare_for_draft builds ForwardBatch while ScheduleBatch still
            # carries the preceding TARGET_VERIFY B*D input. draft_forward
            # replaces input_ids with the actual B*topk rows only afterwards.
            # Eager DP padding records that actual pre-pad width here; using
            # the stale generic scalar would include aligned dummy requests in
            # the explicit-history ngram hash. Direct Graph capture leaves the
            # field unset and therefore keeps the full fixed bucket; replay
            # padding is discarded by the later device real-row mask.
            draft_input_rows = getattr(
                forward_batch, "_original_num_tokens", None
            )
            if draft_input_rows is not None:
                num_token_non_padded = int(draft_input_rows)
        history_num_tokens = min(
            num_tokens,
            num_tokens
            if num_token_non_padded is None
            else int(num_token_non_padded),
        )
        if history_num_tokens == 0:
            return base_hidden_states
        oe_input_ids = input_ids[:history_num_tokens]
        req_lens = ngram_embedding_info.req_lens
        # Decode CUDA graphs round the batch size up and leave initialized
        # one-token rows behind the real requests. Only the first
        # ``history_num_tokens`` rows are real in non-speculative decode.
        # Eager extend/DP-attention metadata already contains one row per real
        # request, whose lengths sum to ``history_num_tokens``.
        if forward_batch.forward_mode.is_target_verify():
            verify_width = int(forward_batch.spec_info.draft_token_num)
            if verify_width <= 0 or history_num_tokens % verify_width != 0:
                raise RuntimeError(
                    "WeLMv4 TARGET_VERIFY ngram input must contain a positive "
                    "fixed-width row group per request, got "
                    f"tokens={history_num_tokens}, width={verify_width}."
                )
            original_batch_size = getattr(
                forward_batch, "_original_batch_size", None
            )
            if original_batch_size is None:
                original_batch_size = forward_batch.batch_size
            real_req_count = min(
                original_batch_size,
                forward_batch.req_pool_indices.shape[0],
                history_num_tokens // verify_width,
            )
            req_lens = torch.full(
                (real_req_count,),
                verify_width,
                dtype=torch.int32,
                device=input_ids.device,
            )
            column_starts = forward_batch.seq_lens[:real_req_count].to(torch.int32)
        elif forward_batch.forward_mode.is_decode():
            real_req_count = min(
                history_num_tokens,
                req_lens.shape[0],
                forward_batch.req_pool_indices.shape[0],
            )
            req_lens = req_lens[:real_req_count]
            column_starts = ngram_embedding_info.column_starts[:real_req_count]
        else:
            real_req_count = min(
                forward_batch.batch_size,
                req_lens.shape[0],
                forward_batch.req_pool_indices.shape[0],
            )
            req_lens = req_lens[:real_req_count]
            column_starts = ngram_embedding_info.column_starts[:real_req_count]
            if is_nextn_model:
                # NextN inputs are shifted by one target row for both prompt
                # draft-extend and DRAFT_EXTEND_V2.
                column_starts = column_starts + 1
        use_npu_fused_hash = (
            _is_npu
            and tuple(self.oe_grams) == (2, 2, 3, 3)
            and len(self.oe_vocab_sizes) == 4
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.extend_start_loc is not None
            )
        )
        npu_hashed_ids = None
        if is_frozen_mtp_decode and use_npu_fused_hash:
            previous1 = forward_batch.spec_info.welmv4_mtp_prev1_tokens
            previous2 = forward_batch.spec_info.welmv4_mtp_prev2_tokens
            if previous1 is None:
                raise RuntimeError("WeLMV4 MTP draft is missing previous token state")
            if previous2 is None:
                rows = forward_batch.req_pool_indices[:real_req_count].to(torch.int64)
                previous_columns = (
                    forward_batch.seq_lens[:real_req_count].to(torch.int64) - 1
                )
                valid = previous_columns >= 0
                previous2 = ngram_embedding_info.token_table[
                    rows, previous_columns.clamp_min(0)
                ]
                previous2 = torch.where(
                    valid, previous2, torch.zeros_like(previous2)
                )
            npu_hashed_ids = welmv4_oe_hash_explicit_history_4way_npu(
                oe_input_ids,
                previous1[:history_num_tokens],
                previous2[:history_num_tokens],
                vocab_size=self.vocab_size,
                oe_vocab_sizes=self.oe_vocab_sizes,
            )
        elif use_npu_fused_hash:
            req_pool_indices = forward_batch.req_pool_indices[:real_req_count]
            if forward_batch.forward_mode.is_decode():
                npu_hashed_ids = welmv4_oe_hash_decode_4way_npu(
                    oe_input_ids,
                    ngram_embedding_info.token_table,
                    req_pool_indices,
                    column_starts,
                    vocab_size=self.vocab_size,
                    oe_vocab_sizes=self.oe_vocab_sizes,
                )
            else:
                extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
                max_req_len = (
                    max(extend_seq_lens_cpu[:real_req_count], default=0)
                    if extend_seq_lens_cpu is not None
                    else (
                        int(forward_batch.spec_info.draft_token_num)
                        if forward_batch.forward_mode.is_target_verify()
                        else history_num_tokens
                    )
                )
                token_offsets = forward_batch.extend_start_loc
                if token_offsets is None:
                    token_offsets = (
                        torch.arange(
                            real_req_count,
                            dtype=torch.int32,
                            device=input_ids.device,
                        )
                        * int(forward_batch.spec_info.draft_token_num)
                    )
                npu_hashed_ids = welmv4_oe_hash_prefill_4way_npu(
                    oe_input_ids,
                    ngram_embedding_info.token_table,
                    req_pool_indices,
                    token_offsets[:real_req_count],
                    req_lens,
                    column_starts,
                    max_req_len=max_req_len,
                    vocab_size=self.vocab_size,
                    oe_vocab_sizes=self.oe_vocab_sizes,
                )
        else:
            req_lens_long = req_lens.to(dtype=torch.long)
            rows = torch.repeat_interleave(
                forward_batch.req_pool_indices[:real_req_count].to(dtype=torch.long),
                req_lens_long,
                output_size=history_num_tokens,
            )
            request_starts = torch.cumsum(req_lens_long, dim=0) - req_lens_long
            flat_starts = torch.repeat_interleave(
                request_starts,
                req_lens_long,
                output_size=history_num_tokens,
            )
            offsets = (
                torch.arange(history_num_tokens, device=input_ids.device) - flat_starts
            )
            token_positions = torch.repeat_interleave(
                column_starts.to(dtype=torch.long),
                req_lens_long,
                output_size=history_num_tokens,
            ) + offsets

            input_ids_ngram = []
            input_ids_ngram_tmp = oe_input_ids
            for g in range(1, max(self.oe_grams)):
                history_positions = token_positions - g
                valid_history = history_positions >= 0
                history_positions = history_positions.clamp_min(0)
                gram_tensor = ngram_embedding_info.token_table[
                    rows, history_positions
                ].to(dtype=input_ids.dtype)
                gram_tensor = torch.where(
                    valid_history, gram_tensor, torch.zeros_like(gram_tensor)
                )
                input_ids_ngram_tmp = input_ids_ngram_tmp + gram_tensor * (
                    self.vocab_size**g
                )
                input_ids_ngram.append(
                    hash_input_ids_vectorized(input_ids_ngram_tmp)
                )

        emb_ngram = []
        for i, vs in enumerate(self.oe_vocab_sizes):
            input_ids_ngram_hashed_tmp = (
                npu_hashed_ids[i]
                if npu_hashed_ids is not None
                else input_ids_ngram[self.oe_grams[i] - 2] % vs
            )
            emb_ngram_tmp = (
                oe_embed_modules[i].forward_local(input_ids_ngram_hashed_tmp)
                if _is_npu
                else oe_embed_modules[i](input_ids_ngram_hashed_tmp)
            )
            emb_ngram.append(emb_ngram_tmp)
        if _is_npu:
            # Four native local lookups + native concat + one OE all-reduce.
            # Base embedding communication remains unchanged in this phase.
            emb_ngram_concat = oe_embed_modules[0].concat_and_reduce_local_outputs(
                emb_ngram, dim=-1
            )
        else:
            emb_ngram_concat = torch.cat(emb_ngram, dim=-1)
        emb_new, _ = oe_up_proj_module(emb_ngram_concat)
        hidden_states = (base_hidden_states[:history_num_tokens] + emb_new) / 2.0
        if history_num_tokens < num_tokens:
            hidden_states = torch.cat(
                (hidden_states, base_hidden_states[history_num_tokens:]), dim=0
            )
        return hidden_states

    def _expand_scale_seq(self, input_ids, forward_batch, hidden_states):
        """Expand hidden_states from (T, D) to (T * scale, D) by interleaving
        main embedding with scale_seq embeddings.

        Layout per original token i:
          [main_emb_i, scale_seq_1_emb_i, ..., scale_seq_N_emb_i]
        """
        scale = self.scale_seq_times + 1
        T = hidden_states.shape[0]
        D = hidden_states.shape[1]

        # (T, D) -> (T, 1, D)
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states_list = [hidden_states]

        for s in range(self.scale_seq_times):
            hs_s = self.scale_seq_embed_tokens_list[s](input_ids)  # (T, D)
            if len(self.oe_grams) > 0 and forward_batch.ngram_embedding_info is not None:
                hs_s = self._compute_oe_embedding(
                    input_ids,
                    forward_batch,
                    hs_s,
                    oe_embed_modules=self.scale_seq_oe_embed_list[s],
                    oe_up_proj_module=self.scale_seq_oe_up_proj_list[s],
                )
            hs_s = hs_s.unsqueeze(1)  # (T, 1, D)
            hidden_states_list.append(hs_s)

        # (T, scale, D) -> (T * scale, D)
        hidden_states = torch.cat(hidden_states_list, dim=1)
        hidden_states = hidden_states.reshape(T * scale, D).contiguous()
        return hidden_states

    def _restore_npu_prefill_deepep_output_layout(
        self,
        hidden_states: torch.Tensor,
        aux_hidden_states: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if forward_batch.welmv4_npu_deepep_full_mirror:
            mirror_num_real_rows = forward_batch.kv_mirror_num_real_rows
            if mirror_num_real_rows is None:
                raise RuntimeError("KV Mirror FULL output metadata is incomplete")
            if hidden_states.shape[0] != mirror_num_real_rows:
                raise RuntimeError(
                    "WeLMv4 FULL KV Mirror output row count changed: "
                    f"{hidden_states.shape[0]} != {mirror_num_real_rows}"
                )
            return hidden_states, aux_hidden_states
        if not forward_batch.welmv4_npu_deepep_scattered:
            return hidden_states, aux_hidden_states

        tp_size = get_tensor_model_parallel_world_size()
        mirror_num_padded_rows = forward_batch.kv_mirror_num_padded_rows
        mirror_num_real_rows = forward_batch.kv_mirror_num_real_rows
        if mirror_num_padded_rows is not None and mirror_num_real_rows is None:
            raise RuntimeError("KV Mirror output metadata is incomplete")
        global_num_rows = mirror_num_padded_rows
        if global_num_rows is None:
            global_num_rows = forward_batch.global_dp_buffer_len
        if global_num_rows is None:
            global_num_rows = hidden_states.shape[0] * tp_size

        def restore_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() < 2:
                raise RuntimeError(
                    "WeLMv4 scattered output restoration requires row tensors"
                )
            if (
                mirror_num_real_rows is not None
                and tensor.shape[0] == mirror_num_real_rows
            ):
                return tensor
            if tensor.shape[0] == global_num_rows:
                restored = tensor
            elif tensor.shape[0] * tp_size == global_num_rows:
                restored = Qwen2MoeDecoderLayer._all_gather_tp_rows(tensor)
            else:
                raise RuntimeError(
                    "Cannot restore WeLMv4 scattered output with "
                    f"{tensor.shape[0]} rows to {global_num_rows} global rows"
                )
            if mirror_num_padded_rows is not None:
                restored = restored[:mirror_num_real_rows]
            return restored

        hidden_states = restore_tensor(hidden_states)
        aux_hidden_states = [restore_tensor(hidden) for hidden in aux_hidden_states]
        return hidden_states, aux_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # Construct immutable request-segment metadata once per eager forward.
        # EXTEND/MIXED positions are independently contiguous per request;
        # speculative modes are deliberately excluded.  DP-attention may have
        # appended a collective-only suffix to positions/K.  Segment metadata
        # must describe only the original request rows: the suffix is ignored
        # by attention and rotating it is both unnecessary and, because its
        # positions are zero-filled, incompatible with contiguous 64-token
        # tiles.  Prefill NPU Graph is disabled for WeLMv4, so no captured
        # buffer needs to own this tensor.
        forward_batch.welmv4_rope_segment_tile_starts = None
        if _is_npu:
            rope_num_position_tokens = positions.numel()
            original_num_tokens = getattr(
                forward_batch, "_original_num_tokens", None
            )
            if (
                original_num_tokens is not None
                and 0 <= int(original_num_tokens) <= rope_num_position_tokens
            ):
                rope_num_position_tokens = int(original_num_tokens)
            tile_starts = build_welmv4_rope_segment_tile_starts(
                forward_batch.extend_seq_lens_cpu,
                batch_size=forward_batch.batch_size,
                num_position_tokens=rope_num_position_tokens,
                ordinary_prefill=(
                    forward_batch.spec_info is None
                    and forward_batch.forward_mode
                    in (ForwardMode.EXTEND, ForwardMode.MIXED)
                ),
            )
            if tile_starts is not None:
                forward_batch.welmv4_rope_segment_tile_starts = torch.tensor(
                    tile_starts,
                    dtype=torch.int32,
                    device=positions.device,
                )
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            if len(self.oe_grams) > 0 and forward_batch.ngram_embedding_info is not None:
                hidden_states = self._compute_oe_embedding(
                    input_ids, forward_batch, hidden_states
                )

            if self.scale_seq_times > 0:
                hidden_states = self._expand_scale_seq(
                    input_ids, forward_batch, hidden_states
                )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if self.start_layer < self.end_layer:
            first_local_layer = self.layers[self.start_layer]
            dp_executor = getattr(first_local_layer, "_welm_dp_executor", None)
            if dp_executor is not None:
                # Allocate batch-owned gather/mask scratch during eager setup
                # (or Graph warmup), then refresh dynamic masks on every real
                # model forward so the refresh operations are captured.
                dp_executor.prepare_forward_scratch(
                    forward_batch=forward_batch,
                    hidden_states=hidden_states,
                )

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions, hidden_states, forward_batch, residual
                    )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            pre_norm_hidden_states = None
            if hidden_states.shape[0] != 0:
                if residual is None:
                    pre_norm_hidden_states = hidden_states
                    hidden_states, _ = self.norm(hidden_states)
                else:
                    last_layer = self.layers[self.end_layer - 1]
                    final_experts_output = getattr(
                        last_layer, "final_mlp_experts_output", None
                    )
                    final_shared_output = getattr(
                        last_layer, "final_mlp_shared_output", None
                    )
                    # Component-wise FP32 rebuild remains restricted to the
                    # single-rank path. TP/EP>1 consumes the layer's already
                    # assembled hidden_states instead.
                    can_rebuild_final_mlp = (
                        final_experts_output is not None
                        and getattr(last_layer.mlp, "tp_size", 1) == 1
                        and not is_dp_attention_enabled()
                    )
                    if can_rebuild_final_mlp:
                        hidden_states = final_experts_output.float() + residual.float()
                        if final_shared_output is not None:
                            hidden_states = hidden_states + final_shared_output.float()
                        pre_norm_hidden_states = hidden_states.to(self.norm.weight.dtype)
                        hidden_states = F.rms_norm(
                            pre_norm_hidden_states,
                            self.norm.weight.shape,
                            self.norm.weight,
                            eps=self.norm.eps,
                        )
                    else:
                        hidden_states = hidden_states.float() + residual.float()
                        pre_norm_hidden_states = hidden_states.to(self.norm.weight.dtype)
                        hidden_states = F.rms_norm(
                            pre_norm_hidden_states,
                            self.norm.weight.shape,
                            self.norm.weight,
                            eps=self.norm.eps,
                        )

        if (
            len(aux_hidden_states) == 0
            and forward_batch.capture_hidden_mode.need_capture()
            and pre_norm_hidden_states is not None
        ):
            aux_hidden_states = [pre_norm_hidden_states]

        hidden_states, aux_hidden_states = (
            self._restore_npu_prefill_deepep_output_layout(
                hidden_states, aux_hidden_states, forward_batch
            )
        )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class WeLMV4MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None and quant_config.get_name() != "modelslim":
            raise NotImplementedError(
                "WeLMv4 currently supports only ModelSlim quantized checkpoints; "
                f"got quantization method {quant_config.get_name()!r}."
            )
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # These managers are process-global because a source and its consumer
        # are ordinary layers in one serial forward.  Reset them when a new
        # model instance is constructed so stale state cannot survive reloads.
        KVMirrorManager.activations_dict_kv.clear()
        LayerManager.decoder_layer.clear()
        LayerManager.prepared_cross_model_pairs.clear()
        if _is_cuda:
            alt_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        elif _is_npu:
            alt_stream = torch.get_device_module().Stream()
        else:
            alt_stream = None
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def bind_finalized_runner_plan(
        self, runner_plan: WelmRunnerParallelPlan
    ) -> None:
        self.model.bind_finalized_runner_plan(runner_plan)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Return the largest real SWA left-history window for cache metadata.

        A checkpoint that only has the generic ``sliding_window`` field with
        ``use_sliding_window=false`` intentionally returns ``None`` here.
        Full-attention marker values are excluded.  The current token is not
        part of this value: eviction receives a pre-length that already excludes
        it, while the WeLM Triton adapter adds it back to the kernel span.
        """
        normalized_windows = get_welmv4_layerwise_sliding_windows(
            self.config,
            context_len=getattr(self.config, "context_len", None),
        )
        swa_windows = [window for window in normalized_windows if window >= 0]
        return max(swa_windows, default=None)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # Every supported top-level forward is serial (TBO/PDMux/speculative
        # Clearing here also recovers cleanly if a previous eager forward
        # failed between an in-model mirror source and consumer. Cross-model
        # MTP mirror tensors live in this forward's model_specific_states.
        KVMirrorManager.activations_dict_kv.clear()
        forward_batch.model_specific_states = None
        model_output = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if isinstance(model_output, tuple):
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
        if self.pp_group.is_last_rank:
            # Contract expanded hidden_states back to logical size for logits.
            # Transformer layers have already processed all T*scale states and
            # written KV cache.  For logits we only need the last state in each
            # scale group (matches MMQ's [:, -1, :] semantic).
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                # Select every scale-th element (last of each group)
                kv_mirror_contracted = (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    hidden_states = hidden_states[indices]
                    if aux_hidden_states is not None:
                        aux_hidden_states = [
                            hidden[indices] for hidden in aux_hidden_states
                        ]

                # Restore forward_batch metadata to logical space so that
                # LogitsProcessor sees the un-expanded lengths.
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            logits_output = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
            logits_output.model_specific_states = forward_batch.model_specific_states
            return logits_output
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

            if (
                len(self.model.oe_grams) > 0
                and forward_batch.ngram_embedding_info is not None
            ):
                forward_batch.hidden_states = self.model._compute_oe_embedding(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

            if self.model.scale_seq_times > 0:
                forward_batch.hidden_states = self.model._expand_scale_seq(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            hidden_states, _ = self.model._restore_npu_prefill_deepep_output_layout(
                hidden_states, [], forward_batch
            )
            forward_batch.hidden_states = hidden_states

            # Contract expanded hidden_states back to logical size
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                kv_mirror_contracted = (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    forward_batch.hidden_states = hidden_states[indices]
                else:
                    forward_batch.hidden_states = hidden_states
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                num_target_layers = int(self.config.num_target_hidden_layers)
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        if is_nextn:
            # nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            next_layer_prefixes = [
                f"model.layers.{i+num_target_layers}" for i in range(num_nextn_layers)
            ]
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        for name, loaded_weight in weights:
            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            logical_layer_id = int(name_list[2])
                            staging_layers = getattr(
                                self.model, "mtp_mirror_staging_layers", {}
                            )
                            is_staged_qkv = (
                                str(logical_layer_id) in staging_layers
                                and any(
                                    f".self_attn.{proj_name}." in name
                                    for proj_name in (
                                        "q_proj",
                                        "k_proj",
                                        "v_proj",
                                        "qkv_proj",
                                    )
                                )
                            )
                            if not is_staged_qkv:
                                continue
                            if name.endswith(".weight"):
                                staging_attn = staging_layers[
                                    str(logical_layer_id)
                                ].self_attn
                                if ".self_attn.qkv_proj." in name:
                                    staging_attn.loaded_qkv_parts.update(
                                        ("q", "k", "v")
                                    )
                                else:
                                    for projection_name, part in (
                                        ("q_proj", "q"),
                                        ("k_proj", "k"),
                                        ("v_proj", "v"),
                                    ):
                                        if (
                                            f".self_attn.{projection_name}."
                                            in name
                                        ):
                                            staging_attn.loaded_qkv_parts.add(part)
                                            break
                            name = name.replace(
                                f"model.layers.{logical_layer_id}.self_attn",
                                "model.mtp_mirror_staging_layers."
                                f"{logical_layer_id}.self_attn",
                                1,
                            )
            else:
                flag = False
                matched_prefix = None
                for next_layer_prefix in next_layer_prefixes:
                    if name.startswith(next_layer_prefix):
                        flag = True
                        matched_prefix = next_layer_prefix
                        break
                if not flag:
                    continue
                # if not name.startswith(nextn_layer_prefix):
                #     continue
                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(matched_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    weight_suffix = int(next_layer_prefix.split(".")[-1])
                    name = name.replace(
                        matched_prefix,
                        f"model.decoder_layers.{weight_suffix-num_target_layers}",
                    )
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            # This is already a fused OE projection parameter. Without this
            # early case, the generic ``up_proj -> gate_up_proj`` mapping below
            # rewrites it to ``oe_gate_gate_up_proj`` and silently skips it.
            if name == "model.oe_gate_up_proj.weight":
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "self_attn.gate_proj" in name:
                    continue
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if weight_name == "up_proj" and "scale_seq_oe_up_proj" in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if name == "model.oe_gate_up_proj.weight":
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attn_sink" in name:
                            start = get_parallel().attn_tp_rank * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
        self.post_init_after_load_weights(is_nextn=is_nextn)

    def get_embed_and_head(self):
        return [
            self.model.embed_tokens,
            self.model.oe_embed,
            self.model.oe_gate_up_proj,
        ], self.lm_head

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed[0]
        self.model.oe_embed = embed[1]
        self.model.oe_gate_up_proj = embed[2]
        self.lm_head = head
        device_module = torch.get_device_module()
        device_module.empty_cache()
        device_module.synchronize()

    def post_init_after_load_weights(self, is_nextn=False):
        total_kv_mirror_layers = getattr(self.model.config, "kv_mirror_layers", [])
        total_kv_mirror_imitated_layers = getattr(
            self.model.config, "kv_mirror_imitated_layers", []
        )
        if is_nextn:
            local_layer_ids = {
                decoder.self_attn.kv_mirror_layer_idx
                for decoder in self.model.decoder_layers
            }
        else:
            for logical_layer_id, staging_layer in getattr(
                self.model, "mtp_mirror_staging_layers", {}
            ).items():
                loaded_parts = staging_layer.self_attn.loaded_qkv_parts
                if loaded_parts != {"q", "k", "v"}:
                    raise RuntimeError(
                        "WeLMV4 target load is missing layer "
                        f"{logical_layer_id} mirror QKV weights; loaded "
                        f"parts={sorted(loaded_parts)}."
                    )
            local_layer_ids = {
                decoder_layer.self_attn.kv_mirror_layer_idx
                for decoder_layer in self.model.layers
            }
            local_layer_ids.update(
                int(layer_id)
                for layer_id in getattr(
                    self.model, "mtp_mirror_staging_layers", {}
                ).keys()
            )
        if _is_npu:
            router_decoder_layers = (
                self.model.decoder_layers if is_nextn else self.model.layers
            )
            for decoder_layer in router_decoder_layers:
                decoder_layer.mlp.prepare_npu_router_compute_weight_t(
                    decoder_layer.post_attention_layernorm.weight.dtype
                )
        # Preserve pair alignment while selecting consumers owned by this
        # model.  In particular, list[-0:] would return the entire source list
        # when a target-only model filters out all MTP consumers.
        active_pairs = [
            (mirror_layer, imitated_layer)
            for mirror_layer, imitated_layer in zip(
                total_kv_mirror_layers, total_kv_mirror_imitated_layers
            )
            if mirror_layer in local_layer_ids
        ]
        kv_mirror_layer_ids = [pair[0] for pair in active_pairs]
        kv_mirror_imitated_layers = [pair[1] for pair in active_pairs]
        LayerManager.post_init(
            kv_mirror_layer_ids, kv_mirror_imitated_layers, is_nextn=is_nextn
        )
        if not is_nextn and hasattr(self.model, "mtp_mirror_staging_layers"):
            for logical_layer_id, staging_layer in list(
                self.model.mtp_mirror_staging_layers.items()
            ):
                layer_id = int(logical_layer_id)
                if LayerManager.decoder_layer.get(layer_id) is staging_layer:
                    LayerManager.decoder_layer.pop(layer_id)
            # Do not carry a second layer48 QKV through quant post-processing
            # or serving; the target source has already absorbed its K/V.
            self.model.mtp_mirror_staging_layers = nn.ModuleDict()
            torch.get_device_module().empty_cache()

    def post_load_weights(self, is_nextn=False, weight_names=None):
        """Run KV-mirror fixups for loaders that bypass ``load_weights``."""
        del weight_names
        self.post_init_after_load_weights(is_nextn=is_nextn)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = WeLMV4MoeForCausalLM
