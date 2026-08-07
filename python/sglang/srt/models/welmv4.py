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
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
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
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.runtime_context import get_parallel

# from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_npu, make_layers

if is_npu():
    from sglang.srt.layers.welmv4_npu_op import mmq_style_router_linear_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()
_WELM_DUMP_PROCESS_DIR = None
_WELM_DUMP_BASE_DIR = None
_WELM_DUMP_PASS_ID = -1
_WELM_REPLAY_PASS_ID = -1
_WELM_REPLAY_LOGGED_KEYS = set()
_WELM_REPLAY_VALIDATED_PASSES = set()
_WELM_REPLAY_POINT_ALIASES = {
    "attn": "attention_input",
    "attention": "attention_input",
    "post_rope": "attention_input",
    "attention_output": "attn_output",
    "gated_attention_output": "gated_attn_output",
    "oproj": "o_proj_out",
    "o_proj": "o_proj_out",
    "norm_input": "norm_inputs",
    "norm_after_attention": "norm_after_attn",
}
_WELM_REPLAY_CONFIGURED_POINTS = frozenset(
    _WELM_REPLAY_POINT_ALIASES.get(point, point)
    for point in (
        item.strip().lower()
        for item in os.getenv("SGLANG_WELM_REPLAY_POINT", "").split(",")
    )
    if point
)


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


def _welm_dump_enabled() -> bool:
    return os.getenv("SGLANG_DUMP_ACTIVATIONS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _welm_should_dump_layer(layer_idx: int) -> bool:
    if not _welm_dump_enabled():
        return False
    layer_idxs = os.getenv("SGLANG_DUMP_ACTIVATIONS_LAYER_IDXS")
    if not layer_idxs:
        return True
    return str(layer_idx) in {x.strip() for x in layer_idxs.split(",") if x.strip()}


def _welm_dump_tensor(name: str, tensor: torch.Tensor) -> None:
    global _WELM_DUMP_PROCESS_DIR
    if not isinstance(tensor, torch.Tensor):
        return
    if _WELM_DUMP_PROCESS_DIR is None:
        process_dir = os.getenv("SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR")
        if process_dir:
            _WELM_DUMP_PROCESS_DIR = Path(process_dir)
        else:
            base_dir = Path(
                os.getenv("SGLANG_DUMP_ACTIVATIONS_DIR", "/tmp/sglang_welm_dump")
            )
            _WELM_DUMP_PROCESS_DIR = (
                base_dir / f"TP0_PP0_Rank0_pid{os.getpid()}" / "Pass00000"
            )
            os.environ["SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR"] = str(
                _WELM_DUMP_PROCESS_DIR
            )
        _WELM_DUMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), _WELM_DUMP_PROCESS_DIR / f"{name}.pt")


def _welm_dump_module_weights(prefix: str, module: nn.Module) -> None:
    """Dump the parameter shards that are resident on this worker.

    SGLang parallel layers replace a checkpoint parameter with the shard owned
    by the current TP/EP rank during weight loading.  Iterating the live module
    (instead of the checkpoint) therefore records the exact tensors consumed
    by this worker.

    Buffers are deliberately excluded: RoPE caches and other buffers are
    runtime state rather than learned weights.  WeLM's KV-mirror QKV tensors
    are an exception because they are unregistered tensors assembled after
    loading; those are dumped explicitly in the decoder layer below.
    """
    if not get_bool_env_var("DUMP_WEIGHT", "false"):
        return
    for name, parameter in module.named_parameters(recurse=True):
        parameter_name = f"{prefix}.{name}" if name else prefix
        _welm_dump_tensor(parameter_name, parameter)


def _welm_start_dump_pass() -> None:
    global _WELM_DUMP_BASE_DIR, _WELM_DUMP_PASS_ID, _WELM_DUMP_PROCESS_DIR
    if not _welm_dump_enabled():
        return
    if _WELM_DUMP_BASE_DIR is None:
        _WELM_DUMP_BASE_DIR = (
            Path(os.getenv("SGLANG_DUMP_ACTIVATIONS_DIR", "/tmp/sglang_welm_dump"))
            / f"TP0_PP0_Rank0_pid{os.getpid()}"
        )
    _WELM_DUMP_PASS_ID += 1
    _WELM_DUMP_PROCESS_DIR = _WELM_DUMP_BASE_DIR / f"Pass{_WELM_DUMP_PASS_ID:05d}"
    os.environ["SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR"] = str(_WELM_DUMP_PROCESS_DIR)
    _WELM_DUMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)


def _welm_replay_points() -> frozenset[str]:
    """Return the optional CUDA-activation replay points.

    This is intentionally an eager-only debugging facility.  The caller is
    responsible for disabling graph capture/replay before enabling it.
    """
    return _WELM_REPLAY_CONFIGURED_POINTS


def _welm_replay_point() -> str:
    """Return a normalized replay-point string for compatibility/debug logs."""
    return ",".join(sorted(_welm_replay_points()))


def _welm_start_replay_pass(forward_batch: ForwardBatch) -> None:
    global _WELM_REPLAY_PASS_ID
    # Model profiling/dummy forwards do not carry request IDs.  Do not consume
    # CUDA dump passes or attempt disk I/O until a real scheduled request runs.
    if _welm_replay_points() and forward_batch.rids:
        _WELM_REPLAY_PASS_ID += 1


def _welm_replay_stage(forward_batch: ForwardBatch) -> str:
    if forward_batch.forward_mode.is_decode():
        return "decode"
    if forward_batch.forward_mode.is_extend():
        return "prefill"
    return "other"


def _welm_replay_layer_ids() -> set[int]:
    value = os.getenv("SGLANG_WELM_REPLAY_LAYER_IDXS")
    if value is None:
        value = os.getenv("SGLANG_DUMP_ACTIVATIONS_LAYER_IDXS", "0")
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            "SGLANG_WELM_REPLAY_LAYER_IDXS must be a comma-separated list "
            f"of integers, got {value!r}."
        ) from exc


def _welm_should_replay(
    point: str,
    forward_batch: ForwardBatch,
    layer_idx: Optional[int] = None,
) -> bool:
    configured_points = _welm_replay_points()
    if not configured_points:
        return False
    if not forward_batch.rids or _WELM_REPLAY_PASS_ID < 0:
        return False
    valid_points = {
        "embedding",
        "qkv",
        "attention_input",
        "attn_output",
        "gated_attn_output",
        "o_proj_out",
        "norm_inputs",
        "norm_after_attn",
    }
    invalid_points = configured_points - valid_points
    if invalid_points:
        raise ValueError(
            "Each comma-separated SGLANG_WELM_REPLAY_POINT value must be one of "
            "embedding, qkv, "
            "attention_input, attn_output, gated_attn_output, o_proj_out, "
            f"norm_inputs, or norm_after_attn, got {sorted(invalid_points)!r}."
        )
    if point not in configured_points:
        return False

    stages = {
        item.strip().lower()
        for item in os.getenv(
            "SGLANG_WELM_REPLAY_STAGES", "prefill,decode"
        ).split(",")
        if item.strip()
    }
    stage = _welm_replay_stage(forward_batch)
    if stage not in stages:
        return False
    if layer_idx is not None and layer_idx not in _welm_replay_layer_ids():
        return False
    return True


def _welm_replay_pass_dir() -> Path:
    if _WELM_REPLAY_PASS_ID < 0:
        raise RuntimeError(
            "WeLM activation replay pass was not initialized before tensor load."
        )

    tp_rank = get_parallel().tp_rank
    directory_template = os.getenv(f"SGLANG_WELM_REPLAY_DIR_TP{tp_rank}")
    if not directory_template:
        directory_template = os.getenv("SGLANG_WELM_REPLAY_DIR")
    if not directory_template:
        raise RuntimeError(
            "WeLM activation replay is enabled but no source directory was "
            "configured. Set SGLANG_WELM_REPLAY_DIR (it may contain "
            "{tp_rank}, {pass_id}, or {pass_name}) or set the rank-specific "
            f"SGLANG_WELM_REPLAY_DIR_TP{tp_rank}."
        )

    try:
        pass_offset = int(os.getenv("SGLANG_WELM_REPLAY_PASS_OFFSET", "0"))
    except ValueError as exc:
        raise ValueError("SGLANG_WELM_REPLAY_PASS_OFFSET must be an integer.") from exc
    source_pass_id = _WELM_REPLAY_PASS_ID + pass_offset
    if source_pass_id < 0:
        raise RuntimeError(
            "The resolved CUDA replay pass is negative: "
            f"npu_pass={_WELM_REPLAY_PASS_ID}, offset={pass_offset}."
        )
    pass_name = f"Pass{source_pass_id:05d}"

    try:
        formatted = directory_template.format(
            tp_rank=tp_rank,
            attn_tp_rank=get_parallel().attn_tp_rank,
            pass_id=source_pass_id,
            pass_name=pass_name,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Invalid placeholder in SGLANG_WELM_REPLAY_DIR: only {tp_rank}, "
            "{attn_tp_rank}, {pass_id}, and {pass_name} are supported."
        ) from exc

    replay_dir = Path(os.path.expandvars(formatted)).expanduser()
    template_has_pass = (
        "{pass_id" in directory_template or "{pass_name" in directory_template
    )
    path_is_explicit_pass = (
        replay_dir.name.startswith("Pass") and replay_dir.name[4:].isdigit()
    )
    if not template_has_pass and not path_is_explicit_pass:
        replay_dir = replay_dir / pass_name
    return replay_dir


def _welm_load_replay_tensor(
    tensor_name: str,
    reference: torch.Tensor,
    *,
    point: str,
    forward_batch: ForwardBatch,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:
    """Load one CPU CUDA dump tensor and move it to the reference device."""
    source_path = _welm_replay_pass_dir() / f"{tensor_name}.pt"
    if not source_path.is_file():
        raise FileNotFoundError(
            "Missing WeLM CUDA replay tensor for "
            f"point={point}, stage={_welm_replay_stage(forward_batch)}, "
            f"layer={layer_idx}, npu_pass={_WELM_REPLAY_PASS_ID}: {source_path}"
        )
    try:
        source = torch.load(source_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose
        # weights_only on torch.load.
        source = torch.load(source_path, map_location="cpu")
    if not isinstance(source, torch.Tensor):
        raise TypeError(f"Replay file does not contain a tensor: {source_path}")
    if tuple(source.shape) != tuple(reference.shape):
        raise RuntimeError(
            "WeLM CUDA replay shape mismatch for "
            f"{tensor_name}: source={tuple(source.shape)}, "
            f"npu={tuple(reference.shape)}, stage={_welm_replay_stage(forward_batch)}, "
            f"npu_pass={_WELM_REPLAY_PASS_ID}, file={source_path}. This normally "
            "means the requests, tokenizer output, batching, or pass offset differ."
        )
    if source.dtype != reference.dtype:
        raise RuntimeError(
            "WeLM CUDA replay dtype mismatch for "
            f"{tensor_name}: source={source.dtype}, npu={reference.dtype}, "
            f"file={source_path}. Refusing an implicit cast because it would "
            "invalidate the precision comparison."
        )

    result = source.to(device=reference.device).contiguous()
    log_key = (point, _welm_replay_stage(forward_batch), layer_idx)
    if log_key not in _WELM_REPLAY_LOGGED_KEYS:
        _WELM_REPLAY_LOGGED_KEYS.add(log_key)
        logger.info(
            "WeLM CUDA activation replay enabled: point=%s stage=%s layer=%s "
            "npu_pass=%d source=%s",
            point,
            _welm_replay_stage(forward_batch),
            layer_idx,
            _WELM_REPLAY_PASS_ID,
            source_path,
        )
    return result


def _welm_validate_replay_positions(
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_idx: int,
) -> None:
    """Reject a same-shape decode dump from the wrong token position/pass."""
    if os.getenv("SGLANG_WELM_REPLAY_VALIDATE_POSITIONS", "1").lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    validation_key = (_WELM_REPLAY_PASS_ID, _welm_replay_stage(forward_batch))
    if validation_key in _WELM_REPLAY_VALIDATED_PASSES:
        return

    tensor_name = f"model.layers.{layer_idx}.self_attn.positions"
    source_path = _welm_replay_pass_dir() / f"{tensor_name}.pt"
    if not source_path.is_file():
        raise FileNotFoundError(
            "Cannot validate WeLM replay positions because the CUDA dump is "
            f"missing {source_path}. Set "
            "SGLANG_WELM_REPLAY_VALIDATE_POSITIONS=0 only if pass alignment "
            "has been verified separately."
        )
    try:
        source_positions = torch.load(
            source_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        source_positions = torch.load(source_path, map_location="cpu")
    current_positions = positions.detach().to(device="cpu")
    positions_match = (
        isinstance(source_positions, torch.Tensor)
        and tuple(source_positions.shape) == tuple(current_positions.shape)
        and torch.equal(
            source_positions.to(torch.int64), current_positions.to(torch.int64)
        )
    )
    if not positions_match:
        source_shape = (
            tuple(source_positions.shape)
            if isinstance(source_positions, torch.Tensor)
            else type(source_positions).__name__
        )
        raise RuntimeError(
            "WeLM CUDA replay positions do not match the current NPU batch: "
            f"source_shape={source_shape}, npu_shape={tuple(current_positions.shape)}, "
            f"stage={_welm_replay_stage(forward_batch)}, "
            f"replay_step={_WELM_REPLAY_PASS_ID}, file={source_path}."
        )
    _WELM_REPLAY_VALIDATED_PASSES.add(validation_key)


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


class LayerManager:
    decoder_layer = dict()
    num_nextn_predict_layers: int = 0
    num_target_layers: int = 0
    num_nextn_predict_layer_idx: List[int] = []

    @staticmethod
    def set_decoder_layer(layer_idx, decoder_layer):
        LayerManager.decoder_layer[layer_idx] = decoder_layer

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

            mirror_qkv_proj_weight = mirror_layer_attn.qkv_proj.weight
            mirror_qkv_proj_bias = getattr(mirror_layer_attn.qkv_proj, "bias", None)
            imitated_qkv_proj_weight = imitated_layer_attn.qkv_proj.weight
            imitated_qkv_proj_bias = getattr(imitated_layer_attn.qkv_proj, "bias", None)
            assert (mirror_qkv_proj_bias is not None) == (
                imitated_qkv_proj_bias is not None
            )

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

        torch.get_device_module().empty_cache()


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
    layer_id: Optional[int] = None,
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
        topk_scores, indices = mmq_style_expert_bias_topk(
            scores, expert_bias, topk, layer_id=layer_id
        )
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
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.expert_bias = torch.nn.Parameter(
            torch.zeros((config.num_experts), dtype=torch.float32)
        )
        self.layer_id = layer_id
        self.num_hidden_layers = config.num_hidden_layers
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
                layer_id=self.layer_id,
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
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.weight.data = self.gate.weight.to(torch.float32)
        if config.shared_expert_intermediate_size > 0:
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
                    if get_moe_a2a_backend().is_deepep()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        return_components: bool = False,
        skip_component_output: bool = False,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        dump_prefix = f"model.layers.{self.layer_id}.mlp"
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.router.input", hidden_states)
            _welm_dump_tensor(f"{dump_prefix}.router.input_fp32", hidden_states_fp32)
        shared_output = None
        if get_moe_a2a_backend().is_deepep() and hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        else:
            if self.shared_expert is not None:
                shared_output = self.shared_expert(hidden_states)
                if self.shared_expert_gate is not None:
                    shared_output = (
                        F.sigmoid(self.shared_expert_gate(hidden_states))
                        * shared_output
                    )
            if _is_npu:
                router_logits = mmq_style_router_linear_npu(
                    hidden_states, self.gate.weight
                )
            else:
                router_logits = mmq_style_router_linear(
                    hidden_states, self.gate.weight
                )
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.router.logits", router_logits)
                router_scores = (
                    torch.softmax(router_logits, dim=-1).type_as(router_logits)
                    if self.router_score_func == "softmax"
                    else torch.sigmoid(router_logits).type_as(router_logits)
                )
                _welm_dump_tensor(f"{dump_prefix}.router.scores", router_scores)
            # Ascend's fused TopK fast path currently ignores custom routing
            # callbacks. WeLM needs its sigmoid/expert-bias callback, so use the
            # shared PyTorch implementation on NPU. CUDA keeps the fused dispatch.
            if _is_npu and self.custom_routing_function is not None:
                topk_output = self.topk.forward_native(hidden_states, router_logits)
            else:
                topk_output = self.topk(hidden_states, router_logits)
        if dump_this_layer and hasattr(topk_output, "topk_weights"):
            _welm_dump_tensor(f"{dump_prefix}.router.topk_scores", topk_output.topk_weights)
            _welm_dump_tensor(f"{dump_prefix}.router.topk_ids", topk_output.topk_ids)
        experts_output = self.experts(hidden_states, topk_output)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.experts_output", experts_output)
        if return_components and skip_component_output:
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
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.shared_output", shared_output)
            if (
                self.layer_id == self.num_hidden_layers - 1
                and self.tp_size == 1
                and not use_reduce_scatter
            ):
                self.last_final_experts_output = experts_output
                self.last_final_shared_output = shared_output
            final_hidden_states = final_hidden_states + shared_output
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.output", final_hidden_states)
        if (
            self.tp_size > 1
            and not use_reduce_scatter
            and not get_moe_a2a_backend().is_deepep()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

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
        sliding_window_size_layerwise=[],
        enable_attn_sink_layerwise=[],
        layer_idx: Optional[int] = None,
        o_norm=False,
        rms_norm_eps: float = 1e-5,
        total_layer_num: int = 1,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

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
        if len(sliding_window_size_layerwise) > layer_idx:
            raw_sliding_window_size = sliding_window_size_layerwise[layer_idx]
            # HF window sizes include the current token, while SGLang
            # attention backends take the number of tokens to the left.
            self.sliding_window_size = (
                raw_sliding_window_size - 1
                if raw_sliding_window_size is not None
                and raw_sliding_window_size > 0
                else -1
            )
        else:
            self.sliding_window_size = -1
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
        self.kv_mirror_layer_idx = (
            layer_idx if not is_nextn else layer_idx + len(LayerManager.decoder_layer)
        )
        if get_global_server_args().speculative_algorithm is not None:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_nextn_predict_layers - 1
            )
        else:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_target_layers - 1
            )
        self.is_nextn = is_nextn

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_o_norm: bool = False,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_idx)
        dump_prefix = f"model.layers.{self.layer_idx}.self_attn"
        if self.kv_mirror_layer_idx in self.kv_mirror_imitated_layers:
            if hasattr(self, "qkv_proj_weight"):
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
                if (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                    and not hasattr(forward_batch, "custom_last_index")
                ):
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                    hidden_states = hidden_states[forward_batch.custom_last_index]
                k, v = KVMirrorManager.get_kv_activation(
                    mirror_layer_number, clear=self.need_clear_kv_cache
                )
                q = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        replay_qkv = _welm_should_replay("qkv", forward_batch, self.layer_idx)
        if replay_qkv:
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_idx
            )
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.q_pre_rope_npu_before_replay", q)
                _welm_dump_tensor(f"{dump_prefix}.k_pre_rope_npu_before_replay", k)
                _welm_dump_tensor(f"{dump_prefix}.v_npu_before_replay", v)
            q = _welm_load_replay_tensor(
                f"{dump_prefix}.q_pre_rope",
                q,
                point="qkv",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
            k = _welm_load_replay_tensor(
                f"{dump_prefix}.k_pre_rope",
                k,
                point="qkv",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
            v = _welm_load_replay_tensor(
                f"{dump_prefix}.v",
                v,
                point="qkv",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.positions", positions)
            if forward_batch.extend_seq_lens is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.extend_seq_lens",
                    forward_batch.extend_seq_lens,
                )
            _welm_dump_tensor(f"{dump_prefix}.q_pre_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_pre_rope", k)
            _welm_dump_tensor(f"{dump_prefix}.v", v)

        q_shape = q.shape
        k_shape = k.shape

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if self.q_norm is not None:
            q_by_head, _ = self.q_norm(q_by_head)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_after_norm", q_by_head.view(q.shape))
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        if self.k_norm is not None:
            k_by_head = mmq_style_k_rms_norm(
                k_by_head.contiguous(),
                self.k_norm.weight,
                self.k_norm.eps,
                layer_id=self.layer_idx,
                stage=(
                    "prefill"
                    if forward_batch.forward_mode.is_extend()
                    else "decode"
                    if forward_batch.forward_mode.is_decode()
                    else None
                ),
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.k_after_norm", k_by_head.view(k.shape))
        k = k_by_head.view(k.shape)

        qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if qk_nope_head_dim > 0:
            if (
                forward_batch.enable_kv_mirror
                and forward_batch.forward_mode.is_extend_without_speculative()
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            ):
                q, k = self.rotary_emb(
                    positions,
                    q,
                    k,
                    last_index=forward_batch.custom_last_index,
                    layer_id=self.layer_idx,
                )
            else:
                q, k = self.rotary_emb(
                    positions, q, k, layer_id=self.layer_idx
                )
            q = q.view(q_shape)
            k = k.view(k_shape)
        else:
            q, k = self.rotary_emb(positions, q, k, layer_id=self.layer_idx)

        replay_attention_input = _welm_should_replay(
            "attention_input", forward_batch, self.layer_idx
        )
        if replay_attention_input:
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_idx
            )
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.q_post_rope_npu_before_replay", q)
                _welm_dump_tensor(f"{dump_prefix}.k_post_rope_npu_before_replay", k)
                _welm_dump_tensor(f"{dump_prefix}.v_npu_before_replay", v)
            q = _welm_load_replay_tensor(
                f"{dump_prefix}.q_post_rope",
                q,
                point="attention_input",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
            k = _welm_load_replay_tensor(
                f"{dump_prefix}.k_post_rope",
                k,
                point="attention_input",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
            v = _welm_load_replay_tensor(
                f"{dump_prefix}.v",
                v,
                point="attention_input",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_post_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_post_rope", k)
            if replay_attention_input:
                # The regular V dump happens before RoPE.  Overwrite it here so
                # the canonical file records the value actually consumed by
                # attention in an attention-input replay run.
                _welm_dump_tensor(f"{dump_prefix}.v", v)

        attn_kwargs = {}
        if self.attn_sink is not None:
            attn_kwargs["sinks"] = self.attn_sink
        attn_output = self.attn(q, k, v, forward_batch, **attn_kwargs)
        replay_attn_output = _welm_should_replay(
            "attn_output", forward_batch, self.layer_idx
        )
        if replay_attn_output:
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_idx
            )
            if dump_this_layer:
                _welm_dump_tensor(
                    f"{dump_prefix}.attn_output_npu_before_replay", attn_output
                )
            attn_output = _welm_load_replay_tensor(
                f"{dump_prefix}.attn_output",
                attn_output,
                point="attn_output",
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.attn_output", attn_output)
        if self.gated_self_attention_headwise:
            attn_shape = attn_output.shape
            gate = self.gate_proj(hidden_states)[0].unsqueeze(
                -1
            )  # (bs * seq_len, num_heads, 1)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.router.0", gate.squeeze(-1)
                )
            attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
            inplace_sigmoid_mul(gate, attn_output)
            attn_output = attn_output.view(attn_shape)
            replay_gated_attn_output = _welm_should_replay(
                "gated_attn_output", forward_batch, self.layer_idx
            )
            if replay_gated_attn_output:
                _welm_validate_replay_positions(
                    positions, forward_batch, self.layer_idx
                )
                if dump_this_layer:
                    _welm_dump_tensor(
                        f"{dump_prefix}.gated_attn_output_npu_before_replay",
                        attn_output,
                    )
                attn_output = _welm_load_replay_tensor(
                    f"{dump_prefix}.gated_attn_output",
                    attn_output,
                    point="gated_attn_output",
                    forward_batch=forward_batch,
                    layer_idx=self.layer_idx,
                )
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.gated_attn_output", attn_output)

        output, _ = self.o_proj(attn_output)
        replay_o_proj_point = None
        if _welm_should_replay("norm_inputs", forward_batch, self.layer_idx):
            # ``norm_inputs`` remains the existing composite replay point: it
            # replaces both o_proj_out here and the FP32 residual below.
            replay_o_proj_point = "norm_inputs"
        elif _welm_should_replay("o_proj_out", forward_batch, self.layer_idx):
            replay_o_proj_point = "o_proj_out"
        replay_o_proj_out = replay_o_proj_point is not None
        o_proj_dump_name = (
            f"model.layers.{self.layer_idx}.attn.mixer.o_proj_out"
        )
        if replay_o_proj_out:
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_idx
            )
            if dump_this_layer:
                _welm_dump_tensor(
                    f"{o_proj_dump_name}_npu_before_replay", output
                )
            output = _welm_load_replay_tensor(
                o_proj_dump_name,
                output,
                point=replay_o_proj_point,
                forward_batch=forward_batch,
                layer_idx=self.layer_idx,
            )
        if dump_this_layer:
            _welm_dump_tensor(o_proj_dump_name, output)
        if self.o_norm is not None and not skip_o_norm:
            output, _ = self.o_norm(output)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.mixer.o_norm_out", output
                )
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_idx}.attn.mixer.0", output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
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
                int(layer_id)
                for layer_id in self.kv_mirror_layers
                if 0 <= int(layer_id) < int(config.num_hidden_layers)
            ),
            default=None,
        )
        self.sliding_window_size_layerwise = getattr(
            config, "sliding_window_size_layerwise", []
        )
        self.enable_attn_sink_layerwise = getattr(
            config, "enable_attn_sink_layerwise", []
        )
        self.ppln = getattr(config, "ppln", False)
        o_norm = getattr(config, "o_norm", False)
        self.prenorm_layer_idx = getattr(config, "prenorm_layer_idx", [])
        logger.debug(
            "WeLMv4 layer %s: ppln=%s, o_norm=%s, prenorm_layer_idx=%s",
            layer_id,
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
            layer_id=layer_id,
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
            sliding_window_size_layerwise=self.sliding_window_size_layerwise,
            enable_attn_sink_layerwise=self.enable_attn_sink_layerwise,
            layer_idx=layer_id,
            o_norm=o_norm and layer_id not in self.prenorm_layer_idx,
            rms_norm_eps=config.rms_norm_eps,
            total_layer_num=total_layer_num,
            is_nextn=is_nextn,
        )
        LayerManager.num_nextn_predict_layers = getattr(
            config, "num_nextn_predict_layers", 0
        )
        self.layer_id = layer_id
        self.is_final_layer = layer_id == total_layer_num - 1 or is_nextn

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        # Qwen2MoE all layers are sparse (include nextn layers)
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=True,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
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
        if (
            is_dp_attention_enabled()
            and self.self_attn.use_o_norm
            and self.layer_scatter_modes.mlp_mode != ScatterMode.FULL
        ):
            raise NotImplementedError(
                "WeLMv4 attention DP with o_norm currently requires the FULL "
                "MoE input layout; EP/A2A or context-parallel MOE_FULL layouts "
                "need a dedicated o_norm-aware communicator path."
            )
        LayerManager.set_decoder_layer(self.self_attn.kv_mirror_layer_idx, self)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        if (
            self.layer_id == 0
            and _welm_should_replay("embedding", forward_batch)
        ):
            _welm_validate_replay_positions(positions, forward_batch, self.layer_id)
            if dump_this_layer:
                _welm_dump_tensor(
                    "model.layers.0.__input__.0_npu_before_replay", hidden_states
                )
            hidden_states = _welm_load_replay_tensor(
                "model.layers.0.__input__.0",
                hidden_states,
                point="embedding",
                forward_batch=forward_batch,
                layer_idx=0,
            )
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_id}.__input__.0", hidden_states)
            _welm_dump_module_weights(
                f"model.layers.{self.layer_id}.__weights__", self
            )

            # LayerManager.post_init() constructs these tensors for KV-mirror
            # source/consumer layers.  They are plain Tensor attributes (not
            # Parameters), and Qwen2MoeAttention.forward() passes them directly
            # to F.linear, so named_parameters() above cannot see them.
            if hasattr(self.self_attn, "qkv_proj_weight"):
                _welm_dump_tensor(
                    f"model.layers.{self.layer_id}.__weights__."
                    "self_attn.qkv_proj_weight",
                    self.self_attn.qkv_proj_weight,
                )
                if self.self_attn.qkv_proj_bias is not None:
                    _welm_dump_tensor(
                        f"model.layers.{self.layer_id}.__weights__."
                        "self_attn.qkv_proj_bias",
                        self.self_attn.qkv_proj_bias,
                    )
        residual_after_layernorm = (
            self.ppln and self.layer_id not in self.prenorm_layer_idx
        )
        use_dp_layer_communicator = is_dp_attention_enabled()
        if use_dp_layer_communicator:
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states, residual, forward_batch
            )
            if residual_after_layernorm:
                residual = hidden_states.to(torch.float32)
        elif residual_after_layernorm:
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
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
            )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.input_layernorm.0", hidden_states
            )
            if residual is not None:
                _welm_dump_tensor(f"model.layers.{self.layer_id}.attn.mixer.1", residual)
        use_mmq_norm_after_attn = residual_after_layernorm and self.self_attn.use_o_norm
        use_dp_o_norm_after_attn = (
            is_dp_attention_enabled() and self.self_attn.use_o_norm
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                # o_norm is nonlinear and must see the attn-TP sum.  The
                # decoder performs it after the explicit all-reduce on DP
                # paths; the MMQ branch additionally fuses residual+rnorm.
                skip_o_norm=use_mmq_norm_after_attn or use_dp_o_norm_after_attn,
            )
        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.layer_id == self.first_target_kv_mirror_layer
        ):
            residual = residual[forward_batch.custom_last_index]
            if is_dp_attention_enabled():
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_rank,
                    set_dp_buffer_len,
                )

                dp_rank = get_attention_dp_rank()
                new_local_num_tokens = hidden_states.shape[0]
                scale = max(getattr(forward_batch, "scale_seq_factor", 1), 1)
                if scale > 1:
                    new_global_num_tokens_gpu = (
                        forward_batch.global_num_tokens_gpu // scale
                    )
                    forward_batch.global_num_tokens_gpu.copy_(
                        new_global_num_tokens_gpu
                    )
                    new_global_num_tokens = [
                        int(x) for x in new_global_num_tokens_gpu.tolist()
                    ]
                    if forward_batch.global_num_tokens_cpu is not None:
                        forward_batch.global_num_tokens_cpu = new_global_num_tokens
                else:
                    forward_batch.global_num_tokens_gpu[dp_rank] = (
                        new_local_num_tokens
                    )
                    new_global_num_tokens = None
                forward_batch.dp_local_start_pos = None
                forward_batch.dp_local_num_tokens = None
                if new_global_num_tokens is not None:
                    if forward_batch.dp_padding_mode.is_max_len():
                        global_dp_buffer_len = max(new_global_num_tokens) * len(
                            new_global_num_tokens
                        )
                    else:
                        global_dp_buffer_len = sum(new_global_num_tokens)
                    forward_batch.global_dp_buffer_len = global_dp_buffer_len
                else:
                    global_dp_buffer_len = forward_batch.global_dp_buffer_len
                set_dp_buffer_len(
                    global_dp_buffer_len,
                    new_local_num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    new_global_num_tokens,
                )

        replay_norm_inputs = _welm_should_replay(
            "norm_inputs", forward_batch, self.layer_id
        )
        if replay_norm_inputs:
            if residual is None:
                raise RuntimeError(
                    "WeLM norm_inputs replay requires a pre-norm residual, "
                    f"but layer {self.layer_id} received residual=None."
                )
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_id
            )
            residual_dump_name = f"model.layers.{self.layer_id}.attn.mixer.1"
            if dump_this_layer:
                _welm_dump_tensor(
                    f"{residual_dump_name}_npu_before_replay", residual
                )
            residual = _welm_load_replay_tensor(
                residual_dump_name,
                residual,
                point="norm_inputs",
                forward_batch=forward_batch,
                layer_idx=self.layer_id,
            )
            if dump_this_layer:
                # Overwrite the earlier pre-attention dump so the canonical
                # file records the residual actually consumed by the norm.
                _welm_dump_tensor(residual_dump_name, residual)

        if use_mmq_norm_after_attn:
            # With attention DP, RowParallelLinear deliberately leaves the
            # output projection as an attn-TP partial.  WeLM applies o_norm
            # before adding the residual, so summing the partials *after* the
            # nonlinear norm would be mathematically wrong.  Reconstruct the
            # full attention output first.
            if is_dp_attention_enabled() and self.attn_tp_size > 1:
                hidden_states = attention_tensor_model_parallel_all_reduce(
                    hidden_states
                )
            hidden_states, residual, hidden_states_fp32 = mmq_style_norm_after_attn(
                hidden_states,
                residual,
                self.self_attn.o_norm.weight,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
            if (
                is_dp_attention_enabled()
                and self.layer_scatter_modes.mlp_mode == ScatterMode.FULL
            ):
                from sglang.srt.layers.dp_attention import (
                    dp_gather_replicate,
                    get_attention_dp_size,
                    get_global_dp_buffer,
                )

                if get_attention_dp_size() != 1:
                    local_hidden_states = hidden_states
                    hidden_states = get_global_dp_buffer(get_tp_group())
                    # The attn-TP all-reduce above made the normalized local
                    # result identical inside each attention-TP group.  Gather
                    # one replica per DP shard; treating every replica as a
                    # partial would multiply the value by attn_tp_size.
                    dp_gather_replicate(
                        hidden_states, local_hidden_states, forward_batch
                    )
                    hidden_states_fp32 = hidden_states.to(torch.float32)
        elif use_dp_o_norm_after_attn:
            # ppln=False checkpoints still may carry o_norm.  Applying it in
            # Attention.forward would normalize each RowParallel partial.
            # Reconstruct the full o_proj result, then perform o_norm,
            # residual addition, post-attention norm, and the DP gather in
            # exactly that order.
            if self.attn_tp_size > 1:
                hidden_states = attention_tensor_model_parallel_all_reduce(
                    hidden_states
                )
            hidden_states, _ = self.self_attn.o_norm(hidden_states)
            (
                hidden_states,
                residual,
                hidden_states_fp32,
            ) = self.post_attention_layernorm(
                hidden_states, residual, clone_fp32_out=True
            )

            from sglang.srt.layers.dp_attention import (
                dp_gather_replicate,
                get_attention_dp_size,
                get_global_dp_buffer,
            )

            if get_attention_dp_size() != 1:
                local_hidden_states = hidden_states
                hidden_states = get_global_dp_buffer(get_tp_group())
                dp_gather_replicate(
                    hidden_states, local_hidden_states, forward_batch
                )
                hidden_states_fp32 = hidden_states.to(torch.float32)
        else:
            if use_dp_layer_communicator:
                hidden_states, residual = self.layer_communicator.prepare_mlp(
                    hidden_states, residual, forward_batch
                )
                hidden_states_fp32 = hidden_states.to(torch.float32)
            else:
                (
                    hidden_states,
                    residual,
                    hidden_states_fp32,
                ) = self.post_attention_layernorm(
                    hidden_states, residual, clone_fp32_out=True
                )
        replay_norm_after_attn = _welm_should_replay(
            "norm_after_attn", forward_batch, self.layer_id
        )
        if replay_norm_after_attn:
            _welm_validate_replay_positions(
                positions, forward_batch, self.layer_id
            )
            norm_dump_prefix = f"model.layers.{self.layer_id}.norm_after_attn"
            if dump_this_layer:
                _welm_dump_tensor(
                    f"{norm_dump_prefix}.output_npu_before_replay", hidden_states
                )
                _welm_dump_tensor(
                    f"{norm_dump_prefix}.output_fp32_npu_before_replay",
                    hidden_states_fp32,
                )
                if residual is not None:
                    _welm_dump_tensor(
                        f"{norm_dump_prefix}.residual_npu_before_replay", residual
                    )
            hidden_states = _welm_load_replay_tensor(
                f"{norm_dump_prefix}.output",
                hidden_states,
                point="norm_after_attn",
                forward_batch=forward_batch,
                layer_idx=self.layer_id,
            )
            hidden_states_fp32 = _welm_load_replay_tensor(
                f"{norm_dump_prefix}.output_fp32",
                hidden_states_fp32,
                point="norm_after_attn",
                forward_batch=forward_batch,
                layer_idx=self.layer_id,
            )
            if residual is not None:
                residual = _welm_load_replay_tensor(
                    f"{norm_dump_prefix}.residual",
                    residual,
                    point="norm_after_attn",
                    forward_batch=forward_batch,
                    layer_idx=self.layer_id,
                )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output", hidden_states
            )
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output_fp32",
                hidden_states_fp32,
            )
            if residual is not None:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_id}.norm_after_attn.residual", residual
                )
        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        self.final_mlp_experts_output = None
        self.final_mlp_shared_output = None
        mlp_output = self.mlp(
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            return_components=dump_this_layer or self.is_final_layer,
            skip_component_output=(
                self.is_final_layer
                and residual is not None
                and not dump_this_layer
                and getattr(self.mlp, "tp_size", 1) == 1
                and not is_dp_attention_enabled()
            ),
        )
        experts_output = None
        shared_output = None
        if isinstance(mlp_output, tuple):
            hidden_states, experts_output, shared_output = mlp_output
        else:
            hidden_states = mlp_output

        if use_dp_layer_communicator:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        if self.is_final_layer:
            self.final_mlp_experts_output = experts_output
            self.final_mlp_shared_output = shared_output
        if dump_this_layer:
            output_with_residual = hidden_states
            if (
                residual is not None
                and experts_output is not None
                and experts_output.shape == residual.shape
            ):
                output_with_residual = experts_output.float() + residual.float()
                if shared_output is not None:
                    output_with_residual = output_with_residual + shared_output.float()
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.mlp.output_with_residual",
                output_with_residual,
            )
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
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
        # The legacy CUDA path optionally keeps the (potentially very large)
        # base/OE embedding tables in mapped pinned host memory.  Keep this
        # opt-in and CUDA-only so the Ascend path never imports a CUDA extension.
        self.use_host_embeddings = (
            _is_cuda and get_global_server_args().enable_over_encoding
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
        if self.pp_group.is_last_rank:
            self.norm = WelmV4FusedRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

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

        dump_oe = _welm_dump_enabled()
        if dump_oe:
            _welm_dump_module_weights("model.oe.__weights__.embed", oe_embed_modules)
            _welm_dump_module_weights(
                "model.oe.__weights__.up_proj", oe_up_proj_module
            )
            _welm_dump_tensor("model.oe.input_ids", input_ids)
            _welm_dump_tensor("model.oe.base_hidden_states", base_hidden_states)

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

        num_token_non_padded = getattr(
            forward_batch, "num_token_non_padded_cpu", None
        )
        history_num_tokens = min(
            num_tokens,
            num_tokens
            if num_token_non_padded is None
            else int(num_token_non_padded),
        )
        if history_num_tokens == 0:
            return base_hidden_states
        oe_input_ids = input_ids[:history_num_tokens]
        req_lens = ngram_embedding_info.req_lens.to(dtype=torch.long)
        # Decode CUDA graphs round the batch size up and leave initialized
        # one-token rows behind the real requests. Only the first
        # ``history_num_tokens`` rows are real in non-speculative decode.
        # Eager extend/DP-attention metadata already contains one row per real
        # request, whose lengths sum to ``history_num_tokens``.
        if forward_batch.forward_mode.is_decode():
            real_req_count = min(
                history_num_tokens,
                req_lens.shape[0],
                forward_batch.req_pool_indices.shape[0],
            )
        else:
            real_req_count = min(
                forward_batch.batch_size,
                req_lens.shape[0],
                forward_batch.req_pool_indices.shape[0],
            )
        req_lens = req_lens[:real_req_count]
        column_starts = ngram_embedding_info.column_starts[:real_req_count]
        rows = torch.repeat_interleave(
            forward_batch.req_pool_indices[:real_req_count].to(dtype=torch.long),
            req_lens,
            output_size=history_num_tokens,
        )
        request_starts = torch.cumsum(req_lens, dim=0) - req_lens
        flat_starts = torch.repeat_interleave(
            request_starts, req_lens, output_size=history_num_tokens
        )
        offsets = (
            torch.arange(history_num_tokens, device=input_ids.device) - flat_starts
        )
        token_positions = torch.repeat_interleave(
            column_starts.to(dtype=torch.long),
            req_lens,
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
            if dump_oe:
                _welm_dump_tensor(f"model.oe.gram{g + 1}.ids", gram_tensor)
            input_ids_ngram_tmp = input_ids_ngram_tmp + gram_tensor * (
                self.vocab_size**g
            )
            input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

        emb_ngram = []
        for i, vs in enumerate(self.oe_vocab_sizes):
            input_ids_ngram_hashed_tmp = input_ids_ngram[self.oe_grams[i] - 2] % vs
            if dump_oe:
                _welm_dump_tensor(
                    f"model.oe.vocab{i}.hashed_ids", input_ids_ngram_hashed_tmp
                )
            emb_ngram_tmp = oe_embed_modules[i](input_ids_ngram_hashed_tmp)
            if dump_oe:
                _welm_dump_tensor(f"model.oe.vocab{i}.embedding", emb_ngram_tmp)
            emb_ngram.append(emb_ngram_tmp)
        emb_new, _ = oe_up_proj_module(torch.cat(emb_ngram, dim=-1))
        hidden_states = (base_hidden_states[:history_num_tokens] + emb_new) / 2.0
        if history_num_tokens < num_tokens:
            hidden_states = torch.cat(
                (hidden_states, base_hidden_states[history_num_tokens:]), dim=0
            )
        if dump_oe:
            _welm_dump_tensor("model.oe.projected", emb_new)
            _welm_dump_tensor("model.oe.output", hidden_states)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        _welm_start_replay_pass(forward_batch)
        _welm_start_dump_pass()
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                if _welm_dump_enabled():
                    _welm_dump_module_weights(
                        "model.embed_tokens.__weights__", self.embed_tokens
                    )
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if _welm_dump_enabled():
                _welm_dump_tensor("model.embed_tokens.output", hidden_states)

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
                if _welm_dump_enabled():
                    _welm_dump_module_weights("model.norm.__weights__", self.norm)
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
                    # In TP>1, component tensors are still pre-all-reduce.
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
        if quant_config is not None:
            raise NotImplementedError(
                "Quantized WeLMv4 checkpoints are not supported by this rebase. "
                "KV-mirror rewrites QKV projections as dense tensors and the "
                "custom routing/clamp path has only been validated for BF16/FP16."
            )
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # These managers are process-global because a source and its consumer
        # are ordinary layers in one serial forward.  Reset them when a new
        # model instance is constructed so stale state cannot survive reloads.
        KVMirrorManager.activations_dict_kv.clear()
        LayerManager.decoder_layer.clear()
        alt_stream = (
            torch.cuda.Stream(device=torch.cuda.current_device()) if _is_cuda else None
        )
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

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Return the largest configured layerwise left window for metadata.

        A checkpoint that only has the generic ``sliding_window`` field with
        ``use_sliding_window=false`` intentionally returns ``None`` here.
        Individual layers still carry their own normalized window values.
        """
        layerwise_windows = (
            getattr(self.config, "sliding_window_size_layerwise", []) or []
        )
        normalized_windows = [
            int(window) - 1
            for window in layerwise_windows
            if window is not None and int(window) > 0
        ]
        return max(normalized_windows, default=None)

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
        # are rejected at startup).  Clearing here also recovers cleanly if a
        # previous eager forward failed between a mirror source and consumer.
        KVMirrorManager.activations_dict_kv.clear()
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

            if _welm_dump_enabled():
                _welm_dump_module_weights("lm_head.__weights__", self.lm_head)
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
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
            _welm_start_replay_pass(forward_batch)
            _welm_start_dump_pass()
            if input_embeds is None:
                if _welm_dump_enabled():
                    _welm_dump_module_weights(
                        "model.embed_tokens.__weights__", self.model.embed_tokens
                    )
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
            if _welm_dump_enabled():
                _welm_dump_module_weights("model.norm.__weights__", self.model.norm)
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
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
            if _welm_dump_enabled():
                _welm_dump_module_weights("lm_head.__weights__", self.lm_head)
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
                num_target_layers = LayerManager.num_target_layers
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
                            continue
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
            local_layer_ids = {
                decoder_layer.self_attn.kv_mirror_layer_idx
                for decoder_layer in self.model.layers
            }
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
