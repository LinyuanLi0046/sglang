"""BF16/MXFP8 MegaMoE for WeLM ordinary-prefill EP shards, not a global backend.

The target runner creates one registered buffer before KV-cache sizing. All
eligible layers share it; decode, mirror consumers and NextN keep their backend.
The custom npu_ops_transformer wheel is imported only during initialization.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

_MAX_LOCAL_ROWS = 16384
_MODE_BF16 = "bf16"
_MODE_MXFP8 = "mxfp8"
_MXFP8_E4M3_TYPE = 24
# Keep buffers alive until explicit graceful/group teardown, not Python GC.
_RUNTIMES: dict[object, WelmPrefillMegaMoE] = {}


def _weight_mode(experts):
    if all(
        getattr(experts, f"{prefix}_weight").dtype == torch.bfloat16
        for prefix in ("w13", "w2")
    ):
        return _MODE_BF16
    if all(
        getattr(experts, f"{prefix}_weight").dtype == torch.float8_e4m3fn
        and getattr(
            getattr(experts, f"{prefix}_kernel", None),
            "use_megamoe_canonical_layout",
            False,
        )
        for prefix in ("w13", "w2")
    ):
        return _MODE_MXFP8
    return None


class WelmPrefillMegaMoE:
    def __init__(self, group, *, config, device, weight_mode=_MODE_BF16):
        from npu_ops_transformer.ops.mega_moe import (
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )

        self.group = group
        self.weight_mode = weight_mode
        self.max_local_rows = _MAX_LOCAL_ROWS
        self._mega_moe = mega_moe
        self._closed = False
        # SymmBuffer queries the name with init_comm=False. Initialize on every
        # participating rank now, not lazily on the first non-empty request.
        group._get_backend(device).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=True
        )
        quant_args = {}
        if weight_mode == _MODE_MXFP8:
            quant_args["dispatch_quant_out_dtype"] = _MXFP8_E4M3_TYPE
        self.symm_buffer = get_symm_buffer_for_mega_moe(
            group,
            num_experts=config.num_experts,
            num_max_tokens_per_rank=_MAX_LOCAL_ROWS,
            num_topk=config.top_k,
            hidden=config.hidden_size,
            intermediate_hidden=(
                0
                if weight_mode == _MODE_MXFP8
                else config.intermediate_size_per_partition
            ),
            max_recv_token_num=0,
            dispatch_quant_mode=4 if weight_mode == _MODE_MXFP8 else 0,
            combine_quant_mode=0,
            comm_alg="",
            **quant_args,
        )

    def can_run(self, num_rows: int) -> bool:
        # DP callers pass the common physical slot, not local real-token count.
        return 0 < num_rows <= self.max_local_rows

    @staticmethod
    def local_valid_rows(shard_rows: int, real_rows: int, attn_tp_rank: int) -> int:
        # Ordinary prefill has a valid prefix followed by padding in each DP
        # slot. ReduceScatter preserves that order inside each attn-TP shard.
        # Use host metadata only; never read a device scalar to skip padding.
        return min(max(real_rows - attn_tp_rank * shard_rows, 0), shard_rows)

    def forward_layer(
        self,
        experts,
        hidden_states,
        topk_output,
        num_valid_rows: int,
        *,
        zero_output_padding: bool = True,
    ):
        # The caller skips DeepEP's -1 masking. TopK IDs are already legal and
        # distinct, including padding rows; only their mixture weights need 0.
        ids = topk_output.topk_ids.to(torch.int32)
        weights = topk_output.topk_weights.to(torch.bfloat16)
        has_padding = num_valid_rows < hidden_states.shape[0]
        if has_padding:
            # TopK returns FP32: the cast above owns a new BF16 tensor. Clear
            # only its suffix, without another full tensor copy or mask kernel.
            weights[num_valid_rows:].zero_()
        quant_args = {}
        if self.weight_mode == _MODE_MXFP8:
            # BF16 x is quantized inside MegaMoE; no external activation quant.
            quant_args = dict(
                scales=None,
                l1_weights_sf=[experts.w13_weight_scale],
                l2_weights_sf=[experts.w2_weight_scale],
                x_active_mask=None,
            )
        output, _ = self._mega_moe(
            hidden_states,
            ids,
            weights,
            # Opted-in experts retain ND during weight postprocessing,
            # in [E, 2I, H]/[E, H, I] order. No per-forward format conversion
            # or extra weight copy; fallback GMMs share these ND weights.
            [experts.w13_weight],
            [experts.w2_weight],
            self.symm_buffer,
            **quant_args,
        )
        if has_padding and zero_output_padding:
            output[num_valid_rows:].zero_()
        return output

    def close(self):
        if not self._closed:
            # The wheel drains device work before freeing registered memory.
            # Never call this on a scheduler exception/wedged-device path.
            self.symm_buffer.destroy()
            self._closed = True


def init_welm_prefill_megamoe(model_runner):
    """Bind loaded BF16/MXFP8 target layers without modifying their weights."""
    from sglang.srt.distributed import (
        get_moe_ep_group,
        get_moe_ep_normal_comm_group,
    )
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_deepep():
        return None

    eligible = []
    for layer in model_runner.model.model.layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if (
            experts is None
            or mlp.is_nextn
            or mlp.is_kv_mirror_consumer
            or not mlp.welm_local_ep_kernel_available
        ):
            continue
        if _weight_mode(experts) is None:
            continue
        config = experts.moe_runner_config
        if config.gemm1_clamp_limit is not None and config.gemm1_clamp_limit > 0:
            continue  # This custom ABI has no routed SwiGLU clamp argument.
        eligible.append(mlp)
    if not eligible:
        return None
    weight_modes = {_weight_mode(mlp.experts) for mlp in eligible}
    if len(weight_modes) != 1:
        # One registered buffer per EP group carries a fixed dispatch quant mode.
        raise ValueError("WeLM MegaMoE requires one expert weight mode per EP group")
    weight_mode = weight_modes.pop()

    plan = model_runner.runner_parallel_plan
    if plan is not None:
        coordinator = plan.moe_ep_normal_group or plan.moe_ep_group
    elif (
        envs.SGLANG_DEEPEP_NORMAL_USE_ALLGATHER.get()
        or envs.SGLANG_DEEPEP_NORMAL_USE_ALLTOALL.get()
    ):
        coordinator = get_moe_ep_normal_comm_group()
    else:
        coordinator = get_moe_ep_group()
    group = coordinator.device_group
    if group in _RUNTIMES:
        raise RuntimeError("WeLM MegaMoE context already initialized for this EP group")
    runtime = WelmPrefillMegaMoE(
        group,
        config=eligible[0].experts.moe_runner_config,
        device=eligible[0].experts.w13_weight.device,
        weight_mode=weight_mode,
    )
    _RUNTIMES[group] = runtime
    for mlp in eligible:
        mlp.welm_prefill_megamoe = runtime
    return runtime


def close_welm_megamoe_runtimes():
    for runtime in _RUNTIMES.values():
        runtime.close()
