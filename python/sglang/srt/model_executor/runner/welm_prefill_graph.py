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
"""WeLM BF16 target-prefill inputs for the existing breakable graph runner.

The transformer body, including native Flash, is captured in independent
Prompt[T] and Mirror[B] graphs. The ordinary logits tail remains eager.
"""

from __future__ import annotations

import copy
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# A narrow capability, not a blanket opt-out of DeepEP's eager protection.
welm_normal_graph_scope: ContextVar[bool] = ContextVar(
    "welm_normal_graph_scope", default=False
)


def parse_capture_batch_sizes(value: str, max_bs: int) -> tuple[int, ...]:
    sizes = sorted({int(item.strip()) for item in value.split(",")})
    if not sizes or sizes[0] <= 0:
        raise ValueError("WeLM prefill graph batch sizes must be positive integers")
    result = tuple(size for size in sizes if size <= max_bs)
    if not result:
        raise ValueError("No WeLM prefill graph batch size fits the request pool")
    return result


def capture_request_lengths(num_tokens: int, batch_size: int) -> list[int]:
    if not 0 < batch_size <= num_tokens:
        raise ValueError("WeLM capture requires 0 < batch_size <= num_tokens")
    quotient, remainder = divmod(num_tokens, batch_size)
    return [quotient + (i < remainder) for i in range(batch_size)]


def padded_flash_cu_seqlens(
    lengths: list[int], capacity: int, max_requests: int
) -> list[int]:
    """Physical TND offsets; seqused_q separately carries the real lengths.

    Give the last metadata slot the trailing physical padding, even when it
    is a real request. Flash uses seqused_q, not that span, for causal lengths.
    The final offset must cover the entire output, including its zeroed tail.
    """
    if (
        not lengths
        or len(lengths) > max_requests
        or any(length <= 0 for length in lengths)
        or sum(lengths) > capacity
    ):
        raise ValueError("Invalid WeLM Flash graph request lengths")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    offsets.extend([offsets[-1]] * (max_requests - len(lengths)))
    offsets[-1] = capacity
    return offsets


def padded_rope_tiles(
    lengths: list[int], capacity: int, max_requests: int | None = None
) -> list[int]:
    """Use the existing segmented kernel's masked zero-length tile contract.

    Empty tiles go BEFORE the first real tile. Their position load is at zero,
    never at the one-past-end sentinel. Real tokens are rotated exactly once.
    No new kernel or valid-tile-count ABI is needed.
    """
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("WeLM graph requests must each have at least one token")
    if sum(lengths) > capacity:
        raise ValueError("Request tokens exceed the WeLM graph capacity")
    starts = []
    offset = 0
    for length in lengths:
        starts.extend(range(offset, offset + length, 64))
        offset += length
    # sum(ceil(E_i / 64)) <= ceil(Tcap / 64) + B - 1.
    max_requests = len(lengths) if max_requests is None else max_requests
    if len(lengths) > max_requests:
        raise ValueError("Request count exceeds the fixed RoPE tile capacity")
    max_tiles = (capacity + 63) // 64 + max_requests - 1
    return [0] * (max_tiles - len(starts)) + starts + [offset]


class WelmPrefillGraphAdapter:
    def __init__(self, runner):
        self.runner = runner
        self.model = runner.layer_model
        self.backend = runner.model_runner.attn_backend
        self.max_tokens = runner.max_num_tokens
        self.device = runner.device
        args = runner.model_runner.server_args
        parallel = get_parallel()
        if (
            runner.model_runner.dtype != torch.bfloat16
            or runner.model_runner.kv_cache_dtype != torch.bfloat16
            or runner.quant_config is not None
        ):
            raise ValueError("WeLM breakable prefill currently requires unquantized BF16")
        if (
            parallel.enable_dp_attention
            or parallel.pp_size != 1
            or parallel.attn_cp_size != 1
            or args.dcp_size != 1
        ):
            raise ValueError(
                "WeLM breakable prefill requires DP attention off and PP=CP=DCP=1"
            )
        if not getattr(self.backend, "use_welm_flash_attn", False):
            raise ValueError("WeLM breakable prefill requires WELM_NPU_USE_FLASH_ATTN=1")
        if getattr(self.model, "is_nextn_model", False) or args.enable_lora:
            raise ValueError("WeLM breakable prefill supports the target model without LoRA")
        if self.model.scale_seq_times > 0:
            raise ValueError("WeLM breakable prefill does not capture scale-seq expansion")
        self.prune = bool(args.enable_kv_mirror)
        self.ep = parallel.moe_ep_size > 1
        if self.ep:
            from sglang.srt.layers.moe import get_moe_a2a_backend

            if (
                not get_moe_a2a_backend().is_deepep()
                or parallel.moe_ep_size != parallel.tp_size
                or not envs.SGLANG_DEEPEP_NORMAL_USE_ALLGATHER.get()
                or envs.SGLANG_DEEPEP_NORMAL_USE_ALLTOALL.get()
                or any(
                    not layer.mlp.supports_welm_local_ep_moe
                    for layer in self.model.layers
                )
            ):
                raise ValueError(
                    "WeLM EP prefill graph requires EP=TP, DeepEP NORMAL AllGather "
                    "and the existing mirror local-sort + AR path"
                )
        self.batch_sizes = parse_capture_batch_sizes(
            envs.SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES.get(),
            min(runner.max_bs, self.max_tokens),
        )
        self.max_requests = max(self.batch_sizes)
        # Allocated outside either graph pool and never shared with decode.
        # Page tables / KV lengths can be shared by T and B graphs: both
        # replays consume the same batch, in order on the forward stream.
        self.flash_inputs = self.backend.create_welm_prefill_graph_metadata(
            self.max_requests
        )
        self.flash_metadata = {}
        self.current_flash_metadata = None
        self.capture_phase = "prompt"
        self.first_mirror = (
            self.model.layers[0].first_target_kv_mirror_layer if self.prune else None
        )
        if self.prune and self.first_mirror is None:
            raise ValueError("WeLM split prefill requires a target mirror suffix")
        self.current_batch = None
        self.captured_states = {}
        self._warned = set()
        self.oe_ids = torch.zeros(
            (len(self.model.oe_grams), self.max_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        self.valid_rows = torch.zeros(self.max_tokens, dtype=torch.bool, device=self.device)
        self.local_valid_rows = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.full_write_locs = torch.zeros(
            self.max_tokens, dtype=torch.int64, device=self.device
        )
        self.swa_write_locs = torch.zeros_like(self.full_write_locs)
        # T-only tiles and fixed Bmax handoff storage remove the T x B key.
        self.rope_tiles = {}
        self.tail_indices = torch.zeros(
            self.max_requests, dtype=torch.int64, device=self.device
        )
        self.mirror_positions = torch.zeros_like(self.tail_indices)
        self.mirror_hidden = None
        self.mirror_residual = None
        self.mirror_residual_is_partial = False
        self.mirror_kv = {}
        self.mtp_kv = {}
        if self.prune:
            for layer in self.model.layers[self.first_mirror : self.model.end_layer]:
                attn = layer.self_attn
                self.mirror_kv[attn.attn.layer_id] = tuple(
                    torch.zeros(
                        (self.max_tokens, attn.kv_size),
                        dtype=runner.model_runner.dtype,
                        device=self.device,
                    )
                    for _ in range(2)
                )
        self.capture_keys = set()
        self._capture_templates = {}
        logger.info(
            "WeLM BF16 breakable prefill: T-only prompt graphs, exact mirror "
            "batches %s, mirror=%s, "
            "MoE=%s; native Flash is captured, LM head/logits stay eager; Gate side "
            "stream and CMO weight prefetch are disabled for graph prefill",
            self.batch_sizes,
            self.prune,
            "EP NORMAL AllGather/local AR" if self.ep else "TP",
        )

    def key(self, capacity: int, batch_size: int | None = None) -> ShapeKey:
        return ShapeKey(
            size=capacity,
            variant_label=f"welm:prompt:mirror={int(self.prune)}",
        )

    def mirror_key(self, batch_size: int) -> ShapeKey:
        return ShapeKey(size=batch_size, variant_label="welm:mirror")

    def reject(self, reason: str) -> bool:
        if reason not in self._warned:
            logger.info("WeLM prefill graph fallback before forward: %s", reason)
            self._warned.add(reason)
        return False

    def can_run(self, batch: ForwardBatch, capacity: int) -> bool:
        if batch.forward_mode != ForwardMode.EXTEND:
            return self.reject("only ordinary target EXTEND is captured")
        if batch.enable_kv_mirror != self.prune:
            return self.reject("mirror setting differs from the captured profile")
        if self.key(capacity) not in self.capture_keys:
            return self.reject("prompt token bucket was not captured")
        if self.prune and self.mirror_key(batch.batch_size) not in self.capture_keys:
            return self.reject("exact mirror request count was not captured")
        if not 0 < batch.batch_size <= self.max_requests:
            return self.reject("request count exceeds fixed metadata capacity")
        if (
            batch.seq_lens_cpu is None
            or int(batch.seq_lens_cpu.max().item()) > self.backend.max_context_len
        ):
            return self.reject("KV length exceeds fixed Flash page-table capacity")
        lengths = batch.extend_seq_lens_cpu
        if (
            lengths is None
            or len(lengths) != batch.batch_size
            or any(length <= 0 for length in lengths)
            or sum(lengths) > capacity
            or sum(lengths) > len(batch.input_ids)
        ):
            return self.reject("invalid ordinary prefill request lengths")
        if self.model.oe_grams and batch.ngram_embedding_info is None:
            return self.reject("missing request n-gram history metadata")
        if self.prune and self.model.layers_to_capture:
            return self.reject("mixed prompt-row and mirror-row auxiliary outputs")
        return True

    def _bind(self, batch: ForwardBatch, capacity: int):
        batch.welm_prefill_graph = self
        batch.welm_prefill_graph_phase = self.capture_phase
        batch.enable_kv_mirror = self.prune
        batch.welm_prefill_oe_ids = self.oe_ids[:, :capacity]
        batch.welm_prefill_token_mask = self.valid_rows[:capacity]
        batch.welm_prefill_full_write_locs = self.full_write_locs[:capacity]
        batch.welm_prefill_swa_write_locs = self.swa_write_locs[:capacity]
        batch.welmv4_rope_segment_tile_starts = self.rope_tiles.get(capacity)

    def prepare_capture(self, batch: ForwardBatch, capacity: int):
        self._bind(batch, capacity)
        # Capture is startup-only. Page zero is the existing reserved dummy
        # page; all real request/cache allocations remain outside this adapter.
        batch.out_cache_loc.zero_()
        batch.req_pool_indices.zero_()
        batch.input_ids.zero_()
        positions = []
        for length in batch.extend_seq_lens_cpu:
            positions.extend(range(length))
        batch.positions.copy_(torch.tensor(positions, device=self.device))
        self.oe_ids.zero_()
        self.valid_rows[:capacity].fill_(True)
        # Dummy attention reads the reserved zero page. Never race writes to
        # that page during capture, including the fused-QKV cache writer.
        self.full_write_locs[:capacity].fill_(-1)
        self.swa_write_locs[:capacity].fill_(-1)
        if batch.num_token_non_padded is not None:
            self.local_valid_rows.copy_(batch.num_token_non_padded)
        else:
            self.local_valid_rows.fill_(capacity)
        self._prepare_tiles(batch, capacity)
        self.backend.prepare_welm_prefill_graph_metadata(
            self.flash_inputs, batch, capture=True
        )
        mirror = batch.welm_prefill_graph_phase == "mirror"
        key = self.mirror_key(batch.batch_size) if mirror else self.key(capacity)
        metadata = copy.copy(self.flash_inputs)
        metadata.welm_flash_schedules = {}
        if mirror:
            bs = batch.batch_size
            metadata.block_tables = metadata.block_tables[:bs]
            if metadata.block_tables_swa is not None:
                metadata.block_tables_swa = metadata.block_tables_swa[:bs]
            metadata.welm_flash_seqused_kv = metadata.welm_flash_seqused_kv[:bs]
            metadata.welm_flash_cu_seqlens_q = torch.arange(
                bs + 1, dtype=torch.int32, device=self.device
            )
            metadata.welm_flash_seqused_q = torch.ones(
                bs, dtype=torch.int32, device=self.device
            )
            metadata.welm_flash_max_seqlen_q = 1
            metadata.welm_flash_mirror_q_lengths = (
                metadata.welm_flash_cu_seqlens_q,
                metadata.welm_flash_seqused_q,
            )
        else:
            metadata.welm_flash_cu_seqlens_q = torch.empty(
                self.max_requests + 1, dtype=torch.int32, device=self.device
            )
            self._prepare_flash_offsets(metadata, batch, capacity)
        self.flash_metadata[key] = metadata
        self.current_flash_metadata = metadata
        self._capture_templates[id(batch)] = (
            batch.num_token_non_padded_cpu,
            batch.global_dp_buffer_len,
            copy.copy(batch.global_num_tokens_cpu),
        )
        self.current_batch = batch

    def _prepare_flash_offsets(self, metadata, batch, capacity):
        offsets = padded_flash_cu_seqlens(
            batch.extend_seq_lens_cpu, capacity, self.max_requests
        )
        staging = torch.tensor(offsets, dtype=torch.int32, device="cpu", pin_memory=True)
        metadata.welm_flash_cu_seqlens_q.copy_(staging, non_blocking=True)

    def _prepare_tiles(self, batch: ForwardBatch, capacity: int):
        values = padded_rope_tiles(
            batch.extend_seq_lens_cpu, capacity, self.max_requests
        )
        key = capacity
        if key not in self.rope_tiles:
            self.rope_tiles[key] = torch.empty(
                len(values), dtype=torch.int32, device=self.device
            )
        # Fresh immutable pinned staging per batch. Do not overwrite a shared
        # CPU slot while an earlier forward-stream H2D may still be pending.
        # torch_npu's async copy records the pinned allocation on that stream.
        host_tiles = torch.tensor(
            values, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.rope_tiles[key].copy_(host_tiles, non_blocking=True)
        batch.welmv4_rope_segment_tile_starts = self.rope_tiles[key]
        offset = 0
        tails = []
        for length in batch.extend_seq_lens_cpu:
            offset += length
            tails.append(offset - 1)
        tails.extend([0] * (self.max_requests - len(tails)))
        host_tails = torch.tensor(
            tails, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.tail_indices.copy_(host_tails, non_blocking=True)

    def before_capture_forward(self, batch: ForwardBatch):
        from sglang.srt.models.welmv4 import KVMirrorManager

        KVMirrorManager.activations_dict_kv.clear()
        # Record schedule producers on EVERY warmup and actual capture. A
        # warmup schedule is not a valid substitute for an in-graph producer.
        key = (
            self.mirror_key(batch.batch_size)
            if batch.welm_prefill_graph_phase == "mirror"
            else self.key(batch.positions.numel())
        )
        self.current_flash_metadata = self.flash_metadata[key]
        self.current_flash_metadata.welm_flash_schedules.clear()
        batch.model_specific_states = None
        batch.__dict__.pop("custom_last_index", None)
        batch.welmv4_npu_deepep_scattered = False
        batch.welmv4_npu_deepep_full_mirror = False
        real_rows, buffer_rows, global_rows = self._capture_templates[id(batch)]
        batch.num_token_non_padded_cpu = real_rows
        batch.global_dp_buffer_len = buffer_rows
        batch.global_num_tokens_cpu = copy.copy(global_rows)
        # This copy is recorded at the start of the body. Mirror later fills
        # the SAME scalar with B, so replay must restore it from a separate,
        # live input rather than capturing a constant fill(Tcap).
        if batch.num_token_non_padded is not None:
            batch.num_token_non_padded.copy_(self.local_valid_rows)
        if batch.global_num_tokens_gpu is not None and global_rows is not None:
            batch.global_num_tokens_gpu.fill_(global_rows[0])
        self.current_batch = batch

    def after_capture(self, key: ShapeKey, batch: ForwardBatch):
        self.capture_keys.add(key)
        if batch.welm_prefill_graph_phase == "prompt":
            self.captured_states[key] = batch.model_specific_states

    def prepare_replay(self, live: ForwardBatch, static: ForwardBatch, capacity: int):
        self._bind(static, capacity)
        static.welm_prefill_graph_phase = "prompt"
        static.ngram_embedding_info = live.ngram_embedding_info
        static.global_num_tokens_cpu = copy.copy(live.global_num_tokens_cpu)
        static.welmv4_npu_deepep_scattered = False
        static.welmv4_npu_deepep_full_mirror = False
        real_rows = sum(live.extend_seq_lens_cpu)
        self.valid_rows[:real_rows].fill_(True)
        self.valid_rows[real_rows:capacity].zero_()
        self.oe_ids[:, real_rows:capacity].zero_()
        if self.model.oe_grams:
            ids = self.model._compute_oe_hashed_ids(live.input_ids[:real_rows], live)
            if ids is None or ids.shape != (len(self.model.oe_grams), real_rows):
                raise RuntimeError("WeLM prepared OE IDs do not match the real request rows")
            self.oe_ids[:, :real_rows].copy_(ids)
        if static.num_token_non_padded is not None:
            self.local_valid_rows.copy_(static.num_token_non_padded)
        else:
            self.local_valid_rows.fill_(real_rows)
        self.full_write_locs[real_rows:capacity].fill_(-1)
        self.full_write_locs[:real_rows].copy_(live.out_cache_loc[:real_rows])
        self.swa_write_locs[real_rows:capacity].fill_(-1)
        if self.backend.use_sliding_window_kv_pool:
            swa = self.backend.token_to_kv_pool.translate_loc_from_full_to_swa(
                live.out_cache_loc[:real_rows]
            )
            self.swa_write_locs[:real_rows].copy_(swa)
        self.backend.prepare_welm_prefill_graph_metadata(self.flash_inputs, live)
        self.current_flash_metadata = self.flash_metadata[self.key(capacity)]
        self._prepare_flash_offsets(self.current_flash_metadata, live, capacity)
        self._prepare_tiles(static, capacity)
        # Arbitrary-position fused QKV may load padded positions as well.
        static.positions[real_rows:capacity].zero_()
        self.current_batch = static

    def finish_prompt(self, hidden_states, residual, positions, batch):
        """T-only graph tail. Persistent outputs live outside both graph pools."""
        from sglang.srt.models.welmv4 import (
            KVMirrorManager,
            WELMV4_MTP_MIRROR_STATES_KEY,
        )

        capacity = positions.numel()
        for layer in self.model.layers[self.first_mirror : self.model.end_layer]:
            attn = layer.self_attn
            source = attn.kv_mirror_imitated_layers[
                attn.kv_mirror_layers.index(attn.kv_mirror_layer_idx)
            ]
            k, v = KVMirrorManager.get_kv_activation(source)
            dst_k, dst_v = self.mirror_kv[attn.attn.layer_id]
            # Raw source tensors can also serve MTP. Never rotate them in place.
            dst_k[:capacity].copy_(k)
            dst_v[:capacity].copy_(v)
            attn.prepare_graph_mirror_key(positions, dst_k[:capacity], batch)
            # Cache writes depend on T, not on the consumer's B-row Q. Finish
            # them here so Mirror[B] only reads its layer's paged cache.
            self.backend.write_welm_prefill_graph_kv(
                attn.attn,
                dst_k[:capacity],
                dst_v[:capacity],
                self.full_write_locs[:capacity],
                self.swa_write_locs[:capacity],
            )

        first = self.model.layers[self.first_mirror]
        hidden_states, residual, partial = first.prepare_graph_mirror_input(
            hidden_states, residual, batch, self.tail_indices
        )
        # Allocated on the first warmup, held strongly for every T and B graph.
        if self.mirror_hidden is None:
            self.mirror_hidden = torch.empty_like(hidden_states)
            self.mirror_residual = torch.empty_like(residual)
        if (
            hidden_states.dtype != self.mirror_hidden.dtype
            or residual.dtype != self.mirror_residual.dtype
        ):
            raise RuntimeError("WeLM mirror handoff dtype changed between captures")
        self.mirror_hidden.copy_(hidden_states)
        self.mirror_residual.copy_(residual)
        self.mirror_residual_is_partial = partial
        self.mirror_positions.copy_(positions.index_select(0, self.tail_indices))

        states = batch.model_specific_states
        if states and WELMV4_MTP_MIRROR_STATES_KEY in states:
            stable = {}
            for consumer, (k, v) in states[WELMV4_MTP_MIRROR_STATES_KEY].items():
                if consumer not in self.mtp_kv:
                    self.mtp_kv[consumer] = tuple(
                        tensor.new_empty((self.max_tokens, *tensor.shape[1:]))
                        for tensor in (k, v)
                    )
                dst_k, dst_v = self.mtp_kv[consumer]
                dst_k[:capacity].copy_(k)
                dst_v[:capacity].copy_(v)
                stable[consumer] = (dst_k[:capacity], dst_v[:capacity])
            batch.model_specific_states = {
                **states,
                WELMV4_MTP_MIRROR_STATES_KEY: stable,
            }
        # Do not give the generic backend a prompt output: its shared final
        # output buffer is allocated only when capturing Mirror[Bmax] later.
        return None

    def begin_mirror(self, batch):
        bs = batch.batch_size
        if self.ep:
            self.model.layers[self.first_mirror]._update_pure_tp_kv_mirror_full_metadata(
                batch, mirror_num_real_rows=bs
            )
        # Residual/norm kernels must not mutate the shared handoff input.
        return (
            self.mirror_hidden[:bs].clone(),
            self.mirror_residual[:bs].clone(),
            self.mirror_positions[:bs],
        )

    def replay(self, key, batch, **kwargs):
        mirror_key = self.mirror_key(batch.batch_size) if self.prune else None
        # Both checks precede either replay. Never run a mixed eager/graph body.
        if key not in self.capture_keys or (
            mirror_key is not None and mirror_key not in self.capture_keys
        ):
            raise RuntimeError("WeLM graph eligibility changed before replay")
        result = self.runner.backend.replay(key, batch, **kwargs)
        if mirror_key is not None:
            result = self.runner.backend.replay(mirror_key, batch, **kwargs)
        self.publish_state(key, batch)
        return result

    def publish_state(self, key: ShapeKey, batch: ForwardBatch):
        # Outer model.forward clears this before invoking the body. The raw
        # source0 tensors are outputs of this bucket's graph and refreshed by
        # its replay; preserve the existing serial target->draft lifetime.
        batch.model_specific_states = self.captured_states.get(key)

    def flash(self, layer, q, k, v, *, mirror: bool, save_kv_cache: bool, sinks=None):
        # Bypass RadixAttention's generic breakable eager wrapper. Explicit
        # metadata avoids changing the backend's eager/decode state at all.
        if save_kv_cache and not mirror:
            self.backend.write_welm_prefill_graph_kv(
                layer,
                k,
                v,
                self.full_write_locs[: q.shape[0]],
                self.swa_write_locs[: q.shape[0]],
            )
        pool = self.backend.token_to_kv_pool
        return self.backend._forward_welm_flash_attention(
            q,
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
            layer,
            sinks,
            mirror_prefill=mirror,
            graph_metadata=self.current_flash_metadata,
        )
