from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend_and_assign,
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_npu, next_power_of_2, is_npu_before_atlas_a5
from sglang.srt.hardware_backend.npu.triton import cache_loc_update, draft_future_replay_prepare_npu_triton

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_npu_before_atlas_a5 = is_npu_before_atlas_a5()
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def _clear_future_replay_buffers(self: EagleDraftInput) -> None:
        self.future_topk_p_buf = None
        self.future_topk_index_buf = None
        self.future_hidden_states_buf = None
        self.future_verified_id_buf = None
        self.future_new_seq_lens_buf = None

    def _get_full_future_relay_prepare_fallback_reason(
        self: EagleDraftInput,
        topk: int,
        num_steps: int,
    ) -> str | None:
        if self.future_indices is None:
            return "missing_future_indices"
        if not self.has_real_future_payload:
            return "future_payload_not_real"
        if self.new_seq_lens is None:
            return "missing_new_seq_lens"
        if self.last_verified_ids is None or self.token_list is None:
            return "missing_placeholder_payload"
        if (
            self.topk_p is None
            or self.topk_index is None
            or self.hidden_states is None
            or self.verified_id is None
        ):
            return "missing_full_relay_payload"
        if self.last_verified_ids.ndim != 1 or self.token_list.ndim != 2:
            return "invalid_placeholder_rank"
        if self.token_list.shape[0] != self.last_verified_ids.shape[0]:
            return "placeholder_batch_mismatch"
        if self.token_list.shape[1] != num_steps:
            return "placeholder_width_mismatch"
        if self.topk_p.ndim != 2 or self.topk_index.ndim != 2:
            return "invalid_full_relay_rank"
        if self.topk_p.shape != self.topk_index.shape:
            return "relay_topk_shape_mismatch"
        if self.topk_p.shape[0] != self.last_verified_ids.shape[0]:
            return "relay_topk_batch_mismatch"
        if self.topk_p.shape[1] != topk:
            return "relay_topk_width_mismatch"
        if self.hidden_states.ndim != 2:
            return "invalid_hidden_states_rank"
        if self.hidden_states.shape[0] != self.last_verified_ids.shape[0]:
            return "relay_hidden_batch_mismatch"
        if self.verified_id.ndim != 1:
            return "invalid_verified_id_rank"
        if self.verified_id.shape[0] != self.last_verified_ids.shape[0]:
            return "relay_verified_batch_mismatch"
        if self.new_seq_lens.ndim != 1:
            return "invalid_new_seq_lens_rank"
        if self.new_seq_lens.shape[0] != self.last_verified_ids.shape[0]:
            return "relay_new_seq_batch_mismatch"
        if torch.any(self.last_verified_ids < 0) or torch.any(self.token_list < 0):
            return "placeholder_not_resolved"
        return None

    def _has_replay_prepare_payload(self: EagleDraftInput) -> bool:
        return (
            self.future_indices is not None
            and self.future_topk_p_buf is not None
            and self.future_topk_index_buf is not None
            and self.future_verified_id_buf is not None
            and self.future_new_seq_lens_buf is not None
            and self.future_hidden_states_buf is not None
        )

    def _get_placeholder_prepare_fallback_reason(
        self: EagleDraftInput,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ) -> str | None:
        if topk != 1:
            return "topk_not_1"
        if not self.uses_future_placeholder:
            return "placeholder_contract_disabled"
        if self.future_indices is None:
            return "missing_future_indices"
        if self.last_verified_ids is None or self.token_list is None:
            return "missing_placeholder_payload"
        if self.last_verified_ids.ndim != 1 or self.token_list.ndim != 2:
            return "invalid_placeholder_rank"
        if self.token_list.shape[0] != self.last_verified_ids.shape[0]:
            return "placeholder_batch_mismatch"
        if self.token_list.shape[1] != num_steps:
            return "placeholder_width_mismatch"
        if torch.any(self.last_verified_ids < 0) or torch.any(self.token_list < 0):
            return "placeholder_not_resolved"
        if (
            self.future_topk_p_buf is None
            or self.future_hidden_states_buf is None
            or self.future_new_seq_lens_buf is None
        ):
            return "replay_only_state_incomplete"
        if getattr(getattr(draft_model_runner, "model", None), "hot_token_id", None) is not None:
            return "hot_token_projection_enabled"
        return None

    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Phase6-C:
        # this path only needs a stream dependency before GPU-side tensor work.
        # avoid blocking the scheduler thread on CPU here.
        batch.maybe_wait_verify_done_event()

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x
            r.decode_batch_idx += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                cur_kv_lens_cpu.to(device=batch.device),
                nxt_kv_lens_cpu.to(device=batch.device),
                out_cache_loc,
                bs,
            )
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            alloc_paged_token_slots_extend_and_assign(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                num_needed_tokens,
            )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_placeholder_decode_proposal_v2(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if self.new_seq_lens is None:
            raise ValueError(
                "prepare_for_placeholder_decode_proposal_v2 requires new_seq_lens."
            )

        batch.spec_info = self
        batch.seq_lens = self.new_seq_lens
        batch.seq_lens_cpu = self.new_seq_lens.detach().cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.extend_seq_lens = None
        batch.extend_prefix_lens = None
        batch.extend_num_tokens = 0
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DECODE
        )
        batch.out_cache_loc = None
        return self.prepare_for_v2_draft(
            req_to_token_pool=req_to_token_pool,
            batch=batch,
            cuda_graph_runner=None,
            draft_model_runner=draft_model_runner,
            topk=topk,
            num_steps=num_steps,
        )

    def prepare_for_placeholder_prefill_proposal_v2(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if self.new_seq_lens is None:
            raise ValueError(
                "prepare_for_placeholder_prefill_proposal_v2 requires new_seq_lens."
            )

        # Prefill proposal should consume the same decode-like proposal contract as
        # the decode path: proposal runs on a clean batch without extend metadata.
        batch.spec_info = self
        batch.seq_lens = self.new_seq_lens
        batch.seq_lens_cpu = self.new_seq_lens.detach().cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.extend_seq_lens = None
        batch.extend_prefix_lens = None
        batch.extend_num_tokens = 0
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DECODE
        )
        batch.out_cache_loc = None
        return self.prepare_for_v2_draft(
            req_to_token_pool=req_to_token_pool,
            batch=batch,
            cuda_graph_runner=None,
            draft_model_runner=draft_model_runner,
            topk=topk,
            num_steps=num_steps,
        )

    def _can_prepare_for_v2_draft_from_placeholder_payload(
        self: EagleDraftInput,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ) -> bool:
        return (
            self._get_placeholder_prepare_fallback_reason(
                draft_model_runner=draft_model_runner,
                topk=topk,
                num_steps=num_steps,
            )
            is None
        )

    def prepare_for_v2_draft_from_placeholder_payload(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
        clear_future_buffers: bool = True,
    ) -> bool:
        if not self._can_prepare_for_v2_draft_from_placeholder_payload(
            draft_model_runner=draft_model_runner,
            topk=topk,
            num_steps=num_steps,
        ):
            return False

        bs = len(batch.seq_lens)
        device = batch.seq_lens.device
        indices = self.future_indices.indices

        # Phase 5 front-half:
        # consume the fields that phase 4 already audits as equivalent
        # (`last_verified_ids`, `token_list[:, 0]`) from placeholder payload,
        # while still reusing replay-only state (`topk_p`, `hidden_states`,
        # `new_seq_lens`) until the later cleanup phase removes replay entirely.
        self.topk_p = self.future_topk_p_buf[indices]
        self.topk_index = self.token_list[:, :topk].to(
            device=device,
            dtype=(
                self.future_topk_index_buf.dtype
                if self.future_topk_index_buf is not None
                else torch.int64
            ),
        )
        self.hidden_states = self.future_hidden_states_buf[indices]
        self.verified_id = self.last_verified_ids.to(
            device=device,
            dtype=(
                self.future_verified_id_buf.dtype
                if self.future_verified_id_buf is not None
                else torch.int32
            ),
        )
        self.new_seq_lens = self.future_new_seq_lens_buf[indices]

        if not batch.forward_mode.is_idle():
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=device,
            )
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        if clear_future_buffers:
            self._clear_future_replay_buffers()
        return True

    def prepare_for_v2_draft_from_full_future_relay(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        topk: int,
        num_steps: int,
        clear_future_buffers: bool = True,
    ) -> bool:
        if (
            self._get_full_future_relay_prepare_fallback_reason(
                topk=topk,
                num_steps=num_steps,
            )
            is not None
        ):
            return False

        bs = len(batch.seq_lens)
        device = batch.seq_lens.device
        self.topk_p = self.topk_p.to(device=device)
        self.topk_index = self.topk_index.to(device=device)
        self.hidden_states = self.hidden_states.to(device=device)
        self.verified_id = self.verified_id.to(device=device)
        self.new_seq_lens = self.new_seq_lens.to(device=device)

        if not batch.forward_mode.is_idle():
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=device,
            )
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        if clear_future_buffers:
            self._clear_future_replay_buffers()
        return True

    def _build_v2_draft_replay_prepare_state(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        topk: int,
        num_steps: int,
    ) -> dict[str, torch.Tensor] | None:
        if not self._has_replay_prepare_payload():
            return None

        bs = len(batch.seq_lens)
        device = batch.input_ids.device
        replay_state = {
            "out_cache_loc": torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=device,
            ),
            "positions": torch.empty(
                (bs * topk,),
                dtype=torch.int64,
                device=device,
            ),
            "topk_p": torch.empty(
                (bs, topk),
                dtype=self.future_topk_p_buf.dtype,
                device=self.future_topk_p_buf.device,
            ),
            "topk_index": torch.empty(
                (bs, topk),
                dtype=self.future_topk_index_buf.dtype,
                device=self.future_topk_index_buf.device,
            ),
            "verified_id": torch.empty(
                (bs,),
                dtype=self.future_verified_id_buf.dtype,
                device=self.future_verified_id_buf.device,
            ),
            "new_seq_lens": torch.empty(
                (bs,),
                dtype=self.future_new_seq_lens_buf.dtype,
                device=self.future_new_seq_lens_buf.device,
            ),
            "hidden_states": torch.empty(
                (bs, self.future_hidden_states_buf.shape[1]),
                dtype=self.future_hidden_states_buf.dtype,
                device=self.future_hidden_states_buf.device,
            ),
        }
        draft_future_replay_prepare_npu_triton(
            future_indices=self.future_indices.indices,
            src_topk_p=self.future_topk_p_buf,
            src_topk_index=self.future_topk_index_buf,
            src_hidden_states=self.future_hidden_states_buf,
            src_verified_id=self.future_verified_id_buf,
            src_new_seq_lens=self.future_new_seq_lens_buf,
            src_seq_lens=batch.seq_lens,
            src_req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            dst_topk_p=replay_state["topk_p"],
            dst_topk_index=replay_state["topk_index"],
            dst_hidden_states=replay_state["hidden_states"],
            dst_verified_id=replay_state["verified_id"],
            dst_new_seq_lens=replay_state["new_seq_lens"],
            dst_positions=replay_state["positions"],
            dst_out_cache_loc=replay_state["out_cache_loc"],
            topk=topk,
            speculative_num_steps=num_steps,
        )
        return replay_state

    def prepare_for_v2_draft_from_replay_payload(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        topk: int,
        num_steps: int,
        clear_future_buffers: bool = True,
    ) -> bool:
        replay_state = self._build_v2_draft_replay_prepare_state(
            req_to_token_pool=req_to_token_pool,
            batch=batch,
            topk=topk,
            num_steps=num_steps,
        )
        if replay_state is None:
            return False

        batch.out_cache_loc = replay_state["out_cache_loc"]
        self.positions = replay_state["positions"]
        self.topk_p = replay_state["topk_p"]
        self.topk_index = replay_state["topk_index"]
        self.verified_id = replay_state["verified_id"]
        self.new_seq_lens = replay_state["new_seq_lens"]
        self.hidden_states = replay_state["hidden_states"]
        if clear_future_buffers:
            self._clear_future_replay_buffers()
        return True

    def _compare_placeholder_prepare_against_replay_prepare(
        self: EagleDraftInput,
        batch: ModelWorkerBatch,
        replay_state: dict[str, torch.Tensor],
    ) -> None:
        mismatch_messages = []

        if self.verified_id.shape != replay_state["verified_id"].shape:
            mismatch_messages.append(
                f"verified_id shape {tuple(self.verified_id.shape)} != {tuple(replay_state['verified_id'].shape)}"
            )
        elif torch.any(self.verified_id != replay_state["verified_id"]):
            mismatch_messages.append("verified_id")

        if self.topk_index.shape != replay_state["topk_index"].shape:
            mismatch_messages.append(
                f"topk_index shape {tuple(self.topk_index.shape)} != {tuple(replay_state['topk_index'].shape)}"
            )
        elif torch.any(self.topk_index != replay_state["topk_index"]):
            mismatch_messages.append("topk_index")

        if self.positions.shape != replay_state["positions"].shape:
            mismatch_messages.append(
                f"positions shape {tuple(self.positions.shape)} != {tuple(replay_state['positions'].shape)}"
            )
        elif torch.any(self.positions != replay_state["positions"]):
            mismatch_messages.append("positions")

        if self.new_seq_lens.shape != replay_state["new_seq_lens"].shape:
            mismatch_messages.append(
                f"new_seq_lens shape {tuple(self.new_seq_lens.shape)} != {tuple(replay_state['new_seq_lens'].shape)}"
            )
        elif torch.any(self.new_seq_lens != replay_state["new_seq_lens"]):
            mismatch_messages.append("new_seq_lens")

        if batch.out_cache_loc.shape != replay_state["out_cache_loc"].shape:
            mismatch_messages.append(
                f"out_cache_loc shape {tuple(batch.out_cache_loc.shape)} != {tuple(replay_state['out_cache_loc'].shape)}"
            )
        elif batch.out_cache_loc.dtype != replay_state["out_cache_loc"].dtype:
            mismatch_messages.append(
                f"out_cache_loc dtype {batch.out_cache_loc.dtype} != {replay_state['out_cache_loc'].dtype}"
            )

        if self.topk_p.shape != replay_state["topk_p"].shape:
            mismatch_messages.append(
                f"topk_p shape {tuple(self.topk_p.shape)} != {tuple(replay_state['topk_p'].shape)}"
            )
        elif self.topk_p.dtype != replay_state["topk_p"].dtype:
            mismatch_messages.append(
                f"topk_p dtype {self.topk_p.dtype} != {replay_state['topk_p'].dtype}"
            )
        elif not torch.allclose(self.topk_p, replay_state["topk_p"]):
            mismatch_messages.append("topk_p")

        if self.hidden_states.shape != replay_state["hidden_states"].shape:
            mismatch_messages.append(
                "hidden_states shape "
                f"{tuple(self.hidden_states.shape)} != {tuple(replay_state['hidden_states'].shape)}"
            )
        elif self.hidden_states.dtype != replay_state["hidden_states"].dtype:
            mismatch_messages.append(
                f"hidden_states dtype {self.hidden_states.dtype} != {replay_state['hidden_states'].dtype}"
            )

        if mismatch_messages:
            logger.warning(
                "Phase6-B consumer compare mismatch at interval=%s: %s.",
                self.future_indices.interval if self.future_indices is not None else None,
                "; ".join(mismatch_messages),
            )
        else:
            logger.debug(
                "Phase6-B consumer compare passed at interval=%s.",
                self.future_indices.interval if self.future_indices is not None else None,
            )

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        bs = len(batch.seq_lens)
        positions_prepared = False
        replay_prepare_state = None
        if not batch.forward_mode.is_idle():
            full_relay_fallback_reason = (
                self._get_full_future_relay_prepare_fallback_reason(
                    topk=topk,
                    num_steps=num_steps,
                )
            )
            prepared_from_full_future_relay = (
                full_relay_fallback_reason is None
                and self.prepare_for_v2_draft_from_full_future_relay(
                    req_to_token_pool=req_to_token_pool,
                    batch=batch,
                    topk=topk,
                    num_steps=num_steps,
                    clear_future_buffers=False,
                )
            )
            placeholder_fallback_reason = self._get_placeholder_prepare_fallback_reason(
                draft_model_runner=draft_model_runner,
                topk=topk,
                num_steps=num_steps,
            )
            prepared_from_placeholder_payload = (
                placeholder_fallback_reason is None
                and self.prepare_for_v2_draft_from_placeholder_payload(
                    req_to_token_pool=req_to_token_pool,
                    batch=batch,
                    draft_model_runner=draft_model_runner,
                    topk=topk,
                    num_steps=num_steps,
                    clear_future_buffers=False,
                )
            )
            future_ready = self._has_replay_prepare_payload()

            if prepared_from_full_future_relay:
                logger.debug(
                    "Phase6-B full-relay prepare hit at interval=%s.",
                    self.future_indices.interval if self.future_indices is not None else None,
                )
                if future_ready:
                    replay_prepare_state = self._build_v2_draft_replay_prepare_state(
                        req_to_token_pool=req_to_token_pool,
                        batch=batch,
                        topk=topk,
                        num_steps=num_steps,
                    )
            elif prepared_from_placeholder_payload:
                logger.debug(
                    "Phase6-B placeholder+replay prepare hit at interval=%s.",
                    self.future_indices.interval if self.future_indices is not None else None,
                )
                replay_prepare_state = self._build_v2_draft_replay_prepare_state(
                    req_to_token_pool=req_to_token_pool,
                    batch=batch,
                    topk=topk,
                    num_steps=num_steps,
                )
            elif future_ready:
                self.prepare_for_v2_draft_from_replay_payload(
                    req_to_token_pool=req_to_token_pool,
                    batch=batch,
                    topk=topk,
                    num_steps=num_steps,
                )
                positions_prepared = True
                if placeholder_fallback_reason is not None:
                    logger.debug(
                        "Phase6-B replay fallback at interval=%s full_relay_reason=%s placeholder_reason=%s.",
                        self.future_indices.interval if self.future_indices is not None else None,
                        full_relay_fallback_reason,
                        placeholder_fallback_reason,
                    )
            else:
                # Assign cache locations
                batch.out_cache_loc = torch.empty(
                    (bs * topk * num_steps,),
                    dtype=torch.int64,
                    device=batch.input_ids.device,
                )
                # FIXME(lsyin): align with the default code path
                assign_draft_cache_locs_page_size_1[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
                if (
                    full_relay_fallback_reason is not None
                    or placeholder_fallback_reason is not None
                ) and (
                    self.future_indices is not None or self.uses_future_placeholder
                ):
                    logger.debug(
                        "Phase6-B default fallback at interval=%s full_relay_reason=%s placeholder_reason=%s.",
                        self.future_indices.interval if self.future_indices is not None else None,
                        full_relay_fallback_reason,
                        placeholder_fallback_reason,
                    )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        if not positions_prepared:
            self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        if replay_prepare_state is not None:
            self._compare_placeholder_prepare_against_replay_prepare(
                batch=batch,
                replay_state=replay_prepare_state,
            )
            self._clear_future_replay_buffers()
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            # Set mamba_track_indices for mamba prefix-cache state tracking
            if get_global_server_args().enable_mamba_extra_buffer():
                batch.mamba_track_indices = torch.stack(
                    [
                        req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                        for req in batch.reqs
                    ]
                ).to(torch.int64)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_length = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        if _is_npu_before_atlas_a5:
            torch.ops.npu.cache_loc_update(
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
            )
        else:
            cache_loc_update(
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
            )

        return out_cache_loc
