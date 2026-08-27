# Copyright 2024-2025 SGLang Team
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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
    welmv4_graph_uses_only_triton_sink,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class EAGLEDraftExtendNpuGraphRunner(EAGLEDraftExtendCudaGraphRunner):
    def __init__(self, eagle_worker: EagleDraftWorker):
        super().__init__(eagle_worker)
        self._welmv4_triton_sink_only = welmv4_graph_uses_only_triton_sink(
            self.model_runner
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_model_specific_buffers(self) -> None:
        self._is_welmv4_nextn = "WeLMV4MoeForCausalLMNextN" in (
            self.model_runner.model_config.hf_config.architectures or []
        )
        if not self._is_welmv4_nextn:
            return
        attention = self.model_runner.model.model.decoder_layers[0].self_attn
        kv_width = int(attention.kv_size)
        device = self.model_runner.device
        dtype = self.model_runner.model_config.dtype
        self._welm_mirror_k = torch.zeros(
            (self.max_num_token, kv_width), device=device, dtype=dtype
        )
        self._welm_mirror_v = torch.zeros_like(self._welm_mirror_k)
        self._welm_mirror_indices = torch.zeros(
            (self.max_num_token,), device=device, dtype=torch.int64
        )

    def _bind_model_specific_capture_inputs(self, spec_info, num_tokens: int) -> None:
        if not self._is_welmv4_nextn:
            return
        from sglang.srt.models.welmv4 import WELMV4_MTP_MIRROR_STATES_KEY

        logical_layer_id = int(
            self.model_runner.model_config.hf_config.num_target_hidden_layers
        )
        spec_info.model_specific_states = {
            WELMV4_MTP_MIRROR_STATES_KEY: {
                logical_layer_id: (
                    self._welm_mirror_k[:num_tokens],
                    self._welm_mirror_v[:num_tokens],
                )
            }
        }
        spec_info.mirrored_kv_indices = self._welm_mirror_indices[:num_tokens]

    def _load_model_specific_replay_inputs(
        self,
        forward_batch,
        *,
        raw_bs: int,
        padded_bs: int,
        num_tokens: int,
    ) -> None:
        if not self._is_welmv4_nextn:
            return
        from sglang.srt.models.welmv4 import WELMV4_MTP_MIRROR_STATES_KEY

        logical_layer_id = int(
            self.model_runner.model_config.hf_config.num_target_hidden_layers
        )
        states = forward_batch.spec_info.model_specific_states or {}
        mirror_states = states.get(WELMV4_MTP_MIRROR_STATES_KEY, {})
        if logical_layer_id not in mirror_states:
            raise RuntimeError(
                f"WeLMV4 draft-extend is missing mirror K/V for layer {logical_layer_id}"
            )
        mirror_k, mirror_v = mirror_states[logical_layer_id]
        if mirror_k.shape[0] != num_tokens or mirror_v.shape[0] != num_tokens:
            raise RuntimeError(
                "WeLMV4 target mirror rows do not match draft-extend rows: "
                f"{mirror_k.shape[0]}/{mirror_v.shape[0]} vs {num_tokens}."
            )
        self._welm_mirror_k[:num_tokens].copy_(mirror_k)
        self._welm_mirror_v[:num_tokens].copy_(mirror_v)
        indices = forward_batch.spec_info.mirrored_kv_indices
        if indices is None:
            self._welm_mirror_indices[:num_tokens].copy_(
                torch.arange(num_tokens, device=self._welm_mirror_indices.device)
            )
        else:
            self._welm_mirror_indices[:num_tokens].copy_(
                indices.to(torch.int64).clamp(min=0)
            )
        padded_tokens = padded_bs * self.captured_req_width
        if padded_tokens > num_tokens:
            self._welm_mirror_k[num_tokens:padded_tokens].zero_()
            self._welm_mirror_v[num_tokens:padded_tokens].zero_()
            self._welm_mirror_indices[num_tokens:padded_tokens].zero_()

    def _replay_graph(self, shape_key, forward_batch):
        hf_config = self.model_runner.model_config.hf_config
        if self._is_welmv4_nextn and self._welmv4_triton_sink_only:
            # The ragged draft-extend metadata was copied to stable device
            # buffers immediately before replay; there is no FIA CPU binding.
            return self.backend.replay(shape_key, forward_batch)
        if not (is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)):
            seq_lens = forward_batch.seq_lens_cpu.tolist() + [0] * (
                self.bs - self.raw_bs
            )
            return self.backend.replay_with_input_update(
                shape_key,
                seq_lens=seq_lens,
                attr_name="actual_seq_lengths_kv",
                attr_type=[],
            )
        else:
            return self.backend.replay(shape_key, forward_batch)
