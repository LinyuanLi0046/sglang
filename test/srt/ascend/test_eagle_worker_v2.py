# Copyright 2026 SGLang Team
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

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class TestEagleV2ZeroBubbleIdleDp(unittest.TestCase):
    def _make_idle_batch(self, global_num_tokens):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True),
            global_num_tokens=global_num_tokens,
            seq_lens=sentinel.seq_lens,
            seq_lens_cpu=sentinel.seq_lens_cpu,
        )

    def _make_worker(self):
        worker = object.__new__(EagleDraftWorker)
        worker.speculative_num_steps = 2
        worker.req_to_token_pool = sentinel.req_to_token_pool
        worker.cuda_graph_runner = None
        worker.draft_runner = sentinel.draft_runner
        worker.topk = 1
        worker.draft_attn_backend = Mock()
        worker.draft_forward_zero_bubble = Mock(
            return_value=(sentinel.topk_p, sentinel.topk_index)
        )
        return worker

    def test_idle_rank_skips_when_all_dp_ranks_are_idle(self):
        worker = self._make_worker()
        batch = self._make_idle_batch([0] * 8)
        draft_input = Mock()

        worker.draft_zero_bubble(batch, sentinel.batch_result, draft_input)

        draft_input.prepare_for_v2_draft.assert_not_called()
        worker.draft_forward_zero_bubble.assert_not_called()

    def test_idle_rank_participates_when_another_dp_rank_has_work(self):
        worker = self._make_worker()
        batch = self._make_idle_batch([1] + [0] * 7)
        next_draft_input = SimpleNamespace(
            hidden_states=sentinel.hidden_states,
            topk_p=sentinel.initial_topk_p,
            topk_index=sentinel.initial_topk_index,
        )
        batch_result = SimpleNamespace(next_draft_input=next_draft_input)
        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                hidden_states=None,
                topk_p=None,
                topk_index=None,
            ),
            forward_mode=SimpleNamespace(is_idle=lambda: True),
        )
        draft_input = Mock()
        draft_input.prepare_for_v2_draft.return_value = (forward_batch, False)

        worker.draft_zero_bubble(batch, batch_result, draft_input)

        draft_input.prepare_for_v2_draft.assert_called_once()
        worker.draft_forward_zero_bubble.assert_called_once_with(forward_batch)
        worker.draft_attn_backend.init_forward_metadata.assert_not_called()
        self.assertIs(next_draft_input.hidden_states, sentinel.hidden_states)
        self.assertIs(next_draft_input.topk_p, sentinel.initial_topk_p)
        self.assertIs(next_draft_input.topk_index, sentinel.initial_topk_index)
        self.assertIs(batch.seq_lens, sentinel.seq_lens)
        self.assertIs(batch.seq_lens_cpu, sentinel.seq_lens_cpu)


if __name__ == "__main__":
    unittest.main()
