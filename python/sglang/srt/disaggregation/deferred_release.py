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
# ==============================================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


@dataclass(frozen=True)
class _PendingRelease:
    fence: int
    req: Req
    is_insert: bool
    req_pool_idx: int
    kv: ReqKvInfo


class DecodeDeferredReleaseGuard:
    """Retain finished PD requests until the last submitted result is ready.

    The scheduler owns this CPU-only queue. A completed step must come from
    the existing copy_done wait, which follows model execution and sampling.
    Keeping the whole Req preserves Full/SWA pages, mappings and the row slot.
    """

    def __init__(self, tree_cache: BasePrefixCache):
        self.tree_cache = tree_cache
        self.latest_submitted = 0
        self.completed = 0
        self._pending: deque[_PendingRelease] = deque()
        self._pending_req_ids: set[int] = set()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def iter_pending_reqs(self) -> Iterator[Req]:
        return (item.req for item in self._pending)

    def on_submitted(self, step: Optional[int]) -> None:
        if step is None:
            return  # PREBUILT without an actual model execution.
        assert step > self.latest_submitted
        self.latest_submitted = step

    def defer_if_inflight(self, req: Req, is_insert: bool) -> bool:
        if id(req) in self._pending_req_ids:
            return True
        if self.latest_submitted <= self.completed:
            return False
        assert req.finished() and req.req_pool_idx is not None and req.kv is not None
        self._pending.append(
            _PendingRelease(
                self.latest_submitted, req, is_insert, req.req_pool_idx, req.kv
            )
        )
        self._pending_req_ids.add(id(req))
        return True

    def on_completed(self, step: Optional[int]) -> None:
        if step is None:
            return
        assert self.completed < step <= self.latest_submitted
        self.completed = step
        while self._pending and self._pending[0].fence <= self.completed:
            item = self._pending[0]
            # A queued Req must not be reset/rebootstrapped or transfer ownership.
            assert item.req.req_pool_idx == item.req_pool_idx
            assert item.req.kv is item.kv
            release_kv_cache(item.req, self.tree_cache, is_insert=item.is_insert)
            self._pending.popleft()
            self._pending_req_ids.remove(id(item.req))
