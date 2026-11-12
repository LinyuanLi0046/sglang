"""Opt-in CPU timings for OpenAI response parsing and output preparation.

Callers must gate construction with request-time logging and ``is_enabled``;
the disabled path must not allocate timing state or read clocks. Synchronous
parser calls alone belong between start/observe, never an await or yield.
``before_sse_yield`` is application readiness, not ASGI/TCP delivery. Nonstream
response_build includes parser work, so those durations must not be added.
Unified streams capture only the first effective delta per choice. All records
use the existing bounded asynchronous PD writer.
"""

from __future__ import annotations

import logging
import time

from sglang.srt.observability.pd_time_stats import emit, identity

_logger = logging.getLogger("sglang.srt.managers.schedule_batch")
_SCALARS = (str, int, float, bool, type(None))


def is_enabled():
    try:
        return _logger.isEnabledFor(logging.INFO)
    except Exception:
        return False


class ParserTimingState(dict):
    """Carry timing alongside parser state without adding parser-visible keys."""

    def __init__(self, timing):
        super().__init__()
        self.timing = timing


class UnifiedStreamTiming(dict):
    """Carry tool parsers and remember the first effective SSE for each choice."""

    def __init__(self, num_choices):
        super().__init__()
        self.pending_choices = num_choices
        self.first_outputs = {}

    @property
    def timing(self):
        # After all choices have emitted content, skip the per-delta probes.
        return self if self.pending_choices else None

    def register_choice(self, rid, index):
        record = self.first_outputs.get(index)
        if record is None:
            record = self.first_outputs[index] = dict(
                rid=rid,
                index=index,
                first_effective_sse_ns=None,
                first_effective_kind=None,
                completion_tokens_at_first_effective=None,
            )
        return record

    def first(self, kind, *, rid, index, completion_tokens=None):
        record = self.register_choice(rid, index)
        if kind != "role" and record["first_effective_sse_ns"] is None:
            record.update(
                first_effective_sse_ns=time.perf_counter_ns(),
                first_effective_kind=kind,
                completion_tokens_at_first_effective=completion_tokens,
            )
            self.pending_choices -= 1

    def finish(self, status):
        for record in self.first_outputs.values():
            emit(
                "UnifiedRequestStats ",
                identity(
                    "unified",
                    None,
                    event="stream_finished",
                    status=status,
                    boundary="before_sse_yield",
                    **record,
                ),
            )


class ResponseTimingState(list):
    """Carry timing without changing the response builder's calling convention."""

    def __init__(self, results, timing):
        super().__init__(results)
        self.timing = timing
        try:
            timing.rid = self[0]["meta_info"]["id"]
        except Exception:
            pass


class ServingTiming:
    """Request-local scalar aggregates; never retain requests or output text."""

    def __init__(self, role, room, **fields):
        self.fields = identity(
            role, room, **{k: v for k, v in fields.items() if isinstance(v, _SCALARS)}
        )
        self.stages = {}
        self.first_events = {}
        self.marks = {}
        self.rid = None
        self.finished = False

    @staticmethod
    def start():
        try:
            return time.perf_counter_ns(), time.thread_time_ns()
        except Exception:
            return None

    def observe(self, stage, start_tuple, input_chars=0, empty=False):
        try:
            if start_tuple is None:
                return
            end_ns, end_cpu_ns = self.start()
            begin_ns, begin_cpu_ns = start_tuple
            wall_ns, cpu_ns = end_ns - begin_ns, end_cpu_ns - begin_cpu_ns
            stats = self.stages.get(stage)
            if stats is None:
                stats = self.stages[stage] = dict(
                    calls=0,
                    wall_ns=0,
                    thread_cpu_ns=0,
                    max_wall_ns=0,
                    max_thread_cpu_ns=0,
                    input_chars=0,
                    max_input_chars=0,
                    empty_outputs=0,
                    max_wall_begin_ns=begin_ns,
                    max_wall_end_ns=end_ns,
                )
            stats["calls"] += 1
            stats["wall_ns"] += wall_ns
            stats["thread_cpu_ns"] += cpu_ns
            stats["input_chars"] += input_chars
            stats["empty_outputs"] += bool(empty)
            stats["max_input_chars"] = max(stats["max_input_chars"], input_chars)
            stats["max_thread_cpu_ns"] = max(stats["max_thread_cpu_ns"], cpu_ns)
            if wall_ns >= stats["max_wall_ns"]:
                stats["max_wall_ns"] = wall_ns
                stats["max_wall_begin_ns"] = begin_ns
                stats["max_wall_end_ns"] = end_ns
        except Exception:
            # Diagnostics must not change parser results or exception handling.
            pass

    def _stage_snapshot(self):
        return {name: dict(values) for name, values in self.stages.items()}

    def first(self, kind, *, rid, index=0, completion_tokens=None):
        try:
            if kind in self.first_events or len(self.first_events) >= 4:
                return
            if self.rid is None:
                self.rid = rid
            event = dict(
                observed_ns=time.perf_counter_ns(),
                rid=rid,
                index=index,
                completion_tokens=completion_tokens,
            )
            self.first_events[kind] = event
            emit(
                "PDServingStats ",
                dict(
                    self.fields,
                    event="first_output_ready",
                    boundary="before_sse_yield",
                    kind=kind,
                    stages=self._stage_snapshot(),
                    **event,
                ),
            )
        except Exception:
            pass

    def mark(self, name):
        try:
            if name not in self.marks:
                self.marks[name] = time.perf_counter_ns()
        except Exception:
            pass

    def finish(self, status, *, rid=None):
        try:
            if self.finished:
                return
            self.finished = True
            emit(
                "PDServingStats ",
                dict(
                    self.fields,
                    event="serving_finished",
                    status=status,
                    rid=rid if rid is not None else self.rid,
                    observed_ns=time.perf_counter_ns(),
                    stages=self._stage_snapshot(),
                    first_events={k: dict(v) for k, v in self.first_events.items()},
                    marks=dict(self.marks),
                ),
            )
        except Exception:
            pass
