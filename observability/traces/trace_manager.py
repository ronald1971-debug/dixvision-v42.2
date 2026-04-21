"""
observability/traces/trace_manager.py
Minimal structured tracing — spans collected in-memory. No external deps.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_ns: int
    end_ns: int = 0
    attrs: dict[str, str] = field(default_factory=dict)


class TraceManager:
    def __init__(self, maxlen: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._spans: list[Span] = []
        self._maxlen = maxlen

    def start(self, name: str, trace_id: str | None = None) -> Span:
        span = Span(
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex,
            name=name,
            start_ns=time.monotonic_ns(),
        )
        return span

    def end(self, span: Span) -> Span:
        span.end_ns = time.monotonic_ns()
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self._maxlen:
                self._spans = self._spans[-self._maxlen :]
        return span

    def recent(self, n: int = 100) -> list[Span]:
        with self._lock:
            return list(self._spans[-n:])


_tm: TraceManager | None = None
_lock = threading.Lock()


def get_trace_manager() -> TraceManager:
    global _tm
    if _tm is None:
        with _lock:
            if _tm is None:
                _tm = TraceManager()
    return _tm
