"""Cross-engine sink protocols for the closed learning loop (P0-3).

Phase 5 wired ``FeedbackCollector`` (Execution → outcome buffer) and
``LearningInterface`` (Intelligence → signal+outcome rows) but nothing
forced ``ExecutionEngine.execute`` to actually feed them. The runtime
gap meant terminal :class:`ExecutionEvent` envelopes never reached the
buffers in production -- only unit-level tests instantiated them.

P0-3 closes the loop by letting ``ExecutionEngine`` accept two opt-in
sinks: a :class:`FeedbackCollector` (same package -- import is
intra-engine and free of B1 conflict) and an
:class:`IntelligenceFeedbackSink` -- a duck-typed protocol that
``intelligence_engine.learning_interface.LearningInterface`` already
satisfies. Routing through a ``core.contracts`` protocol keeps the
B1 cross-engine arrow at a single seam: both engines depend only on
``core.contracts``; neither imports the other.

The protocol is the strict subset of ``LearningInterface`` that
``ExecutionEngine`` needs at the call site: a single ``record``
method that consumes a ``(signal, execution[, mark_price])`` triple.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.contracts.events import ExecutionEvent, SignalEvent

__all__ = ["IntelligenceFeedbackSink"]


@runtime_checkable
class IntelligenceFeedbackSink(Protocol):
    """Duck-typed sink for ``(signal, execution)`` learning rows.

    ``intelligence_engine.learning_interface.LearningInterface`` is the
    canonical implementation. Other adapters (e.g. a ledger-backed
    replay sink) only need to provide the same ``record`` signature.

    Implementations must be deterministic, IO-free, and clock-free:
    callers pass already-realised events whose ``ts_ns`` carries the
    canonical time. The return type is intentionally ``object`` so a
    sink can return its own row type without coupling the protocol to
    a specific dataclass.
    """

    def record(
        self,
        *,
        signal: SignalEvent,
        execution: ExecutionEvent,
        mark_price: float | None = None,
    ) -> object:
        ...
