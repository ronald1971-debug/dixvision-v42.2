"""Decision Trace System — Phase 6 IMMUTABLE WIDGET 3 (DASH-04).

Renders the causal chain that led to each decision, by reading the
canonical 4-event stream and grouping events by the (symbol, ts_ns)
tuple under which the SIGNAL was emitted.

This widget is *purely* a read projection over the
:class:`LedgerReader`. It performs no I/O of its own and never
mutates engine state. (INV-08, INV-37)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.contracts.events import (
    Event,
    ExecutionEvent,
    HazardEvent,
    SignalEvent,
    SystemEvent,
)
from state.ledger.reader import LedgerCursor, LedgerReader


@dataclass(frozen=True, slots=True)
class DecisionTraceStep:
    """One step in a decision-trace chain."""

    ts_ns: int
    kind: str  # EventKind value
    summary: str


@dataclass(frozen=True, slots=True)
class DecisionTraceChain:
    """A grouped causal chain anchored to a specific symbol."""

    symbol: str
    steps: tuple[DecisionTraceStep, ...]


class DecisionTracePanel:
    """DASH-04 — Decision Trace System widget backend."""

    name: str = "decision_trace"
    spec_id: str = "DASH-04"

    def __init__(self, *, ledger: LedgerReader) -> None:
        self._ledger = ledger

    def chains(
        self,
        *,
        cursor: LedgerCursor | None = None,
        limit: int | None = 200,
    ) -> tuple[DecisionTraceChain, ...]:
        events = self._ledger.read(cursor, limit=limit)
        return self._group_by_symbol(events)

    @classmethod
    def _group_by_symbol(
        cls, events: Sequence[Event]
    ) -> tuple[DecisionTraceChain, ...]:
        # Stable grouping by symbol, preserving event order within
        # each chain. SystemEvents and HazardEvents without an owning
        # symbol are bucketed under "<system>".
        groups: dict[str, list[DecisionTraceStep]] = {}
        order: list[str] = []
        for event in events:
            symbol = cls._symbol_for(event)
            if symbol not in groups:
                groups[symbol] = []
                order.append(symbol)
            groups[symbol].append(cls._step_for(event))
        return tuple(
            DecisionTraceChain(symbol=symbol, steps=tuple(groups[symbol]))
            for symbol in order
        )

    @staticmethod
    def _symbol_for(event: Event) -> str:
        if isinstance(event, (SignalEvent, ExecutionEvent)):
            return event.symbol
        return "<system>"

    @staticmethod
    def _step_for(event: Event) -> DecisionTraceStep:
        kind_value = event.kind.value
        if isinstance(event, SignalEvent):
            chain = " -> ".join(event.plugin_chain) if event.plugin_chain else "(none)"
            summary = (
                f"SIGNAL {event.side.value} conf={event.confidence:.2f} "
                f"plugins=[{chain}]"
            )
        elif isinstance(event, ExecutionEvent):
            summary = (
                f"EXEC {event.status.value} {event.side.value} "
                f"qty={event.qty} px={event.price}"
            )
        elif isinstance(event, SystemEvent):
            summary = f"SYSTEM {event.sub_kind.value} from={event.source}"
        elif isinstance(event, HazardEvent):
            summary = (
                f"HAZARD {event.code} sev={event.severity.value} "
                f"src={event.source}"
            )
        else:
            summary = f"{kind_value} (unrecognised event variant)"
        return DecisionTraceStep(
            ts_ns=event.ts_ns, kind=kind_value, summary=summary
        )
