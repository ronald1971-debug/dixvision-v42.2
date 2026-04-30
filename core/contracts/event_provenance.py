"""HARDEN-03 — runtime Triad-Lock provenance assertions (INV-69).

The static lint rules **B20** (only Execution constructs ``ExecutionEvent``),
**B21** (only Intelligence constructs ``SignalEvent``) and **B22** (only
System constructs ``HazardEvent``) catch role-violating *imports* at PR
time. They cannot see anything that goes through dynamic dispatch — a
factory function, a plugin loaded by registry, or a mock that constructs
an event from a path the linter doesn't know about.

This module is the runtime half of the Triad Lock. Every typed event
carries :attr:`produced_by_engine`; receivers call
:func:`assert_event_provenance` and the contract becomes hard:

* a :class:`SignalEvent` with ``produced_by_engine != "intelligence_engine"``
  raises :class:`EventProvenanceError`,
* a :class:`ExecutionEvent` not produced by ``"execution_engine"`` raises,
* a :class:`HazardEvent` not produced by ``"system_engine"`` (or by
  ``"execution_engine"`` for the synthetic ``HAZ-AUTHORITY`` emitted by
  the Execution Gate itself) raises,
* a :class:`SystemEvent` whose ``produced_by_engine`` doesn't appear in
  the SystemEvent producer set raises.

Pairs with HARDEN-01 (B25 / ``ExecutionIntent``) and HARDEN-02 (the
runtime guard at the Execution Gate) — together they form the runtime
defence of the Triad Lock that the user described as the *causal
authority path*.
"""

from __future__ import annotations

from typing import Final

from core.contracts.events import (
    Event,
    EventKind,
    ExecutionEvent,
    HazardEvent,
    SignalEvent,
    SystemEvent,
)

__all__ = [
    "EVENT_PRODUCERS",
    "EventProvenanceError",
    "assert_event_provenance",
    "is_event_provenance_known",
]


class EventProvenanceError(RuntimeError):
    """Raised when a typed event's :attr:`produced_by_engine` does not
    match the Triad-Lock contract for its event class."""


# ---------------------------------------------------------------------------
# Producer registry
# ---------------------------------------------------------------------------
#
# Each entry maps an event class to the closed set of engine names that
# may legally produce that class. The registry is *frozen* — adding a
# producer is a registry change *and* an authority-matrix change. Tests
# read this map directly; receivers go through ``assert_event_provenance``.
#
# Notes:
# * ``HazardEvent`` accepts ``"execution_engine"`` because the Execution
#   Gate itself emits a synthetic ``HAZ-AUTHORITY`` hazard when its
#   runtime guard rejects an :class:`ExecutionIntent` (HARDEN-02). The
#   gate is the only execution-side hazard producer; everything else
#   comes from the ``system_engine`` Dyon domain.
# * ``SystemEvent`` accepts more producers than the other three because
#   it is the cross-engine coordination envelope (heartbeats, plugin
#   lifecycle, patch decisions, calibration reports …). Each entry
#   maps to a documented sub-kind in ``core.contracts.events``.

EVENT_PRODUCERS: Final[dict[type, frozenset[str]]] = {
    # ``intelligence_engine.cognitive`` is the Wave-03 PR-5 operator-
    # approval edge — the only legitimate cognitive-origin signal
    # producer. B26 lint pins the inverse: only that module may stamp
    # the cognitive prefix on a SignalEvent construction.
    SignalEvent: frozenset(
        {"intelligence_engine", "intelligence_engine.cognitive"},
    ),
    ExecutionEvent: frozenset({"execution_engine"}),
    HazardEvent: frozenset({"system_engine", "execution_engine"}),
    SystemEvent: frozenset(
        {
            "intelligence_engine",
            "execution_engine",
            "governance_engine",
            "system_engine",
            "learning_engine",
            "evolution_engine",
            "core.coherence",
        }
    ),
}


def is_event_provenance_known(event: Event) -> bool:
    """Return ``True`` if ``event`` carries a non-empty
    :attr:`produced_by_engine` and the value is in the producer set for
    its class.

    Useful for advisory paths that want to log provenance gaps without
    raising; production receivers should prefer :func:`assert_event_provenance`
    in strict mode.
    """

    producer = getattr(event, "produced_by_engine", "")
    if not producer:
        return False
    expected = EVENT_PRODUCERS.get(type(event))
    return expected is not None and producer in expected


def assert_event_provenance(event: Event, *, strict: bool = True) -> None:
    """Assert ``event.produced_by_engine`` matches its class' producer set.

    Args:
        event: One of the four canonical events. ``MarketTick`` and
            other non-event types are rejected.
        strict: When ``True`` (the default) an empty
            :attr:`produced_by_engine` is rejected. Soft mode (``False``)
            allows the empty string for backwards compatibility — used
            during the migration window where not every producer call
            site has been updated yet.

    Raises:
        EventProvenanceError: when the event class is unknown, the
            producer is not in the allowed set, or strict mode rejects
            an empty value.
    """

    expected = EVENT_PRODUCERS.get(type(event))
    if expected is None:
        raise EventProvenanceError(
            f"unknown event class {type(event).__name__!r}; "
            "add it to EVENT_PRODUCERS"
        )
    producer = getattr(event, "produced_by_engine", "")
    if not producer:
        if strict:
            raise EventProvenanceError(
                f"{type(event).__name__} produced_by_engine is empty; "
                f"expected one of {sorted(expected)} "
                f"(kind={getattr(event, 'kind', EventKind.SIGNAL).value})"
            )
        return
    if producer not in expected:
        raise EventProvenanceError(
            f"{type(event).__name__} produced_by_engine={producer!r} "
            f"violates Triad Lock; expected one of {sorted(expected)}"
        )
