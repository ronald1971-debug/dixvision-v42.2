"""Agent contracts (INV-54 — Phase 10.8).

Stateful intelligence components living under ``intelligence_engine/
agents/`` are permitted to carry private state (unlike pure
:class:`~intelligence_engine.plugins`), but they must pay for that
statefulness with mandatory introspection. The
:class:`AgentIntrospection` Protocol below is the minimal contract
every concrete agent class implements.

Refs:

* ``docs/manifest_v3.3_delta.md`` §1.4 (INV-54 — agent introspection)
* ``docs/directory_tree.md`` §intelligence_engine/agents/
* ``docs/canonical/phase_3_status.md`` (5 AGT-XX agents pending)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class AgentDecisionTrace:
    """Immutable record of a single agent decision.

    Carried in a bounded ring buffer (per :class:`AgentIntrospection`)
    so :meth:`AgentIntrospection.recent_decisions` is O(1) per call
    and never has to scan ``memory_tensor`` or the audit ledger.

    Attributes:
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority,
            T0-04). Captured by the agent at decision time.
        signal_id: Opaque handle of the upstream :class:`SignalEvent`
            (or empty string when the decision is HOLD with no
            originating signal).
        direction: ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        confidence: Agent's confidence in ``[0, 1]``.
        rationale_tags: Tuple of rationale tags drawn from
            ``registry/agent_rationale_tags.yaml`` (allowlisted at
            offline review time; runtime is read-only).
        memory_refs: Content-hashes of memory rows the agent
            consulted to make the decision (empty tuple when the
            agent is stateless across decisions).
    """

    ts_ns: int
    signal_id: str
    direction: str
    confidence: float
    rationale_tags: tuple[str, ...] = ()
    memory_refs: tuple[str, ...] = ()


@runtime_checkable
class AgentIntrospection(Protocol):
    """Mandatory introspection surface for every concrete agent.

    INV-54 invariants:

    1. :meth:`state_snapshot` MUST be **pure**: no side effects, no
       event emission, no PRNG, no clock. Same agent state ⇒ same
       snapshot string-for-string.
    2. :meth:`recent_decisions` MUST be **O(1) per call**: read out
       of an internal bounded ring buffer; never scan memory or
       ledger.
    3. :meth:`state_snapshot` keys MUST subset
       ``registry/agent_state_keys.yaml`` (allowlisted offline).
    4. :attr:`AgentDecisionTrace.rationale_tags` entries MUST subset
       ``registry/agent_rationale_tags.yaml``.

    The ``OperatorRequest(AGENT_INTROSPECT)`` GOV-CP-07 path samples
    these methods and emits the result as a typed ``SystemEvent``
    ledger row, giving HITL on-demand visibility without needing the
    agent to be stopped or replayed.
    """

    @property
    def agent_id(self) -> str:
        """Stable agent identifier (e.g. ``"AGT-01-scalper"``)."""

    def state_snapshot(self) -> Mapping[str, str]:
        """Return a compact deterministic snapshot of agent state.

        Keys subset ``registry/agent_state_keys.yaml``. Values are
        strings (no float drift across replay). Pure: no side
        effects, no clock, no PRNG.
        """

    def recent_decisions(self, n: int) -> Sequence[AgentDecisionTrace]:
        """Return the last ``≤ n`` decisions made by this agent.

        Read from a bounded internal ring buffer; O(1) per call.
        Returned in oldest-to-newest order. ``n <= 0`` yields an
        empty sequence.
        """


@dataclass(frozen=True, slots=True)
class AgentRegistryRow:
    """Read-only metadata about an agent registered in
    ``intelligence_engine/agents/`` (consumed by the dashboard
    /api/agents enumeration endpoint, when wired in a follow-up PR).

    Carried as a plain frozen value so dashboard / discovery code
    does not couple to concrete agent classes.
    """

    agent_id: str
    family: str  # "scalper" | "swing" | "macro" | "lp" | "adversarial"
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "AgentDecisionTrace",
    "AgentIntrospection",
    "AgentRegistryRow",
]
