"""Engine and Plugin Protocols (Phase E0).

Defines the canonical contracts for all six engines and their plugins, and the
tier split between RUNTIME and OFFLINE engines.

Refs:
- ``manifest.md`` §0.2 / §0.2.1 / §0.3 / §0.8
- ``docs/total_recall_index.md`` §33 (ENGINE-01..06), §39 (RUNTIME/OFFLINE
  tier IDs), §40 (authority lint rule set)
- ``build_plan.md`` §Phase E0
- INV-08, INV-11, INV-15

The four runtime engines run on the per-tick canonical event bus and are
strictly deterministic (TEST-01). The two offline engines run on a scheduler,
share a single Python process for cost optimisation, and emit
``UPDATE_PROPOSED`` events through Governance only. Lint rule **L1**
(``tools/authority_lint.py``) keeps the Learning ↔ Evolution domain boundary
explicit even within the shared offline process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from core.contracts.events import Event


class EngineTier(StrEnum):
    """Operational tier of an engine.

    See ``manifest.md`` §0.2.1 and ``total_recall_index.md`` §39.
    """

    RUNTIME = "RUNTIME"
    OFFLINE = "OFFLINE"


class HealthState(StrEnum):
    """Result of :meth:`Engine.check_self`."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


@dataclass(frozen=True)
class HealthStatus:
    """Self-reported engine health.

    Attributes:
        state: Coarse health bucket.
        detail: Free-form detail (must not include secrets).
        plugin_states: ``slot_name -> {plugin_name: state}`` map. Empty when
            no plugins are loaded yet (Phase E0 default).
    """

    state: HealthState
    detail: str = ""
    plugin_states: Mapping[str, Mapping[str, HealthState]] = field(
        default_factory=dict
    )


class PluginLifecycle(StrEnum):
    """Plugin activation state. See PLUGIN-ACT-02."""

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"


@runtime_checkable
class Plugin(Protocol):
    """Plugin contract (PLUGIN-ACT-02 + PLUGIN-ACT-06).

    Plugins are configurable, hot-reloadable units that live inside an engine's
    plugin slot. They never communicate cross-engine directly — outputs are
    surfaced through the engine's ``process()`` return value, which produces
    typed :class:`Event` objects that flow on the canonical bus.
    """

    name: str
    version: str
    lifecycle: PluginLifecycle

    def process(self, event: Event) -> Sequence[Event]:
        """Handle one event, return zero or more events to emit downstream."""
        ...

    def check_self(self) -> HealthStatus:
        """Self-test; called by the host engine's ``check_self()``."""
        ...


@runtime_checkable
class Engine(Protocol):
    """Common engine contract.

    Concrete engines should subclass either :class:`RuntimeEngine` or
    :class:`OfflineEngine` rather than implementing :class:`Engine` directly.
    """

    name: str
    tier: EngineTier
    plugin_slots: Mapping[str, Sequence[Plugin]]

    def process(self, event: Event) -> Sequence[Event]:
        """Process one input event and emit zero or more typed events.

        Cross-engine calls MUST go through the typed event bus
        (INV-08, INV-11). Direct cross-engine imports are forbidden by the
        ``B1`` lint rule (see ``tools/authority_lint.py``).
        """
        ...

    def check_self(self) -> HealthStatus:
        """Return engine + plugin self-test status."""
        ...


@runtime_checkable
class RuntimeEngine(Engine, Protocol):
    """Runtime engine — on the per-tick canonical event bus.

    Strictly deterministic. Same event sequence + same FastRiskCache snapshot
    must produce bit-identical output (INV-15, TEST-01).

    Tier ID assignments (``total_recall_index.md`` §39):

    - ``RUNTIME-ENGINE-01`` Intelligence
    - ``RUNTIME-ENGINE-02`` Execution
    - ``RUNTIME-ENGINE-03`` System
    - ``RUNTIME-ENGINE-04`` Governance
    """

    tier: EngineTier  # = EngineTier.RUNTIME for all subclasses


@runtime_checkable
class OfflineEngine(Engine, Protocol):
    """Offline engine — scheduler-driven; never on the runtime bus.

    May absorb stochasticity (DRL exploration, AI-provider variance, sampling).
    Reads runtime state ONLY through ``state/ledger/reader.py`` (read-only).
    Emits ``UPDATE_PROPOSED`` events through Governance (GOV-G18); these are
    applied at snapshot boundaries by the runtime (INV-15 preserved).

    Tier ID assignments (``total_recall_index.md`` §39):

    - ``OFFLINE-ENGINE-01`` Learning
    - ``OFFLINE-ENGINE-02`` Evolution

    Both offline engines share **one Python process** (cost optimisation,
    ``roadmap.md`` §3 rec #3). The ``L1`` lint rule keeps the Learning ↔
    Evolution domain boundary explicit even in the shared process.
    """

    tier: EngineTier  # = EngineTier.OFFLINE for all subclasses

    def schedule(self) -> str:
        """Return a cron-style expression describing the offline cadence.

        Example: ``"0 */1 * * *"`` (top of every hour).
        """
        ...


__all__ = [
    "Engine",
    "EngineTier",
    "HealthState",
    "HealthStatus",
    "OfflineEngine",
    "Plugin",
    "PluginLifecycle",
    "RuntimeEngine",
]
