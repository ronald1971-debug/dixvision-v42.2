"""Canonical typed contracts shared by all engines.

Importing this package is **always** allowed for every engine — it is the
common allow-listed dependency in the authority-lint rule set
(``tools/authority_lint.py`` rules T1, C2, C3, W1, L1, L2, L3, B1).
"""

from core.contracts.engine import (
    Engine,
    EngineTier,
    HealthState,
    HealthStatus,
    OfflineEngine,
    Plugin,
    PluginLifecycle,
    RuntimeEngine,
)
from core.contracts.events import (
    Event,
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)

__all__ = [
    "Engine",
    "EngineTier",
    "Event",
    "EventKind",
    "ExecutionEvent",
    "ExecutionStatus",
    "HazardEvent",
    "HazardSeverity",
    "HealthState",
    "HealthStatus",
    "OfflineEngine",
    "Plugin",
    "PluginLifecycle",
    "RuntimeEngine",
    "Side",
    "SignalEvent",
    "SystemEvent",
    "SystemEventKind",
]
