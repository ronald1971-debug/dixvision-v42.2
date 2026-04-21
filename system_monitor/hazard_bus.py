"""
system_monitor/hazard_bus.py
Canonical Dyon hazard bus — re-exports the async bus singleton so that
Dyon callers never need to import from ``execution.hazard``.

Rationale: per manifest §6 + §13, hazard detection belongs to the SYSTEM
domain (Dyon), not MARKET (Indira). The async bus singleton stays in one
place; this module is the authority-correct import path for Dyon code.
"""
from __future__ import annotations

from core.authority import Domain, assert_no_adapter_import

# Module-level sanity: Dyon code is about to use this; it MUST NOT have
# dragged in an INDIRA adapter through some indirect import.
assert_no_adapter_import(__name__)

from execution.hazard.async_bus import (  # noqa: E402
    HazardBus,
    HazardEvent,
    HazardSeverity,
    HazardType,
    get_hazard_bus,
)
from execution.hazard.detector import get_hazard_detector  # noqa: E402
from execution.hazard.event_emitter import get_hazard_emitter  # noqa: E402
from execution.hazard.severity_classifier import classify_severity  # noqa: E402

# Legacy alias kept for older callers that still import ``AsyncHazardBus``.
AsyncHazardBus = HazardBus


__all__ = [
    "HazardBus",
    "AsyncHazardBus",
    "HazardEvent",
    "HazardSeverity",
    "HazardType",
    "get_hazard_bus",
    "get_hazard_emitter",
    "get_hazard_detector",
    "classify_severity",
    "Domain",
]
