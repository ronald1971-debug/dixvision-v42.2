"""
execution/hazard/severity_classifier.py
DIX VISION v42.2 — Hazard Severity Classifier
"""
from __future__ import annotations

from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType


def should_halt_trading(event: HazardEvent) -> bool:
    """Return True if this hazard should trigger trading halt."""
    return (
        event.severity == HazardSeverity.CRITICAL or
        event.hazard_type in {
            HazardType.DATA_CORRUPTION_SUSPECTED,
            HazardType.LEDGER_INCONSISTENCY,
            HazardType.API_CONNECTIVITY_FAILURE,
        }
    )

def should_enter_safe_mode(event: HazardEvent) -> bool:
    """Return True if this hazard should trigger safe mode."""
    return (
        event.severity in {HazardSeverity.HIGH, HazardSeverity.CRITICAL} or
        event.hazard_type in {HazardType.FEED_SILENCE, HazardType.EXCHANGE_TIMEOUT}
    )

def classify_response(event: HazardEvent) -> str:
    """Return recommended governance action for this hazard."""
    if event.hazard_type == HazardType.EXCHANGE_TIMEOUT:
        return "CANCEL_ALL_OPEN_ORDERS"
    if event.hazard_type == HazardType.FEED_SILENCE:
        return "PAUSE_NEW_ORDERS"
    if event.hazard_type == HazardType.EXECUTION_LATENCY_SPIKE:
        return "REDUCE_EXPOSURE"
    if event.hazard_type == HazardType.DATA_CORRUPTION_SUSPECTED:
        return "HALT_TRADING"
    if event.hazard_type == HazardType.API_CONNECTIVITY_FAILURE:
        return "HALT_TRADING"
    return "OBSERVE"
