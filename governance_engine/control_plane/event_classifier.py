"""GOV-CP-04 — Event Classifier.

Single switchboard at the front of the Control Plane: every inbound
event is mapped to a :class:`PipelineRoute` describing which CP
modules should run, in which order. Keeps the rest of the pipeline
free of ``isinstance`` chains.

The classifier is **stateless**: same event in → same route out
(INV-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.contracts.events import (
    Event,
    EventKind,
    HazardEvent,
    HazardSeverity,
    SystemEvent,
    SystemEventKind,
)


class PipelineStage(StrEnum):
    POLICY = "POLICY"
    RISK = "RISK"
    COMPLIANCE = "COMPLIANCE"
    STATE_TRANSITION = "STATE_TRANSITION"
    LEDGER = "LEDGER"
    NOOP = "NOOP"


@dataclass(frozen=True, slots=True)
class PipelineRoute:
    """The ordered list of stages an event traverses."""

    stages: tuple[PipelineStage, ...]
    emergency_lock: bool = False
    note: str = ""


_EMPTY_ROUTE = PipelineRoute(stages=(PipelineStage.NOOP,), note="default")


class EventClassifier:
    name: str = "event_classifier"
    spec_id: str = "GOV-CP-04"

    def classify(self, event: Event) -> PipelineRoute:
        if event.kind is EventKind.HAZARD:
            return self._classify_hazard(event)  # type: ignore[arg-type]
        if event.kind is EventKind.SYSTEM:
            return self._classify_system(event)  # type: ignore[arg-type]
        if event.kind is EventKind.SIGNAL:
            return PipelineRoute(
                stages=(
                    PipelineStage.POLICY,
                    PipelineStage.RISK,
                    PipelineStage.COMPLIANCE,
                    PipelineStage.LEDGER,
                ),
                note="signal_to_order_gate",
            )
        if event.kind is EventKind.EXECUTION:
            return PipelineRoute(
                stages=(PipelineStage.LEDGER,),
                note="execution_audit_only",
            )
        return _EMPTY_ROUTE

    # ------------------------------------------------------------------
    # Sub-classifiers
    # ------------------------------------------------------------------

    def _classify_hazard(self, event: HazardEvent) -> PipelineRoute:
        if event.severity in (HazardSeverity.HIGH, HazardSeverity.CRITICAL):
            return PipelineRoute(
                stages=(
                    PipelineStage.STATE_TRANSITION,
                    PipelineStage.LEDGER,
                ),
                emergency_lock=True,
                note=f"hazard:{event.code}:lock",
            )
        return PipelineRoute(
            stages=(PipelineStage.LEDGER,),
            note=f"hazard:{event.code}:audit",
        )

    def _classify_system(self, event: SystemEvent) -> PipelineRoute:
        if event.sub_kind is SystemEventKind.UPDATE_PROPOSED:
            return PipelineRoute(
                stages=(
                    PipelineStage.POLICY,
                    PipelineStage.COMPLIANCE,
                    PipelineStage.LEDGER,
                ),
                note="offline_update_proposal",
            )
        if event.sub_kind is SystemEventKind.PLUGIN_LIFECYCLE:
            return PipelineRoute(
                stages=(
                    PipelineStage.POLICY,
                    PipelineStage.LEDGER,
                ),
                note="plugin_lifecycle_change",
            )
        # HEARTBEAT, HEALTH_REPORT, LEDGER_COMMIT — no governance action
        # required; these are read by Dyon, not written by Governance.
        return _EMPTY_ROUTE


__all__ = ["EventClassifier", "PipelineRoute", "PipelineStage"]
