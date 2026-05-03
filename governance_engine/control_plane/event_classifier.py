"""GOV-CP-04 — Event Classifier.

Single switchboard at the front of the Control Plane: every inbound
event is mapped to a :class:`PipelineRoute` describing which CP
modules should run, in which order. Keeps the rest of the pipeline
free of ``isinstance`` chains.

The classifier is **stateless**: same event in → same route out
(INV-15).

**Hardening-S1 item 4 — fail-closed unknown SystemEventKinds.**

The original architecture review flagged ``_handle_system`` /
``_classify_system`` as silently discarding unknown
:class:`SystemEventKind` values via the default
:data:`_EMPTY_ROUTE`. That made the classifier *default-permissive*:
a future enum addition (or a bus payload with a kind nobody routes)
would simply drop on the floor with no audit trail.

The fix is two-layered:

* The classifier maintains explicit ``_GOVERNANCE_HANDLED_SYSTEM_KINDS``
  (kinds Governance acts on) and ``_AUDIT_ONLY_SYSTEM_KINDS`` (kinds
  routed to NOOP because they're consumed by Dyon / offline engines /
  the ledger reader, not Governance). Anything outside both sets
  returns a dedicated :data:`_UNKNOWN_KIND_ROUTE` whose ``note``
  starts with ``"unknown_system_kind"``.
* The engine's :meth:`process` path inspects the route and writes a
  loud ``UNKNOWN_SYSTEM_KIND`` ledger row before returning, so a
  silent enum addition is *impossible* — every miss surfaces in the
  audit chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

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


# ---------------------------------------------------------------------------
# Hardening-S1 item 4 — explicit SystemEventKind allowlist.
# ---------------------------------------------------------------------------

#: Kinds that drive a governance state change (UPDATE_PROPOSED,
#: PLUGIN_LIFECYCLE). The engine has dedicated handlers for these.
_GOVERNANCE_HANDLED_SYSTEM_KINDS: Final[frozenset[SystemEventKind]] = (
    frozenset(
        {
            SystemEventKind.UPDATE_PROPOSED,
            SystemEventKind.PLUGIN_LIFECYCLE,
        }
    )
)

#: Kinds that are intentionally NOOP for Governance — heartbeats,
#: health reports, ledger-internal events, projection snapshots, and
#: read-only audit surfaces consumed by Dyon, the offline calibrator,
#: the patch pipeline, or the dashboard. Listing each kind here
#: documents that the silence is *deliberate*; missing entries trip
#: the unknown-kind route below.
_AUDIT_ONLY_SYSTEM_KINDS: Final[frozenset[SystemEventKind]] = frozenset(
    {
        SystemEventKind.HEARTBEAT,
        SystemEventKind.HEALTH_REPORT,
        SystemEventKind.LEDGER_COMMIT,
        SystemEventKind.BELIEF_STATE_SNAPSHOT,
        SystemEventKind.PRESSURE_VECTOR_SNAPSHOT,
        SystemEventKind.META_DIVERGENCE,
        SystemEventKind.REWARD_BREAKDOWN,
        SystemEventKind.META_AUDIT,
        SystemEventKind.CALIBRATION_REPORT,
        SystemEventKind.SOURCE_HEARTBEAT,
        SystemEventKind.SOURCE_STALE,
        SystemEventKind.SOURCE_RECOVERED,
        SystemEventKind.SOURCE_FALLBACK_ACTIVATED,
        SystemEventKind.TRADER_OBSERVED,
        SystemEventKind.DECISION_TRACE,
        SystemEventKind.PATCH_PROPOSED,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_DECISION,
    }
)


def _unknown_kind_route(kind: object) -> PipelineRoute:
    """Build the dedicated route for an unrecognised SystemEventKind.

    The route still terminates at :data:`PipelineStage.LEDGER` so the
    engine writes a loud ``UNKNOWN_SYSTEM_KIND`` audit row; the
    ``note`` prefix lets ``governance_engine.engine.process`` detect
    the unknown-kind case explicitly without re-running the
    classifier logic.
    """

    return PipelineRoute(
        stages=(PipelineStage.LEDGER,),
        note=f"unknown_system_kind:{kind!r}",
    )


# Sanity invariant — the two sets must be disjoint and together cover
# every SystemEventKind the codebase knows about. Tested in
# tests/test_event_classifier_fail_closed.py.
_assert_disjoint = (
    _GOVERNANCE_HANDLED_SYSTEM_KINDS & _AUDIT_ONLY_SYSTEM_KINDS == frozenset()
)
assert _assert_disjoint, (
    "Hardening-S1 item 4: "
    "_GOVERNANCE_HANDLED_SYSTEM_KINDS and _AUDIT_ONLY_SYSTEM_KINDS "
    "must be disjoint"
)


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
        # Unknown EventKind — defensive (StrEnum prevents this in
        # practice, but we route to the unknown-kind path so the
        # engine writes a loud audit row instead of dropping silently.
        return _unknown_kind_route(event.kind)

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
        if event.sub_kind in _AUDIT_ONLY_SYSTEM_KINDS:
            return _EMPTY_ROUTE
        # Hardening-S1 item 4 — fail-closed unknown SystemEventKind.
        # A new enum member that nobody routes used to drop silently
        # via the default _EMPTY_ROUTE; now it surfaces a dedicated
        # route whose engine handler writes an UNKNOWN_SYSTEM_KIND
        # audit row.
        return _unknown_kind_route(event.sub_kind)


__all__ = [
    "EventClassifier",
    "PipelineRoute",
    "PipelineStage",
    "_AUDIT_ONLY_SYSTEM_KINDS",
    "_GOVERNANCE_HANDLED_SYSTEM_KINDS",
]
