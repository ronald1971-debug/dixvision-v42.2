"""Patch pipeline ledger surface (BEHAVIOR-P5 / INV-66).

Pure projection helpers that serialise the three patch-pipeline
artefacts (:class:`PatchProposal`, :class:`StageVerdict`,
:class:`PatchApprovalDecision`) into canonical
:class:`SystemEvent` rows for the audit ledger:

* ``PATCH_PROPOSED``      — new proposal registered with the bridge.
* ``PATCH_STAGE_VERDICT`` — one stage finished and recorded a verdict.
* ``PATCH_DECISION``      — terminal decision driven by Governance
  (APPROVED / REJECTED / ROLLED_BACK).

All three payloads use ``json.dumps(sort_keys=True,
separators=(",", ":"))`` so the resulting :class:`SystemEvent` is
byte-identical across replays of the same input (INV-15).

Discipline (B1):
* This module imports only :mod:`core.contracts` types and the patch
  pipeline's own contract types — never any runtime engine.
* Pure functions of their inputs — no clock reads, no PRNG, no I/O.
* The projection rounds-trip losslessly through the matching
  ``*_from_system_event`` helpers (replay parity).
"""

from __future__ import annotations

import json

from core.contracts.events import SystemEvent, SystemEventKind
from core.contracts.learning import PatchProposal
from core.contracts.patch import (
    PatchApprovalDecision,
    PatchStage,
    StageVerdict,
)

PATCH_EVENT_VERSION: int = 1
PATCH_EVENT_SOURCE_PROPOSAL: str = "evolution.patch_pipeline.proposal"
PATCH_EVENT_SOURCE_VERDICT: str = "evolution.patch_pipeline.verdict"
PATCH_EVENT_SOURCE_DECISION: str = "governance.patch_pipeline.decision"

_LEGAL_DECISIONS: frozenset[str] = frozenset(
    {"APPROVED", "REJECTED", "ROLLED_BACK"}
)


# ---------------------------------------------------------------------------
# PATCH_PROPOSED
# ---------------------------------------------------------------------------


def proposal_as_system_event(
    proposal: PatchProposal,
    *,
    source: str = PATCH_EVENT_SOURCE_PROPOSAL,
) -> SystemEvent:
    """Project a :class:`PatchProposal` into a ``PATCH_PROPOSED`` event."""
    if not source:
        raise ValueError("proposal_as_system_event: source must be non-empty")
    if not proposal.patch_id:
        raise ValueError(
            "proposal_as_system_event: patch_id must be non-empty"
        )
    body = {
        "version": PATCH_EVENT_VERSION,
        "patch_id": proposal.patch_id,
        "ts_ns": proposal.ts_ns,
        "source": proposal.source,
        "target_strategy": proposal.target_strategy,
        "touchpoints": list(proposal.touchpoints),
        "rationale": proposal.rationale,
        "meta": {k: proposal.meta[k] for k in sorted(proposal.meta)},
    }
    payload = {
        "proposal": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    return SystemEvent(
        ts_ns=proposal.ts_ns,
        sub_kind=SystemEventKind.PATCH_PROPOSED,
        source=source,
        payload=payload,
    )


def proposal_from_system_event(event: SystemEvent) -> PatchProposal:
    """Reverse of :func:`proposal_as_system_event` (replay-parity)."""
    if event.sub_kind is not SystemEventKind.PATCH_PROPOSED:
        raise ValueError(
            "proposal_from_system_event: event must be PATCH_PROPOSED; "
            f"got {event.sub_kind}"
        )
    raw = event.payload.get("proposal")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "proposal_from_system_event: payload missing 'proposal' string"
        )
    body = json.loads(raw)
    return PatchProposal(
        ts_ns=int(body["ts_ns"]),
        patch_id=str(body["patch_id"]),
        source=str(body["source"]),
        target_strategy=str(body["target_strategy"]),
        touchpoints=tuple(body["touchpoints"]),
        rationale=str(body["rationale"]),
        meta=dict(body["meta"]),
    )


# ---------------------------------------------------------------------------
# PATCH_STAGE_VERDICT
# ---------------------------------------------------------------------------


def verdict_as_system_event(
    *,
    patch_id: str,
    verdict: StageVerdict,
    source: str = PATCH_EVENT_SOURCE_VERDICT,
) -> SystemEvent:
    """Project a :class:`StageVerdict` into a ``PATCH_STAGE_VERDICT`` event."""
    if not source:
        raise ValueError("verdict_as_system_event: source must be non-empty")
    if not patch_id:
        raise ValueError("verdict_as_system_event: patch_id must be non-empty")
    body = {
        "version": PATCH_EVENT_VERSION,
        "patch_id": patch_id,
        "ts_ns": verdict.ts_ns,
        "stage": verdict.stage.value,
        "passed": bool(verdict.passed),
        "detail": verdict.detail,
        "meta": {k: verdict.meta[k] for k in sorted(verdict.meta)},
    }
    payload = {
        "verdict": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    return SystemEvent(
        ts_ns=verdict.ts_ns,
        sub_kind=SystemEventKind.PATCH_STAGE_VERDICT,
        source=source,
        payload=payload,
    )


def verdict_from_system_event(
    event: SystemEvent,
) -> tuple[str, StageVerdict]:
    """Reverse of :func:`verdict_as_system_event` (replay-parity)."""
    if event.sub_kind is not SystemEventKind.PATCH_STAGE_VERDICT:
        raise ValueError(
            "verdict_from_system_event: event must be PATCH_STAGE_VERDICT; "
            f"got {event.sub_kind}"
        )
    raw = event.payload.get("verdict")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "verdict_from_system_event: payload missing 'verdict' string"
        )
    body = json.loads(raw)
    verdict = StageVerdict(
        ts_ns=int(body["ts_ns"]),
        stage=PatchStage(body["stage"]),
        passed=bool(body["passed"]),
        detail=str(body["detail"]),
        meta=dict(body["meta"]),
    )
    return str(body["patch_id"]), verdict


# ---------------------------------------------------------------------------
# PATCH_DECISION
# ---------------------------------------------------------------------------


def decision_as_system_event(
    decision: PatchApprovalDecision,
    *,
    source: str = PATCH_EVENT_SOURCE_DECISION,
) -> SystemEvent:
    """Project a :class:`PatchApprovalDecision` into a ``PATCH_DECISION``
    event."""
    if not source:
        raise ValueError("decision_as_system_event: source must be non-empty")
    if not decision.patch_id:
        raise ValueError(
            "decision_as_system_event: patch_id must be non-empty"
        )
    if decision.decision not in _LEGAL_DECISIONS:
        raise ValueError(
            "decision_as_system_event: decision must be one of "
            f"{sorted(_LEGAL_DECISIONS)}; got {decision.decision!r}"
        )
    body = {
        "version": PATCH_EVENT_VERSION,
        "patch_id": decision.patch_id,
        "ts_ns": decision.ts_ns,
        "decision": decision.decision,
        "reason": decision.reason,
        "final_stage": decision.final_stage.value,
        "meta": {k: decision.meta[k] for k in sorted(decision.meta)},
    }
    payload = {
        "decision": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    return SystemEvent(
        ts_ns=decision.ts_ns,
        sub_kind=SystemEventKind.PATCH_DECISION,
        source=source,
        payload=payload,
    )


def decision_from_system_event(
    event: SystemEvent,
) -> PatchApprovalDecision:
    """Reverse of :func:`decision_as_system_event` (replay-parity)."""
    if event.sub_kind is not SystemEventKind.PATCH_DECISION:
        raise ValueError(
            "decision_from_system_event: event must be PATCH_DECISION; "
            f"got {event.sub_kind}"
        )
    raw = event.payload.get("decision")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "decision_from_system_event: payload missing 'decision' string"
        )
    body = json.loads(raw)
    return PatchApprovalDecision(
        ts_ns=int(body["ts_ns"]),
        patch_id=str(body["patch_id"]),
        decision=str(body["decision"]),
        reason=str(body["reason"]),
        final_stage=PatchStage(body["final_stage"]),
        meta=dict(body["meta"]),
    )


__all__ = [
    "PATCH_EVENT_SOURCE_DECISION",
    "PATCH_EVENT_SOURCE_PROPOSAL",
    "PATCH_EVENT_SOURCE_VERDICT",
    "PATCH_EVENT_VERSION",
    "decision_as_system_event",
    "decision_from_system_event",
    "proposal_as_system_event",
    "proposal_from_system_event",
    "verdict_as_system_event",
    "verdict_from_system_event",
]
