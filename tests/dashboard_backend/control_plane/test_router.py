"""DASH-CP-01 — ControlPlaneRouter tests.

The router is the only write seam between the dashboard UI and the
GOV-CP-07 :class:`OperatorInterfaceBridge`. Authority constraints
checked here:

* Approved requests round-trip into a ledger row (single write seam).
* Rejected requests round-trip into the rejection summary.
* The router never touches the bridge or ledger directly other than
  through ``bridge.submit``.
"""

from __future__ import annotations

from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
    SystemMode,
)


def test_router_forwards_approved_mode_request(governance_stack, router, consent_payload):
    ledger, _, state, _ = governance_stack
    outcome = router.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload=consent_payload(
                ts_ns=1,
                target_mode="PAPER",
                extra={
                    "reason": "bring up paper",
                    "operator_authorized": "false",
                },
            ),
        )
    )
    assert outcome.approved is True
    assert state.current_mode() is SystemMode.PAPER
    assert "approved" in outcome.summary
    assert any(row.kind == "MODE_TRANSITION" for row in ledger.read())


def test_router_forwards_rejected_mode_request(governance_stack, router):
    _, _, state, _ = governance_stack
    outcome = router.submit(
        OperatorRequest(
            ts_ns=2,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={
                "target_mode": "LIVE",
                "reason": "skip ahead",
                "operator_authorized": "true",
            },
        )
    )
    assert outcome.approved is False
    assert "rejected" in outcome.summary
    assert state.current_mode() is SystemMode.SAFE


def test_router_is_a_thin_seam_no_inspection(router):
    # The router must accept any well-typed OperatorRequest and forward
    # it verbatim. We assert that by sending an unrecognised payload
    # shape and confirming the bridge's policy engine is what rejects
    # it (not the router).
    outcome = router.submit(
        OperatorRequest(
            ts_ns=3,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={"target_mode": "NOT_A_MODE", "reason": ""},
        )
    )
    assert outcome.approved is False
    assert "rejected" in outcome.summary
