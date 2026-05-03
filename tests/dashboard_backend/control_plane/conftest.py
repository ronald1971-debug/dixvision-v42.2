"""Shared fixtures for the dashboard control-plane test module."""

from __future__ import annotations

import pytest

from core.contracts.governance import SystemMode
from dashboard_backend.control_plane.router import ControlPlaneRouter
from governance_engine.control_plane import (
    LedgerAuthorityWriter,
    PolicyEngine,
    StateTransitionManager,
)
from governance_engine.control_plane.operator_interface_bridge import (
    OperatorInterfaceBridge,
)


@pytest.fixture
def governance_stack():
    """Build a fresh Governance stack with a real bridge."""

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(
        policy=policy, ledger=ledger, initial_mode=SystemMode.SAFE
    )
    bridge = OperatorInterfaceBridge(
        policy=policy, state_transitions=state, ledger=ledger
    )
    return ledger, policy, state, bridge


@pytest.fixture
def router(governance_stack):
    _, _, _, bridge = governance_stack
    return ControlPlaneRouter(bridge=bridge)


@pytest.fixture
def consent_payload(governance_stack):
    """Build a REQUEST_MODE payload populated with the four
    ``consent_*`` fields expected by ``OperatorInterfaceBridge`` for
    Hardening-S1 item 8 consent-required edges (SAFE→PAPER and
    LIVE→AUTO). Pulls the live ``policy.table_hash`` from the
    governance stack so the consent envelope binds to the same
    policy version the manager will see at validation time.
    """

    _, policy, _, _ = governance_stack

    def _factory(
        *,
        ts_ns: int,
        target_mode: str,
        operator_id: str = "op",
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        payload: dict[str, str] = {
            "target_mode": target_mode,
            "consent_operator_id": operator_id,
            "consent_policy_hash": policy.table_hash,
            "consent_nonce": f"nonce-conftest-{ts_ns}-{target_mode}",
            "consent_ts_ns": str(ts_ns),
        }
        if extra:
            payload.update(extra)
        return payload

    return _factory
