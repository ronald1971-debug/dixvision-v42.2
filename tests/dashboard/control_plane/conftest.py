"""Shared fixtures for the dashboard control-plane test module."""

from __future__ import annotations

import pytest

from core.contracts.governance import SystemMode
from dashboard.control_plane.router import ControlPlaneRouter
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
