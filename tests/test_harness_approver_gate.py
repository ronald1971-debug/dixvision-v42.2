"""Tests for Hardening-S1 item 1 — harness approver opt-in gate.

The harness approval shim is a documented backdoor that wraps a
:class:`SignalEvent` in a fully-approved :class:`ExecutionIntent`
without going through the live governance loop. The architecture
review flagged it as an implicit-authority surface; the fix is the
explicit env-var gate. These tests pin the contract:

* Default (gate closed) — the shim raises
  :class:`HarnessApproverDisabledError` loudly. No silent fallback.
* Env var truthy — the shim runs as before.
* Explicit ``enabled=True`` keyword — overrides a closed gate for one
  call without mutating the process env.
* Explicit ``enabled=False`` — overrides an open gate for one call.
* :func:`is_harness_approver_enabled` — stable boolean check that
  callers can use to branch without tripping the failure path.
"""

from __future__ import annotations

import pytest

from core.contracts.events import Side, SignalEvent
from governance_engine.harness_approver import (
    HARNESS_APPROVER_ENV_VAR,
    HarnessApproverDisabledError,
    approve_signal_for_execution,
    is_harness_approver_enabled,
)


def _signal(ts_ns: int = 1_000) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.9,
        produced_by_engine="intelligence_engine.signal_pipeline.orchestrator",
    )


def test_gate_closed_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env var, the shim refuses to run."""

    monkeypatch.delenv(HARNESS_APPROVER_ENV_VAR, raising=False)
    with pytest.raises(HarnessApproverDisabledError) as excinfo:
        approve_signal_for_execution(_signal(), ts_ns=1_000)
    msg = str(excinfo.value)
    assert HARNESS_APPROVER_ENV_VAR in msg
    assert "B33" in msg


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "On"])
def test_gate_open_via_env(monkeypatch: pytest.MonkeyPatch, truthy: str) -> None:
    """Truthy env values open the gate."""

    monkeypatch.setenv(HARNESS_APPROVER_ENV_VAR, truthy)
    assert is_harness_approver_enabled() is True
    intent = approve_signal_for_execution(_signal(), ts_ns=2_000)
    assert intent.approved_by_governance is True
    assert intent.governance_decision_id == "harness:auto:2000"


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "garbage"])
def test_gate_closed_for_falsy_env(
    monkeypatch: pytest.MonkeyPatch, falsy: str
) -> None:
    """Non-truthy values keep the gate closed and trigger the loud failure."""

    monkeypatch.setenv(HARNESS_APPROVER_ENV_VAR, falsy)
    assert is_harness_approver_enabled() is False
    with pytest.raises(HarnessApproverDisabledError):
        approve_signal_for_execution(_signal(), ts_ns=3_000)


def test_explicit_enabled_true_overrides_closed_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test that wants to opt in for one call without env mutation can."""

    monkeypatch.delenv(HARNESS_APPROVER_ENV_VAR, raising=False)
    intent = approve_signal_for_execution(_signal(), ts_ns=4_000, enabled=True)
    assert intent.approved_by_governance is True


def test_explicit_enabled_false_overrides_open_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller can force-fail even when the env var is set."""

    monkeypatch.setenv(HARNESS_APPROVER_ENV_VAR, "1")
    with pytest.raises(HarnessApproverDisabledError):
        approve_signal_for_execution(_signal(), ts_ns=5_000, enabled=False)


def test_is_harness_approver_enabled_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stable boolean helper never raises."""

    monkeypatch.delenv(HARNESS_APPROVER_ENV_VAR, raising=False)
    assert is_harness_approver_enabled() is False
    monkeypatch.setenv(HARNESS_APPROVER_ENV_VAR, "1")
    assert is_harness_approver_enabled() is True
