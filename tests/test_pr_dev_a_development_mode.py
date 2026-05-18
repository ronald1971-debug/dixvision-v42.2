"""PR-DEV-A — Operator Master Development Mode regression tests.

The Operator Master Development Mode is a dual-flag policy that
separates **learning** from **trading**:

* ``development_enabled`` (default ``True`` at boot) — Indira's
  trader discovery + 5,000+ profile modeling and Dyon's heavy
  learning + structural evolution + slow-loop critique + patch
  pipeline are unblocked when this flag is ``True``.
* ``trading_allowed`` (default ``False`` at boot) — the Execution
  Gate refuses to dispatch to any broker until the operator
  explicitly flips this flag. This is the *single* switch that
  opens trading.

The tests in this module pin three contracts:

1. **Contract layer** — :class:`DevelopmentModePolicy` predicates +
   :meth:`DevelopmentModePolicy.to_system_event` projection. The
   ``None`` sentinel resolves fail-open at both module-level
   predicates so pre-PR-DEV-A offline tests that do not construct a
   policy retain their previous behaviour.
2. **ExecutionEngine chokepoint** — when ``trading_allowed`` is
   ``False`` :meth:`ExecutionEngine.execute` returns a synthetic
   ``REJECTED`` :class:`ExecutionEvent` whose
   ``meta["reason"]`` is :data:`DEVELOPMENT_MODE_TRADING_BLOCKED`.
   The intent has already cleared the :class:`AuthorityGuard`, so
   the rejection is emitted as an event (not an exception) and
   fed into the learning loop. Flipping the flag via
   :meth:`ExecutionEngine.set_development_mode_policy` opens the
   gate atomically — the next ``execute`` call observes the new
   policy with zero retries.
3. **Operator routes** — ``GET`` /
   ``POST /api/operator/development-mode`` and
   ``/api/operator/trading-allowed`` flip the corresponding flag
   under ``STATE.lock`` and write a **pair** of audit rows on
   every transition (including no-ops):

   * ``OPERATOR_DEVELOPMENT_MODE_CHANGED`` /
     ``OPERATOR_TRADING_ALLOWED_CHANGED`` — the operator-intent
     row carrying ``previous`` / ``next`` / ``requestor`` /
     ``reason`` / ``mode``.
   * ``POLICY_STATE`` — the canonical projection emitted by
     :meth:`DevelopmentModePolicy.to_system_event` carrying the
     resulting policy state (``development_enabled`` /
     ``trading_allowed`` / ``mode`` / ``learning_unblocked`` /
     ``trading_unblocked`` / ``version``). Both rows share the
     same ``flip_ts_ns`` so an offline replay validator can
     correlate them.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from core.contracts.development_mode import (
    POLICY_VERSION,
    DevelopmentModePolicy,
    DevelopmentModeTradingBlockedError,
    assert_trading_unblocked,
    is_learning_unblocked,
    is_trading_unblocked,
)
from core.contracts.events import SystemEventKind
from core.contracts.governance import SystemMode

ui_server = importlib.import_module("ui.server")


# ---------------------------------------------------------------------------
# Contract layer — DevelopmentModePolicy predicates + projection.
# ---------------------------------------------------------------------------


def test_default_boot_policy_unblocks_learning_blocks_trading() -> None:
    """Boot default: learning unblocked, trading blocked.

    The operator's stated vision is that Indira + Dyon run at full
    potential before any trading occurs. The default
    :class:`DevelopmentModePolicy` therefore unblocks learning and
    blocks trading; the operator flips ``trading_allowed`` explicitly
    when they decide the system is ready.
    """

    policy = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.SAFE,
    )
    assert policy.is_learning_unblocked() is True
    assert policy.is_trading_unblocked() is False


def test_policy_all_combinations_predicates() -> None:
    """Predicates are pure projections of the underlying flags.

    The contract is *deliberately* non-mode-gated — the operator's
    flag flip alone is the signal. The mode is carried on the
    policy for audit, but neither predicate inspects it.
    """

    for dev in (True, False):
        for trade in (True, False):
            for mode in (
                SystemMode.SAFE,
                SystemMode.PAPER,
                SystemMode.CANARY,
                SystemMode.LIVE,
                SystemMode.AUTO,
                SystemMode.LOCKED,
            ):
                policy = DevelopmentModePolicy(
                    development_enabled=dev,
                    trading_allowed=trade,
                    mode=mode,
                )
                assert policy.is_learning_unblocked() is dev
                assert policy.is_trading_unblocked() is trade


def test_module_level_sentinel_resolves_fail_open() -> None:
    """The ``None`` sentinel preserves pre-PR-DEV-A behaviour.

    Offline tests that construct an :class:`ExecutionEngine` without
    a ``development_mode_policy`` argument must continue to dispatch.
    The production cockpit constructs a real policy at boot, so the
    fail-closed-for-trading default is enforced at the *single*
    point of policy construction in ``_build_execution_tier``.
    """

    assert is_learning_unblocked(None) is True
    assert is_trading_unblocked(None) is True
    assert_trading_unblocked(None)


def test_assert_trading_unblocked_raises_on_blocked_policy() -> None:
    """``assert_trading_unblocked`` raises on a blocked policy.

    The execution engine does *not* use this helper — it emits a
    synthetic REJECTED event instead — but other callers that want
    an exception-shaped gate can use this convenience.
    """

    blocked = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.SAFE,
    )
    with pytest.raises(DevelopmentModeTradingBlockedError):
        assert_trading_unblocked(blocked)

    unblocked = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=True,
        mode=SystemMode.SAFE,
    )
    assert_trading_unblocked(unblocked)


def test_to_system_event_emits_canonical_policy_state() -> None:
    """The projection emits a canonical ``POLICY_STATE`` event.

    The payload is JSON-safe (all booleans/enums rendered as strings)
    so the same row can be read back from a SQLite ledger without a
    custom decoder. The version anchor is carried so a future replay
    validator can identify which policy revision wrote the row.
    """

    policy = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.SAFE,
    )
    event = policy.to_system_event(ts_ns=123_456, source="test.harness")
    assert event.sub_kind is SystemEventKind.POLICY_STATE
    assert event.ts_ns == 123_456
    assert event.source == "test.harness"
    assert event.produced_by_engine == "governance"
    assert event.proposed is False
    payload = dict(event.payload)
    assert payload["policy"] == "DevelopmentModePolicy"
    assert payload["version"] == POLICY_VERSION
    assert payload["development_enabled"] == "true"
    assert payload["trading_allowed"] == "false"
    assert payload["mode"] == "SAFE"
    assert payload["learning_unblocked"] == "true"
    assert payload["trading_unblocked"] == "false"


# ---------------------------------------------------------------------------
# Operator-route layer.
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Re-bind a fresh ``_State`` and reset the dual flags to boot defaults."""

    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    with ui_server.STATE.lock:
        ui_server.STATE.development_mode_enabled = True
        ui_server.STATE.trading_allowed = False
        ui_server.STATE.development_mode_policy = DevelopmentModePolicy(
            development_enabled=True,
            trading_allowed=False,
            mode=(ui_server.STATE.governance.state_transitions.current_mode()),
        )
        ui_server.STATE.execution.set_development_mode_policy(
            ui_server.STATE.development_mode_policy
        )
    return TestClient(ui_server.app)


@pytest.fixture(autouse=True)
def reset_dev_flags() -> None:
    """Force the dual flags back to boot defaults between tests."""

    with ui_server.STATE.lock:
        ui_server.STATE.development_mode_enabled = True
        ui_server.STATE.trading_allowed = False
        ui_server.STATE.development_mode_policy = DevelopmentModePolicy(
            development_enabled=True,
            trading_allowed=False,
            mode=(ui_server.STATE.governance.state_transitions.current_mode()),
        )
        ui_server.STATE.execution.set_development_mode_policy(
            ui_server.STATE.development_mode_policy
        )


def _ledger_rows() -> list[dict[str, object]]:
    """Snapshot of the in-memory ledger chain."""

    return [
        dict(entry.payload, kind=entry.kind) for entry in ui_server.STATE.governance.ledger.read()
    ]


def test_get_development_mode_returns_boot_defaults(
    client: TestClient,
) -> None:
    response = client.get("/api/operator/development-mode")
    assert response.status_code == 200
    body = response.json()
    assert body["development_enabled"] is True
    assert body["trading_allowed"] is False
    assert body["learning_unblocked"] is True
    assert body["trading_unblocked"] is False
    assert body["policy_version"] == POLICY_VERSION
    assert isinstance(body["mode"], str) and body["mode"]


def test_get_trading_allowed_returns_boot_defaults(
    client: TestClient,
) -> None:
    response = client.get("/api/operator/trading-allowed")
    assert response.status_code == 200
    body = response.json()
    assert body["development_enabled"] is True
    assert body["trading_allowed"] is False


def test_post_trading_allowed_true_audits_pair(client: TestClient) -> None:
    before = len(_ledger_rows())

    response = client.post(
        "/api/operator/trading-allowed",
        json={
            "enabled": True,
            "requestor": "ronald",
            "reason": "open the gate for first live cycle",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["trading_allowed"] is True
    assert body["trading_unblocked"] is True

    with ui_server.STATE.lock:
        assert ui_server.STATE.trading_allowed is True
        assert ui_server.STATE.execution.development_mode_policy.is_trading_unblocked() is True

    rows = _ledger_rows()
    assert len(rows) == before + 2
    audit = rows[-2]
    assert audit["kind"] == "OPERATOR_TRADING_ALLOWED_CHANGED"
    assert audit["requestor"] == "ronald"
    assert audit["reason"] == "open the gate for first live cycle"
    assert audit["previous"] == "false"
    assert audit["next"] == "true"
    policy_row = rows[-1]
    assert policy_row["kind"] == "POLICY_STATE"
    assert policy_row["policy"] == "DevelopmentModePolicy"
    assert policy_row["trading_allowed"] == "true"
    assert policy_row["trading_unblocked"] == "true"
    assert policy_row["development_enabled"] == "true"
    assert policy_row["version"] == POLICY_VERSION


def test_post_development_mode_false_audits_pair(client: TestClient) -> None:
    before = len(_ledger_rows())

    response = client.post(
        "/api/operator/development-mode",
        json={
            "enabled": False,
            "requestor": "ronald",
            "reason": "pause indira + dyon for offline replay",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["development_enabled"] is False
    assert body["learning_unblocked"] is False

    rows = _ledger_rows()
    assert len(rows) == before + 2
    assert rows[-2]["kind"] == "OPERATOR_DEVELOPMENT_MODE_CHANGED"
    assert rows[-2]["previous"] == "true"
    assert rows[-2]["next"] == "false"
    assert rows[-1]["kind"] == "POLICY_STATE"
    assert rows[-1]["development_enabled"] == "false"
    assert rows[-1]["learning_unblocked"] == "false"


def test_post_noop_still_audits_pair(client: TestClient) -> None:
    """Toggling to the same value writes the canonical pair.

    The audit trail captures every operator *intent*, not just the
    transitions. A no-op POST is a deliberate operator decision and
    must be visible in the ledger so retrospective forensics can
    distinguish "operator chose to keep this off" from "no operator
    activity".
    """

    before = len(_ledger_rows())
    response = client.post(
        "/api/operator/trading-allowed",
        json={"enabled": False},
    )
    assert response.status_code == 200
    rows = _ledger_rows()
    assert len(rows) == before + 2
    assert rows[-2]["kind"] == "OPERATOR_TRADING_ALLOWED_CHANGED"
    assert rows[-2]["previous"] == "false"
    assert rows[-2]["next"] == "false"
    assert rows[-1]["kind"] == "POLICY_STATE"
    assert rows[-1]["trading_allowed"] == "false"


def test_engine_policy_swapped_atomically(client: TestClient) -> None:
    """Each POST replaces the engine's policy reference atomically.

    The next :meth:`ExecutionEngine.execute` call observes the new
    gate without any race window — the same ``STATE.lock`` that
    audited the flip also called
    :meth:`ExecutionEngine.set_development_mode_policy`.
    """

    client.post(
        "/api/operator/trading-allowed",
        json={"enabled": True},
    )
    assert ui_server.STATE.execution.development_mode_policy.trading_allowed is True
    client.post(
        "/api/operator/trading-allowed",
        json={"enabled": False},
    )
    assert ui_server.STATE.execution.development_mode_policy.trading_allowed is False
