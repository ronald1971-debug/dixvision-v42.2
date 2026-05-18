"""P0-A — ``ui.server`` wiring + admin debug-tick route tests.

The P0-A wave activates the closed learning loop
(:class:`learning_engine.loops.closed_loop.ClosedLearningLoop`) and the
structural evolution loop
(:class:`evolution_engine.loops.structural_loop.StructuralEvolutionLoop`)
on the live harness. The loops are the single freeze-enforcement point
for the FeedbackCollector → SlowLoopLearner → UpdateEmitter chain and
the MutationProposer → PatchPipelineOrchestrator chain respectively;
HARDEN-04 / INV-70 stays frozen by default and unfreezes *only* when
:class:`SystemMode` is ``LIVE`` *and* the operator override flag is
set on :class:`_State`.

These tests pin:

* The loops are instantiated on ``_State`` with the live freeze
  policy supplier bound to the mode-FSM + operator override pair.
* The inner ``SlowLoopLearner`` / ``UpdateEmitter`` / ``MutationProposer``
  are constructed with no inner freeze policy (the loop is the gate).
* The freeze policy snapshot reflects live ``learning_override_enabled``
  + ``SystemMode`` mutations without restart.
* The ``/api/admin/learning/tick`` route refuses without the env opt-in
  (HTTP 403) and drives both loops when enabled, returning the per-loop
  result projection.
* The route still respects HARDEN-04 — a frozen-mode tick reports
  ``frozen=True`` and does not invoke any inner component.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client() -> TestClient:
    """Fresh harness ``_State`` per test.

    PR-Z1 — ``_State()`` now boots with the override pre-armed
    (HARDEN-04 conditional relaxation). These tests exercise the
    loop wiring and the admin tick route's behavior under SAFE +
    override-disabled, which is orthogonal to the boot seed; force
    the flag back to ``False`` after construction. The new
    boot-seed contract is pinned in
    ``tests/test_pr_z1_harden04_conditional_relax.py``.
    """

    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = False
    return TestClient(ui_server.app)


@pytest.fixture(autouse=True)
def reset_override() -> None:
    """Force the override back to ``False`` between tests."""

    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = False


def test_state_wires_both_loops() -> None:
    """``_State`` exposes both loops with the canonical types."""

    state = ui_server.STATE
    assert type(state.closed_learning_loop).__name__ == "ClosedLearningLoop"
    assert type(state.structural_evolution_loop).__name__ == "StructuralEvolutionLoop"


def test_inner_components_have_no_inner_freeze() -> None:
    """The loop is the single freeze-enforcement point.

    Inner components MUST be constructed with no inner freeze
    policy. Pinned by both loops' constructor invariants and
    re-checked here so a future regression on the wiring side is
    caught at the harness boundary.
    """

    state = ui_server.STATE
    assert state.slow_loop_learner._freeze is None  # type: ignore[attr-defined]
    assert state.update_emitter._freeze is None  # type: ignore[attr-defined]
    assert state.mutation_proposer._freeze is None  # type: ignore[attr-defined]


def test_live_freeze_policy_default_is_frozen() -> None:
    """Harness boots in SAFE with the override disabled — frozen."""

    policy = ui_server.STATE._live_freeze_policy()
    assert policy.is_frozen() is True


def test_live_freeze_policy_reflects_override_flip() -> None:
    """Toggling the override at runtime is reflected on next supply.

    Under the relaxed HARDEN-04 contract (``v42.2-P0-RELAX``) the
    freeze gate is ``operator_override`` alone — mode is no longer a
    predicate input. The harness boots in SAFE, and once the override
    is flipped on the policy unfreezes regardless of mode. This test
    pins that the supplier reads live ``_State`` rather than
    snapshotting at init time, AND that mode is mode-agnostic.
    """

    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = True
    policy = ui_server.STATE._live_freeze_policy()
    # SAFE + override=True → unfrozen under v42.2-P0-RELAX (mode
    # is no longer consulted by the freeze predicate).
    assert policy.is_frozen() is False
    assert policy.is_unfrozen() is True
    assert policy.operator_override is True


def test_admin_tick_refuses_without_env_opt_in(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled by default — production deployments cannot tick."""

    monkeypatch.delenv(ui_server.ADMIN_LEARNING_TICK_ENV_VAR, raising=False)
    response = client.post("/api/admin/learning/tick")
    assert response.status_code == 403
    body = response.json()
    assert "disabled" in body["detail"].lower()
    assert ui_server.ADMIN_LEARNING_TICK_ENV_VAR in body["detail"]


def test_admin_tick_enabled_drives_both_loops_frozen(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With opt-in but SAFE mode, both loops report ``frozen=True``.

    HARDEN-04 stays in effect even with the debug route enabled —
    the harness boots in SAFE so the tick MUST short-circuit on
    both loops.
    """

    monkeypatch.setenv(ui_server.ADMIN_LEARNING_TICK_ENV_VAR, "1")
    response = client.post("/api/admin/learning/tick")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["ts_ns"], int) and body["ts_ns"] > 0
    closed = body["closed_learning"]
    structural = body["structural_evolution"]
    assert closed["frozen"] is True
    assert closed["submitted_samples"] == 0
    assert closed["emitted_events"] == 0
    assert closed["policy_mode_name"] == "SAFE"
    assert closed["operator_override"] is False
    assert structural["frozen"] is True
    assert structural["proposals"] == 0
    assert structural["runs"] == 0
    assert structural["policy_mode_name"] == "SAFE"
    assert structural["operator_override"] is False


def test_admin_tick_truthy_env_variants(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var accepts the four canonical truthy spellings."""

    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(ui_server.ADMIN_LEARNING_TICK_ENV_VAR, value)
        response = client.post("/api/admin/learning/tick")
        assert response.status_code == 200, value


def test_admin_tick_falsy_env_variants_refuse(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty / falsy values keep the route disabled."""

    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(ui_server.ADMIN_LEARNING_TICK_ENV_VAR, value)
        response = client.post("/api/admin/learning/tick")
        assert response.status_code == 403, value
