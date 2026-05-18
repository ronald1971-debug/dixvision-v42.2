"""PR-DEV-C — Dyon full-potential pin (Operator Master Development Mode).

PR-DEV-A introduced :class:`DevelopmentModePolicy` (dual-flag policy:
``development_enabled`` defaults ``True``, ``trading_allowed`` defaults
``False``). PR-DEV-B threaded the policy into the
:class:`~intelligence_engine.engine.IntelligenceEngine` via a
:class:`~intelligence_engine.learning_gate.LearningGate` so Indira's
signal-emission surface runs full-bore at boot regardless of mode.

PR-DEV-C unifies the **Dyon-side freeze** with the same dual-flag
policy. The existing :class:`LearningEvolutionFreezePolicy` (HARDEN-04)
already governs both the :class:`ClosedLearningLoop` and the
:class:`StructuralEvolutionLoop` via the boot-default
``learning_override_enabled=True`` flag (PR #376) + the relaxed
single-gate predicate ``operator_override is True`` (PR #392, PR
#414's renumbering chain).

PR-DEV-C wires the live-policy supplier (``_live_freeze_policy``) to
**AND** the two operator flags together so the effective override is
``learning_override_enabled AND development_enabled``. The
operational consequence:

* A single operator flip via
  ``POST /api/operator/development-mode {enabled: false}`` pauses
  **both** Indira (via :class:`LearningGate`) **and** Dyon (via
  :class:`LearningEvolutionFreezePolicy`).
* ``POST /api/operator/learning-override {enabled: false}`` remains
  the Dyon-only switch (operator may pause Dyon while leaving Indira
  running).
* Boot defaults preserved: both flags ``True`` → both loops unfrozen
  → Dyon's adaptive-mutation surface (slow-loop learner, structural
  evolution loop, patch pipeline orchestrator) runs full-bore.

The tests in this module pin three contracts:

1. **Supplier symmetry** — ``_live_freeze_policy`` consults both
   flags. Setting either flag to ``False`` returns a frozen policy.
2. **No-regression for legacy single-flag flips** — flipping just
   ``learning_override_enabled=False`` still freezes the loops
   (existing audit-p1-7 contract). Flipping just
   ``development_enabled=False`` also freezes (new contract).
3. **Resume on re-enable** — flipping both back to ``True`` resumes
   on the very next supplier consultation (no caching). The supplier
   is pure with respect to its callers; concurrent ``STATE.lock``
   contention is the only synchronisation.
"""

from __future__ import annotations

import pytest

import ui.server as ui_server
from core.contracts.development_mode import DevelopmentModePolicy
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)


@pytest.fixture
def state() -> ui_server._State:
    """Rebuild a fresh ``_State`` with boot-default dual flags."""

    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = True
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
    return ui_server.STATE


# ---------------------------------------------------------------------------
# Supplier symmetry
# ---------------------------------------------------------------------------


def test_live_freeze_policy_unfrozen_at_boot_defaults(
    state: ui_server._State,
) -> None:
    """Boot defaults: ``learning_override_enabled=True`` +
    ``development_enabled=True`` → effective override ``True`` →
    HARDEN-04 unfrozen → Dyon's adaptive loops run."""

    policy = state._live_freeze_policy()

    assert isinstance(policy, LearningEvolutionFreezePolicy)
    assert policy.operator_override is True
    assert policy.is_unfrozen() is True
    assert policy.is_frozen() is False


def test_live_freeze_policy_freezes_when_learning_override_false(
    state: ui_server._State,
) -> None:
    """Legacy single-flag flip via
    ``POST /api/operator/learning-override`` still freezes (Dyon-only
    pause)."""

    with state.lock:
        state.learning_override_enabled = False

    policy = state._live_freeze_policy()

    assert policy.operator_override is False
    assert policy.is_frozen() is True


def test_live_freeze_policy_freezes_when_development_enabled_false(
    state: ui_server._State,
) -> None:
    """PR-DEV-C contract: a single flip via
    ``POST /api/operator/development-mode {enabled: false}`` pauses
    Dyon as well as Indira."""

    with state.lock:
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=state.trading_allowed,
            mode=state.governance.state_transitions.current_mode(),
        )

    policy = state._live_freeze_policy()

    assert policy.operator_override is False
    assert policy.is_frozen() is True


def test_live_freeze_policy_freezes_when_both_flags_false(
    state: ui_server._State,
) -> None:
    """Both flags ``False`` → frozen (idempotent intersection)."""

    with state.lock:
        state.learning_override_enabled = False
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=False,
            mode=state.governance.state_transitions.current_mode(),
        )

    policy = state._live_freeze_policy()

    assert policy.operator_override is False
    assert policy.is_frozen() is True


def test_live_freeze_policy_carries_live_mode(
    state: ui_server._State,
) -> None:
    """The policy snapshot includes the live :class:`SystemMode` so
    the audit row recorded by the loop carries the operator-visible
    mode at the time of the freeze decision."""

    policy = state._live_freeze_policy()

    expected_mode = state.governance.state_transitions.current_mode()
    assert policy.mode is expected_mode


# ---------------------------------------------------------------------------
# Resume on re-enable
# ---------------------------------------------------------------------------


def test_live_freeze_policy_resumes_on_dual_re_enable(
    state: ui_server._State,
) -> None:
    """Flip development_enabled off → frozen. Flip back on → unfrozen
    on the **very next** supplier consultation (no caching)."""

    with state.lock:
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=state.trading_allowed,
            mode=state.governance.state_transitions.current_mode(),
        )
    frozen = state._live_freeze_policy()
    assert frozen.is_frozen() is True

    with state.lock:
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=True,
            trading_allowed=state.trading_allowed,
            mode=state.governance.state_transitions.current_mode(),
        )
    resumed = state._live_freeze_policy()
    assert resumed.is_unfrozen() is True


def test_live_freeze_policy_sentinel_keeps_loops_unfrozen(
    state: ui_server._State,
) -> None:
    """Migration sentinel: ``development_mode_policy is None``
    resolves fail-open so pre-PR-DEV-A behaviour is preserved when
    ``learning_override_enabled=True``."""

    with state.lock:
        state.development_mode_policy = None  # type: ignore[assignment]
    policy = state._live_freeze_policy()

    assert policy.is_unfrozen() is True


# ---------------------------------------------------------------------------
# Closed + Structural loop pause-and-resume — wired through the supplier.
# ---------------------------------------------------------------------------


def test_closed_learning_loop_freezes_when_development_disabled(
    state: ui_server._State,
) -> None:
    """End-to-end through the wired supplier: flipping
    ``development_enabled=False`` causes :meth:`ClosedLearningLoop.tick`
    to return a frozen :class:`LoopTickResult` with no submitted
    samples / emitted events on the next tick."""

    with state.lock:
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=state.trading_allowed,
            mode=state.governance.state_transitions.current_mode(),
        )

    result = state.closed_learning_loop.tick(ts_ns=1)

    assert result.frozen is True
    assert result.submitted_samples == ()
    assert result.emitted_events == ()
    assert result.snapshot is None


def test_structural_evolution_loop_freezes_when_development_disabled(
    state: ui_server._State,
) -> None:
    """End-to-end through the wired supplier: flipping
    ``development_enabled=False`` causes
    :meth:`StructuralEvolutionLoop.tick` to return a frozen result on
    the next tick — no proposer / orchestrator invocation."""

    with state.lock:
        state.development_mode_policy = DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=state.trading_allowed,
            mode=state.governance.state_transitions.current_mode(),
        )

    result = state.structural_evolution_loop.tick(ts_ns=1)

    assert result.frozen is True
