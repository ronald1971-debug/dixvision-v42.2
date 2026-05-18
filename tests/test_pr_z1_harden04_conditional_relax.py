"""PR-Z1 + v42.2-P0-RELAX — HARDEN-04 relaxation pin tests.

Two layered changes are pinned here:

1. PR-Z1 boot-seed flip — ``DIXVISION_LEARNING_OVERRIDE`` defaults to
   ``"true"`` so a fresh harness boots pre-armed.
2. ``v42.2-P0-RELAX`` contract relaxation — the
   :class:`LearningEvolutionFreezePolicy.is_unfrozen` predicate was
   relaxed from ``mode is SystemMode.LIVE AND operator_override`` to
   ``operator_override is True``. The closed learning + structural
   evolution loops therefore unfreeze on the very first ``/api/tick``
   after boot under the pre-armed override seed, regardless of FSM
   mode (operators retain re-freeze via
   ``POST /api/operator/learning-override {enabled: false}`` or
   ``DIXVISION_LEARNING_OVERRIDE=false`` at boot).

These tests pin:

* Unset env → ``STATE.learning_override_enabled is True`` (was False
  before PR-Z1).
* Explicit opt-out (``DIXVISION_LEARNING_OVERRIDE=false`` and the
  four other recognised opt-out tokens) → ``False`` — operators
  retain the documented re-freeze path.
* Explicit opt-in (``DIXVISION_LEARNING_OVERRIDE=1`` etc.) still →
  ``True`` (regression guard on the existing seed path).
* Relaxed contract surface: a freshly-constructed
  :class:`LearningEvolutionFreezePolicy` with ``operator_override=
  True`` is **unfrozen in every mode** (SAFE / PAPER / CANARY / LIVE
  / AUTO / LOCKED); with ``operator_override=False`` it is **frozen
  in every mode**.

The tests reload ``ui.server`` under ``monkeypatch`` so a fresh
``_State`` is constructed and the seed path is exercised; the module
is restored at the end so other tests holding references keep
working. The same harness-reload pattern is used in
``tests/test_audit_p1_7_learning_override_route.py``.
"""

from __future__ import annotations

import importlib

import pytest

from core.contracts.governance import SystemMode
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)


def _reload_server_with_override(
    monkeypatch: pytest.MonkeyPatch, override_value: str | None
) -> bool:
    """Reload ``ui.server`` under the given env value; return the seed.

    ``override_value=None`` exercises the unset-env path. Any string
    is set as ``DIXVISION_LEARNING_OVERRIDE``. The harness module is
    reloaded so a fresh ``_State`` is constructed; afterwards it is
    reloaded a second time *without* the env var so the global
    ``STATE`` other tests reference is restored to its canonical
    boot configuration.
    """

    import ui.server as server_module

    monkeypatch.setenv("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")
    if override_value is None:
        monkeypatch.delenv("DIXVISION_LEARNING_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("DIXVISION_LEARNING_OVERRIDE", override_value)

    reloaded = importlib.reload(server_module)
    try:
        return bool(reloaded.STATE.learning_override_enabled)
    finally:
        monkeypatch.delenv("DIXVISION_LEARNING_OVERRIDE", raising=False)
        importlib.reload(server_module)


def test_unset_env_pre_arms_override_default_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-Z1 P0 — unset env → ``learning_override_enabled is True``.

    Before PR-Z1 this was ``False``, leaving the closed learning +
    structural evolution loops silently frozen even after governance
    promoted mode to ``LIVE``. After PR-Z1 the boot seed default is
    ``"true"`` so a fresh harness boots pre-armed.
    """

    assert _reload_server_with_override(monkeypatch, None) is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_opt_in_still_true(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Explicit opt-in tokens continue to pre-arm the override."""

    assert _reload_server_with_override(monkeypatch, value) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_explicit_opt_out_returns_false(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Operators retain the documented re-freeze path.

    Setting ``DIXVISION_LEARNING_OVERRIDE`` to any of the recognised
    opt-out tokens (or the empty string) at boot still seeds the
    override flag to ``False``, freezing adaptive mutations even
    when governance reaches ``LIVE``.
    """

    assert _reload_server_with_override(monkeypatch, value) is False


def test_policy_contract_relax_override_alone_unfreezes() -> None:
    """Relaxed (v42.2-P0-RELAX) HARDEN-04 contract surface.

    Direct operator directive dropped the ``mode is SystemMode.LIVE``
    half of the predicate. The new contract is a single operator-
    gated freeze:

    * ``operator_override=True``  → unfrozen in *every* mode
      (SAFE / PAPER / CANARY / LIVE / AUTO / LOCKED).
    * ``operator_override=False`` → frozen in *every* mode
      (re-freeze path remains intact for the operator route + the
      ``DIXVISION_LEARNING_OVERRIDE=false`` boot pin).

    The execution-side safety chain (kill switch / RiskSnapshot.halted
    / hazard throttle / FSM consent envelopes) is *unchanged* by this
    relaxation — only adaptive mutation is gated by this policy.
    """

    for mode in SystemMode:
        unfrozen = LearningEvolutionFreezePolicy(mode=mode, operator_override=True)
        assert unfrozen.is_unfrozen() is True, mode
        assert unfrozen.is_frozen() is False, mode

        frozen = LearningEvolutionFreezePolicy(mode=mode, operator_override=False)
        assert frozen.is_frozen() is True, mode
        assert frozen.is_unfrozen() is False, mode
