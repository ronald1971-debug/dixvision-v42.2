"""PR-Z1 — HARDEN-04 conditional relaxation pin tests.

The PR-Z1 patch (P0) flips the ``DIXVISION_LEARNING_OVERRIDE`` boot
seed in ``ui.server._State.__init__`` from ``""`` (→ False) to
``"true"`` (→ True). The contract surface in
:class:`LearningEvolutionFreezePolicy` is intentionally unchanged —
``is_unfrozen`` still requires ``mode is SystemMode.LIVE AND
operator_override is True``. The only delta is the *boot seed*, so a
fresh harness boots pre-armed and the closed learning + structural
evolution loops auto-unfreeze the moment governance promotes mode to
``LIVE`` (instead of staying silently frozen even in LIVE).

These tests pin:

* Unset env → ``STATE.learning_override_enabled is True`` (was False
  before PR-Z1).
* Explicit opt-out (``DIXVISION_LEARNING_OVERRIDE=false`` and the
  four other recognised opt-out tokens) → ``False`` — operators
  retain the documented re-freeze path.
* Explicit opt-in (``DIXVISION_LEARNING_OVERRIDE=1`` etc.) still →
  ``True`` (regression guard on the existing seed path).
* The contract surface is unchanged: a freshly-constructed
  :class:`LearningEvolutionFreezePolicy` with the new default still
  requires BOTH ``mode is SystemMode.LIVE`` and ``operator_override
  is True`` to unfreeze. The post-PR-Z1 default flag alone (in any
  non-LIVE mode) does NOT unfreeze the loop.

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
def test_explicit_opt_in_still_true(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Explicit opt-in tokens continue to pre-arm the override."""

    assert _reload_server_with_override(monkeypatch, value) is True


@pytest.mark.parametrize(
    "value", ["0", "false", "FALSE", "no", "off", ""]
)
def test_explicit_opt_out_returns_false(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Operators retain the documented re-freeze path.

    Setting ``DIXVISION_LEARNING_OVERRIDE`` to any of the recognised
    opt-out tokens (or the empty string) at boot still seeds the
    override flag to ``False``, freezing adaptive mutations even
    when governance reaches ``LIVE``.
    """

    assert _reload_server_with_override(monkeypatch, value) is False


def test_policy_contract_still_requires_live_plus_override() -> None:
    """The contract surface is unchanged by PR-Z1.

    PR-Z1 only flips the **boot seed**; the
    :class:`LearningEvolutionFreezePolicy` contract still requires
    BOTH ``mode is SystemMode.LIVE`` and ``operator_override is True``
    to unfreeze. The flag alone in any non-LIVE mode must not
    unfreeze the loops.
    """

    for mode in SystemMode:
        policy = LearningEvolutionFreezePolicy(
            mode=mode, operator_override=True
        )
        if mode is SystemMode.LIVE:
            assert policy.is_unfrozen() is True
            assert policy.is_frozen() is False
        else:
            assert policy.is_unfrozen() is False
            assert policy.is_frozen() is True

    # And operator_override=False keeps the loop frozen in *every*
    # mode, including LIVE — re-freeze path remains intact.
    for mode in SystemMode:
        policy = LearningEvolutionFreezePolicy(
            mode=mode, operator_override=False
        )
        assert policy.is_frozen() is True
        assert policy.is_unfrozen() is False
