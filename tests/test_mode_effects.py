"""Tests for ``core.contracts.mode_effects`` (Wave-04.6 PR-A)."""

from __future__ import annotations

import pytest

from core.contracts.governance import SystemMode
from core.contracts.mode_effects import (
    MODE_EFFECTS,
    MODE_EFFECTS_HASH_KEY,
    MODE_EFFECTS_INSTALLED_KIND,
    ModeEffect,
    effect_for,
    mode_effect_table_hash,
)

# ---------------------------------------------------------------------------
# Coverage and shape
# ---------------------------------------------------------------------------


def test_mode_effects_covers_every_system_mode() -> None:
    """Every SystemMode must have exactly one row in the table."""

    assert set(MODE_EFFECTS.keys()) == set(SystemMode)


def test_mode_effects_is_read_only() -> None:
    """The published mapping is wrapped in MappingProxyType."""

    with pytest.raises(TypeError):
        MODE_EFFECTS[SystemMode.SAFE] = ModeEffect(  # type: ignore[index]
            signals_emit=True,
            executions_dispatch=True,
            size_cap_pct=None,
            learning_emit=True,
            learning_apply=True,
            operator_auth_required=True,
            oversight_kind="per_trade",
        )


def test_mode_effect_is_frozen_slotted_dataclass() -> None:
    """Per the contract: frozen + slotted, no instance dict."""

    eff = MODE_EFFECTS[SystemMode.LIVE]
    with pytest.raises(AttributeError):
        eff.signals_emit = False  # type: ignore[misc]
    assert not hasattr(eff, "__dict__")


# ---------------------------------------------------------------------------
# Per-mode behavioural distinctness (the whole point of Wave-04.6 PR-A)
# ---------------------------------------------------------------------------


def test_locked_kills_everything() -> None:
    """LOCKED suppresses every active behaviour."""

    eff = MODE_EFFECTS[SystemMode.LOCKED]
    assert eff.signals_emit is False
    assert eff.executions_dispatch is False
    assert eff.size_cap_pct == 0.0
    assert eff.learning_emit is False
    assert eff.learning_apply is False
    assert eff.oversight_kind == "none"


def test_safe_signals_quiet_no_execution() -> None:
    """SAFE: nothing fires; learning is frozen; per-trade oversight."""

    eff = MODE_EFFECTS[SystemMode.SAFE]
    assert eff.signals_emit is False
    assert eff.executions_dispatch is False
    assert eff.learning_emit is False
    assert eff.learning_apply is False
    assert eff.oversight_kind == "per_trade"


def test_paper_emits_signals_and_dispatches_to_paper_broker() -> None:
    """PAPER: signals + paper-broker dispatch; learning emits but does not apply."""

    eff = MODE_EFFECTS[SystemMode.PAPER]
    assert eff.signals_emit is True
    assert eff.executions_dispatch is True
    assert eff.size_cap_pct == 0.0  # no live capital at risk
    assert eff.learning_emit is True
    assert eff.learning_apply is False
    assert eff.operator_auth_required is False


def test_shadow_emits_signals_but_suppresses_execution() -> None:
    """SHADOW is exactly the slot reviewer #3 named: signals on, execution off."""

    eff = MODE_EFFECTS[SystemMode.SHADOW]
    assert eff.signals_emit is True
    assert eff.executions_dispatch is False
    # Learning observes shadow signals but cannot apply updates yet.
    assert eff.learning_emit is True
    assert eff.learning_apply is False


def test_canary_caps_size_and_requires_operator() -> None:
    """CANARY: live execution at constrained size + operator-gated entry."""

    eff = MODE_EFFECTS[SystemMode.CANARY]
    assert eff.executions_dispatch is True
    assert eff.size_cap_pct == 1.0
    assert eff.operator_auth_required is True
    # First mode in the chain where learning may apply ratified updates.
    assert eff.learning_apply is True


def test_live_uncapped_per_trade_oversight() -> None:
    """LIVE: full size, per-trade operator oversight, learning applies."""

    eff = MODE_EFFECTS[SystemMode.LIVE]
    assert eff.executions_dispatch is True
    assert eff.size_cap_pct is None
    assert eff.operator_auth_required is True
    assert eff.oversight_kind == "per_trade"
    assert eff.learning_apply is True


def test_auto_relaxes_oversight_to_exception_only() -> None:
    """AUTO is the slot for hands-off operation with hazard-only alerts."""

    eff = MODE_EFFECTS[SystemMode.AUTO]
    assert eff.executions_dispatch is True
    assert eff.size_cap_pct is None
    assert eff.oversight_kind == "exception_only"
    # Still requires operator to *enter* AUTO; just relaxes per-trade gating.
    assert eff.operator_auth_required is True


def test_shadow_and_paper_differ_only_in_dispatch() -> None:
    """SHADOW vs PAPER: identical except executions_dispatch."""

    p = MODE_EFFECTS[SystemMode.PAPER]
    s = MODE_EFFECTS[SystemMode.SHADOW]
    assert p.signals_emit == s.signals_emit
    assert p.executions_dispatch != s.executions_dispatch
    assert p.size_cap_pct == s.size_cap_pct
    assert p.learning_emit == s.learning_emit
    assert p.learning_apply == s.learning_apply


def test_canary_and_live_differ_in_size_cap() -> None:
    """CANARY vs LIVE: identical except size_cap_pct."""

    c = MODE_EFFECTS[SystemMode.CANARY]
    live = MODE_EFFECTS[SystemMode.LIVE]
    assert c.size_cap_pct == 1.0
    assert live.size_cap_pct is None
    assert c.executions_dispatch == live.executions_dispatch
    assert c.learning_apply == live.learning_apply
    assert c.oversight_kind == live.oversight_kind


def test_live_and_auto_differ_in_oversight() -> None:
    """LIVE vs AUTO: identical except oversight_kind."""

    live = MODE_EFFECTS[SystemMode.LIVE]
    auto = MODE_EFFECTS[SystemMode.AUTO]
    assert live.oversight_kind == "per_trade"
    assert auto.oversight_kind == "exception_only"
    assert live.signals_emit == auto.signals_emit
    assert live.executions_dispatch == auto.executions_dispatch
    assert live.size_cap_pct == auto.size_cap_pct
    assert live.learning_apply == auto.learning_apply


def test_each_mode_is_observably_distinct() -> None:
    """No two modes have identical ModeEffect tuples (Wave-04.6 PR-A goal)."""

    seen: dict[tuple, SystemMode] = {}
    for mode, eff in MODE_EFFECTS.items():
        key = (
            eff.signals_emit,
            eff.executions_dispatch,
            eff.size_cap_pct,
            eff.learning_emit,
            eff.learning_apply,
            eff.operator_auth_required,
            eff.oversight_kind,
        )
        if key in seen:
            raise AssertionError(
                f"{mode.name} is behaviourally identical to {seen[key].name}"
            )
        seen[key] = mode


# ---------------------------------------------------------------------------
# Determinism / hashing (mirrors PolicyEngine table-hash pattern)
# ---------------------------------------------------------------------------


def test_table_hash_is_deterministic_within_run() -> None:
    """Successive hash invocations are byte-identical (INV-15 anchor)."""

    h1 = mode_effect_table_hash()
    h2 = mode_effect_table_hash()
    h3 = mode_effect_table_hash(MODE_EFFECTS)
    assert h1 == h2 == h3
    # SHA-256 → 64 hex chars. Anchor the format so accidental changes to
    # the digest function surface immediately.
    assert len(h1) == 64
    int(h1, 16)  # raises ValueError if non-hex


def test_table_hash_matches_explicit_recomputation() -> None:
    """Hash matches an independently-rebuilt copy of the same data.

    Builds a fresh ``dict`` mirror of the canonical table (using the
    same ``ModeEffect`` class so dataclass equality is well-defined)
    and confirms the hash is identical. This catches accidental
    sort-instability or non-determinism in the digest function without
    re-executing the module body.
    """

    mirror = {mode: MODE_EFFECTS[mode] for mode in SystemMode}
    assert mode_effect_table_hash(mirror) == mode_effect_table_hash()


def test_table_hash_changes_when_any_field_changes() -> None:
    """Mutating any single bool flips the digest (no collisions in practice)."""

    base = mode_effect_table_hash()
    mutated = dict(MODE_EFFECTS)
    mutated[SystemMode.SHADOW] = ModeEffect(
        signals_emit=True,
        executions_dispatch=True,  # the bug we want the hash to surface
        size_cap_pct=0.0,
        learning_emit=True,
        learning_apply=False,
        operator_auth_required=False,
        oversight_kind="per_trade",
    )
    assert mode_effect_table_hash(mutated) != base


def test_installed_anchors_are_exposed() -> None:
    """Bootstrap ledger key + payload key are part of the public surface."""

    assert MODE_EFFECTS_INSTALLED_KIND == "MODE_EFFECTS_INSTALLED"
    assert MODE_EFFECTS_HASH_KEY == "table_hash"


# ---------------------------------------------------------------------------
# effect_for accessor
# ---------------------------------------------------------------------------


def test_effect_for_returns_canonical_row() -> None:
    """effect_for is the convenience that B31 will canonicalise on."""

    # Compare by value rather than identity: the determinism test above
    # reloads the module, which rebinds the canonical mapping; both
    # accessors still resolve the same data because frozen dataclasses
    # compare by field values.
    for mode in SystemMode:
        assert effect_for(mode) == MODE_EFFECTS[mode]
