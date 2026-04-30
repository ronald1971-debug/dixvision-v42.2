"""Wave-Stress-Tests — adversarial coverage of the mode-effect table.

The mode-effect table (PR #110, Wave-04.6 PR-A) is the single source of
truth for "what does each engine do under each :class:`SystemMode`".
A silent edit that adds / re-orders / mistypes a row is the kind of
governance hazard the build plan calls out as worst-case (INV-70 /
HARDEN-04). This file hammers the table from every angle:

* Every :class:`SystemMode` member must have a :class:`ModeEffect` row.
* Every row must have field types matching the contract.
* The table hash must be content-only (insensitive to ``dict`` insertion
  order, sensitive to any field flip).
* :func:`equity_notional_cap_qty` must be a pure deterministic function
  of ``(mode, equity, price)`` and must reject negative equity / non-
  positive price.
* Per-mode invariants the build plan documents (e.g. SHADOW emits
  signals but never dispatches; LOCKED is fully gated; CANARY caps at
  1%) must hold.

Backend-only — no UI surface. INV-15: every assertion is a pure
function of the canonical table; no I/O, no clock, no PRNG without a
seeded :class:`random.Random`.
"""

from __future__ import annotations

import math
import random
from dataclasses import fields, replace

import pytest

from core.contracts.governance import SystemMode
from core.contracts.mode_effects import (
    MODE_EFFECTS,
    MODE_EFFECTS_HASH_KEY,
    MODE_EFFECTS_INSTALLED_KIND,
    ModeEffect,
    effect_for,
    equity_notional_cap_qty,
    mode_effect_table_hash,
)

# ---------------------------------------------------------------------------
# 1. Coverage — every SystemMode is in the table, every field is the right type
# ---------------------------------------------------------------------------


def test_every_system_mode_has_a_row() -> None:
    """No SystemMode may be missing from MODE_EFFECTS — silent default = bug."""
    missing = [m for m in SystemMode if m not in MODE_EFFECTS]
    assert missing == [], f"SystemMode(s) missing from MODE_EFFECTS: {missing}"


def test_no_extra_rows_in_table() -> None:
    """The table must not declare modes that aren't in SystemMode."""
    extras = [m for m in MODE_EFFECTS if m not in set(SystemMode)]
    assert extras == [], f"unexpected modes in MODE_EFFECTS: {extras}"


def test_every_row_field_has_expected_type() -> None:
    """Field-level type assertion for every cell in the table."""
    for mode, eff in MODE_EFFECTS.items():
        assert isinstance(eff, ModeEffect), f"{mode}: not a ModeEffect"
        assert isinstance(eff.signals_emit, bool), f"{mode}: signals_emit"
        assert isinstance(eff.executions_dispatch, bool), (
            f"{mode}: executions_dispatch"
        )
        assert eff.size_cap_pct is None or isinstance(
            eff.size_cap_pct, float
        ), f"{mode}: size_cap_pct"
        assert isinstance(eff.learning_emit, bool), f"{mode}: learning_emit"
        assert isinstance(eff.learning_apply, bool), f"{mode}: learning_apply"
        assert isinstance(eff.operator_auth_required, bool), (
            f"{mode}: operator_auth_required"
        )
        assert eff.oversight_kind in {
            "per_trade",
            "exception_only",
            "none",
        }, f"{mode}: oversight_kind"


# ---------------------------------------------------------------------------
# 2. Per-mode invariants documented in the build plan
# ---------------------------------------------------------------------------


def test_locked_is_fully_gated() -> None:
    """LOCKED must be a no-emit, no-dispatch, no-learn cell."""
    eff = effect_for(SystemMode.LOCKED)
    assert eff.signals_emit is False
    assert eff.executions_dispatch is False
    assert eff.learning_emit is False
    assert eff.learning_apply is False


def test_safe_does_not_emit_or_dispatch() -> None:
    """SAFE allows operator action but no autonomous behavior."""
    eff = effect_for(SystemMode.SAFE)
    assert eff.signals_emit is False
    assert eff.executions_dispatch is False


def test_paper_emits_and_dispatches_to_paper_broker() -> None:
    """PAPER is signals-on + executions-on (paper broker)."""
    eff = effect_for(SystemMode.PAPER)
    assert eff.signals_emit is True
    assert eff.executions_dispatch is True


def test_shadow_emits_signals_but_never_dispatches() -> None:
    """SHADOW = signals-on, execution-off (the whole point of SHADOW).

    Reviewer #3's primary callout: SHADOW is required *before* CANARY,
    and is operationally distinct from PAPER (no broker dispatch at all).
    """
    eff = effect_for(SystemMode.SHADOW)
    assert eff.signals_emit is True, "SHADOW must emit signals"
    assert eff.executions_dispatch is False, (
        "SHADOW must NOT dispatch executions — that is the SHADOW invariant"
    )


def test_canary_is_capped_at_one_percent_equity() -> None:
    """CANARY must cap notional at 1% of equity (PR #112 invariant)."""
    eff = effect_for(SystemMode.CANARY)
    assert eff.size_cap_pct == 1.0
    assert eff.executions_dispatch is True
    assert eff.signals_emit is True


def test_live_is_uncapped() -> None:
    """LIVE has no size_cap_pct; risk is bounded by RiskCache only."""
    eff = effect_for(SystemMode.LIVE)
    assert eff.size_cap_pct is None
    assert eff.executions_dispatch is True


def test_auto_uses_exception_only_oversight() -> None:
    """AUTO = LIVE behavior + exception-only operator oversight (PR #115)."""
    eff = effect_for(SystemMode.AUTO)
    assert eff.oversight_kind == "exception_only"
    assert eff.executions_dispatch is True


# ---------------------------------------------------------------------------
# 3. Cross-cutting invariants — relationships between fields
# ---------------------------------------------------------------------------


def test_dispatch_implies_signals_emit() -> None:
    """A mode cannot dispatch executions without first emitting signals."""
    for mode, eff in MODE_EFFECTS.items():
        if eff.executions_dispatch:
            assert eff.signals_emit, (
                f"{mode} dispatches executions but does not emit signals — "
                "Triad-Lock violation: nothing to dispatch"
            )


def test_learning_apply_implies_learning_emit() -> None:
    """Cannot apply learning updates if not emitting them in the first place."""
    for mode, eff in MODE_EFFECTS.items():
        if eff.learning_apply:
            assert eff.learning_emit, (
                f"{mode}: learning_apply=True but learning_emit=False — "
                "would gate every UpdateValidator call on input that "
                "never arrives"
            )


def test_learning_emit_only_when_signals_emit() -> None:
    """Without a signal stream there is no learning input to emit (HARDEN-04).

    LOCKED / SAFE: no signals -> no learning input.
    SHADOW / PAPER / CANARY / LIVE / AUTO: signals on -> learning input
    flows; whether *apply* happens is a separate gate (CANARY/LIVE/AUTO
    only, per PR #114).
    """
    for mode, eff in MODE_EFFECTS.items():
        if eff.learning_emit:
            assert eff.signals_emit, (
                f"{mode}: learning_emit=True but signals_emit=False"
            )


def test_size_cap_zero_means_dispatch_suppressed_or_paper() -> None:
    """``size_cap_pct == 0.0`` is meaningful only for non-equity modes.

    Documented in :func:`equity_notional_cap_qty` docstring: size_cap_pct
    of ``0.0`` is moot for dispatch-suppressed modes (LOCKED, SAFE,
    SHADOW) and for PAPER (paper broker is not equity-bearing).
    """
    for mode, eff in MODE_EFFECTS.items():
        if eff.size_cap_pct == 0.0:
            assert (
                not eff.executions_dispatch
                or mode is SystemMode.PAPER
            ), (
                f"{mode}: size_cap_pct=0.0 but is dispatching to a "
                "non-paper venue — would block every order"
            )


# ---------------------------------------------------------------------------
# 4. Table-hash stability under reordering / mutation
# ---------------------------------------------------------------------------


def test_table_hash_is_deterministic_across_calls() -> None:
    """Hash must be a pure function — repeated calls return the same value."""
    h1 = mode_effect_table_hash()
    h2 = mode_effect_table_hash()
    assert h1 == h2
    assert isinstance(h1, str)
    # SHA-256 hex digest — 64 hex chars.
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_table_hash_is_insensitive_to_dict_order() -> None:
    """Reshuffling the rows must not change the hash (canonical-sort)."""
    canon_hash = mode_effect_table_hash()

    rng = random.Random(20260420)
    items = list(MODE_EFFECTS.items())
    for trial in range(50):
        rng.shuffle(items)
        shuffled = dict(items)
        h = mode_effect_table_hash(shuffled)
        assert h == canon_hash, (
            f"trial {trial}: hash diverged after shuffle"
        )


def test_table_hash_changes_when_any_field_flips() -> None:
    """Every field flip in any row must change the hash (no field is dead)."""
    canon_hash = mode_effect_table_hash()

    for mode in MODE_EFFECTS:
        baseline = MODE_EFFECTS[mode]
        for f in fields(ModeEffect):
            new_val = _flip_field(baseline, f.name)
            mutated = replace(baseline, **{f.name: new_val})
            tbl = dict(MODE_EFFECTS)
            tbl[mode] = mutated
            h = mode_effect_table_hash(tbl)
            assert h != canon_hash, (
                f"flipping {mode}.{f.name} from {getattr(baseline, f.name)!r} "
                f"to {new_val!r} did not change the hash"
            )


def _flip_field(eff: ModeEffect, field_name: str) -> object:
    """Return a value for ``field_name`` that differs from ``eff``."""
    cur = getattr(eff, field_name)
    if isinstance(cur, bool):
        return not cur
    if field_name == "size_cap_pct":
        return None if cur is not None else 0.5
    if field_name == "oversight_kind":
        return "none" if cur != "none" else "per_trade"
    raise AssertionError(f"untested field: {field_name}")


def test_anchor_constants_are_strings() -> None:
    """Wave-04.6 PR-A anchor constants are string-typed and non-empty."""
    assert isinstance(MODE_EFFECTS_INSTALLED_KIND, str)
    assert MODE_EFFECTS_INSTALLED_KIND
    assert isinstance(MODE_EFFECTS_HASH_KEY, str)
    assert MODE_EFFECTS_HASH_KEY


# ---------------------------------------------------------------------------
# 5. effect_for / equity_notional_cap_qty boundary fuzz
# ---------------------------------------------------------------------------


def test_effect_for_round_trips_for_every_mode() -> None:
    for mode in SystemMode:
        assert effect_for(mode) is MODE_EFFECTS[mode]


def test_equity_notional_cap_rejects_negative_equity() -> None:
    with pytest.raises(ValueError, match="equity must be >= 0"):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=-0.01, price=1.0
        )


def test_equity_notional_cap_rejects_zero_price() -> None:
    with pytest.raises(ValueError, match="price must be > 0"):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=10_000.0, price=0.0
        )


def test_equity_notional_cap_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="price must be > 0"):
        equity_notional_cap_qty(
            mode=SystemMode.CANARY, equity=10_000.0, price=-1.5
        )


def test_equity_notional_cap_uncapped_modes_return_none() -> None:
    """LIVE and AUTO are size_cap_pct=None — must surface as None."""
    for mode in (SystemMode.LIVE, SystemMode.AUTO):
        assert (
            equity_notional_cap_qty(
                mode=mode, equity=10_000.0, price=10.0
            )
            is None
        )


def test_equity_notional_cap_zero_pct_modes_return_none() -> None:
    """0.0% caps surface as ``None`` because there is no positive cap.

    Documented contract: ``size_cap_pct`` of ``0.0`` is moot for
    dispatch-suppressed modes and PAPER.
    """
    for mode in (
        SystemMode.LOCKED,
        SystemMode.SAFE,
        SystemMode.SHADOW,
        SystemMode.PAPER,
    ):
        assert (
            equity_notional_cap_qty(
                mode=mode, equity=10_000.0, price=10.0
            )
            is None
        )


def test_equity_notional_cap_canary_at_one_percent() -> None:
    """CANARY's 1% must reflect in qty: equity * 0.01 / price."""
    qty = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=10_000.0, price=100.0
    )
    assert qty is not None
    # 10_000 * 0.01 / 100 = 1.0
    assert math.isclose(qty, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_equity_notional_cap_zero_equity_returns_zero_qty() -> None:
    """Zero equity = zero qty — boundary, not error."""
    qty = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=0.0, price=100.0
    )
    assert qty == 0.0


def test_equity_notional_cap_scales_linearly_with_equity() -> None:
    """qty must scale linearly in equity for a fixed mode + price."""
    qty_low = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=1_000.0, price=10.0
    )
    qty_hi = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=10_000.0, price=10.0
    )
    assert qty_low is not None and qty_hi is not None
    assert math.isclose(qty_hi / qty_low, 10.0, rel_tol=1e-9)


def test_equity_notional_cap_inversely_scales_with_price() -> None:
    """qty must scale inversely with price for a fixed mode + equity."""
    qty_cheap = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=10_000.0, price=10.0
    )
    qty_dear = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=10_000.0, price=100.0
    )
    assert qty_cheap is not None and qty_dear is not None
    assert math.isclose(qty_cheap / qty_dear, 10.0, rel_tol=1e-9)


def test_equity_notional_cap_pure_function() -> None:
    """Determinism — same inputs, byte-identical output across many calls."""
    rng = random.Random(20260420)
    for _ in range(200):
        mode = rng.choice(list(SystemMode))
        equity = rng.uniform(0.0, 1_000_000.0)
        price = rng.uniform(0.000_001, 100_000.0)
        a = equity_notional_cap_qty(mode=mode, equity=equity, price=price)
        b = equity_notional_cap_qty(mode=mode, equity=equity, price=price)
        if a is None:
            assert b is None
        else:
            assert b is not None
            assert math.isclose(a, b, rel_tol=0.0, abs_tol=0.0)


def test_equity_notional_cap_fuzz_does_not_raise_for_valid_inputs() -> None:
    """For every mode and any (equity >= 0, price > 0) it must not raise."""
    rng = random.Random(4242)
    for _ in range(500):
        mode = rng.choice(list(SystemMode))
        equity = rng.uniform(0.0, 1e12)
        price = rng.uniform(1e-9, 1e9)
        # Should not raise.
        qty = equity_notional_cap_qty(mode=mode, equity=equity, price=price)
        if qty is not None:
            assert qty >= 0.0
            assert math.isfinite(qty)


def test_equity_notional_cap_extreme_equity_does_not_overflow() -> None:
    """Float overflow on very large equity must not silently produce inf."""
    qty = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=1e308, price=1.0
    )
    assert qty is not None
    # Result is 1e308 * 0.01 / 1.0 = 1e306; still finite.
    assert math.isfinite(qty)


def test_equity_notional_cap_extreme_small_price_does_not_overflow() -> None:
    """Tiny price + 1% cap: result is large but must remain finite."""
    qty = equity_notional_cap_qty(
        mode=SystemMode.CANARY, equity=1e6, price=1e-9
    )
    assert qty is not None
    # 1e6 * 0.01 / 1e-9 = 1e13 — large but finite.
    assert math.isfinite(qty)
    assert qty > 0.0
