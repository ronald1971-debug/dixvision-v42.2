"""Tests for A-20.3 balance reconciliation.

Coverage:

* ``WalletBalance`` invariant (``total == free + used`` within tolerance)
  + finite-value validation
* ``PositionSnapshot`` validation (empty symbol, non-finite, negative
  collateral, wrong side type)
* ``ReconciliationPolicy`` validation + ``policy_digest`` determinism
  + sensitivity
* ``expected_stake`` projection (freqtrade ``current_stake`` math)
* ``reconcile_wallet`` happy path + drift bucketing (OK / WARNING /
  HAZARD)
* ``reconcile_position`` side-flip → HAZARD; position drift bucketing
* ``reconcile`` aggregate verdict (CONSISTENT / DRIFT_WARNING /
  DRIFT_HAZARD / MISSING_CURRENCY) + sorted iteration determinism
* INV-15 byte-identical replay equality (3 runs)
* AST guards: no engine cross-imports; no typed bus event
  construction; no forbidden imports; ``# ADAPTED FROM`` header
  present
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from execution_engine.protections.reconciliation import (
    DEFAULT_HAZARD_RELATIVE,
    DEFAULT_WARNING_RELATIVE,
    NEW_PIP_DEPENDENCIES,
    DriftSeverity,
    PositionSide,
    PositionSnapshot,
    ReconciliationOutcome,
    ReconciliationPolicy,
    ReconciliationReport,
    WalletBalance,
    expected_stake,
    reconcile,
    reconcile_position,
    reconcile_wallet,
)

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "execution_engine"
    / "protections"
    / "reconciliation.py"
)


# ---------------------------------------------------------------------------
# WalletBalance validation
# ---------------------------------------------------------------------------


def test_wallet_balance_invariant_holds() -> None:
    w = WalletBalance(currency="USDT", free=70.0, used=30.0, total=100.0)
    assert w.total == 100.0


def test_wallet_balance_invariant_violation() -> None:
    with pytest.raises(ValueError):
        WalletBalance(currency="USDT", free=70.0, used=30.0, total=120.0)


def test_wallet_balance_invariant_tolerates_float_noise() -> None:
    # Floating-point noise within INVARIANT_TOTAL_TOLERANCE must pass.
    WalletBalance(currency="USDT", free=70.0, used=30.0, total=100.0 + 1e-9)


def test_wallet_balance_rejects_empty_currency() -> None:
    with pytest.raises(ValueError):
        WalletBalance(currency="", free=0.0, used=0.0, total=0.0)


def test_wallet_balance_rejects_nan() -> None:
    with pytest.raises(ValueError):
        WalletBalance(currency="USDT", free=math.nan, used=0.0, total=math.nan)


def test_wallet_balance_rejects_inf() -> None:
    with pytest.raises(ValueError):
        WalletBalance(currency="USDT", free=math.inf, used=0.0, total=math.inf)


def test_wallet_balance_is_frozen() -> None:
    w = WalletBalance(currency="USDT", free=10.0, used=0.0, total=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.free = 20.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PositionSnapshot validation
# ---------------------------------------------------------------------------


def test_position_snapshot_happy_path() -> None:
    p = PositionSnapshot(
        symbol="BTC/USDT:USDT",
        position=0.5,
        collateral=1_000.0,
        side=PositionSide.LONG,
    )
    assert p.position == 0.5


def test_position_snapshot_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        PositionSnapshot(
            symbol="",
            position=0.5,
            collateral=1_000.0,
            side=PositionSide.LONG,
        )


def test_position_snapshot_rejects_negative_collateral() -> None:
    with pytest.raises(ValueError):
        PositionSnapshot(
            symbol="BTC/USDT:USDT",
            position=0.5,
            collateral=-1.0,
            side=PositionSide.LONG,
        )


def test_position_snapshot_rejects_wrong_side_type() -> None:
    with pytest.raises(TypeError):
        PositionSnapshot(
            symbol="BTC/USDT:USDT",
            position=0.5,
            collateral=1_000.0,
            side="LONG",  # type: ignore[arg-type]
        )


def test_position_snapshot_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        PositionSnapshot(
            symbol="BTC/USDT:USDT",
            position=math.nan,
            collateral=0.0,
            side=PositionSide.LONG,
        )


# ---------------------------------------------------------------------------
# ReconciliationPolicy validation + digest
# ---------------------------------------------------------------------------


def test_policy_defaults_valid() -> None:
    p = ReconciliationPolicy()
    assert p.warning_relative <= p.hazard_relative


@pytest.mark.parametrize(
    "field",
    ["absolute_tolerance", "relative_tolerance", "warning_relative", "hazard_relative"],
)
def test_policy_rejects_negative(field: str) -> None:
    kwargs: dict[str, float] = {field: -0.01}
    with pytest.raises(ValueError):
        ReconciliationPolicy(**kwargs)  # type: ignore[arg-type]


def test_policy_rejects_inverted_buckets() -> None:
    with pytest.raises(ValueError):
        ReconciliationPolicy(warning_relative=0.5, hazard_relative=0.1)


def test_policy_digest_deterministic() -> None:
    p1 = ReconciliationPolicy()
    p2 = ReconciliationPolicy()
    assert p1.policy_digest() == p2.policy_digest()
    assert len(p1.policy_digest()) == 32


def test_policy_digest_sensitive() -> None:
    base = ReconciliationPolicy()
    variants = [
        ReconciliationPolicy(absolute_tolerance=base.absolute_tolerance + 1e-10),
        ReconciliationPolicy(relative_tolerance=base.relative_tolerance + 1e-9),
        ReconciliationPolicy(warning_relative=base.warning_relative + 1e-5),
        ReconciliationPolicy(hazard_relative=base.hazard_relative + 1e-4),
    ]
    base_digest = base.policy_digest()
    digests = {v.policy_digest() for v in variants}
    assert base_digest not in digests
    assert len(digests) == len(variants)


# ---------------------------------------------------------------------------
# expected_stake projection
# ---------------------------------------------------------------------------


def test_expected_stake_freqtrade_formula() -> None:
    # current_stake = start_cap + closed_profit + realized_profit - in_trades
    result = expected_stake(
        start_cap=1_000.0,
        total_closed_profit=150.0,
        total_realized_profit=25.0,
        total_in_trades=300.0,
    )
    assert result == pytest.approx(1_000.0 + 150.0 + 25.0 - 300.0)


def test_expected_stake_rejects_negative_start_cap() -> None:
    with pytest.raises(ValueError):
        expected_stake(
            start_cap=-1.0,
            total_closed_profit=0.0,
            total_realized_profit=0.0,
            total_in_trades=0.0,
        )


def test_expected_stake_rejects_negative_in_trades() -> None:
    with pytest.raises(ValueError):
        expected_stake(
            start_cap=0.0,
            total_closed_profit=0.0,
            total_realized_profit=0.0,
            total_in_trades=-1.0,
        )


# ---------------------------------------------------------------------------
# reconcile_wallet
# ---------------------------------------------------------------------------


def _w(currency: str, total: float) -> WalletBalance:
    return WalletBalance(currency=currency, free=total, used=0.0, total=total)


def test_reconcile_wallet_consistent() -> None:
    policy = ReconciliationPolicy()
    delta = reconcile_wallet(
        expected=_w("USDT", 1_000.0),
        actual=_w("USDT", 1_000.0),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.OK
    assert delta.absolute_delta == 0.0


def test_reconcile_wallet_currency_mismatch() -> None:
    policy = ReconciliationPolicy()
    with pytest.raises(ValueError):
        reconcile_wallet(
            expected=_w("USDT", 1.0),
            actual=_w("USDC", 1.0),
            policy=policy,
        )


def test_reconcile_wallet_warning_bucket() -> None:
    policy = ReconciliationPolicy(
        warning_relative=DEFAULT_WARNING_RELATIVE,
        hazard_relative=DEFAULT_HAZARD_RELATIVE,
    )
    # rel drift = 5e-3 → between warning (1e-3) and hazard (1e-2)
    delta = reconcile_wallet(
        expected=_w("USDT", 1_000.0),
        actual=_w("USDT", 1_005.0),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.WARNING
    assert delta.absolute_delta == pytest.approx(5.0)


def test_reconcile_wallet_hazard_bucket() -> None:
    policy = ReconciliationPolicy()
    # rel drift = 5e-2 → above hazard (1e-2)
    delta = reconcile_wallet(
        expected=_w("USDT", 1_000.0),
        actual=_w("USDT", 1_050.0),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.HAZARD


def test_reconcile_wallet_zero_expected_with_actual_drift() -> None:
    policy = ReconciliationPolicy()
    delta = reconcile_wallet(
        expected=_w("USDT", 0.0),
        actual=_w("USDT", 100.0),
        policy=policy,
    )
    # zero magnitude → relative_delta = abs_delta (no reference)
    assert delta.severity is DriftSeverity.HAZARD


# ---------------------------------------------------------------------------
# reconcile_position
# ---------------------------------------------------------------------------


def _pos(symbol: str, position: float, side: PositionSide) -> PositionSnapshot:
    return PositionSnapshot(symbol=symbol, position=position, collateral=100.0, side=side)


def test_reconcile_position_consistent() -> None:
    policy = ReconciliationPolicy()
    delta = reconcile_position(
        expected=_pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
        actual=_pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.OK


def test_reconcile_position_side_flip_is_hazard() -> None:
    policy = ReconciliationPolicy()
    delta = reconcile_position(
        expected=_pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
        actual=_pos("BTC/USDT:USDT", 0.5, PositionSide.SHORT),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.HAZARD
    assert delta.relative_delta == float("inf")
    assert "side flip" in delta.reason


def test_reconcile_position_symbol_mismatch() -> None:
    policy = ReconciliationPolicy()
    with pytest.raises(ValueError):
        reconcile_position(
            expected=_pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
            actual=_pos("ETH/USDT:USDT", 0.5, PositionSide.LONG),
            policy=policy,
        )


def test_reconcile_position_drift_bucket() -> None:
    policy = ReconciliationPolicy()
    delta = reconcile_position(
        expected=_pos("BTC/USDT:USDT", 1.0, PositionSide.LONG),
        actual=_pos("BTC/USDT:USDT", 1.10, PositionSide.LONG),
        policy=policy,
    )
    assert delta.severity is DriftSeverity.HAZARD


# ---------------------------------------------------------------------------
# reconcile aggregate
# ---------------------------------------------------------------------------


def test_reconcile_aggregate_consistent() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1_000.0)},
        actual_wallets={"USDT": _w("USDT", 1_000.0)},
    )
    assert rep.outcome is ReconciliationOutcome.CONSISTENT
    assert rep.wallet_deltas[0].severity is DriftSeverity.OK
    assert rep.position_deltas == ()


def test_reconcile_aggregate_warning() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1_000.0)},
        actual_wallets={"USDT": _w("USDT", 1_005.0)},
    )
    assert rep.outcome is ReconciliationOutcome.DRIFT_WARNING


def test_reconcile_aggregate_hazard() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1_000.0)},
        actual_wallets={"USDT": _w("USDT", 1_050.0)},
    )
    assert rep.outcome is ReconciliationOutcome.DRIFT_HAZARD


def test_reconcile_aggregate_missing_currency() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1_000.0)},
        actual_wallets={"USDC": _w("USDC", 1_000.0)},
    )
    assert rep.outcome is ReconciliationOutcome.MISSING_CURRENCY
    severities = {d.severity for d in rep.wallet_deltas}
    assert DriftSeverity.MISSING in severities


def test_reconcile_aggregate_position_side_flip_escalates() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1_000.0)},
        actual_wallets={"USDT": _w("USDT", 1_000.0)},
        expected_positions={"BTC/USDT:USDT": _pos("BTC/USDT:USDT", 0.5, PositionSide.LONG)},
        actual_positions={"BTC/USDT:USDT": _pos("BTC/USDT:USDT", 0.5, PositionSide.SHORT)},
    )
    assert rep.outcome is ReconciliationOutcome.DRIFT_HAZARD


def test_reconcile_iteration_order_sorted() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={
            "ZZZ": _w("ZZZ", 1.0),
            "AAA": _w("AAA", 1.0),
            "MMM": _w("MMM", 1.0),
        },
        actual_wallets={
            "ZZZ": _w("ZZZ", 1.0),
            "AAA": _w("AAA", 1.0),
            "MMM": _w("MMM", 1.0),
        },
    )
    assert [d.currency for d in rep.wallet_deltas] == ["AAA", "MMM", "ZZZ"]


def test_reconcile_rejects_negative_now_ns() -> None:
    with pytest.raises(ValueError):
        reconcile(
            now_ns=-1,
            expected_wallets={},
            actual_wallets={},
        )


def test_reconcile_meta_keys_sorted() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1.0)},
        actual_wallets={"USDT": _w("USDT", 1.0)},
        meta={"zzz": "z", "aaa": "a", "mmm": "m"},
    )
    assert list(rep.meta.keys()) == ["aaa", "mmm", "zzz"]


def test_reconcile_meta_default_empty() -> None:
    rep = reconcile(
        now_ns=1,
        expected_wallets={"USDT": _w("USDT", 1.0)},
        actual_wallets={"USDT": _w("USDT", 1.0)},
    )
    assert rep.meta == {}


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _run_reconcile() -> ReconciliationReport:
    return reconcile(
        now_ns=1_234,
        expected_wallets={
            "USDT": _w("USDT", 1_000.0),
            "BTC": _w("BTC", 0.5),
        },
        actual_wallets={
            "USDT": _w("USDT", 1_005.0),
            "BTC": _w("BTC", 0.5),
        },
        expected_positions={
            "BTC/USDT:USDT": _pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
            "ETH/USDT:USDT": _pos("ETH/USDT:USDT", 1.0, PositionSide.SHORT),
        },
        actual_positions={
            "BTC/USDT:USDT": _pos("BTC/USDT:USDT", 0.5, PositionSide.LONG),
            "ETH/USDT:USDT": _pos("ETH/USDT:USDT", 1.0, PositionSide.SHORT),
        },
        meta={"trace_id": "abc"},
    )


def test_three_run_byte_identical_replay() -> None:
    runs = [_run_reconcile() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    digests = {r.snapshot_digest for r in runs}
    assert len(digests) == 1


def test_snapshot_digest_sensitive_to_balance_change() -> None:
    base = _run_reconcile()
    perturbed = reconcile(
        now_ns=1_234,
        expected_wallets={
            "USDT": _w("USDT", 1_000.0),
            "BTC": _w("BTC", 0.5),
        },
        actual_wallets={
            "USDT": _w("USDT", 1_006.0),  # changed
            "BTC": _w("BTC", 0.5),
        },
    )
    assert base.snapshot_digest != perturbed.snapshot_digest


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _module_tree() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def _call_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_no_forbidden_top_level_imports() -> None:
    mods = _imported_modules(_module_tree())
    for forbidden in (
        "random",
        "asyncio",
        "os",
        "datetime",
        "time",
        "numpy",
        "torch",
        "polars",
        "pandas",
        "freqtrade",
    ):
        assert forbidden not in mods, f"forbidden import: {forbidden}"


def test_no_engine_cross_imports() -> None:
    mods = _imported_modules(_module_tree())
    for forbidden in (
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    ):
        for m in mods:
            assert not m.startswith(forbidden), f"forbidden engine import: {m}"


def test_no_typed_bus_event_construction() -> None:
    calls = _call_names(_module_tree())
    for forbidden in (
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "GovernanceDecision",
        "LearningUpdate",
        "PatchProposal",
        "TraderObservation",
    ):
        assert forbidden not in calls, f"forbidden constructor call: {forbidden}"


def test_adapted_from_header_present() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: freqtrade/wallets.py" in src


def test_new_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()
