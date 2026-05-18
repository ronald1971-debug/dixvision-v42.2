"""Balance reconciliation — A-20.3 / EXEC-protections.

Pure-function balance reconciliation between *expected* and *actual*
wallet snapshots, plus a per-pair position reconciliation pass. The
reconciler is a value-object emitter; it never mutates the ledger,
never reads a clock, and never constructs typed bus events.

# ADAPTED FROM: freqtrade/wallets.py (Wallet / PositionWallet shape; expected-balance math)
# GPL-3.0 mitigation: only the *value-object shape* and the
# ``total == free + used`` invariant + the ``current_stake = start_cap
# + tot_profit - tot_in_trades`` projection are reused. No freqtrade
# class is imported, subclassed, or referenced.

Tier discipline
---------------

* OFFLINE_ONLY pure functions — caller supplies all balances and
  ``now_ns``; the reconciler never reads a clock and never talks to a
  venue.
* INV-15 byte-identical replay: deterministic ``BLAKE2b-16``
  ``snapshot_digest`` over the canonical sorted-key text projection
  of both wallet maps + the position map; 3-run replay equality
  pinned in tests.
* B27 / B28 / INV-71 authority symmetry: this module returns value
  objects only — no ``HazardEvent`` / ``SignalEvent`` /
  ``ExecutionEvent`` / ``GovernanceDecision`` / ``LearningUpdate`` /
  ``PatchProposal`` / ``TraderObservation`` constructor calls.
  Pinned by AST tests.
* No ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``evolution_engine`` /
  ``learning_engine`` imports (B1).
* No ``random`` / ``asyncio`` / ``os`` / ``datetime`` / ``time`` /
  ``numpy`` / ``torch`` / ``polars`` / ``pandas`` / ``freqtrade``
  imports.

``NEW_PIP_DEPENDENCIES = ()`` — pure stdlib only.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Final

# ---------------------------------------------------------------------------
# Constants (canonical defaults).
# ---------------------------------------------------------------------------

DEFAULT_ABSOLUTE_TOLERANCE: Final[float] = 1e-8
"""Absolute balance delta below which a currency is considered consistent."""

DEFAULT_RELATIVE_TOLERANCE: Final[float] = 1e-6
"""Relative balance delta (vs expected.total) below which we ignore drift."""

DEFAULT_WARNING_RELATIVE: Final[float] = 1e-3
"""Relative drift above this raises WARNING."""

DEFAULT_HAZARD_RELATIVE: Final[float] = 1e-2
"""Relative drift above this raises HAZARD."""

INVARIANT_TOTAL_TOLERANCE: Final[float] = 1e-6
"""Tolerance for the ``total == free + used`` per-wallet invariant."""

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PositionSide(enum.Enum):
    """Direction for futures positions (mirrors freqtrade ``side``)."""

    LONG = "LONG"
    SHORT = "SHORT"


class ReconciliationOutcome(enum.Enum):
    """High-level reconciler verdict."""

    CONSISTENT = "CONSISTENT"
    DRIFT_WARNING = "DRIFT_WARNING"
    DRIFT_HAZARD = "DRIFT_HAZARD"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    MISSING_CURRENCY = "MISSING_CURRENCY"


class DriftSeverity(enum.Enum):
    """Per-currency / per-position drift bucket."""

    OK = "OK"
    WARNING = "WARNING"
    HAZARD = "HAZARD"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    MISSING = "MISSING"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class WalletBalance:
    """One currency's balance snapshot.

    Mirrors freqtrade ``Wallet`` shape but adds the invariant assertion
    ``total ≈ free + used`` so callers cannot construct nonsensical
    snapshots.
    """

    currency: str
    free: float
    used: float
    total: float

    def __post_init__(self) -> None:
        if not self.currency:
            raise ValueError("currency must be non-empty")
        for name, value in (("free", self.free), ("used", self.used), ("total", self.total)):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{name} must be finite")
        # The total = free + used invariant is the freqtrade convention.
        # We enforce it here so the reconciler can rely on it.
        if not math.isclose(self.total, self.free + self.used, abs_tol=INVARIANT_TOTAL_TOLERANCE):
            raise ValueError(
                f"wallet invariant violated: total={self.total} != "
                f"free({self.free}) + used({self.used})"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """One per-pair futures position snapshot.

    Mirrors freqtrade ``PositionWallet`` shape; ``leverage`` is
    intentionally omitted — see freqtrade comment 'Don't use this -
    it's not guaranteed to be set'.
    """

    symbol: str
    position: float
    collateral: float
    side: PositionSide

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        for name, value in (("position", self.position), ("collateral", self.collateral)):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{name} must be finite")
        if self.collateral < 0:
            raise ValueError("collateral must be >= 0")
        if not isinstance(self.side, PositionSide):
            raise TypeError("side must be a PositionSide enum")


@dataclasses.dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    """Frozen reconciler thresholds."""

    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    warning_relative: float = DEFAULT_WARNING_RELATIVE
    hazard_relative: float = DEFAULT_HAZARD_RELATIVE

    def __post_init__(self) -> None:
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
            ("warning_relative", self.warning_relative),
            ("hazard_relative", self.hazard_relative),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.warning_relative > self.hazard_relative:
            raise ValueError("warning_relative must be <= hazard_relative")

    def canonical_text(self) -> str:
        return (
            f"absolute_tolerance={self.absolute_tolerance!r}|"
            f"relative_tolerance={self.relative_tolerance!r}|"
            f"warning_relative={self.warning_relative!r}|"
            f"hazard_relative={self.hazard_relative!r}"
        )

    def policy_digest(self) -> str:
        return hashlib.blake2b(self.canonical_text().encode("utf-8"), digest_size=16).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class WalletDelta:
    """Per-currency reconciliation delta."""

    currency: str
    expected_total: float
    actual_total: float
    absolute_delta: float
    relative_delta: float
    severity: DriftSeverity
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class PositionDelta:
    """Per-symbol position reconciliation delta."""

    symbol: str
    expected_position: float
    actual_position: float
    absolute_delta: float
    relative_delta: float
    severity: DriftSeverity
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Full reconciliation snapshot returned by :func:`reconcile`."""

    outcome: ReconciliationOutcome
    now_ns: int
    wallet_deltas: tuple[WalletDelta, ...]
    position_deltas: tuple[PositionDelta, ...]
    snapshot_digest: str
    policy_digest: str
    reason: str
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _bucket_severity(
    abs_delta: float, rel_delta: float, policy: ReconciliationPolicy
) -> DriftSeverity:
    if abs_delta <= policy.absolute_tolerance or rel_delta <= policy.relative_tolerance:
        return DriftSeverity.OK
    if rel_delta >= policy.hazard_relative:
        return DriftSeverity.HAZARD
    if rel_delta >= policy.warning_relative:
        return DriftSeverity.WARNING
    return DriftSeverity.OK


def _wallet_canonical_text(wallets: Mapping[str, WalletBalance]) -> str:
    parts = []
    for currency in sorted(wallets):
        w = wallets[currency]
        parts.append(f"{currency}:free={w.free!r},used={w.used!r},total={w.total!r}")
    return "|".join(parts)


def _position_canonical_text(positions: Mapping[str, PositionSnapshot]) -> str:
    parts = []
    for symbol in sorted(positions):
        p = positions[symbol]
        parts.append(f"{symbol}:pos={p.position!r},collat={p.collateral!r},side={p.side.value}")
    return "|".join(parts)


def _snapshot_digest(
    expected: Mapping[str, WalletBalance],
    actual: Mapping[str, WalletBalance],
    expected_positions: Mapping[str, PositionSnapshot],
    actual_positions: Mapping[str, PositionSnapshot],
) -> str:
    text = (
        f"E_W={_wallet_canonical_text(expected)}|"
        f"A_W={_wallet_canonical_text(actual)}|"
        f"E_P={_position_canonical_text(expected_positions)}|"
        f"A_P={_position_canonical_text(actual_positions)}"
    )
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _relative(actual_magnitude: float, abs_delta: float) -> float:
    if actual_magnitude <= 0.0:
        # No reference magnitude → any non-zero delta is "infinite".
        return abs_delta if abs_delta > 0.0 else 0.0
    return abs_delta / actual_magnitude


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_wallet(
    *,
    expected: WalletBalance,
    actual: WalletBalance,
    policy: ReconciliationPolicy,
) -> WalletDelta:
    """Reconcile a single currency. Pure function."""
    if expected.currency != actual.currency:
        raise ValueError(
            f"currency mismatch: expected={expected.currency!r} actual={actual.currency!r}"
        )
    abs_delta = abs(expected.total - actual.total)
    rel_delta = _relative(abs(expected.total), abs_delta)
    severity = _bucket_severity(abs_delta, rel_delta, policy)
    if severity is DriftSeverity.OK:
        reason = "within tolerance"
    else:
        reason = (
            f"drift {severity.value}: |Δ|={abs_delta:.6g} "
            f"rel={rel_delta:.6g} expected={expected.total!r} actual={actual.total!r}"
        )
    return WalletDelta(
        currency=expected.currency,
        expected_total=expected.total,
        actual_total=actual.total,
        absolute_delta=abs_delta,
        relative_delta=rel_delta,
        severity=severity,
        reason=reason,
    )


def reconcile_position(
    *,
    expected: PositionSnapshot,
    actual: PositionSnapshot,
    policy: ReconciliationPolicy,
) -> PositionDelta:
    """Reconcile a single per-pair position. Pure function."""
    if expected.symbol != actual.symbol:
        raise ValueError(f"symbol mismatch: expected={expected.symbol!r} actual={actual.symbol!r}")
    if expected.side is not actual.side:
        # Side flip is always a HAZARD regardless of magnitude.
        return PositionDelta(
            symbol=expected.symbol,
            expected_position=expected.position,
            actual_position=actual.position,
            absolute_delta=abs(expected.position - actual.position),
            relative_delta=float("inf"),
            severity=DriftSeverity.HAZARD,
            reason=(f"side flip: expected={expected.side.value} actual={actual.side.value}"),
        )
    abs_delta = abs(expected.position - actual.position)
    rel_delta = _relative(abs(expected.position), abs_delta)
    severity = _bucket_severity(abs_delta, rel_delta, policy)
    if severity is DriftSeverity.OK:
        reason = "within tolerance"
    else:
        reason = (
            f"position drift {severity.value}: |Δ|={abs_delta:.6g} "
            f"rel={rel_delta:.6g} expected={expected.position!r} "
            f"actual={actual.position!r}"
        )
    return PositionDelta(
        symbol=expected.symbol,
        expected_position=expected.position,
        actual_position=actual.position,
        absolute_delta=abs_delta,
        relative_delta=rel_delta,
        severity=severity,
        reason=reason,
    )


def expected_stake(
    *,
    start_cap: float,
    total_closed_profit: float,
    total_realized_profit: float,
    total_in_trades: float,
) -> float:
    """Reproduce freqtrade's ``current_stake`` projection.

    ``current_stake = start_cap + (total_closed_profit +
    total_realized_profit) - total_in_trades``. Pure function — caller
    supplies every term.
    """
    if start_cap < 0:
        raise ValueError("start_cap must be >= 0")
    if total_in_trades < 0:
        raise ValueError("total_in_trades must be >= 0")
    return start_cap + total_closed_profit + total_realized_profit - total_in_trades


def reconcile(
    *,
    now_ns: int,
    expected_wallets: Mapping[str, WalletBalance],
    actual_wallets: Mapping[str, WalletBalance],
    expected_positions: Mapping[str, PositionSnapshot] | None = None,
    actual_positions: Mapping[str, PositionSnapshot] | None = None,
    policy: ReconciliationPolicy | None = None,
    meta: Mapping[str, str] | None = None,
) -> ReconciliationReport:
    """Reconcile the full wallet + position snapshot.

    Returns a :class:`ReconciliationReport` summarising per-currency
    and per-symbol deltas plus an aggregate outcome bucket.
    """
    if now_ns < 0:
        raise ValueError("now_ns must be >= 0")
    pol = policy if policy is not None else ReconciliationPolicy()
    exp_pos = expected_positions if expected_positions is not None else {}
    act_pos = actual_positions if actual_positions is not None else {}

    wallet_deltas: list[WalletDelta] = []
    position_deltas: list[PositionDelta] = []

    # Wallet pass — iterate the union of currencies in sorted order so
    # the report is deterministic.
    all_currencies = sorted(set(expected_wallets) | set(actual_wallets))
    for currency in all_currencies:
        e = expected_wallets.get(currency)
        a = actual_wallets.get(currency)
        if e is None and a is not None:
            wallet_deltas.append(
                WalletDelta(
                    currency=currency,
                    expected_total=0.0,
                    actual_total=a.total,
                    absolute_delta=abs(a.total),
                    relative_delta=float("inf") if a.total else 0.0,
                    severity=DriftSeverity.MISSING,
                    reason="missing from expected snapshot",
                )
            )
        elif a is None and e is not None:
            wallet_deltas.append(
                WalletDelta(
                    currency=currency,
                    expected_total=e.total,
                    actual_total=0.0,
                    absolute_delta=abs(e.total),
                    relative_delta=float("inf") if e.total else 0.0,
                    severity=DriftSeverity.MISSING,
                    reason="missing from actual snapshot",
                )
            )
        elif e is not None and a is not None:
            wallet_deltas.append(reconcile_wallet(expected=e, actual=a, policy=pol))

    # Position pass — same sorted-union iteration.
    all_symbols = sorted(set(exp_pos) | set(act_pos))
    for symbol in all_symbols:
        e_pos = exp_pos.get(symbol)
        a_pos = act_pos.get(symbol)
        if e_pos is None and a_pos is not None:
            position_deltas.append(
                PositionDelta(
                    symbol=symbol,
                    expected_position=0.0,
                    actual_position=a_pos.position,
                    absolute_delta=abs(a_pos.position),
                    relative_delta=float("inf") if a_pos.position else 0.0,
                    severity=DriftSeverity.MISSING,
                    reason="missing from expected snapshot",
                )
            )
        elif a_pos is None and e_pos is not None:
            position_deltas.append(
                PositionDelta(
                    symbol=symbol,
                    expected_position=e_pos.position,
                    actual_position=0.0,
                    absolute_delta=abs(e_pos.position),
                    relative_delta=float("inf") if e_pos.position else 0.0,
                    severity=DriftSeverity.MISSING,
                    reason="missing from actual snapshot",
                )
            )
        elif e_pos is not None and a_pos is not None:
            position_deltas.append(reconcile_position(expected=e_pos, actual=a_pos, policy=pol))

    # Aggregate verdict — worst-bucket wins; MISSING and HAZARD escalate.
    has_missing = any(
        d.severity is DriftSeverity.MISSING for d in (*wallet_deltas, *position_deltas)
    )
    has_hazard = any(d.severity is DriftSeverity.HAZARD for d in (*wallet_deltas, *position_deltas))
    has_warning = any(
        d.severity is DriftSeverity.WARNING for d in (*wallet_deltas, *position_deltas)
    )
    if has_missing:
        outcome = ReconciliationOutcome.MISSING_CURRENCY
        reason = "missing currency or symbol in one of the snapshots"
    elif has_hazard:
        outcome = ReconciliationOutcome.DRIFT_HAZARD
        reason = "at least one balance / position drifted into HAZARD bucket"
    elif has_warning:
        outcome = ReconciliationOutcome.DRIFT_WARNING
        reason = "at least one balance / position drifted into WARNING bucket"
    else:
        outcome = ReconciliationOutcome.CONSISTENT
        reason = "all balances and positions within tolerance"

    digest = _snapshot_digest(expected_wallets, actual_wallets, exp_pos, act_pos)
    merged_meta: Mapping[str, str]
    if meta:
        merged_meta = dict(sorted(meta.items()))
    else:
        merged_meta = {}

    return ReconciliationReport(
        outcome=outcome,
        now_ns=now_ns,
        wallet_deltas=tuple(wallet_deltas),
        position_deltas=tuple(position_deltas),
        snapshot_digest=digest,
        policy_digest=pol.policy_digest(),
        reason=reason,
        meta=merged_meta,
    )


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_HAZARD_RELATIVE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DEFAULT_WARNING_RELATIVE",
    "DriftSeverity",
    "INVARIANT_TOTAL_TOLERANCE",
    "NEW_PIP_DEPENDENCIES",
    "PositionDelta",
    "PositionSide",
    "PositionSnapshot",
    "ReconciliationOutcome",
    "ReconciliationPolicy",
    "ReconciliationReport",
    "WalletBalance",
    "WalletDelta",
    "expected_stake",
    "reconcile",
    "reconcile_position",
    "reconcile_wallet",
]
