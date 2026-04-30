"""Wave-04 PR-4 — composition engine + compatibility constraint table.

The composition engine is a **pure function** that takes the five
components catalogued in :mod:`intelligence_engine.strategy_library`
plus an optional :class:`~core.contracts.trader_intelligence.PhilosophyProfile`,
and either returns a :class:`ComposedStrategy` proposal or raises
:class:`IncompatibleCompositionError` with structured reason codes.

Why a constraint table — not an LLM, not an oracle:

The trader-intelligence brief flagged the failure mode that motivates
this layer: "without compatibility constraints, mean-reversion
philosophy ⊗ breakout entry composition produces nonsense". The
composition engine therefore enforces a small, hand-curated table of
**incompatibility pairs**. Anything not in the table is allowed; the
table is the explicit list of *what we know breaks*. This is the
opposite of an "LLM decides if this strategy makes sense" layer —
deterministic, auditable, replay-safe (INV-15 / TEST-01).

Authority symmetry: a :class:`ComposedStrategy` is **a proposal**, not
an approved strategy. It carries no execution authority. To enter the
plugin registry it has to be approved through the operator-approval
edge (Wave-03 PR-5) — same gate, no new authority surface. Wave-04.5
wires the approval-edge integration; this PR ships the pure library
(value objects + compose function + constraint table + tests).

This module is library code — anyone can construct a
:class:`ComposedStrategy` (it's a proposal, not an actionable record).
The approval gate that turns a proposal into a live strategy is a
separate authority surface and is deliberately **not** in this PR.

Refs:

* ``intelligence_engine.strategy_library`` — Wave-04 PR-3 components.
* ``core.contracts.trader_intelligence.PhilosophyProfile`` — Wave-04
  PR-1 belief layer.
* ``intelligence_engine.cognitive.approval_edge`` — the operator-
  approval edge (Wave-03 PR-5) that a future PR will route composed
  strategies through.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from core.contracts.trader_intelligence import (
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
)
from intelligence_engine.strategy_library.components import (
    EntryLogic,
    EntryStyle,
    ExitLogic,
    ExitStyle,
    MarketCondition,
    MarketRegime,
    RiskModel,
    SizingStyle,
    StopStyle,
    Timeframe,
)
from intelligence_engine.strategy_library.decomposition import (
    StrategyDecomposition,
    signature_for,
)


class IncompatibilityReason(StrEnum):
    """Stable codes for *why* a composition was rejected.

    Used by tests, audit ledger projections, and the (future) operator
    UI. Adding a new reason is forward-compatible; renaming or removing
    a reason breaks downstream consumers — append-only by convention.
    """

    PHILOSOPHY_VS_ENTRY_STYLE = "PHILOSOPHY_VS_ENTRY_STYLE"
    PHILOSOPHY_VS_EXIT_STYLE = "PHILOSOPHY_VS_EXIT_STYLE"
    PHILOSOPHY_VS_RISK_ATTITUDE = "PHILOSOPHY_VS_RISK_ATTITUDE"
    HORIZON_VS_TIMEFRAME = "HORIZON_VS_TIMEFRAME"
    REGIME_VS_ENTRY_STYLE = "REGIME_VS_ENTRY_STYLE"
    UNKNOWN_DISCRIMINATOR = "UNKNOWN_DISCRIMINATOR"


# ---------------------------------------------------------------------------
# Compatibility constraint table — append-only, pure data
# ---------------------------------------------------------------------------
#
# Each table is the explicit, hand-curated list of incompatibility
# pairs. The default is *allowed*. Anything not in the table here
# can be composed; the table only catalogues what we know breaks.
#
# Edit rule: do not delete entries — they are part of the audit
# contract. To relax a constraint, either drop the matching row
# *and* migrate the audit ledger, or add a more specific override
# in a follow-up PR. To tighten, append a new pair.

BELIEF_THRESHOLD: float = 0.5
"""Minimum belief strength that triggers a philosophy-vs-* constraint.

Beliefs below this threshold are treated as noise — a trader who is
20% mean-reversion and 80% trend-following should not trip the
mean-reversion constraints when they take a breakout entry."""

_BELIEF_VS_ENTRY_BLOCKLIST: Mapping[str, frozenset[EntryStyle]] = {
    # Mean-reversion philosophy ⊗ breakout entry = nonsense
    # (the canonical example from the trader-intelligence brief).
    "mean_reversion": frozenset(
        {EntryStyle.BREAKOUT, EntryStyle.MOMENTUM}
    ),
    # Trend-following philosophy ⊗ mean-reversion entry: the entry
    # logic actively fights the strategy's stated belief.
    "trend_following": frozenset({EntryStyle.MEAN_REVERSION}),
    # Volatility-premium harvesters explicitly fade big moves; a
    # momentum or breakout entry runs the opposite direction.
    "volatility_premium": frozenset(
        {EntryStyle.MOMENTUM, EntryStyle.BREAKOUT}
    ),
}

_BELIEF_VS_EXIT_BLOCKLIST: Mapping[str, frozenset[ExitStyle]] = {
    # A mean-reversion strategy with a trailing-stop exit ratchets
    # against its own thesis (it expects price to *return* to mean,
    # not run further). Documented explicitly so the composition
    # engine catches the mismatch even if the entry-side check misses.
    "mean_reversion": frozenset({ExitStyle.TRAILING_STOP}),
}

_RISK_ATTITUDE_VS_STOP_BLOCKLIST: Mapping[
    RiskAttitude, frozenset[StopStyle]
] = {
    # Conservative trader with no hard stop is the textbook
    # blow-up shape; a wide stop is functionally equivalent for
    # a small account.
    RiskAttitude.CONSERVATIVE: frozenset({StopStyle.NONE, StopStyle.WIDE}),
    # Leveraged trader with a wide stop magnifies the leverage on
    # the very moves that will close the position.
    RiskAttitude.LEVERAGED: frozenset({StopStyle.WIDE, StopStyle.NONE}),
}

_HORIZON_VS_BAR_INTERVAL_BLOCKLIST: Mapping[TimeHorizon, frozenset[str]] = {
    TimeHorizon.SCALP: frozenset({"1h", "1d"}),
    TimeHorizon.INTRADAY: frozenset({"1d"}),
    TimeHorizon.MACRO: frozenset({"1s"}),
}

_REGIME_VS_ENTRY_BLOCKLIST: Mapping[
    MarketRegime, frozenset[EntryStyle]
] = {
    MarketRegime.TRENDING: frozenset({EntryStyle.MEAN_REVERSION}),
    MarketRegime.RANGING: frozenset(
        {EntryStyle.BREAKOUT, EntryStyle.MOMENTUM}
    ),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IncompatibilityFinding:
    """One reason a composition was rejected.

    Attributes:
        reason: Stable code (see :class:`IncompatibilityReason`).
        detail: Short human-readable explanation, deterministic
            (no clocks, no random ids). Suitable for the audit
            ledger and the operator UI.
    """

    reason: IncompatibilityReason
    detail: str


class IncompatibleCompositionError(ValueError):
    """Raised by :func:`compose` when constraints reject the proposal.

    The ``findings`` attribute carries every violation, not just the
    first one — operators need to see the full picture, not whack-a-mole
    one constraint at a time.
    """

    def __init__(self, findings: tuple[IncompatibilityFinding, ...]) -> None:
        if not findings:
            raise ValueError(
                "IncompatibleCompositionError requires at least one finding"
            )
        joined = "; ".join(f"{f.reason}: {f.detail}" for f in findings)
        super().__init__(f"composition rejected: {joined}")
        # Store on a frozen tuple so callers can't mutate the audit
        # trail.
        object.__setattr__(self, "_findings", tuple(findings))

    @property
    def findings(self) -> tuple[IncompatibilityFinding, ...]:
        return self._findings  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ComposedStrategy proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComposedStrategy:
    """Output of :func:`compose` — a proposal, not an approved strategy.

    Attributes:
        composition_id: Caller-supplied stable handle for this
            composition. The composition engine does not generate
            IDs — that's an operator concern.
        decomposition: The 5-tuple of components plus a stable
            ``decomposition_id``. Same shape as
            :class:`StrategyDecomposition` so a composed strategy can
            be re-decomposed and re-signed without round-tripping
            through a different type.
        philosophy: Optional philosophy profile that motivated the
            composition. ``None`` if the operator composed by hand.
        signature: 64-char hex SHA-256 over ``decomposition`` —
            same canonicalisation as
            :func:`intelligence_engine.strategy_library.signature_for`.
        notes: Free-form structural metadata (no PII, no secrets).
    """

    composition_id: str
    decomposition: StrategyDecomposition
    philosophy: PhilosophyProfile | None = None
    signature: str = ""
    notes: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure compose() function
# ---------------------------------------------------------------------------


def _check_unknown_discriminators(
    entry: EntryLogic,
    exit_: ExitLogic,
    risk: RiskModel,
) -> tuple[IncompatibilityFinding, ...]:
    findings: list[IncompatibilityFinding] = []
    if entry.style is EntryStyle.UNKNOWN:
        findings.append(
            IncompatibilityFinding(
                reason=IncompatibilityReason.UNKNOWN_DISCRIMINATOR,
                detail="entry.style is UNKNOWN; refusing to compose",
            )
        )
    if exit_.style is ExitStyle.UNKNOWN:
        findings.append(
            IncompatibilityFinding(
                reason=IncompatibilityReason.UNKNOWN_DISCRIMINATOR,
                detail="exit.style is UNKNOWN; refusing to compose",
            )
        )
    if risk.sizing is SizingStyle.UNKNOWN:
        findings.append(
            IncompatibilityFinding(
                reason=IncompatibilityReason.UNKNOWN_DISCRIMINATOR,
                detail="risk.sizing is UNKNOWN; refusing to compose",
            )
        )
    if risk.stop is StopStyle.UNKNOWN:
        findings.append(
            IncompatibilityFinding(
                reason=IncompatibilityReason.UNKNOWN_DISCRIMINATOR,
                detail="risk.stop is UNKNOWN; refusing to compose",
            )
        )
    return tuple(findings)


def _check_philosophy(
    philosophy: PhilosophyProfile,
    entry: EntryLogic,
    exit_: ExitLogic,
    risk: RiskModel,
) -> tuple[IncompatibilityFinding, ...]:
    findings: list[IncompatibilityFinding] = []
    for belief, strength in philosophy.belief_system.items():
        if strength < BELIEF_THRESHOLD:
            continue
        blocked_entries = _BELIEF_VS_ENTRY_BLOCKLIST.get(belief, frozenset())
        if entry.style in blocked_entries:
            findings.append(
                IncompatibilityFinding(
                    reason=IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE,
                    detail=(
                        f"belief={belief} (strength={strength}) is "
                        f"incompatible with entry.style={entry.style}"
                    ),
                )
            )
        blocked_exits = _BELIEF_VS_EXIT_BLOCKLIST.get(belief, frozenset())
        if exit_.style in blocked_exits:
            findings.append(
                IncompatibilityFinding(
                    reason=IncompatibilityReason.PHILOSOPHY_VS_EXIT_STYLE,
                    detail=(
                        f"belief={belief} (strength={strength}) is "
                        f"incompatible with exit.style={exit_.style}"
                    ),
                )
            )
    blocked_stops = _RISK_ATTITUDE_VS_STOP_BLOCKLIST.get(
        philosophy.risk_attitude, frozenset()
    )
    if risk.stop in blocked_stops:
        findings.append(
            IncompatibilityFinding(
                reason=IncompatibilityReason.PHILOSOPHY_VS_RISK_ATTITUDE,
                detail=(
                    f"risk_attitude={philosophy.risk_attitude} is "
                    f"incompatible with risk.stop={risk.stop}"
                ),
            )
        )
    return tuple(findings)


def _check_horizon(
    philosophy: PhilosophyProfile,
    timeframe: Timeframe,
) -> tuple[IncompatibilityFinding, ...]:
    blocked_intervals = _HORIZON_VS_BAR_INTERVAL_BLOCKLIST.get(
        philosophy.time_horizon, frozenset()
    )
    if timeframe.bar_interval in blocked_intervals:
        return (
            IncompatibilityFinding(
                reason=IncompatibilityReason.HORIZON_VS_TIMEFRAME,
                detail=(
                    f"trader time_horizon={philosophy.time_horizon} "
                    f"cannot run on bar_interval={timeframe.bar_interval}"
                ),
            ),
        )
    return ()


def _check_regime(
    market_condition: MarketCondition,
    entry: EntryLogic,
) -> tuple[IncompatibilityFinding, ...]:
    blocked = _REGIME_VS_ENTRY_BLOCKLIST.get(
        market_condition.regime, frozenset()
    )
    if entry.style in blocked:
        return (
            IncompatibilityFinding(
                reason=IncompatibilityReason.REGIME_VS_ENTRY_STYLE,
                detail=(
                    f"market_condition.regime={market_condition.regime} "
                    f"contradicts entry.style={entry.style}"
                ),
            ),
        )
    return ()


def compose(
    *,
    composition_id: str,
    decomposition_id: str,
    entry: EntryLogic,
    exit_: ExitLogic,
    risk: RiskModel,
    timeframe: Timeframe,
    market_condition: MarketCondition,
    philosophy: PhilosophyProfile | None = None,
    notes: Mapping[str, str] | None = None,
) -> ComposedStrategy:
    """Compose 5 components (+ optional philosophy) into a proposal.

    Returns a :class:`ComposedStrategy` if every constraint passes,
    otherwise raises :class:`IncompatibleCompositionError` carrying
    every finding (not just the first).

    The returned object is a **proposal** — it carries no execution
    authority. Wave-04.5 wires this through the operator-approval
    edge before a composed strategy can flow into the plugin
    registry. Until then, the composition engine produces audit
    records and nothing else.

    Args:
        composition_id: Operator-supplied stable handle.
        decomposition_id: Stable handle for the inner
            :class:`StrategyDecomposition`. Same value as the
            ``decomposition_id`` field of the resulting decomposition.
        entry, exit_, risk, timeframe, market_condition: The five
            reusable components.
        philosophy: Optional philosophy profile. If supplied, the
            philosophy-vs-entry/exit/risk and horizon-vs-timeframe
            constraints are evaluated.
        notes: Optional free-form structural metadata.
    """

    findings: list[IncompatibilityFinding] = []

    findings.extend(_check_unknown_discriminators(entry, exit_, risk))
    if philosophy is not None:
        findings.extend(_check_philosophy(philosophy, entry, exit_, risk))
        findings.extend(_check_horizon(philosophy, timeframe))
    findings.extend(_check_regime(market_condition, entry))

    if findings:
        raise IncompatibleCompositionError(tuple(findings))

    decomp = StrategyDecomposition(
        decomposition_id=decomposition_id,
        entry=entry,
        exit_=exit_,
        risk=risk,
        timeframe=timeframe,
        market_condition=market_condition,
    )
    sig = signature_for(decomp)
    return ComposedStrategy(
        composition_id=composition_id,
        decomposition=decomp,
        philosophy=philosophy,
        signature=sig,
        notes=dict(notes) if notes else {},
    )


# Conviction-style hook is reserved for a future PR — it interacts
# with the system's regime-router (intelligence_engine.strategy_runtime
# .regime_detector), not with the static decomposition. Documented
# here so a future contributor doesn't add a redundant check.
_RESERVED_CONVICTION_STYLES: frozenset[ConvictionStyle] = frozenset(
    {ConvictionStyle.UNKNOWN}
)


__all__ = [
    "BELIEF_THRESHOLD",
    "ComposedStrategy",
    "IncompatibilityFinding",
    "IncompatibilityReason",
    "IncompatibleCompositionError",
    "compose",
]
