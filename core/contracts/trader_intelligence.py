"""Wave-04 PR-1 — Trader-Intelligence first-class structured-knowledge layer.

DIX VISION's pre-Wave-04 architecture models *market data* (Phase E1/E2,
Binance feed, OHLCV ticks) and *its own* internal decisions (Phase 6
``DecisionTrace``, ``META_AUDIT``, ``CALIBRATION_REPORT``). It does
**not** model external *traders* as structured records.

This module fixes that gap. It introduces three frozen / slotted /
hashable dataclasses — :class:`PhilosophyProfile`,
:class:`TraderModel`, and :class:`TraderObservation` — that capture
what an external trader *believes*, *does*, and *consistently
under/over-performs at*, in a form the offline learning + evolution
loop can decompose, recombine, and submit through the existing
operator-approval edge (Wave-03 PR-5).

Design constraints (mirror ``core.contracts.events`` /
``core.contracts.learning``):

* Frozen, slotted, hashable, structural equality (TEST-01 replay parity,
  INV-15 deterministic primitives).
* No callables, no IO, no clocks. Producers stamp ``ts_ns`` from the
  TimeAuthority (T0-04).
* No PII / no secrets. ``trader_id`` is a stable opaque handle from the
  source feed, never a real-name or wallet address.
* Domain records ride on the existing ``SystemEvent`` envelope with the
  new ``TRADER_OBSERVED`` sub-kind. We deliberately do **not** mint a
  fifth canonical event class — keeping the bus 4-typed (Signal /
  Execution / System / Hazard) preserves the Triad-Lock surface area
  HARDEN-02 / HARDEN-03 enforce.

Authority symmetry (HARDEN-06 / INV-71 extension):

* :class:`TraderObservation` may **only** be constructed by
  ``intelligence_engine.trader_modeling.*`` (the dedicated subsystem
  that wraps adapters → SCVS → typed event) or this module (which is
  the contract definition). Lint rule **B29** in ``tools.authority_lint``
  pins the inverse — symmetric to B27 (LearningUpdate) and B28
  (PatchProposal). Outside callers must observe
  :class:`TraderObservation` rows on the typed bus rather than
  synthesising them.

INV-08: only typed records cross domain boundaries.
INV-11: no direct cross-engine method calls; only these records flow.
INV-15: all fields are deterministic primitives.
INV-71 (extended): producer-set symmetry across every actionable
record class — symmetric to LearningUpdate (B27) and PatchProposal
(B28).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Philosophy primitives (StrEnum — discriminators, not free-form text)
# ---------------------------------------------------------------------------


class RiskAttitude(StrEnum):
    """How the trader sizes risk.

    The five canonical buckets are intentionally coarse — the goal is
    *clustering*, not faithful per-trader reproduction. Finer-grained
    attribution lives in :attr:`PhilosophyProfile.belief_system`.
    """

    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    LEVERAGED = "LEVERAGED"
    UNKNOWN = "UNKNOWN"


class TimeHorizon(StrEnum):
    """Holding-period bucket the trader operates in."""

    SCALP = "SCALP"  # seconds → minutes
    INTRADAY = "INTRADAY"  # minutes → hours
    SWING = "SWING"  # days → weeks
    POSITION = "POSITION"  # weeks → months
    MACRO = "MACRO"  # months → years
    UNKNOWN = "UNKNOWN"


class ConvictionStyle(StrEnum):
    """How the trader forms decisions relative to incoming information."""

    REACTIVE = "REACTIVE"  # reacts to confirmed moves
    PREDICTIVE = "PREDICTIVE"  # forecasts moves before confirmation
    CONTRARIAN = "CONTRARIAN"  # explicitly fades consensus
    SYSTEMATIC = "SYSTEMATIC"  # rule-following, no discretion
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# PhilosophyProfile — first-class structured belief layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhilosophyProfile:
    """How the trader *thinks* about markets, encoded as data.

    Replaces "have an LLM imitate the trader" with "extract structured
    fields the composition engine can reason over". Compatibility
    constraints (Wave-04 PR-4) check these fields directly — e.g. a
    mean-reversion belief_system entry is incompatible with a
    breakout-style entry-logic component. Without that gate, the
    composition engine produces nonsense.

    Attributes:
        trader_id: Opaque stable handle from the source feed. Same
            value as the wrapping :class:`TraderModel`.
        belief_system: Mapping of belief tag → strength in ``[0, 1]``.
            Canonical tags (``trend_following``, ``mean_reversion``,
            ``momentum``, ``volatility_premium``, …) live in the
            registry; freeform tags are allowed but ignored by the
            compatibility table.
        risk_attitude: Coarse bucket — see :class:`RiskAttitude`.
        time_horizon: Coarse bucket — see :class:`TimeHorizon`.
        conviction_style: How decisions form — see
            :class:`ConvictionStyle`.
        market_view: Mapping of market-view key → string value
            (e.g. ``{"markets_are_random": "false",
            "regime_persists": "true"}``). Stored as strings for
            replay determinism (no float drift across runs).
        decision_biases: Mapping of bias tag → magnitude in ``[0, 1]``
            (``loss_aversion``, ``overtrading``, ``fomo``,
            ``anchoring``, …). Bias detection is offline; this is a
            structured record of what was detected, not a runtime hook.
    """

    trader_id: str
    belief_system: Mapping[str, float] = field(default_factory=dict)
    risk_attitude: RiskAttitude = RiskAttitude.UNKNOWN
    time_horizon: TimeHorizon = TimeHorizon.UNKNOWN
    conviction_style: ConvictionStyle = ConvictionStyle.UNKNOWN
    market_view: Mapping[str, str] = field(default_factory=dict)
    decision_biases: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraderModel — performance + risk + regime profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraderModel:
    """Structured per-trader profile.

    Captures *what the trader does*, *how well they do it*, and *where
    they do it well*. Paired with :attr:`philosophy` it forms the
    structured input to the composition engine (Wave-04 PR-4).

    The strategy-signature layer (:attr:`strategy_signatures`) is the
    decomposition seam. A signature is a stable hash over a strategy
    decomposition (entry / exit / risk / timeframe / market-condition)
    that the strategy decomposition library (Wave-04 PR-3) will mint.
    Wave-04 PR-1 only declares the field — the registry of canonical
    signatures lands in PR-3.

    Attributes:
        trader_id: Opaque stable handle from the source feed.
        source_feed: Identifier of the SCVS source that produced the
            observation. Mirrors
            ``registry/data_source_registry.yaml`` row IDs (e.g.
            ``"SRC-TRADER-TRADINGVIEW-001"``).
        strategy_signatures: Tuple of decomposition hashes the trader
            consistently runs. Empty for newly-observed traders
            (signatures are minted by Wave-04 PR-3 once enough
            observations accumulate).
        performance_metrics: Mapping of metric tag → value
            (``"win_rate"``, ``"sharpe"``, ``"max_drawdown"``,
            ``"avg_holding_period_min"``, …).
        risk_profile: Mapping of risk-component tag → value
            (``"avg_position_size_pct"``, ``"avg_stop_distance_bps"``,
            ``"leverage_used"``, …).
        regime_performance: Mapping of regime tag → performance score
            (``"trending"``, ``"ranging"``, ``"high_vol"``,
            ``"low_vol"``, …). Lets the composition engine prefer
            traders whose strengths line up with the current regime.
        behavioral_bias: Mapping of bias tag → magnitude. Distinct
            from :attr:`PhilosophyProfile.decision_biases` — that
            describes the trader's *self-reported / inferred*
            psychology, this describes *measured* behaviour
            (e.g. measured ``"chases_breakouts"`` even though the
            trader self-reports as a mean-reversion philosophy).
        philosophy: Optional structured belief layer. ``None`` until
            the offline philosophy-extractor (Wave-04 PR-5) has run on
            the trader's observation history.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    trader_id: str
    source_feed: str
    strategy_signatures: tuple[str, ...] = ()
    performance_metrics: Mapping[str, float] = field(default_factory=dict)
    risk_profile: Mapping[str, float] = field(default_factory=dict)
    regime_performance: Mapping[str, float] = field(default_factory=dict)
    behavioral_bias: Mapping[str, float] = field(default_factory=dict)
    philosophy: PhilosophyProfile | None = None
    meta: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraderObservation — bus-transport record (SystemEvent payload)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraderObservation:
    """Domain record carrying a :class:`TraderModel` onto the typed bus.

    Materialised as a ``SystemEvent(sub_kind=TRADER_OBSERVED)`` so it
    inherits the four-canonical-event Triad-Lock surface area instead
    of minting a fifth. Producer authority is enforced by:

    1. **Static — lint rule B29** (``tools.authority_lint``). Only
       ``intelligence_engine.trader_modeling.*`` and this module may
       construct a :class:`TraderObservation`. Symmetric to B27
       (``LearningUpdate``) and B28 (``PatchProposal``).
    2. **Runtime — SCVS source-liveness FSM**. Adapters that emit
       these are registered as critical sources in
       ``registry/data_source_registry.yaml``; SCVS Phase-2 hazards
       fire if the feed goes stale (SCVS-06 → ``HAZ-13``).
    3. **Wire — provenance assertion**. The wrapping ``SystemEvent``
       carries ``produced_by_engine`` and is checked at receivers via
       :func:`core.contracts.event_provenance.assert_event_provenance`
       — same path HARDEN-03 enforces for every other typed event.

    Attributes:
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority,
            T0-04). Time the observation was *recorded*, not the time
            of the underlying trader action.
        trader_id: Mirrors :attr:`TraderModel.trader_id`. Carried
            top-level for fast dispatch without unpacking the model.
        observation_kind: Coarse discriminator —
            ``"PROFILE_UPDATE"`` (the trader's :class:`TraderModel`
            has been recomputed from accumulated history) or
            ``"SIGNAL_OBSERVED"`` (a single trade / signal was just
            observed and added to the trader's history). Future kinds
            land here without a schema migration.
        model: The current :class:`TraderModel` snapshot.
            ``observation_kind == "PROFILE_UPDATE"`` carries a fully
            recomputed model; ``"SIGNAL_OBSERVED"`` carries the
            *unchanged* model alongside the new signal in :attr:`meta`
            so consumers can distinguish "what we know about the
            trader" from "what just happened".
        meta: Free-form structural metadata (no PII, no secrets). For
            ``"SIGNAL_OBSERVED"`` rows, callers conventionally include
            keys like ``"symbol"``, ``"side"``, ``"entry_price"``,
            ``"observed_ts_ns"``.
    """

    ts_ns: int
    trader_id: str
    observation_kind: str
    model: TraderModel
    meta: Mapping[str, str] = field(default_factory=dict)


# Sentinel constants so callers don't sprinkle string literals.
TRADER_OBSERVATION_PROFILE_UPDATE: str = "PROFILE_UPDATE"
TRADER_OBSERVATION_SIGNAL_OBSERVED: str = "SIGNAL_OBSERVED"


__all__ = [
    "TRADER_OBSERVATION_PROFILE_UPDATE",
    "TRADER_OBSERVATION_SIGNAL_OBSERVED",
    "ConvictionStyle",
    "PhilosophyProfile",
    "RiskAttitude",
    "TimeHorizon",
    "TraderModel",
    "TraderObservation",
]
