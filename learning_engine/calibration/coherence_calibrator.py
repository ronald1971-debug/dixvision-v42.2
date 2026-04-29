"""Coherence calibrator — INV-53 reader side (Phase 6.T1c reader).

This module is the **offline consumer** for the four ledger rows the
hot path emits every tick:

* ``BELIEF_STATE_SNAPSHOT`` — read-only projection of regime + market view.
* ``PRESSURE_VECTOR_SNAPSHOT`` — 5-D constraint summary.
* ``META_AUDIT`` — J3 per-tick meta-controller audit (proposed_side,
  confidence components, sizing components, decision summary).
* ``REWARD_BREAKDOWN`` — per-realised-trade shaped reward + raw PnL +
  per-component contributions.

Per-window the calibrator computes:

1. **Belief calibration.** Mean ``regime_confidence`` across the
   BELIEF_STATE_SNAPSHOT events vs. the directional-match rate of
   the META_AUDIT events whose decision side was honoured by a
   subsequent realised fill (when ``fills`` is provided). The gap
   ``avg_confidence − directional_accuracy`` is the canonical
   "calibrated?" metric — ≈ 0 = well-calibrated, > 0 = overconfident,
   < 0 = underconfident.

2. **Pressure aggregates.** Mean of every pressure dimension across
   the window. Operators read these to detect lens drift (e.g.
   ``pressure_avg_uncertainty`` slowly trending up while realised PnL
   stays flat → entropy term is mis-weighted).

3. **Audit aggregates.** Mean decision confidence + INV-48 fallback
   rate. Combined with the directional match rate this surfaces
   degraded ticks — windows with high fallback rate **and** low
   directional accuracy are immediate Governance signal.

4. **Reward aggregates.** Sum of raw PnL + sum of shaped reward +
   per-component sums (J3 attribution at window scale). The
   calibrator deliberately preserves both the raw and the shaped
   number (INV-47).

The output is a single :class:`CalibrationReport` projected to a
``SystemEvent`` with ``sub_kind=CALIBRATION_REPORT`` (v3.4). The
event is read-only by Governance — it never gates execution; it
only surfaces drift between the runtime's projected lenses and the
realised outcome distribution.

Authority constraints:

* L2 — offline engine, may not import any runtime engine. Inputs are
  ``SystemEvent`` payloads (already structurally projected by the hot
  path) and optional ``ExecutionEvent`` fills. No imports of
  ``intelligence_engine``, ``execution_engine``, ``governance_engine``,
  ``system_engine``.
* L1 — may not import ``evolution_engine``.
* INV-15 — pure: same ``(ts_ns, events, fills)`` ⇒ same report.
* INV-47 — raw PnL is preserved on the report alongside the shaped
  total.
* INV-53 — this is the *reader* side; the hot path is the writer.
* INV-56 (Triad Lock) — this module is governance-blind; it reads
  the ledger and emits a single SystemEvent. It does not construct
  ``ExecutionEvent`` (B21) or ``SignalEvent`` (B22).

Not in scope for v3.4 Wave 2:

* Realised vs predicted *per-pressure-dimension* gap (e.g. realised
  latency_z vs ``pressure.latency``). Requires the runtime to
  ledger realised constraint scalars; that is a follow-on PR.
* INV-55 sim-realism penalty. Lands with Phase 10.1 Strategy Arena.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SystemEvent,
    SystemEventKind,
)

CALIBRATION_REPORT_VERSION = "v3.4-T1c-reader"
CALIBRATOR_SOURCE = "learning_engine.calibration.coherence_calibrator"

# Sentinel: when no audited tick produced a paired fill, the
# directional-accuracy denominator is zero. We expose
# ``audit_match_known=False`` and zero out the rate so downstream
# averages don't get poisoned by NaNs.
_NO_MATCH_SENTINEL = 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Aggregate calibration record for a closed ledger window.

    All ``avg_*`` / ``rate`` fields live in ``[0, 1]`` (or are NaN-free
    sentinel zeros when their denominator is zero — see ``_known``
    flags).

    Fields:
        ts_ns: Report timestamp (caller-supplied; typically the
            ``window_end_ns``).
        window_start_ns: Inclusive window start. ``min(event.ts_ns)``
            when not specified by caller.
        window_end_ns: Inclusive window end. ``max(event.ts_ns)``
            when not specified.

        belief_count: Number of BELIEF_STATE_SNAPSHOT events seen.
        belief_avg_regime_confidence: Mean of ``regime_confidence``
            across belief snapshots (0 when ``belief_count == 0``).

        pressure_count: Number of PRESSURE_VECTOR_SNAPSHOT events.
        pressure_avg_perf / risk / drift / latency / uncertainty:
            Per-dimension means across the window.
        pressure_avg_safety_modifier: Mean of the continuous H2
            damping factor.

        audit_count: Number of META_AUDIT events.
        audit_match_known: True iff at least one META_AUDIT was
            paired with a realised fill (i.e. ``fills`` was non-empty
            and at least one fill's ``ts_ns`` was on or after a
            META_AUDIT's).
        audit_directional_match_rate: Fraction of paired META_AUDIT
            events whose ``decision_side`` matched the realised PnL
            sign of the paired fill. 0.0 when
            ``audit_match_known is False``.
        audit_avg_decision_confidence: Mean of META_AUDIT
            ``decision_confidence``.
        audit_fallback_rate: Fraction of META_AUDIT events whose
            ``decision_fallback`` was true (INV-48 fallback).

        belief_calibration_gap: ``belief_avg_regime_confidence -
            audit_directional_match_rate`` when both are known;
            ``0.0`` when match isn't known. Positive = overconfident.

        reward_count: Number of REWARD_BREAKDOWN events.
        reward_total_raw_pnl: Sum of raw PnL across the window
            (INV-47 preserved).
        reward_total_shaped: Sum of shaped reward across the window.
        reward_components: Sorted tuple of ``(component_name,
            total_contribution)`` aggregated across the window.

        version: ``CALIBRATION_REPORT_VERSION``.
    """

    ts_ns: int
    window_start_ns: int
    window_end_ns: int

    belief_count: int = 0
    belief_avg_regime_confidence: float = 0.0

    pressure_count: int = 0
    pressure_avg_perf: float = 0.0
    pressure_avg_risk: float = 0.0
    pressure_avg_drift: float = 0.0
    pressure_avg_latency: float = 0.0
    pressure_avg_uncertainty: float = 0.0
    pressure_avg_safety_modifier: float = 0.0

    audit_count: int = 0
    audit_match_known: bool = False
    audit_directional_match_rate: float = 0.0
    audit_avg_decision_confidence: float = 0.0
    audit_fallback_rate: float = 0.0

    belief_calibration_gap: float = 0.0

    reward_count: int = 0
    reward_total_raw_pnl: float = 0.0
    reward_total_shaped: float = 0.0
    reward_components: tuple[tuple[str, float], ...] = ()

    version: str = CALIBRATION_REPORT_VERSION

    def to_event(
        self,
        source: str = CALIBRATOR_SOURCE,
    ) -> SystemEvent:
        """Project the report into a ledgerable :class:`SystemEvent`.

        Read-only by Governance (INV-53 / INV-56). The payload is a
        flat string-valued mapping so it round-trips identically
        across the proto / Python / JSON projections.
        """
        payload: dict[str, str] = {
            "version": self.version,
            "window_start_ns": str(self.window_start_ns),
            "window_end_ns": str(self.window_end_ns),
            # Belief
            "belief_count": str(self.belief_count),
            "belief_avg_regime_confidence": (
                f"{self.belief_avg_regime_confidence:.6f}"
            ),
            # Pressure
            "pressure_count": str(self.pressure_count),
            "pressure_avg_perf": f"{self.pressure_avg_perf:.6f}",
            "pressure_avg_risk": f"{self.pressure_avg_risk:.6f}",
            "pressure_avg_drift": f"{self.pressure_avg_drift:.6f}",
            "pressure_avg_latency": f"{self.pressure_avg_latency:.6f}",
            "pressure_avg_uncertainty": (
                f"{self.pressure_avg_uncertainty:.6f}"
            ),
            "pressure_avg_safety_modifier": (
                f"{self.pressure_avg_safety_modifier:.6f}"
            ),
            # Audit
            "audit_count": str(self.audit_count),
            "audit_match_known": "true" if self.audit_match_known else "false",
            "audit_directional_match_rate": (
                f"{self.audit_directional_match_rate:.6f}"
            ),
            "audit_avg_decision_confidence": (
                f"{self.audit_avg_decision_confidence:.6f}"
            ),
            "audit_fallback_rate": f"{self.audit_fallback_rate:.6f}",
            # Belief calibration
            "belief_calibration_gap": f"{self.belief_calibration_gap:.6f}",
            # Reward
            "reward_count": str(self.reward_count),
            "reward_total_raw_pnl": f"{self.reward_total_raw_pnl:.6f}",
            "reward_total_shaped": f"{self.reward_total_shaped:.6f}",
            "reward_component_count": str(len(self.reward_components)),
        }
        for name, total in self.reward_components:
            payload[f"reward_component__{name}"] = f"{total:.6f}"
        return SystemEvent(
            ts_ns=self.ts_ns,
            sub_kind=SystemEventKind.CALIBRATION_REPORT,
            source=source,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Accumulator:
    belief_count: int = 0
    belief_sum_confidence: float = 0.0

    pressure_count: int = 0
    pressure_sum_perf: float = 0.0
    pressure_sum_risk: float = 0.0
    pressure_sum_drift: float = 0.0
    pressure_sum_latency: float = 0.0
    pressure_sum_uncertainty: float = 0.0
    pressure_sum_safety_modifier: float = 0.0

    audit_count: int = 0
    audit_sum_decision_confidence: float = 0.0
    audit_fallback_count: int = 0
    audits: list[tuple[int, Side]] = field(default_factory=list)

    reward_count: int = 0
    reward_sum_raw_pnl: float = 0.0
    reward_sum_shaped: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)

    min_ts_ns: int | None = None
    max_ts_ns: int | None = None


def _track_ts(acc: _Accumulator, ts_ns: int) -> None:
    if acc.min_ts_ns is None or ts_ns < acc.min_ts_ns:
        acc.min_ts_ns = ts_ns
    if acc.max_ts_ns is None or ts_ns > acc.max_ts_ns:
        acc.max_ts_ns = ts_ns


def _f(payload: Mapping[str, str], key: str, default: float = 0.0) -> float:
    raw = payload.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _accumulate_belief(acc: _Accumulator, ev: SystemEvent) -> None:
    acc.belief_count += 1
    acc.belief_sum_confidence += _f(ev.payload, "regime_confidence")


def _accumulate_pressure(acc: _Accumulator, ev: SystemEvent) -> None:
    acc.pressure_count += 1
    p = ev.payload
    acc.pressure_sum_perf += _f(p, "perf")
    acc.pressure_sum_risk += _f(p, "risk")
    acc.pressure_sum_drift += _f(p, "drift")
    acc.pressure_sum_latency += _f(p, "latency")
    acc.pressure_sum_uncertainty += _f(p, "uncertainty")
    acc.pressure_sum_safety_modifier += _f(
        p, "safety_modifier", default=1.0
    )


def _audit_side(payload: Mapping[str, str]) -> Side:
    raw = payload.get("decision_side", Side.HOLD.value)
    try:
        return Side(raw)
    except ValueError:
        return Side.HOLD


def _accumulate_audit(acc: _Accumulator, ev: SystemEvent) -> None:
    acc.audit_count += 1
    acc.audit_sum_decision_confidence += _f(
        ev.payload, "decision_confidence"
    )
    if ev.payload.get("decision_fallback") == "true":
        acc.audit_fallback_count += 1
    acc.audits.append((ev.ts_ns, _audit_side(ev.payload)))


def _accumulate_reward(acc: _Accumulator, ev: SystemEvent) -> None:
    acc.reward_count += 1
    acc.reward_sum_raw_pnl += _f(ev.payload, "raw_pnl")
    acc.reward_sum_shaped += _f(ev.payload, "shaped_reward")
    # Per-component totals: keys of the shape ``c.<name>`` →
    # contribution (matches RewardBreakdown.to_event() payload key
    # convention in learning_engine.lanes.reward_shaping).
    prefix = "c."
    for key, raw in ev.payload.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        try:
            v = float(raw)
        except ValueError:
            continue
        acc.reward_components[name] = (
            acc.reward_components.get(name, 0.0) + v
        )


# ---------------------------------------------------------------------------
# Realised-direction matching
# ---------------------------------------------------------------------------


def _is_realised(fill: ExecutionEvent) -> bool:
    return fill.status in (
        ExecutionStatus.FILLED,
        ExecutionStatus.PARTIALLY_FILLED,
    )


def _realised_pnl(fill: ExecutionEvent) -> float:
    """Best-effort realised-PnL extraction from an ``ExecutionEvent``.

    The payload key ``realised_pnl`` is the canonical channel; when
    not present the calibrator treats the fill as ``0.0`` (no signed
    information about direction-correctness). Callers that want
    proper directional matching populate ``realised_pnl`` on the fill
    meta.
    """
    raw = fill.meta.get("realised_pnl") if fill.meta else None
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _direction_correct(side: Side, realised_pnl: float) -> bool | None:
    """``True`` iff the realised PnL has the sign expected for ``side``.

    Returns ``None`` when there is no signed signal to match against
    (HOLD / PnL == 0). The caller must skip these — they do not
    contribute to either numerator or denominator.
    """
    if side is Side.HOLD:
        return None
    if realised_pnl > 0.0:
        return side is Side.BUY
    if realised_pnl < 0.0:
        return side is Side.SELL
    return None


def _compute_match_rate(
    audits: Sequence[tuple[int, Side]],
    fills: Sequence[ExecutionEvent],
) -> tuple[bool, float]:
    """Pair each audited tick with the next realised fill on or after it.

    Returns ``(match_known, match_rate)``. ``match_known`` is True
    iff at least one audit produced a comparable realised PnL; the
    rate is then the ratio of correctly-directed audits to compared
    audits. ``match_rate`` is ``_NO_MATCH_SENTINEL`` (0.0) when
    ``match_known`` is False.
    """
    if not audits or not fills:
        return False, _NO_MATCH_SENTINEL
    realised: list[tuple[int, float]] = sorted(
        ((f.ts_ns, _realised_pnl(f)) for f in fills if _is_realised(f)),
        key=lambda pair: pair[0],
    )
    if not realised:
        return False, _NO_MATCH_SENTINEL

    sorted_audits = sorted(audits, key=lambda pair: pair[0])

    compared = 0
    correct = 0
    fill_idx = 0
    for audit_ts, side in sorted_audits:
        # Advance the fill cursor to the first fill at or after the
        # audit timestamp.
        while fill_idx < len(realised) and realised[fill_idx][0] < audit_ts:
            fill_idx += 1
        if fill_idx >= len(realised):
            break
        _, pnl = realised[fill_idx]
        verdict = _direction_correct(side, pnl)
        if verdict is None:
            continue
        compared += 1
        if verdict:
            correct += 1

    if compared == 0:
        return False, _NO_MATCH_SENTINEL
    return True, correct / compared


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate_coherence_window(
    *,
    ts_ns: int,
    events: Sequence[SystemEvent],
    fills: Sequence[ExecutionEvent] = (),
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> CalibrationReport:
    """Compute a :class:`CalibrationReport` over a ledger window.

    Pure function — same ``(ts_ns, events, fills, window_start_ns,
    window_end_ns)`` produces the same report (INV-15).

    Args:
        ts_ns: Stamp the report with this timestamp.
        events: A sequence of ``SystemEvent``s (in any order). Only
            BELIEF_STATE_SNAPSHOT, PRESSURE_VECTOR_SNAPSHOT,
            META_AUDIT, and REWARD_BREAKDOWN events are consumed;
            everything else is ignored. Out-of-window events (per
            ``window_start_ns`` / ``window_end_ns`` if provided) are
            also ignored.
        fills: Optional realised ``ExecutionEvent`` fills. When
            provided, the calibrator matches each META_AUDIT to the
            first FILL/PARTIALLY_FILLED on or after it and computes
            an ``audit_directional_match_rate``. When omitted the
            rate reverts to a NaN-free sentinel (``0.0``) and
            ``audit_match_known`` is False.
        window_start_ns: Inclusive window start. When ``None`` the
            calibrator uses ``min(event.ts_ns)`` from the consumed
            events (or ``ts_ns`` itself when no events match).
        window_end_ns: Inclusive window end. ``None`` ⇒
            ``max(event.ts_ns)``.

    Returns:
        A frozen :class:`CalibrationReport`.
    """
    acc = _Accumulator()

    def in_window(t: int) -> bool:
        if window_start_ns is not None and t < window_start_ns:
            return False
        if window_end_ns is not None and t > window_end_ns:
            return False
        return True

    for ev in events:
        if not in_window(ev.ts_ns):
            continue
        sub = ev.sub_kind
        if sub is SystemEventKind.BELIEF_STATE_SNAPSHOT:
            _accumulate_belief(acc, ev)
        elif sub is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT:
            _accumulate_pressure(acc, ev)
        elif sub is SystemEventKind.META_AUDIT:
            _accumulate_audit(acc, ev)
        elif sub is SystemEventKind.REWARD_BREAKDOWN:
            _accumulate_reward(acc, ev)
        else:
            continue
        _track_ts(acc, ev.ts_ns)

    # Fills (used only for directional matching). Tracking ts here
    # ensures the resolved window covers them too when the caller
    # didn't pin window bounds explicitly.
    bounded_fills: list[ExecutionEvent] = []
    for f in fills:
        if not in_window(f.ts_ns):
            continue
        bounded_fills.append(f)
        _track_ts(acc, f.ts_ns)

    resolved_start = (
        window_start_ns
        if window_start_ns is not None
        else (acc.min_ts_ns if acc.min_ts_ns is not None else ts_ns)
    )
    resolved_end = (
        window_end_ns
        if window_end_ns is not None
        else (acc.max_ts_ns if acc.max_ts_ns is not None else ts_ns)
    )

    belief_avg = (
        acc.belief_sum_confidence / acc.belief_count
        if acc.belief_count
        else 0.0
    )
    pcount = max(acc.pressure_count, 1)
    if acc.pressure_count:
        pressure_perf = acc.pressure_sum_perf / pcount
        pressure_risk = acc.pressure_sum_risk / pcount
        pressure_drift = acc.pressure_sum_drift / pcount
        pressure_latency = acc.pressure_sum_latency / pcount
        pressure_uncertainty = acc.pressure_sum_uncertainty / pcount
        pressure_safety = acc.pressure_sum_safety_modifier / pcount
    else:
        pressure_perf = 0.0
        pressure_risk = 0.0
        pressure_drift = 0.0
        pressure_latency = 0.0
        pressure_uncertainty = 0.0
        pressure_safety = 0.0

    if acc.audit_count:
        audit_avg_conf = (
            acc.audit_sum_decision_confidence / acc.audit_count
        )
        audit_fallback = acc.audit_fallback_count / acc.audit_count
    else:
        audit_avg_conf = 0.0
        audit_fallback = 0.0

    match_known, match_rate = _compute_match_rate(acc.audits, bounded_fills)

    if match_known and acc.belief_count:
        calibration_gap = belief_avg - match_rate
    else:
        calibration_gap = 0.0

    reward_components = tuple(
        sorted(acc.reward_components.items(), key=lambda pair: pair[0])
    )

    return CalibrationReport(
        ts_ns=ts_ns,
        window_start_ns=resolved_start,
        window_end_ns=resolved_end,
        belief_count=acc.belief_count,
        belief_avg_regime_confidence=belief_avg,
        pressure_count=acc.pressure_count,
        pressure_avg_perf=pressure_perf,
        pressure_avg_risk=pressure_risk,
        pressure_avg_drift=pressure_drift,
        pressure_avg_latency=pressure_latency,
        pressure_avg_uncertainty=pressure_uncertainty,
        pressure_avg_safety_modifier=pressure_safety,
        audit_count=acc.audit_count,
        audit_match_known=match_known,
        audit_directional_match_rate=match_rate,
        audit_avg_decision_confidence=audit_avg_conf,
        audit_fallback_rate=audit_fallback,
        belief_calibration_gap=calibration_gap,
        reward_count=acc.reward_count,
        reward_total_raw_pnl=acc.reward_sum_raw_pnl,
        reward_total_shaped=acc.reward_sum_shaped,
        reward_components=reward_components,
    )


__all__ = [
    "CALIBRATION_REPORT_VERSION",
    "CALIBRATOR_SOURCE",
    "CalibrationReport",
    "calibrate_coherence_window",
]
