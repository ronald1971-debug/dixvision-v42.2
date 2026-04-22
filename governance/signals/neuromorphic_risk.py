"""
governance/signals/neuromorphic_risk.py
DIX VISION v42.2 — Governance-side neuromorphic sensor (advisory only).

Phase 0 stub: event-emitting scaffolding. No SNN backend yet — Phase 4
replaces the rule-based risk-acceleration detector here with a
rolling-window SNN trained offline on historical risk regimes.

Axioms N1..N8 (immutable_core/neuromorphic_axioms.lean) apply. In
particular:
  - N1: no decision authority — emits events, never approves/rejects;
  - N7: advisory only — governance MAY consume RISK_SIGNAL_EVENT as a
    feature that TIGHTENS (never loosens) constraints. Final decision
    must remain a deterministic hard rule, replayable from ledger.

authority_lint rule C2 forbids this file from importing any of:
  governance.kernel, mind.fast_execute, execution.engine,
  security.operator, system.fast_risk_cache (mutators),
  core.registry (register).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from state.ledger.event_store import append_event

RISK_SIGNAL_TYPES = (
    "RISK_ACCELERATION",
    "REGIME_SHIFT",
    "STRATEGY_INSTABILITY",
    "CORRELATION_BREAKDOWN",
)


@dataclass
class RiskSignalEvent:
    """Wire format for RISK_SIGNAL_EVENT — advisory input into the
    Governance constraint compiler. MUST NOT be treated as a decision."""

    type: str                           # one of RISK_SIGNAL_TYPES
    severity: float                     # 0.0..1.0
    confidence: float                   # 0.0..1.0
    context: str
    timestamp_utc: str
    details: dict[str, Any] = field(default_factory=dict)


class NeuromorphicRisk:
    """Risk-acceleration sensor. Emits events; governance stays the decider."""

    name = "neuromorphic_risk"
    heartbeat_interval: float = 2.0     # seconds — N5 dead-man

    def __init__(self) -> None:
        self._last_emission = time.monotonic()

    def evaluate(self, features: dict[str, Any]) -> RiskSignalEvent | None:
        """Consume a risk-feature dict, maybe emit an advisory event.

        Expected keys (all rolling-window engineered features):
          - drawdown_velocity: float   dDD/dt, normalized
          - variance_ratio: float      rolling_var / baseline_var
          - strategy_pnl_dispersion: float    cross-strategy stddev
          - avg_cross_correlation: float      mean pairwise abs corr
        """
        dd_v = float(features.get("drawdown_velocity", 0.0))
        if dd_v > 0.4:
            return self._emit("RISK_ACCELERATION",
                              severity=min(dd_v, 1.0),
                              confidence=min(dd_v * 1.2, 1.0),
                              context="drawdown_velocity",
                              details={"drawdown_velocity": dd_v})

        var_ratio = float(features.get("variance_ratio", 1.0))
        if var_ratio > 2.5:
            return self._emit("REGIME_SHIFT",
                              severity=min((var_ratio - 2.0) / 3.0, 1.0),
                              confidence=0.6,
                              context="variance_expansion",
                              details={"variance_ratio": var_ratio})

        disp = float(features.get("strategy_pnl_dispersion", 0.0))
        if disp > 0.5:
            return self._emit("STRATEGY_INSTABILITY",
                              severity=min(disp, 1.0),
                              confidence=0.55,
                              context="sharpe_dispersion",
                              details={"strategy_pnl_dispersion": disp})

        corr = float(features.get("avg_cross_correlation", 0.0))
        if corr > 0.8:
            return self._emit("CORRELATION_BREAKDOWN",
                              severity=min((corr - 0.5) / 0.5, 1.0),
                              confidence=0.65,
                              context="cross_corr",
                              details={"avg_cross_correlation": corr})

        return None

    def check_self(self) -> bool:
        """N5 dead-man — advisory sensor must prove it's alive."""
        return (time.monotonic() - self._last_emission) < (self.heartbeat_interval * 3)

    # ── internals ────────────────────────────────────────────────────
    def _emit(self, kind: str, *, severity: float, confidence: float,
              context: str, details: dict[str, Any]) -> RiskSignalEvent:
        from system.time_source import now
        ts = now().utc_time.isoformat()
        event = RiskSignalEvent(
            type=kind, severity=severity, confidence=confidence,
            context=context, timestamp_utc=ts, details=details,
        )
        self._last_emission = time.monotonic()
        try:
            append_event("NEUROMORPHIC", kind, self.name, {
                "severity": severity, "confidence": confidence,
                "context": context, "details": details,
            })
        except Exception:
            pass   # ledger failure never blocks advisory emission
        return event


_singleton: NeuromorphicRisk | None = None


def get_neuromorphic_risk() -> NeuromorphicRisk:
    global _singleton
    if _singleton is None:
        _singleton = NeuromorphicRisk()
    return _singleton
