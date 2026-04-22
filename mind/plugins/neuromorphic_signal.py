"""
mind/plugins/neuromorphic_signal.py
DIX VISION v42.2 — Indira-side neuromorphic sensor (observe + emit only).

Phase 0 stub: event-emitting scaffolding. No SNN model yet — Phase 2
replaces the rule-based spike detector here with a 64-step-window SNN
offline-trained to ONNX (see docs/NEUROMORPHIC_TRIAD_SPEC.md §1).

Axioms N1..N8 (immutable_core/neuromorphic_axioms.lean) strictly bound
what this module is allowed to do:
  - observes L2 microstructure features,
  - emits SPIKE_SIGNAL_EVENT as a ledger event + plugin signal,
  - NEVER decides, executes, or mutates governance / risk state.

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

from . import _BasePlugin

SPIKE_TYPES = (
    "VOLATILITY_SPIKE",
    "OFI_SPIKE",
    "MOMENTUM_IGNITION",
    "LIQUIDITY_SHOCK",
)


@dataclass
class SpikeSignalEvent:
    """Wire format for SPIKE_SIGNAL_EVENT — consumed by Indira as one
    feature among many. MUST NOT be treated as a decision."""

    type: str                           # one of SPIKE_TYPES
    intensity: float                    # 0.0..1.0
    direction: str                      # "UP" | "DOWN" | "NEUTRAL"
    confidence: float                   # 0.0..1.0
    venue: str
    timestamp_utc: str
    sequence: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class NeuromorphicSignalPlugin(_BasePlugin):
    """Microstructure sensor. Emits events; never decides.

    Phase 0: rule-based threshold detector over engineered features.
    Phase 2: SNN backend (snntorch / spikingjelly) with frozen ONNX weights.
    """

    name = "neuromorphic_signal"
    # Dead-man — N5.
    heartbeat_interval: float = 1.0     # seconds

    def __init__(self) -> None:
        # ``_last_tick_seen`` proves the sensor is *alive* (called every
        # evaluation). ``_last_emission`` is kept for diagnostics — an
        # emission also counts as a tick for the dead-man.
        self._last_tick_seen = time.monotonic()
        self._last_emission = time.monotonic()
        self._seq = 0

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Consume a feature dict → maybe emit a spike event.

        Expected features (engineered, NOT raw price):
          - volatility: float     rolling realized vol
          - ofi: float            order-flow imbalance (-1..1)
          - momentum: float       normalized price rate-of-change
          - liquidity_delta: float  book depth change ratio
          - venue: str
        """
        # N5 dead-man: every evaluation is proof-of-life, regardless of
        # whether a spike threshold is crossed. A calm market must not
        # trip the dead-man.
        self._last_tick_seen = time.monotonic()
        venue = str(data.get("venue", "unknown"))
        vol = float(data.get("volatility", 0.0))
        ofi = float(data.get("ofi", 0.0))
        momentum = float(data.get("momentum", 0.0))
        liq = float(data.get("liquidity_delta", 0.0))

        event: SpikeSignalEvent | None = None
        if vol > 0.08:
            event = self._spike("VOLATILITY_SPIKE",
                                intensity=min(vol / 0.2, 1.0),
                                direction="NEUTRAL",
                                venue=venue,
                                details={"volatility": vol})
        elif abs(ofi) > 0.7:
            event = self._spike("OFI_SPIKE",
                                intensity=min(abs(ofi), 1.0),
                                direction="UP" if ofi > 0 else "DOWN",
                                venue=venue,
                                details={"ofi": ofi})
        elif abs(momentum) > 0.6:
            event = self._spike("MOMENTUM_IGNITION",
                                intensity=min(abs(momentum), 1.0),
                                direction="UP" if momentum > 0 else "DOWN",
                                venue=venue,
                                details={"momentum": momentum})
        elif liq < -0.5:
            event = self._spike("LIQUIDITY_SHOCK",
                                intensity=min(abs(liq), 1.0),
                                direction="DOWN",
                                venue=venue,
                                details={"liquidity_delta": liq})

        # Return an advisory signal for the plugin contract. Indira MUST
        # still make the trade decision deterministically.
        if event is None:
            return {"signal": 0.0, "confidence": 0.0,
                    "strategy": self.name, "spike": None}
        return {
            "signal": (event.intensity if event.direction == "UP"
                       else -event.intensity if event.direction == "DOWN"
                       else 0.0),
            "confidence": event.confidence,
            "strategy": self.name,
            "spike": {
                "type": event.type, "intensity": event.intensity,
                "direction": event.direction, "venue": event.venue,
            },
        }

    def check_self(self) -> bool:
        """Dead-man (N5): detector proves it's alive.

        Governance / system dead-man reads this; if False beyond the
        operator grace window, trading halts fail-closed.

        Liveness is proven by *any* ``evaluate()`` call, not just by
        spike emission — otherwise a calm market (no threshold crossed)
        would falsely trip the dead-man.
        """
        return (time.monotonic() - self._last_tick_seen) < (self.heartbeat_interval * 3)

    # ── internals ────────────────────────────────────────────────────
    def _spike(self, kind: str, *, intensity: float, direction: str,
               venue: str, details: dict[str, Any]) -> SpikeSignalEvent:
        from system.time_source import now
        ts = now()
        self._seq += 1
        event = SpikeSignalEvent(
            type=kind, intensity=intensity, direction=direction,
            confidence=intensity,           # placeholder until SNN lands
            venue=venue,
            timestamp_utc=ts.utc_time.isoformat(),
            sequence=self._seq,
            details=details,
        )
        self._last_emission = time.monotonic()
        self._last_tick_seen = self._last_emission
        # N4: every emission writes a ledger row — replayable.
        try:
            append_event("NEUROMORPHIC", kind, self.name, {
                "intensity": event.intensity,
                "direction": event.direction,
                "confidence": event.confidence,
                "venue": event.venue,
                "details": event.details,
            })
        except Exception:
            pass   # ledger failure never blocks sensor emission
        return event
