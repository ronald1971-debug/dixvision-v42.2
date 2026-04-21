"""
mind.strategy_arbiter \u2014 aggregate signals + alpha-decay gate.

Each strategy has a rolling realized-reward window (from episodic memory).
Strategies whose recent edge decays below threshold get auto-demoted to
shadow; the arbiter stops routing their signals to execution. Governance
must re-promote.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from mind.strategies import Strategy, StrategySignal, list_builtins
from state.episodic_memory import get_episodic_memory

DECAY_WINDOW = 50
DECAY_MIN_MEAN = -0.05


@dataclass
class StrategyState:
    name: str
    active: bool = True
    shadow: bool = False
    reward_mean: float = 0.0
    reward_n: int = 0
    last_signal: StrategySignal | None = None


class StrategyArbiter:
    def __init__(self, strategies: list[Strategy] | None = None) -> None:
        self._strategies: dict[str, Strategy] = {
            s.name: s for s in (strategies or list_builtins())
        }
        self._state: dict[str, StrategyState] = {
            n: StrategyState(name=n) for n in self._strategies
        }
        self._lock = threading.RLock()

    def set_shadow(self, name: str, shadow: bool) -> None:
        with self._lock:
            if name in self._state:
                self._state[name].shadow = bool(shadow)

    def set_active(self, name: str, active: bool) -> None:
        with self._lock:
            if name in self._state:
                self._state[name].active = bool(active)

    def state(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return {
                n: {
                    "active": s.active, "shadow": s.shadow,
                    "reward_mean": round(s.reward_mean, 4),
                    "reward_n": s.reward_n,
                    "last": s.last_signal.side if s.last_signal else "",
                } for n, s in self._state.items()
            }

    def refresh_decay(self) -> None:
        mem = get_episodic_memory()
        with self._lock:
            for name, st in self._state.items():
                rewards = mem.reward_window(name, n=DECAY_WINDOW)
                st.reward_n = len(rewards)
                if rewards:
                    mean = sum(rewards) / len(rewards)
                    st.reward_mean = mean
                    if mean < DECAY_MIN_MEAN and not st.shadow:
                        st.shadow = True
                else:
                    st.reward_mean = 0.0

    def propose(self, *, symbol: str, features: dict[str, float]) -> list[StrategySignal]:
        out: list[StrategySignal] = []
        with self._lock:
            for name, strat in self._strategies.items():
                st = self._state[name]
                if not st.active:
                    continue
                sig = strat.propose(symbol=symbol, features=features)
                if sig is None:
                    continue
                st.last_signal = sig
                out.append(sig)
        return out

    def fuse(self, signals: list[StrategySignal]) -> StrategySignal | None:
        """Weighted-sum fusion: shadow strategies contribute 0."""
        if not signals:
            return None
        net = 0.0
        weight = 0.0
        for sig in signals:
            st = self._state.get(sig.strategy)
            w = 0.0 if (st and st.shadow) else 1.0
            if sig.side == "buy":
                net += sig.strength * w
            elif sig.side == "sell":
                net -= sig.strength * w
            weight += w
        if weight == 0.0:
            return None
        strength = net / weight
        if abs(strength) < 0.15:
            side = "flat"
        elif strength > 0:
            side = "buy"
        else:
            side = "sell"
        return StrategySignal(
            strategy="arbiter", symbol=signals[0].symbol, side=side,
            strength=max(-1.0, min(1.0, strength)),
            rationale=f"fused={strength:.3f} ({len(signals)} strategies)",
        )


_singleton: StrategyArbiter | None = None
_lock = threading.Lock()


def get_arbiter() -> StrategyArbiter:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = StrategyArbiter()
    return _singleton


__all__ = ["StrategyArbiter", "StrategyState", "get_arbiter"]
