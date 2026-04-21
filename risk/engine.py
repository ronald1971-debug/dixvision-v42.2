"""risk.engine \u2014 bounded VaR/ES + vol-target sizing + regime label.

Inputs are plain lists/floats; no DB, no network. Caller feeds rolling
windows from the sources layer (bounded via deques). All math is O(n).
Stdlib only \u2014 no numpy dependency.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskSnapshot:
    var_95: float
    var_99: float
    es_975: float
    vol_daily: float
    sharpe_estimate: float
    regime: str                       # TREND | MIXED | RANGE | UNKNOWN
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "var_95": self.var_95, "var_99": self.var_99, "es_975": self.es_975,
            "vol_daily": self.vol_daily,
            "sharpe_estimate": self.sharpe_estimate,
            "regime": self.regime, "n_obs": self.n_obs,
        }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float], m: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def compute_var_es(returns: Sequence[float], *,
                   horizon_days: float = 1.0) -> RiskSnapshot:
    """Historical simulation VaR/ES on a returns window (e.g. daily)."""
    xs = list(returns)
    n = len(xs)
    if n < 5:
        return RiskSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, "UNKNOWN", n)
    m = _mean(xs)
    sd = _stdev(xs, m)
    var_95 = -_percentile(xs, 0.05)
    var_99 = -_percentile(xs, 0.01)
    tail = [-x for x in xs if x <= _percentile(xs, 0.025)]
    es_975 = sum(tail) / len(tail) if tail else 0.0
    scale = math.sqrt(max(0.0, horizon_days))
    sharpe = (m / sd * math.sqrt(252)) if sd > 0 else 0.0
    regime = rolling_regime_label(xs)
    return RiskSnapshot(
        var_95=round(var_95 * scale, 6),
        var_99=round(var_99 * scale, 6),
        es_975=round(es_975 * scale, 6),
        vol_daily=round(sd, 6),
        sharpe_estimate=round(sharpe, 3),
        regime=regime,
        n_obs=n,
    )


def rolling_regime_label(returns: Sequence[float],
                         fast_window: int = 10,
                         slow_window: int = 40) -> str:
    """Cheap regime heuristic: compare fast vs slow vol + autocorr sign.

    TREND  == slow vol rising, mean significantly non-zero
    RANGE  == vol compressing, mean near zero
    MIXED  == neither clearly
    """
    xs = list(returns)
    if len(xs) < max(fast_window, slow_window) + 2:
        return "UNKNOWN"
    fast = xs[-fast_window:]
    slow = xs[-slow_window:]
    m_f, m_s = _mean(fast), _mean(slow)
    sd_f = _stdev(fast, m_f) or 1e-9
    sd_s = _stdev(slow, m_s) or 1e-9
    vol_ratio = sd_s / sd_f
    trend_strength = abs(m_s) / sd_s
    if trend_strength > 0.15 and vol_ratio > 0.9:
        return "TREND"
    if trend_strength < 0.05 and vol_ratio < 1.1:
        return "RANGE"
    return "MIXED"


def position_sizing(*, equity: float, target_vol_annual: float,
                    asset_vol_daily: float,
                    drawdown: float = 0.0,
                    regime: str = "MIXED",
                    max_leverage: float = 3.0) -> float:
    """Vol-targeted sizing with drawdown + regime scaler.

    Returns notional exposure in same units as `equity`.
    """
    if equity <= 0 or asset_vol_daily <= 0:
        return 0.0
    annual = asset_vol_daily * math.sqrt(252)
    base = equity * (target_vol_annual / annual) if annual > 0 else 0.0
    dd_scaler = 1.0
    if drawdown > 0.1:
        dd_scaler = max(0.25, 1.0 - (drawdown - 0.1) * 2.0)
    reg = {"TREND": 1.0, "MIXED": 0.7, "RANGE": 0.5, "UNKNOWN": 0.5}.get(regime, 0.7)
    size = base * dd_scaler * reg
    max_notional = equity * max_leverage
    return round(min(max_notional, max(0.0, size)), 2)


__all__ = ["RiskSnapshot", "compute_var_es",
           "position_sizing", "rolling_regime_label"]
