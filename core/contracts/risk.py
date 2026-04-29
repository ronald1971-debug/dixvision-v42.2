"""Risk-side contracts shared across engine boundaries.

Pure data types — no engine logic, no clock reads, no I/O. May be
imported from any engine package (``ALLOWED_SHARED_PREFIXES`` in
``authority_lint``).

The :class:`RiskSnapshot` here is the canonical, frozen view of the
FastRiskCache that callers consume on the hot path
(``execution_engine.hot_path.FastExecutor``) and that the hazard
throttle layer (``system_engine.coupling.apply_throttle``) tightens
in-line. Centralising it under ``core.contracts`` lets both engines
read it without violating B1 (no cross-engine direct imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """A frozen view of the FastRiskCache for a single tick.

    Attributes:
        version: Monotonic version of the underlying cache; consumed
            by replay determinism (INV-15).
        ts_ns: Timestamp of the snapshot.
        max_position_qty: Per-symbol qty cap; ``None`` = unbounded.
        max_signal_confidence: Floor on signal confidence; signals
            below this are rejected.
        symbol_caps: Optional per-symbol overrides for
            ``max_position_qty``.
        halted: True when governance has halted the hot path
            (kill-switch / SAFE mode); all signals reject.
    """

    version: int
    ts_ns: int
    max_position_qty: float | None = None
    max_signal_confidence: float = 0.0
    symbol_caps: dict[str, float] = field(default_factory=dict)
    halted: bool = False

    def cap_for(self, symbol: str) -> float | None:
        cap = self.symbol_caps.get(symbol)
        if cap is not None:
            return cap
        return self.max_position_qty


__all__ = ["RiskSnapshot"]
