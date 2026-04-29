"""Apply a :class:`ThrottleDecision` to a frozen :class:`RiskSnapshot`.

A pure mutator (INV-64): identical inputs → identical output. The
result is a *new* :class:`RiskSnapshot`; the input is never mutated.

Monotonically restrictive (SAFE-67):

* ``halted`` only goes ``False → True``.
* ``max_signal_confidence`` only rises (``max(...)``).
* ``max_position_qty`` and every entry of ``symbol_caps`` are
  multiplied by ``decision.qty_multiplier ∈ [0, 1]``, so they only
  shrink (or stay unchanged).
* ``version`` and ``ts_ns`` are preserved — the snapshot's identity
  is unchanged; the throttle is a *projection*, not a new
  observation.

Imports :class:`RiskSnapshot` from ``core.contracts.risk`` (not from
``execution_engine``) — keeping the dependency arrows pointed at
shared contracts only (B1 / INV-08 / INV-11).
"""

from __future__ import annotations

from core.contracts.risk import RiskSnapshot
from system_engine.coupling.hazard_throttle import ThrottleDecision


def apply_throttle(
    *,
    snapshot: RiskSnapshot,
    decision: ThrottleDecision,
) -> RiskSnapshot:
    """Project a throttle decision onto a risk snapshot.

    Args:
        snapshot: Frozen snapshot from the FastRiskCache.
        decision: Output of :func:`compute_throttle`.

    Returns:
        A new :class:`RiskSnapshot` with the throttle projected. The
        input snapshot is unchanged.
    """
    halted = snapshot.halted or decision.block

    if decision.confidence_floor > snapshot.max_signal_confidence:
        max_signal_confidence = decision.confidence_floor
    else:
        max_signal_confidence = snapshot.max_signal_confidence

    if decision.qty_multiplier >= 1.0:
        max_position_qty = snapshot.max_position_qty
        symbol_caps = dict(snapshot.symbol_caps)
    else:
        max_position_qty = (
            snapshot.max_position_qty * decision.qty_multiplier
            if snapshot.max_position_qty is not None
            else None
        )
        symbol_caps = {
            sym: cap * decision.qty_multiplier
            for sym, cap in snapshot.symbol_caps.items()
        }

    return RiskSnapshot(
        version=snapshot.version,
        ts_ns=snapshot.ts_ns,
        max_position_qty=max_position_qty,
        max_signal_confidence=max_signal_confidence,
        symbol_caps=symbol_caps,
        halted=halted,
    )


__all__ = ["apply_throttle"]
