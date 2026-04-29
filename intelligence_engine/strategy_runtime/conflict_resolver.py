"""Conflict resolver — Phase 3 / v2-B.

Coalesces multiple :class:`SignalEvent` rows that target the same symbol
in the same tick into a single coordinated decision. Pure-Python,
IO-free, deterministic.

Resolution policy (defaults; tunable via constructor):

1. Group by ``symbol``.
2. Compute confidence-weighted vote per side (BUY / SELL / HOLD).
3. If the winning side's net score is below ``min_net_score`` → emit
   ``HOLD``.
4. Output confidence is the absolute net score, clipped to ``[0.0, 1.0]``.
5. ``plugin_chain`` is the union of contributing chains, in input order.
6. ``meta`` is merged left-to-right for keys that don't already exist on
   the leading signal — the *first* signal in input order wins on
   collision (deterministic).

The resolver does not invent new sides or new symbols; it only ranks +
collapses what was given.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from core.contracts.events import Side, SignalEvent


@dataclass(frozen=True, slots=True)
class ConflictResolution:
    """Audit metadata for one resolution."""

    symbol: str
    winning_side: Side
    net_score: float
    contributing: tuple[str, ...]


class ConflictResolver:
    """Collapse conflicting signals on the same symbol."""

    name: str = "conflict_resolver"
    spec_id: str = "IND-CFR-01"

    def __init__(
        self,
        *,
        min_net_score: float = 0.0,
    ) -> None:
        if min_net_score < 0.0:
            raise ValueError("min_net_score must be >= 0")
        self._min_net_score = min_net_score

    def resolve(
        self,
        signals: Iterable[SignalEvent],
    ) -> tuple[tuple[SignalEvent, ConflictResolution], ...]:
        signals_t = tuple(signals)
        if not signals_t:
            return ()

        groups: dict[str, list[SignalEvent]] = {}
        order: list[str] = []
        for s in signals_t:
            if s.symbol not in groups:
                groups[s.symbol] = []
                order.append(s.symbol)
            groups[s.symbol].append(s)

        out: list[tuple[SignalEvent, ConflictResolution]] = []
        for symbol in order:
            group = groups[symbol]
            out.append(self._resolve_group(symbol, group))
        return tuple(out)

    # -- internals ---------------------------------------------------------

    def _resolve_group(
        self, symbol: str, group: Sequence[SignalEvent]
    ) -> tuple[SignalEvent, ConflictResolution]:
        score: dict[Side, float] = {Side.BUY: 0.0, Side.SELL: 0.0}
        for s in group:
            if s.side is Side.BUY:
                score[Side.BUY] += s.confidence
            elif s.side is Side.SELL:
                score[Side.SELL] += s.confidence
            # HOLD votes do not count toward either side.

        net = score[Side.BUY] - score[Side.SELL]
        if abs(net) <= self._min_net_score:
            winning_side = Side.HOLD
            confidence = 0.0
        elif net > 0:
            winning_side = Side.BUY
            confidence = min(1.0, net)
        else:
            winning_side = Side.SELL
            confidence = min(1.0, -net)

        # Stable plugin chain union (preserve first-occurrence order).
        seen: set[str] = set()
        chain: list[str] = []
        for s in group:
            for p in s.plugin_chain:
                if p not in seen:
                    seen.add(p)
                    chain.append(p)

        # Merge meta: first signal wins on collision.
        merged_meta: dict[str, str] = {}
        for s in group:
            for k, v in s.meta.items():
                if k not in merged_meta:
                    merged_meta[k] = v
        merged_meta["resolved_by"] = self.name

        leader = group[0]
        coalesced = SignalEvent(
            ts_ns=leader.ts_ns,
            symbol=symbol,
            side=winning_side,
            confidence=confidence,
            plugin_chain=tuple(chain),
            meta=merged_meta,
            produced_by_engine="intelligence_engine",
        )
        resolution = ConflictResolution(
            symbol=symbol,
            winning_side=winning_side,
            net_score=net,
            contributing=tuple(s.plugin_chain[0] for s in group if s.plugin_chain),
        )
        return coalesced, resolution


__all__ = ["ConflictResolution", "ConflictResolver"]
