"""
state/projectors/market_state.py
Projects MARKET ledger events into an in-memory read model.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class MarketReadModel:
    last_price_by_asset: dict[str, float] = field(default_factory=dict)
    last_volume_by_asset: dict[str, float] = field(default_factory=dict)


class MarketStateProjector:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = MarketReadModel()

    def apply(self, event: dict) -> None:
        if str(event.get("event_type", "")).upper() != "MARKET":
            return
        p = event.get("payload", {}) or {}
        asset = p.get("asset")
        if not asset:
            return
        with self._lock:
            if "price" in p:
                self._model.last_price_by_asset[asset] = float(p["price"])
            if "volume" in p:
                self._model.last_volume_by_asset[asset] = float(p["volume"])

    def snapshot(self) -> MarketReadModel:
        with self._lock:
            return MarketReadModel(
                last_price_by_asset=dict(self._model.last_price_by_asset),
                last_volume_by_asset=dict(self._model.last_volume_by_asset),
            )


_p: MarketStateProjector | None = None
_lock = threading.Lock()


def get_market_projector() -> MarketStateProjector:
    global _p
    if _p is None:
        with _lock:
            if _p is None:
                _p = MarketStateProjector()
    return _p
