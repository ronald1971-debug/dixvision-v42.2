"""Phase 2 — adapter router unit tests."""

from __future__ import annotations

import pytest

from core.contracts.events import ExecutionEvent, ExecutionStatus, Side, SignalEvent
from execution_engine.adapters.router import (
    AdapterRouter,
    RouterError,
    TradingDomain,
)


class _StubAdapter:
    """Minimal :class:`BrokerAdapter` for routing tests."""

    def __init__(self, name: str) -> None:
        self.name = name

    def submit(
        self,
        signal: SignalEvent,
        mark_price: float,  # noqa: ARG002
    ) -> ExecutionEvent:
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=1.0,
            price=mark_price,
            status=ExecutionStatus.FILLED,
            venue=self.name,
            order_id="o1",
        )


def _signal(domain: str | None = None, venue: str | None = None) -> SignalEvent:
    meta: dict[str, str] = {}
    if domain is not None:
        meta["domain"] = domain
    if venue is not None:
        meta["venue"] = venue
    return SignalEvent(
        ts_ns=1,
        symbol="BTC-USD",
        side=Side.BUY,
        confidence=0.9,
        meta=meta,
    )


def test_register_then_lookup_returns_adapter():
    router = AdapterRouter()
    paper = _StubAdapter("paper")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=paper)
    assert router.adapter_for(_signal(venue="paper")) is paper


def test_default_domain_normal_when_meta_omitted():
    router = AdapterRouter()
    paper = _StubAdapter("paper")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=paper)
    assert router.adapter_for(_signal(venue="paper")) is paper


def test_explicit_domain_routes_to_matching_adapter():
    router = AdapterRouter()
    paper = _StubAdapter("paper")
    burner = _StubAdapter("memecoin")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=paper)
    router.register(
        domain=TradingDomain.MEMECOIN, venue="paper", adapter=burner
    )
    assert (
        router.adapter_for(_signal(domain="MEMECOIN", venue="paper")) is burner
    )


def test_normal_signal_cannot_reach_memecoin_adapter():
    router = AdapterRouter()
    burner = _StubAdapter("memecoin")
    router.register(
        domain=TradingDomain.MEMECOIN, venue="paper", adapter=burner
    )
    with pytest.raises(RouterError):
        router.adapter_for(_signal(domain="NORMAL", venue="paper"))


def test_memecoin_signal_cannot_reach_normal_adapter():
    router = AdapterRouter()
    paper = _StubAdapter("paper")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=paper)
    with pytest.raises(RouterError):
        router.adapter_for(_signal(domain="MEMECOIN", venue="paper"))


def test_unknown_domain_raises():
    router = AdapterRouter()
    with pytest.raises(RouterError):
        router.adapter_for(_signal(domain="WHATEVER", venue="paper"))


def test_missing_venue_raises():
    router = AdapterRouter()
    with pytest.raises(RouterError):
        router.adapter_for(_signal())


def test_register_duplicate_raises():
    router = AdapterRouter()
    a = _StubAdapter("paper")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=a)
    with pytest.raises(ValueError):
        router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=a)


def test_register_blank_venue_raises():
    router = AdapterRouter()
    with pytest.raises(ValueError):
        router.register(
            domain=TradingDomain.NORMAL, venue="", adapter=_StubAdapter("x")
        )


def test_venues_listed_per_domain():
    router = AdapterRouter()
    router.register(
        domain=TradingDomain.NORMAL, venue="binance", adapter=_StubAdapter("b")
    )
    router.register(
        domain=TradingDomain.NORMAL, venue="kraken", adapter=_StubAdapter("k")
    )
    router.register(
        domain=TradingDomain.MEMECOIN,
        venue="paper",
        adapter=_StubAdapter("m"),
    )
    assert router.venues(TradingDomain.NORMAL) == ("binance", "kraken")
    assert router.venues(TradingDomain.MEMECOIN) == ("paper",)


def test_venues_listed_when_constructed_with_plain_str_keys():
    """Constructor accepts plain str domain keys; venues() must still find them."""
    a = _StubAdapter("paper")
    router = AdapterRouter(adapters={(TradingDomain.NORMAL, "paper"): a})
    assert router.venues(TradingDomain.NORMAL) == ("paper",)
    assert router.adapter_for(_signal(domain="NORMAL", venue="paper")) is a


def test_venue_argument_overrides_meta():
    router = AdapterRouter()
    a = _StubAdapter("paper")
    b = _StubAdapter("binance")
    router.register(domain=TradingDomain.NORMAL, venue="paper", adapter=a)
    router.register(domain=TradingDomain.NORMAL, venue="binance", adapter=b)
    sig = _signal(venue="paper")
    assert router.adapter_for(sig, venue="binance") is b
