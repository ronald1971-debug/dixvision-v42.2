"""Unit tests for the S-01 Binance spot adapter (`BrokerAdapter`).

Tests use an in-process fake exchange so the suite never reaches the
real Binance API and never imports ``ccxt`` — the adapter has to
support deterministic replay (INV-15) and the test suite must remain
zero-dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from execution_engine.adapters._live_base import AdapterState
from execution_engine.adapters.base import BrokerAdapter
from execution_engine.adapters.binance import (
    NEW_PIP_DEPENDENCIES,
    BinanceAdapter,
    _safe_float,
)

# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _FakeExchange:
    """Stand-in for ``ccxt.binance`` with a deterministic response queue."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        sandbox_supported: bool = True,
    ) -> None:
        self.responses: list[Any] = responses or []
        self.calls: list[Mapping[str, Any]] = []
        self.sandbox_calls: list[bool] = []
        self.sandbox_supported = sandbox_supported

    def set_sandbox_mode(self, on: bool) -> None:  # noqa: D401  ccxt API shape
        if not self.sandbox_supported:
            raise AttributeError("set_sandbox_mode not supported")
        self.sandbox_calls.append(on)

    def create_order(
        self,
        *,
        symbol: str,
        type: str,  # noqa: A002  matches ccxt signature
        side: str,
        amount: float,
        price: float | None,
        params: Mapping[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        if not self.responses:
            raise RuntimeError("no fake response queued")
        nxt = self.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _signal(
    *,
    side: Side = Side.BUY,
    symbol: str = "BTC/USDT",
    ts_ns: int = 1_000_000,
    meta: Mapping[str, str] | None = None,
) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=0.9,
        plugin_chain=("test",),
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_implements_broker_adapter_protocol() -> None:
    """Adapter satisfies the structural ``BrokerAdapter`` Protocol."""
    adapter = BinanceAdapter()
    assert isinstance(adapter, BrokerAdapter)
    assert adapter.name == "binance_spot"


def test_module_importable_without_ccxt() -> None:
    """Importing the adapter must not require ``ccxt`` at module load."""
    import importlib

    mod = importlib.import_module("execution_engine.adapters.binance")
    assert hasattr(mod, "BinanceAdapter")
    assert "ccxt" in NEW_PIP_DEPENDENCIES


def test_default_construction_is_disconnected_scaffold() -> None:
    adapter = BinanceAdapter()
    adapter.connect()
    status = adapter.status()
    assert status.state is AdapterState.DISCONNECTED
    assert "missing" in status.detail
    assert "api_key" in status.detail
    assert "api_secret" in status.detail


def test_scaffold_mode_rejects_with_structured_meta() -> None:
    adapter = BinanceAdapter()
    adapter.connect()
    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.REJECTED
    assert out.qty == 0.0
    assert out.meta["reason"] == "adapter_not_ready"
    assert out.meta["adapter_state"] == "DISCONNECTED"


def test_partial_credentials_keeps_scaffold_mode() -> None:
    adapter = BinanceAdapter(api_key="K")
    adapter.connect()
    assert adapter.status().state is AdapterState.DISCONNECTED
    assert "api_secret" in adapter.status().detail
    assert "api_key" not in adapter.status().detail.split("api_secret")[1]


def test_injected_exchange_arms_adapter() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(exchange=fake)
    adapter.connect()
    assert adapter.status().state is AdapterState.READY
    assert "injected" in adapter.status().detail


def test_factory_called_with_credentials_and_sandbox_set() -> None:
    captured: dict[str, Any] = {}

    def factory(cfg: Mapping[str, Any]) -> _FakeExchange:
        captured["cfg"] = dict(cfg)
        return _FakeExchange()

    adapter = BinanceAdapter(
        api_key="K",
        api_secret="S",
        exchange_factory=factory,
        sandbox=True,
    )
    adapter.connect()
    assert adapter.status().state is AdapterState.READY
    assert captured["cfg"]["apiKey"] == "K"
    assert captured["cfg"]["secret"] == "S"
    assert captured["cfg"]["enableRateLimit"] is True
    assert "sandbox" in adapter.status().detail


def test_production_mode_skips_sandbox_call() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(
        api_key="K",
        api_secret="S",
        sandbox=False,
        exchange_factory=lambda cfg: fake,
    )
    adapter.connect()
    assert fake.sandbox_calls == []
    assert "production" in adapter.status().detail


def test_factory_failure_drops_to_degraded() -> None:
    def boom(_: Mapping[str, Any]) -> Any:
        raise RuntimeError("network")

    adapter = BinanceAdapter(
        api_key="K",
        api_secret="S",
        exchange_factory=boom,
    )
    adapter.connect()
    assert adapter.status().state is AdapterState.DEGRADED
    assert "RuntimeError" in adapter.status().detail


def test_market_buy_filled_normalisation() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-1",
                "status": "closed",
                "filled": 0.5,
                "amount": 0.5,
                "average": 50_000.0,
                "price": 50_000.0,
                "cost": 25_000.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=0.5)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=49_500.0)

    assert out.status is ExecutionStatus.FILLED
    assert out.qty == 0.5
    assert out.price == 50_000.0
    assert out.order_id == "BNCE-1"
    assert out.venue == "binance:spot"
    assert out.meta["adapter"] == "binance_spot"
    assert out.meta["ccxt_status"] == "closed"
    assert out.meta["order_type"] == "market"
    assert out.meta["filled_qty"] == "0.5"
    assert out.meta["notional_usd"] == "25000"
    # ts_ns is propagated from the signal — no wall-clock reads.
    assert out.ts_ns == 1_000_000

    call = fake.calls[0]
    assert call["symbol"] == "BTC/USDT"
    assert call["type"] == "market"
    assert call["side"] == "buy"
    assert call["amount"] == 0.5
    assert call["price"] is None


def test_limit_order_uses_limit_price_meta() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-2",
                "status": "open",
                "filled": 0.0,
                "amount": 1.0,
                "price": 49_900.0,
                "average": 0.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(
        _signal(meta={"order_type": "limit", "limit_price": "49900"}),
        mark_price=50_000.0,
    )

    call = fake.calls[0]
    assert call["type"] == "limit"
    assert call["price"] == pytest.approx(49_900.0)
    assert out.status is ExecutionStatus.SUBMITTED
    # No fill price yet → falls back to the limit price, then mark.
    assert out.price == 49_900.0
    assert out.meta["order_type"] == "limit"


def test_partial_fill_status_inferred_from_fill_ratio() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-3",
                "status": "closed",
                "filled": 0.4,
                "amount": 1.0,
                "average": 50_000.0,
                "price": 50_000.0,
                "cost": 20_000.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(_signal(meta={"qty": "1.0"}), mark_price=50_000.0)

    assert out.status is ExecutionStatus.PARTIALLY_FILLED
    assert out.qty == 0.4
    assert out.meta["remaining_qty"] == "0.6"


def test_cancelled_status_maps_through() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-4",
                "status": "canceled",
                "filled": 0.0,
                "amount": 0.5,
                "price": 50_000.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=0.5)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.CANCELLED
    assert out.qty == 0.0


def test_unknown_status_falls_back_to_failed() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-5",
                "status": "unicorn",
                "filled": 0.0,
                "amount": 1.0,
                "price": 50_000.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.FAILED


def test_ccxt_exception_classified_into_meta() -> None:
    fake = _FakeExchange([RuntimeError("InsufficientFunds: balance too low for BTCUSDT")])
    adapter = BinanceAdapter(exchange=fake, default_qty=0.1)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.FAILED
    assert out.qty == 0.0
    assert out.meta["reason"] == "ccxt_error"
    assert out.meta["ccxt_error_class"] == "RuntimeError"
    assert "InsufficientFunds" in out.meta["ccxt_error"]


def test_ccxt_error_message_truncated_at_512_chars() -> None:
    long_msg = "x" * 1024
    fake = _FakeExchange([RuntimeError(long_msg)])
    adapter = BinanceAdapter(exchange=fake, default_qty=0.1)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.FAILED
    assert len(out.meta["ccxt_error"]) == 512


def test_hold_signal_rejected_without_calling_exchange() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(_signal(side=Side.HOLD), mark_price=50_000.0)
    assert out.status is ExecutionStatus.REJECTED
    assert out.meta["reason"] == "HOLD signal"
    assert fake.calls == []


def test_non_positive_mark_price_fails_without_calling_exchange() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=0.0)
    assert out.status is ExecutionStatus.FAILED
    assert out.meta["reason"] == "non-positive mark_price"
    assert fake.calls == []


def test_meta_qty_overrides_default_and_validates_nan() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    # NaN qty must be rejected before reaching the exchange
    # (IEEE-754 `not (x >= 0)` pattern from PR #234).
    out = adapter.submit(
        _signal(meta={"qty": "nan"}),
        mark_price=50_000.0,
    )
    assert out.status is ExecutionStatus.REJECTED
    assert out.meta["reason"] == "non-positive qty"
    assert fake.calls == []


def test_zero_qty_rejected_before_exchange_call() -> None:
    fake = _FakeExchange()
    adapter = BinanceAdapter(exchange=fake, default_qty=0.0)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.REJECTED
    assert out.meta["reason"] == "non-positive qty"
    assert fake.calls == []


def test_replay_determinism_two_identical_calls_same_event() -> None:
    """INV-15 — same inputs + same fake response → byte-equal events."""

    def make_response() -> dict[str, Any]:
        return {
            "id": "BNCE-DET",
            "status": "closed",
            "filled": 0.5,
            "amount": 0.5,
            "average": 50_000.0,
            "price": 50_000.0,
            "cost": 25_000.0,
        }

    a = BinanceAdapter(
        exchange=_FakeExchange([make_response()]),
        default_qty=0.5,
    )
    b = BinanceAdapter(
        exchange=_FakeExchange([make_response()]),
        default_qty=0.5,
    )
    a.connect()
    b.connect()
    sig = _signal()
    out_a = a.submit(sig, mark_price=49_500.0)
    out_b = b.submit(sig, mark_price=49_500.0)

    # Two adapters in identical state, identical inputs, identical
    # response — the only difference is the per-adapter monotonic
    # counter, which both have at 1 after a single call.
    assert out_a == out_b


def test_invalid_default_order_type_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="default_order_type"):
        BinanceAdapter(default_order_type="iceberg")  # type: ignore[arg-type]


def test_invalid_default_qty_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="default_qty"):
        BinanceAdapter(default_qty=-1.0)


def test_disconnect_clears_exchange_handle() -> None:
    adapter = BinanceAdapter(exchange=_FakeExchange())
    adapter.connect()
    assert adapter.status().state is AdapterState.READY
    adapter.disconnect()
    assert adapter.status().state is AdapterState.DISCONNECTED
    # After disconnect a submit must reject — we never silently fake a
    # fill (Triad Lock INV-56).
    out = adapter.submit(_signal(), mark_price=50_000.0)
    assert out.status is ExecutionStatus.REJECTED


def test_safe_float_rejects_nan_and_inf() -> None:
    assert _safe_float(None) is None
    assert _safe_float("not a number") is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float(float("-inf")) is None
    assert _safe_float("3.14") == pytest.approx(3.14)
    assert _safe_float(0) == 0.0


def test_filled_qty_falls_back_to_amount_when_filled_missing() -> None:
    """ccxt sometimes omits `filled` on closed orders — must fall back."""
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-AMT",
                "status": "closed",
                "amount": 2.0,
                "average": 100.0,
                "cost": 200.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=2.0)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=100.0)
    assert out.status is ExecutionStatus.FILLED
    assert out.qty == 2.0


def test_average_zero_falls_back_to_price_then_mark() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-AVG",
                "status": "closed",
                "filled": 1.0,
                "amount": 1.0,
                "average": 0.0,
                "price": 0.0,  # both unusable
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=1.0)
    adapter.connect()

    out = adapter.submit(_signal(), mark_price=49_321.0)
    # Both ccxt-side prices are zero → fall back to the inbound mark.
    assert out.price == 49_321.0
    # Sanity: not NaN.
    assert math.isfinite(out.price)


def test_event_carries_engine_provenance() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-PROV",
                "status": "closed",
                "filled": 0.5,
                "amount": 0.5,
                "average": 50.0,
                "price": 50.0,
                "cost": 25.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=0.5)
    adapter.connect()
    out = adapter.submit(_signal(), mark_price=50.0)
    # HARDEN-03 / INV-69 — receiver-side assertions read this label.
    assert out.produced_by_engine == "execution_engine"


def test_returns_execution_event_type() -> None:
    fake = _FakeExchange(
        [
            {
                "id": "BNCE-T",
                "status": "closed",
                "filled": 0.5,
                "amount": 0.5,
                "average": 50.0,
                "price": 50.0,
            }
        ]
    )
    adapter = BinanceAdapter(exchange=fake, default_qty=0.5)
    adapter.connect()
    out = adapter.submit(_signal(), mark_price=50.0)
    assert isinstance(out, ExecutionEvent)
