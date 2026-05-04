"""TradingView Pine-script alert receiver (Paper-S4).

Pure parser + canonical envelope for the operator-controlled
``POST /api/feeds/tradingview/alert`` webhook in :mod:`ui.server`.

Why a parser-only adapter (mirrors :mod:`ui.feeds.tradingview_ideas`):

* TradingView Pine-script ``strategy.entry`` / ``alert()`` hooks fire
  at an operator-configured webhook URL with a JSON body the operator
  composes inside Pine. There is no public TradingView stream we can
  pull; the only honest ingest path is "TradingView pushes, harness
  receives". Building a parser-only seam means the webhook receiver
  in :mod:`ui.server` is a thin shell.
* Paper-S4 — the resulting :class:`SignalEvent` is stamped with
  :class:`SignalTrust.EXTERNAL_LOW` so the governance gate applies the
  per-source confidence cap from
  ``registry/external_signal_trust.yaml`` (row ``tv:tradingview_alert``)
  before the intent reaches the execute chokepoint.

INV-15: parser is pure. ``ts_ns`` is caller-supplied; the parser
never reads a system clock. Two replays of the same input produce
byte-identical output.

Canonical Pine-alert envelope (version 1):

    {
        "version": 1,
        "ticker": "BTCUSDT",
        "side": "BUY" | "SELL",          // or alias "long"/"short"
        "confidence": 0.62,                 // optional, [0.0, 1.0]
        "qty": "0.05",                      // optional, free-form str
        "strategy": "pine_breakout_v3",   // optional, audit only
        "comment": "RSI < 30",             // optional, audit only
    }

The parser tolerates the most common Pine alert shapes:

* TradingView's built-in ``{{strategy.order.action}}`` placeholder
  emits ``"buy"`` / ``"sell"`` (lowercase) — the parser uppercases
  before constructing :class:`Side`.
* Pine integers / floats arrive as JSON numbers; ``confidence`` is
  cast to float and clamped to ``[0.0, 1.0]``. Out-of-range values are
  not rejected (Pine has no clamp helper); the cap is enforced by the
  trust class downstream.
* ``ticker`` is also accepted as ``symbol`` for parity with the
  existing :func:`ui.server.post_signal` envelope.

Failure modes return :data:`None` so the webhook receiver can return a
non-200 ``{"accepted": False, "reason": ...}`` body without raising —
TradingView's webhook engine retries on 4xx/5xx and we do not want
malformed Pine alerts to trigger a retry storm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from core.contracts.events import Side, SignalEvent
from core.contracts.signal_trust import SignalTrust

TRADINGVIEW_ALERT_SOURCE_FEED = "SRC-SIGNAL-TRADINGVIEW-ALERT-001"
"""Stable source id for ``registry/external_signal_trust.yaml``."""

TRADINGVIEW_ALERT_PRODUCED_BY_ENGINE = "ui.feeds.tradingview_alert"
"""HARDEN-03 / INV-69 producer marker for receiver assertions."""


@dataclass(frozen=True, slots=True)
class TradingViewAlertParseResult:
    """Output of :func:`parse_tradingview_alert_payload`.

    The parser deliberately returns the unstamped :class:`SignalEvent`
    only after the webhook receiver layer composes the final
    ``meta`` mapping. This keeps the parser pure and the receiver
    thin.
    """

    signal: SignalEvent
    """Fully-typed SignalEvent stamped EXTERNAL_LOW + source = TV alert."""

    audit_meta: Mapping[str, str]
    """Receiver-friendly free-form audit metadata (e.g. strategy name)."""


def _coerce_side(raw: Any) -> Side | None:
    if not isinstance(raw, str):
        return None
    norm = raw.strip().upper()
    aliases = {"LONG": "BUY", "SHORT": "SELL"}
    norm = aliases.get(norm, norm)
    try:
        return Side(norm)
    except ValueError:
        return None


def _coerce_confidence(raw: Any) -> float:
    """Best-effort confidence coercion clamped to ``[0.0, 1.0]``.

    Defaults to ``0.5`` when missing or unparseable so a caller-side
    Pine alert without an explicit confidence still flows through the
    cap-applying governance gate (the trust class is the real lever).
    """

    if raw is None:
        return 0.5
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.5
    # IEEE-754 NaN compares unequal to everything; the < / > clamps would
    # silently let it through and downstream code (clamp_confidence,
    # SignalEvent.confidence invariant) would either raise or corrupt
    # arithmetic. Treat NaN/inf as "missing" so the webhook stays HTTP 200.
    if v != v or v in (float("inf"), float("-inf")):
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def parse_tradingview_alert_payload(
    payload: Mapping[str, Any] | None,
    *,
    ts_ns: int,
) -> TradingViewAlertParseResult | None:
    """Parse one Pine-script alert envelope into a :class:`SignalEvent`.

    Args:
        payload: Decoded JSON body the operator's Pine alert pushed to
            ``POST /api/feeds/tradingview/alert``. ``None`` or
            non-mapping inputs return ``None``.
        ts_ns: Caller-supplied timestamp in nanoseconds (TimeAuthority
            on the server side; the alert envelope itself is untrusted).

    Returns:
        :data:`None` for unrecognised / malformed payloads (so the
        receiver can answer ``{"accepted": False, ...}`` without
        raising). Otherwise a :class:`TradingViewAlertParseResult` with
        a fully-typed :class:`SignalEvent` stamped with
        :attr:`SignalTrust.EXTERNAL_LOW` and
        ``signal_source = TRADINGVIEW_ALERT_SOURCE_FEED``.
    """

    if not isinstance(payload, Mapping):
        return None

    version = payload.get("version")
    if version is not None and version != 1:
        return None

    ticker = payload.get("ticker") or payload.get("symbol")
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    symbol = ticker.strip()

    side = _coerce_side(payload.get("side") or payload.get("action"))
    if side is None:
        return None

    confidence = _coerce_confidence(payload.get("confidence"))

    audit: dict[str, str] = {}
    qty = payload.get("qty")
    if qty is not None:
        audit["qty"] = str(qty)
    strategy = payload.get("strategy")
    if isinstance(strategy, str) and strategy:
        audit["strategy"] = strategy
    comment = payload.get("comment")
    if isinstance(comment, str) and comment:
        audit["comment"] = comment[:200]

    signal = SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        plugin_chain=("tradingview_alert",),
        meta=dict(audit),
        produced_by_engine=TRADINGVIEW_ALERT_PRODUCED_BY_ENGINE,
        signal_trust=SignalTrust.EXTERNAL_LOW,
        signal_source=TRADINGVIEW_ALERT_SOURCE_FEED,
    )
    return TradingViewAlertParseResult(
        signal=signal,
        audit_meta=MappingProxyType(dict(audit)),
    )


__all__ = [
    "TRADINGVIEW_ALERT_PRODUCED_BY_ENGINE",
    "TRADINGVIEW_ALERT_SOURCE_FEED",
    "TradingViewAlertParseResult",
    "parse_tradingview_alert_payload",
]
