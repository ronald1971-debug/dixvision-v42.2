"""Read-only TradingView trader-feed adapter (SRC-TRADER-TRADINGVIEW-001).

Wave-04 PR-2 — the first trader-feed adapter wired into the
Trader-Intelligence layer (PR-1 contracts in
:mod:`core.contracts.trader_intelligence`). The pump produces
:class:`core.contracts.trader_intelligence.TraderModel` payloads which
the trader-modeling subsystem (sole B29-allowed producer) wraps into
:class:`TraderObservation` records emitted as
``SystemEvent(sub_kind=TRADER_OBSERVED)``.

Why a parser-only adapter (no live WS/scrape):

* TradingView does not expose an unauthenticated public stream for
  trader signals. The two real ingest paths are
  `Pine Script alerts → operator-configured webhook` and
  `TradingView "Ideas" pages` (which require auth + ToS-compatible
  scraping consent). A live WS pump would either need credentials we
  don't have or would scrape pages we shouldn't.
* PR-2's scope per the architectural plan is *the structured-knowledge
  ingest path*, not a specific live source. Building the
  parser + factory + projection now means *any* future ingest channel
  (webhook receiver, alert relay, manual operator paste, screen-
  scraper bot in a separate process) feeds the same pipeline.

Layered split (mirrors :mod:`ui.feeds.binance_public_ws`):

* :func:`parse_tradingview_idea_payload` — pure parser. Takes one
  JSON-decoded payload + a caller-supplied ``ts_ns``. Returns
  ``None`` for unrecognised / malformed payloads (so a webhook
  receiver can skip them without raising) or a tuple
  ``(TraderModel, observation_kind, meta)`` ready for the trader-
  modeling factory.
* The actual :class:`TraderObservation` construction happens inside
  :mod:`intelligence_engine.trader_modeling.aggregator` (the only
  B29-allowed runtime location) — never here.

INV-15: parser is pure. ``ts_ns`` is caller-supplied; the parser
never reads a system clock. Two replays of the same input produce
byte-identical output.

Canonical payload schema (envelope version 1):

    {
        "version": 1,
        "trader_id": "tv:<opaque-handle>",
        "source_feed": "SRC-TRADER-TRADINGVIEW-001",   # optional
        "observation_kind": "PROFILE_UPDATE" | "SIGNAL_OBSERVED",
        "strategy_signatures": ["sig1", ...],          # optional
        "performance_metrics": {...},                  # optional
        "risk_profile": {...},                         # optional
        "regime_performance": {...},                   # optional
        "behavioral_bias": {...},                      # optional
        "philosophy": {                                # optional
            "belief_system": {...},
            "risk_attitude": "BALANCED",
            "time_horizon": "SWING",
            "conviction_style": "REACTIVE",
            "market_view": {...},
            "decision_biases": {...},
        },
        "meta": {...},                                 # optional, all str
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_PROFILE_UPDATE,
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
    TraderModel,
)

#: SCVS source row this adapter is the producer for. The factory
#: defaults this onto the emitted ``TraderModel.source_feed`` if the
#: payload omits it.
TRADINGVIEW_SOURCE_FEED: str = "SRC-TRADER-TRADINGVIEW-001"

#: Wire-format envelope version. Mirrors
#: :data:`intelligence_engine.trader_modeling.observation.OBSERVATION_EVENT_VERSION`.
TRADINGVIEW_PAYLOAD_VERSION: int = 1

_LEGAL_OBSERVATION_KINDS: frozenset[str] = frozenset(
    {
        TRADER_OBSERVATION_PROFILE_UPDATE,
        TRADER_OBSERVATION_SIGNAL_OBSERVED,
    }
)


def _safe_float_map(raw: Any) -> dict[str, float] | None:
    """Coerce a mapping to ``str -> float``; ``None`` if not coercible."""
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            return None
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            return None
    return out


def _safe_str_map(raw: Any) -> dict[str, str] | None:
    """Coerce a mapping to ``str -> str``; ``None`` if not coercible."""
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return None
        out[k] = v
    return out


def _safe_str_tuple(raw: Any) -> tuple[str, ...] | None:
    """Coerce a sequence to ``tuple[str, ...]``; ``None`` if not coercible."""
    if isinstance(raw, str) or not hasattr(raw, "__iter__"):
        return None
    out: list[str] = []
    for v in raw:
        if not isinstance(v, str):
            return None
        out.append(v)
    return tuple(out)


def _parse_philosophy(raw: Any) -> PhilosophyProfile | None:
    """Parse the optional ``philosophy`` sub-payload.

    Returns ``None`` if absent. Returns ``None`` and the parser silently
    drops the philosophy layer if the sub-payload is malformed — the
    enclosing :class:`TraderModel` is still emitted (philosophy is
    optional by design; a missing layer just means the offline
    extractor hasn't run yet).
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    trader_id = raw.get("trader_id")
    if not isinstance(trader_id, str) or not trader_id:
        return None
    belief_system = _safe_float_map(raw.get("belief_system", {})) or {}
    market_view = _safe_str_map(raw.get("market_view", {})) or {}
    decision_biases = _safe_float_map(raw.get("decision_biases", {})) or {}
    try:
        risk_attitude = RiskAttitude(raw.get("risk_attitude", "UNKNOWN"))
        time_horizon = TimeHorizon(raw.get("time_horizon", "UNKNOWN"))
        conviction_style = ConvictionStyle(
            raw.get("conviction_style", "UNKNOWN")
        )
    except ValueError:
        return None
    return PhilosophyProfile(
        trader_id=trader_id,
        belief_system=belief_system,
        risk_attitude=risk_attitude,
        time_horizon=time_horizon,
        conviction_style=conviction_style,
        market_view=market_view,
        decision_biases=decision_biases,
    )


def parse_tradingview_idea_payload(
    payload: Mapping[str, Any] | Any,
    *,
    ts_ns: int,
    source_feed: str = TRADINGVIEW_SOURCE_FEED,
) -> tuple[TraderModel, str, dict[str, str]] | None:
    """Project one TradingView envelope into the trader-modeling inputs.

    Returns ``None`` (never raises) if ``payload`` is unrecognised.
    Returns ``(model, observation_kind, meta)`` on success — the caller
    feeds these to
    :func:`intelligence_engine.trader_modeling.aggregator.make_trader_observation`
    to obtain a :class:`TraderObservation`.

    The parser does **not** construct a :class:`TraderObservation`
    itself — B29 lint forbids construction outside the
    ``intelligence_engine.trader_modeling`` package. The split keeps
    adapters dependency-free of the bus-transport authority.

    Args:
        payload: JSON-decoded envelope (see module docstring for
            schema).
        ts_ns: Caller-supplied monotonic timestamp from the
            TimeAuthority. Not embedded in the returned tuple — the
            aggregator stamps it onto the resulting observation.
        source_feed: Default SCVS source-row id stamped onto
            ``TraderModel.source_feed`` if the payload omits it.

    Returns:
        ``None`` for malformed / non-trader payloads, or a tuple
        ``(model, observation_kind, meta)`` for the aggregator.
    """

    if ts_ns < 0:
        return None
    if not isinstance(payload, Mapping):
        return None

    raw_version = payload.get("version", TRADINGVIEW_PAYLOAD_VERSION)
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        return None
    if version != TRADINGVIEW_PAYLOAD_VERSION:
        return None

    trader_id = payload.get("trader_id")
    if not isinstance(trader_id, str) or not trader_id:
        return None

    observation_kind = payload.get(
        "observation_kind", TRADER_OBSERVATION_PROFILE_UPDATE
    )
    if observation_kind not in _LEGAL_OBSERVATION_KINDS:
        return None

    feed = payload.get("source_feed", source_feed)
    if not isinstance(feed, str) or not feed:
        return None

    signatures = _safe_str_tuple(payload.get("strategy_signatures", ()))
    if signatures is None:
        return None

    performance_metrics = _safe_float_map(
        payload.get("performance_metrics", {})
    )
    risk_profile = _safe_float_map(payload.get("risk_profile", {}))
    regime_performance = _safe_float_map(
        payload.get("regime_performance", {})
    )
    behavioral_bias = _safe_float_map(payload.get("behavioral_bias", {}))
    if (
        performance_metrics is None
        or risk_profile is None
        or regime_performance is None
        or behavioral_bias is None
    ):
        return None

    meta = _safe_str_map(payload.get("meta", {}))
    if meta is None:
        return None

    philosophy = _parse_philosophy(payload.get("philosophy"))
    # Philosophy must agree with the model's trader_id when supplied —
    # otherwise we'd let an adapter ship a TraderModel for trader A
    # carrying a PhilosophyProfile for trader B.
    if philosophy is not None and philosophy.trader_id != trader_id:
        return None

    model = TraderModel(
        trader_id=trader_id,
        source_feed=feed,
        strategy_signatures=signatures,
        performance_metrics=performance_metrics,
        risk_profile=risk_profile,
        regime_performance=regime_performance,
        behavioral_bias=behavioral_bias,
        philosophy=philosophy,
        meta={},  # model.meta is reserved for offline-extractor tags;
                  # adapter-side meta lives on the TraderObservation.
    )
    return model, observation_kind, meta


__all__ = [
    "TRADINGVIEW_PAYLOAD_VERSION",
    "TRADINGVIEW_SOURCE_FEED",
    "parse_tradingview_idea_payload",
]
