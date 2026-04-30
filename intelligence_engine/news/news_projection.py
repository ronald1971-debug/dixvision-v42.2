"""News → Signal projection (Wave-News-Fusion PR-1).

Reviewer #3 (audit v3, "three things that need honest scrutiny", item 2)
flagged the news-to-signal gap as the most important open item: the
:class:`core.contracts.news.NewsItem` rows ingested by the CoinDesk RSS
adapter never reached the meta-controller because no intelligence-side
code projected them into the unified belief state. This module is the
runtime half of that fix — :func:`project_news` is a pure, deterministic
function that turns one ``NewsItem`` into at most one
:class:`core.contracts.events.SignalEvent`, anchored in the current
:class:`core.coherence.belief_state.BeliefState` snapshot.

Design principles
-----------------

1. **Pure / deterministic.** No clocks, no PRNG, no I/O. The same
   ``NewsItem`` (and the same optional ``BeliefState``) always produces
   the same output. Replay determinism (INV-15) is preserved.
2. **Bounded.** Confidence is capped at :data:`_CONFIDENCE_CAP`; the
   keyword tables are frozen module-level ``frozenset``\\s. A news
   headline cannot push the system harder than microstructure-grade
   signals.
3. **Belief-aware.** When a current ``BeliefState`` is supplied the
   projection damps confidence during ``VOL_SPIKE`` / ``UNKNOWN``
   regimes — the system already knows it is in unstable territory and
   a headline must not push it into a vol-spike trade.
4. **B30-compliant.** This module imports ``BeliefState`` directly so
   it is allowed to construct ``SignalEvent`` outside the leaf-producer
   allowlist (``tools/authority_lint.py`` :data:`B30_ALLOWED_LEAF_PRODUCERS`).

Output contract
---------------

* ``side``: deterministic majority over the title + summary tokens
  against frozen BUY / SELL keyword tables. Ties → no signal.
* ``symbol``: resolved from ``NewsItem.meta["symbol"]`` first, then
  from a small symbol-keyword table (BTC / ETH / SOL). Unresolved →
  no signal.
* ``confidence``: ``min(_CONFIDENCE_CAP, _BASE_CONFIDENCE +
  _PER_HIT_INCREMENT * hits) * _belief_damp(current_belief)`` clamped
  to ``[0.0, 1.0]``.
* ``produced_by_engine``: ``"intelligence_engine"`` — the producer set
  in :data:`core.contracts.event_provenance.EVENT_PRODUCERS` for
  ``SignalEvent``. The cognitive prefix is reserved for the
  operator-approval edge (B26 / INV-72).
* ``meta``: stable structural metadata (news source, news guid,
  projection version, raw hit count). No PII, no secrets.
"""

from __future__ import annotations

import re

from core.coherence.belief_state import BeliefState, Regime
from core.contracts.events import Side, SignalEvent
from core.contracts.news import NewsItem

# ---------------------------------------------------------------------------
# Module version — bumped when the projection function changes shape.
# Recorded in every emitted signal so downstream replays can disambiguate
# windows produced by different derivation versions.
# ---------------------------------------------------------------------------
NEWS_PROJECTION_VERSION = "v1"

# Producer name. Must appear in ``EVENT_PRODUCERS[SignalEvent]`` — see
# :mod:`core.contracts.event_provenance`.
_PRODUCER = "intelligence_engine"

# ---------------------------------------------------------------------------
# Frozen keyword tables.
# ---------------------------------------------------------------------------
#
# Tokens are matched case-insensitively against a deterministic
# ``re.findall`` tokenization of ``title + " " + summary``. Tables are
# intentionally short — News-Fusion PR-1 is the *spine*, not the full
# sentiment model. Wave-05 / Wave-06 add the multilingual + multimodal
# layers. Adding a token here is a knowledge change with replay
# implications — bump :data:`NEWS_PROJECTION_VERSION` when doing so.

_BUY_KEYWORDS: frozenset[str] = frozenset(
    {
        "rally",
        "rallies",
        "rallied",
        "surge",
        "surges",
        "surged",
        "soar",
        "soars",
        "soared",
        "jump",
        "jumps",
        "jumped",
        "rise",
        "rises",
        "rose",
        "gain",
        "gains",
        "gained",
        "bullish",
        "breakout",
        "high",
        "highs",
        "record",
        "approve",
        "approved",
        "approval",
        "etf",
        "adoption",
        "milestone",
        "upgrade",
        "support",
        "rebound",
    }
)

_SELL_KEYWORDS: frozenset[str] = frozenset(
    {
        "crash",
        "crashed",
        "plunge",
        "plunges",
        "plunged",
        "fall",
        "falls",
        "fell",
        "drop",
        "drops",
        "dropped",
        "tumble",
        "tumbles",
        "tumbled",
        "slump",
        "slumps",
        "slumped",
        "bearish",
        "breakdown",
        "low",
        "lows",
        "reject",
        "rejected",
        "rejection",
        "ban",
        "banned",
        "hack",
        "hacked",
        "exploit",
        "selloff",
        "downgrade",
        "panic",
        "fear",
    }
)

# Symbol resolution. Mention-based, deterministic, small. Caller-supplied
# ``NewsItem.meta["symbol"]`` overrides the table.
_SYMBOL_KEYWORDS: dict[str, str] = {
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "ether": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
}

# ---------------------------------------------------------------------------
# Confidence shaping.
# ---------------------------------------------------------------------------

_BASE_CONFIDENCE: float = 0.15
_PER_HIT_INCREMENT: float = 0.10
_CONFIDENCE_CAP: float = 0.60

_VOL_SPIKE_DAMP: float = 0.50
_UNKNOWN_REGIME_DAMP: float = 0.75

# Tokenizer — lower-case ASCII words, hyphens kept inside tokens.
_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-z][a-z0-9\-]*")


def _tokenize(text: str) -> list[str]:
    """Deterministic ASCII-word tokenizer."""
    return _TOKEN_PATTERN.findall(text.lower())


def _resolve_symbol(tokens: list[str], news: NewsItem) -> str:
    """Resolve a canonical symbol from ``meta["symbol"]`` then tokens."""
    meta_symbol = news.meta.get("symbol", "")
    if meta_symbol:
        return meta_symbol
    for tok in tokens:
        if tok in _SYMBOL_KEYWORDS:
            return _SYMBOL_KEYWORDS[tok]
    return ""


def _score(tokens: list[str]) -> tuple[Side, int]:
    """Pure side + raw hit count from token bag.

    Ties between ``BUY`` and ``SELL`` collapse to ``HOLD`` so the
    projection emits no signal — it never proposes a coin-flip trade.
    """
    buy_hits = sum(1 for t in tokens if t in _BUY_KEYWORDS)
    sell_hits = sum(1 for t in tokens if t in _SELL_KEYWORDS)
    if buy_hits == 0 and sell_hits == 0:
        return Side.HOLD, 0
    if buy_hits > sell_hits:
        return Side.BUY, buy_hits
    if sell_hits > buy_hits:
        return Side.SELL, sell_hits
    return Side.HOLD, 0


def _belief_damp(belief: BeliefState | None) -> float:
    """Damping factor sourced from the current :class:`BeliefState`.

    News confidence is reduced when the system regime is unstable so a
    single headline cannot push the meta-controller into a high-vol
    trade. ``None`` → 1.0 (no current snapshot, no damping).
    """
    if belief is None:
        return 1.0
    if belief.regime is Regime.VOL_SPIKE:
        return _VOL_SPIKE_DAMP
    if belief.regime is Regime.UNKNOWN:
        return _UNKNOWN_REGIME_DAMP
    return 1.0


def project_news(
    news: NewsItem,
    *,
    current_belief: BeliefState | None = None,
) -> SignalEvent | None:
    """Project one :class:`NewsItem` into at most one :class:`SignalEvent`.

    Returns ``None`` when the headline yields no actionable side or
    when no symbol can be resolved. Pure / deterministic — same inputs
    in, same output out.
    """

    tokens = _tokenize(news.title + " " + news.summary)
    side, hits = _score(tokens)
    if side is Side.HOLD or hits == 0:
        return None

    symbol = _resolve_symbol(tokens, news)
    if not symbol:
        return None

    raw_conf = min(
        _CONFIDENCE_CAP,
        _BASE_CONFIDENCE + _PER_HIT_INCREMENT * hits,
    )
    confidence = max(0.0, min(1.0, raw_conf * _belief_damp(current_belief)))

    return SignalEvent(
        ts_ns=news.ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        plugin_chain=("intelligence_engine.news",),
        meta={
            "news_source": news.source,
            "news_guid": news.guid,
            "projection_version": NEWS_PROJECTION_VERSION,
            "raw_hits": str(hits),
        },
        produced_by_engine=_PRODUCER,
    )


__all__ = ["NEWS_PROJECTION_VERSION", "project_news"]
