"""HAZ-NEWS-SHOCK — news-shock hazard sensor (Wave-News-Fusion PR-2).

Closes the second half of the news→signal gap reviewer #3 (audit v3,
"three things that need honest scrutiny", item 2) called out: with the
projection module landed (Wave-News-Fusion PR-1, PR #118), a single
catalytic headline can push the meta-controller into an entry just as
volatility blows up — exactly when position sizes should be cut, not
expanded. Reviewer #2 (audit v2, tactical recommendation 4) framed the
fix as an *EventGuard*: classify a :class:`NewsItem` against a
deterministic shock rule and route a ``Hazard.NEWS_SHOCK`` to
Governance, which then throttles position sizes for the duration of the
event window via the regular hazard-throttle layer (INV-64).

This module ships the **classifier**. Wiring (the tap that feeds
NewsItems from the news pump into the sensor and the resulting
:class:`HazardEvent`'s onto the governance bus) lands in a follow-up
PR; sensors in this package are stateless / clock-free / IO-free and
deterministic per the package contract above.

Inputs the classifier acts on:

* **Source-side urgency hint.** ``NewsItem.meta["urgency"]`` is
  consulted first — RSS / HTTP adapters that surface a publisher's own
  ``<urgent/>`` or ``priority="high"`` flag get authoritative
  treatment. Recognised values: ``"breaking"``, ``"urgent"``,
  ``"alert"``, ``"flash"``. Case-insensitive.
* **Shock-keyword score.** A short, frozen, ASCII-only token table is
  scored against title + summary. Keywords are intentionally
  *narrower* than the BUY/SELL tables in
  ``intelligence_engine.news.news_projection`` — these are *systemic*
  shock vocabulary (``crash``, ``halt``, ``hack``, ``ban``,
  ``default``, ``sanction``, ``emergency``…) rather than the
  directional sentiment vocabulary (``rally``, ``surge``, ``plunge``…)
  used to size a position. Multilingual / sentence-encoder coverage is
  a Wave-05 concern.

Severity envelope (frozen):

* Source-side ``urgency`` flag set → :class:`HazardSeverity.HIGH`.
* shock-keyword hits ≥ ``high_score_threshold`` (default 3) →
  :class:`HazardSeverity.HIGH`.
* shock-keyword hits ≥ ``medium_score_threshold`` (default 1) →
  :class:`HazardSeverity.MEDIUM`.
* otherwise → no hazard (``()``).

The sensor never returns ``LOW`` or ``CRITICAL``: ``LOW`` would be
indistinguishable from sentiment noise in the throttle layer; ``CRITICAL``
is reserved for system-side breakers (kill-switch level).

A bumped keyword table or threshold MUST bump
:data:`NEWS_SHOCK_VERSION` so replays can disambiguate windows.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from core.contracts.events import HazardEvent, HazardSeverity
from core.contracts.news import NewsItem

NEWS_SHOCK_VERSION: Final[str] = "v1"

_SHOCK_HAZARD_CODE: Final[str] = "HAZ-NEWS-SHOCK"
_SOURCE: Final[str] = "system_engine.hazard_sensors.news_shock"

# Source-side urgency flags publishers expose. Stored lower-case; we
# fold the input on lookup. Values are *not* parsed for nuance — the
# flag is treated as authoritative because the publisher already made
# the editorial call.
_URGENCY_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "breaking",
        "urgent",
        "alert",
        "flash",
    }
)

# Shock vocabulary — systemic / risk-off, intentionally narrower than
# the directional sentiment tables in news_projection.py. ASCII only;
# multilingual coverage is Wave-05.
_SHOCK_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "crash",
        "crashed",
        "crashes",
        "halt",
        "halted",
        "suspended",
        "suspension",
        "hack",
        "hacked",
        "exploit",
        "exploited",
        "exploits",
        "ban",
        "banned",
        "default",
        "defaulted",
        "bankruptcy",
        "bankrupt",
        "insolvent",
        "insolvency",
        "lawsuit",
        "indictment",
        "subpoena",
        "sanction",
        "sanctions",
        "sanctioned",
        "war",
        "attack",
        "attacked",
        "emergency",
        "panic",
        "crisis",
        "shutdown",
        "freeze",
        "frozen",
        "delist",
        "delisted",
        "rugpull",
        "exploiter",
    }
)

# Same pattern as ``intelligence_engine.news.news_projection._TOKEN_PATTERN``
# — hyphenated compounds like ``post-crash`` stay as one token, not two,
# so the projector and the sensor agree on what's a "word". Diverging
# here would let a headline trip the shock sensor while staying invisible
# to the projection (or vice-versa), and the fanout's "hazard before
# signal" ordering implicitly assumes the two pipelines see the same
# tokens. ``test_news_shock_tokenizer_parity_with_news_projection``
# guards this.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9\-]*")


def _tokenize(text: str) -> tuple[str, ...]:
    """Deterministic ASCII-word tokenizer (mirrors news_projection)."""
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _normalise_urgency(meta: Mapping[str, str]) -> str | None:
    """Return a recognised urgency flag or ``None``."""
    raw = meta.get("urgency", "")
    if not raw:
        return None
    flag = raw.strip().lower()
    if flag in _URGENCY_FLAGS:
        return flag
    return None


def _shock_score(news: NewsItem) -> int:
    """Number of shock-keyword hits in title + summary."""
    haystack = _tokenize(f"{news.title} {news.summary}")
    return sum(1 for token in haystack if token in _SHOCK_KEYWORDS)


class NewsShockSensor:
    """HAZ-NEWS-SHOCK classifier.

    Stateless across NewsItems; threshold parameters and version are
    captured in :attr:`HazardEvent.meta` so a downstream replay can
    bind a hazard to the rule that produced it.
    """

    name: str = "news_shock"
    code: str = _SHOCK_HAZARD_CODE
    spec_id: str = _SHOCK_HAZARD_CODE
    source: str = _SOURCE

    __slots__ = (
        "_high_score_threshold",
        "_medium_score_threshold",
    )

    def __init__(
        self,
        *,
        high_score_threshold: int = 3,
        medium_score_threshold: int = 1,
    ) -> None:
        if high_score_threshold < medium_score_threshold:
            raise ValueError(
                "high_score_threshold must be >= medium_score_threshold"
            )
        if medium_score_threshold < 1:
            raise ValueError("medium_score_threshold must be >= 1")
        self._high_score_threshold = high_score_threshold
        self._medium_score_threshold = medium_score_threshold

    def on_news(self, news: NewsItem) -> tuple[HazardEvent, ...]:
        urgency = _normalise_urgency(news.meta)
        score = _shock_score(news)

        if urgency is not None:
            severity = HazardSeverity.HIGH
            reason = "urgency_flag"
        elif score >= self._high_score_threshold:
            severity = HazardSeverity.HIGH
            reason = "shock_score_high"
        elif score >= self._medium_score_threshold:
            severity = HazardSeverity.MEDIUM
            reason = "shock_score_medium"
        else:
            return ()

        meta: dict[str, str] = {
            "news_source": news.source,
            "news_guid": news.guid,
            "shock_score": str(score),
            "reason": reason,
            "version": NEWS_SHOCK_VERSION,
        }
        if urgency is not None:
            meta["urgency"] = urgency

        symbol = news.meta.get("symbol", "")
        if symbol:
            meta["symbol"] = symbol

        detail = (
            f"news_shock score={score} reason={reason} "
            f"source={news.source} guid={news.guid}"
        )

        return (
            HazardEvent(
                ts_ns=news.ts_ns,
                code=self.code,
                severity=severity,
                source=self.source,
                detail=detail,
                meta=meta,
                produced_by_engine="system_engine",
            ),
        )


__all__ = [
    "NEWS_SHOCK_VERSION",
    "NewsShockSensor",
]
