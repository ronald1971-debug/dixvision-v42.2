"""Tests for :mod:`intelligence_engine.news.news_projection`.

Wave-News-Fusion PR-1. The contract is small but reviewer #3 made the
news-to-signal closure structurally important, so the suite explicitly
covers:

* Determinism (replay-safe).
* Side resolution (bullish / bearish / tie / no-keyword).
* Symbol resolution (meta override + keyword fallback + unresolved).
* Confidence shaping (cap, per-hit increment, BeliefState damping).
* Producer + meta stamps that downstream replay code reads.
* B30 lint passes on the new module without an allowlist entry.
"""

from __future__ import annotations

from pathlib import Path

from core.coherence.belief_state import BeliefState, Regime
from core.contracts.event_provenance import (
    EVENT_PRODUCERS,
    assert_event_provenance,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.news import NewsItem
from intelligence_engine.news import NEWS_PROJECTION_VERSION, project_news

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _news(
    *,
    title: str,
    summary: str = "",
    symbol: str | None = None,
    ts_ns: int = 1_700_000_000_000_000_000,
    source: str = "COINDESK",
    guid: str = "guid-1",
) -> NewsItem:
    meta = {"symbol": symbol} if symbol is not None else {}
    return NewsItem(
        ts_ns=ts_ns,
        source=source,
        guid=guid,
        title=title,
        summary=summary,
        meta=meta,
    )


def _belief(regime: Regime) -> BeliefState:
    return BeliefState(
        ts_ns=1_700_000_000_000_000_000,
        regime=regime,
        regime_confidence=0.5,
        consensus_side=Side.HOLD,
        signal_count=0,
        avg_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Side resolution
# ---------------------------------------------------------------------------


def test_bullish_keywords_produce_buy_signal() -> None:
    news = _news(title="Bitcoin rallies to record high on ETF approval")
    sig = project_news(news)
    assert sig is not None
    assert sig.side is Side.BUY
    assert sig.symbol == "BTC-USD"


def test_bearish_keywords_produce_sell_signal() -> None:
    news = _news(title="Ethereum plunges after exchange hack and selloff")
    sig = project_news(news)
    assert sig is not None
    assert sig.side is Side.SELL
    assert sig.symbol == "ETH-USD"


def test_tie_collapses_to_no_signal() -> None:
    # one BUY token (rally), one SELL token (crash), no symbol mention
    # ambiguity is resolved by the score tie collapsing to HOLD.
    news = _news(
        title="Bitcoin rally and crash",
        summary="",
    )
    sig = project_news(news)
    assert sig is None


def test_no_keywords_returns_none() -> None:
    news = _news(title="Bitcoin community holds annual conference")
    sig = project_news(news)
    assert sig is None


def test_summary_tokens_count_for_scoring() -> None:
    news = _news(
        title="Bitcoin update",
        summary="ETF approval drives the rally and a record high",
    )
    sig = project_news(news)
    assert sig is not None
    assert sig.side is Side.BUY


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


def test_meta_symbol_overrides_keyword_resolution() -> None:
    news = _news(
        title="Bitcoin rallies on adoption news",
        symbol="DOGE-USD",
    )
    sig = project_news(news)
    assert sig is not None
    assert sig.symbol == "DOGE-USD"


def test_unknown_symbol_returns_none_even_with_strong_signal() -> None:
    news = _news(title="Asset surges on bullish breakout and ETF approval")
    sig = project_news(news)
    assert sig is None


def test_eth_keyword_resolves_to_eth_usd() -> None:
    news = _news(title="ETH soars after upgrade milestone")
    sig = project_news(news)
    assert sig is not None
    assert sig.symbol == "ETH-USD"


# ---------------------------------------------------------------------------
# Confidence shaping
# ---------------------------------------------------------------------------


def test_single_hit_uses_base_plus_one_increment() -> None:
    news = _news(title="Bitcoin rally")
    sig = project_news(news)
    assert sig is not None
    assert sig.confidence == 0.15 + 0.10  # _BASE + 1 * _PER_HIT


def test_confidence_cap_holds_when_many_hits() -> None:
    news = _news(
        title="Bitcoin rally surge soar jump rise gain bullish breakout",
    )
    sig = project_news(news)
    assert sig is not None
    assert sig.confidence == 0.60  # _CONFIDENCE_CAP


def test_vol_spike_belief_damps_confidence() -> None:
    news = _news(title="Bitcoin rally")
    base_sig = project_news(news)
    damped_sig = project_news(news, current_belief=_belief(Regime.VOL_SPIKE))
    assert base_sig is not None and damped_sig is not None
    assert damped_sig.confidence == base_sig.confidence * 0.50


def test_unknown_belief_damps_confidence() -> None:
    news = _news(title="Bitcoin rally")
    base_sig = project_news(news)
    damped_sig = project_news(news, current_belief=_belief(Regime.UNKNOWN))
    assert base_sig is not None and damped_sig is not None
    assert damped_sig.confidence == base_sig.confidence * 0.75


def test_stable_regime_does_not_damp() -> None:
    news = _news(title="Bitcoin rally")
    base_sig = project_news(news)
    same_sig = project_news(news, current_belief=_belief(Regime.RANGE))
    assert base_sig is not None and same_sig is not None
    assert same_sig.confidence == base_sig.confidence


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_projection_is_deterministic() -> None:
    news = _news(title="Ethereum surges on upgrade adoption")
    a = project_news(news)
    b = project_news(news)
    assert a == b


def test_projection_is_deterministic_with_belief() -> None:
    news = _news(title="Ethereum surges on upgrade adoption")
    belief = _belief(Regime.TREND_UP)
    a = project_news(news, current_belief=belief)
    b = project_news(news, current_belief=belief)
    assert a == b


# ---------------------------------------------------------------------------
# Producer + meta stamps
# ---------------------------------------------------------------------------


def test_producer_is_intelligence_engine_and_passes_provenance() -> None:
    news = _news(title="Bitcoin rally")
    sig = project_news(news)
    assert sig is not None
    assert sig.produced_by_engine == "intelligence_engine"
    assert sig.produced_by_engine in EVENT_PRODUCERS[SignalEvent]
    # Hard runtime contract used by Triad-Lock receivers (HARDEN-03).
    assert_event_provenance(sig)


def test_meta_records_news_provenance_and_version() -> None:
    news = _news(
        title="Bitcoin rally on ETF approval",
        source="COINDESK",
        guid="guid-xyz",
    )
    sig = project_news(news)
    assert sig is not None
    assert sig.meta["news_source"] == "COINDESK"
    assert sig.meta["news_guid"] == "guid-xyz"
    assert sig.meta["projection_version"] == NEWS_PROJECTION_VERSION
    assert int(sig.meta["raw_hits"]) >= 1


def test_plugin_chain_identifies_news_subsystem() -> None:
    news = _news(title="Bitcoin rally")
    sig = project_news(news)
    assert sig is not None
    assert sig.plugin_chain == ("intelligence_engine.news",)


def test_signal_ts_matches_news_ts() -> None:
    news = _news(title="Bitcoin rally", ts_ns=1_700_000_000_111_111_111)
    sig = project_news(news)
    assert sig is not None
    assert sig.ts_ns == 1_700_000_000_111_111_111


# ---------------------------------------------------------------------------
# Belief-state contract (B30 anchor)
# ---------------------------------------------------------------------------


def test_module_imports_belief_state_contract() -> None:
    """News-projection must import the belief-state contract directly so it
    satisfies the unify-intelligence-into-belief-state rule (B30 / reviewer
    #3 audit v3 §2). Verified statically rather than by lint to keep the
    invariant testable independently of the lint module."""
    import ast

    src_path = (
        REPO_ROOT
        / "intelligence_engine"
        / "news"
        / "news_projection.py"
    )
    tree = ast.parse(src_path.read_text())
    imports_belief_state = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "core.coherence.belief_state":
                names = {alias.name for alias in node.names}
                if "BeliefState" in names or "derive_belief_state" in names:
                    imports_belief_state = True
                    break
    assert imports_belief_state, (
        "news_projection.py must import BeliefState (or "
        "derive_belief_state) from core.coherence.belief_state to honour "
        "the unify-intelligence-into-belief-state contract"
    )
