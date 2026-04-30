"""Tests for ``NewsFanout`` (Wave-News-Fusion PR-3).

Closes the loop end-to-end: a single :class:`NewsItem` flows through
the fanout into both the projected :class:`SignalEvent` (via
``project_news``) and zero-or-one :class:`HazardEvent` (via
``NewsShockSensor``).
"""

from __future__ import annotations

from collections.abc import Mapping

from core.coherence.belief_state import BeliefState, Regime
from core.contracts.events import HazardEvent, HazardSeverity, Side, SignalEvent
from core.contracts.news import NewsItem
from system_engine.hazard_sensors import NewsShockSensor
from ui.feeds.news_fanout import NewsFanout

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _news(
    *,
    title: str = "headline",
    summary: str = "",
    urgency: str | None = None,
    symbol: str | None = None,
    ts_ns: int = 1_700_000_000_000_000_000,
    source: str = "COINDESK",
    guid: str = "g-1",
) -> NewsItem:
    meta: dict[str, str] = {}
    if urgency is not None:
        meta["urgency"] = urgency
    if symbol is not None:
        meta["symbol"] = symbol
    return NewsItem(
        ts_ns=ts_ns,
        source=source,
        guid=guid,
        title=title,
        url="https://example.test/x",
        summary=summary,
        published_ts_ns=None,
        meta=meta,
    )


class _Capture:
    def __init__(self) -> None:
        self.signals: list[SignalEvent] = []
        self.hazards: list[HazardEvent] = []
        self.events: list[str] = []

    def signal_sink(self, evt: SignalEvent) -> None:
        self.signals.append(evt)
        self.events.append("signal")

    def hazard_sink(self, evt: HazardEvent) -> None:
        self.hazards.append(evt)
        self.events.append("hazard")


def _make_fanout(
    *,
    sensor: NewsShockSensor | None = None,
    current_belief: BeliefState | None = None,
) -> tuple[NewsFanout, _Capture]:
    cap = _Capture()
    fanout = NewsFanout(
        signal_sink=cap.signal_sink,
        hazard_sink=cap.hazard_sink,
        sensor=sensor if sensor is not None else NewsShockSensor(),
        current_belief=(lambda: current_belief)
        if current_belief is not None
        else None,
    )
    return fanout, cap


# ---------------------------------------------------------------------------
# no-op cases
# ---------------------------------------------------------------------------


def test_neutral_news_with_no_symbol_emits_nothing() -> None:
    fanout, cap = _make_fanout()
    fanout(_news(title="Bitcoin ATM count grows steadily"))
    assert cap.signals == []
    assert cap.hazards == []
    assert cap.events == []


def test_neutral_news_with_symbol_but_no_keywords_emits_nothing() -> None:
    fanout, cap = _make_fanout()
    fanout(_news(title="trading desk update", symbol="BTC-USD"))
    assert cap.signals == []
    assert cap.hazards == []


# ---------------------------------------------------------------------------
# signal-only cases
# ---------------------------------------------------------------------------


def test_bullish_news_with_symbol_emits_signal_only() -> None:
    fanout, cap = _make_fanout()
    fanout(
        _news(
            title="Bitcoin rallies on ETF approval",
            summary="rally surges across crypto",
            symbol="BTC-USD",
        )
    )
    assert len(cap.signals) == 1
    assert cap.signals[0].side is Side.BUY
    assert cap.signals[0].symbol == "BTC-USD"
    assert cap.signals[0].produced_by_engine == "intelligence_engine"
    assert cap.hazards == []


def test_bearish_news_with_keyword_symbol_emits_signal_only() -> None:
    fanout, cap = _make_fanout()
    fanout(
        _news(
            title="ETH falls on regulatory pressure",
            summary="ether tumbles after rejection",
        )
    )
    assert len(cap.signals) == 1
    assert cap.signals[0].side is Side.SELL
    assert cap.signals[0].symbol == "ETH-USD"
    assert cap.hazards == []


# ---------------------------------------------------------------------------
# hazard-only cases
# ---------------------------------------------------------------------------


def test_shock_news_without_symbol_emits_hazard_only() -> None:
    fanout, cap = _make_fanout()
    fanout(
        _news(
            title="exchange halt suspended",
            summary="trading frozen on the venue",
        )
    )
    assert len(cap.hazards) == 1
    assert cap.hazards[0].code == "HAZ-NEWS-SHOCK"
    # no symbol resolved → no signal projected (even though words
    # might score directionally — the projection skips when symbol is
    # unresolved).
    assert cap.signals == []


def test_urgency_flag_alone_emits_hazard() -> None:
    fanout, cap = _make_fanout()
    fanout(_news(title="Fed update", urgency="breaking"))
    assert len(cap.hazards) == 1
    assert cap.hazards[0].severity is HazardSeverity.HIGH
    assert cap.signals == []


# ---------------------------------------------------------------------------
# combined cases — both sinks fire on a single NewsItem
# ---------------------------------------------------------------------------


def test_shock_news_with_symbol_emits_both_signal_and_hazard() -> None:
    fanout, cap = _make_fanout()
    fanout(
        _news(
            title="BTC crash continues",
            summary="bitcoin tumbles after exchange hack",
            symbol="BTC-USD",
        )
    )
    assert len(cap.hazards) == 1
    assert cap.hazards[0].code == "HAZ-NEWS-SHOCK"
    assert len(cap.signals) == 1
    assert cap.signals[0].side is Side.SELL
    assert cap.signals[0].symbol == "BTC-USD"


def test_dispatch_order_hazard_before_signal() -> None:
    """Governance's hazard-throttle layer (INV-64) gets the throttle
    window before the directional signal arrives."""
    fanout, cap = _make_fanout()
    fanout(
        _news(
            title="BTC crash continues",
            summary="bitcoin tumbles after exchange hack",
            symbol="BTC-USD",
        )
    )
    assert cap.events == ["hazard", "signal"]


# ---------------------------------------------------------------------------
# BeliefState wiring
# ---------------------------------------------------------------------------


def _belief(regime: Regime) -> BeliefState:
    return BeliefState(
        ts_ns=1_700_000_000_000_000_000,
        regime=regime,
        regime_confidence=0.9,
        consensus_side=Side.HOLD,
        signal_count=0,
        avg_confidence=0.0,
        symbols=(),
    )


def test_current_belief_callable_is_invoked_with_zero_args() -> None:
    calls: list[None] = []

    def belief_view() -> BeliefState | None:
        calls.append(None)
        return _belief(Regime.RANGE)

    cap = _Capture()
    fanout = NewsFanout(
        signal_sink=cap.signal_sink,
        hazard_sink=cap.hazard_sink,
        sensor=NewsShockSensor(),
        current_belief=belief_view,
    )
    fanout(_news(title="Bitcoin rallies", symbol="BTC-USD"))
    fanout(_news(title="Bitcoin rallies again", symbol="BTC-USD", guid="g-2"))
    assert len(calls) == 2  # invoked once per fanout call


def test_volatility_spike_belief_damps_projected_confidence() -> None:
    """Damping factors live in news_projection; this test confirms the
    fanout actually plumbs the belief through to the projector."""
    high_vol = _belief(Regime.VOL_SPIKE)
    range_belief = _belief(Regime.RANGE)

    fanout_vol, cap_vol = _make_fanout(current_belief=high_vol)
    fanout_range, cap_range = _make_fanout(current_belief=range_belief)

    item = _news(
        title="BTC rallies hard",
        summary="rally surges across the market",
        symbol="BTC-USD",
    )
    fanout_vol(item)
    fanout_range(item)

    assert len(cap_vol.signals) == 1
    assert len(cap_range.signals) == 1
    assert cap_vol.signals[0].confidence < cap_range.signals[0].confidence


def test_no_belief_callable_uses_no_damping() -> None:
    fanout_none, cap_none = _make_fanout()
    fanout_range, cap_range = _make_fanout(
        current_belief=_belief(Regime.RANGE),
    )
    item = _news(
        title="BTC rallies hard",
        summary="rally surges",
        symbol="BTC-USD",
    )
    fanout_none(item)
    fanout_range(item)
    # RANGE has damping factor 1.0; None bypasses entirely. Both
    # paths must produce identical confidences.
    assert cap_none.signals[0].confidence == cap_range.signals[0].confidence


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism_two_runs_produce_identical_outputs() -> None:
    fanout_a, cap_a = _make_fanout()
    fanout_b, cap_b = _make_fanout()
    item = _news(
        title="BTC crash continues",
        summary="bitcoin tumbles after exchange hack",
        symbol="BTC-USD",
    )
    fanout_a(item)
    fanout_b(item)
    assert cap_a.signals == cap_b.signals
    assert cap_a.hazards == cap_b.hazards
    assert cap_a.events == cap_b.events


def test_repeat_call_on_same_fanout_is_stable() -> None:
    fanout, cap = _make_fanout()
    item = _news(
        title="ETH plunges on hack",
        symbol="ETH-USD",
    )
    fanout(item)
    first_signals = list(cap.signals)
    first_hazards = list(cap.hazards)
    fanout(item)
    # second call appends another identical pair
    assert cap.signals[len(first_signals):] == first_signals
    assert cap.hazards[len(first_hazards):] == first_hazards


# ---------------------------------------------------------------------------
# producer split (HARDEN-03)
# ---------------------------------------------------------------------------


def test_signal_keeps_intelligence_engine_producer_stamp() -> None:
    fanout, cap = _make_fanout()
    fanout(_news(title="BTC rallies", symbol="BTC-USD"))
    assert cap.signals[0].produced_by_engine == "intelligence_engine"


def test_hazard_keeps_system_engine_producer_stamp() -> None:
    fanout, cap = _make_fanout()
    fanout(_news(title="exchange hacked exploit"))
    assert cap.hazards[0].produced_by_engine == "system_engine"


# ---------------------------------------------------------------------------
# meta is a Mapping, not a dict — guard the news contract type
# ---------------------------------------------------------------------------


def test_news_meta_mapping_type_does_not_break_fanout() -> None:
    """``NewsItem.meta`` is typed as :class:`Mapping`; ensure both sinks
    accept items where meta is something other than ``dict``."""

    class _RO(Mapping[str, str]):
        def __init__(self, data: dict[str, str]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> str:
            return self._data[key]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    fanout, cap = _make_fanout()
    item = NewsItem(
        ts_ns=1_700_000_000_000_000_000,
        source="COINDESK",
        guid="g-mapping",
        title="exchange hacked exploit BTC crash",
        url="",
        summary="",
        published_ts_ns=None,
        meta=_RO({"symbol": "BTC-USD"}),
    )
    fanout(item)
    assert len(cap.hazards) == 1
    assert len(cap.signals) == 1
    assert cap.signals[0].symbol == "BTC-USD"
