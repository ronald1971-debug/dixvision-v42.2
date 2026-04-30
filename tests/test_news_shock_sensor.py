"""Tests for ``NewsShockSensor`` (Wave-News-Fusion PR-2).

Closes the second half of the news→signal gap reviewer #3 (audit v3,
item 2) called out: with the projection module landed (PR #118), the
sensor classifies a single :class:`NewsItem` against a deterministic
shock rule and emits ``HAZ-NEWS-SHOCK`` :class:`HazardEvent`'s. The
governance hazard-throttle layer (INV-64) downstream then cuts position
sizes for the duration of the event window — the PR-3 wiring.
"""

from __future__ import annotations

from core.contracts.event_provenance import (
    EVENT_PRODUCERS,
    assert_event_provenance,
)
from core.contracts.events import HazardEvent, HazardSeverity
from core.contracts.news import NewsItem
from system_engine.hazard_sensors import (
    NEWS_SHOCK_VERSION,
    NewsShockSensor,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _news(
    *,
    title: str = "",
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
        title=title or "headline",
        url="https://example.test/x",
        summary=summary,
        published_ts_ns=None,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# no-shock cases
# ---------------------------------------------------------------------------


def test_no_keywords_no_urgency_returns_no_hazard() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="Bitcoin treasury company opens new office",
            summary="A measured update with no shock vocabulary.",
        )
    )
    assert out == ()


def test_unrecognised_urgency_value_does_not_trigger() -> None:
    """Source-side urgency must be one of the recognised flags."""
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="market gently moves",
            summary="",
            urgency="medium-priority",  # not in _URGENCY_FLAGS
        )
    )
    assert out == ()


def test_empty_urgency_string_does_not_trigger() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(_news(title="calm", urgency=""))
    assert out == ()


# ---------------------------------------------------------------------------
# urgency-flag cases (HIGH)
# ---------------------------------------------------------------------------


def test_breaking_urgency_emits_high_severity() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(title="Fed announcement", urgency="breaking")
    )
    assert len(out) == 1
    haz = out[0]
    assert haz.severity is HazardSeverity.HIGH
    assert haz.code == "HAZ-NEWS-SHOCK"
    assert haz.meta["reason"] == "urgency_flag"
    assert haz.meta["urgency"] == "breaking"


def test_urgency_is_case_and_whitespace_insensitive() -> None:
    sensor = NewsShockSensor()
    for raw in (" Breaking ", "URGENT", "alert"):
        out = sensor.on_news(_news(title="x", urgency=raw))
        assert len(out) == 1
        assert out[0].severity is HazardSeverity.HIGH


def test_flash_urgency_emits_high_severity() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(_news(title="x", urgency="flash"))
    assert out[0].severity is HazardSeverity.HIGH


# ---------------------------------------------------------------------------
# shock-keyword cases
# ---------------------------------------------------------------------------


def test_single_shock_keyword_emits_medium_severity() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(title="exchange halt under review", summary="")
    )
    assert len(out) == 1
    assert out[0].severity is HazardSeverity.MEDIUM
    assert out[0].meta["reason"] == "shock_score_medium"
    assert out[0].meta["shock_score"] == "1"


def test_three_shock_keywords_emit_high_severity() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="Exchange hacked, exploit found",
            summary="Trading suspended on the venue.",
        )
    )
    assert len(out) == 1
    assert out[0].severity is HazardSeverity.HIGH
    assert out[0].meta["reason"] == "shock_score_high"
    assert int(out[0].meta["shock_score"]) >= 3


def test_shock_score_counts_token_occurrences_not_unique_keywords() -> None:
    """Repeated shock tokens still escalate severity (each occurrence
    is a separate hit)."""
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="crash crash crash",
            summary="",
        )
    )
    assert int(out[0].meta["shock_score"]) == 3
    assert out[0].severity is HazardSeverity.HIGH


def test_keyword_match_is_token_boundary_not_substring() -> None:
    """``crashing`` is *not* in the table; only inflected forms we
    chose. The tokenizer splits on word boundaries and lookup is exact
    against the frozen table."""
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="Crashing course on macro",
            summary="Nothing of substance.",
        )
    )
    assert out == ()


def test_news_shock_tokenizer_parity_with_news_projection() -> None:
    """The shock sensor and the projection MUST tokenise the same way:
    diverging would let a headline trip the shock sensor while staying
    invisible to the projection (or vice-versa) — the fanout's
    'hazard before signal' ordering implicitly assumes the two
    pipelines see the same tokens (Devin Review BUG_pr-review-job-
    2f05103288af44aea37efd6c970b2339_0001 on PR #120)."""
    from intelligence_engine.news.news_projection import (
        _TOKEN_PATTERN as PROJECTION_TOKEN_PATTERN,
    )
    from system_engine.hazard_sensors.news_shock import (
        _TOKEN_PATTERN as SHOCK_TOKEN_PATTERN,
    )

    assert SHOCK_TOKEN_PATTERN.pattern == PROJECTION_TOKEN_PATTERN.pattern

    # Concrete behavioural anchor — hyphenated compounds stay as one
    # token in both modules.
    samples = (
        "post-crash analysis",
        "self-hack disclosure",
        "exchange-hack report",
        "anti-freeze rally",
    )
    for text in samples:
        assert SHOCK_TOKEN_PATTERN.findall(text.lower()) == (
            PROJECTION_TOKEN_PATTERN.findall(text.lower())
        ), text


def test_hyphenated_compound_does_not_split_into_shock_keyword() -> None:
    """``post-crash`` is one token (not ``"crash"``) — so a measured
    retrospective headline doesn't trip the sensor. Mirrors the
    projection module so the sensor and projector see the same
    'words' on hyphenated compounds."""
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="post-crash analysis published today",
            summary="A measured retrospective with no shock vocabulary.",
        )
    )
    assert out == ()


# ---------------------------------------------------------------------------
# urgency dominates score (HIGH stays HIGH)
# ---------------------------------------------------------------------------


def test_urgency_dominates_zero_keyword_score() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(title="calm headline", urgency="breaking")
    )
    assert out[0].severity is HazardSeverity.HIGH
    assert int(out[0].meta["shock_score"]) == 0


# ---------------------------------------------------------------------------
# meta + provenance contract
# ---------------------------------------------------------------------------


def test_hazard_carries_news_provenance_and_version() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(
        _news(
            title="exchange hack",
            summary="",
            symbol="BTC-USD",
            source="COINDESK",
            guid="g-42",
        )
    )
    haz = out[0]
    assert haz.source == "system_engine.hazard_sensors.news_shock"
    assert haz.produced_by_engine == "system_engine"
    assert haz.meta["news_source"] == "COINDESK"
    assert haz.meta["news_guid"] == "g-42"
    assert haz.meta["symbol"] == "BTC-USD"
    assert haz.meta["version"] == NEWS_SHOCK_VERSION


def test_symbol_is_omitted_when_meta_lacks_it() -> None:
    sensor = NewsShockSensor()
    out = sensor.on_news(_news(title="exchange hack"))
    assert "symbol" not in out[0].meta


def test_hazard_passes_event_provenance_assertion() -> None:
    """HARDEN-03 / INV-69 — the produced HazardEvent must satisfy the
    repo-wide producer guard."""
    sensor = NewsShockSensor()
    out = sensor.on_news(_news(title="exchange hack"))
    # raises on violation
    assert_event_provenance(out[0])


def test_system_engine_is_in_event_producers_for_hazard_event() -> None:
    """Sanity check that the producer string we stamp is actually
    accepted by EVENT_PRODUCERS — guards a future regression where the
    set is tightened without updating the sensor."""
    assert "system_engine" in EVENT_PRODUCERS[HazardEvent]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_output() -> None:
    sensor_a = NewsShockSensor()
    sensor_b = NewsShockSensor()
    news = _news(title="exchange hack and ban", urgency="breaking")
    a = sensor_a.on_news(news)
    b = sensor_b.on_news(news)
    assert a == b
    # second call on the same instance is also stable (sensor is
    # stateless across NewsItems).
    assert sensor_a.on_news(news) == a


def test_ts_ns_is_carried_through_unchanged() -> None:
    sensor = NewsShockSensor()
    news = _news(
        title="exchange hack",
        ts_ns=1_700_000_000_111_222_333,
    )
    out = sensor.on_news(news)
    assert out[0].ts_ns == 1_700_000_000_111_222_333


# ---------------------------------------------------------------------------
# threshold configurability + validation
# ---------------------------------------------------------------------------


def test_custom_thresholds_take_effect() -> None:
    sensor = NewsShockSensor(
        high_score_threshold=2,
        medium_score_threshold=1,
    )
    out = sensor.on_news(
        _news(title="exchange hack and exploit", summary="")
    )
    # 2 hits → HIGH under custom thresholds, MEDIUM under defaults.
    assert out[0].severity is HazardSeverity.HIGH


def test_high_below_medium_threshold_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        NewsShockSensor(
            high_score_threshold=1,
            medium_score_threshold=2,
        )


def test_medium_below_one_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        NewsShockSensor(medium_score_threshold=0)
