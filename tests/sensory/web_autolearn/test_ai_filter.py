"""Unit tests for sensory.web_autolearn.ai_filter."""

from __future__ import annotations

import pytest

from sensory.web_autolearn.ai_filter import (
    AIFilter,
    FilterDecision,
    KeywordAIFilter,
)
from sensory.web_autolearn.contracts import RawDocument


def _doc(
    seed_id: str = "s",
    *,
    title: str = "",
    body: str = "",
    fetched_ok: bool = True,
) -> RawDocument:
    return RawDocument(
        ts_ns=1,
        seed_id=seed_id,
        url="https://x",
        title=title,
        body=body,
        fetched_ok=fetched_ok,
    )


def test_keyword_filter_satisfies_protocol() -> None:
    f = KeywordAIFilter(seed_keywords={"s": ("bitcoin",)})
    assert isinstance(f, AIFilter)


def test_filter_decision_xor_invariant() -> None:
    """A FilterDecision must set exactly one of item / drop_reason."""

    from sensory.web_autolearn.contracts import FilteredItem

    with pytest.raises(ValueError, match="exactly one"):
        FilterDecision()  # neither set

    item = FilteredItem(
        ts_ns=1,
        seed_id="s",
        url="https://x",
        title="t",
        body="b",
        score=0.5,
        reason="keyword:x",
    )
    with pytest.raises(ValueError, match="exactly one"):
        FilterDecision(item=item, drop_reason="dropped")  # both set


def test_keyword_filter_drops_failed_fetch() -> None:
    f = KeywordAIFilter(seed_keywords={"s": ("bitcoin",)})
    decision = f.evaluate(_doc(fetched_ok=False, title="bitcoin"))
    assert not decision.passed
    assert decision.drop_reason == "fetch_failed"


def test_keyword_filter_drops_seed_with_no_keywords() -> None:
    f = KeywordAIFilter(seed_keywords={})
    decision = f.evaluate(_doc(title="bitcoin rallies"))
    assert decision.drop_reason == "no_keywords"


def test_keyword_filter_drops_when_no_match() -> None:
    f = KeywordAIFilter(seed_keywords={"s": ("ethereum",)})
    decision = f.evaluate(_doc(title="bitcoin rallies"))
    assert decision.drop_reason == "no_keywords_matched"


def test_keyword_filter_passes_partial_match() -> None:
    f = KeywordAIFilter(
        seed_keywords={"s": ("bitcoin", "ethereum")},
    )
    decision = f.evaluate(_doc(title="Bitcoin rallies past 70k"))
    assert decision.passed
    item = decision.item
    assert item is not None
    assert item.score == pytest.approx(0.5)
    assert item.reason == "keyword:bitcoin"


def test_keyword_filter_full_match_caps_score_at_1() -> None:
    f = KeywordAIFilter(
        seed_keywords={"s": ("a", "b", "c")},
    )
    decision = f.evaluate(_doc(title="a b c d"))
    assert decision.passed
    assert decision.item is not None
    assert decision.item.score == pytest.approx(1.0)


def test_keyword_filter_min_score_threshold() -> None:
    f = KeywordAIFilter(
        seed_keywords={"s": ("a", "b", "c", "d")},
        min_score=0.5,
    )
    # 1/4 = 0.25 < 0.5 -> drop
    decision = f.evaluate(_doc(title="a only"))
    assert decision.drop_reason == "score_below_min"

    # 2/4 = 0.5 >= 0.5 -> pass
    decision = f.evaluate(_doc(title="a b"))
    assert decision.passed


def test_keyword_filter_min_score_bounds() -> None:
    KeywordAIFilter(min_score=0.0)
    KeywordAIFilter(min_score=1.0)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="min_score"):
            KeywordAIFilter(min_score=bad)


def test_keyword_filter_normalizes_keywords() -> None:
    """Keywords are normalized to lowercase + stripped + dedupe."""

    f = KeywordAIFilter(
        seed_keywords={"s": ("  Bitcoin  ", "BITCOIN", "ethereum")},
    )
    decision = f.evaluate(_doc(title="bitcoin only"))
    assert decision.passed
    item = decision.item
    assert item is not None
    # 1 distinct match out of 2 distinct keywords (bitcoin, ethereum)
    assert item.score == pytest.approx(0.5)


def test_keyword_filter_replay_determinism() -> None:
    """Same input -> same FilterDecision (INV-15)."""

    f = KeywordAIFilter(seed_keywords={"s": ("bitcoin",)})
    doc = _doc(title="bitcoin")
    a = f.evaluate(doc)
    b = f.evaluate(doc)
    assert a == b
