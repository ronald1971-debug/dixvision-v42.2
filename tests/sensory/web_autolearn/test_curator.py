"""Unit tests for sensory.web_autolearn.curator."""

from __future__ import annotations

import pytest

from sensory.web_autolearn.contracts import FilteredItem
from sensory.web_autolearn.curator import Curator, CuratorRules


def _item(
    seed_id: str = "s",
    *,
    title: str = "title",
    body: str = "body",
    score: float = 1.0,
) -> FilteredItem:
    return FilteredItem(
        ts_ns=1,
        seed_id=seed_id,
        url="https://x",
        title=title,
        body=body,
        score=score,
        reason="keyword:x",
    )


def test_curator_drops_unknown_seed() -> None:
    rules = CuratorRules.from_mapping(
        {"known": {"topic": "crypto"}}
    )
    curator = Curator(rules=rules)
    out = curator.curate([_item(seed_id="unknown")])
    assert out == ()


def test_curator_drops_below_min_score() -> None:
    rules = CuratorRules.from_mapping(
        {"s": {"topic": "crypto", "min_score": 0.5}}
    )
    curator = Curator(rules=rules)
    out = curator.curate([_item(score=0.3)])
    assert out == ()


def test_curator_passes_at_or_above_min_score() -> None:
    rules = CuratorRules.from_mapping(
        {"s": {"topic": "crypto", "min_score": 0.5}}
    )
    curator = Curator(rules=rules)
    out = curator.curate(
        [_item(score=0.5), _item(score=0.9)]
    )
    assert len(out) == 2


def test_curator_deny_substrings() -> None:
    rules = CuratorRules.from_mapping(
        {
            "s": {
                "topic": "crypto",
                "deny": ["sponsored"],
            }
        }
    )
    curator = Curator(rules=rules)
    out = curator.curate(
        [_item(title="Sponsored Content"), _item(title="real news")]
    )
    assert len(out) == 1
    assert out[0].title == "real news"


def test_curator_allow_substrings_required() -> None:
    rules = CuratorRules.from_mapping(
        {
            "s": {
                "topic": "crypto",
                "allow": ["btc", "eth"],
            }
        }
    )
    curator = Curator(rules=rules)
    out = curator.curate(
        [_item(title="BTC rallies"), _item(title="off-topic")]
    )
    assert len(out) == 1
    assert out[0].title == "BTC rallies"


def test_curator_carries_topic_and_tags() -> None:
    rules = CuratorRules.from_mapping(
        {
            "s": {
                "topic": "crypto",
                "tags": ["spot", "majors", "spot"],
            }
        }
    )
    curator = Curator(rules=rules)
    out = curator.curate([_item()])
    assert out[0].seed_topic == "crypto"
    # tags are deduplicated and sorted
    assert out[0].curator_tags == ("majors", "spot")


def test_curator_rules_from_mapping_defaults() -> None:
    rules = CuratorRules.from_mapping(
        {"s": {"topic": "crypto"}}
    )
    rule = rules.rules["s"]
    assert rule.topic == "crypto"
    assert rule.min_score == 0.0
    assert rule.allow_substrings == ()
    assert rule.deny_substrings == ()
    assert rule.tags == ()


def test_curator_rules_rejects_non_mapping_body() -> None:
    with pytest.raises(ValueError, match="mapping"):
        CuratorRules.from_mapping({"s": "not-a-mapping"})  # type: ignore[arg-type]


def test_curator_rules_rejects_non_string_topic() -> None:
    with pytest.raises(ValueError, match="topic"):
        CuratorRules.from_mapping(
            {"s": {"topic": 123}}  # type: ignore[arg-type]
        )


def test_curator_rules_rejects_non_numeric_min_score() -> None:
    with pytest.raises(ValueError, match="min_score"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "min_score": "0.5"}}  # type: ignore[arg-type]
        )


def test_curator_rules_rejects_bare_string_allow_deny_tags() -> None:
    """YAML ``deny: sponsored`` (bare string, no brackets) iterates per-char.

    Without an explicit string check the curator would silently expand
    ``"sponsored"`` to ``('s','p','o',...)`` and any document containing
    one of those letters would match the deny filter — i.e. almost every
    document. Fail loudly instead.
    """

    with pytest.raises(ValueError, match="allow must be a list"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "allow": "bitcoin"}}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="deny must be a list"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "deny": "sponsored"}}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="tags must be a list"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "tags": "majors"}}  # type: ignore[arg-type]
        )


def test_curator_rules_rejects_bool_min_score() -> None:
    """YAML ``yes``/``true``/``on`` parses as Python ``True`` and ``bool``
    is an ``int`` subclass — reject explicitly so a misspelled YAML line
    cannot silently become ``min_score=1.0``.
    """

    with pytest.raises(ValueError, match="min_score"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "min_score": True}}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="min_score"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "min_score": False}}  # type: ignore[arg-type]
        )


def test_curator_rules_rejects_empty_topic() -> None:
    with pytest.raises(ValueError, match="topic"):
        CuratorRules.from_mapping({"s": {"topic": ""}})


def test_curator_rules_min_score_bounds() -> None:
    with pytest.raises(ValueError, match="min_score"):
        CuratorRules.from_mapping(
            {"s": {"topic": "x", "min_score": 1.5}}
        )


def test_curator_replay_determinism() -> None:
    """Same items + same rules -> same output (INV-15)."""

    rules = CuratorRules.from_mapping(
        {"s": {"topic": "crypto", "min_score": 0.5}}
    )
    curator = Curator(rules=rules)
    items = [_item(score=0.5), _item(score=0.9)]
    a = curator.curate(items)
    b = curator.curate(items)
    assert a == b
