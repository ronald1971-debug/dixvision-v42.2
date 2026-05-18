"""B-07 tests — spaCy NER filter canonical adaptation.

Covers:
1. AST authority pins (no spaCy / engine cross-imports / typed event ctors)
2. Value-object validation (NamedEntity, EnrichedNewsItem)
3. Per-label calculator correctness
4. Overlap resolution + sort stability
5. NewsItem enrichment + batch pipe
6. Determinism (3-run byte equality)
7. Bounds (MAX_TEXT_LENGTH)
"""

from __future__ import annotations

import ast
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.contracts.news import NewsItem  # noqa: E402
from intelligence_engine.news.ner_filter import (  # noqa: E402
    MAX_TEXT_LENGTH,
    NEW_PIP_DEPENDENCIES,
    EntityLabel,
    NamedEntity,
    NERFilterError,
    enrich_news_item,
    entity_summary,
    extract_entities,
    extract_entities_batch,
)

SOURCE_PATH = (REPO_ROOT / "intelligence_engine" / "news" / "ner_filter.py").resolve()
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")
SOURCE_TREE = ast.parse(SOURCE_TEXT)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            out.add(node.module.split(".")[0])
    return out


def test_no_banned_top_level_imports() -> None:
    banned = {
        "spacy",
        "thinc",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "torch",
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "websockets",
        "langsmith",
        "requests",
        "httpx",
    }
    found = _imported_modules(SOURCE_TREE)
    assert not (banned & found), f"banned imports: {banned & found}"


def test_no_typed_event_ctors() -> None:
    """B27 / B28 / INV-71 authority symmetry: no typed bus events."""

    forbidden_names = {
        "PatchProposal",
        "SignalEvent",
        "GovernanceDecision",
        "SystemEvent",
        "ExecutionIntent",
        "FillEvent",
    }
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in forbidden_names:
                pytest.fail(f"forbidden ctor: {target.id}")
            if isinstance(target, ast.Attribute) and target.attr in forbidden_names:
                pytest.fail(f"forbidden ctor: {target.attr}")


def test_no_engine_cross_imports() -> None:
    """B1 engine isolation."""

    forbidden = {
        "governance_engine",
        "system_engine",
        "execution_engine",
        "evolution_engine",
    }
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top not in forbidden, f"forbidden engine import: {node.module}"


def test_adapted_from_header_present() -> None:
    assert "# ADAPTED FROM: spaCy explosion/spaCy" in SOURCE_TEXT


def test_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_no_top_level_io_side_effects() -> None:
    for node in SOURCE_TREE.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
            ),
        ):
            continue
        pytest.fail(f"unexpected top-level node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# NamedEntity validation
# ---------------------------------------------------------------------------


def test_named_entity_rejects_negative_start() -> None:
    with pytest.raises(ValueError):
        NamedEntity(start=-1, end=3, label=EntityLabel.ORG, text="abc")


def test_named_entity_rejects_end_le_start() -> None:
    with pytest.raises(ValueError):
        NamedEntity(start=3, end=3, label=EntityLabel.ORG, text="abc")
    with pytest.raises(ValueError):
        NamedEntity(start=3, end=2, label=EntityLabel.ORG, text="abc")


def test_named_entity_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        NamedEntity(start=0, end=3, label=EntityLabel.ORG, text="")


def test_named_entity_frozen() -> None:
    ent = NamedEntity(start=0, end=5, label=EntityLabel.ORG, text="Apple")
    with pytest.raises((AttributeError, FrozenInstanceError)):
        ent.text = "Tesla"  # type: ignore[misc]


def test_named_entity_sorts() -> None:
    a = NamedEntity(start=10, end=12, label=EntityLabel.GPE, text="US")
    b = NamedEntity(start=0, end=5, label=EntityLabel.ORG, text="Apple")
    assert sorted([a, b]) == [b, a]


# ---------------------------------------------------------------------------
# EntityLabel coverage
# ---------------------------------------------------------------------------


def test_entity_label_canonical_set() -> None:
    labels = {label.value for label in EntityLabel}
    expected = {"ORG", "MONEY", "PERCENT", "GPE", "DATE", "PRODUCT", "TICKER"}
    assert labels == expected


def test_entity_label_sorted_in_definition() -> None:
    values = [label.value for label in EntityLabel]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# Per-label extraction correctness
# ---------------------------------------------------------------------------


def _labels(text: str) -> set[EntityLabel]:
    return {e.label for e in extract_entities(text)}


def _texts(text: str, label: EntityLabel) -> list[str]:
    return [e.text for e in extract_entities(text) if e.label is label]


def test_extract_org_apple() -> None:
    assert "Apple" in _texts("Apple announced new chips today.", EntityLabel.ORG)


def test_extract_org_coinbase() -> None:
    assert "Coinbase" in _texts("Coinbase reported record revenue.", EntityLabel.ORG)


def test_extract_gpe_us() -> None:
    assert "US" in _texts("Inflation rises in US markets.", EntityLabel.GPE)


def test_extract_gpe_two_word() -> None:
    assert "Hong Kong" in _texts("Hong Kong regulators approved the listing.", EntityLabel.GPE)


def test_extract_money_dollar() -> None:
    assert "$100" in _texts("Tesla earnings beat $100 estimates.", EntityLabel.MONEY)


def test_extract_money_with_magnitude() -> None:
    assert "$1.5M" in _texts("The fund raised $1.5M in seed.", EntityLabel.MONEY)


def test_extract_money_usd_prefix() -> None:
    assert "USD 250" in _texts("USD 250 paid out.", EntityLabel.MONEY)


def test_extract_percent_basic() -> None:
    assert "5%" in _texts("Yields jumped 5% overnight.", EntityLabel.PERCENT)


def test_extract_percent_negative() -> None:
    assert "-2.3%" in _texts("Stocks fell -2.3% on the day.", EntityLabel.PERCENT)


def test_extract_date_iso() -> None:
    assert "2024-01-15" in _texts("The vote was on 2024-01-15.", EntityLabel.DATE)


def test_extract_date_quarter() -> None:
    assert "Q1 2024" in _texts("Earnings beat in Q1 2024.", EntityLabel.DATE)


def test_extract_date_short_year() -> None:
    assert "2024" in _texts("Outlook for 2024 is bullish.", EntityLabel.DATE)


def test_extract_product_bitcoin() -> None:
    assert "Bitcoin" in _texts("Bitcoin price hit a new high.", EntityLabel.PRODUCT)


def test_extract_product_ethereum() -> None:
    assert "Ethereum" in _texts("Ethereum upgrade complete.", EntityLabel.PRODUCT)


def test_extract_ticker_dollar_btc() -> None:
    assert "$BTC" in _texts("$BTC traded sideways today.", EntityLabel.TICKER)


def test_extract_ticker_aapl() -> None:
    # AAPL is 4 uppercase — matches plain ticker pattern.
    texts = _texts("AAPL closed flat.", EntityLabel.TICKER)
    assert "AAPL" in texts


def test_extract_empty_text_returns_empty() -> None:
    assert extract_entities("") == ()


def test_extract_irrelevant_text_returns_empty() -> None:
    assert extract_entities("the quick brown fox jumps over lazy dog") == ()


# ---------------------------------------------------------------------------
# Multi-entity text
# ---------------------------------------------------------------------------


def test_multi_entity_text_extracts_all() -> None:
    text = "Apple stock rose 5% in 2024-01-15 trading; Bitcoin hit $50000."
    found = _labels(text)
    assert EntityLabel.ORG in found
    assert EntityLabel.PERCENT in found
    assert EntityLabel.DATE in found
    assert EntityLabel.PRODUCT in found
    assert EntityLabel.MONEY in found


def test_entities_sorted_by_start() -> None:
    text = "Apple stock rose 5% in 2024 trading."
    ents = extract_entities(text)
    starts = [e.start for e in ents]
    assert starts == sorted(starts)


def test_overlap_resolution_keeps_longest() -> None:
    # "USA" is a GPE literal; "AAPL" wouldn't overlap but we test that
    # when two patterns hit the same span the longest wins.
    text = "USA economy expanded."
    ents = extract_entities(text)
    # "USA" matches GPE; the ticker pattern would also match it as a
    # 3-letter all-caps token. After overlap resolution we should keep
    # exactly one entity for that span. Both are length 3, so the
    # canonical tie-break (label alphabetic) picks GPE before TICKER.
    spans = [(e.start, e.end) for e in ents]
    assert spans.count((0, 3)) == 1
    only = [e for e in ents if (e.start, e.end) == (0, 3)][0]
    assert only.label is EntityLabel.GPE


# ---------------------------------------------------------------------------
# EnrichedNewsItem
# ---------------------------------------------------------------------------


def _news(title: str, summary: str = "") -> NewsItem:
    return NewsItem(
        ts_ns=1,
        source="UNIT_TEST",
        guid="g-1",
        title=title,
        summary=summary,
    )


def test_enrich_news_item_basic() -> None:
    item = _news("Apple stock rose 5% today.")
    enriched = enrich_news_item(item)
    assert enriched.item is item
    labels = {e.label for e in enriched.entities}
    assert EntityLabel.ORG in labels
    assert EntityLabel.PERCENT in labels


def test_enrich_news_item_empty_title_summary_okay() -> None:
    item = _news("Apple")
    enriched = enrich_news_item(item)
    assert enriched.entities  # at least one (the ORG)


def test_enrich_news_item_entities_sorted() -> None:
    item = _news("Bitcoin rose 5% but $BTC tickers were quiet.")
    enriched = enrich_news_item(item)
    starts = [e.start for e in enriched.entities]
    assert starts == sorted(starts)


def test_enrich_news_item_rejects_non_news() -> None:
    with pytest.raises(NERFilterError):
        enrich_news_item("not a NewsItem")  # type: ignore[arg-type]


def test_enrich_uses_title_and_summary() -> None:
    item = _news("Apple released", "Tesla also rallied in 2024.")
    enriched = enrich_news_item(item)
    org_texts = {e.text for e in enriched.entities if e.label is EntityLabel.ORG}
    date_texts = {e.text for e in enriched.entities if e.label is EntityLabel.DATE}
    assert "Apple" in org_texts
    assert "Tesla" in org_texts
    assert "2024" in date_texts


def test_enriched_entity_texts() -> None:
    item = _news("Apple Apple Tesla")
    enriched = enrich_news_item(item)
    assert enriched.entity_texts == ("Apple", "Tesla")


def test_enriched_entities_by_label() -> None:
    item = _news("Apple Tesla Bitcoin")
    enriched = enrich_news_item(item)
    orgs = enriched.entities_by_label(EntityLabel.ORG)
    products = enriched.entities_by_label(EntityLabel.PRODUCT)
    assert {e.text for e in orgs} == {"Apple", "Tesla"}
    assert {e.text for e in products} == {"Bitcoin"}


def test_enriched_frozen() -> None:
    enriched = enrich_news_item(_news("Apple"))
    with pytest.raises((AttributeError, FrozenInstanceError)):
        enriched.entities = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch pipe
# ---------------------------------------------------------------------------


def test_extract_batch_preserves_order() -> None:
    items = [_news("Apple stock"), _news("Tesla up 5%"), _news("Bitcoin rallied")]
    result = extract_entities_batch(items)
    assert len(result) == 3
    assert result[0].item is items[0]
    assert result[1].item is items[1]
    assert result[2].item is items[2]


def test_extract_batch_empty() -> None:
    assert extract_entities_batch([]) == ()


def test_extract_batch_rejects_non_news() -> None:
    with pytest.raises(NERFilterError):
        extract_entities_batch([_news("Apple"), "bad"])  # type: ignore[list-item]


def test_extract_batch_does_not_swallow_news_item_validation() -> None:
    # NewsItem itself enforces non-empty title — so we can't even
    # construct a bad input. Confirm the contract is upheld upstream.
    with pytest.raises(ValueError):
        _news("")


def test_entity_summary_counts() -> None:
    batch = extract_entities_batch(
        [
            _news("Apple stock rose 5%"),
            _news("Tesla and Coinbase up"),
            _news("Bitcoin hit $50000"),
        ]
    )
    summary = entity_summary(batch)
    assert summary.get(EntityLabel.ORG, 0) == 3  # Apple, Tesla, Coinbase
    assert summary.get(EntityLabel.PRODUCT, 0) == 1  # Bitcoin
    assert summary.get(EntityLabel.PERCENT, 0) == 1  # 5%
    assert summary.get(EntityLabel.MONEY, 0) == 1  # $50000


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_extract_3_run_byte_identical() -> None:
    text = "Apple rose 5% in 2024-01-15 trading; Bitcoin hit $50000 in the US."
    runs = [extract_entities(text) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_enrich_3_run_byte_identical() -> None:
    item = _news("Apple stock rose 5% today.", "Tesla in 2024 also up.")
    runs = [enrich_news_item(item) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_batch_3_run_byte_identical() -> None:
    items = [
        _news("Apple Tesla Bitcoin"),
        _news("Coinbase rose 7%"),
        _news("$BTC up in 2024"),
    ]
    runs = [extract_entities_batch(items) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_sensitive_to_text_change() -> None:
    a = extract_entities("Apple rose 5%")
    b = extract_entities("Apple rose 6%")
    assert a != b


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_extract_rejects_oversized_text() -> None:
    too_long = "x" * (MAX_TEXT_LENGTH + 1)
    with pytest.raises(NERFilterError):
        extract_entities(too_long)


def test_extract_rejects_non_string() -> None:
    with pytest.raises(NERFilterError):
        extract_entities(123)  # type: ignore[arg-type]


def test_enrich_swallows_oversized_summary() -> None:
    huge = "x" * (MAX_TEXT_LENGTH + 1)
    item = _news("Apple", summary=huge)
    enriched = enrich_news_item(item)
    # extract_entities raises NERFilterError -> swallowed -> empty.
    assert enriched.entities == ()


# ---------------------------------------------------------------------------
# Rule-table invariants
# ---------------------------------------------------------------------------


def test_overlap_resolution_handles_no_candidates() -> None:
    """Defensive: empty input must not blow up overlap resolver."""

    assert extract_entities("") == ()


def test_max_text_length_constant() -> None:
    assert MAX_TEXT_LENGTH == 32_768
