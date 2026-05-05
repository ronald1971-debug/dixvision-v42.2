"""Tests for **B30** — Unify-Intelligence-into-BeliefState.

Reviewer #3 (audit v3, item 2) observed that the news pipeline ends at
ingestion: nothing folds external context into
:class:`core.coherence.belief_state.BeliefState` before signals reach
the meta-controller. B30 is the structural guardrail that prevents
any *future* intelligence-side signal source from re-introducing the
same gap. These tests pin the rule's contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.authority_lint import (
    B30_ALLOWED_LEAF_PRODUCERS,
    _check_b30,
    _module_imports_belief_state,
    lint_repo,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(
    source: str,
    importer: str,
    *,
    relative_path: str = "intelligence_engine/_b30_fixture.py",
) -> list:
    file = REPO_ROOT / relative_path
    tree = ast.parse(source)
    return _check_b30(importer, file, REPO_ROOT, tree)


# ---------------------------------------------------------------------------
# Block: non-leaf intelligence module emits SignalEvent without BeliefState
# ---------------------------------------------------------------------------


def test_b30_blocks_news_projection_without_belief_state() -> None:
    src = (
        "from core.contracts.events import SignalEvent, Side\n"
        "def project(news):\n"
        "    return SignalEvent(ts_ns=1, symbol='BTC-USD', side=Side.BUY,\n"
        "                       confidence=0.5,\n"
        "                       produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.news.news_projection")
    assert [v.rule for v in violations] == ["B30"]
    assert "BeliefState" in violations[0].detail


def test_b30_blocks_macro_projection_without_belief_state() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='BTC-USD', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.macro.fred_projection")
    assert any(v.rule == "B30" for v in violations)


def test_b30_blocks_sentiment_projection_without_belief_state() -> None:
    src = (
        "from core.contracts import events\n"
        "events.SignalEvent(ts_ns=1, symbol='ETH-USD', side='SELL',\n"
        "                   confidence=0.4,\n"
        "                   produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.sentiment.score_projection")
    assert any(v.rule == "B30" for v in violations)


def test_b30_emits_one_violation_per_construction_site() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='A', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
        "SignalEvent(ts_ns=2, symbol='B', side='SELL',\n"
        "            confidence=0.6,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.news.news_projection")
    assert [v.rule for v in violations] == ["B30", "B30"]


# ---------------------------------------------------------------------------
# Allow: BeliefState consumer constructs SignalEvent
# ---------------------------------------------------------------------------


def test_b30_allows_news_projection_when_it_imports_belief_state() -> None:
    src = (
        "from core.contracts.events import SignalEvent, Side\n"
        "from core.coherence.belief_state import BeliefState\n"
        "def project(news, bs: BeliefState):\n"
        "    return SignalEvent(ts_ns=1, symbol='BTC-USD', side=Side.BUY,\n"
        "                       confidence=0.5,\n"
        "                       produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.news.news_projection")
    assert violations == []


def test_b30_allows_via_derive_belief_state_import() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "from core.coherence.belief_state import derive_belief_state\n"
        "_ = derive_belief_state\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.macro.fred_projection")
    assert violations == []


def test_b30_allows_via_full_module_import() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "import core.coherence.belief_state\n"
        "_ = core.coherence.belief_state\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.macro.fred_projection")
    assert violations == []


def test_b30_allows_via_from_core_coherence_import_belief_state() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "from core.coherence import belief_state\n"
        "_ = belief_state\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_engine.news.news_projection")
    assert violations == []


# ---------------------------------------------------------------------------
# Allow: leaf-producer allowlist
# ---------------------------------------------------------------------------


def test_b30_allows_each_leaf_producer_without_belief_state_import() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    for importer in B30_ALLOWED_LEAF_PRODUCERS:
        violations = _run(src, importer)
        assert violations == [], (
            f"leaf producer {importer!r} should be allowed, got {violations}"
        )


def test_b30_leaf_producer_set_matches_expected_canonical_set() -> None:
    """Pin the allowlist's *contents* so accidental drift is caught.

    Adding to this set is allowed *only* via a deliberate update to
    both the lint rule and this test (and a documented rationale in
    the PR description).
    """
    assert B30_ALLOWED_LEAF_PRODUCERS == frozenset(
        {
            "intelligence_engine.engine",
            "intelligence_engine.signal_pipeline",
            "intelligence_engine.strategy_runtime.conflict_resolver",
            "intelligence_engine.plugins.microstructure.microstructure_v1",
            "intelligence_engine.plugins.order_book_pressure.v1",
            "intelligence_engine.cognitive.approval_edge",
        }
    )


# ---------------------------------------------------------------------------
# Skip: out-of-scope modules
# ---------------------------------------------------------------------------


def test_b30_does_not_fire_on_governance_engine() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='governance_engine')\n"
    )
    # Governance can't legally construct SignalEvent at all (B22 blocks
    # it), but B30 should not be the rule that fires.
    violations = _run(src, "governance_engine.policy_engine")
    assert violations == []


def test_b30_does_not_fire_on_execution_engine() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='execution_engine')\n"
    )
    violations = _run(src, "execution_engine.engine")
    assert violations == []


def test_b30_does_not_match_prefix_lookalike_module() -> None:
    """Scope guard uses ``_starts_with_any`` (segment-aware), not raw
    ``str.startswith``. A hypothetical ``intelligence_enginefoo`` module
    must not be swept into the rule."""
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "intelligence_enginefoo.news")
    assert violations == []


def test_b30_does_not_fire_on_core_contracts() -> None:
    src = (
        "from core.contracts.events import SignalEvent\n"
        "SignalEvent(ts_ns=1, symbol='X', side='BUY',\n"
        "            confidence=0.5,\n"
        "            produced_by_engine='intelligence_engine')\n"
    )
    violations = _run(src, "core.contracts.events")
    assert violations == []


def test_b30_does_not_fire_when_no_signal_event_constructed() -> None:
    """Touching SignalEvent as a type annotation must not trip the rule."""
    src = (
        "from core.contracts.events import SignalEvent\n"
        "def consume(signal: SignalEvent) -> None:\n"
        "    pass\n"
    )
    violations = _run(src, "intelligence_engine.news.news_projection")
    assert violations == []


# ---------------------------------------------------------------------------
# Belief-state import detector
# ---------------------------------------------------------------------------


def test_belief_state_detector_handles_import_forms() -> None:
    cases = [
        ("from core.coherence.belief_state import BeliefState", True),
        ("from core.coherence.belief_state import derive_belief_state", True),
        ("from core.coherence.belief_state import BeliefState, Regime", True),
        ("import core.coherence.belief_state", True),
        ("from core.coherence import belief_state", True),
        ("from core.coherence.belief_state import Regime", False),
        ("from core.contracts.events import SignalEvent", False),
        ("from core.coherence import decision_trace", False),
    ]
    for source, expected in cases:
        tree = ast.parse(source)
        assert _module_imports_belief_state(tree) is expected, source


# ---------------------------------------------------------------------------
# Production code is clean — regression guard against drift
# ---------------------------------------------------------------------------


def test_b30_clean_on_repo() -> None:
    violations = [v for v in lint_repo(REPO_ROOT) if v.rule == "B30"]
    assert violations == [], f"unexpected B30 violations: {violations}"
