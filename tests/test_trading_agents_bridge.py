"""Tests for intelligence_engine/agents/trading_agents_bridge.py (C-18).

Covers:
1. AST guards (no forbidden imports, B1 seam, no wall-clock / PRNG,
   no typed event constructors).
2. Module constants (pinned values, directive clauses).
3. Role / config / analysis value objects (frozen, slots, validation).
4. Default extractor (DIRECTION / CONFIDENCE / RATIONALE parsing).
5. Committee runner (clean run, extractor overrides, error paths).
6. INV-15 three-run byte-identical digest.
7. Lazy seam (NotImplementedError, no tradingagents import).
8. Canonical persona hints.
9. LiteLLM role speaker factory.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from intelligence_engine.agents.trading_agents_bridge import (
    ANALYST_ROLES,
    DEFAULT_DIRECTION,
    DEFAULT_RATIONALE,
    DIRECTION_LABELS,
    MAX_ANALYSIS_TEXT_LEN,
    MAX_BRIEF_LEN,
    MAX_PERSONA_LEN,
    NEW_PIP_DEPENDENCIES,
    PORTFOLIO_MANAGER_ROLE,
    RoleAnalysis,
    RoleAnalyst,
    RoleSpeaker,
    TradingAgentsBridgeError,
    TradingAgentsConfig,
    TradingAgentsProposal,
    TradingRole,
    canonical_persona_hint,
    canonical_role_iter,
    default_role_analysis_extractor,
    enable_trading_agents_factory,
    litellm_role_speaker_factory,
    run_trading_agents_committee,
)

MODULE_PATH = Path(
    importlib.util.find_spec(  # type: ignore[union-attr]
        "intelligence_engine.agents.trading_agents_bridge"
    ).origin
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANALYST_PERSONAS = {
    TradingRole.FUNDAMENTALS: "Analyze fundamentals.",
    TradingRole.TECHNICAL: "Analyze technicals.",
    TradingRole.SENTIMENT: "Analyze sentiment.",
    TradingRole.RESEARCHER: "Synthesise analyses.",
}

_PM_PERSONA = "Portfolio manager: weigh all analyses."


def _make_analysts() -> tuple[RoleAnalyst, ...]:
    return tuple(RoleAnalyst(role=role, persona=_ANALYST_PERSONAS[role]) for role in ANALYST_ROLES)


def _make_pm() -> RoleAnalyst:
    return RoleAnalyst(role=TradingRole.PORTFOLIO_MANAGER, persona=_PM_PERSONA)


def _make_config(
    brief: str = "BTC/USDT 1h — what direction?",
) -> TradingAgentsConfig:
    return TradingAgentsConfig(
        brief=brief,
        analysts=_make_analysts(),
        portfolio_manager=_make_pm(),
    )


class DeterministicSpeaker:
    """Fake RoleSpeaker: returns a fixed reply per role."""

    def __init__(self, replies: dict[TradingRole, str] | None = None) -> None:
        if replies is None:
            replies = {
                TradingRole.FUNDAMENTALS: (
                    "DIRECTION: BUY\nCONFIDENCE: 0.8\nRATIONALE: Strong on-chain inflows."
                ),
                TradingRole.TECHNICAL: (
                    "DIRECTION: BUY\nCONFIDENCE: 0.7\nRATIONALE: Breakout above EMA200."
                ),
                TradingRole.SENTIMENT: (
                    "DIRECTION: HOLD\nCONFIDENCE: 0.5\nRATIONALE: Mixed social signals."
                ),
                TradingRole.RESEARCHER: (
                    "DIRECTION: BUY\nCONFIDENCE: 0.6\nRATIONALE: Macro tailwinds align."
                ),
                TradingRole.PORTFOLIO_MANAGER: (
                    "DIRECTION: BUY\nCONFIDENCE: 0.75\n"
                    "RATIONALE: Three of four analysts favour BUY."
                ),
            }
        self._replies = replies

    def speak(
        self,
        *,
        analyst: RoleAnalyst,
        brief: str,
        prior_analyses: tuple[RoleAnalysis, ...],
    ) -> str:
        return self._replies[analyst.role]


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _parse_module() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def test_no_forbidden_top_level_imports() -> None:
    """Module must not top-level import random/time/datetime/secrets/
    os/asyncio/tradingagents/litellm/openai."""

    forbidden = frozenset(
        {
            "random",
            "time",
            "datetime",
            "secrets",
            "os",
            "asyncio",
            "tradingagents",
            "litellm",
            "openai",
            "subprocess",
            "socket",
        }
    )
    tree = _parse_module()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module.split(".")[0] not in forbidden, node.module


def test_only_core_contracts_or_self_imports() -> None:
    """B1 — no execution_engine / governance_engine / system_engine /
    evolution_engine top-level imports."""

    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
    )
    tree = _parse_module()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for pfx in forbidden_prefixes:
                    assert not alias.name.startswith(pfx), alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for pfx in forbidden_prefixes:
                    assert not node.module.startswith(pfx), node.module


def test_no_wall_clock_or_prng() -> None:
    """INV-15 — source must not contain time.time / random / secrets /
    datetime.now calls at any level."""

    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden_call in (
        "time.time()",
        "time.monotonic()",
        "import random",
        "import secrets",
        "datetime.now",
        "datetime.utcnow",
    ):
        assert forbidden_call not in source


def test_no_typed_event_constructors() -> None:
    """Module must never construct SignalEvent / PatchProposal /
    GovernanceDecision / ExecutionIntent."""

    source = MODULE_PATH.read_text(encoding="utf-8")
    for event_type in (
        "SignalEvent(",
        "PatchProposal(",
        "GovernanceDecision(",
        "ExecutionIntent(",
    ):
        assert event_type not in source


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("tradingagents",)


def test_analyst_roles_pinned() -> None:
    assert ANALYST_ROLES == (
        TradingRole.FUNDAMENTALS,
        TradingRole.TECHNICAL,
        TradingRole.SENTIMENT,
        TradingRole.RESEARCHER,
    )


def test_portfolio_manager_role_pinned() -> None:
    assert PORTFOLIO_MANAGER_ROLE is TradingRole.PORTFOLIO_MANAGER


def test_direction_labels_pinned() -> None:
    assert DIRECTION_LABELS == frozenset({"BUY", "SELL", "HOLD"})


def test_canonical_role_iter_covers_enum() -> None:
    roster = list(canonical_role_iter())
    assert set(roster) == set(TradingRole)
    assert len(roster) == len(TradingRole)


# ---------------------------------------------------------------------------
# Value objects — RoleAnalyst
# ---------------------------------------------------------------------------


def test_role_analyst_frozen_and_slotted() -> None:
    a = RoleAnalyst(role=TradingRole.FUNDAMENTALS, persona="Test persona.")
    assert a.role is TradingRole.FUNDAMENTALS
    assert hasattr(a.__class__, "__slots__")
    with pytest.raises(AttributeError):
        a.role = TradingRole.TECHNICAL  # type: ignore[misc]


def test_role_analyst_empty_persona_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        RoleAnalyst(role=TradingRole.FUNDAMENTALS, persona="")


def test_role_analyst_long_persona_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        RoleAnalyst(
            role=TradingRole.FUNDAMENTALS,
            persona="x" * (MAX_PERSONA_LEN + 1),
        )


# ---------------------------------------------------------------------------
# Value objects — TradingAgentsConfig
# ---------------------------------------------------------------------------


def test_config_valid() -> None:
    c = _make_config()
    assert len(c.analysts) == len(ANALYST_ROLES)
    assert c.portfolio_manager.role is TradingRole.PORTFOLIO_MANAGER


def test_config_empty_brief_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        TradingAgentsConfig(
            brief="",
            analysts=_make_analysts(),
            portfolio_manager=_make_pm(),
        )


def test_config_long_brief_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        TradingAgentsConfig(
            brief="x" * (MAX_BRIEF_LEN + 1),
            analysts=_make_analysts(),
            portfolio_manager=_make_pm(),
        )


def test_config_wrong_analyst_count_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        TradingAgentsConfig(
            brief="BTC",
            analysts=(_make_analysts()[0],),
            portfolio_manager=_make_pm(),
        )


def test_config_wrong_analyst_order_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        TradingAgentsConfig(
            brief="BTC",
            analysts=tuple(reversed(_make_analysts())),
            portfolio_manager=_make_pm(),
        )


def test_config_wrong_pm_role_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        TradingAgentsConfig(
            brief="BTC",
            analysts=_make_analysts(),
            portfolio_manager=RoleAnalyst(role=TradingRole.FUNDAMENTALS, persona="PM."),
        )


# ---------------------------------------------------------------------------
# Value objects — RoleAnalysis
# ---------------------------------------------------------------------------


def test_role_analysis_valid() -> None:
    a = RoleAnalysis(
        role=TradingRole.FUNDAMENTALS,
        text="Some analysis.",
        direction="BUY",
        confidence=0.8,
        rationale="On-chain strong.",
    )
    assert a.direction == "BUY"
    assert a.confidence == 0.8


def test_role_analysis_bad_direction_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        RoleAnalysis(
            role=TradingRole.FUNDAMENTALS,
            text=".",
            direction="MOON",
            confidence=0.5,
            rationale=".",
        )


def test_role_analysis_bad_confidence_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        RoleAnalysis(
            role=TradingRole.FUNDAMENTALS,
            text=".",
            direction="BUY",
            confidence=1.5,
            rationale=".",
        )


def test_role_analysis_bool_confidence_rejected() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        RoleAnalysis(
            role=TradingRole.FUNDAMENTALS,
            text=".",
            direction="BUY",
            confidence=True,  # type: ignore[arg-type]
            rationale=".",
        )


# ---------------------------------------------------------------------------
# Default extractor
# ---------------------------------------------------------------------------


def test_extractor_parses_all_fields() -> None:
    text = "DIRECTION: BUY\nCONFIDENCE: 0.9\nRATIONALE: Looks bullish."
    d, c, r = default_role_analysis_extractor(TradingRole.FUNDAMENTALS, text)
    assert d == "BUY"
    assert c == 0.9
    assert r == "Looks bullish."


def test_extractor_defaults_on_missing_fields() -> None:
    text = "No structured output here."
    d, c, r = default_role_analysis_extractor(TradingRole.TECHNICAL, text)
    assert d == DEFAULT_DIRECTION
    assert c == 0.0
    assert r == DEFAULT_RATIONALE


def test_extractor_rejects_unknown_direction() -> None:
    text = "DIRECTION: MOON\nCONFIDENCE: 0.5\nRATIONALE: ?"
    d, c, _r = default_role_analysis_extractor(TradingRole.SENTIMENT, text)
    assert d == DEFAULT_DIRECTION
    assert c == 0.5


def test_extractor_clamps_confidence_range() -> None:
    text = "DIRECTION: BUY\nCONFIDENCE: 2.0\nRATIONALE: Too high."
    _, c, _ = default_role_analysis_extractor(TradingRole.FUNDAMENTALS, text)
    assert c == 0.0


def test_extractor_case_insensitive() -> None:
    text = "direction: sell\nconfidence: 0.6\nrationale: Bearish."
    d, c, r = default_role_analysis_extractor(TradingRole.RESEARCHER, text)
    assert d == "SELL"
    assert c == 0.6
    assert r == "Bearish."


# ---------------------------------------------------------------------------
# Committee runner
# ---------------------------------------------------------------------------


def test_committee_clean_run() -> None:
    config = _make_config()
    speaker = DeterministicSpeaker()
    proposal = run_trading_agents_committee(config=config, speaker=speaker)
    assert isinstance(proposal, TradingAgentsProposal)
    assert proposal.brief == config.brief
    assert len(proposal.analyses) == len(ANALYST_ROLES)
    for analysis, role in zip(proposal.analyses, ANALYST_ROLES, strict=True):
        assert analysis.role is role
    assert proposal.direction == "BUY"
    assert proposal.confidence == 0.75
    assert proposal.proposal_digest


def test_committee_analyses_ordered() -> None:
    config = _make_config()
    speaker = DeterministicSpeaker()
    proposal = run_trading_agents_committee(config=config, speaker=speaker)
    for i, role in enumerate(ANALYST_ROLES):
        assert proposal.analyses[i].role is role


def test_committee_with_custom_extractor() -> None:
    config = _make_config()
    speaker = DeterministicSpeaker()

    def override_extractor(role: TradingRole, text: str) -> tuple[str, float, str]:
        return ("SELL", 0.9, "Always sell.")

    proposal = run_trading_agents_committee(
        config=config, speaker=speaker, extractor=override_extractor
    )
    assert proposal.direction == "SELL"
    assert proposal.confidence == 0.9
    assert proposal.rationale == "Always sell."
    for analysis in proposal.analyses:
        assert analysis.direction == "SELL"


def test_committee_rejects_non_str_speaker_reply() -> None:
    config = _make_config()

    class BadSpeaker:
        def speak(
            self,
            *,
            analyst: RoleAnalyst,
            brief: str,
            prior_analyses: tuple[RoleAnalysis, ...],
        ) -> int:  # type: ignore[override]
            return 42  # type: ignore[return-value]

    with pytest.raises(TradingAgentsBridgeError, match="must be str"):
        run_trading_agents_committee(config=config, speaker=BadSpeaker())  # type: ignore[arg-type]


def test_committee_rejects_oversized_speaker_reply() -> None:
    config = _make_config()

    class BigSpeaker:
        def speak(
            self,
            *,
            analyst: RoleAnalyst,
            brief: str,
            prior_analyses: tuple[RoleAnalysis, ...],
        ) -> str:
            return "x" * (MAX_ANALYSIS_TEXT_LEN + 1)

    with pytest.raises(TradingAgentsBridgeError, match="MAX_ANALYSIS_TEXT_LEN"):
        run_trading_agents_committee(config=config, speaker=BigSpeaker())


# ---------------------------------------------------------------------------
# INV-15 three-run byte-identical digest
# ---------------------------------------------------------------------------


def test_inv15_three_run_byte_identical() -> None:
    config = _make_config()
    speaker = DeterministicSpeaker()
    digests = set()
    for _ in range(3):
        proposal = run_trading_agents_committee(config=config, speaker=speaker)
        digests.add(proposal.proposal_digest)
    assert len(digests) == 1, f"non-deterministic: {digests}"


def test_inv15_change_detection_brief() -> None:
    speaker = DeterministicSpeaker()
    p1 = run_trading_agents_committee(config=_make_config(brief="BTC/USDT 1h"), speaker=speaker)
    p2 = run_trading_agents_committee(config=_make_config(brief="ETH/USDT 1h"), speaker=speaker)
    assert p1.proposal_digest != p2.proposal_digest


def test_inv15_change_detection_role_reply() -> None:
    config = _make_config()
    base = DeterministicSpeaker()
    alt_replies = dict(base._replies)
    alt_replies[TradingRole.FUNDAMENTALS] = "DIRECTION: SELL\nCONFIDENCE: 0.9\nRATIONALE: Changed."
    alt = DeterministicSpeaker(replies=alt_replies)
    p1 = run_trading_agents_committee(config=config, speaker=base)
    p2 = run_trading_agents_committee(config=config, speaker=alt)
    assert p1.proposal_digest != p2.proposal_digest


# ---------------------------------------------------------------------------
# Lazy seam
# ---------------------------------------------------------------------------


def test_enable_trading_agents_factory_raises() -> None:
    with pytest.raises(NotImplementedError):
        enable_trading_agents_factory()


def test_no_tradingagents_import_at_runtime() -> None:
    assert "tradingagents" not in sys.modules


# ---------------------------------------------------------------------------
# Canonical persona hints
# ---------------------------------------------------------------------------


def test_persona_hints_cover_all_roles() -> None:
    for role in TradingRole:
        hint = canonical_persona_hint(role)
        assert isinstance(hint, str)
        assert len(hint) > 0


def test_persona_hint_rejects_non_role() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        canonical_persona_hint("not_a_role")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LiteLLM role speaker factory
# ---------------------------------------------------------------------------


def test_litellm_factory_produces_speaker() -> None:
    def fake_completion(*, messages: list[dict[str, str]]) -> str:
        return "DIRECTION: BUY\nCONFIDENCE: 0.7\nRATIONALE: All good."

    speaker = litellm_role_speaker_factory(completion=fake_completion)
    assert isinstance(speaker, RoleSpeaker)
    config = _make_config()
    proposal = run_trading_agents_committee(config=config, speaker=speaker)
    assert proposal.direction == "BUY"


def test_litellm_factory_rejects_non_callable() -> None:
    with pytest.raises(TradingAgentsBridgeError):
        litellm_role_speaker_factory(completion="not_callable")  # type: ignore[arg-type]


def test_litellm_factory_system_prompt_prefix() -> None:
    calls: list[list[dict[str, str]]] = []

    def capture_completion(*, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "DIRECTION: HOLD\nCONFIDENCE: 0.5\nRATIONALE: Meh."

    speaker = litellm_role_speaker_factory(
        completion=capture_completion,
        system_prompt_prefix="DIX v42.2",
    )
    config = _make_config()
    run_trading_agents_committee(config=config, speaker=speaker)
    assert any(
        "DIX v42.2" in msg["content"] for call in calls for msg in call if msg["role"] == "system"
    )


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(TradingAgentsBridgeError, ValueError)
