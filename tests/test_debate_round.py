"""Tests for B-03 autogen-agentchat canonical adaptation
(:mod:`intelligence_engine.agents.debate_round`)."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from intelligence_engine.agents.debate_round import (
    DEFAULT_RATIONALE,
    DEFAULT_RECOMMENDATION,
    DEFAULT_TERMINATOR,
    MAX_AGENT_NAME_LEN,
    MAX_AGENT_ROLE_LEN,
    MAX_AGENTS,
    MAX_PERSONA_LEN,
    MAX_RATIONALE_LEN,
    MAX_RECOMMENDATION_LEN,
    MAX_ROUNDS,
    MAX_TOPIC_LEN,
    MAX_TURN_TEXT_LEN,
    MIN_AGENTS,
    MIN_ROUNDS,
    NEW_PIP_DEPENDENCIES,
    RECOMMENDATION_LABELS,
    DebateAgent,
    DebateRoundConfig,
    DebateRoundError,
    DebateRoundProposal,
    DebateTurn,
    Speaker,
    default_recommendation_extractor,
    litellm_speaker_factory,
    run_debate_round,
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "intelligence_engine" / "agents" / "debate_round.py"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent(name: str, role: str = "analyst") -> DebateAgent:
    return DebateAgent(
        name=name,
        role=role,
        persona=f"You are {name}, role={role}. Speak briefly.",
    )


@pytest.fixture
def two_agents() -> tuple[DebateAgent, DebateAgent]:
    return (_make_agent("alice", role="bull"), _make_agent("bob", role="bear"))


@pytest.fixture
def config(two_agents: tuple[DebateAgent, DebateAgent]) -> DebateRoundConfig:
    return DebateRoundConfig(
        topic="Should DIX promote strategy S-001?",
        agents=two_agents,
        max_rounds=2,
    )


@dataclass(frozen=True, slots=True)
class _ScriptedSpeaker:
    """Speaker that returns a deterministic reply per agent name.

    ``replies[name]`` is a list — turn ``i`` for that agent returns
    ``replies[name][min(i, len(replies[name]) - 1)]`` so tests can
    cap the list at one element to mean "always respond this way"."""

    replies: Mapping[str, tuple[str, ...]]

    def speak(
        self,
        *,
        agent: DebateAgent,
        transcript: tuple[DebateTurn, ...],
        topic: str,
        round_idx: int,
    ) -> str:
        bank = self.replies[agent.name]
        # How many times has this agent already spoken?
        already = sum(1 for t in transcript if t.agent_name == agent.name)
        idx = min(already, len(bank) - 1)
        return bank[idx]


def _approve_speaker() -> _ScriptedSpeaker:
    return _ScriptedSpeaker(
        replies={
            "alice": (
                "Markets favourable.\nRECOMMENDATION: APPROVE\n"
                "RATIONALE: alpha decay still positive",
            ),
            "bob": ("Risk acceptable.\nRECOMMENDATION: APPROVE\nRATIONALE: drawdown under cap",),
        }
    )


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def test_no_forbidden_imports() -> None:
    """No autogen / litellm / random / time / clock / numpy / torch imports.

    Top-level imports only — production wires LiteLLM via the
    factory; ``random`` / ``time`` etc. break INV-15."""

    forbidden = {
        "autogen",
        "autogen_agentchat",
        "litellm",
        "openai",
        "anthropic",
        "random",
        "time",
        "datetime",
        "secrets",
        "os",
        "asyncio",
        "numpy",
        "torch",
        "scipy",
        "polars",
        "pandas",
        "langsmith",
        "langgraph",
        "langchain",
        "langchain_core",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            assert root not in forbidden, f"forbidden import-from: {mod}"


def test_no_engine_cross_imports() -> None:
    """B1: no execution/governance/system/evolution engine imports."""

    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            assert root not in forbidden_roots, f"forbidden engine cross-import: {mod}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden_roots, f"forbidden engine cross-import: {alias.name}"


def test_does_not_construct_typed_bus_events() -> None:
    """B27 / B28 / INV-71 authority symmetry.

    intelligence_engine/agents/* MUST NOT construct PatchProposal /
    SignalEvent / GovernanceDecision / ExecutionIntent."""

    forbidden_ctors = {
        "PatchProposal",
        "SignalEvent",
        "GovernanceDecision",
        "ExecutionIntent",
        "OrderIntent",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_ctors, (
                f"forbidden typed-event ctor: {node.func.id}"
            )


def test_adapted_from_header_present() -> None:
    """LGPL/license traceability — ADAPTED FROM cites autogen sources."""

    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: microsoft/autogen" in source
    assert "groupchat.py" in source
    assert "conversable_agent.py" in source


def test_new_pip_dependencies_empty() -> None:
    """Module advertises no new pip dependency — both autogen and
    litellm are out-of-tier."""

    assert NEW_PIP_DEPENDENCIES == ()


def test_module_does_not_construct_governance_decision_strings() -> None:
    """Sanity AST pin: ``GovernanceDecision(`` literal must not appear
    in the source (catches future bare-string ctor patterns)."""

    source = _MODULE_PATH.read_text(encoding="utf-8")
    # Allow doc references like "GovernanceDecision proposal" but not
    # the call expression.
    assert "GovernanceDecision(" not in source


# ---------------------------------------------------------------------------
# Value object validation
# ---------------------------------------------------------------------------


def test_debate_agent_validates_name() -> None:
    with pytest.raises(DebateRoundError):
        DebateAgent(name="", role="r", persona="p")
    with pytest.raises(DebateRoundError):
        DebateAgent(name="a" * (MAX_AGENT_NAME_LEN + 1), role="r", persona="p")
    with pytest.raises(DebateRoundError):
        DebateAgent(name=123, role="r", persona="p")  # type: ignore[arg-type]


def test_debate_agent_validates_role() -> None:
    with pytest.raises(DebateRoundError):
        DebateAgent(name="a", role="", persona="p")
    with pytest.raises(DebateRoundError):
        DebateAgent(name="a", role="r" * (MAX_AGENT_ROLE_LEN + 1), persona="p")


def test_debate_agent_validates_persona() -> None:
    with pytest.raises(DebateRoundError):
        DebateAgent(name="a", role="r", persona="")
    with pytest.raises(DebateRoundError):
        DebateAgent(
            name="a",
            role="r",
            persona="p" * (MAX_PERSONA_LEN + 1),
        )


def test_debate_round_config_validates_topic(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(topic="", agents=two_agents, max_rounds=1)
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(
            topic="t" * (MAX_TOPIC_LEN + 1),
            agents=two_agents,
            max_rounds=1,
        )


def test_debate_round_config_requires_min_agents() -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(topic="t", agents=(_make_agent("a"),), max_rounds=1)


def test_debate_round_config_rejects_too_many_agents() -> None:
    agents = tuple(_make_agent(f"a{i}") for i in range(MAX_AGENTS + 1))
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(topic="t", agents=agents, max_rounds=1)


def test_debate_round_config_rejects_duplicate_names() -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(
            topic="t",
            agents=(_make_agent("a"), _make_agent("a")),
            max_rounds=1,
        )


def test_debate_round_config_rejects_non_tuple_agents() -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(
            topic="t",
            agents=[_make_agent("a"), _make_agent("b")],  # type: ignore[arg-type]
            max_rounds=1,
        )


def test_debate_round_config_rejects_non_agent_entry(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(
            topic="t",
            agents=(two_agents[0], "not-an-agent"),  # type: ignore[arg-type]
            max_rounds=1,
        )


@pytest.mark.parametrize("bad", [0, MAX_ROUNDS + 1, -1])
def test_debate_round_config_rejects_bad_max_rounds(
    bad: int, two_agents: tuple[DebateAgent, DebateAgent]
) -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(topic="t", agents=two_agents, max_rounds=bad)


def test_debate_round_config_rejects_bool_max_rounds(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(
            topic="t",
            agents=two_agents,
            max_rounds=True,  # type: ignore[arg-type]
        )


def test_debate_round_config_rejects_empty_terminator(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    with pytest.raises(DebateRoundError):
        DebateRoundConfig(topic="t", agents=two_agents, max_rounds=1, terminator_marker="")


def test_min_rounds_constant() -> None:
    assert MIN_ROUNDS >= 1
    assert MIN_AGENTS >= 2
    assert DEFAULT_TERMINATOR == "TERMINATE"
    assert DEFAULT_RECOMMENDATION == "ABSTAIN"


# ---------------------------------------------------------------------------
# default_recommendation_extractor
# ---------------------------------------------------------------------------


def test_default_extractor_parses_recommendation_and_rationale(
    config: DebateRoundConfig,
) -> None:
    transcript = (
        DebateTurn(
            round_idx=0,
            speaker_idx=0,
            agent_name="alice",
            text="long.\nRECOMMENDATION: APPROVE\nRATIONALE: alpha positive",
        ),
        DebateTurn(
            round_idx=0,
            speaker_idx=1,
            agent_name="bob",
            text="risky.\nRECOMMENDATION: REJECT\nRATIONALE: drawdown breach",
        ),
    )
    rec, rat, votes = default_recommendation_extractor(transcript, config)
    # Tie → lex-asc tiebreak → APPROVE
    assert rec == "APPROVE"
    # Last rationale wins
    assert rat == "drawdown breach"
    assert dict(votes) == {"alice": "APPROVE", "bob": "REJECT"}


def test_default_extractor_plurality_vote(
    config: DebateRoundConfig,
) -> None:
    # alice approves twice, bob rejects once → APPROVE wins on last-turn-only
    transcript = (
        DebateTurn(
            round_idx=0,
            speaker_idx=0,
            agent_name="alice",
            text="RECOMMENDATION: APPROVE",
        ),
        DebateTurn(
            round_idx=0,
            speaker_idx=1,
            agent_name="bob",
            text="RECOMMENDATION: REJECT",
        ),
        DebateTurn(
            round_idx=1,
            speaker_idx=0,
            agent_name="alice",
            text="RECOMMENDATION: APPROVE",
        ),
    )
    rec, _rat, votes = default_recommendation_extractor(transcript, config)
    # alice's *last* turn is APPROVE; bob's only turn is REJECT
    assert votes == {"alice": "APPROVE", "bob": "REJECT"}
    # Tied 1-1 → lex-asc → APPROVE
    assert rec == "APPROVE"


def test_default_extractor_falls_back_to_abstain_on_missing_marker(
    config: DebateRoundConfig,
) -> None:
    transcript = (
        DebateTurn(round_idx=0, speaker_idx=0, agent_name="alice", text="bla"),
        DebateTurn(round_idx=0, speaker_idx=1, agent_name="bob", text="meh"),
    )
    rec, rat, votes = default_recommendation_extractor(transcript, config)
    assert rec == DEFAULT_RECOMMENDATION
    assert rat == DEFAULT_RATIONALE
    assert votes == {"alice": DEFAULT_RECOMMENDATION, "bob": DEFAULT_RECOMMENDATION}


def test_default_extractor_rejects_invalid_label(
    config: DebateRoundConfig,
) -> None:
    transcript = (
        DebateTurn(
            round_idx=0,
            speaker_idx=0,
            agent_name="alice",
            text="RECOMMENDATION: SOMETHING_ELSE",
        ),
        DebateTurn(
            round_idx=0,
            speaker_idx=1,
            agent_name="bob",
            text="RECOMMENDATION: APPROVE",
        ),
    )
    _rec, _rat, votes = default_recommendation_extractor(transcript, config)
    # invalid label maps to ABSTAIN
    assert votes["alice"] == DEFAULT_RECOMMENDATION
    assert votes["bob"] == "APPROVE"


def test_default_extractor_picks_only_uses_last_turn_per_agent(
    config: DebateRoundConfig,
) -> None:
    transcript = (
        DebateTurn(
            round_idx=0,
            speaker_idx=0,
            agent_name="alice",
            text="RECOMMENDATION: APPROVE",
        ),
        DebateTurn(
            round_idx=1,
            speaker_idx=0,
            agent_name="alice",
            text="RECOMMENDATION: REJECT",
        ),
        DebateTurn(
            round_idx=1,
            speaker_idx=1,
            agent_name="bob",
            text="RECOMMENDATION: REJECT",
        ),
    )
    _rec, _rat, votes = default_recommendation_extractor(transcript, config)
    assert votes["alice"] == "REJECT"


def test_default_extractor_rejects_bad_inputs(
    config: DebateRoundConfig,
) -> None:
    with pytest.raises(DebateRoundError):
        default_recommendation_extractor(["not-a-tuple"], config)  # type: ignore[arg-type]
    with pytest.raises(DebateRoundError):
        default_recommendation_extractor((), "not-a-config")  # type: ignore[arg-type]


def test_recommendation_labels_constant() -> None:
    assert "APPROVE" in RECOMMENDATION_LABELS
    assert "REJECT" in RECOMMENDATION_LABELS
    assert "ESCALATE" in RECOMMENDATION_LABELS
    assert "ABSTAIN" in RECOMMENDATION_LABELS


# ---------------------------------------------------------------------------
# run_debate_round — happy path
# ---------------------------------------------------------------------------


def test_run_debate_round_basic(config: DebateRoundConfig) -> None:
    result = run_debate_round(config=config, speaker=_approve_speaker())
    assert isinstance(result, DebateRoundProposal)
    assert result.topic == config.topic
    assert result.n_turns == config.max_rounds * config.n_agents
    assert result.n_rounds == config.max_rounds
    assert result.converged is False
    assert result.recommendation == "APPROVE"
    assert result.votes == {"alice": "APPROVE", "bob": "APPROVE"}


def test_run_debate_round_round_robin_order(config: DebateRoundConfig) -> None:
    result = run_debate_round(config=config, speaker=_approve_speaker())
    expected_speakers = ["alice", "bob", "alice", "bob"]
    assert [t.agent_name for t in result.turns] == expected_speakers


def test_run_debate_round_increments_round_idx(
    config: DebateRoundConfig,
) -> None:
    result = run_debate_round(config=config, speaker=_approve_speaker())
    assert [t.round_idx for t in result.turns] == [0, 0, 1, 1]
    assert [t.speaker_idx for t in result.turns] == [0, 1, 0, 1]


def test_run_debate_round_terminator_short_circuits(
    config: DebateRoundConfig,
) -> None:
    speaker = _ScriptedSpeaker(
        replies={
            "alice": ("Done. TERMINATE",),
            "bob": ("never reached",),
        }
    )
    result = run_debate_round(config=config, speaker=speaker)
    assert result.converged is True
    assert result.n_turns == 1
    assert result.turns[0].agent_name == "alice"


def test_run_debate_round_custom_terminator(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    cfg = DebateRoundConfig(
        topic="t",
        agents=two_agents,
        max_rounds=2,
        terminator_marker="<END>",
    )
    speaker = _ScriptedSpeaker(
        replies={
            "alice": ("hello", "RECOMMENDATION: APPROVE"),
            "bob": ("agreed <END>",),
        }
    )
    result = run_debate_round(config=cfg, speaker=speaker)
    assert result.converged is True
    assert result.n_turns == 2  # alice then bob


def test_run_debate_round_rejects_non_speaker(
    config: DebateRoundConfig,
) -> None:
    class NotASpeaker:
        pass

    with pytest.raises(DebateRoundError):
        run_debate_round(config=config, speaker=NotASpeaker())  # type: ignore[arg-type]


def test_run_debate_round_rejects_non_config() -> None:
    with pytest.raises(DebateRoundError):
        run_debate_round(
            config="not-a-config",  # type: ignore[arg-type]
            speaker=_approve_speaker(),
        )


def test_run_debate_round_rejects_non_string_reply(
    config: DebateRoundConfig,
) -> None:
    class IntSpeaker:
        def speak(
            self,
            *,
            agent: DebateAgent,
            transcript: tuple[DebateTurn, ...],
            topic: str,
            round_idx: int,
        ) -> str:
            return 42  # type: ignore[return-value]

    with pytest.raises(DebateRoundError):
        run_debate_round(config=config, speaker=IntSpeaker())


def test_run_debate_round_rejects_oversized_reply(
    config: DebateRoundConfig,
) -> None:
    big = "x" * (MAX_TURN_TEXT_LEN + 1)

    class HugeSpeaker:
        def speak(
            self,
            *,
            agent: DebateAgent,
            transcript: tuple[DebateTurn, ...],
            topic: str,
            round_idx: int,
        ) -> str:
            return big

    with pytest.raises(DebateRoundError):
        run_debate_round(config=config, speaker=HugeSpeaker())


# ---------------------------------------------------------------------------
# Custom extractor
# ---------------------------------------------------------------------------


def test_run_debate_round_uses_custom_extractor(
    config: DebateRoundConfig,
) -> None:
    def static(
        _turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig
    ) -> tuple[str, str, Mapping[str, str]]:
        return (
            "ESCALATE",
            "manual override",
            {"alice": "ESCALATE", "bob": "ESCALATE"},
        )

    result = run_debate_round(
        config=config,
        speaker=_approve_speaker(),
        recommendation_extractor=static,
    )
    assert result.recommendation == "ESCALATE"
    assert result.rationale == "manual override"
    assert result.votes == {"alice": "ESCALATE", "bob": "ESCALATE"}


def test_custom_extractor_rejects_missing_agent_keys(
    config: DebateRoundConfig,
) -> None:
    def bad(
        _turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig
    ) -> tuple[str, str, Mapping[str, str]]:
        return ("APPROVE", "ok", {"alice": "APPROVE"})

    with pytest.raises(DebateRoundError):
        run_debate_round(
            config=config,
            speaker=_approve_speaker(),
            recommendation_extractor=bad,
        )


def test_custom_extractor_rejects_oversized_recommendation(
    config: DebateRoundConfig,
) -> None:
    big_rec = "A" * (MAX_RECOMMENDATION_LEN + 1)

    def bad(
        _turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig
    ) -> tuple[str, str, Mapping[str, str]]:
        return (big_rec, "ok", {"alice": "APPROVE", "bob": "APPROVE"})

    with pytest.raises(DebateRoundError):
        run_debate_round(
            config=config,
            speaker=_approve_speaker(),
            recommendation_extractor=bad,
        )


def test_custom_extractor_rejects_oversized_rationale(
    config: DebateRoundConfig,
) -> None:
    big_rat = "r" * (MAX_RATIONALE_LEN + 1)

    def bad(
        _turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig
    ) -> tuple[str, str, Mapping[str, str]]:
        return ("APPROVE", big_rat, {"alice": "APPROVE", "bob": "APPROVE"})

    with pytest.raises(DebateRoundError):
        run_debate_round(
            config=config,
            speaker=_approve_speaker(),
            recommendation_extractor=bad,
        )


def test_custom_extractor_rejects_non_tuple_return(
    config: DebateRoundConfig,
) -> None:
    def bad(_turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig):
        return "not-a-tuple"

    with pytest.raises(DebateRoundError):
        run_debate_round(
            config=config,
            speaker=_approve_speaker(),
            recommendation_extractor=bad,  # type: ignore[arg-type]
        )


def test_custom_extractor_rejects_wrong_arity(
    config: DebateRoundConfig,
) -> None:
    def bad(_turns: tuple[DebateTurn, ...], _cfg: DebateRoundConfig):
        return ("APPROVE", "ok")  # missing votes

    with pytest.raises(DebateRoundError):
        run_debate_round(
            config=config,
            speaker=_approve_speaker(),
            recommendation_extractor=bad,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Digest + INV-15 determinism
# ---------------------------------------------------------------------------


def test_proposal_digest_is_hex16(config: DebateRoundConfig) -> None:
    result = run_debate_round(config=config, speaker=_approve_speaker())
    assert len(result.proposal_digest) == 32  # 16 bytes hex
    int(result.proposal_digest, 16)  # parseable hex


def test_proposal_digest_three_run_replay_equality(
    config: DebateRoundConfig,
) -> None:
    """INV-15: same config + speaker → same proposal_digest, three runs."""

    digests = []
    for _ in range(3):
        result = run_debate_round(config=config, speaker=_approve_speaker())
        digests.append(result.proposal_digest)
    assert digests[0] == digests[1] == digests[2]


def test_proposal_digest_differs_with_topic(
    two_agents: tuple[DebateAgent, DebateAgent],
) -> None:
    cfg_a = DebateRoundConfig(topic="topic-a", agents=two_agents, max_rounds=1)
    cfg_b = DebateRoundConfig(topic="topic-b", agents=two_agents, max_rounds=1)
    result_a = run_debate_round(config=cfg_a, speaker=_approve_speaker())
    result_b = run_debate_round(config=cfg_b, speaker=_approve_speaker())
    assert result_a.proposal_digest != result_b.proposal_digest


def test_proposal_digest_differs_with_speaker_reply(
    config: DebateRoundConfig,
) -> None:
    speaker_a = _ScriptedSpeaker(
        replies={
            "alice": ("alpha\nRECOMMENDATION: APPROVE\nRATIONALE: x",),
            "bob": ("beta\nRECOMMENDATION: APPROVE\nRATIONALE: y",),
        }
    )
    speaker_b = _ScriptedSpeaker(
        replies={
            "alice": ("gamma\nRECOMMENDATION: APPROVE\nRATIONALE: x",),
            "bob": ("beta\nRECOMMENDATION: APPROVE\nRATIONALE: y",),
        }
    )
    result_a = run_debate_round(config=config, speaker=speaker_a)
    result_b = run_debate_round(config=config, speaker=speaker_b)
    assert result_a.proposal_digest != result_b.proposal_digest


def test_votes_are_sorted_by_key(config: DebateRoundConfig) -> None:
    """INV-15: per-agent vote mapping iterates in sorted-key order."""

    result = run_debate_round(config=config, speaker=_approve_speaker())
    keys = list(result.votes.keys())
    assert keys == sorted(keys)


def test_proposal_is_frozen(config: DebateRoundConfig) -> None:
    result = run_debate_round(config=config, speaker=_approve_speaker())
    with pytest.raises((AttributeError, TypeError)):
        result.recommendation = "REJECT"  # type: ignore[misc]


def test_debate_turn_is_frozen() -> None:
    turn = DebateTurn(round_idx=0, speaker_idx=0, agent_name="a", text="x")
    with pytest.raises((AttributeError, TypeError)):
        turn.text = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bigger committees + many rounds
# ---------------------------------------------------------------------------


def test_debate_round_with_three_agents() -> None:
    agents = (_make_agent("a"), _make_agent("b"), _make_agent("c"))
    cfg = DebateRoundConfig(topic="t", agents=agents, max_rounds=2)
    speaker = _ScriptedSpeaker(
        replies={
            "a": ("RECOMMENDATION: APPROVE",),
            "b": ("RECOMMENDATION: REJECT",),
            "c": ("RECOMMENDATION: APPROVE",),
        }
    )
    result = run_debate_round(config=cfg, speaker=speaker)
    assert result.n_turns == 6
    assert result.votes == {"a": "APPROVE", "b": "REJECT", "c": "APPROVE"}
    assert result.recommendation == "APPROVE"  # 2 vs 1


def test_round_robin_loops_across_rounds() -> None:
    agents = (_make_agent("a"), _make_agent("b"), _make_agent("c"))
    cfg = DebateRoundConfig(topic="t", agents=agents, max_rounds=3)
    speaker = _ScriptedSpeaker(
        replies={
            "a": ("a-msg",),
            "b": ("b-msg",),
            "c": ("c-msg",),
        }
    )
    result = run_debate_round(config=cfg, speaker=speaker)
    names = [t.agent_name for t in result.turns]
    assert names == ["a", "b", "c"] * 3


# ---------------------------------------------------------------------------
# Speaker Protocol structural typing
# ---------------------------------------------------------------------------


def test_speaker_is_runtime_checkable() -> None:
    assert isinstance(_approve_speaker(), Speaker)


def test_non_speaker_is_not_speaker_instance() -> None:
    class Bare:
        pass

    assert not isinstance(Bare(), Speaker)


# ---------------------------------------------------------------------------
# litellm_speaker_factory — lazy seam
# ---------------------------------------------------------------------------


def test_litellm_speaker_factory_rejects_non_callable() -> None:
    with pytest.raises(DebateRoundError):
        litellm_speaker_factory(completion="not-callable")  # type: ignore[arg-type]


def test_litellm_speaker_factory_rejects_non_str_prefix() -> None:
    def cb(**_kw: object) -> str:
        return ""

    with pytest.raises(DebateRoundError):
        litellm_speaker_factory(
            completion=cb,
            system_prompt_prefix=42,  # type: ignore[arg-type]
        )


def test_litellm_speaker_factory_returns_speaker(
    config: DebateRoundConfig,
) -> None:
    seen: list[list[dict[str, str]]] = []

    def fake_completion(*, messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        return "ok\nRECOMMENDATION: APPROVE\nRATIONALE: looks fine"

    speaker = litellm_speaker_factory(completion=fake_completion)
    result = run_debate_round(config=config, speaker=speaker)
    assert isinstance(speaker, Speaker)
    assert result.n_turns == 4
    # first turn: system + just the "your turn" cue (no transcript yet)
    first = seen[0]
    assert first[0]["role"] == "system"
    assert "alice" in first[0]["content"]
    assert "bull" in first[0]["content"]
    assert first[-1]["role"] == "user"


def test_litellm_speaker_factory_system_prefix_appears(
    config: DebateRoundConfig,
) -> None:
    seen: list[list[dict[str, str]]] = []

    def fake_completion(*, messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        return "RECOMMENDATION: APPROVE"

    speaker = litellm_speaker_factory(
        completion=fake_completion,
        system_prompt_prefix="DIX governance debate.",
    )
    run_debate_round(config=config, speaker=speaker)
    assert "DIX governance debate." in seen[0][0]["content"]


def test_litellm_factory_does_not_import_litellm_at_module_top() -> None:
    """The factory wires LiteLLM at the call site — its presence in the
    module text must not trigger a top-level `import litellm`."""

    src = _MODULE_PATH.read_text(encoding="utf-8")
    # No top-level "import litellm" line; only string references in the
    # docstring/factory body are allowed.
    assert "\nimport litellm" not in src
    assert "\nfrom litellm" not in src
