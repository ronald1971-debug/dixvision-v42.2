"""Tests for intelligence_engine/agents/crew_strategy_council.py (B-04)."""

from __future__ import annotations

import ast
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intelligence_engine.agents import crew_strategy_council as csc  # noqa: E402
from intelligence_engine.agents.crew_strategy_council import (  # noqa: E402
    ARBITER_RATIONALE_KEY,
    ARBITER_RECOMMENDATION_KEY,
    CANONICAL_ROLE_ORDER,
    DEFAULT_RATIONALE,
    DEFAULT_RECOMMENDATION,
    MAX_AGENTS,
    MAX_BACKSTORY_LEN,
    MAX_DESCRIPTION_LEN,
    MAX_GOAL_LEN,
    MAX_OUTPUT_KEY_LEN,
    MAX_OUTPUT_KEYS,
    MAX_OUTPUT_VALUE_LEN,
    MAX_RATIONALE_LEN,
    MAX_TASKS,
    MAX_TOPIC_LEN,
    MIN_AGENTS,
    MIN_TASKS,
    NEW_PIP_DEPENDENCIES,
    RECOMMENDATION_LABELS,
    CouncilAgent,
    CouncilConfig,
    CouncilError,
    CouncilProposal,
    CouncilRole,
    CouncilTask,
    CouncilTaskResult,
    StructuredSpeaker,
    crewai_speaker_factory,
    run_council,
)

SOURCE_PATH = REPO_ROOT / "intelligence_engine" / "agents" / "crew_strategy_council.py"
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")
SOURCE_TREE = ast.parse(SOURCE_TEXT)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "crewai",
        "litellm",
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
        "openai",
        "anthropic",
    }
)


def _walk_module_imports() -> set[str]:
    names: set[str] = set()
    for node in SOURCE_TREE.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_top_level_imports() -> None:
    top = _walk_module_imports()
    for mod in _FORBIDDEN_TOP_LEVEL:
        assert mod not in top, f"crew_strategy_council.py must not top-level import {mod!r}"


def test_no_engine_cross_imports() -> None:
    top = _walk_module_imports()
    forbidden_engines = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
    }
    for eng in forbidden_engines:
        assert eng not in top, f"must not import {eng!r}"
    # And no nested import either.
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.Import):
            for alias in node.names:
                first = alias.name.split(".")[0]
                assert first not in forbidden_engines, first
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                first = node.module.split(".")[0]
                assert first not in forbidden_engines, first


def test_does_not_construct_typed_bus_events() -> None:
    """B27 / B28 / INV-71 authority symmetry — agents tier MUST NOT
    construct typed bus events. Pinned by AST."""

    forbidden_ctors = {
        "PatchProposal",
        "SignalEvent",
        "GovernanceDecision",
        "ExecutionIntent",
    }
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_ctors:
                raise AssertionError(
                    f"crew_strategy_council.py must not construct {name}(...) "
                    "— promotion to typed bus events happens in "
                    "governance_engine, never in intelligence_engine/agents"
                )


def test_adapted_from_header_present() -> None:
    assert "# ADAPTED FROM: crewAIInc/crewAI" in SOURCE_TEXT
    assert "crewai/agent.py" in SOURCE_TEXT
    assert "crewai/crew.py" in SOURCE_TEXT
    assert "crewai/task.py" in SOURCE_TEXT


def test_new_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)


def test_module_does_not_construct_governance_decision_strings() -> None:
    """Sanity pin: no GovernanceDecision construction at all."""
    assert "GovernanceDecision(" not in SOURCE_TEXT


def test_module_carries_advisory_only_clause() -> None:
    """INV-12 advisory clause must be present in the docstring."""
    assert "ADVISORY ONLY" in SOURCE_TEXT


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_canonical_role_order() -> None:
    assert CANONICAL_ROLE_ORDER == (
        CouncilRole.SIGNAL_ANALYST,
        CouncilRole.RISK_OFFICER,
        CouncilRole.REGIME_EXPERT,
        CouncilRole.ARBITER,
    )


def test_recommendation_labels_constant() -> None:
    assert RECOMMENDATION_LABELS == frozenset({"APPROVE", "REJECT", "ESCALATE", "ABSTAIN"})


def test_arbiter_reserved_keys() -> None:
    assert ARBITER_RECOMMENDATION_KEY == "recommendation"
    assert ARBITER_RATIONALE_KEY == "rationale"


def test_bounds_constants() -> None:
    assert MIN_AGENTS == 2
    assert MAX_AGENTS == 8
    assert MIN_TASKS == 1
    assert MAX_TASKS == 16
    assert MAX_TOPIC_LEN > 0
    assert MAX_GOAL_LEN > 0
    assert MAX_BACKSTORY_LEN > 0
    assert MAX_DESCRIPTION_LEN > 0
    assert MAX_OUTPUT_KEY_LEN > 0
    assert MAX_OUTPUT_VALUE_LEN > 0
    assert MAX_OUTPUT_KEYS > 0
    assert MAX_RATIONALE_LEN > 0


# ---------------------------------------------------------------------------
# CouncilAgent validation
# ---------------------------------------------------------------------------


def _agent(
    role: CouncilRole = CouncilRole.SIGNAL_ANALYST,
    goal: str = "Decode microstructure",
    backstory: str = "Senior microstructure quant",
    expected_output_keys: tuple[str, ...] = ("verdict", "evidence"),
) -> CouncilAgent:
    return CouncilAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        expected_output_keys=expected_output_keys,
    )


def test_council_agent_basic() -> None:
    a = _agent()
    assert a.role is CouncilRole.SIGNAL_ANALYST
    assert a.expected_output_keys == ("verdict", "evidence")


def test_council_agent_rejects_non_role() -> None:
    with pytest.raises(CouncilError):
        CouncilAgent(
            role="signal_analyst",  # type: ignore[arg-type]
            goal="g",
            backstory="b",
            expected_output_keys=("k",),
        )


def test_council_agent_rejects_empty_goal() -> None:
    with pytest.raises(CouncilError):
        _agent(goal=" ")


def test_council_agent_rejects_oversized_goal() -> None:
    with pytest.raises(CouncilError):
        _agent(goal="x" * (MAX_GOAL_LEN + 1))


def test_council_agent_rejects_empty_backstory() -> None:
    with pytest.raises(CouncilError):
        _agent(backstory="")


def test_council_agent_rejects_oversized_backstory() -> None:
    with pytest.raises(CouncilError):
        _agent(backstory="b" * (MAX_BACKSTORY_LEN + 1))


def test_council_agent_rejects_non_tuple_keys() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=["a"])  # type: ignore[arg-type]


def test_council_agent_rejects_empty_keys() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=())


def test_council_agent_rejects_too_many_keys() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=tuple(f"k{i}" for i in range(MAX_OUTPUT_KEYS + 1)))


def test_council_agent_rejects_blank_key() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=("verdict", " "))


def test_council_agent_rejects_duplicate_keys() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=("verdict", "verdict"))


def test_council_agent_rejects_oversized_key() -> None:
    with pytest.raises(CouncilError):
        _agent(expected_output_keys=("x" * (MAX_OUTPUT_KEY_LEN + 1),))


def test_council_agent_arbiter_requires_recommendation_key() -> None:
    with pytest.raises(CouncilError):
        _agent(
            role=CouncilRole.ARBITER,
            expected_output_keys=("rationale", "extra"),
        )


def test_council_agent_arbiter_requires_rationale_key() -> None:
    with pytest.raises(CouncilError):
        _agent(
            role=CouncilRole.ARBITER,
            expected_output_keys=("recommendation", "extra"),
        )


def test_council_agent_arbiter_accepts_reserved_keys() -> None:
    a = _agent(
        role=CouncilRole.ARBITER,
        expected_output_keys=("recommendation", "rationale", "summary"),
    )
    assert a.role is CouncilRole.ARBITER


def test_council_agent_is_frozen() -> None:
    a = _agent()
    with pytest.raises(FrozenInstanceError):
        a.goal = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CouncilTask validation
# ---------------------------------------------------------------------------


def _task(
    task_id: str = "t1",
    description: str = "Analyse signal quality",
    role_owner: CouncilRole = CouncilRole.SIGNAL_ANALYST,
) -> CouncilTask:
    return CouncilTask(
        task_id=task_id,
        description=description,
        role_owner=role_owner,
    )


def test_council_task_basic() -> None:
    t = _task()
    assert t.task_id == "t1"
    assert t.role_owner is CouncilRole.SIGNAL_ANALYST


def test_council_task_rejects_blank_id() -> None:
    with pytest.raises(CouncilError):
        _task(task_id="")


def test_council_task_rejects_oversized_id() -> None:
    with pytest.raises(CouncilError):
        _task(task_id="x" * (MAX_OUTPUT_KEY_LEN + 1))


def test_council_task_rejects_blank_description() -> None:
    with pytest.raises(CouncilError):
        _task(description="   ")


def test_council_task_rejects_oversized_description() -> None:
    with pytest.raises(CouncilError):
        _task(description="x" * (MAX_DESCRIPTION_LEN + 1))


def test_council_task_rejects_non_role_owner() -> None:
    with pytest.raises(CouncilError):
        _task(role_owner="signal_analyst")  # type: ignore[arg-type]


def test_council_task_is_frozen() -> None:
    t = _task()
    with pytest.raises(FrozenInstanceError):
        t.description = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CouncilConfig validation
# ---------------------------------------------------------------------------


def _make_canonical_agents() -> tuple[CouncilAgent, ...]:
    return (
        _agent(
            role=CouncilRole.SIGNAL_ANALYST,
            expected_output_keys=("verdict", "evidence"),
        ),
        _agent(
            role=CouncilRole.RISK_OFFICER,
            goal="Bound risk",
            backstory="Senior risk officer",
            expected_output_keys=("risk_level", "risk_notes"),
        ),
        _agent(
            role=CouncilRole.REGIME_EXPERT,
            goal="Classify regime",
            backstory="Macro regime expert",
            expected_output_keys=("regime", "horizon"),
        ),
        _agent(
            role=CouncilRole.ARBITER,
            goal="Synthesise final recommendation",
            backstory="Senior arbiter",
            expected_output_keys=("recommendation", "rationale"),
        ),
    )


def _make_canonical_tasks() -> tuple[CouncilTask, ...]:
    return (
        _task("t_signal", "Analyse signal", CouncilRole.SIGNAL_ANALYST),
        _task("t_risk", "Assess risk", CouncilRole.RISK_OFFICER),
        _task("t_regime", "Classify regime", CouncilRole.REGIME_EXPERT),
        _task("t_arbit", "Synthesise final recommendation", CouncilRole.ARBITER),
    )


def _canonical_config() -> CouncilConfig:
    return CouncilConfig(
        topic="Should we long ES futures?",
        agents=_make_canonical_agents(),
        tasks=_make_canonical_tasks(),
    )


def test_council_config_basic() -> None:
    cfg = _canonical_config()
    assert cfg.topic == "Should we long ES futures?"
    assert len(cfg.agents) == 4
    assert len(cfg.tasks) == 4


def test_council_config_rejects_blank_topic() -> None:
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic=" ",
            agents=_make_canonical_agents(),
            tasks=_make_canonical_tasks(),
        )


def test_council_config_rejects_oversized_topic() -> None:
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="x" * (MAX_TOPIC_LEN + 1),
            agents=_make_canonical_agents(),
            tasks=_make_canonical_tasks(),
        )


def test_council_config_rejects_non_tuple_agents() -> None:
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=list(_make_canonical_agents()),  # type: ignore[arg-type]
            tasks=_make_canonical_tasks(),
        )


def test_council_config_rejects_too_few_agents() -> None:
    arbiter = _agent(
        role=CouncilRole.ARBITER,
        expected_output_keys=("recommendation", "rationale"),
    )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=(arbiter,),
            tasks=(_task("a", "x", CouncilRole.ARBITER),),
        )


def test_council_config_rejects_too_many_agents() -> None:
    # Build MAX_AGENTS + 1 by repeating roles → also catches duplicate
    # but the count check fires first because we keep the tuple size
    # above MAX_AGENTS.
    too_many: list[CouncilAgent] = []
    for i in range(MAX_AGENTS + 1):
        role = CouncilRole.ARBITER if i == 0 else CouncilRole.SIGNAL_ANALYST
        keys: tuple[str, ...] = (
            ("recommendation", "rationale") if role is CouncilRole.ARBITER else (f"k{i}",)
        )
        too_many.append(
            CouncilAgent(
                role=role,
                goal=f"goal {i}",
                backstory=f"backstory {i}",
                expected_output_keys=keys,
            )
        )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=tuple(too_many),
            tasks=_make_canonical_tasks(),
        )


def test_council_config_rejects_duplicate_role() -> None:
    agents = _make_canonical_agents()
    dup = (
        agents[0],
        agents[0],  # duplicate SIGNAL_ANALYST
        agents[3],
    )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=dup,
            tasks=_make_canonical_tasks(),
        )


def test_council_config_requires_arbiter() -> None:
    agents = (
        _agent(role=CouncilRole.SIGNAL_ANALYST),
        _agent(
            role=CouncilRole.RISK_OFFICER,
            goal="x",
            backstory="y",
            expected_output_keys=("risk_level",),
        ),
    )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=agents,
            tasks=(_task("a", "x", CouncilRole.SIGNAL_ANALYST),),
        )


def test_council_config_rejects_non_tuple_tasks() -> None:
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=_make_canonical_agents(),
            tasks=list(_make_canonical_tasks()),  # type: ignore[arg-type]
        )


def test_council_config_rejects_too_few_tasks() -> None:
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=_make_canonical_agents(),
            tasks=(),
        )


def test_council_config_rejects_too_many_tasks() -> None:
    tasks = tuple(_task(f"t{i}", f"d{i}", CouncilRole.SIGNAL_ANALYST) for i in range(MAX_TASKS)) + (
        _task("t_final", "synth", CouncilRole.ARBITER),
    )
    # That's MAX_TASKS + 1
    assert len(tasks) == MAX_TASKS + 1
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=_make_canonical_agents(),
            tasks=tasks,
        )


def test_council_config_rejects_duplicate_task_id() -> None:
    tasks = (
        _task("dup", "a", CouncilRole.SIGNAL_ANALYST),
        _task("dup", "b", CouncilRole.ARBITER),
    )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=_make_canonical_agents(),
            tasks=tasks,
        )


def test_council_config_rejects_task_referencing_missing_role() -> None:
    # Build agents WITHOUT regime expert, but reference it in a task.
    agents = (
        _agent(role=CouncilRole.SIGNAL_ANALYST),
        _agent(
            role=CouncilRole.ARBITER,
            goal="x",
            backstory="y",
            expected_output_keys=("recommendation", "rationale"),
        ),
    )
    tasks = (
        _task("a", "x", CouncilRole.REGIME_EXPERT),
        _task("b", "y", CouncilRole.ARBITER),
    )
    with pytest.raises(CouncilError):
        CouncilConfig(topic="t", agents=agents, tasks=tasks)


def test_council_config_last_task_must_be_arbiter() -> None:
    tasks = (
        _task("t_arbit", "synth", CouncilRole.ARBITER),
        _task("t_signal", "analyse", CouncilRole.SIGNAL_ANALYST),
    )
    with pytest.raises(CouncilError):
        CouncilConfig(
            topic="t",
            agents=_make_canonical_agents(),
            tasks=tasks,
        )


def test_council_config_is_frozen() -> None:
    cfg = _canonical_config()
    with pytest.raises(FrozenInstanceError):
        cfg.topic = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StructuredSpeaker Protocol — structural typing
# ---------------------------------------------------------------------------


class _StubSpeaker:
    def __init__(self, replies: Mapping[str, Mapping[str, str]]) -> None:
        self._replies = replies

    def speak(
        self,
        *,
        agent: CouncilAgent,
        task: CouncilTask,
        topic: str,
        prior_results: tuple[CouncilTaskResult, ...],
    ) -> Mapping[str, str]:
        return self._replies[task.task_id]


def test_speaker_is_runtime_checkable() -> None:
    speaker = _StubSpeaker({})
    assert isinstance(speaker, StructuredSpeaker)


def test_non_speaker_is_not_speaker_instance() -> None:
    class _NotASpeaker:
        pass

    obj = _NotASpeaker()
    assert not isinstance(obj, StructuredSpeaker)


# ---------------------------------------------------------------------------
# Happy-path orchestrator
# ---------------------------------------------------------------------------


def _stub_replies() -> dict[str, dict[str, str]]:
    return {
        "t_signal": {"verdict": "bull", "evidence": "OBI > 0.2"},
        "t_risk": {"risk_level": "moderate", "risk_notes": "VaR within band"},
        "t_regime": {"regime": "trend_up", "horizon": "intraday"},
        "t_arbit": {
            "recommendation": "APPROVE",
            "rationale": "Signal + regime align; risk acceptable",
        },
    }


def test_run_council_basic() -> None:
    cfg = _canonical_config()
    speaker = _StubSpeaker(_stub_replies())
    proposal = run_council(config=cfg, speaker=speaker)
    assert isinstance(proposal, CouncilProposal)
    assert proposal.topic == cfg.topic
    assert len(proposal.results) == 4
    assert proposal.recommendation == "APPROVE"
    assert "Signal + regime align" in proposal.rationale


def test_run_council_preserves_task_order() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    assert [r.task_id for r in proposal.results] == [
        "t_signal",
        "t_risk",
        "t_regime",
        "t_arbit",
    ]
    assert [r.role for r in proposal.results] == [
        CouncilRole.SIGNAL_ANALYST,
        CouncilRole.RISK_OFFICER,
        CouncilRole.REGIME_EXPERT,
        CouncilRole.ARBITER,
    ]


def test_run_council_results_have_sorted_output_keys() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    for r in proposal.results:
        assert list(r.output.keys()) == sorted(r.output.keys())


def test_run_council_rejects_non_config() -> None:
    with pytest.raises(CouncilError):
        run_council(config=object(), speaker=_StubSpeaker({}))  # type: ignore[arg-type]


def test_run_council_rejects_non_speaker() -> None:
    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=object())  # type: ignore[arg-type]


def test_run_council_rejects_non_mapping_reply() -> None:
    class _BadSpeaker:
        def speak(self, **_: object) -> object:  # type: ignore[override]
            return "not a mapping"

    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=_BadSpeaker())  # type: ignore[arg-type]


def test_run_council_rejects_reply_missing_key() -> None:
    replies = _stub_replies()
    replies["t_signal"] = {"verdict": "bull"}  # drop "evidence"
    speaker = _StubSpeaker(replies)
    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=speaker)


def test_run_council_rejects_reply_extra_key() -> None:
    replies = _stub_replies()
    replies["t_signal"] = {
        "verdict": "bull",
        "evidence": "OBI > 0.2",
        "stray": "should-not-be-here",
    }
    speaker = _StubSpeaker(replies)
    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=speaker)


def test_run_council_rejects_non_string_value() -> None:
    replies = _stub_replies()
    replies["t_signal"] = {"verdict": 1, "evidence": "x"}  # type: ignore[dict-item]
    speaker = _StubSpeaker(replies)
    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=speaker)


def test_run_council_rejects_oversized_value() -> None:
    replies = _stub_replies()
    replies["t_signal"] = {
        "verdict": "bull",
        "evidence": "x" * (MAX_OUTPUT_VALUE_LEN + 1),
    }
    speaker = _StubSpeaker(replies)
    cfg = _canonical_config()
    with pytest.raises(CouncilError):
        run_council(config=cfg, speaker=speaker)


# ---------------------------------------------------------------------------
# Arbiter projection
# ---------------------------------------------------------------------------


def test_run_council_invalid_arbiter_label_falls_back_to_default() -> None:
    replies = _stub_replies()
    replies["t_arbit"] = {
        "recommendation": "WAT",
        "rationale": "nope",
    }
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(replies))
    assert proposal.recommendation == DEFAULT_RECOMMENDATION


def test_run_council_blank_rationale_falls_back_to_default() -> None:
    replies = _stub_replies()
    replies["t_arbit"] = {
        "recommendation": "REJECT",
        "rationale": " ",
    }
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(replies))
    assert proposal.rationale == DEFAULT_RATIONALE
    assert proposal.recommendation == "REJECT"


def test_run_council_lowercase_arbiter_label_normalised() -> None:
    replies = _stub_replies()
    replies["t_arbit"] = {
        "recommendation": "approve",
        "rationale": "ok",
    }
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(replies))
    assert proposal.recommendation == "APPROVE"


def test_run_council_oversized_rationale_truncated() -> None:
    replies = _stub_replies()
    big = "x" * (MAX_RATIONALE_LEN + 100)
    # First: oversized value is rejected at speaker boundary (8192 limit).
    # Make rationale exactly MAX_RATIONALE_LEN+10 which is below
    # MAX_OUTPUT_VALUE_LEN but above MAX_RATIONALE_LEN.
    rationale = "x" * (MAX_RATIONALE_LEN + 10)
    assert len(rationale) <= MAX_OUTPUT_VALUE_LEN
    replies["t_arbit"] = {
        "recommendation": "APPROVE",
        "rationale": rationale,
    }
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(replies))
    assert len(proposal.rationale) == MAX_RATIONALE_LEN
    assert big != proposal.rationale


# ---------------------------------------------------------------------------
# Digest determinism — INV-15
# ---------------------------------------------------------------------------


def test_proposal_digest_is_hex16() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    assert len(proposal.proposal_digest) == 32
    int(proposal.proposal_digest, 16)


def test_proposal_digest_three_run_replay_equality() -> None:
    cfg = _canonical_config()
    p1 = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    p2 = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    p3 = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    assert p1.proposal_digest == p2.proposal_digest == p3.proposal_digest


def test_proposal_digest_changes_with_topic() -> None:
    cfg1 = _canonical_config()
    cfg2 = CouncilConfig(
        topic="A different topic entirely",
        agents=cfg1.agents,
        tasks=cfg1.tasks,
    )
    p1 = run_council(config=cfg1, speaker=_StubSpeaker(_stub_replies()))
    p2 = run_council(config=cfg2, speaker=_StubSpeaker(_stub_replies()))
    assert p1.proposal_digest != p2.proposal_digest


def test_proposal_digest_changes_with_output_value() -> None:
    cfg = _canonical_config()
    p1 = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    replies2 = _stub_replies()
    replies2["t_signal"]["evidence"] = "Different evidence"
    p2 = run_council(config=cfg, speaker=_StubSpeaker(replies2))
    assert p1.proposal_digest != p2.proposal_digest


def test_proposal_digest_changes_with_recommendation() -> None:
    cfg = _canonical_config()
    replies_a = _stub_replies()
    replies_b = _stub_replies()
    replies_b["t_arbit"]["recommendation"] = "REJECT"
    p_a = run_council(config=cfg, speaker=_StubSpeaker(replies_a))
    p_b = run_council(config=cfg, speaker=_StubSpeaker(replies_b))
    assert p_a.proposal_digest != p_b.proposal_digest


def test_council_proposal_is_frozen() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    with pytest.raises(FrozenInstanceError):
        proposal.recommendation = "REJECT"  # type: ignore[misc]


def test_council_task_result_is_frozen() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    r = proposal.results[0]
    with pytest.raises(FrozenInstanceError):
        r.task_id = "x"  # type: ignore[misc]


def test_task_result_output_is_immutable_mapping() -> None:
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=_StubSpeaker(_stub_replies()))
    r = proposal.results[0]
    with pytest.raises(TypeError):
        r.output["evidence"] = "tampered"  # type: ignore[index]


def test_run_council_propagates_prior_results_in_order() -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []

    class _ObservingSpeaker:
        def speak(
            self,
            *,
            agent: CouncilAgent,
            task: CouncilTask,
            topic: str,
            prior_results: tuple[CouncilTaskResult, ...],
        ) -> Mapping[str, str]:
            seen.append((task.task_id, tuple(r.task_id for r in prior_results)))
            return _stub_replies()[task.task_id]

    cfg = _canonical_config()
    run_council(config=cfg, speaker=_ObservingSpeaker())  # type: ignore[arg-type]
    assert seen == [
        ("t_signal", ()),
        ("t_risk", ("t_signal",)),
        ("t_regime", ("t_signal", "t_risk")),
        ("t_arbit", ("t_signal", "t_risk", "t_regime")),
    ]


# ---------------------------------------------------------------------------
# crewai_speaker_factory — lazy seam
# ---------------------------------------------------------------------------


def test_crewai_speaker_factory_rejects_non_callable_completion() -> None:
    with pytest.raises(CouncilError):
        crewai_speaker_factory(completion=42)  # type: ignore[arg-type]


def test_crewai_speaker_factory_rejects_non_str_prefix() -> None:
    with pytest.raises(CouncilError):
        crewai_speaker_factory(
            completion=lambda **_: {"a": "b"},
            system_prompt_prefix=42,  # type: ignore[arg-type]
        )


def test_crewai_speaker_factory_returns_speaker() -> None:
    speaker = crewai_speaker_factory(completion=lambda **_: {"verdict": "bull", "evidence": "ok"})
    assert isinstance(speaker, StructuredSpeaker)


def test_crewai_speaker_factory_passes_through_keys() -> None:
    captured: dict[str, object] = {}

    def completion(
        *,
        system: str,
        user: str,
        expected_keys: tuple[str, ...],
    ) -> Mapping[str, str]:
        captured["system"] = system
        captured["user"] = user
        captured["expected_keys"] = expected_keys
        return {k: f"value-{k}" for k in expected_keys}

    speaker = crewai_speaker_factory(
        completion=completion,
        system_prompt_prefix="DIX council",
    )
    cfg = _canonical_config()
    proposal = run_council(config=cfg, speaker=speaker)
    assert proposal.results[0].output["verdict"] == "value-verdict"
    assert "DIX council" in captured["system"]
    assert "Topic:" in captured["user"]
    assert captured["expected_keys"] == ("recommendation", "rationale")


def test_crewai_speaker_factory_does_not_import_litellm_at_top() -> None:
    """The production seam must NOT pull litellm into the module
    top level. (litellm wiring lives in the caller, typically S-12.)"""
    assert "import litellm" not in SOURCE_TEXT
    assert "from litellm" not in SOURCE_TEXT


# ---------------------------------------------------------------------------
# Multi-config independence
# ---------------------------------------------------------------------------


def test_two_independent_configs_have_independent_digests() -> None:
    cfg1 = _canonical_config()
    cfg2 = CouncilConfig(
        topic="Different question",
        agents=cfg1.agents,
        tasks=cfg1.tasks,
    )
    speaker = _StubSpeaker(_stub_replies())
    p1 = run_council(config=cfg1, speaker=speaker)
    p2 = run_council(config=cfg2, speaker=speaker)
    assert p1.proposal_digest != p2.proposal_digest
    # But topic identity within each is preserved.
    assert p1.topic == cfg1.topic
    assert p2.topic == cfg2.topic


# ---------------------------------------------------------------------------
# csc namespace pin (catch accidental renames at module level)
# ---------------------------------------------------------------------------


def test_module_namespace() -> None:
    assert hasattr(csc, "run_council")
    assert hasattr(csc, "CouncilConfig")
    assert hasattr(csc, "CouncilProposal")
    assert hasattr(csc, "CouncilRole")
    assert hasattr(csc, "StructuredSpeaker")
    assert hasattr(csc, "crewai_speaker_factory")
