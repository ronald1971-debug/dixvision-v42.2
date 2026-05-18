# ADAPTED FROM: https://github.com/geekan/MetaGPT (MIT)
#
# Tests for C-14 metagpt — multi-role strategy critique council
# (intelligence_engine/agents/strategy_council.py) and iterative
# critique loop (evolution_engine/critique_loop.py).
"""C-14 tests: critique-round + critique-loop."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from evolution_engine import critique_loop as cl
from evolution_engine.critique_loop import (
    CritiqueLoopError,
    CritiqueLoopResult,
    CritiqueRoundRecord,
    MutationError,
    ProposalMutator,
    run_critique_loop,
)
from intelligence_engine.agents import strategy_council as sc
from intelligence_engine.agents.strategy_council import (
    CANONICAL_CRITIC_ROLES,
    ConsensusError,
    CriticMessage,
    CriticRole,
    CritiqueConsensus,
    CritiqueLog,
    CritiqueRecommendation,
    MessageError,
    RoleSpec,
    RoleSpecError,
    StrategyCouncilError,
    StructuredSpeaker,
    run_critique_round,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_strategy_council_new_pip_dependencies() -> None:
    assert sc.NEW_PIP_DEPENDENCIES == ("metagpt",)


def test_critique_loop_new_pip_dependencies() -> None:
    assert cl.NEW_PIP_DEPENDENCIES == ("metagpt",)


def test_strategy_council_error_hierarchy() -> None:
    assert issubclass(StrategyCouncilError, ValueError)
    assert issubclass(RoleSpecError, StrategyCouncilError)
    assert issubclass(MessageError, StrategyCouncilError)
    assert issubclass(ConsensusError, StrategyCouncilError)


def test_critique_loop_error_hierarchy() -> None:
    assert issubclass(CritiqueLoopError, ValueError)
    assert issubclass(MutationError, CritiqueLoopError)


def test_canonical_critic_roles_order() -> None:
    assert CANONICAL_CRITIC_ROLES == (
        CriticRole.SIGNAL_ANALYST,
        CriticRole.RISK_OFFICER,
        CriticRole.REGIME_EXPERT,
        CriticRole.ARBITER,
    )


# ---------------------------------------------------------------------------
# AST guards — top-level imports + lazy seam
# ---------------------------------------------------------------------------


def _top_level_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


_SC_SRC = Path(sc.__file__).read_text()
_SC_TREE = ast.parse(_SC_SRC)
_CL_SRC = Path(cl.__file__).read_text()
_CL_TREE = ast.parse(_CL_SRC)


_FORBIDDEN_TOP_HEADS = frozenset(
    (
        "metagpt",
        "openai",
        "anthropic",
        "litellm",
        "requests",
        "httpx",
        "asyncio",
        "time",
        "datetime",
        "random",
        "secrets",
    )
)
_B1_FORBIDDEN_MODULES = frozenset(
    (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "learning_engine",
        "core.contracts.events",
    )
)


def test_strategy_council_top_level_imports_clean() -> None:
    for name in _top_level_imports(_SC_TREE):
        head = name.split(".")[0]
        assert head not in _FORBIDDEN_TOP_HEADS, (head, name)
        assert head not in _B1_FORBIDDEN_MODULES, (head, name)
        # Also no evolution_engine cross-import inside the council.
        assert head != "evolution_engine", name


def test_critique_loop_top_level_imports_clean() -> None:
    for name in _top_level_imports(_CL_TREE):
        head = name.split(".")[0]
        assert head not in _FORBIDDEN_TOP_HEADS, (head, name)
        assert head not in _B1_FORBIDDEN_MODULES, (head, name)


def test_no_wall_clock_or_prng_calls_in_modules() -> None:
    forbidden_attrs = {
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("random", "random"),
        ("random", "randint"),
    }
    for tree in (_SC_TREE, _CL_TREE):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                pair = (node.value.id, node.attr)
                assert pair not in forbidden_attrs, pair


def test_strategy_council_no_typed_event_constructors() -> None:
    # B27/B28/INV-71 — the council is on the intelligence-engine /
    # agents side of the boundary and must never construct typed
    # bus events.
    forbidden_calls = frozenset(
        (
            "SignalEvent",
            "ExecutionIntent",
            "PatchProposal",
            "GovernanceDecision",
            "ExecutionResult",
        )
    )
    for tree in (_SC_TREE, _CL_TREE):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, node.func.id


# ---------------------------------------------------------------------------
# RoleSpec
# ---------------------------------------------------------------------------


def _critic_spec(
    role: CriticRole,
    *,
    keys: tuple[str, ...] | None = None,
) -> RoleSpec:
    if keys is None:
        if role is CriticRole.ARBITER:
            keys = ("recommendation", "rationale", "refinements")
        else:
            keys = ("verdict", "notes")
    return RoleSpec(
        role=role,
        goal=f"goal for {role.value}",
        profile=f"profile for {role.value}",
        payload_keys=keys,
    )


@pytest.mark.parametrize("role", list(CriticRole))
def test_role_spec_round_trip(role: CriticRole) -> None:
    spec = _critic_spec(role)
    assert spec.role is role


def test_role_spec_rejects_bad_goal() -> None:
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.SIGNAL_ANALYST,
            goal="",
            profile="x",
            payload_keys=("verdict",),
        )


def test_role_spec_rejects_bad_profile() -> None:
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.SIGNAL_ANALYST,
            goal="x",
            profile="   ",
            payload_keys=("verdict",),
        )


def test_role_spec_rejects_empty_payload_keys() -> None:
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.SIGNAL_ANALYST,
            goal="x",
            profile="x",
            payload_keys=(),
        )


def test_role_spec_rejects_duplicate_payload_keys() -> None:
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.SIGNAL_ANALYST,
            goal="x",
            profile="x",
            payload_keys=("a", "a"),
        )


def test_role_spec_rejects_non_identifier_payload_key() -> None:
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.SIGNAL_ANALYST,
            goal="x",
            profile="x",
            payload_keys=("1bad",),
        )


def test_role_spec_arbiter_requires_reserved_keys() -> None:
    # ARBITER must declare both ``recommendation`` and ``rationale``.
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.ARBITER,
            goal="x",
            profile="x",
            payload_keys=("recommendation",),
        )
    with pytest.raises(RoleSpecError):
        RoleSpec(
            role=CriticRole.ARBITER,
            goal="x",
            profile="x",
            payload_keys=("rationale",),
        )


# ---------------------------------------------------------------------------
# CriticMessage
# ---------------------------------------------------------------------------


def test_critic_message_rejects_non_mapping_payload() -> None:
    with pytest.raises(MessageError):
        CriticMessage(
            sender_role=CriticRole.SIGNAL_ANALYST,
            payload=[("verdict", "ok")],  # type: ignore[arg-type]
        )


def test_critic_message_rejects_empty_payload() -> None:
    with pytest.raises(MessageError):
        CriticMessage(
            sender_role=CriticRole.SIGNAL_ANALYST,
            payload={},
        )


def test_critic_message_rejects_non_string_value() -> None:
    with pytest.raises(MessageError):
        CriticMessage(
            sender_role=CriticRole.SIGNAL_ANALYST,
            payload={"v": 42},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# Deterministic StructuredSpeaker for tests
# ---------------------------------------------------------------------------


class _DeterministicSpeaker:
    """Speaker that emits a fixed structured payload per role,
    optionally driven by a script."""

    def __init__(
        self,
        script: Mapping[CriticRole, Mapping[str, str]],
    ) -> None:
        self._script = dict(script)
        self.calls: list[CriticRole] = []

    def speak(
        self,
        spec: RoleSpec,
        proposal: Mapping[str, str],
        prior: tuple[CriticMessage, ...],
    ) -> Mapping[str, str]:
        self.calls.append(spec.role)
        return self._script[spec.role]


def _full_bundle() -> dict[CriticRole, RoleSpec]:
    return {role: _critic_spec(role) for role in CANONICAL_CRITIC_ROLES}


def _approving_script() -> dict[CriticRole, dict[str, str]]:
    return {
        CriticRole.SIGNAL_ANALYST: {
            "verdict": "alpha intact",
            "notes": "no leakage",
        },
        CriticRole.RISK_OFFICER: {
            "verdict": "size within band",
            "notes": "dd ok",
        },
        CriticRole.REGIME_EXPERT: {
            "verdict": "fits trend regime",
            "notes": "vol stable",
        },
        CriticRole.ARBITER: {
            "recommendation": "APPROVE",
            "rationale": "all three critics approved",
            "refinements": "",
        },
    }


# ---------------------------------------------------------------------------
# run_critique_round — happy path
# ---------------------------------------------------------------------------


def test_run_critique_round_returns_typed_log_and_consensus() -> None:
    speaker = _DeterministicSpeaker(_approving_script())
    log, consensus = run_critique_round(
        proposal_id="p-1",
        proposal={"strategy": "trend"},
        role_bundle=_full_bundle(),
        speaker=speaker,
    )
    assert isinstance(log, CritiqueLog)
    assert isinstance(consensus, CritiqueConsensus)
    assert consensus.proposal_id == "p-1"
    assert consensus.recommendation is CritiqueRecommendation.APPROVE
    assert speaker.calls == list(CANONICAL_CRITIC_ROLES)


def test_run_critique_round_orders_messages_canonically() -> None:
    speaker = _DeterministicSpeaker(_approving_script())
    log, _ = run_critique_round(
        proposal_id="p-1",
        proposal={"x": "y"},
        role_bundle=_full_bundle(),
        speaker=speaker,
    )
    assert tuple(m.sender_role for m in log.messages) == (
        CriticRole.SIGNAL_ANALYST,
        CriticRole.RISK_OFFICER,
        CriticRole.REGIME_EXPERT,
        CriticRole.ARBITER,
    )


def test_run_critique_round_orders_payload_keys_canonically() -> None:
    # Speaker returns keys in **reverse** order; the round must
    # re-emit them in :attr:`RoleSpec.payload_keys` declaration
    # order so identical inputs produce byte-identical messages.
    script = _approving_script()
    script[CriticRole.SIGNAL_ANALYST] = {
        "notes": "no leakage",
        "verdict": "alpha intact",
    }
    speaker = _DeterministicSpeaker(script)
    log, _ = run_critique_round(
        proposal_id="p",
        proposal={"x": "y"},
        role_bundle=_full_bundle(),
        speaker=speaker,
    )
    msg = log.by_role(CriticRole.SIGNAL_ANALYST)
    assert list(msg.payload.keys()) == ["verdict", "notes"]


def test_run_critique_round_rejects_schema_mismatch() -> None:
    script = _approving_script()
    # Drop a required key from the SIGNAL_ANALYST reply.
    script[CriticRole.SIGNAL_ANALYST] = {"verdict": "ok"}
    speaker = _DeterministicSpeaker(script)
    with pytest.raises(MessageError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=speaker,
        )


def test_run_critique_round_rejects_extra_keys() -> None:
    script = _approving_script()
    script[CriticRole.SIGNAL_ANALYST] = {
        "verdict": "ok",
        "notes": "ok",
        "extra": "leaked",
    }
    speaker = _DeterministicSpeaker(script)
    with pytest.raises(MessageError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=speaker,
        )


def test_run_critique_round_rejects_missing_role_in_bundle() -> None:
    bundle = _full_bundle()
    del bundle[CriticRole.RISK_OFFICER]
    with pytest.raises(RoleSpecError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=bundle,
            speaker=_DeterministicSpeaker(_approving_script()),
        )


def test_run_critique_round_rejects_non_string_proposal_value() -> None:
    with pytest.raises(StrategyCouncilError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": 42},  # type: ignore[dict-item]
            role_bundle=_full_bundle(),
            speaker=_DeterministicSpeaker(_approving_script()),
        )


def test_run_critique_round_rejects_bad_speaker() -> None:
    with pytest.raises(StrategyCouncilError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker="not a speaker",  # type: ignore[arg-type]
        )


def test_run_critique_round_arbiter_invalid_recommendation() -> None:
    script = _approving_script()
    script[CriticRole.ARBITER] = {
        "recommendation": "MAYBE",
        "rationale": "",
        "refinements": "",
    }
    speaker = _DeterministicSpeaker(script)
    with pytest.raises(ConsensusError):
        run_critique_round(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=speaker,
        )


def test_run_critique_round_consensus_parses_refinements() -> None:
    script = _approving_script()
    script[CriticRole.ARBITER] = {
        "recommendation": "REFINE",
        "rationale": "tighten stop",
        "refinements": "raise stop to 1.5R\nadd ATR filter",
    }
    speaker = _DeterministicSpeaker(script)
    _, consensus = run_critique_round(
        proposal_id="p",
        proposal={"x": "y"},
        role_bundle=_full_bundle(),
        speaker=speaker,
    )
    assert consensus.recommendation is CritiqueRecommendation.REFINE
    assert consensus.refinements == (
        "raise stop to 1.5R",
        "add ATR filter",
    )


def test_speaker_sees_prior_messages_in_canonical_order() -> None:
    seen: list[tuple[CriticRole, tuple[CriticRole, ...]]] = []

    class _Recorder:
        def speak(
            self,
            spec: RoleSpec,
            proposal: Mapping[str, str],
            prior: tuple[CriticMessage, ...],
        ) -> Mapping[str, str]:
            seen.append(
                (
                    spec.role,
                    tuple(m.sender_role for m in prior),
                )
            )
            return _approving_script()[spec.role]

    run_critique_round(
        proposal_id="p",
        proposal={"x": "y"},
        role_bundle=_full_bundle(),
        speaker=_Recorder(),
    )
    assert seen[0] == (CriticRole.SIGNAL_ANALYST, ())
    assert seen[1] == (
        CriticRole.RISK_OFFICER,
        (CriticRole.SIGNAL_ANALYST,),
    )
    assert seen[2] == (
        CriticRole.REGIME_EXPERT,
        (CriticRole.SIGNAL_ANALYST, CriticRole.RISK_OFFICER),
    )
    assert seen[3] == (
        CriticRole.ARBITER,
        (
            CriticRole.SIGNAL_ANALYST,
            CriticRole.RISK_OFFICER,
            CriticRole.REGIME_EXPERT,
        ),
    )


# ---------------------------------------------------------------------------
# run_critique_round — INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def _digest_log(log: CritiqueLog) -> bytes:
    body_parts: list[str] = [log.proposal_id]
    for m in log.messages:
        body_parts.append(m.sender_role.value)
        for k in sorted(m.payload.keys()):
            body_parts.append(f"{k}={m.payload[k]}")
    return hashlib.blake2b("|".join(body_parts).encode(), digest_size=16).digest()


def test_run_critique_round_inv15_three_run_byte_identical() -> None:
    digests: list[bytes] = []
    for _ in range(3):
        speaker = _DeterministicSpeaker(_approving_script())
        log, _ = run_critique_round(
            proposal_id="p-7",
            proposal={"strategy": "trend"},
            role_bundle=_full_bundle(),
            speaker=speaker,
        )
        digests.append(_digest_log(log))
    assert digests[0] == digests[1] == digests[2]


# ---------------------------------------------------------------------------
# CritiqueConsensus + CritiqueLog value-object validation
# ---------------------------------------------------------------------------


def test_critique_log_rejects_wrong_role_order() -> None:
    msgs = tuple(
        CriticMessage(
            sender_role=r,
            payload={"v": "ok"},
        )
        for r in (
            CriticRole.RISK_OFFICER,
            CriticRole.SIGNAL_ANALYST,
            CriticRole.REGIME_EXPERT,
            CriticRole.ARBITER,
        )
    )
    with pytest.raises(StrategyCouncilError):
        CritiqueLog(proposal_id="p", messages=msgs)


def test_critique_log_rejects_short_messages() -> None:
    with pytest.raises(StrategyCouncilError):
        CritiqueLog(proposal_id="p", messages=())


def test_critique_consensus_rejects_long_rationale() -> None:
    with pytest.raises(ConsensusError):
        CritiqueConsensus(
            proposal_id="p",
            recommendation=CritiqueRecommendation.APPROVE,
            rationale="x" * (sc.MAX_RATIONALE_LEN + 1),
            refinements=(),
        )


def test_critique_consensus_rejects_too_many_refinements() -> None:
    with pytest.raises(ConsensusError):
        CritiqueConsensus(
            proposal_id="p",
            recommendation=CritiqueRecommendation.REFINE,
            rationale="",
            refinements=tuple("r" for _ in range(sc.MAX_REFINEMENTS + 1)),
        )


# ---------------------------------------------------------------------------
# run_critique_loop
# ---------------------------------------------------------------------------


class _IdentityMutator:
    def mutate(
        self,
        proposal: Mapping[str, str],
        consensus: CritiqueConsensus,
    ) -> Mapping[str, str]:
        return dict(proposal)


class _RefineUntilApprove:
    """Speaker that REFINEs N times then APPROVEs."""

    def __init__(self, refine_count: int) -> None:
        self.refine_count = refine_count
        self._calls = 0

    def speak(
        self,
        spec: RoleSpec,
        proposal: Mapping[str, str],
        prior: tuple[CriticMessage, ...],
    ) -> Mapping[str, str]:
        # The arbiter is the last call in each round. Count
        # arbiter visits to know which round we're in.
        if spec.role is CriticRole.ARBITER:
            self._calls += 1
            if self._calls <= self.refine_count:
                return {
                    "recommendation": "REFINE",
                    "rationale": f"round {self._calls}: tighten",
                    "refinements": "tweak the stop",
                }
            return {
                "recommendation": "APPROVE",
                "rationale": "converged",
                "refinements": "",
            }
        return _approving_script()[spec.role]


def test_run_critique_loop_terminates_on_approve() -> None:
    speaker = _DeterministicSpeaker(_approving_script())
    result = run_critique_loop(
        proposal_id="p",
        proposal={"strategy": "trend"},
        role_bundle=_full_bundle(),
        speaker=speaker,
        mutator=_IdentityMutator(),
        max_rounds=4,
    )
    assert isinstance(result, CritiqueLoopResult)
    assert result.terminal_recommendation is CritiqueRecommendation.APPROVE
    assert result.converged is True
    assert len(result.rounds) == 1


def test_run_critique_loop_iterates_until_approve() -> None:
    speaker = _RefineUntilApprove(refine_count=2)
    mutator_calls: list[tuple[CritiqueRecommendation, ...]] = []

    class _Counting(_IdentityMutator):
        def mutate(
            self,
            proposal: Mapping[str, str],
            consensus: CritiqueConsensus,
        ) -> Mapping[str, str]:
            mutator_calls.append((consensus.recommendation,))
            return dict(proposal)

    result = run_critique_loop(
        proposal_id="p",
        proposal={"strategy": "trend"},
        role_bundle=_full_bundle(),
        speaker=speaker,
        mutator=_Counting(),
        max_rounds=4,
    )
    assert result.converged is True
    assert result.terminal_recommendation is CritiqueRecommendation.APPROVE
    assert len(result.rounds) == 3
    # Mutator runs between rounds — once after each REFINE, never
    # after the terminal APPROVE.
    assert mutator_calls == [
        (CritiqueRecommendation.REFINE,),
        (CritiqueRecommendation.REFINE,),
    ]


def test_run_critique_loop_max_rounds_exhausted_returns_refine() -> None:
    speaker = _RefineUntilApprove(refine_count=10)
    result = run_critique_loop(
        proposal_id="p",
        proposal={"strategy": "trend"},
        role_bundle=_full_bundle(),
        speaker=speaker,
        mutator=_IdentityMutator(),
        max_rounds=3,
    )
    assert result.converged is False
    assert result.terminal_recommendation is CritiqueRecommendation.REFINE
    assert len(result.rounds) == 3


def test_run_critique_loop_threads_mutator_into_subsequent_rounds() -> None:
    seen_proposals: list[Mapping[str, str]] = []

    class _Recording:
        def __init__(self) -> None:
            self.calls = 0

        def speak(
            self,
            spec: RoleSpec,
            proposal: Mapping[str, str],
            prior: tuple[CriticMessage, ...],
        ) -> Mapping[str, str]:
            if spec.role is CriticRole.SIGNAL_ANALYST:
                seen_proposals.append(dict(proposal))
            if spec.role is CriticRole.ARBITER:
                self.calls += 1
                if self.calls < 2:
                    return {
                        "recommendation": "REFINE",
                        "rationale": "again",
                        "refinements": "delta",
                    }
                return {
                    "recommendation": "APPROVE",
                    "rationale": "",
                    "refinements": "",
                }
            return _approving_script()[spec.role]

    class _BumpVersion:
        def mutate(
            self,
            proposal: Mapping[str, str],
            consensus: CritiqueConsensus,
        ) -> Mapping[str, str]:
            return {
                "strategy": proposal["strategy"],
                "version": str(int(proposal["version"]) + 1),
            }

    result = run_critique_loop(
        proposal_id="p",
        proposal={"strategy": "trend", "version": "0"},
        role_bundle=_full_bundle(),
        speaker=_Recording(),
        mutator=_BumpVersion(),
        max_rounds=4,
    )
    assert seen_proposals[0]["version"] == "0"
    assert seen_proposals[1]["version"] == "1"
    assert result.final_proposal["version"] == "1"
    assert result.initial_proposal["version"] == "0"


def test_run_critique_loop_inv15_three_run_byte_identical() -> None:
    digests: list[bytes] = []
    for _ in range(3):
        result = run_critique_loop(
            proposal_id="p-9",
            proposal={"strategy": "trend"},
            role_bundle=_full_bundle(),
            speaker=_RefineUntilApprove(refine_count=2),
            mutator=_IdentityMutator(),
            max_rounds=4,
        )
        body_parts: list[str] = [
            result.proposal_id,
            result.terminal_recommendation.value,
            str(result.converged),
            str(len(result.rounds)),
        ]
        for record in result.rounds:
            body_parts.append(str(record.round_index))
            for k in sorted(record.proposal.keys()):
                body_parts.append(f"{k}={record.proposal[k]}")
            body_parts.append(_digest_log(record.log).hex())
        digests.append(hashlib.blake2b("|".join(body_parts).encode(), digest_size=16).digest())
    assert digests[0] == digests[1] == digests[2]


# ---------------------------------------------------------------------------
# Mutator + loop guard rails
# ---------------------------------------------------------------------------


def test_run_critique_loop_rejects_mutator_schema_drift() -> None:
    speaker = _RefineUntilApprove(refine_count=10)

    class _DriftingMutator:
        def mutate(
            self,
            proposal: Mapping[str, str],
            consensus: CritiqueConsensus,
        ) -> Mapping[str, str]:
            out = dict(proposal)
            out["leaked"] = "yes"
            return out

    with pytest.raises(MutationError):
        run_critique_loop(
            proposal_id="p",
            proposal={"strategy": "trend"},
            role_bundle=_full_bundle(),
            speaker=speaker,
            mutator=_DriftingMutator(),
            max_rounds=3,
        )


def test_run_critique_loop_rejects_mutator_non_string_value() -> None:
    speaker = _RefineUntilApprove(refine_count=10)

    class _BadMutator:
        def mutate(
            self,
            proposal: Mapping[str, str],
            consensus: CritiqueConsensus,
        ) -> Mapping[str, str]:
            return {"strategy": 42}  # type: ignore[dict-item]

    with pytest.raises(MutationError):
        run_critique_loop(
            proposal_id="p",
            proposal={"strategy": "trend"},
            role_bundle=_full_bundle(),
            speaker=speaker,
            mutator=_BadMutator(),
            max_rounds=3,
        )


def test_run_critique_loop_rejects_bad_max_rounds() -> None:
    with pytest.raises(CritiqueLoopError):
        run_critique_loop(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=_DeterministicSpeaker(_approving_script()),
            mutator=_IdentityMutator(),
            max_rounds=0,
        )
    with pytest.raises(CritiqueLoopError):
        run_critique_loop(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=_DeterministicSpeaker(_approving_script()),
            mutator=_IdentityMutator(),
            max_rounds=999,
        )
    with pytest.raises(CritiqueLoopError):
        run_critique_loop(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=_DeterministicSpeaker(_approving_script()),
            mutator=_IdentityMutator(),
            max_rounds=True,  # type: ignore[arg-type]
        )


def test_run_critique_loop_rejects_bad_speaker_or_mutator() -> None:
    with pytest.raises(CritiqueLoopError):
        run_critique_loop(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker="bad",  # type: ignore[arg-type]
            mutator=_IdentityMutator(),
            max_rounds=2,
        )
    with pytest.raises(CritiqueLoopError):
        run_critique_loop(
            proposal_id="p",
            proposal={"x": "y"},
            role_bundle=_full_bundle(),
            speaker=_DeterministicSpeaker(_approving_script()),
            mutator="bad",  # type: ignore[arg-type]
            max_rounds=2,
        )


def test_critique_round_record_validates_index() -> None:
    speaker = _DeterministicSpeaker(_approving_script())
    log, consensus = run_critique_round(
        proposal_id="p",
        proposal={"x": "y"},
        role_bundle=_full_bundle(),
        speaker=speaker,
    )
    rec = CritiqueRoundRecord(
        round_index=0,
        proposal={"x": "y"},
        log=log,
        consensus=consensus,
    )
    assert rec.round_index == 0
    with pytest.raises(CritiqueLoopError):
        CritiqueRoundRecord(
            round_index=-1,
            proposal={"x": "y"},
            log=log,
            consensus=consensus,
        )


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_DeterministicSpeaker(_approving_script()), StructuredSpeaker)
    assert isinstance(_IdentityMutator(), ProposalMutator)
