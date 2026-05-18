"""A-17 — Tests for governance invariant verifier (z3-solver adapter)."""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Mapping
from pathlib import Path

import pytest

from governance_engine.control_plane.invariant_verifier import (
    DEFAULT_SMT_RANDOM_SEED,
    DEFAULT_SOLVER_TIMEOUT_MS,
    INVARIANT_AUTONOMY_ESCALATION,
    INVARIANT_NO_GOVERNANCE_BYPASS,
    INVARIANT_POSITION_LIMIT,
    NEW_PIP_DEPENDENCIES,
    AutonomyEscalationProblem,
    GovernanceBypassProblem,
    InProcessSMTBackend,
    InvariantVerifier,
    PositionLimitProblem,
    SMTBackend,
    SolverResult,
    SolverVerdict,
    VerificationReport,
    VerificationStatus,
    z3_backend_factory,
)

_MODULE_PATH = Path("governance_engine/control_plane/invariant_verifier.py")


# ----------------------------------------------------------------------
# Problem validation
# ----------------------------------------------------------------------


class TestPositionLimitProblem:
    def test_round_trip(self) -> None:
        p = PositionLimitProblem(
            max_position=1000.0,
            max_leverage=2.0,
            exposure_cap=2500.0,
        )
        assert p.max_position == 1000.0
        assert p.max_leverage == 2.0
        assert p.exposure_cap == 2500.0

    @pytest.mark.parametrize(
        "max_position,max_leverage,exposure_cap",
        [
            (0.0, 2.0, 100.0),
            (-1.0, 2.0, 100.0),
            (100.0, 0.5, 100.0),
            (100.0, 2.0, 0.0),
            (100.0, 2.0, -10.0),
        ],
    )
    def test_invalid(
        self,
        max_position: float,
        max_leverage: float,
        exposure_cap: float,
    ) -> None:
        with pytest.raises(ValueError):
            PositionLimitProblem(
                max_position=max_position,
                max_leverage=max_leverage,
                exposure_cap=exposure_cap,
            )

    def test_type_check(self) -> None:
        with pytest.raises(TypeError):
            PositionLimitProblem(
                max_position=True,  # type: ignore[arg-type]
                max_leverage=2.0,
                exposure_cap=100.0,
            )

    def test_frozen(self) -> None:
        p = PositionLimitProblem(
            max_position=10.0,
            max_leverage=2.0,
            exposure_cap=100.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.max_position = 5.0  # type: ignore[misc]


class TestAutonomyEscalationProblem:
    def test_round_trip(self) -> None:
        p = AutonomyEscalationProblem(
            mode_ranks=(0, 1, 3, 4, 5),
            allowed_edges=(
                (0, 1),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (4, 0),
            ),
        )
        assert p.mode_ranks == (0, 1, 3, 4, 5)

    def test_unsorted_ranks_rejected(self) -> None:
        with pytest.raises(ValueError):
            AutonomyEscalationProblem(
                mode_ranks=(5, 0, 1),
                allowed_edges=(),
            )

    def test_unknown_edge_rejected(self) -> None:
        with pytest.raises(ValueError):
            AutonomyEscalationProblem(
                mode_ranks=(0, 1),
                allowed_edges=((0, 2),),
            )

    def test_empty_ranks_rejected(self) -> None:
        with pytest.raises(ValueError):
            AutonomyEscalationProblem(
                mode_ranks=(),
                allowed_edges=(),
            )


class TestGovernanceBypassProblem:
    def test_round_trip(self) -> None:
        p = GovernanceBypassProblem(
            nodes=("execution", "governance", "intelligence", "operator"),
            edges=(
                ("operator", "governance"),
                ("governance", "execution"),
                ("intelligence", "governance"),
            ),
            governance_nodes=("governance",),
            source="operator",
            sink="execution",
        )
        assert p.source == "operator"

    def test_unsorted_nodes_rejected(self) -> None:
        with pytest.raises(ValueError):
            GovernanceBypassProblem(
                nodes=("z", "a"),
                edges=(),
                governance_nodes=(),
                source="a",
                sink="z",
            )

    def test_source_is_governance_rejected(self) -> None:
        with pytest.raises(ValueError):
            GovernanceBypassProblem(
                nodes=("gov", "sink"),
                edges=(("gov", "sink"),),
                governance_nodes=("gov",),
                source="gov",
                sink="sink",
            )

    def test_unknown_governance_rejected(self) -> None:
        with pytest.raises(ValueError):
            GovernanceBypassProblem(
                nodes=("a", "b"),
                edges=(),
                governance_nodes=("c",),
                source="a",
                sink="b",
            )


# ----------------------------------------------------------------------
# InProcessSMTBackend behaviour
# ----------------------------------------------------------------------


class TestInProcessPositionLimit:
    def test_safe_returns_unsat(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=2.0,
                exposure_cap=500.0,
            )
        )
        assert result.verdict is SolverVerdict.UNSAT
        assert result.counterexample == {}

    def test_violation_returns_sat_with_witness(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=10.0,
                exposure_cap=500.0,
            )
        )
        assert result.verdict is SolverVerdict.SAT
        assert result.counterexample["leverage"] == "10.0"
        assert result.counterexample["position"] == "100.0"
        assert result.counterexample["product"] == "1000.0"

    def test_boundary_exactly_safe(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=5.0,
                exposure_cap=500.0,
            )
        )
        assert result.verdict is SolverVerdict.UNSAT


class TestInProcessAutonomyEscalation:
    def test_single_step_promotions_pass(self) -> None:
        backend = InProcessSMTBackend()
        # Mirrors the canonical SystemMode lattice (SAFE/PAPER/CANARY/LIVE/AUTO)
        # encoded as ranks 0,1,3,4,5 — the rank=2 gap is intentional, but
        # promotions are still single-step in the lattice (1->3 spans the gap).
        # For this invariant we model rank-adjacency on the *declared* ranks,
        # so use a dense rank space for the legality check itself.
        result = backend.check_autonomy_escalation(
            AutonomyEscalationProblem(
                mode_ranks=(0, 1, 2, 3, 4),
                allowed_edges=(
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 0),  # emergency demotion
                    (3, 0),
                ),
            )
        )
        assert result.verdict is SolverVerdict.UNSAT

    def test_skip_rank_promotion_fails(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_autonomy_escalation(
            AutonomyEscalationProblem(
                mode_ranks=(0, 1, 2, 3),
                allowed_edges=((0, 1), (0, 3)),
            )
        )
        assert result.verdict is SolverVerdict.SAT
        assert result.counterexample == {
            "from_rank": "0",
            "to_rank": "3",
        }

    def test_self_loops_ignored(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_autonomy_escalation(
            AutonomyEscalationProblem(
                mode_ranks=(0, 1, 2),
                allowed_edges=((0, 0), (1, 1), (0, 1), (1, 2)),
            )
        )
        assert result.verdict is SolverVerdict.UNSAT


class TestInProcessGovernanceBypass:
    def test_must_pass_through_governance(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_governance_bypass(
            GovernanceBypassProblem(
                nodes=("execution", "governance", "operator"),
                edges=(
                    ("operator", "governance"),
                    ("governance", "execution"),
                ),
                governance_nodes=("governance",),
                source="operator",
                sink="execution",
            )
        )
        assert result.verdict is SolverVerdict.UNSAT

    def test_bypass_path_violates(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_governance_bypass(
            GovernanceBypassProblem(
                nodes=("execution", "governance", "intelligence", "operator"),
                edges=(
                    ("intelligence", "execution"),  # bypass!
                    ("operator", "governance"),
                    ("governance", "execution"),
                ),
                governance_nodes=("governance",),
                source="intelligence",
                sink="execution",
            )
        )
        assert result.verdict is SolverVerdict.SAT
        assert result.counterexample["from"] == "intelligence"
        assert result.counterexample["to"] == "execution"
        assert result.counterexample["path"] == "intelligence->execution"

    def test_no_route_returns_unsat(self) -> None:
        backend = InProcessSMTBackend()
        result = backend.check_governance_bypass(
            GovernanceBypassProblem(
                nodes=("a", "b", "c"),
                edges=(("a", "b"),),
                governance_nodes=("c",),
                source="a",
                sink="c",
            )
        )
        # No edge into c at all, so no bypass exists.
        assert result.verdict is SolverVerdict.UNSAT

    def test_governance_sink_allowed_as_terminal(self) -> None:
        backend = InProcessSMTBackend()
        # When sink itself is governance, the only "bypass" path
        # would be source->...->sink without crossing any earlier
        # governance hop. The encoding allows sink to terminate.
        result = backend.check_governance_bypass(
            GovernanceBypassProblem(
                nodes=("a", "gov"),
                edges=(("a", "gov"),),
                governance_nodes=("gov",),
                source="a",
                sink="gov",
            )
        )
        assert result.verdict is SolverVerdict.SAT


# ----------------------------------------------------------------------
# InvariantVerifier projection
# ----------------------------------------------------------------------


class TestInvariantVerifier:
    def test_default_backend_is_in_process(self) -> None:
        verifier = InvariantVerifier()
        assert isinstance(verifier.backend, InProcessSMTBackend)

    def test_position_limit_holds(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=2.0,
                exposure_cap=500.0,
            )
        )
        assert report.invariant_id == INVARIANT_POSITION_LIMIT
        assert report.status is VerificationStatus.HOLDS
        assert report.holds is True
        assert report.counterexample == {}

    def test_position_limit_violated_carries_witness(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=10.0,
                exposure_cap=500.0,
            )
        )
        assert report.status is VerificationStatus.VIOLATED
        assert report.holds is False
        assert set(report.counterexample.keys()) == {
            "leverage",
            "position",
            "product",
        }

    def test_counterexample_keys_sorted(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_position_limit(
            PositionLimitProblem(
                max_position=100.0,
                max_leverage=10.0,
                exposure_cap=500.0,
            )
        )
        keys = list(report.counterexample.keys())
        assert keys == sorted(keys)

    def test_autonomy_escalation_holds(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_autonomy_escalation(
            AutonomyEscalationProblem(
                mode_ranks=(0, 1, 2),
                allowed_edges=((0, 1), (1, 2), (2, 0)),
            )
        )
        assert report.status is VerificationStatus.HOLDS
        assert report.invariant_id == INVARIANT_AUTONOMY_ESCALATION

    def test_autonomy_escalation_violated(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_autonomy_escalation(
            AutonomyEscalationProblem(
                mode_ranks=(0, 1, 2, 3),
                allowed_edges=((0, 1), (1, 3)),
            )
        )
        assert report.status is VerificationStatus.VIOLATED

    def test_no_governance_bypass_holds(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_no_governance_bypass(
            GovernanceBypassProblem(
                nodes=("execution", "governance", "operator"),
                edges=(
                    ("operator", "governance"),
                    ("governance", "execution"),
                ),
                governance_nodes=("governance",),
                source="operator",
                sink="execution",
            )
        )
        assert report.status is VerificationStatus.HOLDS
        assert report.invariant_id == INVARIANT_NO_GOVERNANCE_BYPASS

    def test_no_governance_bypass_violated(self) -> None:
        verifier = InvariantVerifier()
        report = verifier.verify_no_governance_bypass(
            GovernanceBypassProblem(
                nodes=("execution", "governance", "intel", "operator"),
                edges=(
                    ("intel", "execution"),
                    ("operator", "governance"),
                    ("governance", "execution"),
                ),
                governance_nodes=("governance",),
                source="intel",
                sink="execution",
            )
        )
        assert report.status is VerificationStatus.VIOLATED

    def test_unknown_verdict_maps_to_unknown(self) -> None:
        class _StubBackend:
            def check_position_limit(self, problem: PositionLimitProblem) -> SolverResult:
                return SolverResult(verdict=SolverVerdict.UNKNOWN)

            def check_autonomy_escalation(self, problem: AutonomyEscalationProblem) -> SolverResult:
                return SolverResult(verdict=SolverVerdict.UNKNOWN)

            def check_governance_bypass(self, problem: GovernanceBypassProblem) -> SolverResult:
                return SolverResult(verdict=SolverVerdict.UNKNOWN)

        verifier = InvariantVerifier(backend=_StubBackend())
        report = verifier.verify_position_limit(
            PositionLimitProblem(
                max_position=10.0,
                max_leverage=2.0,
                exposure_cap=100.0,
            )
        )
        assert report.status is VerificationStatus.UNKNOWN
        assert report.holds is False


# ----------------------------------------------------------------------
# Reports + serialisation
# ----------------------------------------------------------------------


class TestVerificationReport:
    def test_frozen(self) -> None:
        rep = VerificationReport(
            invariant_id="X",
            status=VerificationStatus.HOLDS,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rep.status = VerificationStatus.VIOLATED  # type: ignore[misc]

    def test_invariant_id_required(self) -> None:
        with pytest.raises(ValueError):
            VerificationReport(
                invariant_id="",
                status=VerificationStatus.HOLDS,
            )

    def test_counterexample_keys_must_be_str(self) -> None:
        with pytest.raises(ValueError):
            VerificationReport(
                invariant_id="X",
                status=VerificationStatus.VIOLATED,
                counterexample={"": "v"},
            )


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def _harvest(
    verifier: InvariantVerifier,
) -> tuple[VerificationReport, VerificationReport, VerificationReport]:
    a = verifier.verify_position_limit(
        PositionLimitProblem(
            max_position=100.0,
            max_leverage=10.0,
            exposure_cap=500.0,
        )
    )
    b = verifier.verify_autonomy_escalation(
        AutonomyEscalationProblem(
            mode_ranks=(0, 1, 2, 3),
            allowed_edges=((0, 1), (1, 3), (3, 0)),
        )
    )
    c = verifier.verify_no_governance_bypass(
        GovernanceBypassProblem(
            nodes=("execution", "governance", "intel", "operator"),
            edges=(
                ("intel", "execution"),
                ("operator", "governance"),
                ("governance", "execution"),
            ),
            governance_nodes=("governance",),
            source="intel",
            sink="execution",
        )
    )
    return a, b, c


def test_replay_byte_identical_3_runs() -> None:
    run_a = _harvest(InvariantVerifier())
    run_b = _harvest(InvariantVerifier())
    run_c = _harvest(InvariantVerifier())
    assert run_a == run_b == run_c


# ----------------------------------------------------------------------
# z3 backend factory
# ----------------------------------------------------------------------


class TestZ3BackendFactory:
    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError):
            z3_backend_factory(timeout_ms=0)

    def test_invalid_seed(self) -> None:
        with pytest.raises(ValueError):
            z3_backend_factory(random_seed=-1)

    def test_seed_bool_rejected(self) -> None:
        with pytest.raises(ValueError):
            z3_backend_factory(random_seed=True)  # type: ignore[arg-type]

    def test_raises_without_z3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _blocked_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "z3" or name.startswith("z3."):
                raise ImportError(f"blocked: {name}")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
        # Also evict any cached z3 modules so the lazy import path
        # is forced through __import__.
        import sys

        for name in list(sys.modules):
            if name == "z3" or name.startswith("z3."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        with pytest.raises(RuntimeError, match="z3-solver not installed"):
            z3_backend_factory()


# ----------------------------------------------------------------------
# AST guards
# ----------------------------------------------------------------------


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _toplevel_import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_no_toplevel_z3_import() -> None:
    names = _toplevel_import_names(_module_tree())
    assert "z3" not in names


def test_no_engine_imports() -> None:
    names = _toplevel_import_names(_module_tree())
    forbidden = {
        "execution_engine",
        "evolution_engine",
        "intelligence_engine",
        "system_engine",
        "sensory",
    }
    assert names.isdisjoint(forbidden)


def test_no_clock_or_random_imports() -> None:
    names = _toplevel_import_names(_module_tree())
    forbidden = {
        "asyncio",
        "datetime",
        "os",
        "random",
        "secrets",
        "time",
    }
    assert names.isdisjoint(forbidden), f"forbidden imports found: {names & forbidden}"


def test_no_typed_event_construction() -> None:
    """B27 / B28 / INV-71 — verifier never constructs typed bus events."""
    tree = _module_tree()
    forbidden = {
        "PatchProposal",
        "SignalEvent",
        "HazardEvent",
        "GovernanceDecision",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                raise AssertionError(f"verifier constructs forbidden typed event: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                raise AssertionError(f"verifier constructs forbidden typed event: .{func.attr}")


def test_lazy_z3_import_is_inside_factory() -> None:
    """The only ``import z3`` must live inside the factory body."""
    tree = _module_tree()
    factory: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "z3_backend_factory":
            factory = node
            break
    assert factory is not None, "z3_backend_factory not found"

    found = False
    for sub in ast.walk(factory):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name == "z3":
                    found = True
    assert found, "Lazy import 'import z3' missing inside factory"


def test_pip_dependency_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("z3-solver",)


def test_default_timeout_is_bounded() -> None:
    assert isinstance(DEFAULT_SOLVER_TIMEOUT_MS, int)
    assert DEFAULT_SOLVER_TIMEOUT_MS > 0
    assert DEFAULT_SMT_RANDOM_SEED == 0


# ----------------------------------------------------------------------
# Backend protocol conformance
# ----------------------------------------------------------------------


def test_in_process_satisfies_protocol() -> None:
    backend: SMTBackend = InProcessSMTBackend()
    assert hasattr(backend, "check_position_limit")
    assert hasattr(backend, "check_autonomy_escalation")
    assert hasattr(backend, "check_governance_bypass")


def test_solver_result_validates() -> None:
    with pytest.raises(TypeError):
        SolverResult(verdict="SAT")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SolverResult(
            verdict=SolverVerdict.SAT,
            counterexample={"a": 5},  # type: ignore[dict-item]
        )


def _projection_pure_check(
    backend: SMTBackend,
    problem: PositionLimitProblem,
) -> Mapping[str, str]:
    return dict(backend.check_position_limit(problem).counterexample)


def test_inprocess_purity() -> None:
    backend = InProcessSMTBackend()
    p = PositionLimitProblem(
        max_position=100.0,
        max_leverage=10.0,
        exposure_cap=500.0,
    )
    a = _projection_pure_check(backend, p)
    b = _projection_pure_check(backend, p)
    assert a == b
