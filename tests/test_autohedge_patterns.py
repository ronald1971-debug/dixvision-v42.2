"""Tests for intelligence_engine/agents/autohedge_patterns.py (C-19)."""

from __future__ import annotations

import ast
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from intelligence_engine.agents.autohedge_patterns import (
    AUTOHEDGE_PATTERN_CATALOG,
    NEW_PIP_DEPENDENCIES,
    AutoHedgePatternError,
    AutoHedgePatternRole,
    AutoHedgeRole,
    autohedge_pattern_catalog,
    autohedge_role_for_dix_module,
    canonical_consensus_flow,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "intelligence_engine" / "agents" / "autohedge_patterns.py"


# ===================================================================
# AST guard tests
# ===================================================================


class TestASTGuards:
    """Verify compile-time authority constraints by inspecting the AST."""

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return _MODULE_PATH.read_text()

    @pytest.fixture(scope="class")
    def tree(self, source: str) -> ast.Module:
        return ast.parse(source, filename=str(_MODULE_PATH))

    def test_no_forbidden_runtime_imports(self, source: str) -> None:
        """B1: must not import execution_engine / governance_engine /
        system_engine / evolution_engine / learning_engine."""

        forbidden = (
            "import execution_engine",
            "import governance_engine",
            "import system_engine",
            "import evolution_engine",
            "import learning_engine",
            "from execution_engine",
            "from governance_engine",
            "from system_engine",
            "from evolution_engine",
            "from learning_engine",
        )
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                assert pattern not in stripped, (
                    f"B1 violation: '{pattern}' found in autohedge_patterns.py: {stripped}"
                )

    def test_no_typed_event_constructors(self, source: str) -> None:
        """B27/B28/INV-71: must never construct typed bus events."""

        forbidden_constructors = (
            "SignalEvent(",
            "ExecutionIntent(",
            "GovernanceDecision(",
            "PatchProposal(",
        )
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"', "'")):
                continue
            for ctor in forbidden_constructors:
                assert ctor not in stripped, f"INV-71 violation: '{ctor}' found: {stripped}"

    def test_no_wall_clock_or_prng(self, source: str) -> None:
        """INV-15: must not import random / time / datetime / secrets."""

        forbidden = (
            "import random",
            "import time",
            "import datetime",
            "import secrets",
        )
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                assert pattern not in stripped, f"INV-15 violation: '{pattern}' found: {stripped}"

    def test_b1_seam_no_top_level_framework_import(self, tree: ast.Module) -> None:
        """B1 lazy seam: no top-level ``import autohedge`` or
        ``from autohedge ...`` anywhere."""

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("autohedge"), (
                        f"B1 lazy seam: top-level 'import {alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("autohedge"), (
                    f"B1 lazy seam: top-level 'from {node.module} ...'"
                )


# ===================================================================
# Module constants
# ===================================================================


class TestModuleConstants:
    def test_new_pip_dependencies_empty(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ()

    def test_autohedge_role_enum_members(self) -> None:
        expected = {
            "MARKET_ANALYST",
            "TECHNICAL_ANALYST",
            "RISK_MANAGER",
            "PORTFOLIO_OPTIMIZER",
            "EXECUTION_MANAGER",
        }
        assert {r.value for r in AutoHedgeRole} == expected

    def test_autohedge_role_count(self) -> None:
        assert len(AutoHedgeRole) == 5

    def test_catalog_length_matches_role_count(self) -> None:
        assert len(AUTOHEDGE_PATTERN_CATALOG) == len(AutoHedgeRole)


# ===================================================================
# Value object — AutoHedgePatternRole
# ===================================================================


class TestAutoHedgePatternRole:
    def test_frozen(self) -> None:
        role = AUTOHEDGE_PATTERN_CATALOG[0]
        with pytest.raises(AttributeError):
            role.role = AutoHedgeRole.RISK_MANAGER  # type: ignore[misc]

    def test_slotted(self) -> None:
        assert hasattr(AutoHedgePatternRole, "__slots__")

    def test_fields(self) -> None:
        role = AUTOHEDGE_PATTERN_CATALOG[0]
        assert isinstance(role.role, AutoHedgeRole)
        assert isinstance(role.responsibility, str)
        assert isinstance(role.dix_module, str)
        assert isinstance(role.dix_summary, str)

    def test_reject_empty_responsibility(self) -> None:
        with pytest.raises(AutoHedgePatternError):
            AutoHedgePatternRole(
                role=AutoHedgeRole.MARKET_ANALYST,
                responsibility="   ",
                dix_module="some/path.py",
                dix_summary="something",
            )

    def test_reject_empty_dix_module(self) -> None:
        with pytest.raises(AutoHedgePatternError):
            AutoHedgePatternRole(
                role=AutoHedgeRole.MARKET_ANALYST,
                responsibility="does things",
                dix_module="",
                dix_summary="something",
            )

    def test_reject_empty_dix_summary(self) -> None:
        with pytest.raises(AutoHedgePatternError):
            AutoHedgePatternRole(
                role=AutoHedgeRole.MARKET_ANALYST,
                responsibility="does things",
                dix_module="some/path.py",
                dix_summary="   ",
            )

    def test_reject_bad_role_type(self) -> None:
        with pytest.raises(AutoHedgePatternError):
            AutoHedgePatternRole(
                role="NOT_A_REAL_ROLE",  # type: ignore[arg-type]
                responsibility="does things",
                dix_module="some/path.py",
                dix_summary="something",
            )


# ===================================================================
# Catalog
# ===================================================================


class TestCatalog:
    def test_catalog_returns_tuple(self) -> None:
        cat = autohedge_pattern_catalog()
        assert isinstance(cat, tuple)

    def test_catalog_covers_all_roles(self) -> None:
        roles = {entry.role for entry in autohedge_pattern_catalog()}
        assert roles == set(AutoHedgeRole)

    def test_catalog_unique_dix_modules(self) -> None:
        paths = [entry.dix_module for entry in autohedge_pattern_catalog()]
        assert len(set(paths)) == len(paths)

    def test_dix_module_anchors_exist(self) -> None:
        """Every dix_module path should reference a real file or
        directory in the repo."""
        for entry in autohedge_pattern_catalog():
            target = _REPO_ROOT / entry.dix_module
            assert target.exists(), f"{entry.dix_module} does not exist at {target}"


# ===================================================================
# Reverse lookup
# ===================================================================


class TestReverseLookup:
    def test_known_path_returns_role(self) -> None:
        role = autohedge_role_for_dix_module("governance_engine/control_plane/risk_evaluator.py")
        assert role is AutoHedgeRole.RISK_MANAGER

    def test_unknown_path_returns_none(self) -> None:
        assert autohedge_role_for_dix_module("nonexistent/foo.py") is None

    def test_rejects_non_string(self) -> None:
        with pytest.raises(AutoHedgePatternError):
            autohedge_role_for_dix_module(42)  # type: ignore[arg-type]

    def test_all_catalog_paths_reversible(self) -> None:
        for entry in autohedge_pattern_catalog():
            assert autohedge_role_for_dix_module(entry.dix_module) is entry.role


# ===================================================================
# Consensus flow
# ===================================================================


class TestConsensusFlow:
    def test_flow_returns_tuple(self) -> None:
        flow = canonical_consensus_flow()
        assert isinstance(flow, tuple)

    def test_flow_covers_all_roles(self) -> None:
        flow = canonical_consensus_flow()
        assert set(flow) == set(AutoHedgeRole)

    def test_flow_length_matches(self) -> None:
        flow = canonical_consensus_flow()
        assert len(flow) == len(AutoHedgeRole)

    def test_flow_order_is_fixed(self) -> None:
        flow = canonical_consensus_flow()
        expected_order = (
            AutoHedgeRole.MARKET_ANALYST,
            AutoHedgeRole.TECHNICAL_ANALYST,
            AutoHedgeRole.RISK_MANAGER,
            AutoHedgeRole.PORTFOLIO_OPTIMIZER,
            AutoHedgeRole.EXECUTION_MANAGER,
        )
        assert flow == expected_order

    def test_flow_stable_across_calls(self) -> None:
        assert canonical_consensus_flow() == canonical_consensus_flow()


# ===================================================================
# Error hierarchy
# ===================================================================


class TestErrorHierarchy:
    def test_error_is_value_error(self) -> None:
        assert issubclass(AutoHedgePatternError, ValueError)

    def test_error_instantiation(self) -> None:
        err = AutoHedgePatternError("test message")
        assert str(err) == "test message"
