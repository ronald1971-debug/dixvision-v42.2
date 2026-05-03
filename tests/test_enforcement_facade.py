"""Smoke tests for the ``enforcement`` re-export façade (P0-1c)."""

from __future__ import annotations

import enforcement


def test_authority_matrix_symbols_exposed():
    assert hasattr(enforcement, "AuthorityMatrix")
    assert hasattr(enforcement, "load_authority_matrix")
    assert hasattr(enforcement, "AuthorityActor")
    assert hasattr(enforcement, "AuthorityOverride")
    assert hasattr(enforcement, "ConflictRow")


def test_authority_guard_symbols_exposed():
    assert hasattr(enforcement, "AuthorityGuard")
    assert hasattr(enforcement, "AuthorityViolation")
    assert hasattr(enforcement, "UnauthorizedActorError")
    assert issubclass(enforcement.UnauthorizedActorError, RuntimeError)


def test_policy_engine_symbols_exposed():
    assert hasattr(enforcement, "PolicyEngine")
    assert hasattr(enforcement, "install_policy_table")
    assert hasattr(enforcement, "verify_policy_table_hash")


def test_constraint_engine_symbols_exposed():
    assert hasattr(enforcement, "RuleGraph")
    assert hasattr(enforcement, "compile_rules")
    assert hasattr(enforcement, "RuleKind")
    assert hasattr(enforcement, "RuleSeverity")
    assert hasattr(enforcement, "RuleAction")
    assert hasattr(enforcement, "CompiledRule")


def test_adaptive_freeze_symbols_exposed():
    assert hasattr(enforcement, "LearningEvolutionFreezePolicy")
    assert hasattr(enforcement, "LearningEvolutionFrozenError")
    assert hasattr(enforcement, "assert_unfrozen")
    assert hasattr(enforcement, "is_unfrozen")
    assert issubclass(enforcement.LearningEvolutionFrozenError, RuntimeError)


def test_kill_switch_symbols_exposed():
    assert hasattr(enforcement, "KillSwitch")
    assert hasattr(enforcement, "KillReason")
    assert hasattr(enforcement, "KillRequest")


def test_facade_does_not_introduce_new_state():
    """Re-importing the façade returns identical class objects.

    The package is supposed to be a thin alias — the canonical
    classes still live in their original modules. Sanity-check that
    we did not accidentally fork them.
    """

    from core.constraint_engine import RuleGraph as RuleGraphCanonical
    from core.contracts.learning_evolution_freeze import (
        LearningEvolutionFreezePolicy as FreezeCanonical,
    )
    from execution_engine.execution_gate import (
        AuthorityGuard as AuthorityGuardCanonical,
    )
    from governance_engine.control_plane.policy_engine import (
        PolicyEngine as PolicyEngineCanonical,
    )
    from system.kill_switch import KillSwitch as KillSwitchCanonical
    from system_engine.authority import (
        AuthorityMatrix as AuthorityMatrixCanonical,
    )

    assert enforcement.RuleGraph is RuleGraphCanonical
    assert enforcement.LearningEvolutionFreezePolicy is FreezeCanonical
    assert enforcement.AuthorityGuard is AuthorityGuardCanonical
    assert enforcement.PolicyEngine is PolicyEngineCanonical
    assert enforcement.KillSwitch is KillSwitchCanonical
    assert enforcement.AuthorityMatrix is AuthorityMatrixCanonical


def test_all_export_list_matches_attributes():
    for name in enforcement.__all__:
        assert hasattr(enforcement, name), f"{name} declared in __all__ but missing"
