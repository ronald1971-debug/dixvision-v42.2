"""Tests for the constraint-engine compiler + expression DSL."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.constraint_engine import (
    RuleAction,
    RuleGraph,
    RuleKind,
    RuleSeverity,
    compile_rules,
)
from core.constraint_engine import expr as expr_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES = REPO_ROOT / "registry" / "constraint_rules.yaml"
MATRIX = REPO_ROOT / "registry" / "authority_matrix.yaml"


# ---------------------------------------------------------------------------
# expression DSL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "facts", "expected"),
    [
        ("a > 5", {"a": 10}, True),
        ("a > 5", {"a": 5}, False),
        ("a >= 5", {"a": 5}, True),
        ("a < 5 or a > 10", {"a": 12}, True),
        ("a < 5 or a > 10", {"a": 7}, False),
        ("a == b", {"a": 3, "b": 3}, True),
        ("a == b", {"a": 3, "b": 4}, False),
        ("not a > 5", {"a": 1}, True),
        ("(a < 5) and (b > 0)", {"a": 1, "b": 2}, True),
        ("(a < 5) and (b > 0)", {"a": 1, "b": 0}, False),
    ],
)
def test_expr_evaluate_truth_table(src, facts, expected):
    assert expr_mod.evaluate(expr_mod.parse(src), facts) is expected


def test_expr_free_idents_collects_all_refs():
    ast = expr_mod.parse("a > 5 and (b == c or d < 10)")
    assert expr_mod.free_idents(ast) == frozenset({"a", "b", "c", "d"})


def test_expr_missing_fact_raises():
    ast = expr_mod.parse("missing > 0")
    with pytest.raises(KeyError, match="missing"):
        expr_mod.evaluate(ast, {})


def test_expr_non_numeric_fact_raises_for_ordered_op():
    ast = expr_mod.parse("x > 0")
    with pytest.raises(TypeError, match="numeric"):
        expr_mod.evaluate(ast, {"x": "not a number"})


def test_expr_string_equality_supported_for_eq_ne():
    ast_eq = expr_mod.parse("a == b")
    ast_ne = expr_mod.parse("a != b")
    assert expr_mod.evaluate(ast_eq, {"a": "x", "b": "x"}) is True
    assert expr_mod.evaluate(ast_eq, {"a": "x", "b": "y"}) is False
    assert expr_mod.evaluate(ast_ne, {"a": "x", "b": "y"}) is True


def test_expr_bool_facts_treated_as_int():
    ast = expr_mod.parse("flag == 1")
    assert expr_mod.evaluate(ast, {"flag": True}) is True
    assert expr_mod.evaluate(ast, {"flag": False}) is False


@pytest.mark.parametrize(
    "bad",
    [
        "a >",  # missing right operand
        "a 5",  # no operator
        "a > 5 and",  # trailing and
        "(a > 5",  # unclosed paren
        "a $ 5",  # illegal char
        "",  # empty
    ],
)
def test_expr_parser_rejects_malformed(bad):
    with pytest.raises(ValueError):
        expr_mod.parse(bad)


# ---------------------------------------------------------------------------
# canonical rules file
# ---------------------------------------------------------------------------


def test_canonical_rules_compile():
    g = compile_rules(RULES, matrix_path=MATRIX)
    assert isinstance(g, RuleGraph)
    assert len(g.rules) >= 20
    assert g.version


def test_canonical_rules_owners_are_matrix_actors():
    g = compile_rules(RULES, matrix_path=MATRIX)
    actor_ids = g.matrix.actor_ids
    for r in g.rules:
        assert r.owner in actor_ids, f"{r.id} owner {r.owner!r} not in matrix"


def test_canonical_rules_topo_order_respects_deps():
    g = compile_rules(RULES, matrix_path=MATRIX)
    pos = {rid: i for i, rid in enumerate(g.order)}
    for r in g.rules:
        for dep in r.depends_on:
            assert pos[dep] < pos[r.id], (
                f"{r.id} appears before its dep {dep}"
            )


def test_canonical_rules_kinds_present():
    g = compile_rules(RULES, matrix_path=MATRIX)
    kinds = {r.kind for r in g.rules}
    # The first cut covers every documented family.
    assert {
        RuleKind.INV,
        RuleKind.SAFE,
        RuleKind.HAZ,
        RuleKind.SCVS,
        RuleKind.GOV,
        RuleKind.PERF,
    } <= kinds


def test_canonical_safe09_has_predicate_and_fires():
    g = compile_rules(RULES, matrix_path=MATRIX)
    safe09 = g.get("SAFE-09")
    assert safe09.when_ast is not None
    assert safe09.severity is RuleSeverity.BLOCK
    assert safe09.action is RuleAction.HALT
    fired = g.evaluate(
        {
            "fast_risk_cache_age_ns": 10,
            "fast_risk_cache_max_age_ns": 5,
            "hot_path_latency_ns": 0,
            "hot_path_latency_budget_ns": 1,
            "ledger_commit_latency_ns": 0,
            "ledger_commit_budget_ns": 1,
            "actor": "x",
            "owner": "x",
        }
    )
    assert any(r.id == "SAFE-09" for r in fired)


def test_canonical_evaluate_is_deterministic():
    g = compile_rules(RULES, matrix_path=MATRIX)
    facts = {
        "fast_risk_cache_age_ns": 10,
        "fast_risk_cache_max_age_ns": 1,
        "hot_path_latency_ns": 100,
        "hot_path_latency_budget_ns": 50,
        "ledger_commit_latency_ns": 0,
        "ledger_commit_budget_ns": 1,
        "actor": "x",
        "owner": "x",
    }
    a = [r.id for r in g.evaluate(facts)]
    b = [r.id for r in g.evaluate(facts)]
    assert a == b


def test_rules_owned_by_helper():
    g = compile_rules(RULES, matrix_path=MATRIX)
    for actor in g.matrix.actor_ids:
        owned = g.rules_owned_by(actor)
        assert all(r.owner == actor for r in owned)


def test_rules_of_kind_helper():
    g = compile_rules(RULES, matrix_path=MATRIX)
    inv = g.rules_of_kind(RuleKind.INV)
    assert inv
    assert all(r.kind is RuleKind.INV for r in inv)


# ---------------------------------------------------------------------------
# loader rejection paths
# ---------------------------------------------------------------------------


def _good() -> dict:
    return {
        "version": "v0",
        "rules": [
            {
                "id": "A",
                "kind": "INV",
                "severity": "BLOCK",
                "action": "AUDIT",
                "owner": "governance",
                "description": "x",
            },
            {
                "id": "B",
                "kind": "SAFE",
                "severity": "BLOCK",
                "action": "REJECT",
                "owner": "governance",
                "description": "x",
                "depends_on": ["A"],
            },
        ],
    }


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_load_unknown_owner_rejected(tmp_path):
    body = _good()
    body["rules"][0]["owner"] = "alien"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="owner 'alien'"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_unknown_kind_rejected(tmp_path):
    body = _good()
    body["rules"][0]["kind"] = "ZZZ"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="unknown kind"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_unknown_severity_rejected(tmp_path):
    body = _good()
    body["rules"][0]["severity"] = "INTENSE"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="unknown severity"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_unknown_action_rejected(tmp_path):
    body = _good()
    body["rules"][0]["action"] = "NUKE"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="unknown action"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_dangling_dependency_rejected(tmp_path):
    body = _good()
    body["rules"][1]["depends_on"] = ["GHOST"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="depends on unknown rule"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_self_dependency_rejected(tmp_path):
    body = _good()
    body["rules"][0]["depends_on"] = ["A"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="self-dependency"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_dependency_cycle_rejected(tmp_path):
    body = _good()
    body["rules"][0]["depends_on"] = ["B"]
    body["rules"][1]["depends_on"] = ["A"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="cycle"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_duplicate_rule_id_rejected(tmp_path):
    body = _good()
    body["rules"].append(dict(body["rules"][0]))
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate rule id"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_invalid_when_expression_rejected(tmp_path):
    body = _good()
    body["rules"][0]["when"] = "a >"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="invalid 'when' expression"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_empty_when_rejected(tmp_path):
    body = _good()
    body["rules"][0]["when"] = "   "
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="empty 'when' clause"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_missing_required_key_rejected(tmp_path):
    body = _good()
    del body["rules"][0]["description"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="missing required key 'description'"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_missing_top_level_rules_key(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"version": "v0"}))
    with pytest.raises(ValueError, match="missing required top-level keys"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_empty_rules_list_rejected(tmp_path):
    p = _write(tmp_path, {"version": "v0", "rules": []})
    with pytest.raises(ValueError, match="non-empty list"):
        compile_rules(p, matrix_path=MATRIX)


def test_load_depends_on_not_a_list_rejected(tmp_path):
    body = _good()
    body["rules"][1]["depends_on"] = "A"  # str, not list
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="must be a list"):
        compile_rules(p, matrix_path=MATRIX)


# ---------------------------------------------------------------------------
# graph evaluation
# ---------------------------------------------------------------------------


def test_evaluate_empty_when_rules_never_fire(tmp_path):
    body = _good()  # no 'when' clauses
    p = _write(tmp_path, body)
    g = compile_rules(p, matrix_path=MATRIX)
    assert g.evaluate({}) == ()


def test_evaluate_with_predicate_fires_in_topo_order(tmp_path):
    body = _good()
    body["rules"][0]["when"] = "a > 5"
    body["rules"][1]["when"] = "b < 3"
    p = _write(tmp_path, body)
    g = compile_rules(p, matrix_path=MATRIX)
    fired = g.evaluate({"a": 10, "b": 1})
    ids = [r.id for r in fired]
    # B depends on A → A precedes B in topo order.
    assert ids == ["A", "B"]


def test_get_unknown_rule_raises():
    g = compile_rules(RULES, matrix_path=MATRIX)
    with pytest.raises(KeyError, match="unknown rule"):
        g.get("NEVER")


def test_compiled_rule_is_frozen():
    g = compile_rules(RULES, matrix_path=MATRIX)
    rule = g.rules[0]
    with pytest.raises((AttributeError, TypeError)):
        rule.id = "X"  # type: ignore[misc]
