"""A-07 — tests for ``governance_engine/services/opa_policy.py``."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from core.contracts.governance import DecisionKind, GovernanceDecision
from governance_engine.services import opa_policy
from governance_engine.services.opa_policy import (
    DEFAULT_POLICY_PACKAGE,
    MAX_PAYLOAD_KEYS,
    MAX_PAYLOAD_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    OPA_ADAPTER_VERSION,
    InProcessPolicyTransport,
    OpaPolicyError,
    OpaPolicyEvaluator,
    PolicyDecision,
    PolicyEvaluationError,
    PolicyInput,
    PolicyRule,
    PolicyTransport,
    PolicyTransportError,
    PolicyTransportResult,
    PolicyVerdict,
    build_baseline_rules,
    to_governance_decision,
)

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "governance_engine" / "services" / "opa_policy.py"
)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)
REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = REPO_ROOT / "governance_engine" / "policies"


# ---------------------------------------------------------------------------
# Module-level smoke + invariants
# ---------------------------------------------------------------------------


def test_module_constants() -> None:
    assert NEW_PIP_DEPENDENCIES == ("opa-python-client",)
    assert OPA_ADAPTER_VERSION == "1"
    assert DEFAULT_POLICY_PACKAGE == "dix.governance"
    assert MAX_PAYLOAD_KEYS == 32
    assert MAX_PAYLOAD_VALUE_LEN == 1024


def test_module_has_adapted_from_header() -> None:
    assert "# ADAPTED FROM: open-policy-agent" in MODULE_SOURCE


def test_module_has_no_top_level_opa_import() -> None:
    for node in MODULE_AST.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("opa_client"), (
                    "top-level opa_client import forbidden: " + alias.name
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("opa_client"), (
                "top-level opa_client import forbidden: " + (node.module or "")
            )


def test_module_has_no_forbidden_runtime_imports() -> None:
    """INV-15: no clock / IO / random / async / numerics frameworks."""

    forbidden = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "websockets",
        "numpy",
        "torch",
        "polars",
        "langsmith",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                assert root not in forbidden, f"forbidden import: {node.module}"


def test_module_has_no_os_top_level_import() -> None:
    """``os`` is part of the INV-15 ban — verify separately."""

    for node in MODULE_AST.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "os"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "os"


def test_module_has_no_disallowed_engine_imports() -> None:
    """B1 isolation: only ``core.contracts.governance`` is permitted."""

    disallowed_roots = {
        "system_engine",
        "execution_engine",
        "evolution_engine",
        "intelligence_engine",
        "simulation",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in disallowed_roots, f"disallowed engine import: {node.module}"


def test_module_does_not_construct_forbidden_typed_events() -> None:
    """B27/B28/INV-71: no PatchProposal / SignalEvent construction.

    ``GovernanceDecision`` is permitted because this file lives under
    ``governance_engine/`` — see ``to_governance_decision``.
    """

    forbidden = {"PatchProposal", "SignalEvent"}
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                pytest.fail(f"forbidden constructor call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                pytest.fail(f"forbidden constructor call: {func.attr}")


def test_predictor_protocol_runtime_checkable() -> None:
    assert isinstance(InProcessPolicyTransport(rules=()), PolicyTransport)


def test_lazy_factory_does_not_import_opa_at_module_load() -> None:
    """``opa_http_transport_factory`` must own the lazy import."""

    factory_src = ""
    for node in MODULE_AST.body:
        if isinstance(node, ast.FunctionDef) and node.name == "opa_http_transport_factory":
            factory_src = ast.unparse(node)
            break
    assert factory_src, "factory not found"
    assert "import opa_client" in factory_src


# ---------------------------------------------------------------------------
# PolicyInput
# ---------------------------------------------------------------------------


def _basic_input(**over: object) -> PolicyInput:
    defaults: dict[str, object] = {
        "action": "EXECUTE_ORDER",
        "mode": "LIVE",
        "subject": "operator-1",
        "payload": {"notional_usd": 1000.0, "notional_limit_usd": 10000.0},
    }
    defaults.update(over)
    return PolicyInput.from_mapping(
        action=str(defaults["action"]),
        mode=str(defaults["mode"]),
        subject=str(defaults["subject"]),
        payload=defaults["payload"],  # type: ignore[arg-type]
    )


def test_policy_input_basic() -> None:
    pi = _basic_input()
    assert pi.action == "EXECUTE_ORDER"
    assert pi.mode == "LIVE"
    assert pi.subject == "operator-1"
    assert dict(pi.payload) == {
        "notional_usd": 1000.0,
        "notional_limit_usd": 10000.0,
    }


def test_policy_input_payload_sorted() -> None:
    pi = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"z": 1, "a": 2, "m": 3},
    )
    keys = [k for k, _ in pi.payload]
    assert keys == ["a", "m", "z"]


def test_policy_input_rejects_empty_action() -> None:
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(action="", mode="LIVE", subject="s")


def test_policy_input_rejects_empty_mode() -> None:
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(action="X", mode="", subject="s")


def test_policy_input_rejects_empty_subject() -> None:
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(action="X", mode="LIVE", subject="")


def test_policy_input_rejects_non_str_action() -> None:
    with pytest.raises(TypeError):
        PolicyInput.from_mapping(
            action=1,  # type: ignore[arg-type]
            mode="LIVE",
            subject="s",
        )


def test_policy_input_rejects_oversize_action() -> None:
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(action="x" * 200, mode="LIVE", subject="s")


def test_policy_input_rejects_oversize_payload_value() -> None:
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(
            action="X",
            mode="LIVE",
            subject="s",
            payload={"k": "x" * (MAX_PAYLOAD_VALUE_LEN + 1)},
        )


def test_policy_input_rejects_too_many_keys() -> None:
    payload = {f"k{i}": i for i in range(MAX_PAYLOAD_KEYS + 1)}
    with pytest.raises(ValueError):
        PolicyInput.from_mapping(action="X", mode="LIVE", subject="s", payload=payload)


def test_policy_input_rejects_unsupported_value_type() -> None:
    with pytest.raises(TypeError):
        PolicyInput.from_mapping(
            action="X",
            mode="LIVE",
            subject="s",
            payload={"k": [1, 2, 3]},  # type: ignore[dict-item]
        )


def test_policy_input_rejects_non_mapping_payload() -> None:
    with pytest.raises(TypeError):
        PolicyInput.from_mapping(
            action="X",
            mode="LIVE",
            subject="s",
            payload=[("k", 1)],  # type: ignore[arg-type]
        )


def test_policy_input_rejects_duplicate_payload_key() -> None:
    with pytest.raises(ValueError):
        PolicyInput(
            action="X",
            mode="LIVE",
            subject="s",
            payload=(("k", 1), ("k", 2)),
        )


def test_policy_input_rejects_non_tuple_payload() -> None:
    with pytest.raises(TypeError):
        PolicyInput(
            action="X",
            mode="LIVE",
            subject="s",
            payload=[("k", 1)],  # type: ignore[arg-type]
        )


def test_policy_input_to_canonical_json_byte_stable() -> None:
    pi1 = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"a": 1, "z": 2, "m": 3},
    )
    pi2 = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"m": 3, "z": 2, "a": 1},
    )
    assert pi1.to_canonical_json() == pi2.to_canonical_json()


def test_policy_input_as_payload_round_trip() -> None:
    pi = _basic_input()
    payload = pi.as_payload()
    assert payload == {
        "notional_usd": 1000.0,
        "notional_limit_usd": 10000.0,
    }


# ---------------------------------------------------------------------------
# PolicyTransportResult
# ---------------------------------------------------------------------------


def test_transport_result_basic_approve() -> None:
    r = PolicyTransportResult(
        verdict=PolicyVerdict.APPROVE,
        policy_id="p",
        rule_path="r",
    )
    assert r.verdict is PolicyVerdict.APPROVE
    assert r.rejection_code == ""


def test_transport_result_approve_rejects_code() -> None:
    with pytest.raises(ValueError):
        PolicyTransportResult(
            verdict=PolicyVerdict.APPROVE,
            policy_id="p",
            rule_path="r",
            rejection_code="should_be_empty",
        )


def test_transport_result_reject_requires_code() -> None:
    with pytest.raises(ValueError):
        PolicyTransportResult(
            verdict=PolicyVerdict.REJECT,
            policy_id="p",
            rule_path="r",
        )


def test_transport_result_rejects_bad_verdict() -> None:
    with pytest.raises(TypeError):
        PolicyTransportResult(
            verdict="APPROVE",  # type: ignore[arg-type]
            policy_id="p",
            rule_path="r",
        )


def test_transport_result_rejects_empty_policy_id() -> None:
    with pytest.raises(ValueError):
        PolicyTransportResult(
            verdict=PolicyVerdict.APPROVE,
            policy_id="",
            rule_path="r",
        )


# ---------------------------------------------------------------------------
# PolicyRule
# ---------------------------------------------------------------------------


def _allow_predicate(_pi: PolicyInput) -> bool:
    return False


def test_policy_rule_basic() -> None:
    r = PolicyRule(
        policy_id="p",
        rule_path="r/path",
        predicate=_allow_predicate,
        rejection_code="POLICY_R",
        summary="r",
    )
    assert r.verdict_on_match is PolicyVerdict.REJECT


def test_policy_rule_rejects_empty_policy_id() -> None:
    with pytest.raises(ValueError):
        PolicyRule(
            policy_id="",
            rule_path="r",
            predicate=_allow_predicate,
            rejection_code="X",
        )


def test_policy_rule_rejects_non_callable_predicate() -> None:
    with pytest.raises(TypeError):
        PolicyRule(
            policy_id="p",
            rule_path="r",
            predicate="not-callable",  # type: ignore[arg-type]
            rejection_code="X",
        )


def test_policy_rule_reject_requires_code() -> None:
    with pytest.raises(ValueError):
        PolicyRule(
            policy_id="p",
            rule_path="r",
            predicate=_allow_predicate,
            rejection_code="",
            verdict_on_match=PolicyVerdict.REJECT,
        )


def test_policy_rule_approve_rejects_code() -> None:
    with pytest.raises(ValueError):
        PolicyRule(
            policy_id="p",
            rule_path="r",
            predicate=_allow_predicate,
            rejection_code="X",
            verdict_on_match=PolicyVerdict.APPROVE,
        )


# ---------------------------------------------------------------------------
# InProcessPolicyTransport
# ---------------------------------------------------------------------------


def test_in_process_transport_default_allow() -> None:
    t = InProcessPolicyTransport(rules=())
    out = t.evaluate(_basic_input())
    assert out.verdict is PolicyVerdict.APPROVE
    assert out.rule_path == "default/allow"


def test_in_process_transport_first_match_wins() -> None:
    matched: list[str] = []

    def _p_first(pi: PolicyInput) -> bool:
        matched.append("first")
        return True

    def _p_second(pi: PolicyInput) -> bool:
        matched.append("second")
        return True

    t = InProcessPolicyTransport(
        rules=(
            PolicyRule(
                policy_id="p1",
                rule_path="first",
                predicate=_p_first,
                rejection_code="POLICY_FIRST",
                summary="first",
            ),
            PolicyRule(
                policy_id="p2",
                rule_path="second",
                predicate=_p_second,
                rejection_code="POLICY_SECOND",
                summary="second",
            ),
        )
    )
    out = t.evaluate(_basic_input())
    assert out.policy_id == "p1"
    assert matched == ["first"]


def test_in_process_transport_predicate_error_becomes_transport_error() -> None:
    def _bad(_pi: PolicyInput) -> bool:
        raise ValueError("boom")

    t = InProcessPolicyTransport(
        rules=(
            PolicyRule(
                policy_id="p",
                rule_path="r",
                predicate=_bad,
                rejection_code="X",
            ),
        )
    )
    with pytest.raises(PolicyTransportError):
        t.evaluate(_basic_input())


def test_in_process_transport_rejects_non_policy_input() -> None:
    t = InProcessPolicyTransport(rules=())
    with pytest.raises(PolicyTransportError):
        t.evaluate({"action": "X"})  # type: ignore[arg-type]


def test_in_process_transport_rejects_duplicate_rule_keys() -> None:
    with pytest.raises(ValueError):
        InProcessPolicyTransport(
            rules=(
                PolicyRule(
                    policy_id="p",
                    rule_path="r",
                    predicate=_allow_predicate,
                    rejection_code="X",
                ),
                PolicyRule(
                    policy_id="p",
                    rule_path="r",
                    predicate=_allow_predicate,
                    rejection_code="Y",
                ),
            )
        )


def test_in_process_transport_rejects_non_tuple_rules() -> None:
    with pytest.raises(TypeError):
        InProcessPolicyTransport(rules=[])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Baseline rules
# ---------------------------------------------------------------------------


def test_build_baseline_rules_returns_three() -> None:
    rules = build_baseline_rules()
    assert len(rules) == 3
    ids = {r.policy_id for r in rules}
    assert ids == {
        "dix.governance.position_limits",
        "dix.governance.execution_gates",
        "dix.governance.autonomy_levels",
    }


def test_baseline_rejects_overlimit_position() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="EXECUTE_ORDER",
        mode="LIVE",
        subject="s",
        payload={
            "notional_usd": 50000.0,
            "notional_limit_usd": 10000.0,
        },
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.REJECT
    assert r.rejection_code == "POLICY_POSITION_LIMIT"


def test_baseline_approves_underlimit_position() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="EXECUTE_ORDER",
        mode="LIVE",
        subject="s",
        payload={
            "notional_usd": 100.0,
            "notional_limit_usd": 10000.0,
        },
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.APPROVE


def test_baseline_blocks_execute_in_safe() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="EXECUTE_ORDER",
        mode="SAFE",
        subject="s",
        payload={
            "notional_usd": 100.0,
            "notional_limit_usd": 10000.0,
        },
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.REJECT
    assert r.rejection_code == "POLICY_EXECUTION_GATE"


def test_baseline_escalates_auto_with_hazards() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="STAGE_ORDER",
        mode="AUTO",
        subject="s",
        payload={"active_hazards": 3},
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.ESCALATE
    assert r.rejection_code == "POLICY_AUTONOMY_ESCALATE"


def test_baseline_approves_auto_zero_hazards() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="STAGE_ORDER",
        mode="AUTO",
        subject="s",
        payload={"active_hazards": 0},
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.APPROVE


def test_baseline_position_rule_ignores_missing_keys() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="STAGE_ORDER",
        mode="LIVE",
        subject="s",
        payload={"unrelated": 1},
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.APPROVE


def test_baseline_position_rule_ignores_bool_values() -> None:
    """``bool`` is disjoint from ``int``/``float`` per the spec."""

    t = InProcessPolicyTransport(rules=build_baseline_rules())
    pi = PolicyInput.from_mapping(
        action="STAGE_ORDER",
        mode="LIVE",
        subject="s",
        payload={
            "notional_usd": True,  # type: ignore[dict-item]
            "notional_limit_usd": 10.0,
        },
    )
    r = t.evaluate(pi)
    assert r.verdict is PolicyVerdict.APPROVE


# ---------------------------------------------------------------------------
# OpaPolicyEvaluator
# ---------------------------------------------------------------------------


def test_evaluator_basic_approve() -> None:
    ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=()))
    decision = ev.evaluate(_basic_input())
    assert decision.verdict is PolicyVerdict.APPROVE
    assert len(decision.policy_digest) == 32
    int(decision.policy_digest, 16)


def test_evaluator_rejects_non_policy_input() -> None:
    ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=()))
    with pytest.raises(TypeError):
        ev.evaluate("not-a-policy-input")  # type: ignore[arg-type]


def test_evaluator_fail_closes_on_transport_error() -> None:
    def _boom(_pi: PolicyInput) -> bool:
        raise KeyError("oops")

    ev = OpaPolicyEvaluator(
        transport=InProcessPolicyTransport(
            rules=(
                PolicyRule(
                    policy_id="p",
                    rule_path="r",
                    predicate=_boom,
                    rejection_code="X",
                ),
            )
        )
    )
    decision = ev.evaluate(_basic_input())
    assert decision.verdict is PolicyVerdict.REJECT
    assert decision.rejection_code == "POLICY_TRANSPORT_ERROR"
    assert decision.policy_id == DEFAULT_POLICY_PACKAGE
    assert "transport error" in decision.summary


def test_evaluator_propagates_transport_decision_unchanged() -> None:
    t = InProcessPolicyTransport(rules=build_baseline_rules())
    ev = OpaPolicyEvaluator(transport=t)
    pi = PolicyInput.from_mapping(
        action="EXECUTE_ORDER",
        mode="SAFE",
        subject="s",
        payload={"notional_usd": 1.0, "notional_limit_usd": 10.0},
    )
    decision = ev.evaluate(pi)
    assert decision.rejection_code == "POLICY_EXECUTION_GATE"
    assert decision.policy_id == "dix.governance.execution_gates"


def test_evaluator_rejects_bad_transport_return() -> None:
    class BadTransport:
        def evaluate(self, _pi: PolicyInput) -> object:
            return "not-a-result"

    ev = OpaPolicyEvaluator(transport=BadTransport())  # type: ignore[arg-type]
    with pytest.raises(PolicyEvaluationError):
        ev.evaluate(_basic_input())


def test_evaluator_rejects_bad_transport_type() -> None:
    with pytest.raises(TypeError):
        OpaPolicyEvaluator(transport=42)  # type: ignore[arg-type]


def test_evaluator_three_run_replay_equality() -> None:
    """INV-15: byte-identical decisions across three independent runs."""

    digests: list[str] = []
    for _ in range(3):
        ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=build_baseline_rules()))
        pi = PolicyInput.from_mapping(
            action="EXECUTE_ORDER",
            mode="SAFE",
            subject="op",
            payload={
                "notional_usd": 50.0,
                "notional_limit_usd": 100.0,
            },
        )
        digests.append(ev.evaluate(pi).policy_digest)
    assert len(set(digests)) == 1


def test_evaluator_digest_changes_on_input_change() -> None:
    ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=()))
    a = ev.evaluate(_basic_input())
    b = ev.evaluate(_basic_input(subject="other"))
    assert a.policy_digest != b.policy_digest


def test_evaluator_digest_is_dict_order_independent() -> None:
    ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=()))
    pi1 = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"a": 1, "b": 2, "c": 3},
    )
    pi2 = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"c": 3, "b": 2, "a": 1},
    )
    assert ev.evaluate(pi1).policy_digest == ev.evaluate(pi2).policy_digest


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


def test_policy_decision_rejects_bad_digest() -> None:
    with pytest.raises(ValueError):
        PolicyDecision(
            verdict=PolicyVerdict.APPROVE,
            policy_id="p",
            rule_path="r",
            rejection_code="",
            summary="",
            policy_digest="too-short",
        )


def test_policy_decision_rejects_bad_verdict_type() -> None:
    with pytest.raises(TypeError):
        PolicyDecision(
            verdict="APPROVE",  # type: ignore[arg-type]
            policy_id="p",
            rule_path="r",
            rejection_code="",
            summary="",
            policy_digest="0" * 32,
        )


# ---------------------------------------------------------------------------
# to_governance_decision
# ---------------------------------------------------------------------------


def _approved_decision() -> PolicyDecision:
    return PolicyDecision(
        verdict=PolicyVerdict.APPROVE,
        policy_id="p",
        rule_path="r",
        rejection_code="",
        summary="ok",
        policy_digest="0" * 32,
    )


def _rejected_decision() -> PolicyDecision:
    return PolicyDecision(
        verdict=PolicyVerdict.REJECT,
        policy_id="p",
        rule_path="r",
        rejection_code="POLICY_X",
        summary="no",
        policy_digest="0" * 32,
    )


def _escalated_decision() -> PolicyDecision:
    return PolicyDecision(
        verdict=PolicyVerdict.ESCALATE,
        policy_id="p",
        rule_path="r",
        rejection_code="POLICY_ESC",
        summary="ask op",
        policy_digest="0" * 32,
    )


def test_to_governance_decision_approve() -> None:
    gd = to_governance_decision(_approved_decision(), ts_ns=1_000, kind=DecisionKind.NOOP)
    assert isinstance(gd, GovernanceDecision)
    assert gd.approved is True
    assert gd.rejection_code == ""


def test_to_governance_decision_reject() -> None:
    gd = to_governance_decision(_rejected_decision(), ts_ns=1_000, kind=DecisionKind.REJECTED)
    assert gd.approved is False
    assert gd.rejection_code == "POLICY_X"


def test_to_governance_decision_escalate_is_not_approved() -> None:
    gd = to_governance_decision(_escalated_decision(), ts_ns=1_000, kind=DecisionKind.REJECTED)
    assert gd.approved is False
    assert gd.rejection_code == "POLICY_ESC"


def test_to_governance_decision_rejects_bad_kind() -> None:
    with pytest.raises(TypeError):
        to_governance_decision(
            _approved_decision(),
            ts_ns=1,
            kind="NOOP",  # type: ignore[arg-type]
        )


def test_to_governance_decision_rejects_negative_ts() -> None:
    with pytest.raises(ValueError):
        to_governance_decision(_approved_decision(), ts_ns=-1, kind=DecisionKind.NOOP)


def test_to_governance_decision_rejects_bool_ts() -> None:
    with pytest.raises(TypeError):
        to_governance_decision(
            _approved_decision(),
            ts_ns=True,  # type: ignore[arg-type]
            kind=DecisionKind.NOOP,
        )


def test_to_governance_decision_rejects_non_decision() -> None:
    with pytest.raises(TypeError):
        to_governance_decision(
            object(),  # type: ignore[arg-type]
            ts_ns=1,
            kind=DecisionKind.NOOP,
        )


def test_to_governance_decision_carries_ledger_seq() -> None:
    gd = to_governance_decision(
        _approved_decision(),
        ts_ns=1,
        kind=DecisionKind.NOOP,
        ledger_seq=42,
    )
    assert gd.ledger_seq == 42


# ---------------------------------------------------------------------------
# opa_http_transport_factory
# ---------------------------------------------------------------------------


def test_http_factory_requires_client() -> None:
    with pytest.raises(TypeError):
        opa_policy.opa_http_transport_factory(client=None)


def test_http_factory_rejects_bad_policy_package() -> None:
    with pytest.raises(ValueError):
        opa_policy.opa_http_transport_factory(client=object(), policy_package="")


# ---------------------------------------------------------------------------
# Rego policy files
# ---------------------------------------------------------------------------


def test_policies_dir_has_three_rego_files() -> None:
    files = sorted(f.name for f in POLICIES_DIR.glob("*.rego"))
    assert files == [
        "autonomy_levels.rego",
        "execution_gates.rego",
        "position_limits.rego",
    ]


def test_rego_files_have_adapted_from_header() -> None:
    for rego in POLICIES_DIR.glob("*.rego"):
        text = rego.read_text()
        assert "# ADAPTED FROM: open-policy-agent" in text


def test_rego_packages_match_python_policy_ids() -> None:
    python_ids = {r.policy_id for r in build_baseline_rules()}
    rego_packages: set[str] = set()
    for rego in POLICIES_DIR.glob("*.rego"):
        text = rego.read_text()
        m = re.search(r"^package\s+(\S+)", text, re.MULTILINE)
        assert m, f"missing package in {rego.name}"
        rego_packages.add(m.group(1))
    assert rego_packages == python_ids


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_e2e_baseline_pipeline_emits_governance_decision() -> None:
    ev = OpaPolicyEvaluator(transport=InProcessPolicyTransport(rules=build_baseline_rules()))
    pi = PolicyInput.from_mapping(
        action="EXECUTE_ORDER",
        mode="SAFE",
        subject="op",
        payload={
            "notional_usd": 50.0,
            "notional_limit_usd": 100.0,
        },
    )
    decision = ev.evaluate(pi)
    gd = to_governance_decision(decision, ts_ns=12345, kind=DecisionKind.REJECTED)
    assert gd.approved is False
    assert gd.rejection_code == "POLICY_EXECUTION_GATE"
    assert gd.ts_ns == 12345


def test_e2e_canonical_json_is_replayable() -> None:
    pi = PolicyInput.from_mapping(
        action="X",
        mode="LIVE",
        subject="s",
        payload={"a": 1, "b": "two", "c": 3.5, "d": True, "e": None},
    )
    blob = pi.to_canonical_json()
    parsed = json.loads(blob)
    assert parsed == {
        "action": "X",
        "mode": "LIVE",
        "subject": "s",
        "payload": {"a": 1, "b": "two", "c": 3.5, "d": True, "e": None},
    }


def test_opa_policy_error_hierarchy() -> None:
    assert issubclass(PolicyTransportError, OpaPolicyError)
    assert issubclass(PolicyEvaluationError, OpaPolicyError)
