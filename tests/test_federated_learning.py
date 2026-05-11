"""C-09 flwr — FedAvg federated learning lane tests.

Pins:

* Module surface — exports, version, NEW_PIP_DEPENDENCIES.
* :class:`GradientUpdate` / :class:`FederatedAggregate` validation.
* :func:`fed_avg_aggregate` math correctness.
* :func:`aggregate_round` produces the expected :class:`LearningUpdate`.
* :func:`verify_privacy` rejects raw-data smuggling.
* :func:`canonical_sort_updates` deterministic order.
* INV-15: 3-run byte-identical replay over aggregate digest + LearningUpdate.
* B1 / B27 / B28 / INV-71 AST guardrails: no forbidden top-level imports,
  no runtime-tier imports, no transport-layer typed-event constructors.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from core.contracts.learning import LearningUpdate
from learning_engine.lanes import federated
from learning_engine.lanes.federated import (
    FEDERATED_VERSION,
    MIN_CLIENTS_PER_ROUND,
    NEW_PIP_DEPENDENCIES,
    PRIVACY_FORBIDDEN_META_KEYS,
    FederatedAggregate,
    GradientUpdate,
    aggregate_round,
    canonical_sort_updates,
    fed_avg_aggregate,
    is_valid_round,
    updates_digest,
    verify_privacy,
)

LANE_PATH = Path(federated.__file__)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_version_string() -> None:
    assert FEDERATED_VERSION == "v3.7-C09"


def test_new_pip_dependencies_declared_but_unused() -> None:
    assert NEW_PIP_DEPENDENCIES == ("flwr",)
    src = LANE_PATH.read_text()
    assert "import flwr" not in src
    assert "from flwr" not in src


def test_min_clients_constant() -> None:
    assert MIN_CLIENTS_PER_ROUND == 2


def test_privacy_forbidden_keys() -> None:
    assert "raw_data" in PRIVACY_FORBIDDEN_META_KEYS
    assert "training_data" in PRIVACY_FORBIDDEN_META_KEYS
    assert "dataset" in PRIVACY_FORBIDDEN_META_KEYS
    assert "X" in PRIVACY_FORBIDDEN_META_KEYS
    assert "y" in PRIVACY_FORBIDDEN_META_KEYS


def test_exports_complete() -> None:
    assert set(federated.__all__) == {
        "FEDERATED_VERSION",
        "FederatedAggregate",
        "GradientUpdate",
        "MIN_CLIENTS_PER_ROUND",
        "NEW_PIP_DEPENDENCIES",
        "PRIVACY_FORBIDDEN_META_KEYS",
        "aggregate_round",
        "canonical_sort_updates",
        "fed_avg_aggregate",
        "is_valid_round",
        "updates_digest",
        "verify_privacy",
    }


# ---------------------------------------------------------------------------
# GradientUpdate validation
# ---------------------------------------------------------------------------


def _u(
    *,
    client_id: str = "c1",
    parameter: str = "lr",
    delta: float = 0.1,
    num_samples: int = 10,
    ts_ns: int = 1_000_000,
    meta: dict[str, str] | None = None,
) -> GradientUpdate:
    return GradientUpdate(
        client_id=client_id,
        parameter=parameter,
        delta=delta,
        num_samples=num_samples,
        ts_ns=ts_ns,
        meta=dict(meta or {}),
    )


def test_gradient_update_happy_path() -> None:
    u = _u()
    assert u.client_id == "c1"
    assert u.parameter == "lr"
    assert u.delta == 0.1
    assert u.num_samples == 10


def test_gradient_update_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    u = _u()
    with pytest.raises(FrozenInstanceError):
        u.delta = 0.5  # type: ignore[misc]


def test_gradient_update_rejects_empty_client_id() -> None:
    with pytest.raises(ValueError, match="client_id"):
        _u(client_id="")


def test_gradient_update_rejects_empty_parameter() -> None:
    with pytest.raises(ValueError, match="parameter"):
        _u(parameter="")


def test_gradient_update_rejects_non_float_delta() -> None:
    with pytest.raises(TypeError, match="delta"):
        GradientUpdate(
            client_id="c1",
            parameter="lr",
            delta=1,  # type: ignore[arg-type]
            num_samples=10,
            ts_ns=1,
        )


def test_gradient_update_rejects_nan_delta() -> None:
    with pytest.raises(ValueError, match="finite"):
        _u(delta=float("nan"))


def test_gradient_update_rejects_inf_delta() -> None:
    with pytest.raises(ValueError, match="finite"):
        _u(delta=float("inf"))


def test_gradient_update_rejects_bool_num_samples() -> None:
    with pytest.raises(TypeError, match="num_samples"):
        GradientUpdate(
            client_id="c1",
            parameter="lr",
            delta=0.1,
            num_samples=True,  # type: ignore[arg-type]
            ts_ns=1,
        )


def test_gradient_update_rejects_negative_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        _u(num_samples=-1)


def test_gradient_update_allows_zero_num_samples() -> None:
    u = _u(num_samples=0)
    assert u.num_samples == 0


def test_gradient_update_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        _u(ts_ns=-1)


def test_gradient_update_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns"):
        GradientUpdate(
            client_id="c1",
            parameter="lr",
            delta=0.1,
            num_samples=10,
            ts_ns=True,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# FederatedAggregate validation
# ---------------------------------------------------------------------------


def _agg(
    *,
    round_id: str = "r1",
    parameter: str = "lr",
    n_clients: int = 2,
    aggregated_delta: float = 0.05,
    total_samples: int = 20,
    ts_ns: int = 1,
    digest: str = "0" * 32,
) -> FederatedAggregate:
    return FederatedAggregate(
        round_id=round_id,
        parameter=parameter,
        n_clients=n_clients,
        aggregated_delta=aggregated_delta,
        total_samples=total_samples,
        ts_ns=ts_ns,
        digest=digest,
    )


def test_federated_aggregate_happy_path() -> None:
    a = _agg()
    assert a.round_id == "r1"
    assert a.n_clients == 2


def test_federated_aggregate_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    a = _agg()
    with pytest.raises(FrozenInstanceError):
        a.n_clients = 5  # type: ignore[misc]


def test_federated_aggregate_rejects_empty_round_id() -> None:
    with pytest.raises(ValueError, match="round_id"):
        _agg(round_id="")


def test_federated_aggregate_rejects_negative_n_clients() -> None:
    with pytest.raises(ValueError, match="n_clients"):
        _agg(n_clients=-1)


def test_federated_aggregate_rejects_nan_aggregated_delta() -> None:
    with pytest.raises(ValueError, match="aggregated_delta"):
        _agg(aggregated_delta=float("nan"))


def test_federated_aggregate_rejects_bad_digest_length() -> None:
    with pytest.raises(ValueError, match="digest"):
        _agg(digest="abcd")


def test_federated_aggregate_rejects_uppercase_digest() -> None:
    with pytest.raises(ValueError, match="digest"):
        _agg(digest="A" * 32)


# ---------------------------------------------------------------------------
# verify_privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(PRIVACY_FORBIDDEN_META_KEYS))
def test_verify_privacy_rejects_each_forbidden_key(key: str) -> None:
    u = _u(meta={key: "smuggled"})
    with pytest.raises(ValueError, match="raw-data"):
        verify_privacy(u)


def test_verify_privacy_accepts_safe_meta() -> None:
    u = _u(meta={"version": "1", "client_region": "eu"})
    verify_privacy(u)


def test_verify_privacy_accepts_empty_meta() -> None:
    u = _u(meta={})
    verify_privacy(u)


# ---------------------------------------------------------------------------
# canonical_sort_updates
# ---------------------------------------------------------------------------


def test_canonical_sort_orders_by_client_id_then_ts() -> None:
    updates = [
        _u(client_id="b", ts_ns=2),
        _u(client_id="a", ts_ns=10),
        _u(client_id="a", ts_ns=5),
        _u(client_id="c", ts_ns=1),
    ]
    sorted_ = canonical_sort_updates(updates)
    assert [(u.client_id, u.ts_ns) for u in sorted_] == [
        ("a", 5),
        ("a", 10),
        ("b", 2),
        ("c", 1),
    ]


def test_canonical_sort_is_stable_three_runs() -> None:
    updates = [
        _u(client_id="b", ts_ns=2),
        _u(client_id="a", ts_ns=5),
    ]
    r1 = canonical_sort_updates(updates)
    r2 = canonical_sort_updates(updates)
    r3 = canonical_sort_updates(updates)
    assert r1 == r2 == r3


def test_canonical_sort_returns_tuple() -> None:
    out = canonical_sort_updates([_u()])
    assert isinstance(out, tuple)


# ---------------------------------------------------------------------------
# updates_digest — INV-15
# ---------------------------------------------------------------------------


def test_updates_digest_three_run_identical() -> None:
    updates = [
        _u(client_id="b", delta=0.2, num_samples=5, ts_ns=10),
        _u(client_id="a", delta=0.1, num_samples=10, ts_ns=20),
    ]
    d1 = updates_digest(updates)
    d2 = updates_digest(updates)
    d3 = updates_digest(updates)
    assert d1 == d2 == d3
    assert len(d1) == 32


def test_updates_digest_is_order_invariant_under_sort() -> None:
    a = [
        _u(client_id="b", delta=0.2, num_samples=5, ts_ns=10),
        _u(client_id="a", delta=0.1, num_samples=10, ts_ns=20),
    ]
    b = list(reversed(a))
    assert updates_digest(a) == updates_digest(b)


def test_updates_digest_distinguishes_different_deltas() -> None:
    a = [_u(client_id="a", delta=0.1, num_samples=10, ts_ns=1)]
    b = [_u(client_id="a", delta=0.2, num_samples=10, ts_ns=1)]
    assert updates_digest(a) != updates_digest(b)


def test_updates_digest_distinguishes_different_samples() -> None:
    a = [_u(client_id="a", delta=0.1, num_samples=10, ts_ns=1)]
    b = [_u(client_id="a", delta=0.1, num_samples=11, ts_ns=1)]
    assert updates_digest(a) != updates_digest(b)


# ---------------------------------------------------------------------------
# is_valid_round
# ---------------------------------------------------------------------------


def test_is_valid_round_empty_is_invalid() -> None:
    assert is_valid_round([]) is False


def test_is_valid_round_two_clients() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    assert is_valid_round(updates) is True


def test_is_valid_round_one_client_fails_default_min() -> None:
    updates = [_u(client_id="a")]
    assert is_valid_round(updates) is False


def test_is_valid_round_one_client_passes_with_min_1() -> None:
    updates = [_u(client_id="a", num_samples=5)]
    assert is_valid_round(updates, min_clients=1) is True


def test_is_valid_round_rejects_mixed_parameters() -> None:
    updates = [_u(client_id="a", parameter="lr"), _u(client_id="b", parameter="bs")]
    assert is_valid_round(updates) is False


def test_is_valid_round_rejects_zero_total_samples() -> None:
    updates = [_u(client_id="a", num_samples=0), _u(client_id="b", num_samples=0)]
    assert is_valid_round(updates) is False


def test_is_valid_round_treats_dupes_by_client_id() -> None:
    updates = [
        _u(client_id="a", ts_ns=1),
        _u(client_id="a", ts_ns=2),  # retry: same client
    ]
    assert is_valid_round(updates) is False  # only 1 distinct client


def test_is_valid_round_rejects_min_clients_zero() -> None:
    with pytest.raises(ValueError, match="min_clients"):
        is_valid_round([_u()], min_clients=0)


# ---------------------------------------------------------------------------
# fed_avg_aggregate
# ---------------------------------------------------------------------------


def test_fed_avg_aggregate_equal_weights() -> None:
    updates = [
        _u(client_id="a", delta=0.1, num_samples=10),
        _u(client_id="b", delta=0.3, num_samples=10),
    ]
    result, total = fed_avg_aggregate(updates)
    assert math.isclose(result, 0.2)
    assert total == 20


def test_fed_avg_aggregate_unequal_weights() -> None:
    updates = [
        _u(client_id="a", delta=0.0, num_samples=10),
        _u(client_id="b", delta=1.0, num_samples=30),
    ]
    result, total = fed_avg_aggregate(updates)
    assert math.isclose(result, 0.75)  # (0*10 + 1*30)/40
    assert total == 40


def test_fed_avg_aggregate_single_client() -> None:
    updates = [_u(client_id="a", delta=0.42, num_samples=7)]
    result, total = fed_avg_aggregate(updates)
    assert result == 0.42
    assert total == 7


def test_fed_avg_aggregate_negative_deltas() -> None:
    updates = [
        _u(client_id="a", delta=-0.5, num_samples=5),
        _u(client_id="b", delta=0.5, num_samples=5),
    ]
    result, _ = fed_avg_aggregate(updates)
    assert math.isclose(result, 0.0)


def test_fed_avg_aggregate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one update"):
        fed_avg_aggregate([])


def test_fed_avg_aggregate_rejects_zero_total_samples() -> None:
    updates = [_u(client_id="a", num_samples=0), _u(client_id="b", num_samples=0)]
    with pytest.raises(ValueError, match="num_samples"):
        fed_avg_aggregate(updates)


def test_fed_avg_aggregate_is_order_invariant() -> None:
    updates = [
        _u(client_id="a", delta=0.1, num_samples=10),
        _u(client_id="b", delta=0.3, num_samples=20),
        _u(client_id="c", delta=-0.2, num_samples=15),
    ]
    r1, t1 = fed_avg_aggregate(updates)
    r2, t2 = fed_avg_aggregate(list(reversed(updates)))
    assert r1 == r2
    assert t1 == t2


def test_fed_avg_aggregate_three_run_identical() -> None:
    updates = [
        _u(client_id="a", delta=0.123, num_samples=11, ts_ns=1),
        _u(client_id="b", delta=-0.456, num_samples=7, ts_ns=2),
        _u(client_id="c", delta=0.789, num_samples=13, ts_ns=3),
    ]
    r1 = fed_avg_aggregate(updates)
    r2 = fed_avg_aggregate(updates)
    r3 = fed_avg_aggregate(updates)
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# aggregate_round — produces LearningUpdate
# ---------------------------------------------------------------------------


def _round_inputs() -> dict:
    return {
        "round_id": "round-7",
        "strategy_id": "strat-A",
        "parameter": "lr",
        "current_value": 0.5,
        "updates": [
            _u(client_id="a", parameter="lr", delta=0.1, num_samples=10, ts_ns=1),
            _u(client_id="b", parameter="lr", delta=0.3, num_samples=10, ts_ns=2),
        ],
        "ts_ns": 99,
    }


def test_aggregate_round_returns_aggregate_and_update() -> None:
    agg, upd = aggregate_round(**_round_inputs())
    assert isinstance(agg, FederatedAggregate)
    assert isinstance(upd, LearningUpdate)
    assert agg.parameter == "lr"
    assert agg.n_clients == 2
    assert agg.total_samples == 20
    assert math.isclose(agg.aggregated_delta, 0.2)
    assert upd.parameter == "lr"
    assert upd.strategy_id == "strat-A"
    assert upd.ts_ns == 99


def test_aggregate_round_old_value_equals_current() -> None:
    inputs = _round_inputs()
    agg, upd = aggregate_round(**inputs)
    assert upd.old_value == repr(inputs["current_value"])


def test_aggregate_round_new_value_folds_in_aggregate() -> None:
    inputs = _round_inputs()
    agg, upd = aggregate_round(**inputs)
    expected = inputs["current_value"] + agg.aggregated_delta
    assert upd.new_value == repr(expected)


def test_aggregate_round_reason_contains_round_id_and_digest() -> None:
    inputs = _round_inputs()
    agg, upd = aggregate_round(**inputs)
    assert inputs["round_id"] in upd.reason
    assert "federated_fedavg" in upd.reason
    assert agg.digest in upd.reason


def test_aggregate_round_meta_carries_lane_and_round() -> None:
    inputs = _round_inputs()
    agg, upd = aggregate_round(**inputs)
    assert upd.meta["lane"] == "federated"
    assert upd.meta["version"] == FEDERATED_VERSION
    assert upd.meta["round_id"] == inputs["round_id"]
    assert upd.meta["digest"] == agg.digest
    assert upd.meta["n_clients"] == "2"
    assert upd.meta["total_samples"] == "20"


def test_aggregate_round_rejects_empty_round_id() -> None:
    inputs = _round_inputs()
    inputs["round_id"] = ""
    with pytest.raises(ValueError, match="round_id"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_empty_strategy_id() -> None:
    inputs = _round_inputs()
    inputs["strategy_id"] = ""
    with pytest.raises(ValueError, match="strategy_id"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_empty_parameter() -> None:
    inputs = _round_inputs()
    inputs["parameter"] = ""
    with pytest.raises(ValueError, match="parameter"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_nan_current() -> None:
    inputs = _round_inputs()
    inputs["current_value"] = float("nan")
    with pytest.raises(ValueError, match="current_value"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_negative_ts() -> None:
    inputs = _round_inputs()
    inputs["ts_ns"] = -1
    with pytest.raises(ValueError, match="ts_ns"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_mismatched_parameter_on_client() -> None:
    inputs = _round_inputs()
    inputs["updates"] = [
        _u(client_id="a", parameter="lr"),
        _u(client_id="b", parameter="other"),
    ]
    with pytest.raises(ValueError, match="parameter mismatch"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_single_client_at_default_min() -> None:
    inputs = _round_inputs()
    inputs["updates"] = [_u(client_id="a", parameter="lr")]
    with pytest.raises(ValueError, match="invalid round"):
        aggregate_round(**inputs)


def test_aggregate_round_rejects_privacy_violation() -> None:
    inputs = _round_inputs()
    inputs["updates"] = [
        _u(client_id="a", parameter="lr", meta={"raw_data": "leak"}),
        _u(client_id="b", parameter="lr"),
    ]
    with pytest.raises(ValueError, match="raw-data"):
        aggregate_round(**inputs)


def test_aggregate_round_three_run_byte_identical() -> None:
    inputs = _round_inputs()
    r1 = aggregate_round(**inputs)
    r2 = aggregate_round(**inputs)
    r3 = aggregate_round(**inputs)
    assert r1 == r2
    assert r2 == r3
    assert r1[0].digest == r2[0].digest == r3[0].digest
    assert r1[1] == r2[1] == r3[1]


def test_aggregate_round_input_order_invariant() -> None:
    inputs = _round_inputs()
    forward = aggregate_round(**inputs)
    backward_inputs = dict(inputs)
    backward_inputs["updates"] = list(reversed(inputs["updates"]))
    backward = aggregate_round(**backward_inputs)
    assert forward[0].aggregated_delta == backward[0].aggregated_delta
    assert forward[0].digest == backward[0].digest
    assert forward[1].new_value == backward[1].new_value


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


FORBIDDEN_TOP_LEVEL_IMPORTS: frozenset[str] = frozenset(
    {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "subprocess",
        "socket",
        "ssl",
        "flwr",
        "numpy",
        "torch",
        "polars",
        "pandas",
        "requests",
        "httpx",
        "aiohttp",
        "tornado",
        "sqlite3",
    },
)

RUNTIME_TIER_PREFIXES: frozenset[str] = frozenset(
    {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
    },
)

FORBIDDEN_EVENT_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "SystemEvent",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "PatchProposal",
    },
)


def _module_ast() -> ast.Module:
    return ast.parse(LANE_PATH.read_text(), filename=str(LANE_PATH))


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_ast()
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in FORBIDDEN_TOP_LEVEL_IMPORTS:
                offenders.append(f"from {node.module} import ...")
    assert offenders == [], f"forbidden top-level imports: {offenders}"


def test_no_runtime_tier_imports() -> None:
    tree = _module_ast()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in RUNTIME_TIER_PREFIXES:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in RUNTIME_TIER_PREFIXES:
                offenders.append(f"from {node.module} import ...")
    assert offenders == [], f"forbidden runtime-tier imports: {offenders}"


def test_no_typed_event_constructors() -> None:
    """Lane must not construct transport-layer typed events.

    :class:`LearningUpdate` is a domain record (not a transport event) and is
    therefore allowed. The forbidden set covers the five transport-layer
    classes: ``SystemEvent`` / ``HazardEvent`` / ``SignalEvent`` /
    ``ExecutionEvent`` / ``PatchProposal``.
    """
    tree = _module_ast()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_EVENT_CONSTRUCTORS:
            offenders.append(func.id)
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_EVENT_CONSTRUCTORS:
            offenders.append(func.attr)
    assert offenders == [], f"transport-layer typed-event constructors: {offenders}"


def test_module_does_not_import_flwr() -> None:
    src = LANE_PATH.read_text()
    assert "import flwr" not in src
    assert "from flwr" not in src
