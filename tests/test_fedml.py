"""C-10 fedml — tests for federated_fedml topology lane.

Pins:

* Module surface (constants, exports)
* Topology enum membership
* GroupAssignment / HierarchicalRoundResult / RingStep / RingRoundResult validation
* is_valid_group_partition: orphans / overlaps / duplicate group_id
* partition_into_groups: correct bucketing, canonical sort within bucket
* hierarchical_digest: 3-run identical, distinguishes inputs
* hierarchical_aggregate: math equivalence to flat FedAvg, per-group records,
    parameter mismatch / privacy / empty / zero-samples reject, INV-15
    3-run byte-identical replay, input order invariant
* ring_aggregate: math equivalence to flat FedAvg, per-step running sums,
    parameter mismatch / privacy / empty / zero-samples reject, INV-15
    3-run byte-identical replay
* AST guardrails: no forbidden top-level imports, no runtime-tier imports,
    no transport-layer typed-event constructors, never imports fedml or flwr
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path

import pytest

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.federated import (
    FederatedAggregate,
    GradientUpdate,
    fed_avg_aggregate,
)
from learning_engine.lanes.federated_fedml import (
    FEDML_VERSION,
    NEW_PIP_DEPENDENCIES,
    FederationTopology,
    GroupAssignment,
    HierarchicalRoundResult,
    RingRoundResult,
    RingStep,
    hierarchical_aggregate,
    hierarchical_digest,
    is_valid_group_partition,
    partition_into_groups,
    ring_aggregate,
)

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "learning_engine" / "lanes" / "federated_fedml.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _u(
    *,
    client_id: str = "c1",
    parameter: str = "lr",
    delta: float = 0.1,
    num_samples: int = 10,
    ts_ns: int = 1_000,
    meta: Mapping[str, str] | None = None,
) -> GradientUpdate:
    return GradientUpdate(
        client_id=client_id,
        parameter=parameter,
        delta=delta,
        num_samples=num_samples,
        ts_ns=ts_ns,
        meta=dict(meta) if meta else {},
    )


def _round_inputs() -> dict[str, object]:
    return {
        "round_id": "round-1",
        "strategy_id": "S1",
        "parameter": "lr",
        "current_value": 0.5,
        "updates": [
            _u(client_id="a", delta=0.0, num_samples=10),
            _u(client_id="b", delta=0.5, num_samples=20),
            _u(client_id="c", delta=1.0, num_samples=30),
            _u(client_id="d", delta=-0.2, num_samples=40),
        ],
        "groups": [
            GroupAssignment(group_id="g0", client_ids=("a", "b")),
            GroupAssignment(group_id="g1", client_ids=("c", "d")),
        ],
        "ts_ns": 5_000,
    }


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_version() -> None:
    assert FEDML_VERSION == "v3.7-C10"


def test_module_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("fedml",)


def test_topology_values() -> None:
    assert FederationTopology.FLAT.value == "flat"
    assert FederationTopology.HIERARCHICAL.value == "hierarchical"
    assert FederationTopology.RING.value == "ring"


def test_topology_membership() -> None:
    assert set(FederationTopology) == {
        FederationTopology.FLAT,
        FederationTopology.HIERARCHICAL,
        FederationTopology.RING,
    }


# ---------------------------------------------------------------------------
# GroupAssignment validation
# ---------------------------------------------------------------------------


def test_group_assignment_happy_path() -> None:
    g = GroupAssignment(group_id="g0", client_ids=("a", "b"))
    assert g.group_id == "g0"
    assert g.client_ids == ("a", "b")


def test_group_assignment_frozen() -> None:
    g = GroupAssignment(group_id="g0", client_ids=("a",))
    with pytest.raises((AttributeError, Exception)):
        g.group_id = "other"  # type: ignore[misc]


def test_group_assignment_empty_group_id() -> None:
    with pytest.raises(ValueError, match="group_id"):
        GroupAssignment(group_id="", client_ids=("a",))


def test_group_assignment_empty_client_ids() -> None:
    with pytest.raises(ValueError, match="client_ids"):
        GroupAssignment(group_id="g0", client_ids=())


def test_group_assignment_duplicate_client_id() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        GroupAssignment(group_id="g0", client_ids=("a", "a"))


def test_group_assignment_empty_client_in_list() -> None:
    with pytest.raises(ValueError, match="client_id"):
        GroupAssignment(group_id="g0", client_ids=("a", ""))


# ---------------------------------------------------------------------------
# HierarchicalRoundResult / RingStep / RingRoundResult validation
# ---------------------------------------------------------------------------


def _stub_agg(round_id: str = "r1") -> FederatedAggregate:
    return FederatedAggregate(
        round_id=round_id,
        parameter="lr",
        n_clients=1,
        aggregated_delta=0.1,
        total_samples=10,
        ts_ns=1_000,
        digest="0" * 32,
    )


def test_hierarchical_round_result_happy_path() -> None:
    agg = _stub_agg()
    r = HierarchicalRoundResult(
        round_id="r1",
        parameter="lr",
        n_groups=1,
        group_aggregates=(agg,),
        root_aggregate=agg,
        ts_ns=1_000,
        digest="a" * 32,
    )
    assert r.n_groups == 1
    assert r.group_aggregates == (agg,)


def test_hierarchical_round_result_n_groups_mismatch() -> None:
    agg = _stub_agg()
    with pytest.raises(ValueError, match="group_aggregates length"):
        HierarchicalRoundResult(
            round_id="r1",
            parameter="lr",
            n_groups=2,
            group_aggregates=(agg,),
            root_aggregate=agg,
            ts_ns=1_000,
            digest="a" * 32,
        )


def test_hierarchical_round_result_bad_digest() -> None:
    agg = _stub_agg()
    with pytest.raises(ValueError, match="digest"):
        HierarchicalRoundResult(
            round_id="r1",
            parameter="lr",
            n_groups=1,
            group_aggregates=(agg,),
            root_aggregate=agg,
            ts_ns=1_000,
            digest="nope",
        )


def test_hierarchical_round_result_empty_round_id() -> None:
    agg = _stub_agg()
    with pytest.raises(ValueError, match="round_id"):
        HierarchicalRoundResult(
            round_id="",
            parameter="lr",
            n_groups=1,
            group_aggregates=(agg,),
            root_aggregate=agg,
            ts_ns=1_000,
            digest="a" * 32,
        )


def test_hierarchical_round_result_negative_ts() -> None:
    agg = _stub_agg()
    with pytest.raises(ValueError, match="ts_ns"):
        HierarchicalRoundResult(
            round_id="r1",
            parameter="lr",
            n_groups=1,
            group_aggregates=(agg,),
            root_aggregate=agg,
            ts_ns=-1,
            digest="a" * 32,
        )


def test_ring_step_happy_path() -> None:
    s = RingStep(step_index=0, client_id="a", running_weighted_sum=1.0, running_total_samples=10)
    assert s.step_index == 0


def test_ring_step_negative_index() -> None:
    with pytest.raises(ValueError, match="step_index"):
        RingStep(step_index=-1, client_id="a", running_weighted_sum=0.0, running_total_samples=0)


def test_ring_step_nan_sum() -> None:
    with pytest.raises(ValueError, match="running_weighted_sum"):
        RingStep(
            step_index=0,
            client_id="a",
            running_weighted_sum=float("nan"),
            running_total_samples=10,
        )


def test_ring_round_result_happy_path() -> None:
    step = RingStep(step_index=0, client_id="a", running_weighted_sum=1.0, running_total_samples=10)
    r = RingRoundResult(
        round_id="r1",
        parameter="lr",
        ring_order=("a",),
        aggregated_delta=0.1,
        total_samples=10,
        steps=(step,),
        ts_ns=1_000,
        digest="b" * 32,
    )
    assert r.ring_order == ("a",)


def test_ring_round_result_order_steps_mismatch() -> None:
    step = RingStep(step_index=0, client_id="a", running_weighted_sum=1.0, running_total_samples=10)
    with pytest.raises(ValueError, match="ring_order length"):
        RingRoundResult(
            round_id="r1",
            parameter="lr",
            ring_order=("a", "b"),
            aggregated_delta=0.1,
            total_samples=10,
            steps=(step,),
            ts_ns=1_000,
            digest="b" * 32,
        )


# ---------------------------------------------------------------------------
# is_valid_group_partition
# ---------------------------------------------------------------------------


def test_partition_valid_basic() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a", "b"))]
    assert is_valid_group_partition(updates, groups) is True


def test_partition_valid_multi_group() -> None:
    updates = [_u(client_id="a"), _u(client_id="b"), _u(client_id="c")]
    groups = [
        GroupAssignment(group_id="g0", client_ids=("a",)),
        GroupAssignment(group_id="g1", client_ids=("b", "c")),
    ]
    assert is_valid_group_partition(updates, groups) is True


def test_partition_orphan_client() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a",))]
    assert is_valid_group_partition(updates, groups) is False


def test_partition_extra_group_client() -> None:
    updates = [_u(client_id="a")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a", "ghost"))]
    assert is_valid_group_partition(updates, groups) is False


def test_partition_overlapping_groups() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [
        GroupAssignment(group_id="g0", client_ids=("a",)),
        GroupAssignment(group_id="g1", client_ids=("a", "b")),
    ]
    assert is_valid_group_partition(updates, groups) is False


def test_partition_duplicate_group_id() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [
        GroupAssignment(group_id="dup", client_ids=("a",)),
        GroupAssignment(group_id="dup", client_ids=("b",)),
    ]
    assert is_valid_group_partition(updates, groups) is False


def test_partition_empty_updates_invalid() -> None:
    groups = [GroupAssignment(group_id="g0", client_ids=("a",))]
    assert is_valid_group_partition([], groups) is False


def test_partition_empty_groups_invalid() -> None:
    updates = [_u(client_id="a")]
    assert is_valid_group_partition(updates, []) is False


# ---------------------------------------------------------------------------
# partition_into_groups
# ---------------------------------------------------------------------------


def test_partition_into_groups_basic() -> None:
    updates = [
        _u(client_id="a", delta=0.1),
        _u(client_id="b", delta=0.2),
        _u(client_id="c", delta=0.3),
    ]
    groups = [
        GroupAssignment(group_id="g0", client_ids=("a",)),
        GroupAssignment(group_id="g1", client_ids=("b", "c")),
    ]
    buckets = partition_into_groups(updates, groups)
    assert list(buckets.keys()) == ["g0", "g1"]
    assert len(buckets["g0"]) == 1
    assert len(buckets["g1"]) == 2


def test_partition_into_groups_lex_order_keys() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [
        GroupAssignment(group_id="zz", client_ids=("a",)),
        GroupAssignment(group_id="aa", client_ids=("b",)),
    ]
    buckets = partition_into_groups(updates, groups)
    assert list(buckets.keys()) == ["aa", "zz"]


def test_partition_into_groups_rejects_orphan() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a",))]
    with pytest.raises(ValueError, match="not a valid partition"):
        partition_into_groups(updates, groups)


def test_partition_into_groups_returns_tuples() -> None:
    updates = [_u(client_id="a")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a",))]
    buckets = partition_into_groups(updates, groups)
    assert isinstance(buckets["g0"], tuple)


# ---------------------------------------------------------------------------
# hierarchical_digest
# ---------------------------------------------------------------------------


def test_hierarchical_digest_three_run_identical() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a", "b"))]
    d1 = hierarchical_digest(updates, groups)
    d2 = hierarchical_digest(updates, groups)
    d3 = hierarchical_digest(updates, groups)
    assert d1 == d2 == d3
    assert len(d1) == 32


def test_hierarchical_digest_input_order_invariant() -> None:
    updates_a = [_u(client_id="a"), _u(client_id="b")]
    updates_b = [_u(client_id="b"), _u(client_id="a")]
    groups = [GroupAssignment(group_id="g0", client_ids=("a", "b"))]
    assert hierarchical_digest(updates_a, groups) == hierarchical_digest(updates_b, groups)


def test_hierarchical_digest_group_order_invariant() -> None:
    updates = [_u(client_id="a"), _u(client_id="b")]
    groups_a = [
        GroupAssignment(group_id="g0", client_ids=("a",)),
        GroupAssignment(group_id="g1", client_ids=("b",)),
    ]
    groups_b = [
        GroupAssignment(group_id="g1", client_ids=("b",)),
        GroupAssignment(group_id="g0", client_ids=("a",)),
    ]
    assert hierarchical_digest(updates, groups_a) == hierarchical_digest(updates, groups_b)


def test_hierarchical_digest_distinguishes_inputs() -> None:
    updates_a = [_u(client_id="a", delta=0.1)]
    updates_b = [_u(client_id="a", delta=0.2)]
    groups = [GroupAssignment(group_id="g0", client_ids=("a",))]
    assert hierarchical_digest(updates_a, groups) != hierarchical_digest(updates_b, groups)


# ---------------------------------------------------------------------------
# hierarchical_aggregate
# ---------------------------------------------------------------------------


def test_hierarchical_aggregate_returns_typed_pair() -> None:
    result, update = hierarchical_aggregate(**_round_inputs())  # type: ignore[arg-type]
    assert isinstance(result, HierarchicalRoundResult)
    assert isinstance(update, LearningUpdate)


def test_hierarchical_aggregate_math_equivalence_to_flat_fedavg() -> None:
    """Hierarchical aggregation must equal flat FedAvg on the same updates."""
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    flat_delta, flat_samples = fed_avg_aggregate(inputs["updates"])  # type: ignore[arg-type]
    assert math.isclose(result.root_aggregate.aggregated_delta, flat_delta)
    assert result.root_aggregate.total_samples == flat_samples


def test_hierarchical_aggregate_two_group_partitions_have_expected_means() -> None:
    """g0 = avg(a,b) weighted; g1 = avg(c,d) weighted."""
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    # g0: (0.0*10 + 0.5*20)/30 = 10/30
    # g1: (1.0*30 + -0.2*40)/70 = 22/70
    g0, g1 = result.group_aggregates
    assert math.isclose(g0.aggregated_delta, 10.0 / 30.0)
    assert math.isclose(g1.aggregated_delta, 22.0 / 70.0)
    assert g0.total_samples == 30
    assert g1.total_samples == 70


def test_hierarchical_aggregate_n_groups_matches() -> None:
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    assert result.n_groups == 2
    assert len(result.group_aggregates) == 2


def test_hierarchical_aggregate_groups_in_lex_order() -> None:
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    ids = [g.round_id for g in result.group_aggregates]
    assert ids == sorted(ids)


def test_hierarchical_aggregate_root_round_id_is_outer() -> None:
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    assert result.root_aggregate.round_id == "round-1"


def test_hierarchical_aggregate_group_round_ids_are_namespaced() -> None:
    inputs = _round_inputs()
    result, _ = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    assert result.group_aggregates[0].round_id == "round-1::g0"
    assert result.group_aggregates[1].round_id == "round-1::g1"


def test_hierarchical_aggregate_learning_update_meta() -> None:
    inputs = _round_inputs()
    _, update = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    assert update.meta["lane"] == "federated_fedml"
    assert update.meta["topology"] == "hierarchical"
    assert update.meta["version"] == FEDML_VERSION
    assert update.meta["round_id"] == "round-1"
    assert update.meta["n_groups"] == "2"
    assert update.meta["total_samples"] == "100"


def test_hierarchical_aggregate_new_value_folding() -> None:
    inputs = _round_inputs()
    result, update = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    expected_new = 0.5 + result.root_aggregate.aggregated_delta
    assert math.isclose(float(update.new_value), expected_new)
    assert math.isclose(float(update.old_value), 0.5)


def test_hierarchical_aggregate_parameter_mismatch_rejected() -> None:
    inputs = _round_inputs()
    bad_updates = list(inputs["updates"])  # type: ignore[arg-type]
    bad_updates[0] = _u(client_id="a", parameter="other_param", delta=0.0, num_samples=10)
    inputs["updates"] = bad_updates
    with pytest.raises(ValueError, match="parameter mismatch"):
        hierarchical_aggregate(**inputs)  # type: ignore[arg-type]


def test_hierarchical_aggregate_privacy_violation_rejected() -> None:
    inputs = _round_inputs()
    bad_updates = list(inputs["updates"])  # type: ignore[arg-type]
    bad_updates[0] = _u(
        client_id="a",
        delta=0.0,
        num_samples=10,
        meta={"raw_data": "leaked"},
    )
    inputs["updates"] = bad_updates
    with pytest.raises(ValueError, match="raw-data"):
        hierarchical_aggregate(**inputs)  # type: ignore[arg-type]


def test_hierarchical_aggregate_empty_round_id_rejected() -> None:
    inputs = _round_inputs()
    inputs["round_id"] = ""
    with pytest.raises(ValueError, match="round_id"):
        hierarchical_aggregate(**inputs)  # type: ignore[arg-type]


def test_hierarchical_aggregate_invalid_partition_rejected() -> None:
    inputs = _round_inputs()
    inputs["groups"] = [GroupAssignment(group_id="g0", client_ids=("a",))]
    with pytest.raises(ValueError, match="not a valid partition"):
        hierarchical_aggregate(**inputs)  # type: ignore[arg-type]


def test_hierarchical_aggregate_three_run_byte_identical() -> None:
    inputs = _round_inputs()
    r1, u1 = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    r2, u2 = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    r3, u3 = hierarchical_aggregate(**inputs)  # type: ignore[arg-type]
    assert r1 == r2 == r3
    assert u1 == u2 == u3
    assert r1.digest == r2.digest == r3.digest


def test_hierarchical_aggregate_input_order_invariant() -> None:
    inputs_a = _round_inputs()
    inputs_b = _round_inputs()
    inputs_b["updates"] = list(reversed(inputs_b["updates"]))  # type: ignore[arg-type]
    r_a, _ = hierarchical_aggregate(**inputs_a)  # type: ignore[arg-type]
    r_b, _ = hierarchical_aggregate(**inputs_b)  # type: ignore[arg-type]
    assert math.isclose(
        r_a.root_aggregate.aggregated_delta,
        r_b.root_aggregate.aggregated_delta,
    )
    assert r_a.digest == r_b.digest


# ---------------------------------------------------------------------------
# ring_aggregate
# ---------------------------------------------------------------------------


def _ring_inputs() -> dict[str, object]:
    return {
        "round_id": "round-2",
        "strategy_id": "S1",
        "parameter": "lr",
        "current_value": 1.0,
        "updates": [
            _u(client_id="a", delta=0.0, num_samples=10),
            _u(client_id="b", delta=1.0, num_samples=30),
        ],
        "ts_ns": 5_000,
    }


def test_ring_aggregate_returns_typed_pair() -> None:
    result, update = ring_aggregate(**_ring_inputs())  # type: ignore[arg-type]
    assert isinstance(result, RingRoundResult)
    assert isinstance(update, LearningUpdate)


def test_ring_aggregate_math_equivalence_to_flat_fedavg() -> None:
    inputs = _ring_inputs()
    result, _ = ring_aggregate(**inputs)  # type: ignore[arg-type]
    flat_delta, flat_samples = fed_avg_aggregate(inputs["updates"])  # type: ignore[arg-type]
    assert math.isclose(result.aggregated_delta, flat_delta)
    assert result.total_samples == flat_samples


def test_ring_aggregate_specific_math() -> None:
    """Two clients (0.0*10 + 1.0*30)/40 = 0.75."""
    inputs = _ring_inputs()
    result, _ = ring_aggregate(**inputs)  # type: ignore[arg-type]
    assert math.isclose(result.aggregated_delta, 0.75)
    assert result.total_samples == 40


def test_ring_aggregate_per_step_running_sums() -> None:
    inputs = _ring_inputs()
    result, _ = ring_aggregate(**inputs)  # type: ignore[arg-type]
    s0, s1 = result.steps
    # canonical sort is by (client_id, ts_ns), so 'a' first
    assert s0.client_id == "a"
    assert math.isclose(s0.running_weighted_sum, 0.0)
    assert s0.running_total_samples == 10
    assert s1.client_id == "b"
    assert math.isclose(s1.running_weighted_sum, 30.0)
    assert s1.running_total_samples == 40


def test_ring_aggregate_ring_order_is_unique_sorted() -> None:
    inputs = _ring_inputs()
    result, _ = ring_aggregate(**inputs)  # type: ignore[arg-type]
    assert result.ring_order == ("a", "b")


def test_ring_aggregate_new_value_folding() -> None:
    inputs = _ring_inputs()
    _, update = ring_aggregate(**inputs)  # type: ignore[arg-type]
    assert math.isclose(float(update.old_value), 1.0)
    assert math.isclose(float(update.new_value), 1.0 + 0.75)


def test_ring_aggregate_learning_update_meta() -> None:
    inputs = _ring_inputs()
    _, update = ring_aggregate(**inputs)  # type: ignore[arg-type]
    assert update.meta["lane"] == "federated_fedml"
    assert update.meta["topology"] == "ring"
    assert update.meta["version"] == FEDML_VERSION
    assert update.meta["round_id"] == "round-2"
    assert update.meta["n_steps"] == "2"


def test_ring_aggregate_parameter_mismatch_rejected() -> None:
    inputs = _ring_inputs()
    bad_updates = list(inputs["updates"])  # type: ignore[arg-type]
    bad_updates[0] = _u(client_id="a", parameter="other", delta=0.0, num_samples=10)
    inputs["updates"] = bad_updates
    with pytest.raises(ValueError, match="parameter mismatch"):
        ring_aggregate(**inputs)  # type: ignore[arg-type]


def test_ring_aggregate_privacy_violation_rejected() -> None:
    inputs = _ring_inputs()
    bad_updates = list(inputs["updates"])  # type: ignore[arg-type]
    bad_updates[0] = _u(
        client_id="a",
        delta=0.0,
        num_samples=10,
        meta={"training_data": "leaked"},
    )
    inputs["updates"] = bad_updates
    with pytest.raises(ValueError, match="raw-data"):
        ring_aggregate(**inputs)  # type: ignore[arg-type]


def test_ring_aggregate_three_run_byte_identical() -> None:
    inputs = _ring_inputs()
    r1, u1 = ring_aggregate(**inputs)  # type: ignore[arg-type]
    r2, u2 = ring_aggregate(**inputs)  # type: ignore[arg-type]
    r3, u3 = ring_aggregate(**inputs)  # type: ignore[arg-type]
    assert r1 == r2 == r3
    assert u1 == u2 == u3
    assert r1.digest == r2.digest == r3.digest


def test_ring_aggregate_input_order_invariant_for_final_number() -> None:
    inputs_a = _ring_inputs()
    inputs_b = _ring_inputs()
    inputs_b["updates"] = list(reversed(inputs_b["updates"]))  # type: ignore[arg-type]
    r_a, _ = ring_aggregate(**inputs_a)  # type: ignore[arg-type]
    r_b, _ = ring_aggregate(**inputs_b)  # type: ignore[arg-type]
    assert math.isclose(r_a.aggregated_delta, r_b.aggregated_delta)
    assert r_a.total_samples == r_b.total_samples
    assert r_a.digest == r_b.digest


def test_ring_aggregate_empty_round_id_rejected() -> None:
    inputs = _ring_inputs()
    inputs["round_id"] = ""
    with pytest.raises(ValueError, match="round_id"):
        ring_aggregate(**inputs)  # type: ignore[arg-type]


def test_ring_aggregate_negative_ts_rejected() -> None:
    inputs = _ring_inputs()
    inputs["ts_ns"] = -1
    with pytest.raises(ValueError, match="ts_ns"):
        ring_aggregate(**inputs)  # type: ignore[arg-type]


def test_ring_aggregate_nan_current_value_rejected() -> None:
    inputs = _ring_inputs()
    inputs["current_value"] = float("nan")
    with pytest.raises(ValueError, match="current_value"):
        ring_aggregate(**inputs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AST guardrails — pin INV-15, B1, B27/B28/INV-71
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "subprocess",
        "socket",
        "ssl",
        "fedml",
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


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                assert root not in _FORBIDDEN_TOP_LEVEL, f"forbidden top-level import: {n.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            root = node.module.split(".")[0]
            assert root not in _FORBIDDEN_TOP_LEVEL, (
                f"forbidden top-level import-from: {node.module}"
            )


_RUNTIME_TIERS = frozenset(
    {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "system_engine",
    },
)


def test_no_runtime_tier_imports() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                assert root not in _RUNTIME_TIERS, f"forbidden runtime-tier import: {n.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            assert root not in _RUNTIME_TIERS, f"forbidden runtime-tier import-from: {node.module}"


_TRANSPORT_EVENT_CONSTRUCTORS = frozenset(
    {
        "SystemEvent",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "PatchProposal",
    },
)


def test_no_transport_event_constructors() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _TRANSPORT_EVENT_CONSTRUCTORS:
                pytest.fail(
                    f"transport-layer typed-event constructor found: {name} (B27/B28/INV-71)",
                )


def test_module_never_imports_fedml_or_flwr() -> None:
    """NEW_PIP_DEPENDENCIES is declared but the package is never imported."""
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                assert not n.name.startswith("fedml"), (
                    "fedml must remain a lazy seam — never imported"
                )
                assert not n.name.startswith("flwr"), (
                    "flwr must remain a lazy seam — never imported"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            assert not node.module.startswith("fedml"), (
                "fedml must remain a lazy seam — never imported"
            )
            assert not node.module.startswith("flwr"), (
                "flwr must remain a lazy seam — never imported"
            )
