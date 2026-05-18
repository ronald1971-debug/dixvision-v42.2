"""Tests for C-11 federated_openfl — declarative multi-round federation plans.

OFFLINE-only deterministic plan execution. Mirrors the test discipline of
``tests/test_federated.py`` (C-09) and ``tests/test_fedml.py`` (C-10):

* module surface + lazy seam,
* value-object validation (happy path + frozen + edge cases),
* plan digest stability + collaborator-order invariance,
* execute_plan math equivalence to a sequence of flat FedAvg rounds,
* INV-15 3-run byte-identical replay,
* privacy guards (inherited from C-09),
* AST guardrails (forbidden imports, transport-layer typed-event
  constructors, runtime-tier imports).
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.federated import fed_avg_aggregate
from learning_engine.lanes.federated_openfl import (
    NEW_PIP_DEPENDENCIES,
    OPENFL_VERSION,
    FederationPlan,
    MultiRoundReport,
    RoundContribution,
    RoundReport,
    execute_plan,
    plan_digest,
)

OPENFL_PATH = Path("learning_engine/lanes/federated_openfl.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(
    n_rounds: int = 3,
    collaborators: tuple[str, ...] = ("c-alpha", "c-beta", "c-gamma"),
    min_collab: int = 2,
    parameter: str = "lr",
    plan_id: str = "plan-A",
) -> FederationPlan:
    return FederationPlan(
        plan_id=plan_id,
        parameter=parameter,
        collaborator_ids=collaborators,
        n_rounds=n_rounds,
        min_collaborators_per_round=min_collab,
    )


def _contrib(
    round_index: int,
    collaborator_id: str,
    delta: float,
    num_samples: int,
    ts_ns: int = 100,
) -> RoundContribution:
    return RoundContribution(
        round_index=round_index,
        collaborator_id=collaborator_id,
        delta=delta,
        num_samples=num_samples,
        ts_ns=ts_ns,
    )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_version_string(self) -> None:
        assert OPENFL_VERSION == "v3.7-C11"

    def test_new_pip_dependencies(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ("openfl",)

    def test_exports(self) -> None:
        from learning_engine.lanes import federated_openfl as mod

        for name in (
            "FederationPlan",
            "RoundContribution",
            "RoundReport",
            "MultiRoundReport",
            "execute_plan",
            "plan_digest",
        ):
            assert hasattr(mod, name), name


# ---------------------------------------------------------------------------
# FederationPlan validation
# ---------------------------------------------------------------------------


class TestFederationPlan:
    def test_happy_path(self) -> None:
        p = _plan()
        assert p.plan_id == "plan-A"
        assert p.parameter == "lr"
        assert p.collaborator_ids == ("c-alpha", "c-beta", "c-gamma")
        assert p.n_rounds == 3
        assert p.min_collaborators_per_round == 2
        assert p.aggregator_id == "aggregator-0"
        assert p.strategy == "fedavg"

    def test_frozen(self) -> None:
        p = _plan()
        with pytest.raises(FrozenInstanceError):
            p.plan_id = "other"  # type: ignore[misc]

    def test_collaborators_sorted_for_stability(self) -> None:
        p = FederationPlan(
            plan_id="p",
            parameter="x",
            collaborator_ids=("z", "a", "m"),
            n_rounds=1,
            min_collaborators_per_round=1,
        )
        assert p.collaborator_ids == ("a", "m", "z")

    def test_empty_plan_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=1,
            )

    def test_empty_parameter_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=1,
            )

    def test_empty_aggregator_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=1,
                aggregator_id="",
            )

    def test_unsupported_strategy_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=1,
                strategy="median",
            )

    def test_empty_collaborator_list_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=(),
                n_rounds=1,
                min_collaborators_per_round=1,
            )

    def test_duplicate_collaborators_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "a"),
                n_rounds=1,
                min_collaborators_per_round=1,
            )

    def test_empty_collaborator_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", ""),
                n_rounds=1,
                min_collaborators_per_round=1,
            )

    def test_zero_rounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=0,
                min_collaborators_per_round=1,
            )

    def test_negative_min_collab_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=0,
            )

    def test_min_collab_exceeds_collab_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=3,
            )

    def test_meta_non_str_rejected(self) -> None:
        with pytest.raises(TypeError):
            FederationPlan(
                plan_id="p",
                parameter="x",
                collaborator_ids=("a", "b"),
                n_rounds=1,
                min_collaborators_per_round=1,
                meta={"k": 1},  # type: ignore[dict-item]
            )


# ---------------------------------------------------------------------------
# plan_digest
# ---------------------------------------------------------------------------


class TestPlanDigest:
    def test_digest_is_32_hex(self) -> None:
        d = plan_digest(_plan())
        assert len(d) == 32
        assert all(c in "0123456789abcdef" for c in d)

    def test_digest_3_run_byte_identical(self) -> None:
        p = _plan()
        digests = {plan_digest(p) for _ in range(3)}
        assert len(digests) == 1

    def test_digest_order_invariant_for_collaborators(self) -> None:
        p1 = FederationPlan(
            plan_id="p",
            parameter="x",
            collaborator_ids=("a", "b", "c"),
            n_rounds=2,
            min_collaborators_per_round=1,
        )
        p2 = FederationPlan(
            plan_id="p",
            parameter="x",
            collaborator_ids=("c", "a", "b"),
            n_rounds=2,
            min_collaborators_per_round=1,
        )
        assert plan_digest(p1) == plan_digest(p2)

    def test_digest_distinguishes_plan_id(self) -> None:
        p1 = _plan(plan_id="A")
        p2 = _plan(plan_id="B")
        assert plan_digest(p1) != plan_digest(p2)

    def test_digest_distinguishes_parameter(self) -> None:
        p1 = _plan(parameter="lr")
        p2 = _plan(parameter="momentum")
        assert plan_digest(p1) != plan_digest(p2)

    def test_digest_distinguishes_n_rounds(self) -> None:
        p1 = _plan(n_rounds=2)
        p2 = _plan(n_rounds=3)
        assert plan_digest(p1) != plan_digest(p2)


# ---------------------------------------------------------------------------
# RoundContribution validation
# ---------------------------------------------------------------------------


class TestRoundContribution:
    def test_happy_path(self) -> None:
        c = _contrib(0, "c-alpha", 0.5, 10)
        assert c.round_index == 0
        assert c.collaborator_id == "c-alpha"
        assert c.delta == 0.5
        assert c.num_samples == 10

    def test_frozen(self) -> None:
        c = _contrib(0, "c-alpha", 0.5, 10)
        with pytest.raises(FrozenInstanceError):
            c.delta = 1.0  # type: ignore[misc]

    def test_negative_round_index_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(-1, "c-alpha", 0.5, 10)

    def test_empty_collaborator_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(0, "", 0.5, 10)

    def test_nan_delta_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(0, "c-alpha", float("nan"), 10)

    def test_inf_delta_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(0, "c-alpha", float("inf"), 10)

    def test_negative_num_samples_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(0, "c-alpha", 0.5, -1)

    def test_negative_ts_ns_rejected(self) -> None:
        with pytest.raises(ValueError):
            _contrib(0, "c-alpha", 0.5, 10, ts_ns=-1)

    def test_as_gradient_update_carries_parameter(self) -> None:
        c = _contrib(0, "c-alpha", 0.5, 10)
        gu = c.as_gradient_update("lr")
        assert gu.client_id == "c-alpha"
        assert gu.parameter == "lr"
        assert gu.delta == 0.5
        assert gu.num_samples == 10


# ---------------------------------------------------------------------------
# RoundReport / MultiRoundReport validation
# ---------------------------------------------------------------------------


class TestRoundReport:
    def _r(self, **kw: object) -> RoundReport:
        defaults: dict[str, object] = dict(
            plan_id="p",
            round_index=0,
            parameter="lr",
            n_collaborators=2,
            aggregated_delta=0.5,
            total_samples=20,
            ts_ns=100,
            digest="a" * 32,
        )
        defaults.update(kw)
        return RoundReport(**defaults)  # type: ignore[arg-type]

    def test_happy_path(self) -> None:
        r = self._r()
        assert r.round_index == 0

    def test_frozen(self) -> None:
        r = self._r()
        with pytest.raises(FrozenInstanceError):
            r.round_index = 5  # type: ignore[misc]

    def test_empty_plan_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._r(plan_id="")

    def test_short_digest_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._r(digest="abc")

    def test_nan_delta_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._r(aggregated_delta=float("nan"))


class TestMultiRoundReport:
    def _mr(
        self,
        rounds: tuple[RoundReport, ...] | None = None,
        **kw: object,
    ) -> MultiRoundReport:
        if rounds is None:
            rounds = (
                RoundReport(
                    plan_id="p",
                    round_index=0,
                    parameter="lr",
                    n_collaborators=2,
                    aggregated_delta=0.5,
                    total_samples=20,
                    ts_ns=100,
                    digest="a" * 32,
                ),
                RoundReport(
                    plan_id="p",
                    round_index=1,
                    parameter="lr",
                    n_collaborators=2,
                    aggregated_delta=0.25,
                    total_samples=20,
                    ts_ns=100,
                    digest="b" * 32,
                ),
            )
        defaults: dict[str, object] = dict(
            plan_id="p",
            plan_digest="c" * 32,
            parameter="lr",
            n_rounds=2,
            rounds=rounds,
            initial_value=1.0,
            final_value=1.75,
            ts_ns=100,
            digest="d" * 32,
        )
        defaults.update(kw)
        return MultiRoundReport(**defaults)  # type: ignore[arg-type]

    def test_happy_path(self) -> None:
        m = self._mr()
        assert m.n_rounds == 2

    def test_n_rounds_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            self._mr(n_rounds=3)

    def test_out_of_order_rounds_rejected(self) -> None:
        rounds = (
            RoundReport(
                plan_id="p",
                round_index=1,
                parameter="lr",
                n_collaborators=2,
                aggregated_delta=0.5,
                total_samples=20,
                ts_ns=100,
                digest="a" * 32,
            ),
            RoundReport(
                plan_id="p",
                round_index=0,
                parameter="lr",
                n_collaborators=2,
                aggregated_delta=0.25,
                total_samples=20,
                ts_ns=100,
                digest="b" * 32,
            ),
        )
        with pytest.raises(ValueError):
            self._mr(rounds=rounds, n_rounds=2)

    def test_round_plan_id_mismatch_rejected(self) -> None:
        rounds = (
            RoundReport(
                plan_id="other",
                round_index=0,
                parameter="lr",
                n_collaborators=2,
                aggregated_delta=0.5,
                total_samples=20,
                ts_ns=100,
                digest="a" * 32,
            ),
            RoundReport(
                plan_id="p",
                round_index=1,
                parameter="lr",
                n_collaborators=2,
                aggregated_delta=0.25,
                total_samples=20,
                ts_ns=100,
                digest="b" * 32,
            ),
        )
        with pytest.raises(ValueError):
            self._mr(rounds=rounds, n_rounds=2)

    def test_round_parameter_mismatch_rejected(self) -> None:
        rounds = (
            RoundReport(
                plan_id="p",
                round_index=0,
                parameter="lr",
                n_collaborators=2,
                aggregated_delta=0.5,
                total_samples=20,
                ts_ns=100,
                digest="a" * 32,
            ),
            RoundReport(
                plan_id="p",
                round_index=1,
                parameter="momentum",
                n_collaborators=2,
                aggregated_delta=0.25,
                total_samples=20,
                ts_ns=100,
                digest="b" * 32,
            ),
        )
        with pytest.raises(ValueError):
            self._mr(rounds=rounds, n_rounds=2)


# ---------------------------------------------------------------------------
# execute_plan — happy path + math equivalence
# ---------------------------------------------------------------------------


class TestExecutePlanHappy:
    def test_single_round_plan(self) -> None:
        p = _plan(n_rounds=1)
        contribs = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 10),
            _contrib(0, "c-gamma", 0.0, 10),
        ]
        report, update = execute_plan(
            plan=p,
            contributions=contribs,
            initial_value=0.0,
            ts_ns=100,
        )
        assert isinstance(report, MultiRoundReport)
        assert isinstance(update, LearningUpdate)
        assert report.n_rounds == 1
        assert report.parameter == "lr"
        assert report.plan_id == "plan-A"
        assert report.plan_digest == plan_digest(p)
        assert len(report.rounds) == 1
        r0 = report.rounds[0]
        # FedAvg over (0.5*10, 0.25*10, 0.0*10) / 30 = 7.5/30 = 0.25
        assert r0.aggregated_delta == pytest.approx(0.25)
        assert r0.total_samples == 30
        assert r0.n_collaborators == 3
        assert report.initial_value == 0.0
        assert report.final_value == pytest.approx(0.25)

    def test_multi_round_equivalence_to_flat_fedavg(self) -> None:
        """Per-round delta == flat FedAvg over per-round updates."""
        p = _plan(n_rounds=2)
        contribs = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 30),
            _contrib(1, "c-alpha", 0.1, 5),
            _contrib(1, "c-gamma", -0.05, 15),
        ]
        report, _ = execute_plan(
            plan=p,
            contributions=contribs,
            initial_value=1.0,
            ts_ns=200,
        )
        # Round 0 expected: (0.5*10 + 0.25*30) / 40 = 12.5/40 = 0.3125
        # Round 1 expected: (0.1*5 + -0.05*15) / 20 = -0.25/20 = -0.0125
        r0 = report.rounds[0]
        r1 = report.rounds[1]
        assert r0.aggregated_delta == pytest.approx(0.3125)
        assert r1.aggregated_delta == pytest.approx(-0.0125)
        assert r0.total_samples == 40
        assert r1.total_samples == 20
        # Final value folds both deltas onto initial:
        assert report.final_value == pytest.approx(1.0 + 0.3125 + -0.0125)

    def test_final_value_equals_sum_of_round_deltas(self) -> None:
        p = _plan(n_rounds=3)
        contribs = [
            _contrib(0, "c-alpha", 0.4, 5),
            _contrib(0, "c-beta", 0.6, 5),
            _contrib(1, "c-alpha", 0.2, 5),
            _contrib(1, "c-beta", 0.0, 5),
            _contrib(2, "c-alpha", -0.1, 5),
            _contrib(2, "c-gamma", -0.3, 5),
        ]
        report, _ = execute_plan(
            plan=p,
            contributions=contribs,
            initial_value=10.0,
            ts_ns=300,
        )
        expected = 10.0 + sum(r.aggregated_delta for r in report.rounds)
        assert report.final_value == pytest.approx(expected)

    def test_learning_update_payload(self) -> None:
        p = _plan(n_rounds=2)
        contribs = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 10),
            _contrib(1, "c-alpha", 0.1, 5),
            _contrib(1, "c-beta", 0.0, 5),
        ]
        report, update = execute_plan(
            plan=p,
            contributions=contribs,
            initial_value=1.0,
            ts_ns=400,
        )
        assert update.ts_ns == 400
        assert update.strategy_id == "aggregator-0"
        assert update.parameter == "lr"
        assert update.old_value == repr(1.0)
        assert update.new_value == repr(report.final_value)
        assert update.meta["lane"] == "federated_openfl"
        assert update.meta["plan_id"] == "plan-A"
        assert update.meta["plan_digest"] == plan_digest(p)
        assert update.meta["n_rounds"] == "2"
        assert update.meta["report_digest"] == report.digest
        assert update.meta["version"] == OPENFL_VERSION

    def test_contribution_order_invariant(self) -> None:
        p = _plan(n_rounds=2)
        order_a = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 30),
            _contrib(1, "c-alpha", 0.1, 5),
            _contrib(1, "c-gamma", -0.05, 15),
        ]
        order_b = list(reversed(order_a))
        r_a, _ = execute_plan(
            plan=p,
            contributions=order_a,
            initial_value=1.0,
            ts_ns=500,
        )
        r_b, _ = execute_plan(
            plan=p,
            contributions=order_b,
            initial_value=1.0,
            ts_ns=500,
        )
        assert r_a.digest == r_b.digest
        assert r_a.final_value == r_b.final_value

    def test_3_run_byte_identical_replay(self) -> None:
        p = _plan(n_rounds=2)
        contribs = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 30),
            _contrib(1, "c-alpha", 0.1, 5),
            _contrib(1, "c-gamma", -0.05, 15),
        ]
        digests = set()
        finals = set()
        for _ in range(3):
            report, _ = execute_plan(
                plan=p,
                contributions=contribs,
                initial_value=1.0,
                ts_ns=500,
            )
            digests.add(report.digest)
            finals.add(report.final_value)
        assert len(digests) == 1
        assert len(finals) == 1

    def test_round_digest_matches_independent_fedavg(self) -> None:
        """Per-round aggregated_delta numerically agrees with flat fed_avg."""
        p = _plan(n_rounds=1)
        contribs = [
            _contrib(0, "c-alpha", 0.5, 10),
            _contrib(0, "c-beta", 0.25, 30),
            _contrib(0, "c-gamma", 0.0, 20),
        ]
        report, _ = execute_plan(
            plan=p,
            contributions=contribs,
            initial_value=0.0,
            ts_ns=100,
        )
        gus = [c.as_gradient_update("lr") for c in contribs]
        agg_delta, total = fed_avg_aggregate(gus)
        assert report.rounds[0].aggregated_delta == pytest.approx(agg_delta)
        assert report.rounds[0].total_samples == total


# ---------------------------------------------------------------------------
# execute_plan — rejection cases
# ---------------------------------------------------------------------------


class TestExecutePlanRejection:
    def test_unknown_collaborator_rejected(self) -> None:
        p = _plan(n_rounds=1)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[
                    _contrib(0, "c-alpha", 0.5, 10),
                    _contrib(0, "outsider", 0.25, 10),
                ],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_round_index_out_of_range_rejected(self) -> None:
        p = _plan(n_rounds=2)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[
                    _contrib(0, "c-alpha", 0.5, 10),
                    _contrib(0, "c-beta", 0.25, 10),
                    _contrib(5, "c-alpha", 0.1, 5),
                ],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_duplicate_collaborator_in_round_rejected(self) -> None:
        p = _plan(n_rounds=1)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[
                    _contrib(0, "c-alpha", 0.5, 10),
                    _contrib(0, "c-alpha", 0.25, 10),
                ],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_round_below_min_collab_rejected(self) -> None:
        p = _plan(n_rounds=1, min_collab=2)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[_contrib(0, "c-alpha", 0.5, 10)],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_one_empty_round_rejected_with_min_collab(self) -> None:
        p = _plan(n_rounds=2, min_collab=1)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[_contrib(0, "c-alpha", 0.5, 10)],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_privacy_meta_key_rejected(self) -> None:
        p = _plan(n_rounds=1)
        bad = RoundContribution(
            round_index=0,
            collaborator_id="c-alpha",
            delta=0.5,
            num_samples=10,
            ts_ns=100,
            meta={"raw_data": "leak"},
        )
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[bad, _contrib(0, "c-beta", 0.25, 10)],
                initial_value=0.0,
                ts_ns=100,
            )

    @pytest.mark.parametrize(
        "key",
        [
            "raw_data",
            "training_data",
            "dataset",
            "samples",
            "features",
            "labels",
            "X",
            "y",
        ],
    )
    def test_all_privacy_keys_rejected(self, key: str) -> None:
        p = _plan(n_rounds=1)
        bad = RoundContribution(
            round_index=0,
            collaborator_id="c-alpha",
            delta=0.5,
            num_samples=10,
            ts_ns=100,
            meta={key: "leak"},
        )
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[bad, _contrib(0, "c-beta", 0.25, 10)],
                initial_value=0.0,
                ts_ns=100,
            )

    def test_negative_ts_ns_rejected(self) -> None:
        p = _plan(n_rounds=1)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[
                    _contrib(0, "c-alpha", 0.5, 10),
                    _contrib(0, "c-beta", 0.25, 10),
                ],
                initial_value=0.0,
                ts_ns=-1,
            )

    def test_nan_initial_value_rejected(self) -> None:
        p = _plan(n_rounds=1)
        with pytest.raises(ValueError):
            execute_plan(
                plan=p,
                contributions=[
                    _contrib(0, "c-alpha", 0.5, 10),
                    _contrib(0, "c-beta", 0.25, 10),
                ],
                initial_value=float("nan"),
                ts_ns=100,
            )


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _parse_module() -> ast.Module:
    return ast.parse(OPENFL_PATH.read_text(encoding="utf-8"))


def _top_level_imports() -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(_parse_module()):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


FORBIDDEN_TOP_LEVEL = {
    "time",
    "datetime",
    "random",
    "asyncio",
    "os",
    "subprocess",
    "socket",
    "ssl",
    "openfl",
    "flwr",
    "fedml",
    "numpy",
    "torch",
    "polars",
    "pandas",
    "requests",
    "httpx",
    "aiohttp",
    "tornado",
    "sqlite3",
}

RUNTIME_TIERS = {
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "evolution_engine",
    "system_engine",
}

TRANSPORT_TYPED_EVENTS = {
    "SystemEvent",
    "HazardEvent",
    "SignalEvent",
    "ExecutionEvent",
    "PatchProposal",
}


class TestASTGuardrails:
    def test_no_forbidden_top_level_imports(self) -> None:
        names = _top_level_imports()
        overlap = names & FORBIDDEN_TOP_LEVEL
        assert overlap == set(), f"forbidden top-level imports: {overlap}"

    def test_no_runtime_tier_imports(self) -> None:
        names = _top_level_imports()
        overlap = names & RUNTIME_TIERS
        assert overlap == set(), f"runtime-tier imports: {overlap}"

    def test_openfl_never_imported(self) -> None:
        names = _top_level_imports()
        assert "openfl" not in names

    def test_no_transport_typed_event_constructors(self) -> None:
        tree = _parse_module()
        bad: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in TRANSPORT_TYPED_EVENTS:
                    bad.add(node.func.id)
        assert bad == set(), f"transport-layer typed-event constructors: {bad}"

    def test_module_uses_learning_update_only(self) -> None:
        text = OPENFL_PATH.read_text(encoding="utf-8")
        assert "LearningUpdate(" in text
        for name in TRANSPORT_TYPED_EVENTS:
            assert f"{name}(" not in text, f"{name}( appears in module"

    def test_no_random_or_time_calls(self) -> None:
        text = OPENFL_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "time.time",
            "time.monotonic",
            "time.perf_counter",
            "time.time_ns",
            "datetime.now",
            "datetime.utcnow",
            "random.random",
            "random.randint",
            "asyncio.run",
            "asyncio.sleep",
        ):
            assert forbidden not in text, f"{forbidden} appears in module"
