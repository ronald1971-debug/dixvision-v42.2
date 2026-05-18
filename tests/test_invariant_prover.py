"""I-24 — tests for the canonical invariant prover surface."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tools.invariant_prover import (
    INT_DOMAIN_MAX,
    INT_DOMAIN_MIN,
    MAX_INVARIANT_NAME_LEN,
    MAX_SAMPLES,
    MIN_SAMPLES,
    NEW_PIP_DEPENDENCIES,
    PROVER_VERSION,
    Counterexample,
    Invariant,
    PostCondition,
    ProofResult,
    ProofSuite,
    ProofTask,
    ProofVerdict,
    ProverError,
    bounded_int,
    enable_crosshair_factory,
    nonneg_int,
    positive_int,
    prove,
    prove_suite,
)

# ---------------------------------------------------------------------------
# Constants / module identity
# ---------------------------------------------------------------------------


def test_prover_version_is_pinned() -> None:
    assert PROVER_VERSION == "v1.0-I24"


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("crosshair-tool",)


def test_sample_bounds() -> None:
    assert MIN_SAMPLES == 1
    assert MAX_SAMPLES == 1_000_000


def test_int_domain_is_int32_signed() -> None:
    assert INT_DOMAIN_MIN == -(2**31)
    assert INT_DOMAIN_MAX == 2**31 - 1


def test_max_invariant_name_len() -> None:
    assert MAX_INVARIANT_NAME_LEN == 128


# ---------------------------------------------------------------------------
# ProofVerdict enum
# ---------------------------------------------------------------------------


def test_proof_verdict_values() -> None:
    assert ProofVerdict.PROVED.value == "PROVED"
    assert ProofVerdict.PROVED_SOFT.value == "PROVED_SOFT"
    assert ProofVerdict.COUNTEREXAMPLE.value == "COUNTEREXAMPLE"
    assert ProofVerdict.PRECONDITION_UNSATISFIABLE.value == ("PRECONDITION_UNSATISFIABLE")


def test_proof_verdict_count() -> None:
    assert len(list(ProofVerdict)) == 4


# ---------------------------------------------------------------------------
# Invariant validation
# ---------------------------------------------------------------------------


def test_invariant_constructs_valid() -> None:
    inv = Invariant(name="x_positive", arity=1, predicate=lambda x: x > 0)
    assert inv.name == "x_positive"
    assert inv.arity == 1


def test_invariant_is_frozen_and_slotted() -> None:
    inv = Invariant(name="x", arity=1, predicate=lambda x: True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.arity = 2  # type: ignore[misc]
    assert not hasattr(inv, "__dict__")


def test_invariant_rejects_empty_name() -> None:
    with pytest.raises(ProverError):
        Invariant(name="", arity=1, predicate=lambda x: True)


def test_invariant_rejects_oversize_name() -> None:
    with pytest.raises(ProverError):
        Invariant(
            name="x" * (MAX_INVARIANT_NAME_LEN + 1),
            arity=1,
            predicate=lambda x: True,
        )


def test_invariant_rejects_zero_arity() -> None:
    with pytest.raises(ProverError):
        Invariant(name="x", arity=0, predicate=lambda: True)


def test_invariant_rejects_bool_arity() -> None:
    with pytest.raises(ProverError):
        Invariant(name="x", arity=True, predicate=lambda x: True)  # type: ignore[arg-type]


def test_invariant_rejects_non_callable_predicate() -> None:
    with pytest.raises(ProverError):
        Invariant(name="x", arity=1, predicate="not callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PostCondition validation
# ---------------------------------------------------------------------------


def test_postcondition_constructs_valid() -> None:
    pc = PostCondition(name="out_nonneg", predicate=lambda x, out: out >= 0)
    assert pc.name == "out_nonneg"


def test_postcondition_is_frozen_and_slotted() -> None:
    pc = PostCondition(name="x", predicate=lambda x, out: True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pc.name = "y"  # type: ignore[misc]
    assert not hasattr(pc, "__dict__")


def test_postcondition_rejects_empty_name() -> None:
    with pytest.raises(ProverError):
        PostCondition(name="", predicate=lambda x, out: True)


def test_postcondition_rejects_non_callable() -> None:
    with pytest.raises(ProverError):
        PostCondition(name="x", predicate=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProofTask validation
# ---------------------------------------------------------------------------


def _trivial_post() -> tuple[PostCondition, ...]:
    return (PostCondition(name="always_true", predicate=lambda *args: True),)


def test_task_constructs_valid() -> None:
    task = ProofTask(
        name="t",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=_trivial_post(),
    )
    assert task.name == "t"
    assert task.max_samples == 1024


def test_task_is_frozen_and_slotted() -> None:
    task = ProofTask(
        name="t",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=_trivial_post(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.arity = 2  # type: ignore[misc]
    assert not hasattr(task, "__dict__")


def test_task_rejects_empty_name() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=_trivial_post(),
        )


def test_task_rejects_non_callable_target() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target="not callable",  # type: ignore[arg-type]
            arity=1,
            preconditions=(),
            postconditions=_trivial_post(),
        )


def test_task_rejects_zero_arity() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=0,
            preconditions=(),
            postconditions=_trivial_post(),
        )


def test_task_rejects_bool_arity() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=True,  # type: ignore[arg-type]
            preconditions=(),
            postconditions=_trivial_post(),
        )


def test_task_rejects_non_tuple_preconditions() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=[],  # type: ignore[arg-type]
            postconditions=_trivial_post(),
        )


def test_task_rejects_non_tuple_postconditions() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=[PostCondition(name="x", predicate=lambda *a: True)],  # type: ignore[arg-type]
        )


def test_task_rejects_empty_postconditions() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=(),
        )


def test_task_rejects_precondition_arity_mismatch() -> None:
    pre = Invariant(name="bad_arity", arity=2, predicate=lambda x, y: True)
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(pre,),
            postconditions=_trivial_post(),
        )


def test_task_rejects_non_invariant_precondition() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=("not an invariant",),  # type: ignore[arg-type]
            postconditions=_trivial_post(),
        )


def test_task_rejects_non_postcondition_postcondition() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=("not a postcondition",),  # type: ignore[arg-type]
        )


def test_task_rejects_below_min_samples() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=_trivial_post(),
            max_samples=0,
        )


def test_task_rejects_above_max_samples() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=_trivial_post(),
            max_samples=MAX_SAMPLES + 1,
        )


def test_task_rejects_bool_max_samples() -> None:
    with pytest.raises(ProverError):
        ProofTask(
            name="t",
            target=lambda x: x,
            arity=1,
            preconditions=(),
            postconditions=_trivial_post(),
            max_samples=True,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Counterexample validation
# ---------------------------------------------------------------------------


def test_counterexample_constructs_valid() -> None:
    cx = Counterexample(
        inputs=(1, 2),
        output=3,
        violated_postcondition="sum_check",
    )
    assert cx.inputs == (1, 2)


def test_counterexample_is_frozen_and_slotted() -> None:
    cx = Counterexample(inputs=(1,), output=0, violated_postcondition="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cx.output = 1  # type: ignore[misc]
    assert not hasattr(cx, "__dict__")


def test_counterexample_rejects_non_tuple_inputs() -> None:
    with pytest.raises(ProverError):
        Counterexample(
            inputs=[1],  # type: ignore[arg-type]
            output=0,
            violated_postcondition="x",
        )


def test_counterexample_rejects_bool_input() -> None:
    with pytest.raises(ProverError):
        Counterexample(
            inputs=(True,),
            output=0,
            violated_postcondition="x",
        )


def test_counterexample_rejects_empty_postcondition_name() -> None:
    with pytest.raises(ProverError):
        Counterexample(inputs=(1,), output=0, violated_postcondition="")


# ---------------------------------------------------------------------------
# ProofResult validation
# ---------------------------------------------------------------------------


def test_result_proved_soft_constructs() -> None:
    r = ProofResult(
        task_name="t",
        verdict=ProofVerdict.PROVED_SOFT,
        samples_drawn=10,
        samples_satisfying_preconditions=10,
    )
    assert r.backend == "stdlib"


def test_result_is_frozen_and_slotted() -> None:
    r = ProofResult(
        task_name="t",
        verdict=ProofVerdict.PROVED_SOFT,
        samples_drawn=1,
        samples_satisfying_preconditions=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.samples_drawn = 2  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_counterexample_without_cx_field() -> None:
    with pytest.raises(ProverError):
        ProofResult(
            task_name="t",
            verdict=ProofVerdict.COUNTEREXAMPLE,
            samples_drawn=1,
            samples_satisfying_preconditions=1,
        )


def test_result_rejects_non_counterexample_with_cx_field() -> None:
    cx = Counterexample(inputs=(1,), output=0, violated_postcondition="x")
    with pytest.raises(ProverError):
        ProofResult(
            task_name="t",
            verdict=ProofVerdict.PROVED_SOFT,
            samples_drawn=1,
            samples_satisfying_preconditions=1,
            counterexample=cx,
        )


def test_result_rejects_negative_samples_drawn() -> None:
    with pytest.raises(ProverError):
        ProofResult(
            task_name="t",
            verdict=ProofVerdict.PROVED_SOFT,
            samples_drawn=-1,
            samples_satisfying_preconditions=0,
        )


def test_result_rejects_pre_count_exceeding_drawn() -> None:
    with pytest.raises(ProverError):
        ProofResult(
            task_name="t",
            verdict=ProofVerdict.PROVED_SOFT,
            samples_drawn=1,
            samples_satisfying_preconditions=2,
        )


def test_result_rejects_empty_backend() -> None:
    with pytest.raises(ProverError):
        ProofResult(
            task_name="t",
            verdict=ProofVerdict.PROVED_SOFT,
            samples_drawn=1,
            samples_satisfying_preconditions=1,
            backend="",
        )


def test_result_rejects_empty_task_name() -> None:
    with pytest.raises(ProverError):
        ProofResult(
            task_name="",
            verdict=ProofVerdict.PROVED_SOFT,
            samples_drawn=1,
            samples_satisfying_preconditions=1,
        )


# ---------------------------------------------------------------------------
# prove() — stdlib backend
# ---------------------------------------------------------------------------


def test_prove_abs_is_always_nonneg() -> None:
    task = ProofTask(
        name="abs_nonneg",
        target=abs,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="out_ge_zero", predicate=lambda x, out: out >= 0),),
        max_samples=256,
    )
    r = prove(task, seed=42)
    assert r.verdict is ProofVerdict.PROVED_SOFT
    assert r.samples_drawn == 256
    assert r.samples_satisfying_preconditions == 256
    assert r.backend == "stdlib"


def test_prove_finds_counterexample_for_broken_postcondition() -> None:
    # Claim: x + x is always > 0 for every int. Counterexample: x = 0
    # (or any negative).
    task = ProofTask(
        name="double_strictly_positive",
        target=lambda x: x + x,
        arity=1,
        preconditions=(),
        postconditions=(
            PostCondition(
                name="strictly_positive",
                predicate=lambda x, out: out > 0,
            ),
        ),
        max_samples=512,
    )
    r = prove(task, seed=0)
    assert r.verdict is ProofVerdict.COUNTEREXAMPLE
    assert r.counterexample is not None
    assert r.counterexample.violated_postcondition == "strictly_positive"


def test_prove_respects_preconditions() -> None:
    task = ProofTask(
        name="positive_double",
        target=lambda x: x + x,
        arity=1,
        preconditions=(positive_int(),),
        postconditions=(PostCondition(name="positive_out", predicate=lambda x, out: out > 0),),
        max_samples=512,
    )
    r = prove(task, seed=99)
    assert r.verdict is ProofVerdict.PROVED_SOFT
    assert r.samples_satisfying_preconditions > 0


def test_prove_emits_precondition_unsat_when_no_input_qualifies() -> None:
    impossible = Invariant(
        name="impossible",
        arity=1,
        predicate=lambda x: False,
    )
    task = ProofTask(
        name="vacuous",
        target=lambda x: x,
        arity=1,
        preconditions=(impossible,),
        postconditions=_trivial_post(),
        max_samples=64,
    )
    r = prove(task, seed=0)
    assert r.verdict is ProofVerdict.PRECONDITION_UNSATISFIABLE
    assert r.samples_satisfying_preconditions == 0
    assert r.samples_drawn == 64


def test_prove_catches_target_raises_as_counterexample() -> None:
    def raising_target(x: int) -> int:
        if x % 7 == 0:
            raise ValueError("boom")
        return x

    task = ProofTask(
        name="raises_on_mod7",
        target=raising_target,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="anything", predicate=lambda x, out: True),),
        max_samples=1024,
    )
    r = prove(task, seed=0)
    assert r.verdict is ProofVerdict.COUNTEREXAMPLE
    assert r.counterexample is not None
    assert r.counterexample.violated_postcondition == "target_raised"


def test_prove_two_arity() -> None:
    task = ProofTask(
        name="add_commutative",
        target=lambda x, y: x + y,
        arity=2,
        preconditions=(),
        postconditions=(
            PostCondition(
                name="commutes",
                predicate=lambda x, y, out: out == y + x,
            ),
        ),
        max_samples=256,
    )
    r = prove(task, seed=7)
    assert r.verdict is ProofVerdict.PROVED_SOFT


def test_prove_rejects_non_task() -> None:
    with pytest.raises(TypeError):
        prove("not a task", seed=0)  # type: ignore[arg-type]


def test_prove_rejects_bool_seed() -> None:
    task = ProofTask(
        name="t",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=_trivial_post(),
    )
    with pytest.raises(TypeError):
        prove(task, seed=True)  # type: ignore[arg-type]


def test_prove_rejects_negative_seed() -> None:
    task = ProofTask(
        name="t",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=_trivial_post(),
    )
    with pytest.raises(ProverError):
        prove(task, seed=-1)


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _replay_task() -> ProofTask:
    return ProofTask(
        name="replay_target",
        target=lambda x, y: x * y,
        arity=2,
        preconditions=(),
        postconditions=(
            PostCondition(
                name="signs_match",
                predicate=lambda x, y, out: (x > 0 and y > 0) or out <= 0,
            ),
        ),
        max_samples=128,
    )


def test_inv15_three_run_byte_identical() -> None:
    task = _replay_task()
    r1 = prove(task, seed=11)
    r2 = prove(task, seed=11)
    r3 = prove(task, seed=11)
    assert r1 == r2 == r3


def test_inv15_different_seeds_can_diverge() -> None:
    task = _replay_task()
    r0 = prove(task, seed=0)
    r1 = prove(task, seed=1)
    # Either both PROVED_SOFT or one of them has a counterexample;
    # in either case the sample sequences must differ across seeds.
    if r0.verdict is r1.verdict is ProofVerdict.PROVED_SOFT:
        # Same verdict, but different sample paths → byte-different
        # only if a counterexample was found mid-run; here both
        # exhausted the budget, so samples_drawn match. The point is
        # that the seed plumbing is plumbed, not that verdicts diverge.
        assert r0.samples_drawn == r1.samples_drawn
    # Always must have non-equal seed-driven counterexample inputs when
    # both produce one.
    if r0.counterexample and r1.counterexample:
        assert r0.counterexample.inputs != r1.counterexample.inputs or (
            r0.counterexample.violated_postcondition != r1.counterexample.violated_postcondition
        )


# ---------------------------------------------------------------------------
# Suite / SuiteReport
# ---------------------------------------------------------------------------


def test_suite_constructs_valid() -> None:
    task = ProofTask(
        name="abs_nonneg",
        target=abs,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="nonneg", predicate=lambda x, out: out >= 0),),
        max_samples=16,
    )
    suite = ProofSuite(name="dix_invariants", tasks=(task,))
    assert suite.name == "dix_invariants"


def test_suite_is_frozen_and_slotted() -> None:
    suite = ProofSuite(name="x", tasks=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        suite.name = "y"  # type: ignore[misc]
    assert not hasattr(suite, "__dict__")


def test_suite_rejects_empty_name() -> None:
    with pytest.raises(ProverError):
        ProofSuite(name="", tasks=())


def test_suite_rejects_non_tuple_tasks() -> None:
    with pytest.raises(ProverError):
        ProofSuite(name="x", tasks=[])  # type: ignore[arg-type]


def test_suite_rejects_duplicate_task_names() -> None:
    task = ProofTask(
        name="dup",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=_trivial_post(),
        max_samples=4,
    )
    with pytest.raises(ProverError):
        ProofSuite(name="x", tasks=(task, task))


def test_suite_rejects_non_task_entry() -> None:
    with pytest.raises(ProverError):
        ProofSuite(name="x", tasks=("not a task",))  # type: ignore[arg-type]


def test_prove_suite_runs_every_task() -> None:
    task_a = ProofTask(
        name="abs_nonneg",
        target=abs,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="nonneg", predicate=lambda x, out: out >= 0),),
        max_samples=16,
    )
    task_b = ProofTask(
        name="identity_invariant",
        target=lambda x: x,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="equal", predicate=lambda x, out: out == x),),
        max_samples=16,
    )
    suite = ProofSuite(name="batch", tasks=(task_a, task_b))
    report = prove_suite(suite, seed=0)
    assert report.suite_name == "batch"
    assert len(report.results) == 2
    assert report.all_clear()
    assert report.counterexamples() == ()


def test_prove_suite_reports_counterexamples() -> None:
    task_good = ProofTask(
        name="abs_nonneg",
        target=abs,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="nonneg", predicate=lambda x, out: out >= 0),),
        max_samples=64,
    )
    task_bad = ProofTask(
        name="claim_strictly_positive",
        target=lambda x: x + x,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="strictly_positive", predicate=lambda x, out: out > 0),),
        max_samples=256,
    )
    suite = ProofSuite(name="mixed", tasks=(task_good, task_bad))
    report = prove_suite(suite, seed=0)
    assert not report.all_clear()
    cxs = report.counterexamples()
    assert len(cxs) == 1
    assert cxs[0].task_name == "claim_strictly_positive"


def test_prove_suite_rejects_non_suite() -> None:
    with pytest.raises(TypeError):
        prove_suite("not a suite", seed=0)  # type: ignore[arg-type]


def test_prove_suite_rejects_bool_seed() -> None:
    suite = ProofSuite(name="x", tasks=())
    with pytest.raises(TypeError):
        prove_suite(suite, seed=True)  # type: ignore[arg-type]


def test_prove_suite_rejects_negative_seed() -> None:
    suite = ProofSuite(name="x", tasks=())
    with pytest.raises(ProverError):
        prove_suite(suite, seed=-1)


def test_prove_suite_byte_identical_replay() -> None:
    task_a = ProofTask(
        name="t1",
        target=lambda x: x * 2,
        arity=1,
        preconditions=(),
        postconditions=(PostCondition(name="double", predicate=lambda x, out: out == 2 * x),),
        max_samples=32,
    )
    task_b = ProofTask(
        name="t2",
        target=lambda x, y: x + y,
        arity=2,
        preconditions=(),
        postconditions=(PostCondition(name="sum", predicate=lambda x, y, out: out == x + y),),
        max_samples=32,
    )
    suite = ProofSuite(name="repro", tasks=(task_a, task_b))
    r1 = prove_suite(suite, seed=3)
    r2 = prove_suite(suite, seed=3)
    r3 = prove_suite(suite, seed=3)
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# Convenience pre-condition factories
# ---------------------------------------------------------------------------


def test_positive_int_factory_accepts_positive_only() -> None:
    inv = positive_int()
    assert inv.predicate(1) is True
    assert inv.predicate(0) is False
    assert inv.predicate(-1) is False
    assert inv.predicate(True) is False  # bool rejected


def test_nonneg_int_factory_accepts_zero() -> None:
    inv = nonneg_int()
    assert inv.predicate(0) is True
    assert inv.predicate(1) is True
    assert inv.predicate(-1) is False


def test_bounded_int_factory_inclusive_bounds() -> None:
    inv = bounded_int(low=-5, high=5)
    assert inv.predicate(-5) is True
    assert inv.predicate(0) is True
    assert inv.predicate(5) is True
    assert inv.predicate(6) is False
    assert inv.predicate(-6) is False


def test_bounded_int_factory_rejects_inverted_bounds() -> None:
    with pytest.raises(ProverError):
        bounded_int(low=5, high=-5)


def test_bounded_int_factory_rejects_bool_bounds() -> None:
    with pytest.raises(ProverError):
        bounded_int(low=True, high=1)  # type: ignore[arg-type]


def test_bounded_int_factory_default_name_is_informative() -> None:
    inv = bounded_int(low=0, high=10)
    assert "bounded_int[0..10]" == inv.name


def test_bounded_int_factory_custom_name() -> None:
    inv = bounded_int(low=0, high=10, name="custom_name")
    assert inv.name == "custom_name"


# ---------------------------------------------------------------------------
# Lazy seam — CrossHair backend
# ---------------------------------------------------------------------------


def test_enable_crosshair_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("crosshair")
    except ImportError:
        with pytest.raises(ImportError, match="crosshair"):
            enable_crosshair_factory()
    else:
        # If crosshair is installed, the factory must succeed.
        prover = enable_crosshair_factory()
        assert callable(prover)


def test_enable_crosshair_factory_rejects_unknown_overrides() -> None:
    try:
        importlib.import_module("crosshair")
    except ImportError:
        pytest.skip("crosshair not installed")
    with pytest.raises(ProverError):
        enable_crosshair_factory(overrides={"unknown_key": 1})


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "invariant_prover.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_crosshair_import() -> None:
    assert all(not name.startswith("crosshair") for name in _top_level_imports(_module_ast()))


def test_no_top_level_hypothesis_import() -> None:
    assert all(not name.startswith("hypothesis") for name in _top_level_imports(_module_ast()))


def test_no_top_level_z3_import() -> None:
    assert all(not name.startswith("z3") for name in _top_level_imports(_module_ast()))


def test_no_top_level_time_or_random_import() -> None:
    banned = {"time", "random", "datetime", "asyncio"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_io_imports() -> None:
    banned = {"subprocess", "socket", "urllib", "requests", "httpx", "aiohttp"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_engine_imports() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "core.contracts.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def test_crosshair_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith("crosshair"):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside enable_crosshair_factory"
                    )
                    assert parent.name == "enable_crosshair_factory", (
                        f"{name} imported in {parent.name!r} — must be "
                        "inside enable_crosshair_factory"
                    )


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


# ---------------------------------------------------------------------------
# Reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import tools.invariant_prover as mod1

    importlib.reload(mod1)
    import tools.invariant_prover as mod2

    assert mod1.PROVER_VERSION == mod2.PROVER_VERSION
    assert mod1.MAX_SAMPLES == mod2.MAX_SAMPLES
    assert mod1.ProofVerdict.PROVED is mod2.ProofVerdict.PROVED
