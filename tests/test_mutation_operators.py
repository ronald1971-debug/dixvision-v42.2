"""Tests for evolution_engine.genetic.mutation_operators (A-04.1)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from evolution_engine.genetic import mutation_operators as mo
from evolution_engine.genetic.mutation_operators import (
    MAX_META_KEY_LEN,
    MAX_META_KEYS,
    MAX_META_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    OPERATOR_DE_BINOMIAL_CROSSOVER,
    OPERATOR_DE_CURRENT_TO_BEST_1,
    OPERATOR_DE_RAND_1,
    OPERATOR_GAUSSIAN,
    OPERATOR_POLYNOMIAL,
    MutationOperatorError,
    de_binomial_crossover,
    de_current_to_best_1,
    de_rand_1,
    gaussian_mutate,
    polynomial_mutate,
)
from evolution_engine.genetic.strategy_chromosome import (
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
)

# ---------------------------------------------------------------------------
# Module file path / AST authority pins
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "evolution_engine" / "genetic" / "mutation_operators.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _three_kind_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="alpha", kind=ParameterKind.CONTINUOUS, low=-1.0, high=1.0),
        ParameterSpec(name="lr", kind=ParameterKind.LOG_CONTINUOUS, low=1e-4, high=1e-1),
        ParameterSpec(name="window", kind=ParameterKind.INTEGER, low=2.0, high=64.0),
    )


def _chromosome(values: tuple[float, ...], *, version: int = 0) -> StrategyChromosome:
    return StrategyChromosome(
        strategy_id="strat-A",
        specs=_three_kind_specs(),
        values=values,
        version=version,
    )


@pytest.fixture
def parent() -> StrategyChromosome:
    return _chromosome((0.25, 1e-3, 16.0))


@pytest.fixture
def parent_a() -> StrategyChromosome:
    return _chromosome((0.0, 1e-2, 8.0), version=2)


@pytest.fixture
def parent_b() -> StrategyChromosome:
    return _chromosome((-0.5, 1e-3, 32.0), version=1)


@pytest.fixture
def parent_c() -> StrategyChromosome:
    return _chromosome((0.5, 5e-3, 4.0), version=3)


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


def test_no_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_module_has_adapted_from_header() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: facebookresearch/nevergrad" in text
    assert "optimizerlib.py" in text


def test_operator_tags_distinct_strings() -> None:
    tags = {
        OPERATOR_GAUSSIAN,
        OPERATOR_POLYNOMIAL,
        OPERATOR_DE_RAND_1,
        OPERATOR_DE_CURRENT_TO_BEST_1,
        OPERATOR_DE_BINOMIAL_CROSSOVER,
    }
    assert len(tags) == 5
    for tag in tags:
        assert isinstance(tag, str) and tag


def test_meta_caps_positive() -> None:
    assert MAX_META_KEYS > 0
    assert MAX_META_KEY_LEN > 0
    assert MAX_META_VALUE_LEN > 0


def test_public_surface() -> None:
    assert set(mo.__all__) == {
        "MAX_META_KEYS",
        "MAX_META_KEY_LEN",
        "MAX_META_VALUE_LEN",
        "MutationOperatorError",
        "NEW_PIP_DEPENDENCIES",
        "OPERATOR_DE_BINOMIAL_CROSSOVER",
        "OPERATOR_DE_CURRENT_TO_BEST_1",
        "OPERATOR_DE_RAND_1",
        "OPERATOR_GAUSSIAN",
        "OPERATOR_POLYNOMIAL",
        "de_binomial_crossover",
        "de_current_to_best_1",
        "de_rand_1",
        "gaussian_mutate",
        "polynomial_mutate",
    }


def test_mutation_operator_error_is_value_error() -> None:
    assert issubclass(MutationOperatorError, ValueError)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def test_no_forbidden_imports() -> None:
    imports = _imports(_module_ast())
    forbidden = {
        "nevergrad",
        "numpy",
        "torch",
        "scipy",
        "evotorch",
        "deap",
        "pymoo",
        "random",
        "secrets",
        "os",
        "time",
        "datetime",
        "asyncio",
    }
    for f in forbidden:
        assert not any(imp == f or imp.startswith(f + ".") for imp in imports), (
            f"forbidden import found: {f}"
        )


def test_no_engine_cross_imports() -> None:
    imports = _imports(_module_ast())
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "ui",
    )
    for imp in imports:
        for prefix in forbidden_prefixes:
            assert not (imp == prefix or imp.startswith(prefix + ".")), (
                f"engine cross-import found: {imp}"
            )


def test_does_not_construct_patch_proposal() -> None:
    """B27 / B28 / INV-71: this leaf must not construct PatchProposal —
    only ``evolution_engine.patch_pipeline`` /
    ``evolution_engine.genetic.pipeline`` (A-04.2) does."""

    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "PatchProposal", "mutation_operators must not construct PatchProposal"


def test_no_clock_or_io() -> None:
    """No time / clock / file IO calls anywhere in the module."""

    tree = _module_ast()
    forbidden_attrs = {
        "time_ns",
        "monotonic_ns",
        "monotonic",
        "perf_counter",
        "now",
        "today",
        "open",
        "read",
        "write",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attrs or _attr_chain_starts_with(node, ("math",)), (
                f"forbidden attribute call: {ast.dump(node)}"
            )


def _attr_chain_starts_with(node: ast.Attribute, allowed_roots: tuple[str, ...]) -> bool:
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in allowed_roots


def test_module_is_pure_function_layer() -> None:
    """Top-level definitions are functions, frozen-stateless helpers,
    constants, or the error class — no mutable globals."""

    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.Expr, ast.AnnAssign, ast.Assign)):
            continue
        raise AssertionError(f"unexpected top-level node: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_splitmix64_is_deterministic() -> None:
    a = mo._splitmix64(0)
    b = mo._splitmix64(0)
    assert a == b
    assert mo._splitmix64(1) != mo._splitmix64(2)


def test_uniform01_in_open_unit_interval() -> None:
    for k in range(50):
        u = mo._uniform01(42, k, 7)
        assert 0.0 < u <= 1.0


def test_uniform01_deterministic() -> None:
    u1 = mo._uniform01(1, 2, 3, 4)
    u2 = mo._uniform01(1, 2, 3, 4)
    assert u1 == u2


def test_gauss_pair_independent_seeds() -> None:
    a = mo._gauss_pair(seed=1, generation=0, individual=0, dim=0)
    b = mo._gauss_pair(seed=2, generation=0, individual=0, dim=0)
    assert a != b


def test_gauss_pair_reproducible() -> None:
    a = mo._gauss_pair(seed=11, generation=5, individual=3, dim=1)
    b = mo._gauss_pair(seed=11, generation=5, individual=3, dim=1)
    assert a == b


# ---------------------------------------------------------------------------
# Encoded-bounds helpers
# ---------------------------------------------------------------------------


def test_encoded_bounds_log_continuous() -> None:
    specs = _three_kind_specs()
    bounds = mo._encoded_bounds(specs)
    assert bounds[0] == (-1.0, 1.0)  # CONTINUOUS pass-through
    lo_log, hi_log = bounds[1]
    assert math.isclose(lo_log, math.log10(1e-4))
    assert math.isclose(hi_log, math.log10(1e-1))
    assert bounds[2] == (2.0, 64.0)  # INTEGER pass-through


def test_encode_decode_roundtrip_log() -> None:
    specs = _three_kind_specs()
    encoded = mo._encode_vector(specs, (0.25, 1e-3, 16.0))
    assert math.isclose(encoded[1], math.log10(1e-3))


# ---------------------------------------------------------------------------
# Gaussian mutate
# ---------------------------------------------------------------------------


def test_gaussian_mutate_returns_chromosome(parent: StrategyChromosome) -> None:
    out = gaussian_mutate(chromosome=parent, sigma=0.1, seed=1, generation=0, individual=0)
    assert isinstance(out, StrategyChromosome)
    assert out.strategy_id == parent.strategy_id
    assert out.specs == parent.specs
    assert out.version == parent.version + 1


def test_gaussian_mutate_in_bounds(parent: StrategyChromosome) -> None:
    for ind in range(20):
        out = gaussian_mutate(chromosome=parent, sigma=0.5, seed=42, generation=0, individual=ind)
        assert -1.0 <= out.values[0] <= 1.0
        assert 1e-4 <= out.values[1] <= 1e-1
        assert 2.0 <= out.values[2] <= 64.0
        assert int(out.values[2]) == out.values[2]


def test_gaussian_mutate_deterministic(parent: StrategyChromosome) -> None:
    a = gaussian_mutate(chromosome=parent, sigma=0.1, seed=7, generation=2, individual=3)
    b = gaussian_mutate(chromosome=parent, sigma=0.1, seed=7, generation=2, individual=3)
    assert a.values == b.values
    assert chromosome_digest(a) == chromosome_digest(b)


def test_gaussian_mutate_seed_divergence(parent: StrategyChromosome) -> None:
    a = gaussian_mutate(chromosome=parent, sigma=0.2, seed=1, generation=0, individual=0)
    b = gaussian_mutate(chromosome=parent, sigma=0.2, seed=2, generation=0, individual=0)
    assert a.values != b.values


def test_gaussian_mutate_individual_divergence(parent: StrategyChromosome) -> None:
    a = gaussian_mutate(chromosome=parent, sigma=0.2, seed=1, generation=0, individual=0)
    b = gaussian_mutate(chromosome=parent, sigma=0.2, seed=1, generation=0, individual=1)
    assert a.values != b.values


def test_gaussian_mutate_meta_carries_operator_tag(parent: StrategyChromosome) -> None:
    out = gaussian_mutate(chromosome=parent, sigma=0.1, seed=1, generation=0, individual=0)
    assert out.meta["operator"] == OPERATOR_GAUSSIAN
    assert out.meta["seed"] == "1"
    assert out.meta["generation"] == "0"
    assert out.meta["individual"] == "0"
    assert out.meta["parent_digest"] == chromosome_digest(parent)


def test_gaussian_mutate_zero_sigma_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(chromosome=parent, sigma=0.0, seed=1, generation=0, individual=0)


def test_gaussian_mutate_negative_sigma_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(chromosome=parent, sigma=-0.1, seed=1, generation=0, individual=0)


def test_gaussian_mutate_3run_replay_equality(parent: StrategyChromosome) -> None:
    digests = [
        chromosome_digest(
            gaussian_mutate(chromosome=parent, sigma=0.3, seed=99, generation=4, individual=2)
        )
        for _ in range(3)
    ]
    assert len(set(digests)) == 1


# ---------------------------------------------------------------------------
# Polynomial mutate
# ---------------------------------------------------------------------------


def test_polynomial_mutate_in_bounds(parent: StrategyChromosome) -> None:
    for ind in range(30):
        out = polynomial_mutate(chromosome=parent, eta=20.0, seed=11, generation=0, individual=ind)
        assert -1.0 <= out.values[0] <= 1.0
        assert 1e-4 <= out.values[1] <= 1e-1
        assert 2.0 <= out.values[2] <= 64.0


def test_polynomial_mutate_deterministic(parent: StrategyChromosome) -> None:
    a = polynomial_mutate(chromosome=parent, eta=20.0, seed=1, generation=0, individual=0)
    b = polynomial_mutate(chromosome=parent, eta=20.0, seed=1, generation=0, individual=0)
    assert a.values == b.values


def test_polynomial_mutate_concentrated_at_high_eta(parent: StrategyChromosome) -> None:
    """Larger eta concentrates the mutation closer to the parent."""

    parent_alpha = parent.values[0]
    high_eta_dev: list[float] = []
    low_eta_dev: list[float] = []
    for ind in range(50):
        a = polynomial_mutate(chromosome=parent, eta=200.0, seed=5, generation=0, individual=ind)
        b = polynomial_mutate(chromosome=parent, eta=2.0, seed=5, generation=0, individual=ind)
        high_eta_dev.append(abs(a.values[0] - parent_alpha))
        low_eta_dev.append(abs(b.values[0] - parent_alpha))
    assert sum(high_eta_dev) / len(high_eta_dev) < sum(low_eta_dev) / len(low_eta_dev)


def test_polynomial_mutate_meta_tag(parent: StrategyChromosome) -> None:
    out = polynomial_mutate(chromosome=parent, eta=20.0, seed=1, generation=0, individual=0)
    assert out.meta["operator"] == OPERATOR_POLYNOMIAL
    assert out.meta["parent_digest"] == chromosome_digest(parent)


def test_polynomial_mutate_zero_eta_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        polynomial_mutate(chromosome=parent, eta=0.0, seed=1, generation=0, individual=0)


def test_polynomial_mutate_3run_replay_equality(parent: StrategyChromosome) -> None:
    digests = [
        chromosome_digest(
            polynomial_mutate(chromosome=parent, eta=20.0, seed=33, generation=1, individual=4)
        )
        for _ in range(3)
    ]
    assert len(set(digests)) == 1


# ---------------------------------------------------------------------------
# DE/rand/1
# ---------------------------------------------------------------------------


def test_de_rand_1_returns_chromosome(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    out = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=1, generation=0, individual=0)
    assert isinstance(out, StrategyChromosome)
    assert out.strategy_id == parent_a.strategy_id
    assert out.specs == parent_a.specs
    assert out.version == max(parent_a.version, parent_b.version, parent_c.version) + 1


def test_de_rand_1_in_bounds(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    for f in (0.1, 0.5, 0.9, 1.5):
        out = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=f, seed=1, generation=0, individual=0)
        assert -1.0 <= out.values[0] <= 1.0
        assert 1e-4 <= out.values[1] <= 1e-1
        assert 2.0 <= out.values[2] <= 64.0


def test_de_rand_1_deterministic(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    a = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=7, generation=2, individual=3)
    b = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=7, generation=2, individual=3)
    assert a.values == b.values


def test_de_rand_1_meta_records_three_parents(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    out = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=1, generation=0, individual=0)
    assert out.meta["operator"] == OPERATOR_DE_RAND_1
    assert out.meta["parent_a_digest"] == chromosome_digest(parent_a)
    assert out.meta["parent_b_digest"] == chromosome_digest(parent_b)
    assert out.meta["parent_c_digest"] == chromosome_digest(parent_c)


def test_de_rand_1_strategy_id_mismatch_rejected(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
) -> None:
    other = StrategyChromosome(
        strategy_id="strat-Z",
        specs=parent_a.specs,
        values=parent_a.values,
        version=parent_a.version,
    )
    with pytest.raises(MutationOperatorError):
        de_rand_1(a=parent_a, b=parent_b, c=other, F=0.5, seed=1, generation=0, individual=0)


def test_de_rand_1_specs_mismatch_rejected(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
) -> None:
    other_specs = (ParameterSpec(name="alpha", kind=ParameterKind.CONTINUOUS, low=-5.0, high=5.0),)
    other = StrategyChromosome(
        strategy_id=parent_a.strategy_id,
        specs=other_specs,
        values=(0.0,),
        version=0,
    )
    with pytest.raises(MutationOperatorError):
        de_rand_1(a=parent_a, b=parent_b, c=other, F=0.5, seed=1, generation=0, individual=0)


def test_de_rand_1_zero_F_rejected(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    with pytest.raises(MutationOperatorError):
        de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.0, seed=1, generation=0, individual=0)


def test_de_rand_1_3run_replay_equality(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    digests = [
        chromosome_digest(
            de_rand_1(
                a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=99, generation=4, individual=2
            )
        )
        for _ in range(3)
    ]
    assert len(set(digests)) == 1


def test_de_rand_1_equal_parents_b_c_yields_a(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
) -> None:
    """If b == c, then ``F * (b - c) == 0`` so the mutant equals a."""

    out = de_rand_1(a=parent_a, b=parent_b, c=parent_b, F=0.7, seed=1, generation=0, individual=0)
    for spec, va, vo in zip(parent_a.specs, parent_a.values, out.values, strict=True):
        if spec.kind is ParameterKind.INTEGER:
            assert vo == va
        elif spec.kind is ParameterKind.LOG_CONTINUOUS:
            assert math.isclose(vo, va, rel_tol=1e-9, abs_tol=1e-12)
        else:
            assert math.isclose(vo, va, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# DE/current-to-best/1
# ---------------------------------------------------------------------------


def test_de_current_to_best_1_in_bounds(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    for ind in range(20):
        out = de_current_to_best_1(
            target=parent,
            best=parent_c,
            a=parent_a,
            b=parent_b,
            F=0.5,
            seed=1,
            generation=0,
            individual=ind,
        )
        assert -1.0 <= out.values[0] <= 1.0
        assert 1e-4 <= out.values[1] <= 1e-1
        assert 2.0 <= out.values[2] <= 64.0


def test_de_current_to_best_1_deterministic(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    a = de_current_to_best_1(
        target=parent,
        best=parent_c,
        a=parent_a,
        b=parent_b,
        F=0.5,
        seed=7,
        generation=2,
        individual=3,
    )
    b = de_current_to_best_1(
        target=parent,
        best=parent_c,
        a=parent_a,
        b=parent_b,
        F=0.5,
        seed=7,
        generation=2,
        individual=3,
    )
    assert a.values == b.values


def test_de_current_to_best_1_meta_records_four_parents(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    out = de_current_to_best_1(
        target=parent,
        best=parent_c,
        a=parent_a,
        b=parent_b,
        F=0.5,
        seed=1,
        generation=0,
        individual=0,
    )
    assert out.meta["operator"] == OPERATOR_DE_CURRENT_TO_BEST_1
    assert out.meta["target_digest"] == chromosome_digest(parent)
    assert out.meta["best_digest"] == chromosome_digest(parent_c)
    assert out.meta["parent_a_digest"] == chromosome_digest(parent_a)
    assert out.meta["parent_b_digest"] == chromosome_digest(parent_b)


def test_de_current_to_best_1_target_eq_best_a_eq_b_yields_target(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
) -> None:
    out = de_current_to_best_1(
        target=parent,
        best=parent,
        a=parent_a,
        b=parent_a,
        F=0.7,
        seed=1,
        generation=0,
        individual=0,
    )
    for spec, vt, vo in zip(parent.specs, parent.values, out.values, strict=True):
        if spec.kind is ParameterKind.INTEGER:
            assert vo == vt
        else:
            assert math.isclose(vo, vt, rel_tol=1e-9, abs_tol=1e-12)


def test_de_current_to_best_1_specs_mismatch_rejected(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
) -> None:
    other_specs = (ParameterSpec(name="alpha", kind=ParameterKind.CONTINUOUS, low=0.0, high=2.0),)
    bad = StrategyChromosome(
        strategy_id=parent.strategy_id,
        specs=other_specs,
        values=(1.0,),
        version=0,
    )
    with pytest.raises(MutationOperatorError):
        de_current_to_best_1(
            target=parent,
            best=bad,
            a=parent_a,
            b=parent_b,
            F=0.5,
            seed=1,
            generation=0,
            individual=0,
        )


def test_de_current_to_best_1_3run_replay_equality(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    digests = [
        chromosome_digest(
            de_current_to_best_1(
                target=parent,
                best=parent_c,
                a=parent_a,
                b=parent_b,
                F=0.5,
                seed=99,
                generation=4,
                individual=2,
            )
        )
        for _ in range(3)
    ]
    assert len(set(digests)) == 1


# ---------------------------------------------------------------------------
# DE binomial crossover
# ---------------------------------------------------------------------------


def test_de_binomial_crossover_returns_chromosome(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    out = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.5, seed=1, generation=0, individual=0
    )
    assert isinstance(out, StrategyChromosome)
    assert out.strategy_id == parent.strategy_id
    assert out.specs == parent.specs
    assert out.version == max(parent.version, parent_c.version) + 1


def test_de_binomial_crossover_at_least_one_donor_dim(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    """CR=0 should still pick exactly one donor dim (the forced index)."""

    out = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.0, seed=1, generation=0, individual=0
    )
    diffs = sum(1 for vt, vo in zip(parent.values, out.values, strict=True) if vt != vo)
    assert diffs == 1


def test_de_binomial_crossover_cr_one_takes_all_donor(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    """CR=1 — every dim is donor (forced index is also donor)."""

    out = de_binomial_crossover(
        target=parent, donor=parent_c, CR=1.0, seed=1, generation=0, individual=0
    )
    assert out.values == parent_c.values


def test_de_binomial_crossover_deterministic(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    a = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.5, seed=1, generation=0, individual=0
    )
    b = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.5, seed=1, generation=0, individual=0
    )
    assert a.values == b.values


def test_de_binomial_crossover_meta_tag(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    out = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.5, seed=1, generation=0, individual=0
    )
    assert out.meta["operator"] == OPERATOR_DE_BINOMIAL_CROSSOVER
    assert out.meta["target_digest"] == chromosome_digest(parent)
    assert out.meta["donor_digest"] == chromosome_digest(parent_c)


def test_de_binomial_crossover_cr_out_of_range(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    with pytest.raises(MutationOperatorError):
        de_binomial_crossover(
            target=parent, donor=parent_c, CR=1.5, seed=1, generation=0, individual=0
        )
    with pytest.raises(MutationOperatorError):
        de_binomial_crossover(
            target=parent, donor=parent_c, CR=-0.1, seed=1, generation=0, individual=0
        )


def test_de_binomial_crossover_3run_replay_equality(
    parent: StrategyChromosome, parent_c: StrategyChromosome
) -> None:
    digests = [
        chromosome_digest(
            de_binomial_crossover(
                target=parent,
                donor=parent_c,
                CR=0.5,
                seed=99,
                generation=4,
                individual=2,
            )
        )
        for _ in range(3)
    ]
    assert len(set(digests)) == 1


def test_de_binomial_crossover_specs_mismatch_rejected(
    parent: StrategyChromosome,
) -> None:
    other_specs = (ParameterSpec(name="alpha", kind=ParameterKind.CONTINUOUS, low=-1.0, high=1.0),)
    bad = StrategyChromosome(
        strategy_id=parent.strategy_id, specs=other_specs, values=(0.0,), version=0
    )
    with pytest.raises(MutationOperatorError):
        de_binomial_crossover(target=parent, donor=bad, CR=0.5, seed=1, generation=0, individual=0)


# ---------------------------------------------------------------------------
# Extra-meta validation
# ---------------------------------------------------------------------------


def test_extra_meta_pass_through(parent: StrategyChromosome) -> None:
    out = gaussian_mutate(
        chromosome=parent,
        sigma=0.1,
        seed=1,
        generation=0,
        individual=0,
        extra_meta={"trial": "T-001", "harness": "sim"},
    )
    assert out.meta["trial"] == "T-001"
    assert out.meta["harness"] == "sim"


def test_extra_meta_reserved_key_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"operator": "spoof"},
        )


def test_extra_meta_too_many_keys_rejected(parent: StrategyChromosome) -> None:
    too_many = {f"k{i}": "v" for i in range(MAX_META_KEYS + 1)}
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
            extra_meta=too_many,
        )


def test_extra_meta_long_key_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"k" * (MAX_META_KEY_LEN + 1): "v"},
        )


def test_extra_meta_long_value_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"k": "v" * (MAX_META_VALUE_LEN + 1)},
        )


def test_extra_meta_non_string_value_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
            extra_meta={"k": 1},  # type: ignore[dict-item]
        )


def test_meta_is_immutable(parent: StrategyChromosome) -> None:
    out = gaussian_mutate(chromosome=parent, sigma=0.1, seed=1, generation=0, individual=0)
    with pytest.raises((TypeError, AttributeError)):
        out.meta["evil"] = "x"  # type: ignore[index]


def test_meta_keys_sorted_for_determinism(parent: StrategyChromosome) -> None:
    out = gaussian_mutate(
        chromosome=parent,
        sigma=0.1,
        seed=1,
        generation=0,
        individual=0,
        extra_meta={"zeta": "z", "alpha": "a", "mu": "m"},
    )
    keys = list(out.meta.keys())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


def test_chromosome_type_validation(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome="not-a-chromosome",  # type: ignore[arg-type]
            sigma=0.1,
            seed=1,
            generation=0,
            individual=0,
        )


def test_seed_type_validation(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=1.5,  # type: ignore[arg-type]
            generation=0,
            individual=0,
        )


def test_negative_generation_rejected(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(chromosome=parent, sigma=0.1, seed=1, generation=-1, individual=0)


def test_bool_not_accepted_as_int(parent: StrategyChromosome) -> None:
    with pytest.raises(MutationOperatorError):
        gaussian_mutate(
            chromosome=parent,
            sigma=0.1,
            seed=True,  # type: ignore[arg-type]
            generation=0,
            individual=0,
        )


# ---------------------------------------------------------------------------
# Integer round-trip (banker's rounding via unpack)
# ---------------------------------------------------------------------------


def test_integer_kind_always_integer_valued(parent: StrategyChromosome) -> None:
    for ind in range(50):
        out = gaussian_mutate(chromosome=parent, sigma=0.4, seed=2024, generation=0, individual=ind)
        assert int(out.values[2]) == out.values[2]


def test_polynomial_integer_kind_always_integer_valued(parent: StrategyChromosome) -> None:
    for ind in range(50):
        out = polynomial_mutate(
            chromosome=parent, eta=20.0, seed=2024, generation=0, individual=ind
        )
        assert int(out.values[2]) == out.values[2]


# ---------------------------------------------------------------------------
# Cross-operator independence
# ---------------------------------------------------------------------------


def test_operators_emit_distinct_meta_tags(
    parent: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    parent_c: StrategyChromosome,
) -> None:
    g = gaussian_mutate(chromosome=parent, sigma=0.1, seed=1, generation=0, individual=0)
    p = polynomial_mutate(chromosome=parent, eta=20.0, seed=1, generation=0, individual=0)
    d1 = de_rand_1(a=parent_a, b=parent_b, c=parent_c, F=0.5, seed=1, generation=0, individual=0)
    d2 = de_current_to_best_1(
        target=parent,
        best=parent_c,
        a=parent_a,
        b=parent_b,
        F=0.5,
        seed=1,
        generation=0,
        individual=0,
    )
    bx = de_binomial_crossover(
        target=parent, donor=parent_c, CR=0.5, seed=1, generation=0, individual=0
    )
    tags = {
        g.meta["operator"],
        p.meta["operator"],
        d1.meta["operator"],
        d2.meta["operator"],
        bx.meta["operator"],
    }
    assert tags == {
        OPERATOR_GAUSSIAN,
        OPERATOR_POLYNOMIAL,
        OPERATOR_DE_RAND_1,
        OPERATOR_DE_CURRENT_TO_BEST_1,
        OPERATOR_DE_BINOMIAL_CROSSOVER,
    }
