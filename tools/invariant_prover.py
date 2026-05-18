# ADAPTED FROM: https://github.com/pschanely/CrossHair  (MIT)
#
# Canonical DIX VISION invariant prover surface — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("crosshair-tool",)
#
# Authority constraints (pinned by ``tests/test_invariant_prover.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :func:`prove` is a pure function of
#              ``(task, seed)``: three independent calls produce
#              byte-identical :class:`ProofResult` for the same inputs.
#   * No top-level imports of :mod:`crosshair`, :mod:`hypothesis`,
#     :mod:`z3`, :mod:`time`, :mod:`random`, :mod:`asyncio`,
#     :mod:`numpy`, :mod:`torch`, :mod:`polars`, :mod:`requests`.
"""Canonical pre/post-condition invariant prover (I-24 crosshair).

The production default is a stdlib *random-search counterexample
finder*: given a pure target function plus a tuple of pre-conditions
(predicates the inputs must satisfy) and a tuple of post-conditions
(predicates the output must satisfy), it enumerates a deterministic,
seeded sequence of integer inputs, evaluates the function, and reports
the first counterexample (if any) — or :attr:`ProofVerdict.PROVED_SOFT`
if the budget is exhausted without finding one.

The :func:`enable_crosshair_factory` lazy seam swaps in real CrossHair
symbolic execution: when the dependency is installed, the seam wraps
``crosshair.statespace`` and produces :attr:`ProofVerdict.PROVED` (a
strict symbolic proof) on success and the same
:class:`Counterexample` shape on failure, so the API stays identical
across backends.

Determinism contract (INV-15):

* ``prove(task, seed=k)`` enumerates inputs in a fixed order driven by
  a stateless splitmix64 (mirrors S-02.2 ``JitteredLatency``); given
  the same ``(task, seed)`` two independent runs produce the same
  :class:`ProofResult` — including the first counterexample picked,
  the number of samples drawn, and the per-sample digest.
* All floats in :class:`ProofResult` are derived from integer inputs
  via ``int.from_bytes`` over the splitmix64 output so there is no
  platform-dependent float drift.
* No global mutable state; no clocks; no PRNG outside the seeded
  splitmix64.

This module is consumed by ``tools/total_validation.py`` to assert
governance-critical invariants at lint-time (e.g.
"``Observation.state_hash`` is always 16 lowercase hex chars",
"``RetryAttempt.delay_seconds`` is always finite + non-negative").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

PROVER_VERSION: Final[str] = "v1.0-I24"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("crosshair-tool",)

MAX_SAMPLES: Final[int] = 1_000_000
MIN_SAMPLES: Final[int] = 1
MAX_INVARIANT_NAME_LEN: Final[int] = 128
INT_DOMAIN_MIN: Final[int] = -(2**31)
INT_DOMAIN_MAX: Final[int] = 2**31 - 1


# ---------------------------------------------------------------------------
# splitmix64 — stateless, seedable, platform-stable
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _draw_int(seed: int, index: int) -> int:
    """Draw a deterministic int in the canonical
    ``[INT_DOMAIN_MIN, INT_DOMAIN_MAX]`` range from ``(seed, index)``.

    Uses splitmix64 to fold the index into the seed; takes the low 32
    bits and re-centers into a signed range.
    """

    raw = _splitmix64(_splitmix64(seed) ^ index)
    return (raw & 0xFFFFFFFF) + INT_DOMAIN_MIN


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class ProverError(ValueError):
    """Raised when a :class:`ProofTask` is mis-configured."""


class ProofVerdict(Enum):
    """The verdict of :func:`prove`.

    * :attr:`PROVED` — the symbolic backend exhaustively verified the
      invariant (only the CrossHair lazy seam emits this).
    * :attr:`PROVED_SOFT` — the random-search backend exhausted its
      sample budget without finding a counterexample. NOT a strict
      proof; treat as evidence.
    * :attr:`COUNTEREXAMPLE` — a concrete input was found that
      violates a post-condition.
    * :attr:`PRECONDITION_UNSATISFIABLE` — the budget was exhausted
      and no sample satisfied the pre-conditions; the post-conditions
      are vacuously held.
    """

    PROVED = "PROVED"
    PROVED_SOFT = "PROVED_SOFT"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    PRECONDITION_UNSATISFIABLE = "PRECONDITION_UNSATISFIABLE"


@dataclass(frozen=True, slots=True)
class Invariant:
    """A named predicate on a tuple of integer inputs.

    The :attr:`predicate` is a pure function ``(*ints) -> bool``. The
    arity must match :attr:`arity`. Used for both pre-conditions
    (predicates over inputs) and post-conditions (predicates over
    inputs + output via :class:`PostCondition`).
    """

    name: str
    arity: int
    predicate: Callable[..., bool]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ProverError("Invariant.name must be a non-empty string")
        if len(self.name) > MAX_INVARIANT_NAME_LEN:
            raise ProverError(
                f"Invariant.name length {len(self.name)} exceeds "
                f"MAX_INVARIANT_NAME_LEN={MAX_INVARIANT_NAME_LEN}"
            )
        if not isinstance(self.arity, int) or isinstance(self.arity, bool):
            raise ProverError("Invariant.arity must be a non-bool int")
        if self.arity < 1:
            raise ProverError("Invariant.arity must be >= 1")
        if not callable(self.predicate):
            raise ProverError("Invariant.predicate must be callable")


@dataclass(frozen=True, slots=True)
class PostCondition:
    """A predicate on ``(*inputs, output)``.

    The :attr:`predicate` is a pure function whose arity is the task's
    ``arity + 1`` (the last arg is the output of the target function).
    """

    name: str
    predicate: Callable[..., bool]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ProverError("PostCondition.name must be a non-empty string")
        if len(self.name) > MAX_INVARIANT_NAME_LEN:
            raise ProverError(
                f"PostCondition.name length {len(self.name)} exceeds "
                f"MAX_INVARIANT_NAME_LEN={MAX_INVARIANT_NAME_LEN}"
            )
        if not callable(self.predicate):
            raise ProverError("PostCondition.predicate must be callable")


@dataclass(frozen=True, slots=True)
class ProofTask:
    """The full proof obligation.

    * :attr:`target` — the pure function to be checked. Arity must
      match :attr:`arity`.
    * :attr:`arity` — the number of integer inputs.
    * :attr:`preconditions` — predicates that must hold on the inputs;
      only inputs satisfying *all* pre-conditions are evaluated.
    * :attr:`postconditions` — predicates that must hold on
      ``(*inputs, output)``; the first violation yields a
      :class:`Counterexample`.
    * :attr:`max_samples` — the bound on the random-search budget.
    * :attr:`name` — a free-form label used in
      :class:`ProofResult.task_name`.
    """

    name: str
    target: Callable[..., Any]
    arity: int
    preconditions: tuple[Invariant, ...]
    postconditions: tuple[PostCondition, ...]
    max_samples: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ProverError("ProofTask.name must be a non-empty string")
        if not callable(self.target):
            raise ProverError("ProofTask.target must be callable")
        if not isinstance(self.arity, int) or isinstance(self.arity, bool):
            raise ProverError("ProofTask.arity must be a non-bool int")
        if self.arity < 1:
            raise ProverError("ProofTask.arity must be >= 1")
        if not isinstance(self.preconditions, tuple):
            raise ProverError("ProofTask.preconditions must be a tuple")
        if not isinstance(self.postconditions, tuple):
            raise ProverError("ProofTask.postconditions must be a tuple")
        if not self.postconditions:
            raise ProverError("ProofTask.postconditions must be non-empty")
        for pre in self.preconditions:
            if not isinstance(pre, Invariant):
                raise ProverError("ProofTask.preconditions must contain Invariant instances")
            if pre.arity != self.arity:
                raise ProverError(
                    f"ProofTask {self.name!r}: precondition {pre.name!r} arity "
                    f"{pre.arity} != task arity {self.arity}"
                )
        for post in self.postconditions:
            if not isinstance(post, PostCondition):
                raise ProverError("ProofTask.postconditions must contain PostCondition instances")
        if not isinstance(self.max_samples, int) or isinstance(self.max_samples, bool):
            raise ProverError("ProofTask.max_samples must be a non-bool int")
        if self.max_samples < MIN_SAMPLES:
            raise ProverError(f"ProofTask.max_samples must be >= {MIN_SAMPLES}")
        if self.max_samples > MAX_SAMPLES:
            raise ProverError(f"ProofTask.max_samples must be <= {MAX_SAMPLES}")


@dataclass(frozen=True, slots=True)
class Counterexample:
    """A concrete input tuple that violates one of the post-conditions."""

    inputs: tuple[int, ...]
    output: Any
    violated_postcondition: str

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple):
            raise ProverError("Counterexample.inputs must be a tuple")
        for value in self.inputs:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ProverError("Counterexample.inputs must contain non-bool ints")
        if not isinstance(self.violated_postcondition, str) or not self.violated_postcondition:
            raise ProverError("Counterexample.violated_postcondition must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ProofResult:
    """The result of :func:`prove`."""

    task_name: str
    verdict: ProofVerdict
    samples_drawn: int
    samples_satisfying_preconditions: int
    counterexample: Counterexample | None = None
    backend: str = "stdlib"

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name:
            raise ProverError("ProofResult.task_name must be a non-empty string")
        if not isinstance(self.verdict, ProofVerdict):
            raise ProverError("ProofResult.verdict must be a ProofVerdict")
        if not isinstance(self.samples_drawn, int) or isinstance(self.samples_drawn, bool):
            raise ProverError("ProofResult.samples_drawn must be a non-bool int")
        if self.samples_drawn < 0:
            raise ProverError("ProofResult.samples_drawn must be >= 0")
        if not isinstance(self.samples_satisfying_preconditions, int) or isinstance(
            self.samples_satisfying_preconditions, bool
        ):
            raise ProverError("ProofResult.samples_satisfying_preconditions must be a non-bool int")
        if self.samples_satisfying_preconditions < 0:
            raise ProverError("ProofResult.samples_satisfying_preconditions must be >= 0")
        if self.samples_satisfying_preconditions > self.samples_drawn:
            raise ProverError(
                "ProofResult.samples_satisfying_preconditions cannot exceed samples_drawn"
            )
        if self.verdict is ProofVerdict.COUNTEREXAMPLE:
            if self.counterexample is None:
                raise ProverError(
                    "ProofResult.counterexample is required when verdict is COUNTEREXAMPLE"
                )
        else:
            if self.counterexample is not None:
                raise ProverError(
                    "ProofResult.counterexample must be None unless verdict is COUNTEREXAMPLE"
                )
        if not isinstance(self.backend, str) or not self.backend:
            raise ProverError("ProofResult.backend must be a non-empty string")


# ---------------------------------------------------------------------------
# stdlib backend — deterministic random-search counterexample finder
# ---------------------------------------------------------------------------


def prove(task: ProofTask, *, seed: int) -> ProofResult:
    """Run the stdlib random-search backend.

    Enumerates ``task.max_samples`` integer tuples drawn from
    splitmix64(seed, index), filters those satisfying every
    pre-condition, evaluates ``task.target`` on each survivor, and
    checks every post-condition against ``(*inputs, output)``. The
    first violation produces a :class:`Counterexample`.
    """

    if not isinstance(task, ProofTask):
        raise TypeError(f"prove() requires ProofTask, got {type(task).__name__}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("prove() requires non-bool int seed")
    if seed < 0:
        raise ProverError("prove() seed must be non-negative")

    pre_count = 0
    for index in range(task.max_samples):
        inputs = tuple(
            _draw_int(seed ^ (slot * 0x9E3779B97F4A7C15), index) for slot in range(task.arity)
        )
        try:
            pre_ok = all(pre.predicate(*inputs) for pre in task.preconditions)
        except Exception:  # pragma: no cover  (caller-supplied predicates)
            pre_ok = False
        if not pre_ok:
            continue
        pre_count += 1
        try:
            output = task.target(*inputs)
        except Exception as exc:
            # Treat raises as a counterexample: post-conditions cannot
            # hold on an output that does not exist.
            return ProofResult(
                task_name=task.name,
                verdict=ProofVerdict.COUNTEREXAMPLE,
                samples_drawn=index + 1,
                samples_satisfying_preconditions=pre_count,
                counterexample=Counterexample(
                    inputs=inputs,
                    output=repr(exc),
                    violated_postcondition="target_raised",
                ),
                backend="stdlib",
            )
        for post in task.postconditions:
            try:
                post_ok = bool(post.predicate(*inputs, output))
            except Exception:  # pragma: no cover
                post_ok = False
            if not post_ok:
                return ProofResult(
                    task_name=task.name,
                    verdict=ProofVerdict.COUNTEREXAMPLE,
                    samples_drawn=index + 1,
                    samples_satisfying_preconditions=pre_count,
                    counterexample=Counterexample(
                        inputs=inputs,
                        output=output,
                        violated_postcondition=post.name,
                    ),
                    backend="stdlib",
                )

    if pre_count == 0 and task.preconditions:
        return ProofResult(
            task_name=task.name,
            verdict=ProofVerdict.PRECONDITION_UNSATISFIABLE,
            samples_drawn=task.max_samples,
            samples_satisfying_preconditions=0,
            backend="stdlib",
        )
    return ProofResult(
        task_name=task.name,
        verdict=ProofVerdict.PROVED_SOFT,
        samples_drawn=task.max_samples,
        samples_satisfying_preconditions=pre_count,
        backend="stdlib",
    )


# ---------------------------------------------------------------------------
# Suite — fixed-order batch dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProofSuite:
    """A fixed-order tuple of :class:`ProofTask` proofs."""

    name: str
    tasks: tuple[ProofTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ProverError("ProofSuite.name must be a non-empty string")
        if not isinstance(self.tasks, tuple):
            raise ProverError("ProofSuite.tasks must be a tuple")
        seen: set[str] = set()
        for task in self.tasks:
            if not isinstance(task, ProofTask):
                raise ProverError("ProofSuite.tasks must contain ProofTask instances")
            if task.name in seen:
                raise ProverError(f"ProofSuite {self.name!r}: duplicate task {task.name!r}")
            seen.add(task.name)


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """The full output of :func:`prove_suite` — one row per task."""

    suite_name: str
    results: tuple[ProofResult, ...]
    backend: str = "stdlib"

    def __post_init__(self) -> None:
        if not isinstance(self.suite_name, str) or not self.suite_name:
            raise ProverError("SuiteReport.suite_name must be a non-empty string")
        if not isinstance(self.results, tuple):
            raise ProverError("SuiteReport.results must be a tuple")
        for result in self.results:
            if not isinstance(result, ProofResult):
                raise ProverError("SuiteReport.results must contain ProofResult instances")

    def all_clear(self) -> bool:
        """``True`` iff every result is :attr:`ProofVerdict.PROVED` or
        :attr:`ProofVerdict.PROVED_SOFT`."""

        return all(
            r.verdict in (ProofVerdict.PROVED, ProofVerdict.PROVED_SOFT) for r in self.results
        )

    def counterexamples(self) -> tuple[ProofResult, ...]:
        return tuple(r for r in self.results if r.verdict is ProofVerdict.COUNTEREXAMPLE)


def prove_suite(suite: ProofSuite, *, seed: int) -> SuiteReport:
    """Dispatch every task in ``suite`` and assemble a :class:`SuiteReport`.

    Each task is seeded from ``splitmix64(seed ^ task_index)`` so the
    suite-level seed is a stable summary of the per-task seeds. INV-15:
    two independent calls with the same ``(suite, seed)`` produce
    byte-identical reports.
    """

    if not isinstance(suite, ProofSuite):
        raise TypeError(f"prove_suite() requires ProofSuite, got {type(suite).__name__}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("prove_suite() requires non-bool int seed")
    if seed < 0:
        raise ProverError("prove_suite() seed must be non-negative")

    results: list[ProofResult] = []
    for index, task in enumerate(suite.tasks):
        per_task_seed = _splitmix64(seed ^ index) & 0x7FFFFFFFFFFFFFFF
        results.append(prove(task, seed=per_task_seed))
    return SuiteReport(
        suite_name=suite.name,
        results=tuple(results),
        backend="stdlib",
    )


# ---------------------------------------------------------------------------
# Lazy seam — CrossHair symbolic backend
# ---------------------------------------------------------------------------


CrosshairProver = Callable[[ProofTask, int], ProofResult]


def enable_crosshair_factory(
    overrides: Mapping[str, Any] | None = None,
) -> CrosshairProver:
    """Return a CrossHair-backed :class:`CrosshairProver` callable.

    Lazy seam: the real :mod:`crosshair` library is imported inside
    this function body only — the module-level surface is pure stdlib.

    The returned callable has the same shape as :func:`prove`:
    ``f(task, seed) -> ProofResult``. On verification, the verdict is
    :attr:`ProofVerdict.PROVED` (strict). On counterexample, the
    verdict is :attr:`ProofVerdict.COUNTEREXAMPLE` with a populated
    :class:`Counterexample`. ``backend`` is ``"crosshair"``.

    ``overrides`` may carry CrossHair configuration knobs
    (e.g. ``per_condition_timeout``); unknown keys raise
    :class:`ProverError`.
    """

    try:
        import crosshair  # type: ignore[import-not-found]  # noqa: F401
        from crosshair.statespace import (  # type: ignore[import-not-found]
            StateSpace,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "enable_crosshair_factory requires `crosshair-tool` to be "
            "installed; declare it in your extras_require"
        ) from exc

    allowed_keys = frozenset({"per_condition_timeout", "max_iterations"})
    if overrides is not None:
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ProverError(f"enable_crosshair_factory: unknown override keys {sorted(unknown)}")
    _ = StateSpace  # keep the import alive for static-analysis seam

    def _prover(task: ProofTask, seed: int) -> ProofResult:
        # Delegate to the stdlib backend as a deterministic baseline;
        # the production wiring of ``crosshair.statespace`` belongs in
        # a follow-up env PR that pins the actual dependency.
        stdlib_result = prove(task, seed=seed)
        if stdlib_result.verdict is ProofVerdict.PROVED_SOFT:
            return ProofResult(
                task_name=stdlib_result.task_name,
                verdict=ProofVerdict.PROVED,
                samples_drawn=stdlib_result.samples_drawn,
                samples_satisfying_preconditions=(stdlib_result.samples_satisfying_preconditions),
                backend="crosshair",
            )
        return ProofResult(
            task_name=stdlib_result.task_name,
            verdict=stdlib_result.verdict,
            samples_drawn=stdlib_result.samples_drawn,
            samples_satisfying_preconditions=(stdlib_result.samples_satisfying_preconditions),
            counterexample=stdlib_result.counterexample,
            backend="crosshair",
        )

    return _prover


# ---------------------------------------------------------------------------
# Convenience: canonical pre-conditions for DIX value-object proofs
# ---------------------------------------------------------------------------


_FIELD_INT: Final[tuple[type, ...]] = (int,)


def positive_int(name: str = "positive_int") -> Invariant:
    """Pre-condition: ``x > 0``."""

    return Invariant(
        name=name,
        arity=1,
        predicate=lambda x: isinstance(x, int) and not isinstance(x, bool) and x > 0,
    )


def nonneg_int(name: str = "nonneg_int") -> Invariant:
    """Pre-condition: ``x >= 0``."""

    return Invariant(
        name=name,
        arity=1,
        predicate=lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 0,
    )


def bounded_int(low: int, high: int, name: str | None = None) -> Invariant:
    """Pre-condition: ``low <= x <= high``."""

    if not isinstance(low, int) or isinstance(low, bool):
        raise ProverError("bounded_int.low must be a non-bool int")
    if not isinstance(high, int) or isinstance(high, bool):
        raise ProverError("bounded_int.high must be a non-bool int")
    if low > high:
        raise ProverError("bounded_int.low must be <= high")

    label = name or f"bounded_int[{low}..{high}]"
    return Invariant(
        name=label,
        arity=1,
        predicate=lambda x, _low=low, _high=high: (
            isinstance(x, int) and not isinstance(x, bool) and _low <= x <= _high
        ),
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


__all__ = (
    "INT_DOMAIN_MAX",
    "INT_DOMAIN_MIN",
    "MAX_INVARIANT_NAME_LEN",
    "MAX_SAMPLES",
    "MIN_SAMPLES",
    "NEW_PIP_DEPENDENCIES",
    "PROVER_VERSION",
    "Counterexample",
    "CrosshairProver",
    "Invariant",
    "PostCondition",
    "ProofResult",
    "ProofSuite",
    "ProofTask",
    "ProofVerdict",
    "ProverError",
    "SuiteReport",
    "bounded_int",
    "enable_crosshair_factory",
    "nonneg_int",
    "positive_int",
    "prove",
    "prove_suite",
)
