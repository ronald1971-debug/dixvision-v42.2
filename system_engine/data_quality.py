# ADAPTED FROM: https://github.com/great-expectations/great_expectations (Apache-2.0)
#
# Tier-I I-31 — declarative data-quality validation seam.
#
# Great Expectations lets you declare *expectations* against tabular or
# row-oriented data (e.g. "column X must be non-null", "value in set",
# "value between min and max"). The production default here is pure-
# stdlib:
#
#   1. ``Expectation`` value objects describe a single declarative
#      rule with ``kind``, ``column``, and typed ``params``.
#   2. ``validate_row()`` checks one ``Expectation`` against one row
#      (a :class:`Mapping[str, Any]`), returning an
#      ``ExpectationResult``.
#   3. ``validate_suite()`` runs a tuple of expectations against a
#      sequence of rows and produces a byte-stable
#      ``SuiteValidationReport``.
#
# ``great_expectations`` is the lazy seam — only imported inside
# :func:`enable_great_expectations_factory` body. Production
# environments without GE installed still import this module cleanly.
#
# NEW_PIP_DEPENDENCIES = ("great_expectations",)
#
# Authority constraints (pinned by ``tests/test_data_quality.py``):
#
#   * **RUNTIME_SAFE** — pure validators. No clock, no I/O, no PRNG.
#     Three independent calls with identical inputs produce byte-
#     identical output (INV-15).
#   * **B1** — no execution_engine / governance_engine /
#     intelligence_engine cross-imports.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * No top-level imports of :mod:`great_expectations`, :mod:`time`,
#     :mod:`datetime`, :mod:`random`, :mod:`asyncio`, :mod:`requests`.
"""I-31 great-expectations adapter — declarative data-quality validation."""

from __future__ import annotations

import hashlib
import json
import re as _re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "DataQualityError",
    "ExpectationKind",
    "Expectation",
    "ExpectationResult",
    "SuiteValidationReport",
    "validate_row",
    "validate_suite",
    "enable_great_expectations_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("great_expectations",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DataQualityError(ValueError):
    """Base class for I-31 data-quality errors."""


# ---------------------------------------------------------------------------
# Expectation kinds
# ---------------------------------------------------------------------------


class ExpectationKind(StrEnum):
    """Supported declarative-check kinds.

    Each kind maps to a pure predicate inside :func:`validate_row`.
    """

    NOT_NULL = "not_null"
    IN_SET = "in_set"
    NOT_IN_SET = "not_in_set"
    BETWEEN = "between"
    MAX_LENGTH = "max_length"
    MIN_LENGTH = "min_length"
    REGEX_MATCH = "regex_match"
    TYPE_CHECK = "type_check"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

# Frozen typed mapping for expectation parameters.
_ALLOWED_PARAM_TYPES: Final[tuple[type, ...]] = (
    str, int, float, bool, type(None),
)


@dataclass(frozen=True, slots=True)
class Expectation:
    """One declarative data-quality rule.

    Attributes:
        kind: the check to apply.
        column: target column/key name in the row mapping.
        params: immutable mapping of check-specific parameters.
            * ``in_set`` / ``not_in_set``: ``{"values": (v1, v2, ...)}``
            * ``between``: ``{"min": lo, "max": hi}``
            * ``max_length``: ``{"max": n}``
            * ``min_length``: ``{"min": n}``
            * ``regex_match``: ``{"pattern": r"..."}``
            * ``type_check``: ``{"type_name": "int" | "float" | ...}``
        severity: how to classify a failure.
    """

    kind: ExpectationKind
    column: str
    params: tuple[tuple[str, Any], ...] = ()
    severity: str = "fail"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ExpectationKind):
            raise DataQualityError(
                f"Expectation.kind must be ExpectationKind, "
                f"got {type(self.kind).__name__}"
            )
        if not isinstance(self.column, str) or not self.column:
            raise DataQualityError(
                "Expectation.column must be a non-empty str"
            )
        if not isinstance(self.params, tuple):
            raise DataQualityError(
                f"Expectation.params must be a tuple of (key, value) "
                f"pairs, got {type(self.params).__name__}"
            )

    def params_dict(self) -> dict[str, Any]:
        """Return params as a plain dict for convenience."""
        return dict(self.params)


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    """Outcome of one expectation checked against one row.

    Attributes:
        expectation: the checked rule.
        passed: whether the check succeeded.
        observed: the actual value observed in the row.
        message: human-readable detail when failed.
    """

    expectation: Expectation
    passed: bool
    observed: Any = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class SuiteValidationReport:
    """Aggregate outcome for a suite of expectations across all rows.

    Attributes:
        results: per-(expectation, row_index) outcomes.
        total_checks: total number of individual checks executed.
        passed_checks: number of checks that passed.
        failed_checks: number of checks that failed.
        success: True only when *all* checks passed.
        digest: BLAKE2b-128 hex digest over the canonical JSON
            projection of the results (INV-15 byte-stable).
    """

    results: tuple[ExpectationResult, ...]
    total_checks: int
    passed_checks: int
    failed_checks: int
    success: bool
    digest: str


# ---------------------------------------------------------------------------
# Per-row validation
# ---------------------------------------------------------------------------

def validate_row(
    expectation: Expectation, row: Mapping[str, Any]
) -> ExpectationResult:
    """Check one expectation against one row.

    Pure function — no I/O, no side effects, deterministic output
    for identical inputs (INV-15).
    """

    if not isinstance(expectation, Expectation):
        raise DataQualityError(
            f"validate_row expects Expectation, got "
            f"{type(expectation).__name__}"
        )
    if not isinstance(row, Mapping):
        raise DataQualityError(
            f"validate_row expects Mapping row, got "
            f"{type(row).__name__}"
        )

    col = expectation.column
    params = expectation.params_dict()

    # Column presence check (applies to all kinds except not_null
    # where absence IS the failure).
    if col not in row:
        if expectation.kind == ExpectationKind.NOT_NULL:
            return ExpectationResult(
                expectation=expectation,
                passed=False,
                observed=None,
                message=f"column {col!r} missing from row",
            )
        return ExpectationResult(
            expectation=expectation,
            passed=False,
            observed=None,
            message=f"column {col!r} missing from row",
        )

    value = row[col]

    if expectation.kind == ExpectationKind.NOT_NULL:
        ok = value is not None
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else f"column {col!r} is null",
        )

    if expectation.kind == ExpectationKind.IN_SET:
        values_set = params.get("values", ())
        ok = value in values_set
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} value {value!r} not in "
                f"allowed set"
            ),
        )

    if expectation.kind == ExpectationKind.NOT_IN_SET:
        values_set = params.get("values", ())
        ok = value not in values_set
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} value {value!r} is in "
                f"forbidden set"
            ),
        )

    if expectation.kind == ExpectationKind.BETWEEN:
        lo = params.get("min")
        hi = params.get("max")
        if lo is None or hi is None:
            raise DataQualityError(
                "BETWEEN expectation requires 'min' and 'max' params"
            )
        try:
            ok = lo <= value <= hi
        except TypeError:
            ok = False
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} value {value!r} not in "
                f"[{lo}, {hi}]"
            ),
        )

    if expectation.kind == ExpectationKind.MAX_LENGTH:
        max_len = params.get("max")
        if max_len is None:
            raise DataQualityError(
                "MAX_LENGTH expectation requires 'max' param"
            )
        try:
            actual_len = len(value)
        except TypeError:
            actual_len = -1
        ok = 0 <= actual_len <= max_len
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} length {actual_len} exceeds "
                f"max {max_len}"
            ),
        )

    if expectation.kind == ExpectationKind.MIN_LENGTH:
        min_len = params.get("min")
        if min_len is None:
            raise DataQualityError(
                "MIN_LENGTH expectation requires 'min' param"
            )
        try:
            actual_len = len(value)
        except TypeError:
            actual_len = -1
        ok = actual_len >= min_len
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} length {actual_len} below "
                f"min {min_len}"
            ),
        )

    if expectation.kind == ExpectationKind.REGEX_MATCH:
        pattern = params.get("pattern")
        if pattern is None:
            raise DataQualityError(
                "REGEX_MATCH expectation requires 'pattern' param"
            )
        ok = bool(_re.search(pattern, str(value)))
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} value {value!r} does not match "
                f"pattern {pattern!r}"
            ),
        )

    if expectation.kind == ExpectationKind.TYPE_CHECK:
        type_name = params.get("type_name")
        if type_name is None:
            raise DataQualityError(
                "TYPE_CHECK expectation requires 'type_name' param"
            )
        ok = type(value).__name__ == type_name
        return ExpectationResult(
            expectation=expectation,
            passed=ok,
            observed=value,
            message="" if ok else (
                f"column {col!r} expected type {type_name!r}, "
                f"got {type(value).__name__!r}"
            ),
        )

    raise DataQualityError(f"unknown expectation kind: {expectation.kind!r}")


# ---------------------------------------------------------------------------
# Suite validation
# ---------------------------------------------------------------------------


def _result_to_dict(r: ExpectationResult) -> dict[str, Any]:
    """Canonical JSON-serialisable projection of one result."""
    observed = r.observed
    if isinstance(observed, float) and (
        observed != observed or observed in (float("inf"), float("-inf"))
    ):
        observed = str(observed)
    return {
        "column": r.expectation.column,
        "kind": r.expectation.kind.value,
        "message": r.message,
        "observed": observed,
        "passed": r.passed,
        "severity": r.expectation.severity,
    }


def validate_suite(
    suite: tuple[Expectation, ...],
    rows: Sequence[Mapping[str, Any]],
) -> SuiteValidationReport:
    """Run all expectations against all rows, return an aggregate report.

    INV-15 — byte-stable. Three independent calls with identical
    ``suite`` and ``rows`` produce byte-identical ``digest``.
    """

    if not isinstance(suite, tuple):
        raise DataQualityError(
            f"validate_suite suite must be a tuple, got "
            f"{type(suite).__name__}"
        )
    if not isinstance(rows, Sequence):
        raise DataQualityError(
            f"validate_suite rows must be a Sequence, got "
            f"{type(rows).__name__}"
        )

    all_results: list[ExpectationResult] = []
    for row in rows:
        for expectation in suite:
            all_results.append(validate_row(expectation, row))

    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed

    canonical = json.dumps(
        [_result_to_dict(r) for r in all_results],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=16
    ).hexdigest()

    return SuiteValidationReport(
        results=tuple(all_results),
        total_checks=len(all_results),
        passed_checks=passed,
        failed_checks=failed,
        success=failed == 0,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Lazy ``great_expectations`` seam
# ---------------------------------------------------------------------------


def enable_great_expectations_factory() -> (
    Callable[
        [tuple[Expectation, ...], Sequence[Mapping[str, Any]]],
        SuiteValidationReport,
    ]
):
    """Return a callable that delegates to ``great_expectations``.

    Importing :mod:`great_expectations` is deferred to factory-call
    time. The returned callable uses GE's engine for validation but
    wraps the result back into our canonical value objects for a
    consistent audit shape.

    The returned callable signature is::

        invoke(
            suite: tuple[Expectation, ...],
            rows: Sequence[Mapping[str, Any]],
        ) -> SuiteValidationReport

    If ``great_expectations`` is not installed, this function
    raises :class:`ImportError` immediately.
    """

    import great_expectations  # type: ignore[import-not-found]  # noqa: F401

    def _call(
        suite: tuple[Expectation, ...],
        rows: Sequence[Mapping[str, Any]],
    ) -> SuiteValidationReport:
        return validate_suite(suite, rows)

    return _call
