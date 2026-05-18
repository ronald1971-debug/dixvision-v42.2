"""Tests for system_engine.data_quality (I-31 great-expectations adapter).

Pinned authority constraints:
* RUNTIME_SAFE — pure validators, no I/O / clock / PRNG.
* INV-15 — three independent calls produce byte-identical digests.
* B1 — no cross-tier engine imports.
* B27 / B28 / INV-71 — no typed-event constructors.
"""

from __future__ import annotations

from typing import Any

import pytest

from system_engine.data_quality import (
    NEW_PIP_DEPENDENCIES,
    DataQualityError,
    Expectation,
    ExpectationKind,
    SuiteValidationReport,
    validate_row,
    validate_suite,
)

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_new_pip_dependencies() -> None:
    assert "great_expectations" in NEW_PIP_DEPENDENCIES


# ---------------------------------------------------------------------------
# Expectation construction
# ---------------------------------------------------------------------------


def test_expectation_valid() -> None:
    e = Expectation(
        kind=ExpectationKind.NOT_NULL,
        column="price",
    )
    assert e.kind == ExpectationKind.NOT_NULL
    assert e.column == "price"
    assert e.params == ()


def test_expectation_with_params() -> None:
    e = Expectation(
        kind=ExpectationKind.IN_SET,
        column="side",
        params=(("values", ("BUY", "SELL")),),
    )
    assert e.params_dict() == {"values": ("BUY", "SELL")}


def test_expectation_bad_kind() -> None:
    with pytest.raises(DataQualityError, match="ExpectationKind"):
        Expectation(kind="unknown", column="x")  # type: ignore[arg-type]


def test_expectation_empty_column() -> None:
    with pytest.raises(DataQualityError, match="non-empty str"):
        Expectation(kind=ExpectationKind.NOT_NULL, column="")


def test_expectation_bad_params_type() -> None:
    with pytest.raises(DataQualityError, match="tuple"):
        Expectation(
            kind=ExpectationKind.NOT_NULL,
            column="x",
            params={"a": 1},  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# validate_row — NOT_NULL
# ---------------------------------------------------------------------------


def test_not_null_passes() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="price")
    r = validate_row(e, {"price": 42.0})
    assert r.passed is True
    assert r.observed == 42.0


def test_not_null_fails_on_none() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="price")
    r = validate_row(e, {"price": None})
    assert r.passed is False
    assert "null" in r.message


def test_not_null_fails_on_missing_column() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="price")
    r = validate_row(e, {"volume": 100})
    assert r.passed is False
    assert "missing" in r.message


# ---------------------------------------------------------------------------
# validate_row — IN_SET / NOT_IN_SET
# ---------------------------------------------------------------------------


def test_in_set_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.IN_SET,
        column="side",
        params=(("values", ("BUY", "SELL")),),
    )
    r = validate_row(e, {"side": "BUY"})
    assert r.passed is True


def test_in_set_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.IN_SET,
        column="side",
        params=(("values", ("BUY", "SELL")),),
    )
    r = validate_row(e, {"side": "HOLD"})
    assert r.passed is False


def test_not_in_set_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.NOT_IN_SET,
        column="status",
        params=(("values", ("DELETED", "ARCHIVED")),),
    )
    r = validate_row(e, {"status": "ACTIVE"})
    assert r.passed is True


def test_not_in_set_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.NOT_IN_SET,
        column="status",
        params=(("values", ("DELETED", "ARCHIVED")),),
    )
    r = validate_row(e, {"status": "DELETED"})
    assert r.passed is False


# ---------------------------------------------------------------------------
# validate_row — BETWEEN
# ---------------------------------------------------------------------------


def test_between_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.BETWEEN,
        column="score",
        params=(("min", 0.0), ("max", 1.0)),
    )
    r = validate_row(e, {"score": 0.5})
    assert r.passed is True


def test_between_boundary_inclusive() -> None:
    e = Expectation(
        kind=ExpectationKind.BETWEEN,
        column="score",
        params=(("min", 0), ("max", 100)),
    )
    assert validate_row(e, {"score": 0}).passed is True
    assert validate_row(e, {"score": 100}).passed is True


def test_between_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.BETWEEN,
        column="score",
        params=(("min", 0.0), ("max", 1.0)),
    )
    r = validate_row(e, {"score": 1.5})
    assert r.passed is False


def test_between_missing_params() -> None:
    e = Expectation(
        kind=ExpectationKind.BETWEEN,
        column="score",
    )
    with pytest.raises(DataQualityError, match="min.*max"):
        validate_row(e, {"score": 0.5})


# ---------------------------------------------------------------------------
# validate_row — MAX_LENGTH / MIN_LENGTH
# ---------------------------------------------------------------------------


def test_max_length_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.MAX_LENGTH,
        column="name",
        params=(("max", 10),),
    )
    r = validate_row(e, {"name": "alice"})
    assert r.passed is True


def test_max_length_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.MAX_LENGTH,
        column="name",
        params=(("max", 3),),
    )
    r = validate_row(e, {"name": "alice"})
    assert r.passed is False
    assert "exceeds" in r.message


def test_max_length_missing_param() -> None:
    e = Expectation(
        kind=ExpectationKind.MAX_LENGTH,
        column="name",
    )
    with pytest.raises(DataQualityError, match="max"):
        validate_row(e, {"name": "alice"})


def test_min_length_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.MIN_LENGTH,
        column="name",
        params=(("min", 3),),
    )
    r = validate_row(e, {"name": "alice"})
    assert r.passed is True


def test_min_length_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.MIN_LENGTH,
        column="name",
        params=(("min", 10),),
    )
    r = validate_row(e, {"name": "alice"})
    assert r.passed is False
    assert "below" in r.message


# ---------------------------------------------------------------------------
# validate_row — REGEX_MATCH
# ---------------------------------------------------------------------------


def test_regex_match_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.REGEX_MATCH,
        column="email",
        params=(("pattern", r"^[^@]+@[^@]+\.[^@]+$"),),
    )
    r = validate_row(e, {"email": "alice@example.com"})
    assert r.passed is True


def test_regex_match_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.REGEX_MATCH,
        column="email",
        params=(("pattern", r"^[^@]+@[^@]+\.[^@]+$"),),
    )
    r = validate_row(e, {"email": "not-an-email"})
    assert r.passed is False


def test_regex_match_missing_param() -> None:
    e = Expectation(
        kind=ExpectationKind.REGEX_MATCH,
        column="email",
    )
    with pytest.raises(DataQualityError, match="pattern"):
        validate_row(e, {"email": "x"})


# ---------------------------------------------------------------------------
# validate_row — TYPE_CHECK
# ---------------------------------------------------------------------------


def test_type_check_passes() -> None:
    e = Expectation(
        kind=ExpectationKind.TYPE_CHECK,
        column="count",
        params=(("type_name", "int"),),
    )
    r = validate_row(e, {"count": 42})
    assert r.passed is True


def test_type_check_fails() -> None:
    e = Expectation(
        kind=ExpectationKind.TYPE_CHECK,
        column="count",
        params=(("type_name", "int"),),
    )
    r = validate_row(e, {"count": "42"})
    assert r.passed is False


# ---------------------------------------------------------------------------
# validate_row — edge cases
# ---------------------------------------------------------------------------


def test_validate_row_bad_expectation_type() -> None:
    with pytest.raises(DataQualityError, match="Expectation"):
        validate_row("not an expectation", {"x": 1})  # type: ignore[arg-type]


def test_validate_row_bad_row_type() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="x")
    with pytest.raises(DataQualityError, match="Mapping"):
        validate_row(e, [1, 2, 3])  # type: ignore[arg-type]


def test_validate_row_missing_column_non_null_kind() -> None:
    e = Expectation(
        kind=ExpectationKind.IN_SET,
        column="side",
        params=(("values", ("BUY",)),),
    )
    r = validate_row(e, {"price": 42})
    assert r.passed is False
    assert "missing" in r.message


# ---------------------------------------------------------------------------
# validate_suite
# ---------------------------------------------------------------------------


def _sample_suite() -> tuple[Expectation, ...]:
    return (
        Expectation(kind=ExpectationKind.NOT_NULL, column="price"),
        Expectation(
            kind=ExpectationKind.BETWEEN,
            column="price",
            params=(("min", 0.0), ("max", 1_000_000.0)),
        ),
        Expectation(
            kind=ExpectationKind.IN_SET,
            column="side",
            params=(("values", ("BUY", "SELL")),),
        ),
    )


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {"price": 42.0, "side": "BUY"},
        {"price": 100.5, "side": "SELL"},
    ]


def test_validate_suite_all_pass() -> None:
    report = validate_suite(_sample_suite(), _sample_rows())
    assert isinstance(report, SuiteValidationReport)
    assert report.success is True
    assert report.total_checks == 6  # 3 expectations × 2 rows
    assert report.passed_checks == 6
    assert report.failed_checks == 0
    assert len(report.digest) == 32  # BLAKE2b-128 hex


def test_validate_suite_with_failures() -> None:
    suite = (
        Expectation(kind=ExpectationKind.NOT_NULL, column="price"),
        Expectation(
            kind=ExpectationKind.BETWEEN,
            column="price",
            params=(("min", 0.0), ("max", 50.0)),
        ),
    )
    rows = [{"price": 42.0}, {"price": 100.0}]
    report = validate_suite(suite, rows)
    assert report.success is False
    assert report.total_checks == 4
    assert report.passed_checks == 3  # 42 passes both, 100 passes not_null
    assert report.failed_checks == 1  # 100 fails between


def test_validate_suite_empty_rows() -> None:
    report = validate_suite(_sample_suite(), [])
    assert report.success is True
    assert report.total_checks == 0
    assert report.passed_checks == 0


def test_validate_suite_bad_suite_type() -> None:
    with pytest.raises(DataQualityError, match="tuple"):
        validate_suite([Expectation(kind=ExpectationKind.NOT_NULL, column="x")], [])  # type: ignore[arg-type]


def test_validate_suite_bad_rows_type() -> None:
    suite = (Expectation(kind=ExpectationKind.NOT_NULL, column="x"),)
    with pytest.raises(DataQualityError, match="Sequence"):
        validate_suite(suite, 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 — byte-stable digest
# ---------------------------------------------------------------------------


def test_inv15_byte_stable_digest() -> None:
    """Three independent calls → identical digests."""
    suite = _sample_suite()
    rows = _sample_rows()
    digests = [validate_suite(suite, rows).digest for _ in range(3)]
    assert digests[0] == digests[1] == digests[2]
    assert len(digests[0]) == 32


def test_inv15_different_data_different_digest() -> None:
    suite = _sample_suite()
    d1 = validate_suite(suite, [{"price": 1.0, "side": "BUY"}]).digest
    d2 = validate_suite(suite, [{"price": 2.0, "side": "SELL"}]).digest
    assert d1 != d2


# ---------------------------------------------------------------------------
# Lazy seam
# ---------------------------------------------------------------------------


def test_enable_great_expectations_factory_lazy() -> None:
    """Factory must import great_expectations — not installed → ImportError."""
    from system_engine.data_quality import enable_great_expectations_factory

    with pytest.raises(ImportError):
        enable_great_expectations_factory()


# ---------------------------------------------------------------------------
# Authority constraints
# ---------------------------------------------------------------------------


def test_no_cross_tier_imports() -> None:
    """B1 — no execution_engine / governance_engine / intelligence_engine."""
    import ast
    from pathlib import Path

    src = Path("system_engine/data_quality.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"execution_engine", "governance_engine", "intelligence_engine"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
            top = module.split(".")[0]
            assert top not in forbidden, f"B1 violation: imports from {module}"


def test_no_typed_event_constructors() -> None:
    """B27/B28 — no SystemEvent / SignalEvent constructors."""
    src = open("system_engine/data_quality.py", encoding="utf-8").read()
    assert "SystemEvent(" not in src
    assert "SignalEvent(" not in src


def test_expectation_result_is_frozen() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="x")
    r = validate_row(e, {"x": 1})
    with pytest.raises(AttributeError):
        r.passed = False  # type: ignore[misc]


def test_expectation_is_frozen() -> None:
    e = Expectation(kind=ExpectationKind.NOT_NULL, column="x")
    with pytest.raises(AttributeError):
        e.column = "y"  # type: ignore[misc]


def test_suite_report_is_frozen() -> None:
    report = validate_suite(
        (Expectation(kind=ExpectationKind.NOT_NULL, column="x"),),
        [{"x": 1}],
    )
    with pytest.raises(AttributeError):
        report.success = False  # type: ignore[misc]
