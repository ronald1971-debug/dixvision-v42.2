# ADAPTED FROM: duckdb/duckdb tools/pythonpkg — test contract mirrors
#   the DuckDB Python cursor surface (execute / fetchall / description)
#   without requiring duckdb at test time.
"""A-14 duckdb → ledger-query analytics tests.

Covers the OFFLINE_ONLY ``learning_engine/analytics/ledger_query.py``
coordinator:

* Frozen contracts (validation in ``__post_init__``).
* :class:`LedgerAnalytics.fetch_rows` projection + filter + limit.
* :class:`LedgerAnalytics.group_by` aggregations (count/sum/avg/min/max).
* :class:`LedgerAnalytics.percentile` over typed numeric columns.
* :class:`LedgerAnalytics.count` over filtered rows.
* INV-15 byte-identical replay — 3 runs of every query path produce
  byte-equal ``result_digest`` hex strings.
* AST invariants:

  - ``import duckdb`` does **not** appear at module top-level.
  - No engine cross-imports (B1 isolation).
  - No clock / time / datetime / random / asyncio / os imports
    (INV-15).
  - No construction of ``PatchProposal`` / ``SignalEvent`` /
    ``GovernanceDecision`` (B27 / B28 / INV-71 authority symmetry).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from learning_engine.analytics.ledger_query import (
    NEW_PIP_DEPENDENCIES,
    AggregateSpec,
    AnalyticsBackend,
    GroupBySpec,
    InProcessAnalyticsBackend,
    LedgerAnalytics,
    QueryRequest,
    QueryResult,
)

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "learning_engine"
    / "analytics"
    / "ledger_query.py"
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _rows() -> tuple[dict[str, object], ...]:
    return (
        {"symbol": "BTC", "side": "BUY", "qty": 1.0, "pnl_usd": 10.0},
        {"symbol": "BTC", "side": "SELL", "qty": 2.0, "pnl_usd": -5.0},
        {"symbol": "ETH", "side": "BUY", "qty": 3.0, "pnl_usd": 7.0},
        {"symbol": "ETH", "side": "SELL", "qty": 1.0, "pnl_usd": -2.0},
        {"symbol": "SOL", "side": "BUY", "qty": 5.0, "pnl_usd": 1.0},
    )


def _backend() -> InProcessAnalyticsBackend:
    return InProcessAnalyticsBackend(rows=_rows())


def _analytics() -> LedgerAnalytics:
    return LedgerAnalytics(backend=_backend())


# ----------------------------------------------------------------------
# Module-level contract
# ----------------------------------------------------------------------


def test_module_declares_duckdb_pip_dep() -> None:
    assert NEW_PIP_DEPENDENCIES == ("duckdb",)


def test_in_process_backend_implements_protocol() -> None:
    assert isinstance(_backend(), AnalyticsBackend)


# ----------------------------------------------------------------------
# QueryRequest validation
# ----------------------------------------------------------------------


def test_query_request_rejects_empty_columns() -> None:
    with pytest.raises(ValueError):
        QueryRequest(table="trades", columns=())


def test_query_request_rejects_bad_table() -> None:
    with pytest.raises(ValueError):
        QueryRequest(table="bad-name!", columns=("x",))


def test_query_request_rejects_negative_limit() -> None:
    with pytest.raises(ValueError):
        QueryRequest(table="trades", columns=("symbol",), limit=-1)


def test_query_request_rejects_bool_limit() -> None:
    with pytest.raises(TypeError):
        QueryRequest(
            table="trades",
            columns=("symbol",),
            limit=True,  # type: ignore[arg-type]
        )


def test_query_request_validates_unknown_columns() -> None:
    backend = _backend()
    with pytest.raises(ValueError, match="unknown columns"):
        backend.execute(QueryRequest(table="trades", columns=("nonexistent_column",)))


def test_query_request_validates_unknown_filter_columns() -> None:
    backend = _backend()
    with pytest.raises(ValueError, match="unknown columns"):
        backend.execute(
            QueryRequest(
                table="trades",
                columns=("symbol",),
                filters={"nonexistent_column": "x"},
            )
        )


# ----------------------------------------------------------------------
# Fetch rows
# ----------------------------------------------------------------------


def test_fetch_rows_returns_all_when_no_filter() -> None:
    analytics = _analytics()
    result = analytics.fetch_rows(QueryRequest(table="trades", columns=("symbol", "pnl_usd")))
    assert result.row_count() == 5
    assert result.request_columns == ("symbol", "pnl_usd")


def test_fetch_rows_filters_by_equality() -> None:
    analytics = _analytics()
    result = analytics.fetch_rows(
        QueryRequest(
            table="trades",
            columns=("symbol", "pnl_usd"),
            filters={"symbol": "BTC"},
        )
    )
    assert result.row_count() == 2
    assert all(row[0] == "BTC" for row in result.rows)


def test_fetch_rows_orders_deterministically() -> None:
    analytics = _analytics()
    result_a = analytics.fetch_rows(QueryRequest(table="trades", columns=("symbol", "qty")))
    result_b = analytics.fetch_rows(QueryRequest(table="trades", columns=("symbol", "qty")))
    assert result_a.rows == result_b.rows


def test_fetch_rows_respects_limit() -> None:
    analytics = _analytics()
    result = analytics.fetch_rows(QueryRequest(table="trades", columns=("symbol",), limit=2))
    assert result.row_count() == 2


def test_fetch_rows_explicit_order_by() -> None:
    analytics = _analytics()
    result = analytics.fetch_rows(
        QueryRequest(
            table="trades",
            columns=("symbol", "pnl_usd"),
            order_by=("pnl_usd",),
        )
    )
    pnls = [row[1] for row in result.rows]
    assert pnls == sorted(pnls)


# ----------------------------------------------------------------------
# Aggregates / group-by
# ----------------------------------------------------------------------


def test_group_by_count_per_symbol() -> None:
    analytics = _analytics()
    result = analytics.group_by(
        GroupBySpec(
            table="trades",
            group_keys=("symbol",),
            aggregates=(AggregateSpec(op="count", column="pnl_usd", alias="n"),),
        )
    )
    assert result.rows == (("BTC", 2), ("ETH", 2), ("SOL", 1))


def test_group_by_sum_per_symbol() -> None:
    analytics = _analytics()
    result = analytics.group_by(
        GroupBySpec(
            table="trades",
            group_keys=("symbol",),
            aggregates=(AggregateSpec(op="sum", column="pnl_usd", alias="total"),),
        )
    )
    table = {row[0]: row[1] for row in result.rows}
    assert table == {"BTC": 5.0, "ETH": 5.0, "SOL": 1.0}


def test_group_by_min_max_avg() -> None:
    analytics = _analytics()
    result = analytics.group_by(
        GroupBySpec(
            table="trades",
            group_keys=("symbol",),
            aggregates=(
                AggregateSpec(op="min", column="pnl_usd", alias="lo"),
                AggregateSpec(op="max", column="pnl_usd", alias="hi"),
                AggregateSpec(op="avg", column="pnl_usd", alias="mu"),
            ),
        )
    )
    by_sym = {row[0]: row[1:] for row in result.rows}
    assert by_sym["BTC"] == (-5.0, 10.0, 2.5)
    assert by_sym["ETH"] == (-2.0, 7.0, 2.5)
    assert by_sym["SOL"] == (1.0, 1.0, 1.0)


def test_group_by_with_filter() -> None:
    analytics = _analytics()
    result = analytics.group_by(
        GroupBySpec(
            table="trades",
            group_keys=("symbol",),
            aggregates=(AggregateSpec(op="count", column="pnl_usd", alias="n"),),
            filters={"side": "BUY"},
        )
    )
    by_sym = {row[0]: row[1] for row in result.rows}
    assert by_sym == {"BTC": 1, "ETH": 1, "SOL": 1}


def test_aggregate_spec_rejects_bad_op() -> None:
    with pytest.raises(ValueError):
        AggregateSpec(
            op="median",  # type: ignore[arg-type]
            column="pnl_usd",
            alias="m",
        )


def test_group_by_spec_rejects_empty_aggregates() -> None:
    with pytest.raises(ValueError):
        GroupBySpec(
            table="trades",
            group_keys=("symbol",),
            aggregates=(),
        )


# ----------------------------------------------------------------------
# Percentile / count helpers
# ----------------------------------------------------------------------


def test_count_filters_rows() -> None:
    analytics = _analytics()
    assert analytics.count("trades", filters={"side": "BUY"}) == 3
    assert analytics.count("trades", filters={"symbol": "BTC"}) == 2


def test_percentile_returns_nearest_rank() -> None:
    analytics = _analytics()
    request = QueryRequest(table="trades", columns=("pnl_usd",))
    p50 = analytics.percentile(request, column="pnl_usd", percentile=0.5)
    p99 = analytics.percentile(request, column="pnl_usd", percentile=0.99)
    assert p50 == 1.0
    assert p99 == 7.0


def test_percentile_rejects_unsupported_value() -> None:
    analytics = _analytics()
    request = QueryRequest(table="trades", columns=("pnl_usd",))
    with pytest.raises(ValueError):
        analytics.percentile(request, column="pnl_usd", percentile=0.75)


# ----------------------------------------------------------------------
# INV-15 byte-identical replay
# ----------------------------------------------------------------------


def test_fetch_rows_digest_is_deterministic() -> None:
    analytics = _analytics()
    request = QueryRequest(
        table="trades",
        columns=("symbol", "side", "qty", "pnl_usd"),
    )
    digests = {analytics.fetch_rows(request).result_digest for _ in range(3)}
    assert len(digests) == 1


def test_group_by_digest_is_deterministic() -> None:
    analytics = _analytics()
    spec = GroupBySpec(
        table="trades",
        group_keys=("symbol",),
        aggregates=(
            AggregateSpec(op="count", column="pnl_usd", alias="n"),
            AggregateSpec(op="sum", column="pnl_usd", alias="total"),
        ),
    )
    digests = {analytics.group_by(spec).result_digest for _ in range(3)}
    assert len(digests) == 1


def test_digest_changes_when_rows_change() -> None:
    a = LedgerAnalytics(backend=_backend())
    b = LedgerAnalytics(
        backend=InProcessAnalyticsBackend(
            rows=_rows() + ({"symbol": "DOGE", "side": "BUY", "qty": 1.0, "pnl_usd": 0.5},)
        )
    )
    request = QueryRequest(
        table="trades",
        columns=("symbol", "side", "qty", "pnl_usd"),
    )
    assert a.fetch_rows(request).result_digest != b.fetch_rows(request).result_digest


# ----------------------------------------------------------------------
# QueryResult validation
# ----------------------------------------------------------------------


def test_query_result_rejects_bad_digest_length() -> None:
    with pytest.raises(ValueError):
        QueryResult(
            request_table="trades",
            request_columns=("symbol",),
            rows=(("BTC",),),
            result_digest="short",
        )


def test_query_result_rejects_row_width_mismatch() -> None:
    with pytest.raises(ValueError):
        QueryResult(
            request_table="trades",
            request_columns=("symbol", "qty"),
            rows=(("BTC",),),
            result_digest="0" * 32,
        )


# ----------------------------------------------------------------------
# AST guard: lazy duckdb import + no engine cross-imports + INV-15
# ----------------------------------------------------------------------


def _parse() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def test_no_toplevel_duckdb_import() -> None:
    tree = _parse()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "duckdb", (
                    "duckdb must be lazy-imported inside the factory body"
                )
        if isinstance(node, ast.ImportFrom):
            assert node.module != "duckdb"


def test_no_runtime_engine_imports() -> None:
    """B1: analytics module must not import runtime engines."""
    banned = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
        "intelligence_engine.meta_controller.hot_path",
    }
    tree = _parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in banned:
                assert not node.module.startswith(prefix), (
                    f"{prefix} import banned from learning_engine/analytics/ledger_query.py"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in banned:
                    assert not alias.name.startswith(prefix), f"{prefix} import banned"


def test_no_clock_or_random_imports() -> None:
    """INV-15: no clock / random / IO / asyncio imports."""
    banned = {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "subprocess",
        "socket",
        "websockets",
        "requests",
        "urllib",
        "urllib.request",
    }
    tree = _parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in banned, f"INV-15 violation: import {alias.name!r}"
        if isinstance(node, ast.ImportFrom):
            assert node.module not in banned, f"INV-15 violation: from {node.module!r}"


def test_no_typed_event_construction() -> None:
    """B27 / B28 / INV-71: analytics never constructs typed bus events."""
    banned_calls = {"PatchProposal", "SignalEvent", "GovernanceDecision"}
    tree = _parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id not in banned_calls, (
                    f"B27/B28/INV-71 authority symmetry violation: "
                    f"learning_engine/analytics may not construct "
                    f"{func.id}"
                )
            if isinstance(func, ast.Attribute):
                assert func.attr not in banned_calls, f"B27/B28/INV-71 violation: {func.attr}"
