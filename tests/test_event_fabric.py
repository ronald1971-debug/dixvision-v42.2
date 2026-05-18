"""Tests for C-01 ``system_engine.streaming.event_fabric``.

Covers:

* Operator value-object surface (frozen + slotted).
* ``Dataflow`` immutable-builder semantics + ``dataflow_digest``.
* ``run_dataflow`` execution order: identity / map / filter /
  key_by + reduce / key_by + tumbling_window.
* INV-15 byte-identical replay (3-run equality on a multi-operator
  pipeline).
* Cross-process worker bridge: in-process driver of
  :func:`fabric_worker_main` over ``queue.Queue``.
* :func:`bytewax_dataflow_factory` raises ``NotImplementedError``
  until the research-acceptance gate ships.
* AST guardrails:
  - no top-level ``bytewax`` / ``random`` / ``time`` /
    ``datetime`` / ``asyncio`` / ``os`` imports.
  - no construction of typed events ``PatchProposal`` /
    ``HazardEvent`` / ``SignalEvent`` / ``ExecutionEvent`` /
    ``SystemEvent`` (B27 / B28 / INV-71 authority symmetry —
    the fabric is a transport, not an emitter).
  - no imports from runtime tiers ``intelligence_engine``,
    ``execution_engine``, ``governance_engine``,
    ``evolution_engine`` (B1).
"""

from __future__ import annotations

import ast
import queue
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from system_engine.streaming import event_fabric
from system_engine.streaming.event_fabric import (
    EVENT_FABRIC_VERSION,
    Dataflow,
    EventFabricSentinel,
    FabricResult,
    FilterOp,
    KeyByOp,
    MapOp,
    Operator,
    ReduceOp,
    TumblingWindowOp,
    bytewax_dataflow_factory,
    drain_queue,
    fabric_worker_main,
    run_dataflow,
)

# ---------------------------------------------------------------------------
# Module-level helpers used by AST guardrails and structural tests.
# ---------------------------------------------------------------------------

EVENT_FABRIC_PATH = Path(event_fabric.__file__)


# ---------------------------------------------------------------------------
# Operator value-object surface.
# ---------------------------------------------------------------------------


def test_event_fabric_version_is_pinned() -> None:
    assert EVENT_FABRIC_VERSION == 1


def test_operator_subclasses_are_frozen() -> None:
    op = MapOp(fn=lambda x: x)
    with pytest.raises(FrozenInstanceError):
        op.name = "mutated"  # type: ignore[misc]


def test_operator_subclasses_inherit_marker() -> None:
    assert issubclass(MapOp, Operator)
    assert issubclass(FilterOp, Operator)
    assert issubclass(KeyByOp, Operator)
    assert issubclass(ReduceOp, Operator)
    assert issubclass(TumblingWindowOp, Operator)


def test_operator_subclasses_carry_default_names() -> None:
    assert MapOp(fn=lambda x: x).name == "map"
    assert FilterOp(predicate=lambda x: True).name == "filter"
    assert KeyByOp(key_fn=lambda x: "k").name == "key_by"
    assert ReduceOp(init=list, step=lambda a, x: a + [x]).name == "reduce"
    assert (
        TumblingWindowOp(
            window_ns=1,
            ts_fn=lambda x: 0,
            init=list,
            step=lambda a, x: a + [x],
            finalize=lambda a, k, b, s: a,
        ).name
        == "tumbling_window"
    )


# ---------------------------------------------------------------------------
# Dataflow builder semantics.
# ---------------------------------------------------------------------------


def test_dataflow_then_is_immutable_builder() -> None:
    df0 = Dataflow(name="t")
    df1 = df0.map(lambda x: x + 1)
    df2 = df1.filter(lambda x: x > 0)

    assert df0.operators == ()
    assert len(df1.operators) == 1
    assert len(df2.operators) == 2
    assert df1 is not df0
    assert df2 is not df1


def test_dataflow_chain_helpers_match_operator_types() -> None:
    df = (
        Dataflow(name="chain")
        .map(lambda x: x)
        .filter(lambda x: True)
        .key_by(lambda x: "k")
        .reduce(lambda: 0, lambda a, x: a + 1)
    )
    types = [type(op).__name__ for op in df.operators]
    assert types == ["MapOp", "FilterOp", "KeyByOp", "ReduceOp"]


def test_dataflow_tumbling_window_rejects_non_positive_window() -> None:
    df = Dataflow(name="t").key_by(lambda x: "k")
    with pytest.raises(ValueError):
        df.tumbling_window(
            window_ns=0,
            ts_fn=lambda x: 0,
            init=list,
            step=lambda a, x: a + [x],
            finalize=lambda a, k, b, s: a,
        )
    with pytest.raises(ValueError):
        df.tumbling_window(
            window_ns=-1,
            ts_fn=lambda x: 0,
            init=list,
            step=lambda a, x: a + [x],
            finalize=lambda a, k, b, s: a,
        )


def test_dataflow_digest_is_16_hex_and_stable() -> None:
    df = Dataflow(name="t").map(lambda x: x).filter(lambda x: True)
    digest = df.dataflow_digest()
    assert len(digest) == 16
    assert int(digest, 16) >= 0
    # Stable across re-construction with the same name + operator chain.
    again = Dataflow(name="t").map(lambda x: x).filter(lambda x: True)
    assert again.dataflow_digest() == digest


def test_dataflow_digest_changes_with_name_or_chain() -> None:
    base = Dataflow(name="t").map(lambda x: x)
    renamed = Dataflow(name="u").map(lambda x: x)
    extended = base.filter(lambda x: True)
    assert base.dataflow_digest() != renamed.dataflow_digest()
    assert base.dataflow_digest() != extended.dataflow_digest()


def test_dataflow_digest_includes_window_ns() -> None:
    a = (
        Dataflow(name="t")
        .key_by(lambda x: "k")
        .tumbling_window(
            window_ns=1_000,
            ts_fn=lambda x: 0,
            init=list,
            step=lambda acc, x: acc + [x],
            finalize=lambda acc, k, b, s: tuple(acc),
        )
    )
    b = (
        Dataflow(name="t")
        .key_by(lambda x: "k")
        .tumbling_window(
            window_ns=2_000,
            ts_fn=lambda x: 0,
            init=list,
            step=lambda acc, x: acc + [x],
            finalize=lambda acc, k, b, s: tuple(acc),
        )
    )
    assert a.dataflow_digest() != b.dataflow_digest()


# ---------------------------------------------------------------------------
# run_dataflow — execution semantics.
# ---------------------------------------------------------------------------


def test_run_dataflow_identity_emits_original_events() -> None:
    df = Dataflow(name="identity")
    out = run_dataflow(df, [1, 2, 3])
    assert tuple(r.payload for r in out) == (1, 2, 3)
    assert tuple(r.seq for r in out) == (0, 1, 2)
    assert all(r.operator_name == "identity" for r in out)


def test_run_dataflow_map_then_filter_is_insertion_order() -> None:
    df = Dataflow(name="t").map(lambda x: x * 2).filter(lambda x: x % 4 == 0)
    out = run_dataflow(df, [1, 2, 3, 4])
    # map -> [2,4,6,8]; filter keeps multiples of 4 -> [4,8]
    assert tuple(r.payload for r in out) == (4, 8)


def test_run_dataflow_key_by_then_reduce_preserves_first_seen_key_order() -> None:
    df = (
        Dataflow(name="t")
        .key_by(lambda x: x["sym"])
        .reduce(
            init=lambda: 0,
            step=lambda acc, item: acc + item["qty"],
        )
    )
    events = [
        {"sym": "BTC", "qty": 1},
        {"sym": "ETH", "qty": 2},
        {"sym": "BTC", "qty": 3},
        {"sym": "SOL", "qty": 4},
        {"sym": "ETH", "qty": 5},
    ]
    out = run_dataflow(df, events)
    # First-seen key order: BTC, ETH, SOL.
    assert tuple((r.key, r.payload) for r in out) == (
        ("BTC", 4),
        ("ETH", 7),
        ("SOL", 4),
    )


def test_run_dataflow_tumbling_window_sorts_by_bucket_then_key() -> None:
    df = (
        Dataflow(name="t")
        .key_by(lambda x: x["sym"])
        .tumbling_window(
            window_ns=1_000_000_000,
            ts_fn=lambda x: x["ts_ns"],
            init=lambda: 0,
            step=lambda acc, item: acc + item["qty"],
            finalize=lambda acc, k, b, s: {"key": k, "bucket": b, "start_ns": s, "sum": acc},
        )
    )
    events = [
        {"sym": "BTC", "ts_ns": 500_000_000, "qty": 1},  # bucket 0
        {"sym": "ETH", "ts_ns": 600_000_000, "qty": 10},  # bucket 0
        {"sym": "BTC", "ts_ns": 1_200_000_000, "qty": 2},  # bucket 1
        {"sym": "BTC", "ts_ns": 1_400_000_000, "qty": 3},  # bucket 1
        {"sym": "AAA", "ts_ns": 500_000_000, "qty": 100},  # bucket 0 (sorts before BTC)
    ]
    out = run_dataflow(df, events)
    # Sorted by (bucket asc, key asc):
    # bucket 0 -> AAA(100), BTC(1), ETH(10); bucket 1 -> BTC(5)
    rendered = [(r.bucket_idx, r.key, r.payload["sum"]) for r in out]
    assert rendered == [
        (0, "AAA", 100),
        (0, "BTC", 1),
        (0, "ETH", 10),
        (1, "BTC", 5),
    ]


def test_reduce_rejects_unkeyed_input() -> None:
    df = Dataflow(name="t").reduce(init=lambda: 0, step=lambda a, x: a + 1)
    with pytest.raises(TypeError):
        run_dataflow(df, [1, 2, 3])


def test_tumbling_window_rejects_unkeyed_input() -> None:
    df = Dataflow(name="t").tumbling_window(
        window_ns=10,
        ts_fn=lambda x: 0,
        init=lambda: 0,
        step=lambda a, x: a + 1,
        finalize=lambda a, k, b, s: a,
    )
    with pytest.raises(TypeError):
        run_dataflow(df, [1, 2, 3])


def test_tumbling_window_rejects_non_int_ts() -> None:
    df = (
        Dataflow(name="t")
        .key_by(lambda x: "k")
        .tumbling_window(
            window_ns=10,
            ts_fn=lambda x: "not-int",  # type: ignore[return-value]
            init=lambda: 0,
            step=lambda a, x: a + 1,
            finalize=lambda a, k, b, s: a,
        )
    )
    with pytest.raises(TypeError):
        run_dataflow(df, [{}])


def test_reduce_rejects_non_str_key() -> None:
    df = Dataflow(name="t").reduce(init=lambda: 0, step=lambda a, x: a + 1)
    with pytest.raises(TypeError):
        # Hand-craft an already-keyed tuple stream with a non-str key.
        run_dataflow(df, [(1, "item")])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# INV-15 — byte-identical replay equality.
# ---------------------------------------------------------------------------


def _make_replay_dataflow() -> Dataflow:
    return (
        Dataflow(name="replay")
        .map(lambda x: {**x, "qty_x2": x["qty"] * 2})
        .filter(lambda x: x["qty_x2"] > 1)
        .key_by(lambda x: x["sym"])
        .tumbling_window(
            window_ns=1_000_000_000,
            ts_fn=lambda x: x["ts_ns"],
            init=lambda: {"count": 0, "sum_qty": 0.0},
            step=lambda acc, item: {
                "count": acc["count"] + 1,
                "sum_qty": acc["sum_qty"] + item["qty_x2"],
            },
            finalize=lambda acc, k, b, s: {
                "key": k,
                "bucket": b,
                "start_ns": s,
                "count": acc["count"],
                "sum_qty": acc["sum_qty"],
            },
        )
    )


def _make_replay_events() -> list[dict[str, object]]:
    return [
        {"sym": "BTC", "ts_ns": 1_000_000_000, "qty": 1.0},
        {"sym": "ETH", "ts_ns": 1_500_000_000, "qty": 2.0},
        {"sym": "BTC", "ts_ns": 2_200_000_000, "qty": 3.0},
        {"sym": "BTC", "ts_ns": 2_700_000_000, "qty": 4.0},
        {"sym": "ETH", "ts_ns": 3_100_000_000, "qty": 0.4},  # qty_x2 == 0.8 → filtered out
        {"sym": "AAA", "ts_ns": 1_100_000_000, "qty": 5.0},
    ]


def test_replay_equality_three_runs() -> None:
    df = _make_replay_dataflow()
    events = _make_replay_events()
    run_a = run_dataflow(df, events)
    run_b = run_dataflow(df, events)
    run_c = run_dataflow(df, events)
    assert run_a == run_b == run_c


def test_replay_digest_is_chain_stable() -> None:
    digest_a = _make_replay_dataflow().dataflow_digest()
    digest_b = _make_replay_dataflow().dataflow_digest()
    digest_c = _make_replay_dataflow().dataflow_digest()
    assert digest_a == digest_b == digest_c


def test_fabric_result_is_frozen_and_slotted() -> None:
    fr = FabricResult(seq=0, operator_name="map", payload=1)
    with pytest.raises(FrozenInstanceError):
        fr.seq = 99  # type: ignore[misc]
    assert FabricResult.__slots__  # dataclass slots=True populates __slots__


# ---------------------------------------------------------------------------
# Cross-process worker bridge — driven in-process via queue.Queue.
# ---------------------------------------------------------------------------


def test_fabric_worker_main_drains_to_outbound_then_emits_sentinel() -> None:
    df = Dataflow(name="x").map(lambda x: x * 10)
    inbound: queue.Queue[object] = queue.Queue()
    outbound: queue.Queue[object] = queue.Queue()
    for ev in [1, 2, 3, 4]:
        inbound.put(ev)
    inbound.put(EventFabricSentinel())

    fabric_worker_main(df, inbound, outbound, batch_size=2)  # type: ignore[arg-type]

    drained = drain_queue(outbound, timeout_s=1.0)  # type: ignore[arg-type]
    assert tuple(r.payload for r in drained) == (10, 20, 30, 40)


def test_fabric_worker_main_handles_partial_final_batch() -> None:
    df = Dataflow(name="x").map(lambda x: x + 1)
    inbound: queue.Queue[object] = queue.Queue()
    outbound: queue.Queue[object] = queue.Queue()
    for ev in [1, 2, 3]:  # batch_size=2, so final batch is partial (1 item).
        inbound.put(ev)
    inbound.put(EventFabricSentinel())

    fabric_worker_main(df, inbound, outbound, batch_size=2)  # type: ignore[arg-type]

    drained = drain_queue(outbound, timeout_s=1.0)  # type: ignore[arg-type]
    assert tuple(r.payload for r in drained) == (2, 3, 4)


def test_fabric_worker_main_rejects_zero_batch_size() -> None:
    df = Dataflow(name="x")
    with pytest.raises(ValueError):
        fabric_worker_main(df, queue.Queue(), queue.Queue(), batch_size=0)  # type: ignore[arg-type]


def test_drain_queue_rejects_unexpected_payload() -> None:
    outbound: queue.Queue[object] = queue.Queue()
    outbound.put("not-a-tuple")
    with pytest.raises(TypeError):
        drain_queue(outbound, timeout_s=0.1)  # type: ignore[arg-type]


def test_drain_queue_timeout_raises() -> None:
    outbound: queue.Queue[object] = queue.Queue()
    with pytest.raises(TimeoutError):
        drain_queue(outbound, timeout_s=0.05)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bytewax lazy factory — pinned NotImplementedError.
# ---------------------------------------------------------------------------


def test_bytewax_dataflow_factory_is_gated() -> None:
    with pytest.raises(NotImplementedError) as ei:
        bytewax_dataflow_factory()
    msg = str(ei.value)
    assert "research-acceptance" in msg
    assert "shadow-equivalence" in msg


# ---------------------------------------------------------------------------
# AST guardrails — module hygiene.
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(EVENT_FABRIC_PATH.read_text(encoding="utf-8"))


def test_module_has_adapted_from_header() -> None:
    text = EVENT_FABRIC_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: bytewax" in text


def test_module_has_no_top_level_forbidden_imports() -> None:
    forbidden = {
        "bytewax",
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
    }
    seen: list[str] = []
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            seen.append(node.module.split(".")[0])
    offenders = sorted(forbidden.intersection(seen))
    assert offenders == [], f"forbidden imports present: {offenders}"


def test_module_has_no_runtime_tier_imports() -> None:
    forbidden = {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
    }
    seen: list[str] = []
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            seen.append(node.module.split(".")[0])
    offenders = sorted(forbidden.intersection(seen))
    assert offenders == [], f"runtime-tier imports present: {offenders}"


def test_module_never_constructs_typed_events() -> None:
    """B27 / B28 / INV-71 authority symmetry — fabric is transport only."""

    banned = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }
    offenders: list[str] = []
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned:
                offenders.append(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in banned:
                offenders.append(func.attr)
    assert offenders == [], f"forbidden typed-event constructors: {offenders}"
