"""Tests for C-02 ``system_engine.streaming.faust_bus``.

Covers:

* Value-object surface (``Record`` / ``Topic`` / ``Agent`` / ``Table`` /
  ``TumblingWindowSpec`` / ``SendOp`` / ``AgentContext`` / ``App`` /
  ``FaustResult`` / ``InboundEvent``) — all frozen, slotted.
* ``App`` immutable-builder semantics — ``topic`` / ``table`` /
  ``agent`` return new instances; ``app_digest`` stable.
* ``run_app`` execution: agent fan-out order, ``SendOp`` BFS
  forwarding, flat-table emission, ``TumblingEventTimeWindow`` flush
  ordered ``(bucket_idx asc, key asc)``.
* INV-15 byte-identical replay — 3-run equality on a multi-agent +
  windowed-table pipeline; seed-order independence; dict-insertion
  invariance.
* Cross-process worker bridge — driven in-process via
  ``queue.Queue``.
* :func:`faust_app_factory` raises ``NotImplementedError`` until the
  research-acceptance gate ships.
* AST guardrails:
  - no top-level ``faust`` / ``random`` / ``time`` / ``datetime`` /
    ``asyncio`` / ``os`` / ``numpy`` / ``torch`` / ``polars`` imports.
  - no construction of typed events ``PatchProposal`` /
    ``HazardEvent`` / ``SignalEvent`` / ``ExecutionEvent`` /
    ``SystemEvent`` (B27 / B28 / INV-71 authority symmetry — the bus
    is a transport, not an emitter).
  - no imports from runtime tiers ``intelligence_engine``,
    ``execution_engine``, ``governance_engine``,
    ``evolution_engine`` (B1).
  - ``# ADAPTED FROM:`` upstream-source header present.
"""

from __future__ import annotations

import ast
import queue
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from system_engine.streaming import faust_bus
from system_engine.streaming.faust_bus import (
    FAUST_BUS_VERSION,
    Agent,
    AgentContext,
    App,
    FaustBusSentinel,
    FaustResult,
    InboundEvent,
    Record,
    SendOp,
    Table,
    Topic,
    TumblingWindowSpec,
    bucket_index,
    drain_queue,
    faust_app_factory,
    faust_worker_main,
    run_app,
)

# ---------------------------------------------------------------------------
# Module-level helpers used by AST guardrails and structural tests.
# ---------------------------------------------------------------------------

FAUST_BUS_PATH = Path(faust_bus.__file__)


def _module_ast() -> ast.Module:
    return ast.parse(FAUST_BUS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Value-object surface.
# ---------------------------------------------------------------------------


def test_version_is_pinned() -> None:
    assert FAUST_BUS_VERSION == 1


def test_new_pip_dependencies_pin() -> None:
    assert faust_bus.NEW_PIP_DEPENDENCIES == ("faust-streaming",)


def test_record_is_frozen_and_slotted_marker_base() -> None:
    @dataclass(frozen=True, slots=True)
    class TradeRecord(Record):
        ts_ns: int
        sym: str
        qty: float

    r = TradeRecord(ts_ns=1, sym="BTC", qty=1.0)
    with pytest.raises(FrozenInstanceError):
        r.qty = 99.0  # type: ignore[misc]
    assert isinstance(r, Record)


def test_topic_is_frozen() -> None:
    t = Topic(name="market")
    with pytest.raises(FrozenInstanceError):
        t.name = "other"  # type: ignore[misc]


def test_agent_is_frozen() -> None:
    a = Agent(topic_name="market", fn=lambda ctx, ev: None, name="a")
    with pytest.raises(FrozenInstanceError):
        a.name = "x"  # type: ignore[misc]


def test_table_is_frozen() -> None:
    tbl: Table[int] = Table(name="counts", default=lambda: 0)
    with pytest.raises(FrozenInstanceError):
        tbl.name = "y"  # type: ignore[misc]


def test_tumbling_window_spec_is_frozen() -> None:
    spec = TumblingWindowSpec(window_ns=1, ts_fn=lambda x: 0)
    with pytest.raises(FrozenInstanceError):
        spec.window_ns = 2  # type: ignore[misc]


def test_send_op_is_frozen() -> None:
    s = SendOp(topic_name="t", payload=1)
    with pytest.raises(FrozenInstanceError):
        s.payload = 2  # type: ignore[misc]


def test_agent_context_is_frozen() -> None:
    ctx = AgentContext(tables={})
    with pytest.raises(FrozenInstanceError):
        ctx.current_ts_ns = 1  # type: ignore[misc]


def test_inbound_event_is_frozen() -> None:
    ev = InboundEvent(topic_name="t", payload=1)
    with pytest.raises(FrozenInstanceError):
        ev.payload = 2  # type: ignore[misc]


def test_faust_result_is_frozen_and_slotted() -> None:
    r = FaustResult(seq=0, kind="event", topic_name="t", payload=1)
    with pytest.raises(FrozenInstanceError):
        r.seq = 99  # type: ignore[misc]
    assert FaustResult.__slots__


def test_table_tumbling_rejects_non_positive_window() -> None:
    tbl: Table[int] = Table(name="t", default=lambda: 0)
    with pytest.raises(ValueError):
        tbl.tumbling(window_ns=0, ts_fn=lambda x: 0)
    with pytest.raises(ValueError):
        tbl.tumbling(window_ns=-1, ts_fn=lambda x: 0)


def test_bucket_index_helper() -> None:
    assert bucket_index(0, 1000) == 0
    assert bucket_index(999, 1000) == 0
    assert bucket_index(1000, 1000) == 1
    assert bucket_index(2_500_000_000, 1_000_000_000) == 2
    with pytest.raises(ValueError):
        bucket_index(10, 0)


# ---------------------------------------------------------------------------
# App immutable-builder semantics.
# ---------------------------------------------------------------------------


def test_app_topic_returns_new_instance() -> None:
    a0 = App(id="x")
    a1 = a0.topic("market")
    assert a0 is not a1
    assert a0.topics == ()
    assert a1.topics == (Topic(name="market"),)


def test_app_topic_rejects_duplicate() -> None:
    a = App(id="x").topic("market")
    with pytest.raises(ValueError):
        a.topic("market")


def test_app_table_rejects_duplicate() -> None:
    a = App(id="x").table("counts", default=lambda: 0)
    with pytest.raises(ValueError):
        a.table("counts", default=lambda: 0)


def test_app_agent_requires_registered_topic() -> None:
    a = App(id="x")
    with pytest.raises(ValueError):
        a.agent("market", fn=lambda ctx, ev: None)


def test_app_agent_defaults_name() -> None:
    a = App(id="x").topic("market").agent("market", fn=lambda ctx, ev: None)
    assert a.agents[0].name == "agent_0"


def test_app_digest_is_16_hex_and_stable() -> None:
    def build() -> App:
        return (
            App(id="x")
            .topic("market")
            .table("c", default=lambda: 0)
            .agent("market", fn=lambda ctx, ev: None, name="a0")
        )

    digest_a = build().app_digest()
    digest_b = build().app_digest()
    assert digest_a == digest_b
    assert len(digest_a) == 16
    assert int(digest_a, 16) >= 0


def test_app_digest_changes_on_topic_addition() -> None:
    base = App(id="x").topic("a")
    extended = base.topic("b")
    assert base.app_digest() != extended.app_digest()


# ---------------------------------------------------------------------------
# run_app — empty / single-agent / multi-agent paths.
# ---------------------------------------------------------------------------


def test_run_app_empty_stream_returns_empty_tuple() -> None:
    a = App(id="x").topic("market")
    assert run_app(a, ()) == ()


def test_run_app_single_agent_emits_send_ops_in_yield_order() -> None:
    def agent_fn(ctx: AgentContext, ev: int) -> list[SendOp]:
        return [
            SendOp(topic_name="out", payload=ev),
            SendOp(topic_name="out", payload=ev * 10),
        ]

    a = App(id="x").topic("in").topic("out").agent("in", fn=agent_fn)
    results = run_app(
        a,
        [InboundEvent(topic_name="in", payload=1)],
    )
    payloads = [r.payload for r in results if r.kind == "event"]
    assert payloads == [1, 10]


def test_run_app_agents_fire_in_registration_order() -> None:
    log: list[str] = []

    def first(ctx: AgentContext, ev: int) -> None:
        log.append(f"first:{ev}")

    def second(ctx: AgentContext, ev: int) -> None:
        log.append(f"second:{ev}")

    a = (
        App(id="x")
        .topic("in")
        .agent("in", fn=first, name="first")
        .agent("in", fn=second, name="second")
    )
    run_app(a, [InboundEvent(topic_name="in", payload=7)])
    assert log == ["first:7", "second:7"]


def test_run_app_rejects_non_send_op_yield() -> None:
    def bad(ctx: AgentContext, ev: int) -> list[object]:
        return [{"topic": "out", "payload": ev}]  # type: ignore[list-item]

    a = App(id="x").topic("in").agent("in", fn=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_app(a, [InboundEvent(topic_name="in", payload=1)])


def test_run_app_event_to_event_bfs() -> None:
    """SendOp outputs join the inbound tail (BFS), not a recursive stack."""

    def doubler(ctx: AgentContext, ev: int) -> list[SendOp]:
        if ev < 4:
            return [SendOp(topic_name="in", payload=ev + 1)]
        return []

    a = App(id="x").topic("in").agent("in", fn=doubler)
    results = run_app(a, [InboundEvent(topic_name="in", payload=1)])
    payloads = [r.payload for r in results if r.kind == "event"]
    assert payloads == [2, 3, 4]


# ---------------------------------------------------------------------------
# Flat keyed table.
# ---------------------------------------------------------------------------


def test_run_app_flat_table_emits_sorted_keys() -> None:
    def aggregator(ctx: AgentContext, ev: dict[str, object]) -> None:
        sym = str(ev["sym"])
        qty = float(ev["qty"])
        cells = ctx.tables["totals"]
        cells[sym] = cells.get(sym, 0.0) + qty

    a = (
        App(id="x")
        .topic("trades")
        .table("totals", default=lambda: 0.0)
        .agent("trades", fn=aggregator)
    )
    inbound = [
        InboundEvent(topic_name="trades", payload={"sym": "BTC", "qty": 1.0}),
        InboundEvent(topic_name="trades", payload={"sym": "ETH", "qty": 2.0}),
        InboundEvent(topic_name="trades", payload={"sym": "BTC", "qty": 0.5}),
    ]
    results = run_app(a, inbound)
    rows = [r for r in results if r.kind == "table"]
    assert [(r.key, r.payload) for r in rows] == [
        ("BTC", 1.5),
        ("ETH", 2.0),
    ]


# ---------------------------------------------------------------------------
# Tumbling event-time window — the CEP differentiator.
# ---------------------------------------------------------------------------


def _trade_ts(ev: dict[str, object]) -> int:
    return int(ev["ts_ns"])


def _windowed_app() -> App:
    def windowed_aggregator(
        ctx: AgentContext,
        ev: dict[str, object],
    ) -> None:
        assert ctx.current_ts_ns is not None
        bucket = bucket_index(ctx.current_ts_ns, 1_000_000_000)
        key = str(ev["sym"])
        cells = ctx.tables["totals"]
        cell_key = (bucket, key)
        cells[cell_key] = cells.get(cell_key, 0.0) + float(ev["qty"])

    return (
        App(id="cep")
        .topic("trades")
        .table(
            "totals",
            default=lambda: 0.0,
            window=TumblingWindowSpec(
                window_ns=1_000_000_000,
                ts_fn=_trade_ts,
            ),
        )
        .agent("trades", fn=windowed_aggregator)
    )


def _windowed_events() -> list[InboundEvent]:
    return [
        InboundEvent(
            topic_name="trades",
            payload={"ts_ns": 500_000_000, "sym": "BTC", "qty": 1.0},
        ),
        InboundEvent(
            topic_name="trades",
            payload={"ts_ns": 1_500_000_000, "sym": "BTC", "qty": 2.0},
        ),
        InboundEvent(
            topic_name="trades",
            payload={"ts_ns": 1_700_000_000, "sym": "ETH", "qty": 5.0},
        ),
        InboundEvent(
            topic_name="trades",
            payload={"ts_ns": 600_000_000, "sym": "AAA", "qty": 3.0},
        ),
        InboundEvent(
            topic_name="trades",
            payload={"ts_ns": 2_500_000_000, "sym": "BTC", "qty": 4.0},
        ),
    ]


def test_tumbling_window_emits_in_bucket_then_key_order() -> None:
    results = run_app(_windowed_app(), _windowed_events())
    rows = [r for r in results if r.kind == "window"]
    actual = [(r.bucket_idx, r.key, r.payload) for r in rows]
    assert actual == [
        (0, "AAA", 3.0),
        (0, "BTC", 1.0),
        (1, "BTC", 2.0),
        (1, "ETH", 5.0),
        (2, "BTC", 4.0),
    ]


def test_tumbling_window_agent_context_carries_event_time() -> None:
    seen: list[int | None] = []

    def probe(ctx: AgentContext, ev: dict[str, object]) -> None:
        seen.append(ctx.current_ts_ns)

    a = (
        App(id="x")
        .topic("trades")
        .table(
            "totals",
            default=lambda: 0,
            window=TumblingWindowSpec(window_ns=1_000_000_000, ts_fn=_trade_ts),
        )
        .agent("trades", fn=probe)
    )
    run_app(
        a,
        [
            InboundEvent(
                topic_name="trades",
                payload={"ts_ns": 700_000_000, "sym": "BTC", "qty": 1.0},
            )
        ],
    )
    assert seen == [700_000_000]


def test_tumbling_window_rejects_non_tuple_cell() -> None:
    def bad_aggregator(ctx: AgentContext, ev: dict[str, object]) -> None:
        ctx.tables["totals"]["plain-key"] = 1.0  # not a (bucket, key) tuple

    a = (
        App(id="x")
        .topic("trades")
        .table(
            "totals",
            default=lambda: 0.0,
            window=TumblingWindowSpec(window_ns=1_000_000_000, ts_fn=_trade_ts),
        )
        .agent("trades", fn=bad_aggregator)
    )
    with pytest.raises(TypeError):
        run_app(
            a,
            [
                InboundEvent(
                    topic_name="trades",
                    payload={"ts_ns": 0, "sym": "BTC", "qty": 1.0},
                )
            ],
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay determinism.
# ---------------------------------------------------------------------------


def test_replay_equality_three_runs() -> None:
    app = _windowed_app()
    events = _windowed_events()
    run_a = run_app(app, events)
    run_b = run_app(app, events)
    run_c = run_app(app, events)
    assert run_a == run_b == run_c


def test_replay_digest_is_app_stable() -> None:
    digest_a = _windowed_app().app_digest()
    digest_b = _windowed_app().app_digest()
    digest_c = _windowed_app().app_digest()
    assert digest_a == digest_b == digest_c


def test_replay_insertion_order_independence_on_windowed_table() -> None:
    """Permuting events that fall into different buckets must NOT change
    the final emission order."""

    base = _windowed_events()
    permuted = [base[2], base[0], base[4], base[3], base[1]]
    run_base = tuple(r for r in run_app(_windowed_app(), base) if r.kind == "window")
    run_perm = tuple(r for r in run_app(_windowed_app(), permuted) if r.kind == "window")
    assert [(r.bucket_idx, r.key, r.payload) for r in run_base] == [
        (r.bucket_idx, r.key, r.payload) for r in run_perm
    ]


# ---------------------------------------------------------------------------
# Cross-process worker bridge — driven in-process via queue.Queue.
# ---------------------------------------------------------------------------


def test_faust_worker_main_drains_to_outbound_then_emits_sentinel() -> None:
    def fanout(ctx: AgentContext, ev: int) -> list[SendOp]:
        return [SendOp(topic_name="out", payload=ev * 10)]

    a = App(id="x").topic("in").topic("out").agent("in", fn=fanout)
    inbound: queue.Queue[object] = queue.Queue()
    outbound: queue.Queue[object] = queue.Queue()
    for v in [1, 2, 3, 4]:
        inbound.put(InboundEvent(topic_name="in", payload=v))
    inbound.put(FaustBusSentinel())

    faust_worker_main(a, inbound, outbound, batch_size=2)  # type: ignore[arg-type]

    drained = drain_queue(outbound, timeout_s=1.0)  # type: ignore[arg-type]
    assert tuple(r.payload for r in drained if r.kind == "event") == (10, 20, 30, 40)


def test_faust_worker_main_handles_partial_final_batch() -> None:
    def passthrough(ctx: AgentContext, ev: int) -> list[SendOp]:
        return [SendOp(topic_name="out", payload=ev + 1)]

    a = App(id="x").topic("in").topic("out").agent("in", fn=passthrough)
    inbound: queue.Queue[object] = queue.Queue()
    outbound: queue.Queue[object] = queue.Queue()
    for v in [1, 2, 3]:
        inbound.put(InboundEvent(topic_name="in", payload=v))
    inbound.put(FaustBusSentinel())

    faust_worker_main(a, inbound, outbound, batch_size=2)  # type: ignore[arg-type]

    drained = drain_queue(outbound, timeout_s=1.0)  # type: ignore[arg-type]
    assert tuple(r.payload for r in drained if r.kind == "event") == (2, 3, 4)


def test_faust_worker_main_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError):
        faust_worker_main(App(id="x"), queue.Queue(), queue.Queue(), batch_size=0)  # type: ignore[arg-type]


def test_faust_worker_main_rejects_non_inbound_event() -> None:
    inbound: queue.Queue[object] = queue.Queue()
    outbound: queue.Queue[object] = queue.Queue()
    inbound.put("not-an-inbound-event")
    inbound.put(FaustBusSentinel())
    with pytest.raises(TypeError):
        faust_worker_main(App(id="x"), inbound, outbound)  # type: ignore[arg-type]


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
# faust lazy factory — pinned NotImplementedError.
# ---------------------------------------------------------------------------


def test_faust_app_factory_is_gated() -> None:
    with pytest.raises(NotImplementedError) as ei:
        faust_app_factory()
    msg = str(ei.value)
    assert "research-acceptance" in msg
    assert "shadow-equivalence" in msg


# ---------------------------------------------------------------------------
# AST guardrails — module hygiene.
# ---------------------------------------------------------------------------


def test_module_has_adapted_from_header() -> None:
    text = FAUST_BUS_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: robinhood/faust" in text


def test_module_has_no_top_level_forbidden_imports() -> None:
    forbidden = {
        "faust",
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
    """B27 / B28 / INV-71 authority symmetry — bus is transport only."""

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
