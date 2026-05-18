"""C-44 — Tests for core/event_cognition/lava_patterns.py.

Authority pins, port semantics, process FIFO discipline, graph
validation, and INV-15 byte-identical replay.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from core.event_cognition import lava_patterns as lp
from core.event_cognition.lava_patterns import (
    LAVA_PATTERNS_VERSION,
    NEW_PIP_DEPENDENCIES,
    LavaCompositionError,
    LavaEdge,
    LavaGraph,
    LavaInPort,
    LavaOutPort,
    LavaPortError,
    LavaProcess,
    LavaScheduler,
    LavaSpike,
    PassthroughProcess,
)

MODULE_PATH = Path(lp.__file__)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)


# ============================================================== authority


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: lava-nc/lava")


def test_authority_pip_dependencies_empty() -> None:
    """PATTERN_ONLY tier — zero external deps."""

    assert NEW_PIP_DEPENDENCIES == ()


def test_authority_version_string_stable() -> None:
    assert LAVA_PATTERNS_VERSION == "lava-patterns/v1"


def _iter_top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _iter_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_authority_no_vendor_imports() -> None:
    forbidden = {
        "lava",
        "nengo",
        "norse",
        "torch",
        "snntorch",
        "bindsnet",
        "brian2",
        "numpy",
        "pandas",
        "polars",
        "jax",
    }
    imports = _iter_imports(MODULE_AST)
    for name in imports:
        root = name.split(".")[0]
        assert root not in forbidden, f"forbidden vendor import: {name}"


def test_authority_no_engine_cross_imports() -> None:
    imports = _iter_imports(MODULE_AST)
    forbidden_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
    )
    for name in imports:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), f"forbidden cross-engine import: {name}"


def test_authority_no_runtime_imports() -> None:
    forbidden_roots = {
        "time",
        "datetime",
        "random",
        "os",
        "sys",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "logging",
        "threading",
        "queue",
    }
    tops = _iter_top_level_imports(MODULE_AST)
    for name in tops:
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden top-level runtime import: {name}"


def test_authority_allowed_stdlib_imports_only() -> None:
    allowed_roots = {
        "__future__",
        "asyncio",
        "math",
        "collections",
        "dataclasses",
        "typing",
    }
    tops = _iter_top_level_imports(MODULE_AST)
    for name in tops:
        root = name.split(".")[0]
        assert root in allowed_roots, f"unexpected import: {name}"


# ============================================================== LavaSpike


def test_spike_is_frozen_slotted() -> None:
    s = LavaSpike[float](ts_ns=1, payload=1.0)
    assert "__slots__" in LavaSpike.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.ts_ns = 2  # type: ignore[misc]


def test_spike_rejects_negative_ts() -> None:
    with pytest.raises(LavaPortError):
        LavaSpike[float](ts_ns=-1, payload=1.0)


def test_spike_rejects_non_int_ts() -> None:
    with pytest.raises(LavaPortError):
        LavaSpike[float](ts_ns=1.5, payload=1.0)  # type: ignore[arg-type]


def test_spike_rejects_bool_ts() -> None:
    with pytest.raises(LavaPortError):
        LavaSpike[float](ts_ns=True, payload=1.0)  # type: ignore[arg-type]


def test_spike_rejects_non_str_source_port_id() -> None:
    with pytest.raises(LavaPortError):
        LavaSpike[float](
            ts_ns=1,
            payload=1.0,
            source_port_id=1,  # type: ignore[arg-type]
        )


# ============================================================== LavaInPort


def test_in_port_construction() -> None:
    p = LavaInPort[int](port_id="x", capacity=4)
    assert p.port_id == "x"
    assert p.capacity == 4
    assert not p.is_bound
    assert not p.is_closed
    assert p.bound_out_port_id == ""


def test_in_port_rejects_empty_id() -> None:
    with pytest.raises(LavaPortError):
        LavaInPort[int](port_id="", capacity=1)


def test_in_port_rejects_oversize_id() -> None:
    with pytest.raises(LavaPortError):
        LavaInPort[int](port_id="x" * 129, capacity=1)


def test_in_port_rejects_zero_capacity() -> None:
    with pytest.raises(LavaPortError):
        LavaInPort[int](port_id="x", capacity=0)


def test_in_port_rejects_negative_capacity() -> None:
    with pytest.raises(LavaPortError):
        LavaInPort[int](port_id="x", capacity=-1)


def test_in_port_rejects_oversize_capacity() -> None:
    with pytest.raises(LavaPortError):
        LavaInPort[int](port_id="x", capacity=10_000)


def test_in_port_bind_once() -> None:
    p = LavaInPort[int](port_id="x")
    p.bind("o1")
    assert p.is_bound
    with pytest.raises(LavaPortError):
        p.bind("o2")


def test_in_port_push_after_close_rejected() -> None:
    async def main() -> None:
        p = LavaInPort[int](port_id="x", capacity=1)
        await p.close()
        with pytest.raises(LavaPortError):
            await p.push(LavaSpike[int](ts_ns=1, payload=1))

    asyncio.run(main())


def test_in_port_push_rejects_non_spike() -> None:
    async def main() -> None:
        p = LavaInPort[int](port_id="x", capacity=1)
        with pytest.raises(LavaPortError):
            await p.push("not-a-spike")  # type: ignore[arg-type]

    asyncio.run(main())


def test_in_port_double_close_idempotent() -> None:
    async def main() -> None:
        p = LavaInPort[int](port_id="x", capacity=1)
        await p.close()
        await p.close()
        first = await p.receive()
        assert first is None

    asyncio.run(main())


# ============================================================== LavaOutPort


def test_out_port_construction() -> None:
    p = LavaOutPort[int](port_id="o")
    assert p.port_id == "o"
    assert p.fanout == 0
    assert p.sink_ids == ()


def test_out_port_rejects_empty_id() -> None:
    with pytest.raises(LavaPortError):
        LavaOutPort[int](port_id="")


def test_out_port_connect_requires_in_port() -> None:
    p = LavaOutPort[int](port_id="o")
    with pytest.raises(LavaPortError):
        p.connect("not-a-port")  # type: ignore[arg-type]


def test_out_port_connect_records_sink() -> None:
    out = LavaOutPort[int](port_id="o")
    in_port = LavaInPort[int](port_id="i")
    out.connect(in_port)
    assert out.fanout == 1
    assert out.sink_ids == ("i",)
    assert in_port.bound_out_port_id == "o"


def test_out_port_connect_rejects_duplicate_sink() -> None:
    out = LavaOutPort[int](port_id="o")
    in_port = LavaInPort[int](port_id="i")
    out.connect(in_port)
    with pytest.raises(LavaPortError):
        out.connect(in_port)


def test_out_port_send_after_close_rejected() -> None:
    async def main() -> None:
        out = LavaOutPort[int](port_id="o")
        await out.close()
        with pytest.raises(LavaPortError):
            await out.send(LavaSpike[int](ts_ns=1, payload=1))

    asyncio.run(main())


def test_out_port_send_rejects_non_spike() -> None:
    async def main() -> None:
        out = LavaOutPort[int](port_id="o")
        with pytest.raises(LavaPortError):
            await out.send("not-a-spike")  # type: ignore[arg-type]

    asyncio.run(main())


def test_out_port_retags_source_port_id() -> None:
    async def main() -> None:
        out = LavaOutPort[int](port_id="o")
        in_port = LavaInPort[int](port_id="i", capacity=2)
        out.connect(in_port)
        await out.send(LavaSpike[int](ts_ns=1, payload=42))
        delivered = await in_port.receive()
        assert isinstance(delivered, LavaSpike)
        assert delivered.payload == 42
        assert delivered.source_port_id == "o"

    asyncio.run(main())


def test_out_port_multicast_to_multiple_sinks() -> None:
    async def main() -> None:
        out = LavaOutPort[int](port_id="o")
        s1 = LavaInPort[int](port_id="s1", capacity=2)
        s2 = LavaInPort[int](port_id="s2", capacity=2)
        out.connect(s1)
        out.connect(s2)
        await out.send(LavaSpike[int](ts_ns=1, payload=7))
        m1 = await s1.receive()
        m2 = await s2.receive()
        assert m1 is not None and m1.payload == 7
        assert m2 is not None and m2.payload == 7

    asyncio.run(main())


# ============================================================== LavaProcess


def test_process_requires_non_empty_id() -> None:
    with pytest.raises(LavaCompositionError):
        LavaProcess(process_id="")


def test_process_rejects_non_tuple_in_ports() -> None:
    with pytest.raises(LavaCompositionError):
        LavaProcess(
            process_id="p",
            in_ports=[LavaInPort(port_id="i")],  # type: ignore[arg-type]
        )


def test_process_rejects_non_lava_in_ports() -> None:
    with pytest.raises(LavaCompositionError):
        LavaProcess(
            process_id="p",
            in_ports=("not-a-port",),  # type: ignore[arg-type]
        )


def test_process_rejects_duplicate_in_port_ids() -> None:
    a = LavaInPort[int](port_id="dup")
    b = LavaInPort[int](port_id="dup")
    with pytest.raises(LavaCompositionError):
        LavaProcess(process_id="p", in_ports=(a, b))


def test_process_rejects_duplicate_out_port_ids() -> None:
    a = LavaOutPort[int](port_id="dup")
    b = LavaOutPort[int](port_id="dup")
    with pytest.raises(LavaCompositionError):
        LavaProcess(process_id="p", out_ports=(a, b))


def test_process_rejects_oversize_id() -> None:
    with pytest.raises(LavaCompositionError):
        LavaProcess(process_id="x" * 129)


def test_process_emit_rejects_unregistered_out_port() -> None:
    async def main() -> None:
        p = LavaProcess(process_id="p")
        rogue = LavaOutPort[int](port_id="rogue")
        with pytest.raises(LavaCompositionError):
            await p.emit(rogue, LavaSpike[int](ts_ns=1, payload=1))

    asyncio.run(main())


def test_process_with_no_in_ports_closes_out_ports() -> None:
    async def main() -> None:
        out = LavaOutPort[int](port_id="o")
        p = LavaProcess(process_id="p", out_ports=(out,))
        await p.run()
        assert out.is_closed

    asyncio.run(main())


def test_passthrough_process_basic() -> None:
    async def main() -> None:
        in_port = LavaInPort[int](port_id="i", capacity=4)
        out_port = LavaOutPort[int](port_id="o")
        sink_in = LavaInPort[int](port_id="s", capacity=4)
        out_port.connect(sink_in)
        process = PassthroughProcess[int](
            process_id="pt",
            in_ports=(in_port,),
            out_ports=(out_port,),
        )
        in_port.bind("source")
        await in_port.push(LavaSpike[int](ts_ns=1, payload=10))
        await in_port.push(LavaSpike[int](ts_ns=2, payload=20))
        await in_port.close()
        await process.run()
        assert process.consumed_count == 2
        assert process.emitted_count == 2
        s1 = await sink_in.receive()
        s2 = await sink_in.receive()
        s3 = await sink_in.receive()
        assert s1 is not None and s1.payload == 10
        assert s2 is not None and s2.payload == 20
        assert s3 is None  # close propagated

    asyncio.run(main())


def test_passthrough_with_no_out_ports_is_noop() -> None:
    async def main() -> None:
        in_port = LavaInPort[int](port_id="i", capacity=2)
        process = PassthroughProcess[int](
            process_id="pt",
            in_ports=(in_port,),
            out_ports=(),
        )
        in_port.bind("source")
        await in_port.push(LavaSpike[int](ts_ns=1, payload=10))
        await in_port.close()
        await process.run()
        assert process.consumed_count == 1
        assert process.emitted_count == 0

    asyncio.run(main())


def test_process_fifo_per_port() -> None:
    """Each in-port preserves spike FIFO order."""

    async def main() -> None:
        in_port = LavaInPort[int](port_id="i", capacity=8)
        out_port = LavaOutPort[int](port_id="o")
        sink = LavaInPort[int](port_id="s", capacity=8)
        out_port.connect(sink)
        process = PassthroughProcess[int](
            process_id="p",
            in_ports=(in_port,),
            out_ports=(out_port,),
        )
        in_port.bind("source")
        for i in range(5):
            await in_port.push(LavaSpike[int](ts_ns=i, payload=i))
        await in_port.close()
        await process.run()
        out_seq: list[int] = []
        while True:
            spike = await sink.receive()
            if spike is None:
                break
            out_seq.append(spike.payload)
        assert out_seq == [0, 1, 2, 3, 4]

    asyncio.run(main())


def test_process_listener_invoked() -> None:
    async def main() -> None:
        in_port = LavaInPort[int](port_id="i", capacity=8)
        process = PassthroughProcess[int](process_id="p", in_ports=(in_port,), out_ports=())
        seen: list[int] = []

        async def listener(_p: LavaInPort[Any], s: LavaSpike[Any]) -> None:
            seen.append(s.payload)

        process.add_spike_listener(listener)
        in_port.bind("source")
        await in_port.push(LavaSpike[int](ts_ns=1, payload=1))
        await in_port.push(LavaSpike[int](ts_ns=2, payload=2))
        await in_port.close()
        await process.run()
        assert seen == [1, 2]

    asyncio.run(main())


def test_process_listener_must_be_callable() -> None:
    p = LavaProcess(process_id="p")
    with pytest.raises(LavaCompositionError):
        p.add_spike_listener("not-callable")


def test_process_listener_remove_unknown_raises() -> None:
    p = LavaProcess(process_id="p")
    with pytest.raises(LavaCompositionError):
        p.remove_spike_listener(lambda *a, **kw: None)


# ============================================================== LavaGraph


def test_graph_rejects_non_tuple_processes() -> None:
    with pytest.raises(LavaCompositionError):
        LavaGraph(processes=[LavaProcess(process_id="p")])  # type: ignore[arg-type]


def test_graph_rejects_non_process_entries() -> None:
    with pytest.raises(LavaCompositionError):
        LavaGraph(processes=("not-a-process",))  # type: ignore[arg-type]


def test_graph_rejects_duplicate_process_ids() -> None:
    a = LavaProcess(process_id="dup")
    b = LavaProcess(process_id="dup")
    with pytest.raises(LavaCompositionError):
        LavaGraph(processes=(a, b))


def test_graph_rejects_dangling_input() -> None:
    in_port = LavaInPort[int](port_id="i")
    process = LavaProcess(process_id="p", in_ports=(in_port,))
    graph = LavaGraph(processes=(process,))
    with pytest.raises(LavaCompositionError):
        graph.validate()


def test_graph_connect_records_edge() -> None:
    in_port = LavaInPort[int](port_id="i")
    out_port = LavaOutPort[int](port_id="o")
    src = LavaProcess(process_id="src", out_ports=(out_port,))
    sink = LavaProcess(process_id="sink", in_ports=(in_port,))
    graph = LavaGraph(processes=(src, sink))
    graph.connect(out_port, in_port)
    assert graph.edge_count() == 1
    edge = graph.edges[0]
    assert isinstance(edge, LavaEdge)
    assert edge.producer_process_id == "src"
    assert edge.consumer_process_id == "sink"
    graph.validate()


def test_graph_connect_rejects_unowned_in_port() -> None:
    out_port = LavaOutPort[int](port_id="o")
    src = LavaProcess(process_id="src", out_ports=(out_port,))
    graph = LavaGraph(processes=(src,))
    rogue = LavaInPort[int](port_id="rogue")
    with pytest.raises(LavaCompositionError):
        graph.connect(out_port, rogue)


def test_graph_connect_rejects_non_lava_args() -> None:
    in_port = LavaInPort[int](port_id="i")
    sink = LavaProcess(process_id="sink", in_ports=(in_port,))
    graph = LavaGraph(processes=(sink,))
    with pytest.raises(LavaCompositionError):
        graph.connect("not-a-port", in_port)  # type: ignore[arg-type]
    out_port = LavaOutPort[int](port_id="o")
    with pytest.raises(LavaCompositionError):
        graph.connect(out_port, "not-a-port")  # type: ignore[arg-type]


def test_graph_detects_two_node_cycle() -> None:
    a_in = LavaInPort[int](port_id="a.in")
    a_out = LavaOutPort[int](port_id="a.out")
    b_in = LavaInPort[int](port_id="b.in")
    b_out = LavaOutPort[int](port_id="b.out")
    a = LavaProcess(process_id="a", in_ports=(a_in,), out_ports=(a_out,))
    b = LavaProcess(process_id="b", in_ports=(b_in,), out_ports=(b_out,))
    graph = LavaGraph(processes=(a, b))
    graph.connect(a_out, b_in)
    graph.connect(b_out, a_in)
    with pytest.raises(LavaCompositionError):
        graph.validate()


def test_graph_accepts_diamond() -> None:
    root_out = LavaOutPort[int](port_id="root.out")
    l_in = LavaInPort[int](port_id="l.in")
    l_out = LavaOutPort[int](port_id="l.out")
    r_in = LavaInPort[int](port_id="r.in")
    r_out = LavaOutPort[int](port_id="r.out")
    s_in_l = LavaInPort[int](port_id="s.l")
    s_in_r = LavaInPort[int](port_id="s.r")
    root = LavaProcess(process_id="root", out_ports=(root_out,))
    left = LavaProcess(process_id="left", in_ports=(l_in,), out_ports=(l_out,))
    right = LavaProcess(process_id="right", in_ports=(r_in,), out_ports=(r_out,))
    sink = LavaProcess(process_id="sink", in_ports=(s_in_l, s_in_r))
    graph = LavaGraph(processes=(root, left, right, sink))
    # Root fans out to both branches.
    root_out.connect(l_in)
    root_out.connect(r_in)
    graph.edges = graph.edges + (
        LavaEdge(
            out_port_id="root.out",
            in_port_id="l.in",
            producer_process_id="root",
            consumer_process_id="left",
        ),
        LavaEdge(
            out_port_id="root.out",
            in_port_id="r.in",
            producer_process_id="root",
            consumer_process_id="right",
        ),
    )
    graph.connect(l_out, s_in_l)
    graph.connect(r_out, s_in_r)
    graph.validate()


# ============================================================== LavaScheduler


def test_scheduler_requires_graph() -> None:
    with pytest.raises(LavaCompositionError):
        LavaScheduler(graph="not-a-graph")  # type: ignore[arg-type]


def test_scheduler_rejects_invalid_timeout() -> None:
    async def main() -> None:
        graph = LavaGraph(processes=())
        scheduler = LavaScheduler(graph=graph)
        with pytest.raises(LavaCompositionError):
            await scheduler.run(timeout_s=0.0)
        with pytest.raises(LavaCompositionError):
            await scheduler.run(timeout_s=-1.0)
        with pytest.raises(LavaCompositionError):
            await scheduler.run(timeout_s=float("nan"))
        with pytest.raises(LavaCompositionError):
            await scheduler.run(timeout_s=float("inf"))

    asyncio.run(main())


def _build_two_stage_graph() -> tuple[LavaOutPort[int], LavaScheduler, PassthroughProcess[int]]:
    source_out = LavaOutPort[int](port_id="source.out")
    mid_in = LavaInPort[int](port_id="mid.in", capacity=64)
    mid_out = LavaOutPort[int](port_id="mid.out")
    sink_in = LavaInPort[int](port_id="sink.in", capacity=64)
    midway = PassthroughProcess[int](
        process_id="midway",
        in_ports=(mid_in,),
        out_ports=(mid_out,),
    )
    sink = PassthroughProcess[int](process_id="sink", in_ports=(sink_in,), out_ports=())
    graph = LavaGraph(processes=(midway, sink))
    graph.connect(source_out, mid_in)
    graph.connect(mid_out, sink_in)
    graph.validate()
    return source_out, LavaScheduler(graph=graph), sink


def test_scheduler_end_to_end() -> None:
    async def main() -> list[LavaSpike[Any]]:
        source_out, scheduler, _sink = _build_two_stage_graph()
        await source_out.send(LavaSpike[int](ts_ns=1, payload=10))
        await source_out.send(LavaSpike[int](ts_ns=2, payload=20))
        await source_out.close()
        return await scheduler.run()

    spikes = asyncio.run(main())
    assert [s.payload for s in spikes] == [10, 20]


def test_scheduler_records_consumed_emitted() -> None:
    async def main() -> tuple[int, int, int]:
        source_out, scheduler, sink = _build_two_stage_graph()
        for i in range(7):
            await source_out.send(LavaSpike[int](ts_ns=i, payload=i))
        await source_out.close()
        await scheduler.run()
        return (
            scheduler.graph.processes[0].consumed_count,
            scheduler.graph.processes[0].emitted_count,
            sink.consumed_count,
        )

    consumed, emitted, sink_consumed = asyncio.run(main())
    assert consumed == 7
    assert emitted == 7
    assert sink_consumed == 7


def test_scheduler_timeout_when_source_never_closes() -> None:
    async def main() -> None:
        source_out, scheduler, _sink = _build_two_stage_graph()
        # No close → scheduler must time out.
        with pytest.raises(asyncio.TimeoutError):
            await scheduler.run(timeout_s=0.1)
        # Make sure leftover tasks don't poison the loop.
        await source_out.close()

    asyncio.run(main())


# ============================================================== INV-15


def test_scheduler_replay_deterministic() -> None:
    """Three identical runs produce byte-identical sink output."""

    async def one_run(seed_values: list[int]) -> list[tuple[int, int]]:
        source_out, scheduler, _sink = _build_two_stage_graph()
        for i, v in enumerate(seed_values):
            await source_out.send(LavaSpike[int](ts_ns=i, payload=v))
        await source_out.close()
        spikes = await scheduler.run()
        return [(s.ts_ns, s.payload) for s in spikes]

    payload = list(range(20))
    r1 = asyncio.run(one_run(payload))
    r2 = asyncio.run(one_run(payload))
    r3 = asyncio.run(one_run(payload))
    assert r1 == r2 == r3


def test_no_polling_when_idle() -> None:
    """A process awaiting on an empty in-port doesn't spin.

    We assert that ``on_spike`` is invoked exactly zero times if
    nothing is sent before the in-port is closed.
    """

    async def main() -> int:
        in_port = LavaInPort[int](port_id="i", capacity=4)
        process = PassthroughProcess[int](process_id="p", in_ports=(in_port,), out_ports=())
        in_port.bind("ext")
        await in_port.close()
        await process.run()
        return process.consumed_count

    consumed = asyncio.run(main())
    assert consumed == 0


# ============================================================== Edge tests


def test_edge_value_object_is_frozen_slotted() -> None:
    e = LavaEdge(
        out_port_id="o",
        in_port_id="i",
        producer_process_id="src",
        consumer_process_id="dst",
    )
    assert "__slots__" in LavaEdge.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.out_port_id = "x"  # type: ignore[misc]


# ============================================================== __all__


def test_all_exports_stable() -> None:
    expected = {
        "LAVA_PATTERNS_VERSION",
        "LavaCompositionError",
        "LavaEdge",
        "LavaGraph",
        "LavaInPort",
        "LavaOutPort",
        "LavaPortError",
        "LavaProcess",
        "LavaScheduler",
        "LavaSpike",
        "NEW_PIP_DEPENDENCIES",
        "PassthroughProcess",
    }
    assert set(lp.__all__) == expected
