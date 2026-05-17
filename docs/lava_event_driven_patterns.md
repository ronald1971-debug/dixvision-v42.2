# Lava-style event-driven patterns

`core/event_cognition/lava_patterns.py` distils the **behavioural
contract** of Intel's
[Lava](https://github.com/lava-nc/lava) (BSD-3-Clause) neuromorphic
framework into a pure asyncio + stdlib pattern.

## Why pattern-only

Lava ships a hardware-targeted runtime that drives Loihi 2 and the
underlying neuromorphic substrate. None of that is useful inside
DIX, where the runtime is plain Python + asyncio. **What is useful
is the architectural discipline Lava enforces.** This module
captures that discipline, not the hardware.

`tools/authority_lint` therefore classifies the module as
`OFFLINE_ONLY | ADVISORY (PATTERN_ONLY)`:

- No vendor imports anywhere in the module.
- Pure stdlib (`asyncio`, `dataclasses`, `typing`, `math`).
- No clock reads — every spike carries a caller-supplied
  `ts_ns` (CONST-04).
- No global state, no cross-engine imports (CONST-19).

## Lava primitives mirrored

| Lava primitive | DIX mirror | Notes |
|----------------|-----------|-------|
| `AbstractProcess` (with `Vars` + ports) | `LavaProcess` | Compute-on-spike actor; no shared mutable state between actors. |
| `InPort`        | `LavaInPort`  | Typed bounded `asyncio.Queue`; FIFO per producer; single-bind. |
| `OutPort`       | `LavaOutPort` | Typed fan-out; broadcasts to every connected sink. |
| Process composition (Plug & Play) | `LavaGraph` + `LavaScheduler` | Declarative wiring + cycle / dangling-input validation. |
| Spike            | `LavaSpike[T]` | Immutable `(ts_ns, payload, source_port_id)`. |

Only the **functional contract** is preserved: typed bounded
ports, compute-on-spike actors, declarative DAG composition. The
Loihi runtime, the hardware dispatcher, the Magma compiler — all
of that is intentionally absent.

## The three guarantees

1. **Sensory nodes do not poll.** A `LavaProcess` only computes
   when a spike arrives on one of its in-ports. Idle = literally
   idle (the underlying coroutine is suspended on
   `asyncio.Queue.get`).
2. **Inputs are typed and bounded.** `LavaInPort` rejects
   `capacity <= 0` and `capacity > 4096`. A slow consumer blocks
   its producer via `asyncio.Queue.put` backpressure — there is no
   silent unbounded buffering.
3. **Computation is functional.** Each `on_spike` call takes one
   input spike and emits zero or more output spikes via `emit`.
   No clock reads, no shared mutable state between actors.

## Where this applies in DIX

The intended consumers are **sensory nodes** under
`sensory/` that today poll on a fixed interval. Examples:

- `sensory/neuromorphic/snn_lif.py` (B-14) — could be wired so the
  detector only fires when an upstream `SignalEvent` arrives,
  rather than on every tick.
- `sensory/neuromorphic/nengo_cognitive.py` (C-43) — same idea for
  the cognitive ensemble: spike on regime-update events instead
  of polling.
- `sensory/web_autolearn/` — crawlers can be modelled as
  `LavaProcess` subclasses awaiting on a URL-discovery in-port.

The migration is **opt-in**: nothing changes for existing nodes
until they are rewritten to subclass `LavaProcess`. The Lava
pattern is registered as `LAVA_PATTERNS_VERSION = "lava-patterns/v1"`
so audits can identify which sensors have been migrated.

## Determinism (INV-15)

- `LavaInPort` is FIFO per producer (`asyncio.Queue` semantics).
- `LavaProcess.run` consumes one spike at a time. Given the same
  input sequence per port, every actor produces byte-identical
  output spikes across runs.
- `LavaScheduler` schedules concurrent tasks via
  `asyncio.gather`, but each `LavaProcess` has its own coroutine
  and never reads from a shared mutable list — so ordering inside
  a process is deterministic even if the scheduler ordering
  between processes is not.

## Example

```python
import asyncio
from core.event_cognition.lava_patterns import (
    LavaGraph,
    LavaInPort,
    LavaOutPort,
    LavaScheduler,
    LavaSpike,
    PassthroughProcess,
)


async def example() -> list[LavaSpike[float]]:
    source_out = LavaOutPort[float](port_id="source.out")
    mid_in = LavaInPort[float](port_id="mid.in", capacity=4)
    mid_out = LavaOutPort[float](port_id="mid.out")
    sink_in = LavaInPort[float](port_id="sink.in", capacity=4)

    midway = PassthroughProcess[float](
        process_id="midway",
        in_ports=(mid_in,),
        out_ports=(mid_out,),
    )
    sink = PassthroughProcess[float](
        process_id="sink",
        in_ports=(sink_in,),
        out_ports=(),
    )

    graph = LavaGraph(processes=(midway, sink))
    graph.connect(source_out, mid_in)
    graph.connect(mid_out, sink_in)
    graph.validate()

    scheduler = LavaScheduler(graph=graph)
    await source_out.send(LavaSpike(ts_ns=1, payload=1.5))
    await source_out.send(LavaSpike(ts_ns=2, payload=2.5))
    await source_out.close()
    return await scheduler.run()


asyncio.run(example())
```

## What is intentionally absent

- **No Loihi runtime**: there is no hardware dispatcher, no
  Magma compiler, no neuron-model library.
- **No vendor imports**: `lava-nc` is not in `pyproject.toml`.
  The pattern is captured at the abstraction level only.
- **No clock**: every spike's `ts_ns` is caller-supplied.
- **No reflection / dynamic code**: subclasses define their
  behaviour by overriding `on_spike` — no `Process.run_cfg` or
  Loihi-specific descriptors.

## Authority lint

- Module: `core/event_cognition/lava_patterns.py`
- Classification: `OFFLINE_ONLY | ADVISORY (PATTERN_ONLY)`
- Imports allowed: `asyncio`, `dataclasses`, `typing`, `math`,
  `collections.abc`
- Imports forbidden: everything else
- Tests under `tests/test_lava_patterns.py` pin the import set
  via AST.
