"""C-01 bytewax — Python-Native Stream Processing.

OFFLINE_ONLY adaptation of bytewax's :class:`Dataflow` programming
model applied to DIX's 4-event contract. The pipeline runs as a
SEPARATE PROCESS feeding DIX via a :mod:`multiprocessing` queue.
This module itself NEVER imports the bytewax PyPI package at module
load time — the in-process Python translation is sufficient for
offline replay tests and for the cross-process queue bridge that
production callers wire up.

See :mod:`system_engine.streaming.event_fabric` for the public
surface.
"""

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
    spawn_fabric_worker,
)

__all__ = [
    "EVENT_FABRIC_VERSION",
    "Dataflow",
    "EventFabricSentinel",
    "FabricResult",
    "FilterOp",
    "KeyByOp",
    "MapOp",
    "Operator",
    "ReduceOp",
    "TumblingWindowOp",
    "bytewax_dataflow_factory",
    "drain_queue",
    "fabric_worker_main",
    "run_dataflow",
    "spawn_fabric_worker",
]
