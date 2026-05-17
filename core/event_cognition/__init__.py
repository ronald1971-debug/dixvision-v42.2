"""Event-cognition primitives — compute-on-spike patterns.

This package hosts pattern-only abstractions distilled from
external neuromorphic / event-driven references (e.g. Intel
Lava). No hardware dependency, no vendor imports. See
:mod:`core.event_cognition.lava_patterns`.
"""

from core.event_cognition.lava_patterns import (
    LAVA_PATTERNS_VERSION,
    LavaCompositionError,
    LavaGraph,
    LavaInPort,
    LavaOutPort,
    LavaPortError,
    LavaProcess,
    LavaScheduler,
    LavaSpike,
    PassthroughProcess,
)

__all__ = [
    "LAVA_PATTERNS_VERSION",
    "LavaCompositionError",
    "LavaGraph",
    "LavaInPort",
    "LavaOutPort",
    "LavaPortError",
    "LavaProcess",
    "LavaScheduler",
    "LavaSpike",
    "PassthroughProcess",
]
