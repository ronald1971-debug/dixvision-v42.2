"""NEUR-01 — Indira pulse / microstructure signal extractor.

Pure function over an observed window of trades. Emits a
:class:`PulseSignal` describing the directional pulse and its
normalized intensity.

Authority:
    No imports from any engine. No clock reads. No I/O.

Determinism (INV-15):
    Given the same input window the function must return an equal
    PulseSignal. The function does NOT call ``time.time_ns()`` or
    any other clock — the caller stamps ``ts_ns``. ``random`` is
    not used.

The pulse formula (v1, intentionally simple — to be tightened by
later layers, not by sensor fanciness):

* Compute signed flow ``F = sum(side_sign(t) * size(t)) for t in window``
  where ``side_sign(BUY) = +1``, ``side_sign(SELL) = -1``.
* Compute total magnitude ``M = sum(size(t))``.
* If ``M == 0`` → :data:`POLARITY_NEUTRAL`, intensity 0.
* Otherwise polarity ← sign of ``F``, intensity ← ``|F| / M`` clipped
  to ``[0, 1]``.

That ratio is the "trader-pulse" of the window: 1.0 means perfectly
one-sided flow, 0.0 means perfectly balanced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sensory.neuromorphic.contracts import (
    POLARITY_LONG,
    POLARITY_NEUTRAL,
    POLARITY_SHORT,
    PulseSignal,
)

_SIDE_BUY: str = "BUY"
_SIDE_SELL: str = "SELL"


@dataclass(frozen=True, slots=True)
class TradeSample:
    """One observed trade (the input element to NEUR-01).

    Attributes:
        side: Aggressor side, ``"BUY"`` or ``"SELL"``.
        size: Traded size. Must be finite and ``> 0``.
    """

    side: str
    size: float

    def __post_init__(self) -> None:
        if self.side not in (_SIDE_BUY, _SIDE_SELL):
            raise ValueError(
                "TradeSample.side must be 'BUY' or 'SELL'"
            )
        # Reject NaN, +inf, -inf, 0 and negatives in one branch.
        if not (self.size > 0 and self.size < float("inf")):
            raise ValueError(
                "TradeSample.size must be finite and > 0"
            )


def extract_pulse(
    *,
    ts_ns: int,
    source: str,
    symbol: str,
    window: Sequence[TradeSample] | Iterable[TradeSample],
    evidence: Mapping[str, str] | None = None,
) -> PulseSignal:
    """Extract a :class:`PulseSignal` from a window of trades.

    Args:
        ts_ns: Caller-supplied window-close timestamp.
        source: Stable source identifier.
        symbol: Per-instrument identifier.
        window: Iterable of :class:`TradeSample`. May be empty — an
            empty window emits a NEUTRAL pulse with intensity 0 and
            ``sample_count`` 1 (the caller already decided the bar
            closed; signaling absence is itself a signal).

            Note: ``sample_count`` is clipped to ``>= 1`` because
            :class:`PulseSignal` rejects 0; the empty-window case
            still represents one observation: "the bar closed empty".

            Use ``evidence={"empty_window": "true"}`` if you want a
            downstream consumer to distinguish "balanced flow" from
            "no flow at all".
        evidence: Optional structural metadata. ``None`` becomes the
            empty mapping.

    Returns:
        :class:`PulseSignal` with polarity in
        ``{LONG, SHORT, NEUTRAL}`` and intensity in ``[0.0, 1.0]``.

    Raises:
        ValueError: If any sample fails :class:`TradeSample`'s own
            validation.
    """
    samples = list(window)

    signed_flow: float = 0.0
    magnitude: float = 0.0
    for sample in samples:
        sign = 1.0 if sample.side == _SIDE_BUY else -1.0
        signed_flow += sign * sample.size
        magnitude += sample.size

    if magnitude == 0.0:
        polarity = POLARITY_NEUTRAL
        intensity = 0.0
    else:
        if signed_flow > 0.0:
            polarity = POLARITY_LONG
        elif signed_flow < 0.0:
            polarity = POLARITY_SHORT
        else:
            polarity = POLARITY_NEUTRAL
        ratio = abs(signed_flow) / magnitude
        # Clip defensively even though the algebra gives [0, 1].
        intensity = max(0.0, min(1.0, ratio))

    sample_count = max(1, len(samples))
    return PulseSignal(
        ts_ns=ts_ns,
        source=source,
        symbol=symbol,
        polarity=polarity,
        intensity=intensity,
        sample_count=sample_count,
        evidence=dict(evidence) if evidence else {},
    )


__all__ = ["TradeSample", "extract_pulse"]
