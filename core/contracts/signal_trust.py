"""SignalTrust class — provenance label for SignalEvent (Paper-S1).

Every :class:`~core.contracts.events.SignalEvent` carries a
``signal_trust`` field that classifies *who* produced it, so the
governance / approval gate can apply the right confidence cap before
the signal turns into an :class:`~execution_engine.contracts.ExecutionIntent`.

Three classes, in order of trust:

* ``INTERNAL`` — produced by an in-process Indira plugin or
  intelligence engine path. No additional cap is applied;
  whatever ``confidence`` was emitted is what the position
  sizer sees (subject to all other governance / hazard /
  drift gates).
* ``EXTERNAL_LOW`` — read-only ingestion from a *low-trust*
  external platform (e.g. retail TradingView webhook). The
  governance gate clamps ``confidence`` to the per-source cap
  in ``registry/external_signal_trust.yaml``; if the registry
  has no entry the conservative built-in default
  (``DEFAULT_LOW_CAP``) applies.
* ``EXTERNAL_MED`` — read-only ingestion from a *medium-trust*
  external platform with verifiable backtest provenance
  (e.g. QuantConnect with a Sharpe gate + author allowlist).
  Higher cap (``DEFAULT_MED_CAP``).

Authority
---------
* No cross-engine imports — this module only re-exports primitives.
* No clock, no PRNG (INV-15 — pure value object).
* The actual cap is applied at the governance gate (Paper-S5+);
  this contract module just defines the labels and ships
  conservative defaults so a regression cannot accidentally
  promote an external signal to ``INTERNAL`` confidence.
"""

from __future__ import annotations

from enum import StrEnum

DEFAULT_LOW_CAP: float = 0.5
"""Conservative confidence cap when an EXTERNAL_LOW signal has no
per-source override in ``registry/external_signal_trust.yaml``."""

DEFAULT_MED_CAP: float = 0.7
"""Conservative confidence cap when an EXTERNAL_MED signal has no
per-source override."""


class SignalTrust(StrEnum):
    """Trust class for the producer of a :class:`SignalEvent`."""

    INTERNAL = "INTERNAL"
    EXTERNAL_LOW = "EXTERNAL_LOW"
    EXTERNAL_MED = "EXTERNAL_MED"


def default_cap_for(trust: SignalTrust) -> float | None:
    """Return the conservative built-in confidence cap for *trust*.

    ``INTERNAL`` returns :data:`None` — no cap applied; the original
    confidence is whatever the producer emitted (subject to other
    governance gates).

    ``EXTERNAL_LOW`` returns :data:`DEFAULT_LOW_CAP`,
    ``EXTERNAL_MED`` returns :data:`DEFAULT_MED_CAP`.

    Per-source overrides (more permissive *or* more restrictive) are
    loaded from ``registry/external_signal_trust.yaml`` by
    :func:`core.contracts.external_signal_trust.load_external_signal_trust`.
    """

    if trust is SignalTrust.INTERNAL:
        return None
    if trust is SignalTrust.EXTERNAL_LOW:
        return DEFAULT_LOW_CAP
    if trust is SignalTrust.EXTERNAL_MED:
        return DEFAULT_MED_CAP
    raise ValueError(f"unknown SignalTrust class: {trust!r}")


def clamp_confidence(confidence: float, cap: float | None) -> float:
    """Apply *cap* to *confidence*; pass through if cap is ``None``.

    Out-of-range confidences (outside ``[0.0, 1.0]``) raise
    :class:`ValueError` so a bad upstream value cannot bypass the cap
    via overflow.
    """

    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0.0, 1.0]; got {confidence}")
    if cap is None:
        return confidence
    if not (0.0 <= cap <= 1.0):
        raise ValueError(f"cap must be in [0.0, 1.0]; got {cap}")
    return min(confidence, cap)


__all__ = [
    "DEFAULT_LOW_CAP",
    "DEFAULT_MED_CAP",
    "SignalTrust",
    "clamp_confidence",
    "default_cap_for",
]
