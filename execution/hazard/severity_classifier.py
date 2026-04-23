"""
execution/hazard/severity_classifier.py
DIX VISION v42.2 — Hazard Severity Classifier

Four pure functions consulted by governance's hazard consumer to
decide what to do with each `SYSTEM_HAZARD_EVENT`:

* :func:`should_halt_trading`
* :func:`should_enter_safe_mode`
* :func:`classify_severity`
* :func:`classify_response`

Polyglot dual-backend
---------------------
When the Rust extension ``dixvision_py_system`` is importable **and**
exports the full ``hazard_*`` surface, this module proxies to Rust
(canonical; identical semantics, lower per-call cost on the hazard
consumer loop). Otherwise it falls back to the pure-Python reference
below. Both backends satisfy the invariants exercised by
``tests/test_hazard_severity_parity.py``.

The classifier accepts strings for ``hazard_type`` and ``severity``
so the same call sites work whether the caller is holding a
``HazardEvent`` (Python dataclass), a protobuf message, or a raw
ledger row. Callers that want to reject unknown variants at ingest
time should use :func:`is_known_hazard_type` /
:func:`is_known_severity` (not auto-enforced — the classifier
functions themselves are tolerant so a new variant on the producer
side never crashes the consumer).
"""
from __future__ import annotations

from typing import Union

from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType

# ---------------------------------------------------------------- Rust backend
try:
    import dixvision_py_system as _rs  # type: ignore[import-not-found]

    _HAVE_RUST = all(
        hasattr(_rs, fn)
        for fn in (
            "hazard_should_halt_trading",
            "hazard_should_enter_safe_mode",
            "hazard_classify_severity",
            "hazard_classify_response",
        )
    )
except ImportError:  # pragma: no cover - backend selection branch
    _rs = None
    _HAVE_RUST = False


# -------------------------------------------------------------- normalise args

_HazardTypeArg = Union[HazardType, str]
_SeverityArg = Union[HazardSeverity, str, None]


def _type_str(hazard_type: _HazardTypeArg) -> str:
    """Return the canonical string form of a hazard type.

    Accepts either a :class:`HazardType` enum member or the raw
    string token. ``str``-backed enums produce the token via their
    ``.value`` attribute; plain strings pass through unchanged.
    """
    if isinstance(hazard_type, HazardType):
        return hazard_type.value
    return str(hazard_type)


def _severity_str(severity: _SeverityArg) -> str:
    if severity is None:
        return ""
    if isinstance(severity, HazardSeverity):
        return severity.value
    return str(severity)


# ----------------------------------------------------------- public surface

def should_halt_trading(event_or_type: Union[HazardEvent, _HazardTypeArg],
                        severity: _SeverityArg = None) -> bool:
    """Return True if this hazard should trigger trading halt.

    Two calling conventions, matched to how the module has been used
    historically in the code base:

    * ``should_halt_trading(event)`` — inspects ``event.hazard_type``
      and ``event.severity``.
    * ``should_halt_trading(hazard_type, severity)`` — direct string
      or enum arguments (the form the Rust FFI uses).
    """
    ht, sev = _unpack(event_or_type, severity)
    if _HAVE_RUST:
        return bool(_rs.hazard_should_halt_trading(ht, sev))
    return _py_should_halt_trading(ht, sev)


def should_enter_safe_mode(event_or_type: Union[HazardEvent, _HazardTypeArg],
                           severity: _SeverityArg = None) -> bool:
    """Return True if this hazard should trigger safe mode."""
    ht, sev = _unpack(event_or_type, severity)
    if _HAVE_RUST:
        return bool(_rs.hazard_should_enter_safe_mode(ht, sev))
    return _py_should_enter_safe_mode(ht, sev)


def classify_severity(event_or_type: Union[HazardEvent, _HazardTypeArg],
                      severity: _SeverityArg = None) -> HazardSeverity:
    """Return the effective severity for a hazard event.

    Some hazard types (data-corruption, ledger inconsistency) are
    promoted to ``CRITICAL`` regardless of the severity the emitter
    assigned; the rest are returned as-is. Always returns a
    :class:`HazardSeverity` enum for the classifier-consumer's type
    comfort, even when the caller passed a plain string.
    """
    ht, sev = _unpack(event_or_type, severity)
    if _HAVE_RUST:
        out = _rs.hazard_classify_severity(ht, sev)
    else:
        out = _py_classify_severity(ht, sev)
    # ``HazardSeverity`` is a str-backed Enum; constructing from the
    # token is cheap and preserves the (type, str) duck-type contract.
    try:
        return HazardSeverity(out)
    except ValueError:
        # Unknown severity tokens fall through. Do not raise — the
        # classifier's contract is "never crash on a novel variant".
        return HazardSeverity.LOW  # conservative default


def classify_response(event_or_type: Union[HazardEvent, _HazardTypeArg]) -> str:
    """Return recommended governance action for this hazard.

    Unknown types fall through to ``"OBSERVE"`` — the conservative
    choice, since we never halt trading on a type we do not
    understand.
    """
    if isinstance(event_or_type, HazardEvent):
        ht = event_or_type.hazard_type.value
    else:
        ht = _type_str(event_or_type)
    if _HAVE_RUST:
        return str(_rs.hazard_classify_response(ht))
    return _py_classify_response(ht)


def is_known_hazard_type(hazard_type: _HazardTypeArg) -> bool:
    """Return True if ``hazard_type`` is a documented variant.

    Pure-Python check (no FFI round-trip needed) — the set is small
    and static.
    """
    try:
        HazardType(_type_str(hazard_type))
        return True
    except ValueError:
        return False


def is_known_severity(severity: _SeverityArg) -> bool:
    """Return True if ``severity`` is a documented variant."""
    try:
        HazardSeverity(_severity_str(severity))
        return True
    except ValueError:
        return False


# --------------------------------------------------------------- plumbing

def _unpack(event_or_type: Union[HazardEvent, _HazardTypeArg],
            severity: _SeverityArg) -> tuple[str, str]:
    if isinstance(event_or_type, HazardEvent):
        return event_or_type.hazard_type.value, event_or_type.severity.value
    return _type_str(event_or_type), _severity_str(severity)


# ------------------------------------------------------ pure-Python reference

_CRITICAL_TYPES = frozenset({
    "DATA_CORRUPTION_SUSPECTED",
    "LEDGER_INCONSISTENCY",
})

_HALT_ON_TYPE = frozenset({
    "DATA_CORRUPTION_SUSPECTED",
    "LEDGER_INCONSISTENCY",
    "API_CONNECTIVITY_FAILURE",
})

_SAFE_MODE_ON_TYPE = frozenset({
    "FEED_SILENCE",
    "EXCHANGE_TIMEOUT",
})


def _py_should_halt_trading(hazard_type: str, severity: str) -> bool:
    return severity == "CRITICAL" or hazard_type in _HALT_ON_TYPE


def _py_should_enter_safe_mode(hazard_type: str, severity: str) -> bool:
    return severity in ("HIGH", "CRITICAL") or hazard_type in _SAFE_MODE_ON_TYPE


def _py_classify_severity(hazard_type: str, severity: str) -> str:
    if hazard_type in _CRITICAL_TYPES:
        return "CRITICAL"
    return severity


def _py_classify_response(hazard_type: str) -> str:
    if hazard_type == "EXCHANGE_TIMEOUT":
        return "CANCEL_ALL_OPEN_ORDERS"
    if hazard_type == "FEED_SILENCE":
        return "PAUSE_NEW_ORDERS"
    if hazard_type == "EXECUTION_LATENCY_SPIKE":
        return "REDUCE_EXPOSURE"
    if hazard_type in ("DATA_CORRUPTION_SUSPECTED", "API_CONNECTIVITY_FAILURE"):
        return "HALT_TRADING"
    return "OBSERVE"
