"""Hazard throttle policy + observer (BEHAVIOR-P3).

Translates observed :class:`HazardEvent` records into a deterministic
:class:`ThrottleDecision` that tightens the next :class:`RiskSnapshot`.

Design notes
------------

* **Pure FSM (INV-15 / INV-64).** :func:`compute_throttle` is a pure
  function of (observations, ``now_ns``, config). The optional
  :class:`HazardObserver` adds bounded window state so a runtime
  caller can ``observe()`` events as they arrive and read the
  current throttle on every tick. No clock reads anywhere — the
  caller supplies ``now_ns`` (CONST-04 in the constraint engine).
* **Monotonically restrictive (SAFE-67).** Every aggregation rule
  *tightens*: ``min`` of qty multipliers, ``max`` of confidence
  floors, ``or`` of block flags. A hazard can never *relax* an
  active throttle.
* **Composes with emergency LOCK (SAFE-68).** CRITICAL/HIGH still
  routes through Governance's emergency-lock path
  (``EventClassifier._classify_hazard`` → ``StateTransitionManager``).
  This module additionally returns ``block=True`` for those
  severities so the throttle layer *also* halts the hot path
  immediately, without waiting for the Governance Mode FSM round
  trip.

The taxonomy here is intentionally minimal: a default rule per
:class:`HazardSeverity` plus optional per-code overrides. Codes the
operator hasn't classified explicitly fall back to the severity
default — there is no "unknown hazard" silent pass-through.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from core.contracts.events import HazardEvent, HazardSeverity

HAZARD_THROTTLE_VERSION = "v1.0.0"


def _check_unit_interval(name: str, value: float) -> None:
    if not isinstance(value, int | float):  # type: ignore[unreachable]
        raise TypeError(f"{name} must be a real number")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HazardSeverityRule:
    """Default throttle action for a :class:`HazardSeverity`.

    Attributes:
        qty_multiplier: Multiplier applied to ``max_position_qty`` and
            every entry of ``symbol_caps``. ``1.0`` = no throttle,
            ``0.0`` = no new exposure. Must be ``∈ [0.0, 1.0]``.
        confidence_floor: Floor placed on ``max_signal_confidence``
            (only raised, never lowered). Must be ``∈ [0.0, 1.0]``.
        block: When true, ``snapshot.halted`` is forced true.
        active_window_ns: How long an observation of this severity
            keeps influencing the throttle. Older observations are
            ignored (decay). Must be ``> 0``.
    """

    qty_multiplier: float
    confidence_floor: float
    block: bool
    active_window_ns: int

    def __post_init__(self) -> None:
        _check_unit_interval("qty_multiplier", self.qty_multiplier)
        _check_unit_interval("confidence_floor", self.confidence_floor)
        if not isinstance(self.block, bool):  # type: ignore[unreachable]
            raise TypeError("block must be bool")
        if self.active_window_ns <= 0:
            raise ValueError("active_window_ns must be > 0")


@dataclass(frozen=True, slots=True)
class HazardCodeOverride:
    """Per-code override of the severity default.

    Each unset field falls back to the matched
    :class:`HazardSeverityRule`.
    """

    code: str
    qty_multiplier: float | None = None
    confidence_floor: float | None = None
    block: bool | None = None
    active_window_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must not be empty")
        if self.qty_multiplier is not None:
            _check_unit_interval("qty_multiplier", self.qty_multiplier)
        if self.confidence_floor is not None:
            _check_unit_interval("confidence_floor", self.confidence_floor)
        if self.block is not None and not isinstance(self.block, bool):  # type: ignore[unreachable]
            raise TypeError("block must be bool")
        if self.active_window_ns is not None and self.active_window_ns <= 0:
            raise ValueError("active_window_ns must be > 0")


_DEFAULT_RULES: dict[HazardSeverity, HazardSeverityRule] = {
    HazardSeverity.INFO: HazardSeverityRule(
        qty_multiplier=1.0,
        confidence_floor=0.0,
        block=False,
        active_window_ns=60_000_000_000,  # 60 s
    ),
    HazardSeverity.LOW: HazardSeverityRule(
        qty_multiplier=0.75,
        confidence_floor=0.0,
        block=False,
        active_window_ns=120_000_000_000,  # 2 min
    ),
    HazardSeverity.MEDIUM: HazardSeverityRule(
        qty_multiplier=0.5,
        confidence_floor=0.6,
        block=False,
        active_window_ns=300_000_000_000,  # 5 min
    ),
    HazardSeverity.HIGH: HazardSeverityRule(
        qty_multiplier=0.0,
        confidence_floor=1.0,
        block=True,
        active_window_ns=600_000_000_000,  # 10 min
    ),
    HazardSeverity.CRITICAL: HazardSeverityRule(
        qty_multiplier=0.0,
        confidence_floor=1.0,
        block=True,
        active_window_ns=600_000_000_000,  # 10 min
    ),
}


@dataclass(frozen=True, slots=True)
class HazardThrottleConfig:
    """Throttle policy table: one severity rule per :class:`HazardSeverity`,
    optional per-code overrides.

    Construct with ``HazardThrottleConfig.default()`` for the canonical
    defaults wired into the runtime (LOW=0.75x / MEDIUM=0.5x@0.6 floor /
    HIGH+CRITICAL=block).
    """

    severity_rules: tuple[tuple[HazardSeverity, HazardSeverityRule], ...]
    code_overrides: tuple[HazardCodeOverride, ...] = ()
    version: str = HAZARD_THROTTLE_VERSION

    def __post_init__(self) -> None:
        seen: set[HazardSeverity] = set()
        for severity, _rule in self.severity_rules:
            if severity in seen:
                raise ValueError(
                    f"duplicate severity rule for {severity.name}"
                )
            seen.add(severity)
        for required in HazardSeverity:
            if required not in seen:
                raise ValueError(
                    f"missing severity rule for {required.name}"
                )
        codes: set[str] = set()
        for override in self.code_overrides:
            if override.code in codes:
                raise ValueError(
                    f"duplicate code override for {override.code}"
                )
            codes.add(override.code)

    def rule_for(self, severity: HazardSeverity) -> HazardSeverityRule:
        for s, rule in self.severity_rules:
            if s is severity:
                return rule
        # Cannot occur — __post_init__ enforces full coverage.
        raise KeyError(severity.name)  # pragma: no cover

    def override_for(self, code: str) -> HazardCodeOverride | None:
        for override in self.code_overrides:
            if override.code == code:
                return override
        return None

    @classmethod
    def default(cls) -> HazardThrottleConfig:
        return cls(
            severity_rules=tuple(
                (s, _DEFAULT_RULES[s]) for s in HazardSeverity
            ),
        )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HazardObservation:
    """The window-storable subset of a :class:`HazardEvent`.

    A pure record — no references to mutable state. Constructed via
    :meth:`from_event` from the canonical bus event, or directly by
    tests.
    """

    ts_ns: int
    code: str
    severity: HazardSeverity
    source: str

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if not self.code:
            raise ValueError("code must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")

    @classmethod
    def from_event(cls, event: HazardEvent) -> HazardObservation:
        return cls(
            ts_ns=event.ts_ns,
            code=event.code,
            severity=event.severity,
            source=event.source,
        )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThrottleDecision:
    """The deterministic result of one :func:`compute_throttle` call.

    Attributes:
        block: ``True`` if any active hazard's effective rule sets
            ``block``. Forces ``snapshot.halted = True``.
        qty_multiplier: ``min`` over all active rules. Always
            ``∈ [0.0, 1.0]``.
        confidence_floor: ``max`` over all active rules. Always
            ``∈ [0.0, 1.0]``.
        contributing_codes: Distinct hazard codes whose observations
            were inside the active window, sorted. Empty when no
            hazard is active.
        version: Policy version that produced the decision (matches
            ``config.version``).
    """

    block: bool
    qty_multiplier: float
    confidence_floor: float
    contributing_codes: tuple[str, ...]
    version: str

    @property
    def is_throttled(self) -> bool:
        return (
            self.block
            or self.qty_multiplier < 1.0
            or self.confidence_floor > 0.0
        )


_NEUTRAL_DECISION_TEMPLATE = ThrottleDecision(
    block=False,
    qty_multiplier=1.0,
    confidence_floor=0.0,
    contributing_codes=(),
    version=HAZARD_THROTTLE_VERSION,
)


def _neutral_decision(version: str) -> ThrottleDecision:
    if version == HAZARD_THROTTLE_VERSION:
        return _NEUTRAL_DECISION_TEMPLATE
    return ThrottleDecision(
        block=False,
        qty_multiplier=1.0,
        confidence_floor=0.0,
        contributing_codes=(),
        version=version,
    )


# ---------------------------------------------------------------------------
# Pure aggregator
# ---------------------------------------------------------------------------


def compute_throttle(
    *,
    observations: Sequence[HazardObservation],
    now_ns: int,
    config: HazardThrottleConfig,
) -> ThrottleDecision:
    """Aggregate active hazard observations into a deterministic
    throttle decision.

    Decay rule (SAFE-67): an observation is *active* iff
    ``now_ns - obs.ts_ns < window_ns_for(obs)``. Future-dated
    observations (``ts_ns > now_ns``) are also considered active —
    they are by definition fresh and tightening the throttle on
    fresh data is conservative.

    Aggregation:
        block = any(rule.block for active)
        qty_multiplier = min(rule.qty_multiplier for active, default=1.0)
        confidence_floor = max(rule.confidence_floor for active, default=0.0)
        contributing_codes = sorted({obs.code for active})
    """
    if now_ns < 0:
        raise ValueError("now_ns must be >= 0")

    block = False
    qty_multiplier = 1.0
    confidence_floor = 0.0
    codes: set[str] = set()

    for obs in observations:
        rule = _effective_rule(obs.code, obs.severity, config)
        age = now_ns - obs.ts_ns
        if age >= rule.active_window_ns:
            continue  # decayed
        if rule.block:
            block = True
        if rule.qty_multiplier < qty_multiplier:
            qty_multiplier = rule.qty_multiplier
        if rule.confidence_floor > confidence_floor:
            confidence_floor = rule.confidence_floor
        codes.add(obs.code)

    if not codes and not block:
        return _neutral_decision(config.version)

    return ThrottleDecision(
        block=block,
        qty_multiplier=qty_multiplier,
        confidence_floor=confidence_floor,
        contributing_codes=tuple(sorted(codes)),
        version=config.version,
    )


def _effective_rule(
    code: str,
    severity: HazardSeverity,
    config: HazardThrottleConfig,
) -> HazardSeverityRule:
    base = config.rule_for(severity)
    override = config.override_for(code)
    if override is None:
        return base
    return HazardSeverityRule(
        qty_multiplier=(
            override.qty_multiplier
            if override.qty_multiplier is not None
            else base.qty_multiplier
        ),
        confidence_floor=(
            override.confidence_floor
            if override.confidence_floor is not None
            else base.confidence_floor
        ),
        block=override.block if override.block is not None else base.block,
        active_window_ns=(
            override.active_window_ns
            if override.active_window_ns is not None
            else base.active_window_ns
        ),
    )


# ---------------------------------------------------------------------------
# Stateful observer (bounded ring buffer)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HazardObserver:
    """Bounded ring-buffer that accumulates hazard observations and
    answers ``current_throttle(now_ns)``.

    The observer is *stateful* (the runtime calls ``observe()`` from
    the event bus) but still INV-15 deterministic: same sequence of
    ``observe`` / ``current_throttle`` calls → identical results.
    No clock reads — every ``current_throttle`` call passes
    ``now_ns`` explicitly.

    The buffer is bounded (default ``1024`` slots). Once full, the
    oldest observation is evicted on insert. This is a conservative
    fallback: in the steady state observations decay out of the
    active window long before the buffer fills.
    """

    config: HazardThrottleConfig = field(
        default_factory=HazardThrottleConfig.default
    )
    capacity: int = 1024
    _buffer: deque[HazardObservation] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buffer = deque(maxlen=self.capacity)

    def observe(self, obs: HazardObservation | HazardEvent) -> None:
        if isinstance(obs, HazardEvent):
            obs = HazardObservation.from_event(obs)
        self._buffer.append(obs)

    def observe_many(
        self, items: Iterable[HazardObservation | HazardEvent]
    ) -> None:
        for item in items:
            self.observe(item)

    def current_throttle(self, *, now_ns: int) -> ThrottleDecision:
        return compute_throttle(
            observations=tuple(self._buffer),
            now_ns=now_ns,
            config=self.config,
        )

    def active_observations(
        self, *, now_ns: int
    ) -> tuple[HazardObservation, ...]:
        """Return observations still inside their active window.

        Useful for telemetry / decision-trace callers (BEHAVIOR-P4).
        """
        out: list[HazardObservation] = []
        for obs in self._buffer:
            rule = _effective_rule(obs.code, obs.severity, self.config)
            if now_ns - obs.ts_ns < rule.active_window_ns:
                out.append(obs)
        return tuple(out)

    def __len__(self) -> int:
        return len(self._buffer)


__all__ = [
    "HAZARD_THROTTLE_VERSION",
    "HazardCodeOverride",
    "HazardObservation",
    "HazardObserver",
    "HazardSeverityRule",
    "HazardThrottleConfig",
    "ThrottleDecision",
    "compute_throttle",
]
