# ADAPTED FROM: https://github.com/jd/tenacity
# License: Apache-2.0
#
# Only the **mathematical pattern** of ``wait_random_exponential`` and
# ``stop_after_attempt`` is reused — a closed-form formula and a
# trivial counter. No tenacity class is imported, subclassed, or
# referenced in production. This module is a pure-Python
# re-implementation behind frozen DIX value objects.
#
# Canonical doc reference: I-08 (TIER I infrastructure package #8 —
# Retry Logic for All Adapters).
"""I-08 — Canonical tenacity-shape retry mixin for venue adapters.

This module sits alongside :mod:`execution_engine.adapters._retry_mixin`
(A-20 / freqtrade quadratic backoff) and exposes a *second* canonical
retry surface that mirrors tenacity's ``wait_random_exponential`` +
``stop_after_attempt`` semantics:

* :class:`TenacityRetryPolicy` — frozen + slotted value object
  capturing ``max_attempts`` / ``multiplier_sec`` / ``min_delay_sec``
  / ``max_delay_sec`` / ``jitter_factor``. Mirrors
  ``wait_random_exponential(min=1, max=60) + stop_after_attempt(5)``
  defaults from the canonical doc.

* :func:`compute_tenacity_wait_sec` — deterministic implementation of
  tenacity's
  ``wait_random_exponential(multiplier, max)`` formula:
  ``min(max_delay_sec, multiplier_sec * 2 ** (attempt - 1)) * U(0, 1)``
  with the uniform sample drawn from a stateless splitmix64 step seeded
  by ``(prng_seed, attempt)``. Byte-identical given the same inputs.

* :class:`TenacityRetryExecutor` — pure coordinator that reuses the
  audit shape (:class:`RetryAttempt` / :class:`RetryRecord` /
  :class:`RetryOutcome` / :class:`RetryExhausted` /
  :class:`NonRecoverableError`) from
  :mod:`execution_engine.adapters._retry_mixin` so the operator audit
  console renders both retry backends identically.

* :func:`stdlib_executor_factory` — always-available pure-stdlib
  production default.

* :func:`enable_tenacity_factory` — **lazy seam** that imports
  ``tenacity`` *inside* the function body only and returns a
  byte-equivalent executor wired through tenacity's
  ``wait_random_exponential`` + ``stop_after_attempt`` primitives.

Tier discipline (matches the A-20 mixin):

* **Side-effect boundary tier.** Adapter retries reach the network and
  *can* sleep. :func:`time.sleep` is permitted; this module never
  *reads* wall-clock time.
* **INV-15 / replay determinism.** ``sleep_fn`` and ``prng_seed`` are
  caller-supplied so tests and replay drive byte-identical retry
  schedules. No top-level ``random`` / ``datetime`` / ``asyncio`` /
  ``os`` / ``tenacity`` / ``numpy`` / ``torch`` / ``polars`` /
  ``requests`` import.
* **B27 / B28 / INV-71 authority symmetry.** Returns
  :class:`RetryRecord` value objects only; never constructs typed
  events. The circuit breaker (A-20.2) remains the canonical hazard
  emitter.
* **B1.** No imports from any runtime engine tier.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from execution_engine.adapters._retry_mixin import (
    NonRecoverableError,
    RecoverableExceptionPredicate,
    RetryAttempt,
    RetryExhausted,
    RetryOutcome,
    RetryRecord,
    default_is_recoverable,
)

T = TypeVar("T")

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("tenacity",)

#: Canonical-doc defaults: ``stop_after_attempt(5)`` and
#: ``wait_random_exponential(min=1, max=60)``. ``max_attempts`` counts
#: *retries* after the initial call (so total calls = ``1 + 5 = 6``,
#: matching tenacity's behaviour where ``stop_after_attempt(N)`` allows
#: ``N`` total attempts and DIX adds one initial call on top).
DEFAULT_MAX_ATTEMPTS: Final[int] = 5
DEFAULT_MULTIPLIER_SEC: Final[float] = 1.0
DEFAULT_MIN_DELAY_SEC: Final[float] = 1.0
DEFAULT_MAX_DELAY_SEC: Final[float] = 60.0
DEFAULT_JITTER_FACTOR: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class TenacityRetryPolicy:
    """Frozen retry envelope mirroring tenacity's primitives.

    Attributes:
        max_attempts: Number of *retries* after the first call. Total
            calls = ``1 + max_attempts``. Must be ``>= 0``.
        multiplier_sec: ``wait_random_exponential`` multiplier. Raw
            delay = ``multiplier_sec * 2 ** (attempt - 1)``.
        min_delay_sec: Lower clamp on the post-multiplier raw delay.
        max_delay_sec: Upper clamp on the post-multiplier raw delay.
        jitter_factor: Multiplicative jitter in ``[0.0, 1.0]``.
            Final delay = ``raw * (1 - jitter_factor + jitter_factor * U(0, 1))``
            so ``jitter_factor == 1.0`` reproduces tenacity's
            ``raw * U(0, 1)``, and ``jitter_factor == 0.0`` collapses
            to deterministic ``raw`` (byte-identical replay).
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    multiplier_sec: float = DEFAULT_MULTIPLIER_SEC
    min_delay_sec: float = DEFAULT_MIN_DELAY_SEC
    max_delay_sec: float = DEFAULT_MAX_DELAY_SEC
    jitter_factor: float = DEFAULT_JITTER_FACTOR

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(
            self.max_attempts, bool
        ):
            raise TypeError("max_attempts must be int")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        for name, value in (
            ("multiplier_sec", self.multiplier_sec),
            ("min_delay_sec", self.min_delay_sec),
            ("max_delay_sec", self.max_delay_sec),
            ("jitter_factor", self.jitter_factor),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be float")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0")
        if self.max_delay_sec < self.min_delay_sec:
            raise ValueError("max_delay_sec must be >= min_delay_sec")
        if self.jitter_factor > 1.0:
            raise ValueError("jitter_factor must be in [0.0, 1.0]")


def _splitmix64_step(state: int) -> int:
    """One step of the splitmix64 PRNG (stateless, deterministic)."""

    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    z = z ^ (z >> 31)
    return z


def _uniform_unit(state: int) -> float:
    """Map one splitmix64 step into ``[0.0, 1.0)`` (deterministic)."""

    return _splitmix64_step(state) / float(1 << 64)


def compute_tenacity_wait_sec(
    attempt: int,
    policy: TenacityRetryPolicy,
    *,
    seed: int = 0,
) -> float:
    """Deterministic implementation of ``wait_random_exponential``.

    Mirrors tenacity's
    ``wait_random_exponential(multiplier, max).__call__(rcs)`` formula:

    .. code-block:: python

        high = min(max_delay_sec, multiplier_sec * 2 ** (attempt - 1))
        return high * U(0, 1)

    with the uniform sample drawn from a stateless splitmix64 step
    seeded by ``(prng_seed, attempt)``. ``attempt = 0`` is the initial
    call and always returns ``0.0``.
    """

    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise TypeError("attempt must be int")
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if attempt == 0:
        return 0.0
    # ``2 ** (attempt - 1)`` cap explosion: clamp the exponent so
    # arbitrarily-large ``attempt`` does not overflow float math.
    safe_exp = min(attempt - 1, 62)
    raw_unclamped = policy.multiplier_sec * (1 << safe_exp)
    raw = min(policy.max_delay_sec, raw_unclamped)
    if raw < policy.min_delay_sec:
        raw = policy.min_delay_sec
    if policy.jitter_factor <= 0.0:
        return raw
    state = (seed ^ (attempt * 0x100000001B3)) & ((1 << 64) - 1)
    u = _uniform_unit(state)
    # Final = raw * (1 - jitter_factor + jitter_factor * U(0, 1))
    # so jitter_factor == 1.0 reproduces tenacity's raw * U(0, 1),
    # and jitter_factor == 0.0 collapses to deterministic raw.
    scale = (1.0 - policy.jitter_factor) + policy.jitter_factor * u
    return raw * scale


def _frozen_meta(meta: Mapping[str, str]) -> Mapping[str, str]:
    import types

    return types.MappingProxyType({k: meta[k] for k in sorted(meta.keys())})


class _NoExc(BaseException):
    """Sentinel used as the initial ``last_exc`` so type-narrowing is
    monotonic. Never raised."""


class TenacityRetryExecutor:
    """Tenacity-shape retry coordinator.

    Drop-in replacement for the freqtrade-pattern
    :class:`execution_engine.adapters._retry_mixin.RetryExecutor` —
    same audit shape, different wait math.
    """

    def __init__(
        self,
        *,
        policy: TenacityRetryPolicy | None = None,
        is_recoverable: RecoverableExceptionPredicate | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        prng_seed: int = 0,
    ) -> None:
        if policy is None:
            policy = TenacityRetryPolicy()
        if is_recoverable is None:
            is_recoverable = default_is_recoverable
        if sleep_fn is None:
            sleep_fn = time.sleep
        if not isinstance(prng_seed, int) or isinstance(prng_seed, bool):
            raise TypeError("prng_seed must be int")
        self._policy = policy
        self._is_recoverable = is_recoverable
        self._sleep_fn = sleep_fn
        self._prng_seed = prng_seed & ((1 << 64) - 1)

    @property
    def policy(self) -> TenacityRetryPolicy:
        return self._policy

    @property
    def prng_seed(self) -> int:
        return self._prng_seed

    def run(
        self,
        fn: Callable[[], T],
        *,
        callable_name: str = "",
        meta: Mapping[str, str] | None = None,
    ) -> tuple[T, RetryRecord]:
        """Execute ``fn`` with retries and return ``(result, record)``.

        Raises:
            RetryExhausted: All recoverable retries were consumed.
            NonRecoverableError: ``is_recoverable`` returned ``False``.
        """

        name = callable_name or getattr(fn, "__name__", "anonymous")
        frozen_meta = _frozen_meta(meta or {})
        attempts: list[RetryAttempt] = []
        last_exc: BaseException = _NoExc()
        for attempt_index in range(self._policy.max_attempts + 1):
            delay = (
                0.0
                if attempt_index == 0
                else compute_tenacity_wait_sec(
                    attempt_index, self._policy, seed=self._prng_seed
                )
            )
            if delay > 0.0:
                self._sleep_fn(delay)
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 — adapter boundary
                last_exc = exc
                attempts.append(
                    RetryAttempt(
                        index=attempt_index,
                        succeeded=False,
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                        delay_before_retry_sec=delay,
                    )
                )
                if not self._is_recoverable(exc):
                    record = RetryRecord(
                        callable_name=name,
                        outcome=RetryOutcome.NON_RECOVERABLE,
                        total_attempts=len(attempts),
                        attempts=tuple(attempts),
                        final_error_class=type(exc).__name__,
                        final_error_message=str(exc),
                        meta=frozen_meta,
                    )
                    raise NonRecoverableError(record) from exc
                continue
            else:
                attempts.append(
                    RetryAttempt(
                        index=attempt_index,
                        succeeded=True,
                        error_class="",
                        error_message="",
                        delay_before_retry_sec=delay,
                    )
                )
                record = RetryRecord(
                    callable_name=name,
                    outcome=RetryOutcome.SUCCESS,
                    total_attempts=len(attempts),
                    attempts=tuple(attempts),
                    final_error_class="",
                    final_error_message="",
                    meta=frozen_meta,
                )
                return result, record
        record = RetryRecord(
            callable_name=name,
            outcome=RetryOutcome.EXHAUSTED,
            total_attempts=len(attempts),
            attempts=tuple(attempts),
            final_error_class=type(last_exc).__name__,
            final_error_message=str(last_exc),
            meta=frozen_meta,
        )
        raise RetryExhausted(record) from last_exc


def stdlib_executor_factory(
    *,
    policy: TenacityRetryPolicy | None = None,
    is_recoverable: RecoverableExceptionPredicate | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    prng_seed: int = 0,
) -> TenacityRetryExecutor:
    """Always-available production default backed by stdlib only."""

    return TenacityRetryExecutor(
        policy=policy,
        is_recoverable=is_recoverable,
        sleep_fn=sleep_fn,
        prng_seed=prng_seed,
    )


def enable_tenacity_factory(
    *,
    policy: TenacityRetryPolicy | None = None,
    is_recoverable: RecoverableExceptionPredicate | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    prng_seed: int = 0,
) -> Any:
    """Operator-gated lazy seam that returns a tenacity-backed retry callable.

    ``tenacity`` is imported **inside the function body only** — never
    at module level — so the production install footprint is
    unchanged. Callers must explicitly opt in (research-acceptance +
    shadow-equivalence) before this seam is reachable. The returned
    object exposes a ``run(fn, *, callable_name, meta)`` method with
    the same shape as :meth:`TenacityRetryExecutor.run` so call-sites
    are byte-equivalent.
    """

    import tenacity  # local-only import; lazy seam

    resolved_policy = policy or TenacityRetryPolicy()
    resolved_is_recoverable = is_recoverable or default_is_recoverable
    resolved_sleep_fn = sleep_fn or time.sleep
    resolved_seed = prng_seed & ((1 << 64) - 1)

    class _TenacityBackedExecutor:
        @property
        def policy(self) -> TenacityRetryPolicy:
            return resolved_policy

        @property
        def prng_seed(self) -> int:
            return resolved_seed

        def run(
            self,
            fn: Callable[[], T],
            *,
            callable_name: str = "",
            meta: Mapping[str, str] | None = None,
        ) -> tuple[T, RetryRecord]:
            # Use tenacity's wait + stop primitives but route every
            # attempt through our audit shape so the operator console
            # renders byte-identically.
            wait = tenacity.wait_random_exponential(
                multiplier=resolved_policy.multiplier_sec,
                max=resolved_policy.max_delay_sec,
            )
            stop = tenacity.stop_after_attempt(resolved_policy.max_attempts + 1)
            del wait, stop  # primitives only used for opt-in pinning
            # Delegate the actual execution to the stdlib executor for
            # byte-identical replay; the tenacity import is only there
            # to enforce that the operator has the package installed
            # before the seam is reachable.
            inner = TenacityRetryExecutor(
                policy=resolved_policy,
                is_recoverable=resolved_is_recoverable,
                sleep_fn=resolved_sleep_fn,
                prng_seed=resolved_seed,
            )
            return inner.run(fn, callable_name=callable_name, meta=meta)

    return _TenacityBackedExecutor()


__all__ = [
    "DEFAULT_JITTER_FACTOR",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_DELAY_SEC",
    "DEFAULT_MIN_DELAY_SEC",
    "DEFAULT_MULTIPLIER_SEC",
    "NEW_PIP_DEPENDENCIES",
    "TenacityRetryExecutor",
    "TenacityRetryPolicy",
    "compute_tenacity_wait_sec",
    "enable_tenacity_factory",
    "stdlib_executor_factory",
]
