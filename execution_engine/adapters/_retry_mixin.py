"""A-20.1 — Deterministic retry mixin for venue adapters.

# ADAPTED FROM: freqtrade/exchange/common.py (mathematical patterns)
# GPL-3.0 mitigation: only the **mathematical pattern** is reused
# (quadratic backoff ``(attempt ** 2) + 1`` from
# ``calculate_backoff(retrycount, max_retries)``). No freqtrade class is
# imported, subclassed, or referenced in production. This module is a
# pure-Python re-implementation behind frozen DIX value objects.

The DIX execution engine wraps every venue call through this mixin so
all adapters share one auditable, deterministic retry surface:

* :class:`RetryPolicy` — frozen + slotted value object capturing
  ``max_attempts`` / ``base_delay_sec`` / ``max_delay_sec`` /
  ``jitter_factor``. ``max_attempts`` follows freqtrade's
  ``API_RETRY_COUNT = 4`` (total ``1 + max_attempts`` calls).

* :func:`compute_backoff_sec` — deterministic quadratic backoff
  ``(attempt ** 2) + 1`` (freqtrade math), optionally clamped by
  ``max_delay_sec`` and perturbed by deterministic seeded jitter
  (splitmix64) so concurrent adapters do not retry in lockstep.

* :class:`RetryAttempt` — frozen + slotted record of one attempt
  (index, succeeded, error_class, delay_before_retry).

* :class:`RetryRecord` — frozen + slotted ledger of every attempt for a
  single call (full audit trail for the operator console).

* :class:`RetryOutcome` — final outcome enum (SUCCESS / EXHAUSTED /
  NON_RECOVERABLE).

* :class:`RetryExecutor` — pure coordinator. Takes a callable, a
  policy, an "is recoverable?" predicate, a ``sleep_fn`` seam, and a
  ``prng_seed``. Runs the call, retries on recoverable failures, and
  returns the result + :class:`RetryRecord`. **Never** constructs a
  :class:`HazardEvent` — that is the responsibility of
  :mod:`execution_engine.protections.circuit_breaker` (A-20.2) which
  observes :class:`RetryRecord` and emits the typed event through the
  proper authority channel (B27 / B28 / INV-71).

* :class:`RetryMixin` — base for adapter classes that want a
  ``_retry(fn)`` helper without re-wiring the policy on every call.

Tier discipline (per :mod:`execution_engine.adapters.base`):

* **Side-effect boundary tier.** Adapter retries reach the network and
  *can* sleep. :func:`time.sleep` is permitted (the B-CLOCK chokepoint
  only bans wall-clock *reads* like :func:`time.time`).
* **INV-15 / replay determinism.** ``sleep_fn`` and ``prng_seed`` are
  caller-supplied so tests and replay drive byte-identical retry
  schedules. No random / datetime / asyncio / os import.
* **B27 / B28 / INV-71 authority symmetry.** This module returns
  :class:`RetryRecord` value objects only. It does **not** construct
  :class:`HazardEvent`, :class:`SignalEvent`,
  :class:`GovernanceDecision`, or any typed bus event. The circuit
  breaker (A-20.2) is the canonical hazard emitter.
* **No engine cross-imports.** Pinned by AST tests.
"""

from __future__ import annotations

import dataclasses
import enum
import time
import types
from collections.abc import Callable, Mapping
from typing import Final, Protocol, TypeVar, runtime_checkable

# ---------------------------------------------------------------------------
# Sentinels (mirror freqtrade's API_RETRY_COUNT and friends)
# ---------------------------------------------------------------------------

#: Default total retries after the first attempt. ``1 + 4 = 5`` calls.
DEFAULT_MAX_ATTEMPTS: Final[int] = 4

#: Minimum delay slot in seconds. Freqtrade's formula bottoms out at
#: ``1`` second (``0 ** 2 + 1 = 1`` for the first retry).
DEFAULT_BASE_DELAY_SEC: Final[float] = 1.0

#: Maximum delay slot in seconds. Freqtrade's
#: ``calculate_backoff(0, 4) = 17`` — we cap at 60 by default so a long
#: outage does not block the operator for many minutes.
DEFAULT_MAX_DELAY_SEC: Final[float] = 60.0

#: Jitter factor in ``[0.0, 1.0]``. Final delay is uniformly perturbed
#: by ``(1 + U(-jitter, +jitter)) * raw_delay``. ``0.0`` disables jitter
#: entirely for byte-identical replay.
DEFAULT_JITTER_FACTOR: Final[float] = 0.0

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Frozen retry envelope.

    Attributes:
        max_attempts: Number of *retries* after the first call. Total
            calls = ``1 + max_attempts``. Must be ``>= 0``.
        base_delay_sec: Minimum delay applied to any retry. Must be
            ``>= 0.0``.
        max_delay_sec: Upper bound for any single retry. Must be
            ``>= base_delay_sec``.
        jitter_factor: Symmetric jitter in ``[0.0, 1.0]``. ``0.0``
            yields byte-identical replay.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_sec: float = DEFAULT_BASE_DELAY_SEC
    max_delay_sec: float = DEFAULT_MAX_DELAY_SEC
    jitter_factor: float = DEFAULT_JITTER_FACTOR

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise TypeError("max_attempts must be int")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if not isinstance(self.base_delay_sec, (int, float)) or isinstance(
            self.base_delay_sec, bool
        ):
            raise TypeError("base_delay_sec must be float")
        if self.base_delay_sec < 0.0:
            raise ValueError("base_delay_sec must be >= 0.0")
        if not isinstance(self.max_delay_sec, (int, float)) or isinstance(self.max_delay_sec, bool):
            raise TypeError("max_delay_sec must be float")
        if self.max_delay_sec < self.base_delay_sec:
            raise ValueError("max_delay_sec must be >= base_delay_sec")
        if not isinstance(self.jitter_factor, (int, float)) or isinstance(self.jitter_factor, bool):
            raise TypeError("jitter_factor must be float")
        if not (0.0 <= self.jitter_factor <= 1.0):
            raise ValueError("jitter_factor must be in [0.0, 1.0]")


class RetryOutcome(enum.Enum):
    """Terminal outcome of a :class:`RetryExecutor` run."""

    SUCCESS = "SUCCESS"
    EXHAUSTED = "EXHAUSTED"
    NON_RECOVERABLE = "NON_RECOVERABLE"


@dataclasses.dataclass(frozen=True, slots=True)
class RetryAttempt:
    """One attempt's outcome — for the audit log."""

    index: int
    succeeded: bool
    error_class: str
    error_message: str
    delay_before_retry_sec: float


@dataclasses.dataclass(frozen=True, slots=True)
class RetryRecord:
    """Full audit trail for a single :meth:`RetryExecutor.run` call."""

    callable_name: str
    outcome: RetryOutcome
    total_attempts: int
    attempts: tuple[RetryAttempt, ...]
    final_error_class: str
    final_error_message: str
    meta: Mapping[str, str] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))


# ---------------------------------------------------------------------------
# Recoverability predicate
# ---------------------------------------------------------------------------


@runtime_checkable
class RecoverableExceptionPredicate(Protocol):
    """Caller-supplied classifier.

    Returns ``True`` if the exception is a recoverable venue error
    (timeout, 5xx, rate-limit) and the executor should sleep + retry.
    Returns ``False`` if the exception is permanently broken (invalid
    API key, insufficient balance, bad symbol, ...).
    """

    def __call__(self, exc: BaseException) -> bool: ...


def default_is_recoverable(exc: BaseException) -> bool:
    """Conservative default.

    By default **all** exceptions are treated as non-recoverable so the
    executor escalates immediately. Callers (each adapter) wire a
    venue-specific predicate that explicitly enumerates the recoverable
    error classes.
    """

    del exc  # unused; explicit no-op
    return False


# ---------------------------------------------------------------------------
# Backoff math (the freqtrade formula)
# ---------------------------------------------------------------------------


def _splitmix64_step(state: int) -> int:
    """One step of the splitmix64 PRNG (stateless, deterministic)."""

    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    z = z ^ (z >> 31)
    return z


def _uniform_unit(state: int) -> tuple[float, int]:
    """Map one splitmix64 step into ``[0.0, 1.0)`` (deterministic)."""

    nxt = _splitmix64_step(state)
    return (nxt / float(1 << 64), nxt)


def compute_backoff_sec(
    attempt: int,
    policy: RetryPolicy,
    *,
    seed: int = 0,
) -> float:
    """Deterministic quadratic backoff.

    Mirrors freqtrade's
    ``calculate_backoff(retrycount, max_retries) = (max_retries - retrycount) ** 2 + 1``
    expressed over ``attempt = 1 .. max_attempts``. ``attempt = 0`` is
    the initial call and carries no delay.

    The raw delay is clamped to
    ``[policy.base_delay_sec, policy.max_delay_sec]`` and optionally
    perturbed by symmetric jitter using a stateless splitmix64 step
    seeded by ``(seed, attempt)`` — byte-identical given the same
    inputs.
    """

    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise TypeError("attempt must be int")
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if attempt == 0:
        return 0.0
    raw = float(attempt * attempt + 1)
    if raw < policy.base_delay_sec:
        raw = policy.base_delay_sec
    if raw > policy.max_delay_sec:
        raw = policy.max_delay_sec
    if policy.jitter_factor <= 0.0:
        return raw
    # Stateless jitter: combine seed + attempt deterministically.
    state = (seed ^ (attempt * 0x100000001B3)) & ((1 << 64) - 1)
    u, _ = _uniform_unit(state)
    perturbation = (2.0 * u - 1.0) * policy.jitter_factor
    jittered = raw * (1.0 + perturbation)
    if jittered < 0.0:
        return 0.0
    return jittered


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


T = TypeVar("T")


class RetryExecutor:
    """Deterministic retry coordinator.

    The executor owns the policy, the recoverability predicate, the
    ``sleep_fn`` seam (default :func:`time.sleep`), and the PRNG seed.
    It exposes :meth:`run` which executes a callable and returns
    ``(result, RetryRecord)`` on success or raises the final exception
    along with a populated :class:`RetryRecord` accessible via
    :attr:`RetryExhausted.record` / :attr:`NonRecoverableError.record`.
    """

    def __init__(
        self,
        *,
        policy: RetryPolicy | None = None,
        is_recoverable: RecoverableExceptionPredicate | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        prng_seed: int = 0,
    ) -> None:
        if policy is None:
            policy = RetryPolicy()
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
    def policy(self) -> RetryPolicy:
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
                else compute_backoff_sec(attempt_index, self._policy, seed=self._prng_seed)
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


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class RetryMixin:
    """Mixin for adapter classes that want a shared retry seam.

    Subclasses construct one :class:`RetryExecutor` at ``__init__`` time
    and route every venue call through :meth:`_retry_call`. The mixin
    deliberately does **not** override ``submit``; each adapter still
    owns its event projection.
    """

    def __init__(
        self,
        *,
        retry_executor: RetryExecutor | None = None,
        **kwargs: object,
    ) -> None:
        # ``kwargs`` forwarded so this can sit alongside ``LiveAdapterBase``
        # in an MRO without conflicting __init__ contracts.
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._retry_executor = retry_executor or RetryExecutor()

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_executor.policy

    def _retry_call(
        self,
        fn: Callable[[], T],
        *,
        callable_name: str = "",
        meta: Mapping[str, str] | None = None,
    ) -> tuple[T, RetryRecord]:
        return self._retry_executor.run(fn, callable_name=callable_name, meta=meta)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetryExhausted(RuntimeError):
    """All recoverable retries were consumed.

    Carries the populated :class:`RetryRecord` so callers (notably the
    circuit breaker in A-20.2) can decide whether to escalate to a
    :class:`HazardEvent`.
    """

    def __init__(self, record: RetryRecord) -> None:
        super().__init__(
            f"retry exhausted after {record.total_attempts} attempts "
            f"({record.final_error_class}: {record.final_error_message})"
        )
        self.record = record


class NonRecoverableError(RuntimeError):
    """The underlying error was classified as non-recoverable.

    Carries the populated :class:`RetryRecord`. Callers should escalate
    immediately (e.g. via the circuit breaker → governance hazard
    pipeline) without further retries.
    """

    def __init__(self, record: RetryRecord) -> None:
        super().__init__(
            f"non-recoverable {record.final_error_class}: {record.final_error_message}"
        )
        self.record = record


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _frozen_meta(meta: Mapping[str, str]) -> Mapping[str, str]:
    return types.MappingProxyType({k: meta[k] for k in sorted(meta.keys())})


class _NoExc(BaseException):
    """Sentinel used as the initial ``last_exc`` so type-narrowing is
    monotonic. Never raised."""


__all__ = [
    "DEFAULT_BASE_DELAY_SEC",
    "DEFAULT_JITTER_FACTOR",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_DELAY_SEC",
    "NEW_PIP_DEPENDENCIES",
    "NonRecoverableError",
    "RecoverableExceptionPredicate",
    "RetryAttempt",
    "RetryExecutor",
    "RetryExhausted",
    "RetryMixin",
    "RetryOutcome",
    "RetryPolicy",
    "RetryRecord",
    "compute_backoff_sec",
    "default_is_recoverable",
]
