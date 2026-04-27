"""Deterministic retry classifier — Phase 2 / v2-C.

Maps a venue / network failure code to one of three classes and a
bounded retry decision. The mapping is data-driven so per-venue
overrides go via configuration, not code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RetryClassification(StrEnum):
    TRANSIENT = "TRANSIENT"
    THROTTLED = "THROTTLED"
    PERMANENT = "PERMANENT"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Outcome of one retry-policy lookup.

    ``backoff_ns`` is monotonic-time delta; the caller schedules the
    retry (we never call ``time.sleep`` inside the policy — keeps it
    deterministic).
    """

    classification: RetryClassification
    should_retry: bool
    attempt: int
    backoff_ns: int
    reason: str


# Sane defaults — venues override via :class:`RetryPolicy(**kwargs)`.
_DEFAULT_TRANSIENT_CODES: frozenset[str] = frozenset(
    {
        "TIMEOUT",
        "NETWORK_ERROR",
        "TEMPORARY_DISCONNECT",
        "EXCHANGE_UNREACHABLE",
        "STALE_QUOTE",
    }
)

_DEFAULT_THROTTLED_CODES: frozenset[str] = frozenset(
    {
        "RATE_LIMIT",
        "TOO_MANY_REQUESTS",
        "BURST_LIMIT",
    }
)

_DEFAULT_PERMANENT_CODES: frozenset[str] = frozenset(
    {
        "INSUFFICIENT_BALANCE",
        "BAD_SYMBOL",
        "INVALID_ORDER",
        "REJECTED_BY_RISK",
        "AUTH_FAILED",
    }
)


@dataclass(slots=True)
class RetryPolicy:
    """Deterministic retry policy.

    Args:
        max_attempts: Inclusive upper bound on retry attempts; first
            try is attempt ``1``.
        base_backoff_ns: Initial backoff (nanoseconds).
        backoff_factor: Multiplicative factor between attempts.
        max_backoff_ns: Cap on ``backoff_ns`` (clamped before return).
        transient_codes / throttled_codes / permanent_codes: Override
            the default mappings.
    """

    name: str = "retry_policy"
    spec_id: str = "EXEC-LC-04"
    max_attempts: int = 3
    base_backoff_ns: int = 100_000_000  # 100 ms
    backoff_factor: float = 2.0
    max_backoff_ns: int = 5_000_000_000  # 5 s
    transient_codes: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_TRANSIENT_CODES
    )
    throttled_codes: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_THROTTLED_CODES
    )
    permanent_codes: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_PERMANENT_CODES
    )

    def classify(self, error_code: str) -> RetryClassification:
        if error_code in self.permanent_codes:
            return RetryClassification.PERMANENT
        if error_code in self.throttled_codes:
            return RetryClassification.THROTTLED
        if error_code in self.transient_codes:
            return RetryClassification.TRANSIENT
        # Unknown errors default to PERMANENT — never retry blindly.
        return RetryClassification.PERMANENT

    def decide(self, *, error_code: str, attempt: int) -> RetryDecision:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        cls = self.classify(error_code)
        if cls is RetryClassification.PERMANENT:
            return RetryDecision(
                classification=cls,
                should_retry=False,
                attempt=attempt,
                backoff_ns=0,
                reason=f"permanent: {error_code}",
            )
        if attempt >= self.max_attempts:
            return RetryDecision(
                classification=cls,
                should_retry=False,
                attempt=attempt,
                backoff_ns=0,
                reason="max_attempts_exceeded",
            )
        # Throttled retries get longer backoff than transients.
        floor = self.base_backoff_ns
        if cls is RetryClassification.THROTTLED:
            floor *= 4
        backoff = int(floor * (self.backoff_factor ** (attempt - 1)))
        backoff = min(backoff, self.max_backoff_ns)
        return RetryDecision(
            classification=cls,
            should_retry=True,
            attempt=attempt,
            backoff_ns=backoff,
            reason=f"{cls.name.lower()}: {error_code}",
        )


__all__ = ["RetryClassification", "RetryDecision", "RetryPolicy"]
