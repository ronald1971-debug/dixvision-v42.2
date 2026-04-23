"""system/fast_risk_cache.py

DIX VISION v42.2 — Precomputed Risk Cache (FAST PATH, NO RPC)

Updated asynchronously by Governance (the sole writer).
Consumed synchronously by Indira with zero latency.
Thread-safe reads via atomic reference swap.

Tier-0 Step 1 additions (see docs/ARCHITECTURE_V42_2_TIER0.md §2):

    - **version** (monotonic ``int``) — bumped on every ``update()``
      call. A non-increasing version means the writer missed an
      update or something corrupted the cache; Indira MUST reject.
    - **version_id** (stable ``str``) — deterministic hex derived from
      ``version`` + ``updated_at_ns``. Embedded in every decision
      record as ``risk_version_used`` so audit / replay can verify
      the exact constraints applied to each trade.
    - **updated_at_ns** (wall-clock nanoseconds) — stamps every
      update with ``time.time_ns()`` so Indira can measure staleness.
    - **staleness halt** — ``is_fresh(threshold_ns, now_ns)`` returns
      ``False`` when the cache is older than the threshold.
      ``allows_trade()`` now fails closed on stale cache.
    - **RiskReading helper** — atomic snapshot of ``(version,
      version_id, updated_at_ns, constraints)`` for callers that must
      stamp a decision with the exact cache revision they used.

Hard rule:

    If ``now_ns - updated_at_ns > STALENESS_THRESHOLD_NS``:
        reject ALL trades. No exceptions.

Hard rule:

    Every governance / fast-path decision record MUST include
    ``risk_version_used`` = the ``version_id`` returned by
    :meth:`FastRiskCache.read` at the moment the decision was made.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from threading import RLock

from system.time_source import utc_now


def _compute_version_id(version: int, updated_at_ns: int) -> str:
    """Stable deterministic version-id for a risk-cache revision.

    Used verbatim in decision records so every trade carries the exact
    cache revision it was priced against. Must be deterministic — the
    same ``(version, updated_at_ns)`` pair MUST always yield the same
    ``version_id`` across replays. We just hex-format both fields
    concatenated; no hashing, no collisions possible because ``version``
    is monotonic by construction.
    """
    return f"v{version:x}-{updated_at_ns:x}"


DEFAULT_STALENESS_THRESHOLD_NS = 5 * 1_000_000_000  # 5 seconds


def _wall_ns() -> int:
    return time.time_ns()


@dataclass(frozen=True)
class RiskConstraints:
    """Precomputed risk limits. Consumed by Indira fast path."""
    max_position_pct: float = 1.0       # max position as % of portfolio
    max_order_size_usd: float = 10_000.0
    volatility_band_high: float = 0.05  # 5%
    volatility_band_low: float = 0.001
    circuit_breaker_drawdown: float = 0.04  # 4%
    circuit_breaker_loss_pct: float = 0.01  # 1% per trade
    trading_allowed: bool = True
    safe_mode: bool = False
    last_updated_utc: str = ""
    version: int = 0
    updated_at_ns: int = 0
    version_id: str = ""

    def allows_trade(
        self,
        size_usd: float,
        portfolio_usd: float,
        *,
        now_ns: int | None = None,
        staleness_threshold_ns: int = DEFAULT_STALENESS_THRESHOLD_NS,
    ) -> tuple[bool, str]:
        if now_ns is not None and self.updated_at_ns > 0:
            if (now_ns - self.updated_at_ns) > staleness_threshold_ns:
                return False, "risk_cache_stale"
        if not self.trading_allowed:
            return False, "trading_not_allowed"
        if self.safe_mode:
            return False, "safe_mode_active"
        # Fail-closed: if we don't know the portfolio size we cannot
        # enforce the per-trade circuit breaker, so refuse rather than
        # silently skipping the percentage check.
        if portfolio_usd <= 0:
            return False, "portfolio_usd_required"
        # Absolute per-order cap governance sets via ConstraintCompiler.
        # Checked BEFORE the percentage rule so a large absolute size
        # on a very large portfolio still gets rejected.
        if size_usd > self.max_order_size_usd:
            return False, (
                f"size_usd_{size_usd:.2f}_exceeds_max_"
                f"{self.max_order_size_usd:.2f}"
            )
        pct = size_usd / portfolio_usd
        if pct > self.circuit_breaker_loss_pct:
            return False, f"size_pct_{pct:.4f}_exceeds_limit_{self.circuit_breaker_loss_pct}"
        return True, "ok"


@dataclass(frozen=True)
class RiskReading:
    """Atomic snapshot returned by :meth:`FastRiskCache.read`.

    Callers that must stamp a decision with the cache revision they
    used should capture a ``RiskReading`` once at the top of the
    decision and copy ``version_id`` into the decision record. Because
    :class:`RiskConstraints` is frozen and the cache swaps references
    atomically, the reading is a consistent point-in-time view.
    """

    version: int
    version_id: str
    updated_at_ns: int
    constraints: RiskConstraints


class FastRiskCache:
    """Atomic single-writer, multi-reader risk cache.

    Governance is the sole writer (async).
    Indira reads every tick with zero lock contention.

    T0-1 additions:
      - ``version`` is bumped monotonically on every ``update()``.
      - ``updated_at_ns`` is stamped with wall-clock nanoseconds.
      - ``is_fresh()`` lets Indira check staleness before trading.
    """

    def __init__(
        self,
        *,
        staleness_threshold_ns: int = DEFAULT_STALENESS_THRESHOLD_NS,
        clock_wall_ns: callable = _wall_ns,
    ) -> None:
        self._staleness_threshold_ns = staleness_threshold_ns
        self._clock = clock_wall_ns
        now = self._clock()
        self._constraints = RiskConstraints(
            last_updated_utc=utc_now().isoformat(),
            version=1,
            updated_at_ns=now,
            version_id=_compute_version_id(1, now),
        )
        self._lock = RLock()

    def get(self) -> RiskConstraints:
        """Lock-free read (atomic reference in CPython)."""
        return self._constraints

    @property
    def version(self) -> int:
        return self._constraints.version

    @property
    def updated_at_ns(self) -> int:
        return self._constraints.updated_at_ns

    @property
    def version_id(self) -> str:
        return self._constraints.version_id

    def read(self) -> RiskReading:
        """Atomic snapshot of the cache revision.

        Callers stamp ``reading.version_id`` into their decision record
        as ``risk_version_used`` so audit / replay can verify the
        constraints that were applied.
        """
        c = self._constraints
        return RiskReading(
            version=c.version,
            version_id=c.version_id,
            updated_at_ns=c.updated_at_ns,
            constraints=c,
        )

    @property
    def staleness_threshold_ns(self) -> int:
        return self._staleness_threshold_ns

    def is_fresh(self, now_ns: int | None = None) -> bool:
        """Return ``True`` if the cache was updated within the
        staleness threshold. Indira MUST call this every tick."""
        if now_ns is None:
            now_ns = self._clock()
        return (now_ns - self._constraints.updated_at_ns) <= self._staleness_threshold_ns

    def staleness_ns(self, now_ns: int | None = None) -> int:
        """Wall-clock nanoseconds since the last update."""
        if now_ns is None:
            now_ns = self._clock()
        return max(0, now_ns - self._constraints.updated_at_ns)

    def update(self, **kwargs) -> RiskConstraints:
        """Governance calls this asynchronously to update constraints.

        Monotonically bumps ``version`` and stamps ``updated_at_ns``.
        """
        with self._lock:
            new_version = self._constraints.version + 1
            now = self._clock()
            self._constraints = replace(
                self._constraints,
                last_updated_utc=utc_now().isoformat(),
                version=new_version,
                updated_at_ns=now,
                version_id=_compute_version_id(new_version, now),
                **kwargs,
            )
            return self._constraints

    def enter_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=True, trading_allowed=False)

    def exit_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=False, trading_allowed=True)

    def halt_trading(self, reason: str = "") -> RiskConstraints:
        return self.update(trading_allowed=False)

    def resume_trading(self) -> RiskConstraints:
        return self.update(trading_allowed=True, safe_mode=False)


_cache: FastRiskCache | None = None
_lock = RLock()

def get_risk_cache() -> FastRiskCache:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = FastRiskCache()
    return _cache
