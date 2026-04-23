"""
system/fast_risk_cache.py
DIX VISION v42.2 — Precomputed Risk Cache (FAST PATH, NO RPC)

Updated asynchronously by Governance.
Consumed synchronously by Indira with zero latency.
Thread-safe reads via atomic reference swap.

# Backends

This module has two interchangeable backends and picks one at import time:

* **Rust (``dixvision_py_system``)** — PyO3 extension built from
  ``rust/py_system/``. Preferred when the wheel is installed. Read
  path is a lock-free ``ArcSwap::load()`` on the Rust side; the
  Python side only pays a single FFI trampoline + tuple marshalling
  per call and reconstructs the ``RiskConstraints`` dataclass locally.
* **Pure Python (fallback)** — the original CPython-atomic-reference
  implementation kept verbatim so pre-polyglot ledgers still replay
  on a box without the Rust wheel (no cargo, restricted glibc, etc).

Both backends satisfy the same invariants:

* ``get()`` returns an immutable ``RiskConstraints`` snapshot without
  ever blocking a concurrent writer (single-writer, multi-reader).
* ``update(**kwargs)`` serialises writes and atomically swaps in a
  fresh snapshot, stamping ``last_updated_utc`` to the current
  UTC-ISO string.
* ``enter_safe_mode`` / ``exit_safe_mode`` / ``halt_trading`` /
  ``resume_trading`` are convenience ``update()`` calls with the
  same semantics on both sides.
* ``allows_trade(size_usd, portfolio_usd)`` is a pure function on the
  ``RiskConstraints`` dataclass and therefore backend-independent.

Selection is observable via ``backend()``. Tests in
``tests/test_fast_risk_cache_parity.py`` assert both backends meet
the invariants by constructing instances via ``make_python_cache()``
and ``make_rust_cache()`` directly, without reloading this module
or mutating environment variables.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Optional

from system.time_source import utc_now


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

    def allows_trade(self, size_usd: float, portfolio_usd: float) -> tuple[bool, str]:
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


# ---------------------------------------------------------------------------
# Pure-Python backend. Also the reference implementation the parity
# suite validates the Rust port against.
# ---------------------------------------------------------------------------


class _PythonFastRiskCache:
    """CPython atomic-reference implementation. Attribute assignment on
    a Python object is effectively atomic under the GIL, so readers
    never see a half-built ``RiskConstraints``."""

    __slots__ = ("_constraints", "_lock")

    def __init__(self) -> None:
        self._constraints = RiskConstraints(
            last_updated_utc=utc_now().isoformat()
        )
        self._lock = RLock()

    def get(self) -> RiskConstraints:
        """Lock-free read (atomic reference in CPython)."""
        return self._constraints

    def update(self, **kwargs: object) -> RiskConstraints:
        """Governance calls this asynchronously to update constraints."""
        with self._lock:
            self._constraints = replace(
                self._constraints,
                last_updated_utc=utc_now().isoformat(),
                **kwargs,  # type: ignore[arg-type]
            )
            return self._constraints

    def enter_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=True, trading_allowed=False)

    def exit_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=False, trading_allowed=True)

    def halt_trading(self, reason: str = "") -> RiskConstraints:
        del reason  # Reason is logged by callers; cache just records the flag.
        return self.update(trading_allowed=False)

    def resume_trading(self) -> RiskConstraints:
        return self.update(trading_allowed=True, safe_mode=False)


def make_python_cache() -> _PythonFastRiskCache:
    """Construct a fresh Python-backed risk cache with its own state.
    Callers: the module-level default when the Rust wheel is
    unavailable, and the parity test suite."""
    return _PythonFastRiskCache()


# ---------------------------------------------------------------------------
# Rust backend wrapper. Imported lazily so a missing wheel falls
# through cleanly to the Python backend without an ``ImportError``
# at module load time.
# ---------------------------------------------------------------------------


# Field order MUST match ``rust/py_system/src/lib.rs::RiskTuple``.
# Any reshuffle breaks the ABI — treat this list and the Rust type
# alias as one diff.
_RUST_TUPLE_FIELDS = (
    "max_position_pct",
    "max_order_size_usd",
    "volatility_band_high",
    "volatility_band_low",
    "circuit_breaker_drawdown",
    "circuit_breaker_loss_pct",
    "trading_allowed",
    "safe_mode",
    "last_updated_utc",
)


def _from_rust_tuple(t: tuple) -> RiskConstraints:
    """Reconstruct a ``RiskConstraints`` from the 9-tuple returned by
    the PyO3 seam. Keep this helper symmetrical with the Rust
    ``snapshot_to_tuple`` function."""
    return RiskConstraints(**dict(zip(_RUST_TUPLE_FIELDS, t)))


def _to_rust_patch(
    max_position_pct: Optional[float] = None,
    max_order_size_usd: Optional[float] = None,
    volatility_band_high: Optional[float] = None,
    volatility_band_low: Optional[float] = None,
    circuit_breaker_drawdown: Optional[float] = None,
    circuit_breaker_loss_pct: Optional[float] = None,
    trading_allowed: Optional[bool] = None,
    safe_mode: Optional[bool] = None,
) -> tuple:
    """Marshal Python kwargs → positional 8-tuple for ``risk_update``.
    ``None`` slots leave the corresponding field unchanged."""
    return (
        max_position_pct,
        max_order_size_usd,
        volatility_band_high,
        volatility_band_low,
        circuit_breaker_drawdown,
        circuit_breaker_loss_pct,
        trading_allowed,
        safe_mode,
    )


class _RustFastRiskCache:
    """Thin wrapper delegating to the process-global Rust cache.

    The underlying ``FastRiskCache`` on the Rust side is a process
    singleton (``OnceLock`` inside the PyO3 seam). Constructing
    multiple instances of this wrapper therefore does NOT give you
    independent caches; tests that need isolation use the Python
    backend via ``make_python_cache()``."""

    __slots__ = ("_rust",)

    def __init__(self, rust_module: object) -> None:
        self._rust = rust_module

    def get(self) -> RiskConstraints:
        return _from_rust_tuple(self._rust.risk_get())  # type: ignore[attr-defined]

    def update(self, **kwargs: object) -> RiskConstraints:
        patch = _to_rust_patch(**kwargs)  # type: ignore[arg-type]
        return _from_rust_tuple(self._rust.risk_update(patch))  # type: ignore[attr-defined]

    def enter_safe_mode(self) -> RiskConstraints:
        return _from_rust_tuple(self._rust.risk_enter_safe_mode())  # type: ignore[attr-defined]

    def exit_safe_mode(self) -> RiskConstraints:
        return _from_rust_tuple(self._rust.risk_exit_safe_mode())  # type: ignore[attr-defined]

    def halt_trading(self, reason: str = "") -> RiskConstraints:
        del reason  # Reason is logged by callers; cache just records the flag.
        return _from_rust_tuple(self._rust.risk_halt_trading())  # type: ignore[attr-defined]

    def resume_trading(self) -> RiskConstraints:
        return _from_rust_tuple(self._rust.risk_resume_trading())  # type: ignore[attr-defined]


def make_rust_cache() -> Optional[_RustFastRiskCache]:
    """Return a Rust-backed risk cache if the ``dixvision_py_system``
    wheel is importable; otherwise ``None``.

    The underlying Rust cache is a process singleton, so any number
    of wrappers handed out from this factory share the same state.
    That's the intended semantics — governance and Indira are both
    meant to see the same snapshot. Tests that need isolation use
    ``make_python_cache()``."""
    try:
        import dixvision_py_system as _rust  # type: ignore[import-not-found]
    except ImportError:
        return None
    return _RustFastRiskCache(_rust)


# ``FastRiskCache`` is the public name external callers (governance,
# Indira, tests) instantiate. Aliasing it to the Python backend keeps
# the import surface unchanged from the pre-polyglot era; the Rust
# backend is selected through ``get_risk_cache()`` below.
FastRiskCache = _PythonFastRiskCache


# ---------------------------------------------------------------------------
# Module-level default. Picks the Rust backend if the wheel was built
# and installed; otherwise the pure-Python reference. Both are always
# available via the ``make_*_cache`` factories above.
# ---------------------------------------------------------------------------


_rust_impl: Optional[_RustFastRiskCache] = make_rust_cache()
_BACKEND_NAME: str = "rust" if _rust_impl is not None else "python"
_cache: Optional[object] = None
_init_lock = RLock()


def get_risk_cache() -> object:
    """Process-wide risk cache. First call constructs the cache using
    whichever backend was selected at import time; subsequent calls
    return the same instance.

    Return type is intentionally ``object`` rather than
    ``FastRiskCache`` because at runtime the instance may be the
    Rust-backed wrapper; both expose the same duck-typed methods
    (``get``, ``update``, ``enter_safe_mode``, ``exit_safe_mode``,
    ``halt_trading``, ``resume_trading``). Callers interact through
    these methods, not through ``isinstance`` checks."""
    global _cache
    if _cache is None:
        with _init_lock:
            if _cache is None:
                _cache = _rust_impl if _rust_impl is not None else make_python_cache()
    return _cache


def backend() -> str:
    """Which backend is active in this process. Test-hook only —
    runtime code MUST NOT branch on the return value. Returns either
    ``"rust"`` or ``"python"``."""
    return _BACKEND_NAME
