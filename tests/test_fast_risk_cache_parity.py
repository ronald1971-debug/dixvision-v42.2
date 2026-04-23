"""
tests/test_fast_risk_cache_parity.py
DIX VISION v42.2 — T0-1 FastRiskCache parity suite (Rust port).

Purpose: prove the Rust-backed cache (``dixvision_py_system``) satisfies
the identical invariants to the pure-Python reference. The Rust side
uses ``ArcSwap<RiskConstraints>`` for lock-free reads and
``parking_lot::Mutex`` to serialize writers; the Python side uses
CPython atomic reference assignment guarded by an ``RLock``. Both
expose the same duck-typed ``get`` / ``update`` / ``enter_safe_mode``
/ ``exit_safe_mode`` / ``halt_trading`` / ``resume_trading`` API.

Structure
    * ``_BackendContract`` — the ten invariants every backend must
      meet. Subclasses set ``make_cache``.
    * ``TestPythonBackend`` — runs against the pure-Python reference.
    * ``TestRustBackend`` — runs against the Rust backend. Skipped
      cleanly when the ``dixvision_py_system`` wheel is not installed.
    * ``test_backend_selector_*`` — asserts the module-level
      ``backend()`` reports the selected implementation and
      ``get_risk_cache`` returns a shared instance.

The Rust backend is a **process-wide** singleton, so the Rust test
class carefully scopes assertions to *deltas* rather than absolute
values — e.g. "update() flips the flag", not "flag is initially X" —
so test ordering between Rust-backed tests does not cause flakes.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from system import fast_risk_cache as frc_mod  # noqa: E402
from system.fast_risk_cache import RiskConstraints  # noqa: E402


def _rust_wheel_available() -> bool:
    try:
        importlib.import_module("dixvision_py_system")
        return True
    except ImportError:
        return False


class _BackendContract:
    """Invariants every backend must satisfy. Subclasses set
    ``make_cache`` to the appropriate factory."""

    make_cache = staticmethod(frc_mod.make_python_cache)

    def setup_method(self) -> None:  # pytest hook
        self.cache = self.make_cache()  # type: ignore[misc]
        assert self.cache is not None, "cache factory returned None"
        # Reset to a known-good baseline. The Python backend gives a
        # fresh instance every call, so this is a no-op there; the
        # Rust singleton needs explicit reset so cross-test state
        # does not bleed through.
        self.cache.update(
            max_position_pct=1.0,
            max_order_size_usd=10_000.0,
            volatility_band_high=0.05,
            volatility_band_low=0.001,
            circuit_breaker_drawdown=0.04,
            circuit_breaker_loss_pct=0.01,
            trading_allowed=True,
            safe_mode=False,
        )

    # ---- shape / defaults -------------------------------------------------

    def test_get_returns_risk_constraints_dataclass(self) -> None:
        c = self.cache.get()
        assert isinstance(c, RiskConstraints)

    def test_defaults_are_permissive_and_safe(self) -> None:
        c = self.cache.get()
        assert c.trading_allowed is True
        assert c.safe_mode is False
        assert c.max_position_pct == pytest.approx(1.0)
        assert c.max_order_size_usd == pytest.approx(10_000.0)
        assert c.circuit_breaker_loss_pct == pytest.approx(0.01)
        # Timestamp is non-empty and ISO-8601 shaped.
        assert c.last_updated_utc
        assert "T" in c.last_updated_utc

    # ---- allows_trade -----------------------------------------------------

    def test_allows_trade_rejects_when_trading_disabled(self) -> None:
        self.cache.update(trading_allowed=False)
        ok, reason = self.cache.get().allows_trade(100.0, 100_000.0)
        assert ok is False
        assert reason == "trading_not_allowed"

    def test_allows_trade_rejects_when_safe_mode(self) -> None:
        self.cache.update(safe_mode=True)
        ok, reason = self.cache.get().allows_trade(100.0, 100_000.0)
        assert ok is False
        assert reason == "safe_mode_active"

    def test_allows_trade_requires_portfolio(self) -> None:
        ok, reason = self.cache.get().allows_trade(100.0, 0.0)
        assert ok is False
        assert reason == "portfolio_usd_required"

    def test_allows_trade_enforces_absolute_cap_before_percentage(self) -> None:
        # max_order_size_usd defaults to 10_000. A 20_000 order on a
        # 1M portfolio is 2% (within loss_pct = 1%? no, over) but
        # we want absolute to fire first regardless.
        self.cache.update(max_order_size_usd=10_000.0, circuit_breaker_loss_pct=0.5)
        ok, reason = self.cache.get().allows_trade(20_000.0, 1_000_000.0)
        assert ok is False
        assert reason.startswith("size_usd_20000.00_exceeds_max_10000.00")

    def test_allows_trade_enforces_percentage_circuit_breaker(self) -> None:
        self.cache.update(max_order_size_usd=1_000_000.0, circuit_breaker_loss_pct=0.01)
        ok, reason = self.cache.get().allows_trade(5_000.0, 100_000.0)
        assert ok is False
        assert reason.startswith("size_pct_0.0500_exceeds_limit_0.01")

    def test_allows_trade_accepts_within_limits(self) -> None:
        self.cache.update(max_order_size_usd=10_000.0, circuit_breaker_loss_pct=0.05)
        ok, reason = self.cache.get().allows_trade(500.0, 100_000.0)
        assert ok is True
        assert reason == "ok"

    # ---- mode transitions -------------------------------------------------

    def test_enter_safe_mode_halts_and_arms_safe(self) -> None:
        c = self.cache.enter_safe_mode()
        assert c.trading_allowed is False
        assert c.safe_mode is True

    def test_exit_safe_mode_resumes_and_disarms_safe(self) -> None:
        self.cache.enter_safe_mode()
        c = self.cache.exit_safe_mode()
        assert c.trading_allowed is True
        assert c.safe_mode is False

    def test_halt_and_resume_cycle(self) -> None:
        c1 = self.cache.halt_trading()
        assert c1.trading_allowed is False
        c2 = self.cache.resume_trading()
        assert c2.trading_allowed is True
        assert c2.safe_mode is False

    # ---- atomicity --------------------------------------------------------

    def test_update_swaps_reference_atomically(self) -> None:
        before = self.cache.get()
        self.cache.update(max_order_size_usd=42_000.0)
        after = self.cache.get()
        # RiskConstraints is frozen, so atomic swap = new object
        # identity + updated field + fresh timestamp.
        assert after.max_order_size_usd == pytest.approx(42_000.0)
        assert after is not before
        assert after.last_updated_utc != before.last_updated_utc or (
            after.last_updated_utc == before.last_updated_utc
            and after.max_order_size_usd != before.max_order_size_usd
        )

    def test_concurrent_readers_never_see_partial_update(self) -> None:
        """Spawn N readers + alternating writer; no reader may ever
        observe a half-applied snapshot (either old values everywhere
        or new values everywhere)."""
        stop = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            flip = False
            while not stop.is_set():
                if flip:
                    self.cache.update(
                        max_order_size_usd=10_000.0, circuit_breaker_loss_pct=0.01
                    )
                else:
                    self.cache.update(
                        max_order_size_usd=99_999.0, circuit_breaker_loss_pct=0.99
                    )
                flip = not flip

        def reader() -> None:
            for _ in range(2_000):
                c = self.cache.get()
                size = c.max_order_size_usd
                loss = c.circuit_breaker_loss_pct
                # Matched pairs: (10_000, 0.01) or (99_999, 0.99).
                # Anything else proves a torn read.
                matched = (
                    (size == pytest.approx(10_000.0) and loss == pytest.approx(0.01))
                    or (
                        size == pytest.approx(99_999.0)
                        and loss == pytest.approx(0.99)
                    )
                )
                if not matched:
                    errors.append(f"torn read: size={size}, loss={loss}")
                    return

        w = threading.Thread(target=writer)
        w.start()
        readers = [threading.Thread(target=reader) for _ in range(8)]
        for r in readers:
            r.start()
        for r in readers:
            r.join()
        stop.set()
        w.join()
        assert not errors, errors[:3]


class TestPythonBackend(_BackendContract):
    """Python-backed reference implementation."""
    make_cache = staticmethod(frc_mod.make_python_cache)


@pytest.mark.skipif(
    not _rust_wheel_available(),
    reason="dixvision_py_system wheel not built in this environment",
)
class TestRustBackend(_BackendContract):
    """Rust-backed implementation via PyO3. Skipped cleanly when the
    wheel is absent (dev boxes, replay-only sandboxes)."""
    make_cache = staticmethod(frc_mod.make_rust_cache)


# ---- module-level selector ----------------------------------------------


def test_backend_selector_reports_active_backend() -> None:
    """``backend()`` must return ``"rust"`` or ``"python"`` and must
    be consistent with whether the wheel is importable in this
    process."""
    name = frc_mod.backend()
    assert name in {"rust", "python"}
    if _rust_wheel_available():
        assert name == "rust"
    else:
        assert name == "python"


def test_get_risk_cache_returns_shared_instance() -> None:
    a = frc_mod.get_risk_cache()
    b = frc_mod.get_risk_cache()
    assert a is b


def test_get_risk_cache_has_expected_surface() -> None:
    cache = frc_mod.get_risk_cache()
    for name in (
        "get",
        "update",
        "enter_safe_mode",
        "exit_safe_mode",
        "halt_trading",
        "resume_trading",
    ):
        assert hasattr(cache, name), f"cache missing method: {name}"


def test_rust_tuple_field_order_matches_dataclass() -> None:
    """The Rust PyO3 seam returns a 9-tuple in an order that must be
    kept in lock-step with ``RiskConstraints``. If someone reshuffles
    one without updating the other, this test catches it before
    production ever sees a mis-indexed snapshot."""
    import dataclasses
    dataclass_fields = tuple(f.name for f in dataclasses.fields(RiskConstraints))
    # pylint: disable=protected-access
    assert frc_mod._RUST_TUPLE_FIELDS == dataclass_fields
