"""
tests/test_hazard_severity_parity.py

Parity suite for the hazard severity classifier
(execution/hazard/severity_classifier.py). Each backend
(pure-Python reference + Rust via dixvision_py_system) must satisfy
identical invariants. The Rust test class is skipped when the
extension is not importable in the current environment.
"""
from __future__ import annotations

import pytest

import execution.hazard.severity_classifier as sc
from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType

# -- backend availability probe ----------------------------------------------

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
except ImportError:  # pragma: no cover
    _HAVE_RUST = False


# -- reusable invariant tests -------------------------------------------------

class _InvariantTests:
    """Invariants every backend must satisfy.

    Subclasses patch the module-level ``_HAVE_RUST`` flag (plus the
    ``_rs`` handle) to force the selector into one backend, then run
    the same tests. Any divergence would make the test fail in one
    subclass but not the other.
    """

    # Must be provided by the subclass ----------------------------------
    backend_name: str = ""

    # ------------------------------------------------------ should_halt_trading

    def test_halt_on_any_critical_severity(self) -> None:
        for ty in HazardType:
            assert sc.should_halt_trading(ty, HazardSeverity.CRITICAL), (
                f"{self.backend_name}: CRITICAL {ty.value} must halt"
            )

    def test_halt_on_always_halting_types(self) -> None:
        for ty in (
            HazardType.DATA_CORRUPTION_SUSPECTED,
            HazardType.LEDGER_INCONSISTENCY,
            HazardType.API_CONNECTIVITY_FAILURE,
        ):
            for sev in HazardSeverity:
                assert sc.should_halt_trading(ty, sev), (
                    f"{self.backend_name}: {sev.value} {ty.value} must halt"
                )

    def test_does_not_halt_on_low_medium_other_types(self) -> None:
        assert not sc.should_halt_trading(HazardType.EXCHANGE_TIMEOUT, "LOW")
        assert not sc.should_halt_trading(HazardType.FEED_SILENCE, "MEDIUM")
        assert not sc.should_halt_trading(HazardType.EXECUTION_LATENCY_SPIKE, "LOW")

    # -------------------------------------------------- should_enter_safe_mode

    def test_safe_mode_on_high_or_critical(self) -> None:
        for ty in HazardType:
            for sev in (HazardSeverity.HIGH, HazardSeverity.CRITICAL):
                assert sc.should_enter_safe_mode(ty, sev)

    def test_safe_mode_on_feed_silence_or_exchange_timeout_any_severity(self) -> None:
        for ty in (HazardType.FEED_SILENCE, HazardType.EXCHANGE_TIMEOUT):
            for sev in HazardSeverity:
                assert sc.should_enter_safe_mode(ty, sev)

    def test_safe_mode_does_not_trigger_on_low_unrelated(self) -> None:
        assert not sc.should_enter_safe_mode(HazardType.MEMORY_PRESSURE, "LOW")
        assert not sc.should_enter_safe_mode(HazardType.CPU_OVERLOAD, "MEDIUM")

    # -------------------------------------------------------- classify_severity

    def test_severity_promotion_for_data_corruption(self) -> None:
        assert sc.classify_severity(
            HazardType.DATA_CORRUPTION_SUSPECTED, HazardSeverity.LOW
        ) == HazardSeverity.CRITICAL
        assert sc.classify_severity(
            HazardType.LEDGER_INCONSISTENCY, HazardSeverity.MEDIUM
        ) == HazardSeverity.CRITICAL

    def test_severity_passes_through_non_critical_types(self) -> None:
        assert sc.classify_severity(
            HazardType.FEED_SILENCE, HazardSeverity.HIGH
        ) == HazardSeverity.HIGH
        assert sc.classify_severity(
            HazardType.EXECUTION_LATENCY_SPIKE, HazardSeverity.MEDIUM
        ) == HazardSeverity.MEDIUM

    # -------------------------------------------------------- classify_response

    def test_response_map_is_stable(self) -> None:
        assert sc.classify_response(HazardType.EXCHANGE_TIMEOUT) == "CANCEL_ALL_OPEN_ORDERS"
        assert sc.classify_response(HazardType.FEED_SILENCE) == "PAUSE_NEW_ORDERS"
        assert sc.classify_response(HazardType.EXECUTION_LATENCY_SPIKE) == "REDUCE_EXPOSURE"
        assert sc.classify_response(HazardType.DATA_CORRUPTION_SUSPECTED) == "HALT_TRADING"
        assert sc.classify_response(HazardType.API_CONNECTIVITY_FAILURE) == "HALT_TRADING"

    def test_response_falls_through_to_observe(self) -> None:
        # LEDGER_INCONSISTENCY has no explicit response entry → OBSERVE.
        # (The halting decision is separate, handled by should_halt_trading.)
        assert sc.classify_response(HazardType.LEDGER_INCONSISTENCY) == "OBSERVE"
        assert sc.classify_response(HazardType.SYSTEM_DEGRADATION) == "OBSERVE"
        assert sc.classify_response(HazardType.MEMORY_PRESSURE) == "OBSERVE"

    def test_response_on_unknown_type_is_observe(self) -> None:
        # classify_response accepts raw strings too. Never crash on
        # novel variants — conservative default.
        assert sc.classify_response("TOTALLY_NOVEL_TYPE") == "OBSERVE"

    # ----------------------------------------- calling convention: HazardEvent

    def test_accepts_hazard_event(self) -> None:
        evt = HazardEvent(
            hazard_type=HazardType.API_CONNECTIVITY_FAILURE,
            severity=HazardSeverity.LOW,
            source="parity-test",
        )
        assert sc.should_halt_trading(evt)
        # safe mode is orthogonal — LOW severity on a non-feed/timeout
        # type does not trigger safe mode even though it halts.
        assert not sc.should_enter_safe_mode(evt)
        assert sc.classify_severity(evt) == HazardSeverity.LOW  # not promoted
        assert sc.classify_response(evt) == "HALT_TRADING"

        feed = HazardEvent(
            hazard_type=HazardType.FEED_SILENCE,
            severity=HazardSeverity.LOW,
            source="parity-test",
        )
        assert sc.should_enter_safe_mode(feed)
        assert not sc.should_halt_trading(feed)

    def test_accepts_plain_strings(self) -> None:
        # Callers crossing language boundaries may hold raw strings.
        assert sc.should_halt_trading("DATA_CORRUPTION_SUSPECTED", "LOW")
        assert sc.classify_severity("DATA_CORRUPTION_SUSPECTED", "LOW") == HazardSeverity.CRITICAL

    # ---------------------------------------- helpers: known-variant predicates

    def test_is_known_type_predicate(self) -> None:
        for ty in HazardType:
            assert sc.is_known_hazard_type(ty)
            assert sc.is_known_hazard_type(ty.value)
        assert not sc.is_known_hazard_type("NOT_A_REAL_TYPE")
        assert not sc.is_known_hazard_type("")

    def test_is_known_severity_predicate(self) -> None:
        for sev in HazardSeverity:
            assert sc.is_known_severity(sev)
            assert sc.is_known_severity(sev.value)
        assert not sc.is_known_severity("VERY_BAD")
        assert not sc.is_known_severity("")


# -- Python backend (always available) ---------------------------------------

class TestPythonBackend(_InvariantTests):
    backend_name = "python"

    @pytest.fixture(autouse=True)
    def _force_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the module to use the pure-Python reference regardless
        of whether the Rust wheel is built in this environment."""
        monkeypatch.setattr(sc, "_HAVE_RUST", False)


# -- Rust backend (skipped when the wheel is not built) ---------------------

@pytest.mark.skipif(
    not _HAVE_RUST,
    reason="dixvision_py_system not built; run `maturin develop -m rust/py_system/Cargo.toml`",
)
class TestRustBackend(_InvariantTests):
    backend_name = "rust"

    @pytest.fixture(autouse=True)
    def _force_rust(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the module to use the Rust backend even if the
        selector would otherwise deselect it for some reason (e.g.
        partial surface). We only reach this class when the probe at
        module level confirmed availability."""
        monkeypatch.setattr(sc, "_HAVE_RUST", True)
        monkeypatch.setattr(sc, "_rs", _rs)
