"""
execution/confirmations/reconciliation.py
Periodically reconciles our portfolio read-model against each adapter's
reported balances. Mismatches → HAZARD.RECONCILIATION_DRIFT event.
"""
from __future__ import annotations

import threading

from execution.adapter_router import get_adapter_router
from execution.hazard.async_bus import HazardSeverity, HazardType
from execution.hazard.event_emitter import get_hazard_emitter
from state.ledger.writer import get_writer


class Reconciliation:
    def __init__(self, interval_s: float = 60.0, tolerance_pct: float = 0.5) -> None:
        self._interval_s = interval_s
        self._tolerance = tolerance_pct
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer = get_writer()
        self._emitter = get_hazard_emitter("reconciliation")
        self._last_report: dict[str, float] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DIX-Reconciliation"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                pass
            self._stop.wait(self._interval_s)

    def run_once(self) -> dict[str, float]:
        router = get_adapter_router()
        report: dict[str, float] = {}
        for entry in router.entries():
            try:
                report[entry.name] = float(entry.adapter.get_balance("USDT"))
            except Exception:
                continue
        # diff
        drift: dict[str, float] = {}
        for name, bal in report.items():
            prev = self._last_report.get(name, bal)
            if prev <= 0:
                continue
            delta_pct = abs(bal - prev) / prev * 100.0
            if delta_pct > self._tolerance:
                drift[name] = delta_pct
        if drift:
            self._writer.write("SYSTEM", "RECONCILIATION_DRIFT", "reconciliation", {
                "drift_pct": drift, "report": report,
            })
            self._emitter.emit(
                HazardType.DATA_CORRUPTION_SUSPECTED,
                HazardSeverity.HIGH,
                {"drift": drift},
            )
        self._last_report = report
        return report


_r: Reconciliation | None = None
_lock = threading.Lock()


def get_reconciliation() -> Reconciliation:
    global _r
    if _r is None:
        with _lock:
            if _r is None:
                _r = Reconciliation()
    return _r
