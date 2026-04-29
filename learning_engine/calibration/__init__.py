"""Coherence calibration — INV-53 reader side (Phase 6.T1c reader).

Offline package. Pure functions only. Reads BELIEF_STATE_SNAPSHOT,
PRESSURE_VECTOR_SNAPSHOT, META_AUDIT, and REWARD_BREAKDOWN events
from the ledger and emits CALIBRATION_REPORT SystemEvents. Never
gates execution.
"""

from learning_engine.calibration.coherence_calibrator import (
    CALIBRATION_REPORT_VERSION,
    CALIBRATOR_SOURCE,
    CalibrationReport,
    calibrate_coherence_window,
)

__all__ = [
    "CALIBRATION_REPORT_VERSION",
    "CALIBRATOR_SOURCE",
    "CalibrationReport",
    "calibrate_coherence_window",
]
