"""
translation/audit_log.py
DIX VISION v42.2 — Translation Audit Log
"""
from __future__ import annotations

from system.audit_logger import get_audit_logger


def log_translation(source: str, input_type: str, output_type: str,
                    success: bool, error: str = "") -> None:
    get_audit_logger().log("TRANSLATION", source, {
        "input_type": input_type, "output_type": output_type,
        "success": success, "error": error
    })
