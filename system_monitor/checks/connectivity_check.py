"""
system_monitor/checks/connectivity_check.py
Simple TCP reachability check for an endpoint.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str


def check_connectivity(host: str, port: int, timeout_s: float = 2.0) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return CheckResult(True, f"{host}:{port} reachable")
    except Exception as e:  # noqa: BLE001 - want all exceptions
        return CheckResult(False, f"{host}:{port} unreachable: {e}")
