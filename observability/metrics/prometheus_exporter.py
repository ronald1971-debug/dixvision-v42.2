"""
observability/metrics/prometheus_exporter.py
Prometheus text-format exporter. No HTTP server embedded — callers wire the
output into their own handler (cockpit, /metrics etc.).
"""
from __future__ import annotations

from typing import Any


def _escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_line(name: str, labels: dict[str, str], value: float) -> str:
    if labels:
        lbl = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lbl}}} {float(value)}"
    return f"{name} {float(value)}"


def render_prometheus_text(snapshot: dict[str, Any]) -> str:
    """
    Render a metrics snapshot as Prometheus text-format.

    Snapshot shape is flexible:
      { "metric_name": value }
      { "metric_name": {"labels": {...}, "value": v} }
      { "metric_name": [ {"labels": {...}, "value": v}, ... ] }
    """
    out = []
    for name in sorted(snapshot.keys()):
        val = snapshot[name]
        safe_name = name.replace(".", "_").replace("-", "_")
        if isinstance(val, (int, float)):
            out.append(_format_line(safe_name, {}, float(val)))
        elif isinstance(val, dict) and "value" in val:
            labels = val.get("labels", {}) or {}
            out.append(_format_line(safe_name, labels, float(val["value"])))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "value" in item:
                    out.append(_format_line(
                        safe_name, item.get("labels", {}) or {},
                        float(item["value"]),
                    ))
    return "\n".join(out) + "\n"
