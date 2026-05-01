"""
system/config_schema.py

DIX VISION v42.2 — Tier-0 Step 12: Config schema validation.

Pydantic-backed schema for every runtime-knob in the system. The
schema defines:
    - the authoritative default value for each knob
    - the type and allowed range of each knob
    - which module owns each knob

Rules:

    1. Every configurable knob MUST be declared in a section model
       below. Raw dict access via ``system.config.get("x.y.z")``
       continues to work for backwards compatibility but the key MUST
       still resolve to a declared field on a section model.
    2. System startup MUST call :func:`load_config` exactly once; any
       validation failure is a FATAL boot-time error. The kernel must
       NOT swallow ``ConfigValidationError`` and continue — it must
       escalate to the bootstrap kill-switch path.
    3. Hot-path callers must NOT read config during a decision; config
       values must be snapshotted at boot and frozen.

See docs/ARCHITECTURE_V42_2_TIER0.md §14 for the binding contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# ─────────────────────────────────────────────────────────────────────
# Section models
# ─────────────────────────────────────────────────────────────────────


class _Section(BaseModel):
    """Base class; every section rejects unknown keys (FAIL FAST)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GuardianConfig(_Section):
    """system.enforcement.runtime_guardian knobs."""

    check_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    heartbeat_timeout_seconds: float = Field(default=10.0, gt=0, le=300)


class RiskConfig(_Section):
    """risk.* knobs (portfolio-level guardrails)."""

    max_drawdown_pct: float = Field(default=4.0, gt=0, le=100)
    max_loss_per_trade_pct: float = Field(default=1.0, gt=0, le=100)
    fast_path_max_latency_ms: float = Field(default=5.0, gt=0, le=1000)


class HazardConfig(_Section):
    """execution.hazard.* detector thresholds."""

    feed_silence_threshold_seconds: float = Field(default=5.0, gt=0, le=600)
    latency_spike_threshold_ms: float = Field(default=100.0, gt=0, le=10_000)


class DataConfig(_Section):
    """data paths (relative to repo root unless absolute)."""

    audit_log: str = Field(default="data/audit.jsonl")
    incidents: str = Field(default="data/incidents.jsonl")
    snapshots: str = Field(default="data/snapshots")

    @field_validator("audit_log", "incidents", "snapshots")
    @classmethod
    def _reject_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("data path may not be empty")
        return v


class LedgerConfig(_Section):
    """ledger storage knobs."""

    db_path: str = Field(default="data/sqlite/ledger.db")

    @field_validator("db_path")
    @classmethod
    def _reject_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ledger.db_path may not be empty")
        return v


class SystemConfig(BaseModel):
    """Root config model — all sections are required, all extras rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    guardian: GuardianConfig = Field(default_factory=GuardianConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    hazard: HazardConfig = Field(default_factory=HazardConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)

    def as_dotted(self) -> dict[str, Any]:
        """Flatten into dot-path form matching the legacy get('a.b')
        interface in ``system.config``."""
        out: dict[str, Any] = {}
        for section_name, section in self.model_dump().items():
            for field_name, value in section.items():
                out[f"{section_name}.{field_name}"] = value
        return out


# ─────────────────────────────────────────────────────────────────────
# FAIL-FAST loader
# ─────────────────────────────────────────────────────────────────────


class ConfigValidationError(Exception):
    """Raised at boot when the config is invalid. FATAL — bootstrap
    must treat this as a kill-switch trigger in prod."""


def _dotted_to_nested(values: dict[str, Any]) -> dict[str, Any]:
    """Expand ``{"risk.max_drawdown_pct": 4}`` into
    ``{"risk": {"max_drawdown_pct": 4}}``."""
    nested: dict[str, Any] = {}
    for key, value in values.items():
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigValidationError(
                    f"config key {key!r} conflicts with scalar at {part!r}"
                )
        cursor[parts[-1]] = value
    return nested


def load_config(
    overrides: dict[str, Any] | None = None,
    *,
    yaml_path: str | Path | None = None,
) -> SystemConfig:
    """Build and validate a :class:`SystemConfig`.

    Parameters
    ----------
    overrides:
        Either a nested dict (``{"risk": {"max_drawdown_pct": 3.5}}``)
        OR a flat dotted dict (``{"risk.max_drawdown_pct": 3.5}``).
        Flat form is auto-expanded.
    yaml_path:
        Optional YAML file; if present, merged BELOW ``overrides``
        (overrides win).

    Raises
    ------
    ConfigValidationError
        If any field is unknown, mistyped, or out of range. FATAL —
        callers should NOT catch except to log + escalate.
    """
    base: dict[str, Any] = {}
    if yaml_path is not None:
        import yaml  # deferred import — yaml not needed in the hot path

        p = Path(yaml_path)
        if not p.is_file():
            raise ConfigValidationError(f"config YAML missing: {p}")
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"invalid YAML at {p}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigValidationError(
                f"config YAML at {p} must be a mapping, got {type(loaded).__name__}"
            )
        base = loaded

    if overrides:
        flat_keys = [k for k in overrides if "." in k]
        if flat_keys and len(flat_keys) == len(overrides):
            # all-flat form
            merged_flat = {**_flatten(base), **overrides}
            nested = _dotted_to_nested(merged_flat)
        else:
            nested = {**base, **overrides}
    else:
        nested = base

    try:
        return SystemConfig(**nested)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"config schema validation failed:\n{exc}"
        ) from exc


def _flatten(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in nested.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


__all__ = [
    "ConfigValidationError",
    "DataConfig",
    "GuardianConfig",
    "HazardConfig",
    "LedgerConfig",
    "RiskConfig",
    "SystemConfig",
    "load_config",
]
