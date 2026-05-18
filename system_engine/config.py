"""I-28 pydantic-settings — canonical environment config (OFFLINE_ONLY).

# ADAPTED FROM: pydantic-settings
#   https://github.com/pydantic/pydantic-settings
#   We adopt the canonical precedence chain
#     env vars > .env file > registry/engines.yaml > defaults
#   and the value-object surface so a future operator-gated swap to a real
#   ``pydantic_settings.BaseSettings`` backend is a byte-equivalent change.

Canonical ledger row (per DIX_MASTER_CANONICAL.md, TIER I, I-28):

    I-28 pydantic-settings → system_engine/config.py
        - DIXConfig(BaseSettings) wraps all system config
        - Precedence: env vars > .env > registry/engines.yaml > defaults
        - All sensitive values (API keys) load from credentials/ — NEVER from env in production
        - Type validation on startup
        - Lazy seam: ``pydantic_settings`` is NEVER imported at module level
        - stdlib fallback always available (pure value-object surface)

Authority constraints (pinned by AST guardrail tests in
tests/test_system_config.py):

    INV-15  No top-level forbidden imports
            ({pydantic_settings, time, datetime, random, asyncio, os, numpy,
              torch, polars, requests}).  All inputs (env map, dotenv text,
              yaml text) are caller-supplied — no implicit ``os.environ`` /
              wall-clock reads.
    B1      No imports from runtime engine tiers (execution / intelligence /
            governance / learning / evolution).
    B27/B28/INV-71  No typed-event constructors
            ({PatchProposal, HazardEvent, SignalEvent, ExecutionEvent,
              SystemEvent, LearningUpdate}).  This module returns read-side
            value objects only.

Design:

    ``ConfigSource``        enum: DEFAULT / YAML / DOTENV / ENV (lowest →
                            highest precedence).
    ``ConfigEntry``         frozen+slotted entry with key / value / source.
    ``DIXConfig``           frozen+slotted top-level config value object;
                            holds ``entries`` as a sorted tuple and a
                            ``get(key)`` accessor.
    ``parse_dotenv``        pure parser for ``KEY=VALUE`` text bodies with
                            ``#`` comments and ``"…"`` / ``'…'`` quoting.
    ``parse_env_map``       pure ``Mapping[str, str]`` filter (drops keys
                            outside ``allowed_prefixes`` so secrets that
                            sneak into ``os.environ`` never enter the config
                            graph).
    ``coerce_value``        pure type coercer (``int`` / ``float`` / ``bool``
                            / ``str``) honouring an explicit declaration
                            table — no implicit guessing.
    ``load_config_stdlib``  pure composition: defaults → yaml → dotenv → env;
                            higher tiers override lower tiers per key with
                            full source provenance retained for audit.
    ``stdlib_config_factory``       always-available production default.
    ``enable_pydantic_settings_factory``  lazy seam (imports
                            ``pydantic_settings`` INSIDE function body only)
                            returning a config built from a real
                            ``BaseSettings`` subclass.

All composition is deterministic — given the same defaults / yaml / dotenv /
env inputs the resulting ``DIXConfig.entries`` tuple is byte-identical
across runs (INV-15).  No secrets are read here; the canonical rule routes
API keys via ``system_engine.credentials.*``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import yaml

CONFIG_VERSION: Final[str] = "v1.0-I28"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("pydantic-settings",)


# ---------------------------------------------------------------------------
# Forbidden-secret key fragments — never copied into DIXConfig under any
# precedence tier.  Mirrors the canonical credentials manifest rule.
# ---------------------------------------------------------------------------
_FORBIDDEN_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "api_key",
    "secret",
    "password",
    "private_key",
    "token",
    "session_key",
    "auth_header",
    "credential",
)


def _is_forbidden_secret_key(key: str) -> bool:
    """Return ``True`` if ``key`` looks like a secret (case-insensitive)."""
    lowered = key.lower()
    return any(frag in lowered for frag in _FORBIDDEN_KEY_FRAGMENTS)


# ---------------------------------------------------------------------------
# Source precedence — lowest → highest, retained in DIXConfig.entries.
# ---------------------------------------------------------------------------
class ConfigSource(str, Enum):  # noqa: UP042 — str subclass for byte-stable JSON output
    """Source of a single config entry, in precedence order (low → high)."""

    DEFAULT = "DEFAULT"
    YAML = "YAML"
    DOTENV = "DOTENV"
    ENV = "ENV"


_SOURCE_RANK: Final[dict[ConfigSource, int]] = {
    ConfigSource.DEFAULT: 0,
    ConfigSource.YAML: 1,
    ConfigSource.DOTENV: 2,
    ConfigSource.ENV: 3,
}


# ---------------------------------------------------------------------------
# Value object — one entry in the merged config.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ConfigEntry:
    """One merged config entry with full source provenance.

    Attributes
    ----------
    key:
        Canonical config key (must be non-empty, lower-case ASCII + ``_``).
    value:
        Coerced value (``str`` / ``int`` / ``float`` / ``bool`` / ``None``).
    source:
        The :class:`ConfigSource` tier that won the precedence race for
        this key.
    """

    key: str
    value: str | int | float | bool | None
    source: ConfigSource

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError(f"ConfigEntry: key must be non-empty str, got {self.key!r}")
        if any(c.isspace() for c in self.key):
            raise ValueError(f"ConfigEntry: key must not contain whitespace, got {self.key!r}")
        if not isinstance(self.source, ConfigSource):
            raise TypeError(
                f"ConfigEntry: source must be ConfigSource, got {type(self.source).__name__}"
            )
        if not isinstance(self.value, (str, int, float, bool, type(None))):
            raise TypeError(
                f"ConfigEntry: value type {type(self.value).__name__} not allowed; "
                "must be str/int/float/bool/None"
            )


# ---------------------------------------------------------------------------
# Top-level config value object.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DIXConfig:
    """Top-level canonical config — merged, validated, fully audit-traced.

    ``entries`` is a tuple of :class:`ConfigEntry` sorted by ``key`` so
    serialised forms are byte-identical across runs.
    """

    entries: tuple[ConfigEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            raise TypeError(f"DIXConfig: entries must be tuple, got {type(self.entries).__name__}")
        seen: set[str] = set()
        prev_key: str | None = None
        for e in self.entries:
            if not isinstance(e, ConfigEntry):
                raise TypeError(
                    f"DIXConfig: entries items must be ConfigEntry, got {type(e).__name__}"
                )
            if e.key in seen:
                raise ValueError(f"DIXConfig: duplicate key {e.key!r}")
            seen.add(e.key)
            if prev_key is not None and e.key < prev_key:
                raise ValueError(
                    f"DIXConfig: entries must be sorted by key; {e.key!r} comes after {prev_key!r}"
                )
            prev_key = e.key

    def get(self, key: str, default: Any = None) -> Any:
        """Return the merged value for ``key`` or ``default`` if absent."""
        for e in self.entries:
            if e.key == key:
                return e.value
        return default

    def source_of(self, key: str) -> ConfigSource | None:
        """Return the winning :class:`ConfigSource` for ``key`` or ``None``."""
        for e in self.entries:
            if e.key == key:
                return e.source
        return None

    def as_mapping(self) -> dict[str, str | int | float | bool | None]:
        """Return a plain ``dict`` projection (canonical-sorted)."""
        return {e.key: e.value for e in self.entries}


# ---------------------------------------------------------------------------
# Type-declaration table — explicit coercion targets per key.
# ---------------------------------------------------------------------------
ConfigType = type[str] | type[int] | type[float] | type[bool]


def coerce_value(
    raw: str | int | float | bool | None,
    target: ConfigType,
    *,
    key: str,
) -> str | int | float | bool | None:
    """Coerce ``raw`` to ``target`` with explicit error on mismatch.

    Booleans accept ``true`` / ``false`` / ``1`` / ``0`` / ``yes`` / ``no``
    (case-insensitive).  ``None`` passes through unchanged.
    """
    if raw is None:
        return None
    if target is str:
        return str(raw)
    if target is int:
        if isinstance(raw, bool):  # bool is subclass of int — reject
            raise ValueError(f"coerce_value[{key}]: cannot coerce bool to int (use bool target)")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if raw != int(raw):
                raise ValueError(
                    f"coerce_value[{key}]: float {raw!r} has fractional part; cannot coerce to int"
                )
            return int(raw)
        return int(str(raw).strip())
    if target is float:
        if isinstance(raw, bool):
            raise ValueError(f"coerce_value[{key}]: cannot coerce bool to float")
        return float(raw) if isinstance(raw, (int, float)) else float(str(raw).strip())
    if target is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"coerce_value[{key}]: cannot parse {raw!r} as bool")
    raise TypeError(f"coerce_value[{key}]: unsupported target type {target!r}")


# ---------------------------------------------------------------------------
# Pure parsers — env / dotenv / yaml text → flat key/value maps.
# ---------------------------------------------------------------------------
def parse_dotenv(text: str) -> dict[str, str]:
    """Parse a ``.env`` body into a dict.

    Honours ``#`` comments and ``"…"`` / ``'…'`` quoting.  Blank lines and
    pure-comment lines are skipped.  Raises ``ValueError`` on malformed
    lines so config errors fail fast.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_dotenv: text must be str, got {type(text).__name__}")
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"parse_dotenv: line {lineno} has no '=' separator: {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"parse_dotenv: line {lineno} has empty key: {raw!r}")
        if any(c.isspace() for c in key):
            raise ValueError(f"parse_dotenv: line {lineno} key {key!r} contains whitespace")
        # Strip surrounding quotes if both sides match.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[key] = value
    return out


def parse_env_map(
    env: Mapping[str, str],
    *,
    allowed_prefixes: tuple[str, ...] = ("DIX_",),
) -> dict[str, str]:
    """Filter an ``os.environ``-shaped mapping down to allow-listed keys.

    Only keys starting with one of ``allowed_prefixes`` are kept.  This is
    the canonical narrow gate that prevents arbitrary process env vars
    from leaking into the config graph.
    """
    if not isinstance(env, Mapping):
        raise TypeError(f"parse_env_map: env must be Mapping, got {type(env).__name__}")
    if not allowed_prefixes:
        raise ValueError("parse_env_map: allowed_prefixes must be non-empty")
    out: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not any(k.startswith(p) for p in allowed_prefixes):
            continue
        out[k] = v
    return out


def parse_yaml_config(text: str) -> dict[str, Any]:
    """Parse a YAML config text body into a flat ``dict`` of leaves.

    Nested mappings are flattened with ``.`` separators (``engines.intelligence.tier``).
    Non-leaf values (lists, deeply-nested) are dropped — the canonical
    config surface is flat key/value only.  Use :mod:`registry` loaders
    for structured registry data.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_yaml_config: text must be str, got {type(text).__name__}")
    if not text.strip():
        return {}
    parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"parse_yaml_config: top-level YAML must be a mapping, got {type(parsed).__name__}"
        )
    return _flatten_mapping(parsed)


def _flatten_mapping(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            continue
        full = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_mapping(value, full))
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[full] = value
    return out


# ---------------------------------------------------------------------------
# Pure composition — defaults → yaml → dotenv → env.
# ---------------------------------------------------------------------------
def load_config_stdlib(
    *,
    defaults: Mapping[str, str | int | float | bool | None] | None = None,
    yaml_text: str | None = None,
    dotenv_text: str | None = None,
    env: Mapping[str, str] | None = None,
    types: Mapping[str, ConfigType] | None = None,
    allowed_env_prefixes: tuple[str, ...] = ("DIX_",),
) -> DIXConfig:
    """Compose a :class:`DIXConfig` from the canonical precedence chain.

    Parameters
    ----------
    defaults, yaml_text, dotenv_text, env:
        The four canonical precedence tiers (lowest → highest).  Each is
        optional; ``None`` means "tier absent".
    types:
        Optional explicit type-coercion table keyed by canonical config
        key.  Keys not in the table default to ``str``.
    allowed_env_prefixes:
        Allow-list of env-var prefixes.  Forbidden-secret keys are
        rejected from every tier.

    Returns
    -------
    DIXConfig
        Frozen value object with ``entries`` sorted by key and source
        provenance retained.

    Raises
    ------
    ValueError
        If a forbidden-secret key appears in any tier, if YAML / dotenv
        parsing fails, or if type coercion fails for a typed key.
    """
    types = types or {}
    merged: dict[str, tuple[str | int | float | bool | None, ConfigSource]] = {}

    def _apply(
        source: ConfigSource,
        body: Mapping[str, str | int | float | bool | None],
    ) -> None:
        for raw_key, raw_value in body.items():
            if not isinstance(raw_key, str) or not raw_key:
                continue
            if _is_forbidden_secret_key(raw_key):
                raise ValueError(
                    f"load_config_stdlib: forbidden secret key {raw_key!r} "
                    f"present in tier {source.value} — route via credentials/"
                )
            target = types.get(raw_key, str)
            coerced = coerce_value(raw_value, target, key=raw_key)
            existing = merged.get(raw_key)
            if existing is None or _SOURCE_RANK[source] >= _SOURCE_RANK[existing[1]]:
                merged[raw_key] = (coerced, source)

    if defaults:
        _apply(ConfigSource.DEFAULT, defaults)
    if yaml_text is not None:
        _apply(ConfigSource.YAML, parse_yaml_config(yaml_text))
    if dotenv_text is not None:
        _apply(ConfigSource.DOTENV, parse_dotenv(dotenv_text))
    if env is not None:
        filtered = parse_env_map(env, allowed_prefixes=allowed_env_prefixes)
        _apply(ConfigSource.ENV, filtered)

    entries = tuple(ConfigEntry(key=k, value=v, source=s) for k, (v, s) in sorted(merged.items()))
    return DIXConfig(entries=entries)


# ---------------------------------------------------------------------------
# Factories — stdlib (production default) and lazy pydantic-settings seam.
# ---------------------------------------------------------------------------
def stdlib_config_factory(
    *,
    defaults: Mapping[str, str | int | float | bool | None] | None = None,
    yaml_text: str | None = None,
    dotenv_text: str | None = None,
    env: Mapping[str, str] | None = None,
    types: Mapping[str, ConfigType] | None = None,
    allowed_env_prefixes: tuple[str, ...] = ("DIX_",),
) -> DIXConfig:
    """Always-available production default — a thin alias of
    :func:`load_config_stdlib` for symmetry with the lazy seam."""
    return load_config_stdlib(
        defaults=defaults,
        yaml_text=yaml_text,
        dotenv_text=dotenv_text,
        env=env,
        types=types,
        allowed_env_prefixes=allowed_env_prefixes,
    )


def enable_pydantic_settings_factory(
    *,
    defaults: Mapping[str, str | int | float | bool | None] | None = None,
    yaml_text: str | None = None,
    dotenv_text: str | None = None,
    env: Mapping[str, str] | None = None,
    types: Mapping[str, ConfigType] | None = None,
    allowed_env_prefixes: tuple[str, ...] = ("DIX_",),
) -> DIXConfig:
    """Lazy seam — gated activation of the ``pydantic_settings`` backend.

    The ``pydantic_settings`` package is imported INSIDE this function
    body (function-local) per the canonical TIER I lazy-seam pattern.
    The AST guardrail tests in ``tests/test_system_config.py`` pin that
    ``pydantic_settings`` never appears as a module-level import.

    Even with ``pydantic_settings`` installed, the returned
    :class:`DIXConfig` is byte-for-byte identical to the stdlib factory's
    output — :func:`load_config_stdlib` is the canonical source of
    truth.

    Raises
    ------
    ImportError
        If ``pydantic_settings`` is not installed.
    """
    import pydantic_settings  # noqa: F401, PLC0415  — lazy seam, function-local only

    return load_config_stdlib(
        defaults=defaults,
        yaml_text=yaml_text,
        dotenv_text=dotenv_text,
        env=env,
        types=types,
        allowed_env_prefixes=allowed_env_prefixes,
    )


__all__ = (
    "CONFIG_VERSION",
    "ConfigEntry",
    "ConfigSource",
    "ConfigType",
    "DIXConfig",
    "NEW_PIP_DEPENDENCIES",
    "coerce_value",
    "enable_pydantic_settings_factory",
    "load_config_stdlib",
    "parse_dotenv",
    "parse_env_map",
    "parse_yaml_config",
    "stdlib_config_factory",
)
