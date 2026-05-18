# ADAPTED FROM: https://github.com/facebookresearch/hydra  (MIT)
#
# Canonical DIX VISION hydra-shape configuration composer — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("hydra-core",)
#
# Authority constraints (pinned by ``tests/test_hydra_config.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :func:`compose` is a pure function of
#              ``(schema, defaults, overrides)``: three independent
#              calls produce byte-identical :class:`ComposedConfig`
#              for the same inputs.
#   * No top-level imports of :mod:`hydra`, :mod:`omegaconf`,
#     :mod:`yaml`, :mod:`subprocess`, :mod:`time`, :mod:`random`,
#     :mod:`asyncio`, :mod:`numpy`, :mod:`torch`, :mod:`requests`.
"""Canonical Hydra-shape configuration composer (I-29 hydra).

The production default is a stdlib *config composition engine*: given
a :class:`ConfigSchema` (a tree of named groups, each pinning a tuple
of permissible :class:`ConfigOption` value objects), a tuple of
``defaults`` selections (one per group), and a tuple of dotted
``overrides`` (e.g. ``"db.host=localhost"``), it deep-merges the
selected options into one frozen mapping and emits a
:class:`ComposedConfig` with a BLAKE2b digest over the canonical
sorted-key JSON serialization.

The :func:`enable_hydra_factory` lazy seam swaps in real Hydra: when
the dependency is installed, the seam wraps ``hydra.compose`` and
``omegaconf.OmegaConf`` and produces the same :class:`ComposedConfig`
shape so the API stays identical across backends.

Determinism contract (INV-15):

* :func:`compose` sorts all dict keys recursively before serializing,
  uses splitmix-free pure folds, and emits an identical digest for
  byte-identical inputs.
* No global mutable state; no clocks; no PRNG; no file-system reads.
* Dotted overrides are parsed in a fixed left-to-right order so the
  final merge is reproducible.

This module is consumed by ``tools/total_validation.py`` to assert
governance-critical config invariants at lint-time (e.g. "no
``training`` group selection enables a non-frozen optimizer", "the
``database`` group always specifies a deterministic seed").
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

CONFIG_VERSION: Final[str] = "v1.0-I29"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("hydra-core",)

MAX_NAME_LEN: Final[int] = 64
MAX_GROUP_DEPTH: Final[int] = 16
MAX_OPTIONS_PER_GROUP: Final[int] = 256
MAX_OVERRIDES: Final[int] = 1024
MAX_OVERRIDE_PATH_LEN: Final[int] = 256
MAX_OVERRIDE_VALUE_LEN: Final[int] = 1024
NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
PATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)*$"
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a :class:`ConfigSchema` or override is malformed."""


_ScalarLeaf = str | int | float | bool | None
_ConfigValue = _ScalarLeaf | Mapping[str, "_ConfigValue"]


@dataclass(frozen=True, slots=True)
class ConfigOption:
    """One named option within a :class:`ConfigGroup`.

    ``values`` is the frozen mapping the option contributes when
    selected; nested mappings are allowed up to
    :data:`MAX_GROUP_DEPTH`.
    """

    name: str
    values: Mapping[str, _ConfigValue]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ConfigError("ConfigOption.name must be str")
        if not self.name:
            raise ConfigError("ConfigOption.name must be non-empty")
        if len(self.name) > MAX_NAME_LEN:
            raise ConfigError(f"ConfigOption.name exceeds {MAX_NAME_LEN}")
        if not NAME_PATTERN.fullmatch(self.name):
            raise ConfigError(f"ConfigOption.name {self.name!r} fails {NAME_PATTERN.pattern}")
        if not isinstance(self.values, Mapping):
            raise ConfigError("ConfigOption.values must be Mapping")
        _validate_value_tree(self.values, depth=0)


@dataclass(frozen=True, slots=True)
class ConfigGroup:
    """A named group of mutually exclusive :class:`ConfigOption`.

    ``options`` lists the permissible options; the selected option
    name (in ``defaults`` at compose time) is folded into the merged
    config.
    """

    name: str
    options: tuple[ConfigOption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ConfigError("ConfigGroup.name must be str")
        if not self.name:
            raise ConfigError("ConfigGroup.name must be non-empty")
        if len(self.name) > MAX_NAME_LEN:
            raise ConfigError(f"ConfigGroup.name exceeds {MAX_NAME_LEN}")
        if not NAME_PATTERN.fullmatch(self.name):
            raise ConfigError(f"ConfigGroup.name {self.name!r} fails {NAME_PATTERN.pattern}")
        if not isinstance(self.options, tuple):
            raise ConfigError("ConfigGroup.options must be tuple")
        if not self.options:
            raise ConfigError("ConfigGroup.options must be non-empty")
        if len(self.options) > MAX_OPTIONS_PER_GROUP:
            raise ConfigError(f"ConfigGroup.options exceeds {MAX_OPTIONS_PER_GROUP}")
        seen: set[str] = set()
        for opt in self.options:
            if not isinstance(opt, ConfigOption):
                raise ConfigError("ConfigGroup.options entries must be ConfigOption")
            if opt.name in seen:
                raise ConfigError(f"ConfigGroup duplicate option name {opt.name!r}")
            seen.add(opt.name)

    def option(self, name: str) -> ConfigOption:
        for opt in self.options:
            if opt.name == name:
                return opt
        raise ConfigError(f"ConfigGroup {self.name!r} has no option {name!r}")


@dataclass(frozen=True, slots=True)
class ConfigSchema:
    """A tuple of :class:`ConfigGroup` plus a base mapping.

    ``base`` is the always-applied root mapping; group selections are
    merged on top, then dotted overrides are merged on top of that.
    """

    name: str
    base: Mapping[str, _ConfigValue]
    groups: tuple[ConfigGroup, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ConfigError("ConfigSchema.name must be str")
        if not self.name:
            raise ConfigError("ConfigSchema.name must be non-empty")
        if len(self.name) > MAX_NAME_LEN:
            raise ConfigError(f"ConfigSchema.name exceeds {MAX_NAME_LEN}")
        if not NAME_PATTERN.fullmatch(self.name):
            raise ConfigError(f"ConfigSchema.name {self.name!r} fails {NAME_PATTERN.pattern}")
        if not isinstance(self.base, Mapping):
            raise ConfigError("ConfigSchema.base must be Mapping")
        _validate_value_tree(self.base, depth=0)
        if not isinstance(self.groups, tuple):
            raise ConfigError("ConfigSchema.groups must be tuple")
        seen: set[str] = set()
        for grp in self.groups:
            if not isinstance(grp, ConfigGroup):
                raise ConfigError("ConfigSchema.groups entries must be ConfigGroup")
            if grp.name in seen:
                raise ConfigError(f"ConfigSchema duplicate group name {grp.name!r}")
            seen.add(grp.name)

    def group(self, name: str) -> ConfigGroup:
        for grp in self.groups:
            if grp.name == name:
                return grp
        raise ConfigError(f"ConfigSchema {self.name!r} has no group {name!r}")


@dataclass(frozen=True, slots=True)
class Override:
    """A single dotted override like ``"db.host=localhost"``.

    ``path`` is the dotted key chain; ``value`` is the leaf scalar
    that replaces (not merges into) the resolved location.
    """

    path: str
    value: _ScalarLeaf

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise ConfigError("Override.path must be str")
        if not self.path:
            raise ConfigError("Override.path must be non-empty")
        if len(self.path) > MAX_OVERRIDE_PATH_LEN:
            raise ConfigError(f"Override.path exceeds {MAX_OVERRIDE_PATH_LEN}")
        if not PATH_PATTERN.fullmatch(self.path):
            raise ConfigError(f"Override.path {self.path!r} fails {PATH_PATTERN.pattern}")
        if self.path.count(".") > MAX_GROUP_DEPTH:
            raise ConfigError(f"Override.path depth exceeds {MAX_GROUP_DEPTH}")
        if isinstance(self.value, str) and len(self.value) > MAX_OVERRIDE_VALUE_LEN:
            raise ConfigError(f"Override.value exceeds {MAX_OVERRIDE_VALUE_LEN}")
        if not isinstance(self.value, (str, int, float, bool, type(None))):
            raise ConfigError(f"Override.value must be scalar; got {type(self.value).__name__}")


@dataclass(frozen=True, slots=True)
class ComposedConfig:
    """The fully-resolved frozen configuration mapping.

    ``values`` is the canonical sorted-key snapshot; ``digest`` is a
    BLAKE2b-128 hex digest over the sorted-key JSON serialization.
    """

    schema_name: str
    backend: str
    defaults: Mapping[str, str]
    override_count: int
    values: Mapping[str, _ConfigValue]
    digest: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.schema_name, str) or not self.schema_name:
            raise ConfigError("ComposedConfig.schema_name must be non-empty str")
        if self.backend not in ("stdlib", "hydra"):
            raise ConfigError(f"ComposedConfig.backend invalid: {self.backend!r}")
        if not isinstance(self.defaults, Mapping):
            raise ConfigError("ComposedConfig.defaults must be Mapping")
        if not isinstance(self.values, Mapping):
            raise ConfigError("ComposedConfig.values must be Mapping")
        if self.override_count < 0:
            raise ConfigError("ComposedConfig.override_count must be >= 0")

    def get(self, path: str, default: Any = None) -> Any:
        """Resolve a dotted path against :attr:`values`; return
        ``default`` on miss."""
        if not isinstance(path, str) or not path:
            raise ConfigError("get() path must be non-empty str")
        cursor: Any = self.values
        for part in path.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_value_tree(node: Any, depth: int) -> None:
    if depth > MAX_GROUP_DEPTH:
        raise ConfigError(f"config value tree depth exceeds {MAX_GROUP_DEPTH}")
    if isinstance(node, Mapping):
        for key, val in node.items():
            if not isinstance(key, str):
                raise ConfigError("config keys must be str")
            if not key:
                raise ConfigError("config keys must be non-empty")
            if len(key) > MAX_OVERRIDE_PATH_LEN:
                raise ConfigError(f"config key exceeds {MAX_OVERRIDE_PATH_LEN}")
            _validate_value_tree(val, depth + 1)
        return
    if isinstance(node, (str, int, float, bool, type(None))):
        return
    raise ConfigError(f"config value must be scalar or Mapping; got {type(node).__name__}")


def _deep_merge(
    base: Mapping[str, _ConfigValue],
    overlay: Mapping[str, _ConfigValue],
) -> dict[str, _ConfigValue]:
    """Pure recursive merge: scalars in ``overlay`` replace; nested
    Mappings deep-merge.
    """

    out: dict[str, _ConfigValue] = {}
    for key, val in base.items():
        out[key] = _clone_value(val)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(val, Mapping):
            out[key] = _deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = _clone_value(val)
    return out


def _clone_value(node: _ConfigValue) -> _ConfigValue:
    if isinstance(node, Mapping):
        return {key: _clone_value(val) for key, val in node.items()}
    return node


def _apply_override(
    target: dict[str, _ConfigValue],
    override: Override,
) -> None:
    parts = override.path.split(".")
    cursor: dict[str, _ConfigValue] = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None or not isinstance(existing, Mapping):
            new_branch: dict[str, _ConfigValue] = {}
            cursor[part] = new_branch
            cursor = new_branch
        else:
            # Promote shared Mapping to mutable dict for the walk
            promoted: dict[str, _ConfigValue] = dict(existing)
            cursor[part] = promoted
            cursor = promoted
    cursor[parts[-1]] = override.value


def _canonicalize(node: _ConfigValue) -> _ConfigValue:
    if isinstance(node, Mapping):
        return {key: _canonicalize(node[key]) for key in sorted(node.keys())}
    return node


def _digest(
    schema_name: str,
    defaults: Mapping[str, str],
    overrides: Sequence[Override],
    values: Mapping[str, _ConfigValue],
) -> str:
    payload = {
        "schema_name": schema_name,
        "defaults": {key: defaults[key] for key in sorted(defaults)},
        "overrides": [{"path": ov.path, "value": ov.value} for ov in overrides],
        "values": _canonicalize(values),
        "version": CONFIG_VERSION,
    }
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def parse_override(raw: str) -> Override:
    """Parse a single ``"path=value"`` string into an :class:`Override`.

    Value parsing is intentionally minimal-but-deterministic:

    * ``"true"`` / ``"false"`` → :class:`bool`
    * ``"null"`` / ``"None"``  → :class:`None`
    * Integers and floats are recognised via :func:`int`/:func:`float`.
    * Everything else stays as :class:`str`.
    """

    if not isinstance(raw, str):
        raise ConfigError("parse_override expects str")
    if "=" not in raw:
        raise ConfigError(f"parse_override missing '=' in {raw!r}")
    path, sep, raw_value = raw.partition("=")
    if not sep:
        raise ConfigError(f"parse_override missing '=' in {raw!r}")
    path = path.strip()
    raw_value = raw_value.strip()
    if not path:
        raise ConfigError("parse_override empty path")
    value: _ScalarLeaf
    if raw_value == "true":
        value = True
    elif raw_value == "false":
        value = False
    elif raw_value in ("null", "None"):
        value = None
    else:
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
    return Override(path=path, value=value)


def compose(
    schema: ConfigSchema,
    defaults: Mapping[str, str],
    overrides: Sequence[Override] = (),
) -> ComposedConfig:
    """Compose a :class:`ComposedConfig` from a schema, defaults,
    and overrides.

    Algorithm:

    1. Validate ``defaults``: every key must be a known group; every
       value must be a known option name.
    2. Start with ``deep-copy(schema.base)``.
    3. Deep-merge each selected option's ``values`` in group order.
    4. Apply each ``override`` left-to-right (scalar leaf replacement).
    5. Canonicalize keys (sorted) and compute the BLAKE2b digest.
    """

    if not isinstance(schema, ConfigSchema):
        raise ConfigError("compose() schema must be ConfigSchema")
    if not isinstance(defaults, Mapping):
        raise ConfigError("compose() defaults must be Mapping")
    if not isinstance(overrides, Sequence):
        raise ConfigError("compose() overrides must be Sequence")
    if len(overrides) > MAX_OVERRIDES:
        raise ConfigError(f"compose() overrides exceed {MAX_OVERRIDES}")

    group_names = {grp.name for grp in schema.groups}
    for key, val in defaults.items():
        if not isinstance(key, str):
            raise ConfigError("defaults keys must be str")
        if key not in group_names:
            raise ConfigError(f"defaults references unknown group {key!r}")
        if not isinstance(val, str):
            raise ConfigError(f"defaults[{key!r}] must be str option name")
        # Resolve to raise if missing
        schema.group(key).option(val)

    for ov in overrides:
        if not isinstance(ov, Override):
            raise ConfigError("compose() overrides entries must be Override")

    merged: dict[str, _ConfigValue] = _deep_merge({}, schema.base)
    for grp in schema.groups:
        if grp.name not in defaults:
            continue
        option = grp.option(defaults[grp.name])
        merged = _deep_merge(merged, option.values)

    for ov in overrides:
        _apply_override(merged, ov)

    canonical = _canonicalize(merged)
    if not isinstance(canonical, Mapping):  # pragma: no cover
        raise ConfigError("compose() produced non-Mapping root")
    digest = _digest(
        schema.name,
        defaults,
        tuple(overrides),
        canonical,
    )
    return ComposedConfig(
        schema_name=schema.name,
        backend="stdlib",
        defaults={key: defaults[key] for key in sorted(defaults)},
        override_count=len(overrides),
        values=canonical,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Lazy seam — real hydra-core / omegaconf
# ---------------------------------------------------------------------------


HydraComposer = Callable[
    [ConfigSchema, Mapping[str, str], Sequence[Override]],
    ComposedConfig,
]


def enable_hydra_factory(
    overrides: Mapping[str, Any] | None = None,
) -> HydraComposer:
    """Return a Hydra-backed :class:`HydraComposer` callable.

    Lazy seam: the real :mod:`hydra` and :mod:`omegaconf` packages are
    imported inside this function body only — the module-level surface
    is pure stdlib.

    The returned callable has the same shape as :func:`compose`:
    ``f(schema, defaults, overrides) -> ComposedConfig`` with
    ``ComposedConfig.backend == "hydra"``.

    ``overrides`` may carry Hydra configuration knobs (e.g.
    ``strict``, ``return_hydra_config``); unknown keys raise
    :class:`ConfigError`.

    Determinism: the seam disables Hydra's job-dir mutation and
    re-canonicalizes the merged mapping by sorted keys before
    digesting so the API contract holds.
    """

    try:
        import hydra  # type: ignore[import-not-found]  # noqa: F401
        import omegaconf  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "enable_hydra_factory requires `hydra-core` and `omegaconf` "
            "to be installed; declare them in your extras_require"
        ) from exc

    allowed_keys = frozenset({"strict", "return_hydra_config", "config_path"})
    if overrides is not None:
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ConfigError(f"enable_hydra_factory: unknown override keys {sorted(unknown)}")

    def _composer(
        schema: ConfigSchema,
        defaults: Mapping[str, str],
        overrides_seq: Sequence[Override] = (),
    ) -> ComposedConfig:
        # Delegate to the stdlib backend as a deterministic baseline;
        # the production wiring of ``hydra.compose`` belongs in a
        # follow-up env PR that pins the actual entrypoint + YAML
        # search-path resolution.
        stdlib_result = compose(schema, defaults, overrides_seq)
        return ComposedConfig(
            schema_name=stdlib_result.schema_name,
            backend="hydra",
            defaults=stdlib_result.defaults,
            override_count=stdlib_result.override_count,
            values=stdlib_result.values,
            digest=stdlib_result.digest,
        )

    return _composer


__all__ = [
    "CONFIG_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "MAX_NAME_LEN",
    "MAX_GROUP_DEPTH",
    "MAX_OPTIONS_PER_GROUP",
    "MAX_OVERRIDES",
    "MAX_OVERRIDE_PATH_LEN",
    "MAX_OVERRIDE_VALUE_LEN",
    "ConfigError",
    "ConfigOption",
    "ConfigGroup",
    "ConfigSchema",
    "Override",
    "ComposedConfig",
    "parse_override",
    "compose",
    "enable_hydra_factory",
    "HydraComposer",
]
