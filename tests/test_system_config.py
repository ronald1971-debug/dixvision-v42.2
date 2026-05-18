"""Tests for I-28 ``system_engine/config.py`` — canonical pydantic-settings adapter.

Covers:

  * Module surface (exports, ``NEW_PIP_DEPENDENCIES`` shape, canonical defaults).
  * ``ConfigSource`` / ``ConfigEntry`` / ``DIXConfig`` frozen+slotted + validation.
  * Pure parsers (``parse_dotenv`` / ``parse_env_map`` / ``parse_yaml_config``).
  * ``coerce_value`` for str / int / float / bool.
  * ``load_config_stdlib`` precedence (defaults < yaml < dotenv < env).
  * Forbidden-secret-key rejection at every tier.
  * ``stdlib_config_factory`` always available.
  * ``enable_pydantic_settings_factory`` lazy seam (skip if missing).
  * INV-15 byte-identical 3-run replay (BLAKE2b-16 digest equality).
  * AST guardrails — no top-level forbidden imports, ``pydantic_settings``
    only in seam, no typed-event ctors (B27/B28/INV-71), no B1 runtime-tier
    imports, no wall-clock reads.
"""

from __future__ import annotations

import ast
import importlib.util
from hashlib import blake2b
from pathlib import Path

import pytest

import system_engine.config as cfg


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------
def test_module_exports() -> None:
    expected = {
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
    }
    assert expected.issubset(set(dir(cfg)))
    assert set(cfg.__all__) == expected


def test_new_pip_dependencies_shape() -> None:
    assert isinstance(cfg.NEW_PIP_DEPENDENCIES, tuple)
    assert cfg.NEW_PIP_DEPENDENCIES == ("pydantic-settings",)


def test_config_version_pinned() -> None:
    assert cfg.CONFIG_VERSION == "v1.0-I28"


def test_config_source_enum_values() -> None:
    assert cfg.ConfigSource.DEFAULT.value == "DEFAULT"
    assert cfg.ConfigSource.YAML.value == "YAML"
    assert cfg.ConfigSource.DOTENV.value == "DOTENV"
    assert cfg.ConfigSource.ENV.value == "ENV"


# ---------------------------------------------------------------------------
# ConfigEntry — frozen + slotted + validation
# ---------------------------------------------------------------------------
def test_config_entry_frozen_slotted() -> None:
    e = cfg.ConfigEntry(key="dix_mode", value="paper", source=cfg.ConfigSource.DEFAULT)
    with pytest.raises((AttributeError, Exception)):
        e.key = "other"  # type: ignore[misc]
    assert "__slots__" in type(e).__dict__


def test_config_entry_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cfg.ConfigEntry(key="", value="x", source=cfg.ConfigSource.DEFAULT)


def test_config_entry_rejects_whitespace_key() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        cfg.ConfigEntry(key="dix mode", value="x", source=cfg.ConfigSource.DEFAULT)


def test_config_entry_rejects_bad_source_type() -> None:
    with pytest.raises(TypeError, match="ConfigSource"):
        cfg.ConfigEntry(key="dix_mode", value="x", source="DEFAULT")  # type: ignore[arg-type]


def test_config_entry_rejects_bad_value_type() -> None:
    with pytest.raises(TypeError, match="not allowed"):
        cfg.ConfigEntry(key="dix_mode", value=[1, 2], source=cfg.ConfigSource.DEFAULT)  # type: ignore[arg-type]


def test_config_entry_accepts_none_value() -> None:
    e = cfg.ConfigEntry(key="dix_mode", value=None, source=cfg.ConfigSource.DEFAULT)
    assert e.value is None


# ---------------------------------------------------------------------------
# DIXConfig — frozen + slotted + validation
# ---------------------------------------------------------------------------
def _entry(k: str, v: str, src: cfg.ConfigSource = cfg.ConfigSource.DEFAULT) -> cfg.ConfigEntry:
    return cfg.ConfigEntry(key=k, value=v, source=src)


def test_dix_config_frozen_slotted() -> None:
    c = cfg.DIXConfig(entries=(_entry("a", "1"),))
    with pytest.raises(AttributeError):
        c.entries = ()  # type: ignore[misc]
    assert "__slots__" in type(c).__dict__


def test_dix_config_rejects_unsorted() -> None:
    with pytest.raises(ValueError, match="sorted"):
        cfg.DIXConfig(entries=(_entry("b", "1"), _entry("a", "2")))


def test_dix_config_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        cfg.DIXConfig(entries=(_entry("a", "1"), _entry("a", "2")))


def test_dix_config_rejects_non_tuple_entries() -> None:
    with pytest.raises(TypeError, match="tuple"):
        cfg.DIXConfig(entries=[_entry("a", "1")])  # type: ignore[arg-type]


def test_dix_config_get_returns_value() -> None:
    c = cfg.DIXConfig(entries=(_entry("a", "1"), _entry("b", "2")))
    assert c.get("a") == "1"
    assert c.get("b") == "2"
    assert c.get("missing", "fallback") == "fallback"


def test_dix_config_source_of() -> None:
    c = cfg.DIXConfig(entries=(_entry("a", "1", cfg.ConfigSource.ENV),))
    assert c.source_of("a") == cfg.ConfigSource.ENV
    assert c.source_of("missing") is None


def test_dix_config_as_mapping_sorted() -> None:
    c = cfg.DIXConfig(entries=(_entry("a", "1"), _entry("b", "2")))
    assert list(c.as_mapping().items()) == [("a", "1"), ("b", "2")]


# ---------------------------------------------------------------------------
# parse_dotenv
# ---------------------------------------------------------------------------
def test_parse_dotenv_basic() -> None:
    text = "DIX_MODE=paper\nDIX_TIER=A\n"
    assert cfg.parse_dotenv(text) == {"DIX_MODE": "paper", "DIX_TIER": "A"}


def test_parse_dotenv_skips_comments_and_blank() -> None:
    text = "# top comment\n\nDIX_MODE=paper\n  # indented comment\n"
    assert cfg.parse_dotenv(text) == {"DIX_MODE": "paper"}


def test_parse_dotenv_strips_double_quotes() -> None:
    assert cfg.parse_dotenv('DIX_MODE="paper"\n') == {"DIX_MODE": "paper"}


def test_parse_dotenv_strips_single_quotes() -> None:
    assert cfg.parse_dotenv("DIX_MODE='paper'\n") == {"DIX_MODE": "paper"}


def test_parse_dotenv_rejects_no_separator() -> None:
    with pytest.raises(ValueError, match="separator"):
        cfg.parse_dotenv("DIX_MODE_paper\n")


def test_parse_dotenv_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="empty key"):
        cfg.parse_dotenv("=paper\n")


def test_parse_dotenv_rejects_whitespace_key() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        cfg.parse_dotenv("DIX MODE=paper\n")


def test_parse_dotenv_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        cfg.parse_dotenv(b"x=y")  # type: ignore[arg-type]


def test_parse_dotenv_empty_text() -> None:
    assert cfg.parse_dotenv("") == {}


# ---------------------------------------------------------------------------
# parse_env_map
# ---------------------------------------------------------------------------
def test_parse_env_map_filters_by_prefix() -> None:
    env = {"DIX_MODE": "paper", "PATH": "/usr/bin", "DIX_TIER": "A"}
    out = cfg.parse_env_map(env)
    assert out == {"DIX_MODE": "paper", "DIX_TIER": "A"}


def test_parse_env_map_multiple_prefixes() -> None:
    env = {"DIX_MODE": "a", "FOO_X": "b", "BAR_Y": "c"}
    out = cfg.parse_env_map(env, allowed_prefixes=("DIX_", "FOO_"))
    assert out == {"DIX_MODE": "a", "FOO_X": "b"}


def test_parse_env_map_rejects_empty_prefixes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cfg.parse_env_map({"X": "y"}, allowed_prefixes=())


def test_parse_env_map_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        cfg.parse_env_map("DIX_MODE=paper")  # type: ignore[arg-type]


def test_parse_env_map_drops_non_string_values() -> None:
    env: dict[str, object] = {"DIX_MODE": 42}  # type: ignore[assignment]
    out = cfg.parse_env_map(env)  # type: ignore[arg-type]
    assert out == {}


# ---------------------------------------------------------------------------
# parse_yaml_config
# ---------------------------------------------------------------------------
def test_parse_yaml_config_flat() -> None:
    text = "dix_mode: paper\ndix_tier: A\n"
    assert cfg.parse_yaml_config(text) == {"dix_mode": "paper", "dix_tier": "A"}


def test_parse_yaml_config_nested_flattened() -> None:
    text = "engines:\n  intelligence:\n    tier: RUNTIME\n"
    assert cfg.parse_yaml_config(text) == {"engines.intelligence.tier": "RUNTIME"}


def test_parse_yaml_config_drops_lists() -> None:
    text = "items:\n  - a\n  - b\nmode: paper\n"
    assert cfg.parse_yaml_config(text) == {"mode": "paper"}


def test_parse_yaml_config_empty_text() -> None:
    assert cfg.parse_yaml_config("") == {}
    assert cfg.parse_yaml_config("   ") == {}


def test_parse_yaml_config_rejects_non_mapping_root() -> None:
    with pytest.raises(ValueError, match="mapping"):
        cfg.parse_yaml_config("- a\n- b\n")


def test_parse_yaml_config_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        cfg.parse_yaml_config(b"x: y")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# coerce_value
# ---------------------------------------------------------------------------
def test_coerce_value_none_passthrough() -> None:
    assert cfg.coerce_value(None, str, key="k") is None


def test_coerce_value_str_target() -> None:
    assert cfg.coerce_value(42, str, key="k") == "42"
    assert cfg.coerce_value("paper", str, key="k") == "paper"


def test_coerce_value_int_target() -> None:
    assert cfg.coerce_value("42", int, key="k") == 42
    assert cfg.coerce_value(42, int, key="k") == 42


def test_coerce_value_int_rejects_bool() -> None:
    with pytest.raises(ValueError, match="bool to int"):
        cfg.coerce_value(True, int, key="k")


def test_coerce_value_int_rejects_fractional_float() -> None:
    with pytest.raises(ValueError, match="fractional"):
        cfg.coerce_value(1.5, int, key="k")


def test_coerce_value_float_target() -> None:
    assert cfg.coerce_value("1.5", float, key="k") == 1.5
    assert cfg.coerce_value(2, float, key="k") == 2.0


def test_coerce_value_float_rejects_bool() -> None:
    with pytest.raises(ValueError, match="bool to float"):
        cfg.coerce_value(True, float, key="k")


def test_coerce_value_bool_target_true_forms() -> None:
    for v in ("true", "True", "1", "yes", "on", True, 1):
        assert cfg.coerce_value(v, bool, key="k") is True


def test_coerce_value_bool_target_false_forms() -> None:
    for v in ("false", "False", "0", "no", "off", False, 0):
        assert cfg.coerce_value(v, bool, key="k") is False


def test_coerce_value_bool_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        cfg.coerce_value("maybe", bool, key="k")


def test_coerce_value_rejects_unsupported_target() -> None:
    with pytest.raises(TypeError, match="unsupported target"):
        cfg.coerce_value("x", list, key="k")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_config_stdlib — precedence
# ---------------------------------------------------------------------------
def test_load_config_stdlib_defaults_only() -> None:
    c = cfg.load_config_stdlib(defaults={"a": "1", "b": "2"})
    assert c.get("a") == "1"
    assert c.source_of("a") == cfg.ConfigSource.DEFAULT


def test_load_config_stdlib_yaml_overrides_defaults() -> None:
    c = cfg.load_config_stdlib(
        defaults={"a": "1"},
        yaml_text="a: 2\n",
    )
    assert c.get("a") == "2"
    assert c.source_of("a") == cfg.ConfigSource.YAML


def test_load_config_stdlib_dotenv_overrides_yaml() -> None:
    c = cfg.load_config_stdlib(
        defaults={"a": "1"},
        yaml_text="a: 2\n",
        dotenv_text="a=3\n",
    )
    assert c.get("a") == "3"
    assert c.source_of("a") == cfg.ConfigSource.DOTENV


def test_load_config_stdlib_env_overrides_dotenv() -> None:
    c = cfg.load_config_stdlib(
        defaults={"a": "1"},
        yaml_text="a: 2\n",
        dotenv_text="a=3\n",
        env={"DIX_a": "4"},
        allowed_env_prefixes=("DIX_",),
    )
    assert c.get("DIX_a") == "4"
    assert c.source_of("DIX_a") == cfg.ConfigSource.ENV


def test_load_config_stdlib_entries_sorted() -> None:
    c = cfg.load_config_stdlib(defaults={"z": "1", "a": "2", "m": "3"})
    keys = [e.key for e in c.entries]
    assert keys == sorted(keys)


def test_load_config_stdlib_with_types() -> None:
    c = cfg.load_config_stdlib(
        defaults={"max_workers": "8", "enabled": "true", "rate": "1.5"},
        types={"max_workers": int, "enabled": bool, "rate": float},
    )
    assert c.get("max_workers") == 8
    assert c.get("enabled") is True
    assert c.get("rate") == 1.5


def test_load_config_stdlib_empty_returns_empty_config() -> None:
    c = cfg.load_config_stdlib()
    assert c.entries == ()


# ---------------------------------------------------------------------------
# Forbidden-secret-key rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "openai_api_key",
        "secret",
        "client_secret",
        "password",
        "db_password",
        "private_key",
        "auth_token",
        "session_key",
        "auth_header",
        "credential",
        "API_KEY",
        "Secret_X",
    ],
)
def test_load_config_stdlib_rejects_forbidden_in_defaults(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden secret"):
        cfg.load_config_stdlib(defaults={key: "x"})


def test_load_config_stdlib_rejects_forbidden_in_yaml() -> None:
    with pytest.raises(ValueError, match="forbidden secret"):
        cfg.load_config_stdlib(yaml_text="api_key: x\n")


def test_load_config_stdlib_rejects_forbidden_in_dotenv() -> None:
    with pytest.raises(ValueError, match="forbidden secret"):
        cfg.load_config_stdlib(dotenv_text="API_KEY=x\n")


def test_load_config_stdlib_rejects_forbidden_in_env() -> None:
    with pytest.raises(ValueError, match="forbidden secret"):
        cfg.load_config_stdlib(env={"DIX_API_KEY": "x"})


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def test_stdlib_config_factory_always_available() -> None:
    c = cfg.stdlib_config_factory(defaults={"a": "1"})
    assert isinstance(c, cfg.DIXConfig)
    assert c.get("a") == "1"


def test_enable_pydantic_settings_factory_skip_if_missing() -> None:
    if importlib.util.find_spec("pydantic_settings") is None:
        with pytest.raises(ImportError):
            cfg.enable_pydantic_settings_factory(defaults={"a": "1"})
        pytest.skip("pydantic_settings not installed — seam skip path verified")
    c = cfg.enable_pydantic_settings_factory(defaults={"a": "1"})
    assert isinstance(c, cfg.DIXConfig)
    assert c.get("a") == "1"


def test_enable_pydantic_settings_factory_matches_stdlib_when_present() -> None:
    if importlib.util.find_spec("pydantic_settings") is None:
        pytest.skip("pydantic_settings not installed")
    defaults = {"a": "1", "b": "2"}
    c_stdlib = cfg.stdlib_config_factory(defaults=defaults)
    c_seam = cfg.enable_pydantic_settings_factory(defaults=defaults)
    assert c_stdlib.entries == c_seam.entries


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------
def _build_canonical_inputs() -> dict[str, object]:
    return {
        "defaults": {
            "dix_mode": "paper",
            "max_workers": "4",
            "rate": "1.0",
            "enabled": "false",
        },
        "yaml_text": (
            "dix_mode: paper\n"
            "engines:\n"
            "  intelligence:\n"
            "    tier: RUNTIME\n"
            "  system:\n"
            "    tier: RUNTIME\n"
        ),
        "dotenv_text": "DIX_LEDGER_PATH=/var/lib/dix/ledger.db\nmax_workers=8\n",
        "env": {"DIX_LEDGER_PATH": "/tmp/ledger.db", "DIX_TIER": "A"},
        "types": {"max_workers": int, "rate": float, "enabled": bool},
    }


def _digest(c: cfg.DIXConfig) -> str:
    raw = "\n".join(f"{e.key}\x1e{e.value!r}\x1e{e.source.value}" for e in c.entries).encode(
        "utf-8"
    )
    return blake2b(raw, digest_size=16).hexdigest()


def test_inv15_byte_identical_three_run_replay() -> None:
    inputs = _build_canonical_inputs()
    digests: set[str] = set()
    for _ in range(3):
        c = cfg.load_config_stdlib(**inputs)  # type: ignore[arg-type]
        digests.add(_digest(c))
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# AST guardrails — top-level forbidden imports / typed events / B1 / wall-clock
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(cfg.__file__)
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_SOURCE, filename=str(_MODULE_PATH))


def _top_level_imports() -> set[str]:
    out: set[str] = set()
    for node in _MODULE_AST.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def test_ast_no_forbidden_top_level_imports() -> None:
    forbidden = {
        "pydantic_settings",
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "requests",
    }
    top = _top_level_imports()
    overlap = top & forbidden
    assert not overlap, f"Forbidden top-level imports present: {overlap}"


def test_ast_pydantic_settings_only_in_lazy_seam() -> None:
    # Find every `import pydantic_settings` / `from pydantic_settings…`
    # node in the AST.  Each must be nested inside a FunctionDef body
    # whose name is `enable_pydantic_settings_factory`.
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            if "pydantic_settings" not in names:
                continue
            # Walk back to enclosing FunctionDef.
            enclosing = _find_enclosing_function(_MODULE_AST, node)
            assert enclosing is not None, "pydantic_settings imported outside a function"
            assert enclosing.name == "enable_pydantic_settings_factory", (
                f"pydantic_settings imported inside {enclosing.name!r}, "
                "must be inside enable_pydantic_settings_factory only"
            )


def _find_enclosing_function(root: ast.AST, target: ast.AST) -> ast.FunctionDef | None:
    for fn in ast.walk(root):
        if isinstance(fn, ast.FunctionDef):
            for child in ast.walk(fn):
                if child is target:
                    return fn
    return None


def test_ast_no_typed_event_constructors() -> None:
    forbidden_ctors = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "LearningUpdate",
    }
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Call):
            target = node.func
            name: str | None = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name and name in forbidden_ctors:
                pytest.fail(
                    f"Forbidden typed-event constructor {name!r} called in system_engine/config.py"
                )


def test_ast_no_runtime_tier_imports() -> None:
    forbidden_tiers = {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    }
    for node in _MODULE_AST.body:
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module.split(".")[0]]
        for m in modules:
            assert m not in forbidden_tiers, (
                f"B1 violation: system_engine/config.py imports runtime tier {m!r}"
            )


def test_ast_no_wall_clock_reads() -> None:
    forbidden_calls = {
        ("time", "time"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("time", "time_ns"),
        ("time", "perf_counter"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
    }
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                pair = (value.id, node.func.attr)
                assert pair not in forbidden_calls, (
                    f"Wall-clock read {value.id}.{node.func.attr}() detected"
                )
