"""I-29 — tests for the canonical hydra-shape configuration composer."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tools.hydra_config import (
    CONFIG_VERSION,
    MAX_GROUP_DEPTH,
    MAX_NAME_LEN,
    MAX_OPTIONS_PER_GROUP,
    MAX_OVERRIDE_PATH_LEN,
    MAX_OVERRIDE_VALUE_LEN,
    MAX_OVERRIDES,
    NEW_PIP_DEPENDENCIES,
    ComposedConfig,
    ConfigError,
    ConfigGroup,
    ConfigOption,
    ConfigSchema,
    Override,
    compose,
    enable_hydra_factory,
    parse_override,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_config_version_is_pinned() -> None:
    assert CONFIG_VERSION == "v1.0-I29"


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("hydra-core",)


def test_max_lengths_are_pinned() -> None:
    assert MAX_NAME_LEN == 64
    assert MAX_GROUP_DEPTH == 16
    assert MAX_OPTIONS_PER_GROUP == 256
    assert MAX_OVERRIDES == 1024
    assert MAX_OVERRIDE_PATH_LEN == 256
    assert MAX_OVERRIDE_VALUE_LEN == 1024


# ---------------------------------------------------------------------------
# ConfigOption
# ---------------------------------------------------------------------------


def test_config_option_construction_happy_path() -> None:
    opt = ConfigOption(name="postgres", values={"host": "db.local"})
    assert opt.name == "postgres"
    assert opt.values == {"host": "db.local"}


def test_config_option_is_frozen_and_slotted() -> None:
    opt = ConfigOption(name="postgres", values={"port": 5432})
    with pytest.raises(dataclasses.FrozenInstanceError):
        opt.name = "x"  # type: ignore[misc]
    assert not hasattr(opt, "__dict__")


def test_config_option_name_must_be_str() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name=123, values={})  # type: ignore[arg-type]


def test_config_option_name_non_empty() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name="", values={})


def test_config_option_name_pattern() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name="1bad", values={})


def test_config_option_name_length() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name="A" * 65, values={})


def test_config_option_values_must_be_mapping() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name="x", values=[("a", 1)])  # type: ignore[arg-type]


def test_config_option_values_scalar_only_leaves() -> None:
    with pytest.raises(ConfigError):
        ConfigOption(name="x", values={"a": object()})  # type: ignore[dict-item]


def test_config_option_nested_mapping_ok() -> None:
    opt = ConfigOption(name="x", values={"a": {"b": {"c": 1}}})
    assert opt.values["a"]["b"]["c"] == 1


# ---------------------------------------------------------------------------
# ConfigGroup
# ---------------------------------------------------------------------------


def _basic_options() -> tuple[ConfigOption, ...]:
    return (
        ConfigOption(name="postgres", values={"host": "db.pg"}),
        ConfigOption(name="sqlite", values={"host": "db.sqlite"}),
    )


def test_config_group_happy_path() -> None:
    grp = ConfigGroup(name="db", options=_basic_options())
    assert grp.name == "db"
    assert len(grp.options) == 2


def test_config_group_options_must_be_tuple() -> None:
    with pytest.raises(ConfigError):
        ConfigGroup(name="db", options=list(_basic_options()))  # type: ignore[arg-type]


def test_config_group_options_non_empty() -> None:
    with pytest.raises(ConfigError):
        ConfigGroup(name="db", options=())


def test_config_group_options_duplicate_name() -> None:
    with pytest.raises(ConfigError):
        ConfigGroup(
            name="db",
            options=(
                ConfigOption(name="pg", values={}),
                ConfigOption(name="pg", values={}),
            ),
        )


def test_config_group_option_resolve() -> None:
    grp = ConfigGroup(name="db", options=_basic_options())
    assert grp.option("postgres").values == {"host": "db.pg"}


def test_config_group_option_missing() -> None:
    grp = ConfigGroup(name="db", options=_basic_options())
    with pytest.raises(ConfigError):
        grp.option("oracle")


def test_config_group_name_must_be_str() -> None:
    with pytest.raises(ConfigError):
        ConfigGroup(name=1, options=_basic_options())  # type: ignore[arg-type]


def test_config_group_name_pattern() -> None:
    with pytest.raises(ConfigError):
        ConfigGroup(name="_bad", options=_basic_options())


# ---------------------------------------------------------------------------
# ConfigSchema
# ---------------------------------------------------------------------------


def _basic_schema() -> ConfigSchema:
    return ConfigSchema(
        name="App",
        base={"name": "dixvision", "version": "v42.2"},
        groups=(
            ConfigGroup(name="db", options=_basic_options()),
            ConfigGroup(
                name="log",
                options=(
                    ConfigOption(name="text", values={"format": "text"}),
                    ConfigOption(name="json", values={"format": "json"}),
                ),
            ),
        ),
    )


def test_config_schema_happy_path() -> None:
    schema = _basic_schema()
    assert schema.name == "App"
    assert schema.base["name"] == "dixvision"
    assert len(schema.groups) == 2


def test_config_schema_duplicate_group() -> None:
    with pytest.raises(ConfigError):
        ConfigSchema(
            name="App",
            base={},
            groups=(
                ConfigGroup(name="db", options=_basic_options()),
                ConfigGroup(name="db", options=_basic_options()),
            ),
        )


def test_config_schema_groups_must_be_tuple() -> None:
    with pytest.raises(ConfigError):
        ConfigSchema(
            name="App",
            base={},
            groups=[ConfigGroup(name="db", options=_basic_options())],  # type: ignore[arg-type]
        )


def test_config_schema_group_resolve() -> None:
    schema = _basic_schema()
    assert schema.group("log").name == "log"


def test_config_schema_group_missing() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        schema.group("cache")


def test_config_schema_base_must_be_mapping() -> None:
    with pytest.raises(ConfigError):
        ConfigSchema(name="App", base=[], groups=())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


def test_override_happy_path() -> None:
    ov = Override(path="db.host", value="localhost")
    assert ov.path == "db.host"
    assert ov.value == "localhost"


def test_override_frozen_slotted() -> None:
    ov = Override(path="db.host", value="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ov.path = "y"  # type: ignore[misc]
    assert not hasattr(ov, "__dict__")


def test_override_path_pattern() -> None:
    with pytest.raises(ConfigError):
        Override(path="1bad.foo", value=1)


def test_override_path_empty() -> None:
    with pytest.raises(ConfigError):
        Override(path="", value=1)


def test_override_path_too_long() -> None:
    with pytest.raises(ConfigError):
        Override(path="a" * 257, value=1)


def test_override_depth_cap() -> None:
    deep = ".".join(["a"] * 18)
    with pytest.raises(ConfigError):
        Override(path=deep, value=1)


def test_override_value_must_be_scalar() -> None:
    with pytest.raises(ConfigError):
        Override(path="db.host", value=[1])  # type: ignore[arg-type]


def test_override_value_string_length() -> None:
    with pytest.raises(ConfigError):
        Override(path="db.host", value="x" * 1025)


# ---------------------------------------------------------------------------
# parse_override
# ---------------------------------------------------------------------------


def test_parse_override_string() -> None:
    ov = parse_override("db.host=localhost")
    assert ov.path == "db.host"
    assert ov.value == "localhost"


def test_parse_override_int() -> None:
    ov = parse_override("db.port=5432")
    assert ov.value == 5432


def test_parse_override_float() -> None:
    ov = parse_override("training.lr=0.01")
    assert ov.value == pytest.approx(0.01)


def test_parse_override_true() -> None:
    ov = parse_override("flags.enable=true")
    assert ov.value is True


def test_parse_override_false() -> None:
    ov = parse_override("flags.enable=false")
    assert ov.value is False


def test_parse_override_null() -> None:
    ov = parse_override("opt.host=null")
    assert ov.value is None


def test_parse_override_none() -> None:
    ov = parse_override("opt.host=None")
    assert ov.value is None


def test_parse_override_missing_equals() -> None:
    with pytest.raises(ConfigError):
        parse_override("db.host")


def test_parse_override_empty_path() -> None:
    with pytest.raises(ConfigError):
        parse_override("=value")


def test_parse_override_non_str() -> None:
    with pytest.raises(ConfigError):
        parse_override(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compose() happy paths
# ---------------------------------------------------------------------------


def test_compose_no_groups_returns_base() -> None:
    schema = ConfigSchema(name="App", base={"x": 1}, groups=())
    result = compose(schema, defaults={})
    assert result.values == {"x": 1}
    assert result.backend == "stdlib"
    assert result.override_count == 0


def test_compose_selects_group_default() -> None:
    schema = _basic_schema()
    result = compose(schema, defaults={"db": "postgres", "log": "json"})
    assert result.values["host"] == "db.pg"
    assert result.values["format"] == "json"
    assert result.values["name"] == "dixvision"


def test_compose_apply_override() -> None:
    schema = _basic_schema()
    result = compose(
        schema,
        defaults={"db": "postgres"},
        overrides=(Override(path="host", value="db.override"),),
    )
    assert result.values["host"] == "db.override"
    assert result.override_count == 1


def test_compose_apply_nested_override_creates_branch() -> None:
    schema = ConfigSchema(name="App", base={}, groups=())
    result = compose(
        schema,
        defaults={},
        overrides=(
            Override(path="db.host", value="x"),
            Override(path="db.port", value=5432),
        ),
    )
    assert result.values == {"db": {"host": "x", "port": 5432}}


def test_compose_override_last_wins() -> None:
    schema = ConfigSchema(name="App", base={"host": "a"}, groups=())
    result = compose(
        schema,
        defaults={},
        overrides=(
            Override(path="host", value="b"),
            Override(path="host", value="c"),
        ),
    )
    assert result.values["host"] == "c"


def test_compose_deep_merge_preserves_unrelated_keys() -> None:
    schema = ConfigSchema(
        name="App",
        base={"db": {"host": "base", "pool": 5}},
        groups=(
            ConfigGroup(
                name="db",
                options=(ConfigOption(name="prod", values={"db": {"host": "prod"}}),),
            ),
        ),
    )
    result = compose(schema, defaults={"db": "prod"})
    db_section = result.values["db"]
    assert isinstance(db_section, dict)
    assert db_section["host"] == "prod"
    assert db_section["pool"] == 5


def test_compose_get_returns_nested_value() -> None:
    schema = ConfigSchema(name="App", base={"a": {"b": {"c": 42}}}, groups=())
    result = compose(schema, defaults={})
    assert result.get("a.b.c") == 42


def test_compose_get_returns_default_on_miss() -> None:
    schema = ConfigSchema(name="App", base={"a": 1}, groups=())
    result = compose(schema, defaults={})
    assert result.get("missing.path", default="fallback") == "fallback"


def test_compose_canonicalizes_sorted_keys() -> None:
    schema = ConfigSchema(name="App", base={"z": 1, "a": 2}, groups=())
    result = compose(schema, defaults={})
    assert list(result.values.keys()) == ["a", "z"]


def test_compose_determinism_three_runs() -> None:
    schema = _basic_schema()
    r1 = compose(
        schema,
        defaults={"db": "postgres", "log": "json"},
        overrides=(
            Override(path="extra.x", value=1),
            Override(path="extra.y", value="two"),
        ),
    )
    r2 = compose(
        schema,
        defaults={"db": "postgres", "log": "json"},
        overrides=(
            Override(path="extra.x", value=1),
            Override(path="extra.y", value="two"),
        ),
    )
    r3 = compose(
        schema,
        defaults={"db": "postgres", "log": "json"},
        overrides=(
            Override(path="extra.x", value=1),
            Override(path="extra.y", value="two"),
        ),
    )
    assert r1.digest == r2.digest == r3.digest
    assert r1.values == r2.values == r3.values


def test_compose_digest_changes_with_override() -> None:
    schema = ConfigSchema(name="App", base={}, groups=())
    r1 = compose(schema, defaults={})
    r2 = compose(
        schema,
        defaults={},
        overrides=(Override(path="x", value=1),),
    )
    assert r1.digest != r2.digest


# ---------------------------------------------------------------------------
# compose() error paths
# ---------------------------------------------------------------------------


def test_compose_schema_type() -> None:
    with pytest.raises(ConfigError):
        compose("not-a-schema", defaults={})  # type: ignore[arg-type]


def test_compose_defaults_type() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(schema, defaults=[("db", "postgres")])  # type: ignore[arg-type]


def test_compose_overrides_type() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(schema, defaults={}, overrides=42)  # type: ignore[arg-type]


def test_compose_overrides_entries_type() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(
            schema,
            defaults={},
            overrides=["db.host=x"],  # type: ignore[list-item]
        )


def test_compose_unknown_group() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(schema, defaults={"cache": "redis"})


def test_compose_unknown_option() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(schema, defaults={"db": "oracle"})


def test_compose_defaults_value_must_be_str() -> None:
    schema = _basic_schema()
    with pytest.raises(ConfigError):
        compose(schema, defaults={"db": 1})  # type: ignore[dict-item]


def test_compose_overrides_cap() -> None:
    schema = ConfigSchema(name="App", base={}, groups=())
    many = tuple(Override(path=f"x.k{i}", value=i) for i in range(MAX_OVERRIDES + 1))
    with pytest.raises(ConfigError):
        compose(schema, defaults={}, overrides=many)


# ---------------------------------------------------------------------------
# ComposedConfig
# ---------------------------------------------------------------------------


def test_composed_config_frozen_slotted() -> None:
    schema = ConfigSchema(name="App", base={}, groups=())
    result = compose(schema, defaults={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.backend = "x"  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_composed_config_backend_validated() -> None:
    with pytest.raises(ConfigError):
        ComposedConfig(
            schema_name="App",
            backend="bogus",
            defaults={},
            override_count=0,
            values={},
        )


def test_composed_config_override_count_non_negative() -> None:
    with pytest.raises(ConfigError):
        ComposedConfig(
            schema_name="App",
            backend="stdlib",
            defaults={},
            override_count=-1,
            values={},
        )


def test_composed_config_schema_name_validated() -> None:
    with pytest.raises(ConfigError):
        ComposedConfig(
            schema_name="",
            backend="stdlib",
            defaults={},
            override_count=0,
            values={},
        )


def test_composed_config_get_path_validation() -> None:
    schema = ConfigSchema(name="App", base={}, groups=())
    result = compose(schema, defaults={})
    with pytest.raises(ConfigError):
        result.get("")


# ---------------------------------------------------------------------------
# Lazy seam
# ---------------------------------------------------------------------------


def test_enable_hydra_factory_without_dep_raises_import_error() -> None:
    try:
        import hydra  # type: ignore[import-not-found]  # noqa: F401

        pytest.skip("hydra installed; nothing to assert")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        enable_hydra_factory()


def test_enable_hydra_factory_returns_composer_when_present() -> None:
    try:
        import hydra  # type: ignore[import-not-found]  # noqa: F401
        import omegaconf  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("hydra-core/omegaconf not installed")
    composer = enable_hydra_factory()
    schema = ConfigSchema(name="App", base={"x": 1}, groups=())
    result = composer(schema, {}, ())
    assert result.backend == "hydra"
    assert result.values == {"x": 1}


def test_enable_hydra_factory_rejects_unknown_override_keys() -> None:
    try:
        import hydra  # type: ignore[import-not-found]  # noqa: F401
        import omegaconf  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("hydra-core/omegaconf not installed")
    with pytest.raises(ConfigError):
        enable_hydra_factory(overrides={"bogus_key": 1})


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "hydra_config.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_hydra_import() -> None:
    imports = _top_level_imports(_module_ast())
    assert "hydra" not in imports
    assert "omegaconf" not in imports


def test_no_top_level_yaml_import() -> None:
    assert "yaml" not in _top_level_imports(_module_ast())


def test_no_top_level_subprocess_import() -> None:
    assert "subprocess" not in _top_level_imports(_module_ast())


def test_no_top_level_time_or_random_import() -> None:
    banned = {"time", "random", "datetime", "asyncio"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_network_imports() -> None:
    banned = {"socket", "urllib", "requests", "httpx", "aiohttp"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_engine_imports() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "core.contracts.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


def test_hydra_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name in ("hydra", "omegaconf"):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside enable_hydra_factory"
                    )
                    assert parent.name == "enable_hydra_factory", (
                        f"{name} imported in {parent.name!r} — must be inside enable_hydra_factory"
                    )


# ---------------------------------------------------------------------------
# Realistic compose demo
# ---------------------------------------------------------------------------


def test_realistic_training_config() -> None:
    schema = ConfigSchema(
        name="Training",
        base={
            "seed": 42,
            "trainer": {"max_epochs": 10, "lr": 0.001},
            "data": {"batch_size": 32},
        },
        groups=(
            ConfigGroup(
                name="optimizer",
                options=(
                    ConfigOption(
                        name="adam",
                        values={"trainer": {"optimizer": "adam"}},
                    ),
                    ConfigOption(
                        name="sgd",
                        values={"trainer": {"optimizer": "sgd"}},
                    ),
                ),
            ),
            ConfigGroup(
                name="precision",
                options=(
                    ConfigOption(
                        name="fp32",
                        values={"trainer": {"precision": "fp32"}},
                    ),
                    ConfigOption(
                        name="bf16",
                        values={"trainer": {"precision": "bf16"}},
                    ),
                ),
            ),
        ),
    )
    result = compose(
        schema,
        defaults={"optimizer": "adam", "precision": "bf16"},
        overrides=(
            parse_override("trainer.lr=0.0005"),
            parse_override("trainer.max_epochs=20"),
            parse_override("data.batch_size=64"),
            parse_override("data.shuffle=true"),
        ),
    )
    trainer = result.values["trainer"]
    assert isinstance(trainer, dict)
    assert trainer["optimizer"] == "adam"
    assert trainer["precision"] == "bf16"
    assert trainer["lr"] == pytest.approx(0.0005)
    assert trainer["max_epochs"] == 20
    data = result.values["data"]
    assert isinstance(data, dict)
    assert data["batch_size"] == 64
    assert data["shuffle"] is True


# ---------------------------------------------------------------------------
# Reload idempotency — runs last
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import tools.hydra_config as mod1

    importlib.reload(mod1)
    import tools.hydra_config as mod2

    assert mod1.CONFIG_VERSION == mod2.CONFIG_VERSION
    assert mod1.MAX_GROUP_DEPTH == mod2.MAX_GROUP_DEPTH
