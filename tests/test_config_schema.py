"""
tests/test_config_schema.py — T0-12 config schema validation tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from system.config_schema import (
    ConfigValidationError,
    SystemConfig,
    load_config,
)


def test_defaults_are_valid() -> None:
    cfg = load_config()
    assert isinstance(cfg, SystemConfig)
    assert cfg.risk.max_drawdown_pct == 4.0
    assert cfg.guardian.check_interval_seconds == 2.0


def test_flat_overrides_win_over_defaults() -> None:
    cfg = load_config({"risk.max_drawdown_pct": 2.5})
    assert cfg.risk.max_drawdown_pct == 2.5


def test_nested_overrides_win_over_defaults() -> None:
    cfg = load_config({"risk": {"max_drawdown_pct": 3.0}})
    assert cfg.risk.max_drawdown_pct == 3.0


def test_unknown_section_fails_fast() -> None:
    with pytest.raises(ConfigValidationError):
        load_config({"nonexistent_section": {"foo": 1}})


def test_unknown_key_fails_fast() -> None:
    with pytest.raises(ConfigValidationError):
        load_config({"risk.wat_is_dis": 42})


def test_out_of_range_value_fails_fast() -> None:
    with pytest.raises(ConfigValidationError):
        load_config({"risk.max_drawdown_pct": -1})
    with pytest.raises(ConfigValidationError):
        load_config({"risk.max_drawdown_pct": 1000})


def test_wrong_type_fails_fast() -> None:
    with pytest.raises(ConfigValidationError):
        load_config({"risk.max_drawdown_pct": "nope"})


def test_empty_data_path_fails_fast() -> None:
    with pytest.raises(ConfigValidationError):
        load_config({"data.audit_log": ""})


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(Exception):
        cfg.risk.max_drawdown_pct = 99.0  # type: ignore[misc]


def test_as_dotted_round_trips() -> None:
    cfg = load_config()
    flat = cfg.as_dotted()
    assert flat["risk.max_drawdown_pct"] == 4.0
    assert flat["guardian.check_interval_seconds"] == 2.0
    assert "data.snapshots" in flat


def test_yaml_override_loads_fail_fast(tmp_path: Path) -> None:
    good = tmp_path / "ok.yaml"
    good.write_text("risk:\n  max_drawdown_pct: 1.25\n", encoding="utf-8")
    cfg = load_config(yaml_path=good)
    assert cfg.risk.max_drawdown_pct == 1.25

    bad = tmp_path / "bad.yaml"
    bad.write_text("risk:\n  max_drawdown_pct: -9\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_config(yaml_path=bad)


def test_yaml_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError):
        load_config(yaml_path=tmp_path / "nope.yaml")


def test_yaml_non_mapping_fails_fast(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_config(yaml_path=p)


def test_yaml_syntax_error_fails_fast(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("risk:\n  - oops: not valid\n  - [[[\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_config(yaml_path=p)


def test_overrides_win_over_yaml(tmp_path: Path) -> None:
    y = tmp_path / "c.yaml"
    y.write_text("risk:\n  max_drawdown_pct: 1.0\n", encoding="utf-8")
    cfg = load_config({"risk.max_drawdown_pct": 2.2}, yaml_path=y)
    assert cfg.risk.max_drawdown_pct == 2.2
