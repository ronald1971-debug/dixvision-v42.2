"""Smoke tests for sensory/web_autolearn/seeds.yaml (WEBLEARN-10)."""

from __future__ import annotations

from pathlib import Path

import yaml

from sensory.web_autolearn.curator import CuratorRules

_SEEDS = Path(__file__).resolve().parents[3] / (
    "sensory/web_autolearn/seeds.yaml"
)


def test_seeds_yaml_exists() -> None:
    assert _SEEDS.is_file()


def test_seeds_yaml_parses_to_none_or_mapping() -> None:
    """Empty seeds.yaml parses to None; populated parses to a mapping."""

    parsed = yaml.safe_load(_SEEDS.read_text(encoding="utf-8"))
    assert parsed is None or isinstance(parsed, dict)


def test_seeds_yaml_can_feed_curator_rules() -> None:
    """An empty seeds.yaml yields a no-op CuratorRules.

    Confirms the parser path used by the harness boots cleanly even
    when the operator hasn't added any seeds yet (the system must
    degrade to a no-op rather than crash).
    """

    parsed = yaml.safe_load(_SEEDS.read_text(encoding="utf-8"))
    rules = CuratorRules.from_mapping(parsed or {})
    assert dict(rules.rules) == {}


def test_data_source_registry_paths_resolve_at_import() -> None:
    """``sensory.web_autolearn.contracts.NewsItem`` and ``SocialPost``
    must be importable so ``registry/data_source_registry.yaml``
    references resolve at SCVS schema-validation time.
    """

    import importlib

    mod = importlib.import_module("sensory.web_autolearn.contracts")
    assert hasattr(mod, "NewsItem")
    assert hasattr(mod, "SocialPost")
