"""SCVS Phase 1 — registry loader + consumption tracker + lint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from system_engine.scvs import (
    ConsumptionDeclaration,
    ConsumptionInput,
    SourceCategory,
    SourceRegistry,
    discover_consumption_declarations,
    load_consumption_declaration,
    load_source_registry,
    validate_scvs,
)
from tools import scvs_lint

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "registry" / "data_source_registry.yaml"


# ---------------------------------------------------------------------------
# source_registry
# ---------------------------------------------------------------------------


def test_canonical_registry_loads_clean() -> None:
    reg = load_source_registry(REGISTRY_PATH)
    assert reg.version == "v0.1.0"
    assert len(reg.sources) > 0
    # IDs are unique by construction (loader rejects duplicates).
    assert len(reg.ids) == len(reg.sources)
    # Enabled rows in the registry as of Wave-04 PR-2:
    #   * SRC-MARKET-BINANCE-001 — read-only public WS pump (no creds).
    #   * SRC-AI-{OPENAI,GEMINI,GROK,DEEPSEEK,DEVIN}-001 — flipped on
    #     by Wave-03 PR-6 so the registry-driven HTTP chat dispatcher
    #     (``intelligence_engine.cognitive.chat.http_chat_transport``)
    #     can route to each backend. Each row's API key is read from
    #     ``os.environ`` on every turn, so a row with a missing key
    #     fails-fast with :class:`TransientProviderError` and the
    #     adapter falls through to the next eligible provider — i.e.
    #     ``enabled: true`` does not commit the operator to keying
    #     every row at startup.
    #   * SRC-TRADER-TRADINGVIEW-001 — Wave-04 PR-2 trader-feed ingest
    #     path (read-only structured envelope). Parser is in
    #     ``ui.feeds.tradingview_ideas``; aggregator (sole B29-allowed
    #     producer of ``TraderObservation``) is in
    #     ``intelligence_engine.trader_modeling``.
    #   * SRC-NEWS-COINDESK-001 — Wave-04.5 PR-1 news ingest path
    #     (read-only public RSS, no auth). Pump is in
    #     ``ui.feeds.coindesk_rss`` and emits
    #     ``core.contracts.news.NewsItem`` rows into the harness.
    #   * SRC-MACRO-FRED-001 — Wave-04.5 PR-2 macro ingest path. Pump
    #     is in ``ui.feeds.fred_http`` and emits
    #     ``core.contracts.macro.MacroObservation`` rows into the
    #     harness. ``auth: required`` — the API key is read from
    #     ``os.environ['FRED_API_KEY']`` at pump construction; absent
    #     keys keep the row visible in the registry but the pump is
    #     simply not started, mirroring the AI-provider pattern.
    #   * SRC-MACRO-BLS-001 — Wave-04.5 PR-3 macro ingest path. Pump
    #     is in ``ui.feeds.bls_http`` and emits the same
    #     ``MacroObservation`` schema as FRED so downstream projection
    #     can fan both. ``auth: required`` — the registration key is
    #     read from ``os.environ['BLS_REGISTRATION_KEY']``; same
    #     opt-in startup pattern as FRED.
    assert reg.enabled_ids == frozenset(
        {
            "SRC-MARKET-BINANCE-001",
            "SRC-AI-OPENAI-001",
            "SRC-AI-GEMINI-001",
            "SRC-AI-GROK-001",
            "SRC-AI-DEEPSEEK-001",
            "SRC-AI-DEVIN-001",
            "SRC-TRADER-TRADINGVIEW-001",
            "SRC-NEWS-COINDESK-001",
            "SRC-MACRO-FRED-001",
            "SRC-MACRO-BLS-001",
        }
    )


def test_canonical_registry_has_all_categories() -> None:
    reg = load_source_registry(REGISTRY_PATH)
    cats = {s.category for s in reg.sources}
    # Phase 1 must cover at least every category the operator listed.
    expected = {
        SourceCategory.MARKET,
        SourceCategory.NEWS,
        SourceCategory.SOCIAL,
        SourceCategory.ONCHAIN,
        SourceCategory.MACRO,
        SourceCategory.REGULATORY,
        SourceCategory.DEV,
        SourceCategory.ALT,
        SourceCategory.AI,
        SourceCategory.SYNTHETIC,
    }
    missing = expected - cats
    assert not missing, f"missing categories in registry: {missing}"


def test_registry_rejects_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    },
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x2",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_source_registry(p)


def test_registry_rejects_unknown_category(tmp_path: Path) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-X-001",
                        "name": "x",
                        "category": "not-a-real-category",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="not a SourceCategory"):
        load_source_registry(p)


def test_registry_rejects_bad_id_prefix(tmp_path: Path) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "BAD-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="must start with 'SRC-'"):
        load_source_registry(p)


def test_registry_rejects_bad_auth(tmp_path: Path) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "bearer",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="auth"):
        load_source_registry(p)


# ---------------------------------------------------------------------------
# consumption_tracker
# ---------------------------------------------------------------------------


def _write_consumes(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body))


def test_load_consumption_declaration_minimal(tmp_path: Path) -> None:
    p = tmp_path / "consumes.yaml"
    _write_consumes(
        p,
        {
            "module": "intelligence_engine.signal_pipeline",
            "inputs": [
                {"source_id": "SRC-MARKET-BINANCE-001", "required": True}
            ],
        },
    )
    decl = load_consumption_declaration(p)
    assert decl.module == "intelligence_engine.signal_pipeline"
    assert decl.inputs == (
        ConsumptionInput(source_id="SRC-MARKET-BINANCE-001", required=True),
    )


def test_load_consumption_rejects_bad_source_id(tmp_path: Path) -> None:
    p = tmp_path / "consumes.yaml"
    _write_consumes(
        p,
        {
            "module": "x.y",
            "inputs": [{"source_id": "BAD-001"}],
        },
    )
    with pytest.raises(ValueError, match="must start with 'SRC-'"):
        load_consumption_declaration(p)


def test_load_consumption_rejects_duplicate_source_id(tmp_path: Path) -> None:
    p = tmp_path / "consumes.yaml"
    _write_consumes(
        p,
        {
            "module": "x.y",
            "inputs": [
                {"source_id": "SRC-MARKET-X-001"},
                {"source_id": "SRC-MARKET-X-001"},
            ],
        },
    )
    with pytest.raises(ValueError, match="duplicate source_id"):
        load_consumption_declaration(p)


def test_discover_consumption_declarations(tmp_path: Path) -> None:
    _write_consumes(
        tmp_path / "engine_a" / "consumes.yaml",
        {"module": "engine_a", "inputs": [{"source_id": "SRC-MARKET-X-001"}]},
    )
    _write_consumes(
        tmp_path / "engine_b" / "sub" / "consumes.yaml",
        {"module": "engine_b.sub", "inputs": []},
    )
    decls = discover_consumption_declarations([tmp_path])
    modules = {d.module for d in decls}
    assert modules == {"engine_a", "engine_b.sub"}


def test_discover_handles_dotted_ancestor_path(tmp_path: Path) -> None:
    """Hidden-directory filter is relative to the scan root.

    Regression test: an earlier implementation walked ``p.parts`` of the
    *absolute* path, which silently skipped every ``consumes.yaml``
    when the repo happened to live under a dotted ancestor directory
    (e.g. ``/home/user/.projects/myrepo``).
    """

    hidden_root = tmp_path / ".projects" / "repo" / "engine_a"
    _write_consumes(
        hidden_root / "consumes.yaml",
        {"module": "engine_a", "inputs": [{"source_id": "SRC-MARKET-X-001"}]},
    )
    decls = discover_consumption_declarations([hidden_root.parent])
    assert {d.module for d in decls} == {"engine_a"}


def test_discover_still_skips_dotted_subdir(tmp_path: Path) -> None:
    """A dotted subdir below the scan root is still skipped."""

    _write_consumes(
        tmp_path / ".cache" / "consumes.yaml",
        {"module": "junk", "inputs": []},
    )
    _write_consumes(
        tmp_path / "engine_a" / "consumes.yaml",
        {"module": "engine_a", "inputs": []},
    )
    decls = discover_consumption_declarations([tmp_path])
    assert {d.module for d in decls} == {"engine_a"}


def test_discover_rejects_duplicate_module(tmp_path: Path) -> None:
    _write_consumes(
        tmp_path / "a" / "consumes.yaml",
        {"module": "shared.mod", "inputs": []},
    )
    _write_consumes(
        tmp_path / "b" / "consumes.yaml",
        {"module": "shared.mod", "inputs": []},
    )
    with pytest.raises(ValueError, match="duplicate consumes.yaml module"):
        discover_consumption_declarations([tmp_path])


# ---------------------------------------------------------------------------
# lint — SCVS-01 / SCVS-02
# ---------------------------------------------------------------------------


def _registry(*sources: tuple[str, bool]) -> SourceRegistry:
    """Build a minimal SourceRegistry from (id, enabled) tuples."""

    from system_engine.scvs.source_registry import SourceDeclaration

    rows = tuple(
        SourceDeclaration(
            id=sid,
            name=sid,
            category=SourceCategory.MARKET,
            provider="x",
            endpoint="https://x",
            schema="x.X",
            auth="none",
            enabled=enabled,
            critical=False,
        )
        for sid, enabled in sources
    )
    return SourceRegistry(version="v0.1.0", sources=rows)


def _decl(module: str, *source_ids: str) -> ConsumptionDeclaration:
    return ConsumptionDeclaration(
        module=module,
        inputs=tuple(
            ConsumptionInput(source_id=sid, required=True) for sid in source_ids
        ),
        path=Path(f"/fake/{module}/consumes.yaml"),
    )


def test_lint_clean_when_all_enabled_consumed_and_no_phantom() -> None:
    reg = _registry(("SRC-A", True), ("SRC-B", False))
    decls = [_decl("mod_x", "SRC-A")]
    assert validate_scvs(reg, decls) == ()


def test_lint_scvs01_unused_enabled_source() -> None:
    reg = _registry(("SRC-A", True), ("SRC-B", True))
    decls = [_decl("mod_x", "SRC-A")]
    violations = validate_scvs(reg, decls)
    rules = {v.rule for v in violations}
    assert rules == {"SCVS-01"}
    assert any("SRC-B" in v.detail for v in violations)


def test_lint_scvs01_does_not_flag_disabled_unused_source() -> None:
    # enabled: false rows are exempt — they're placeholders for
    # not-yet-wired adapters.
    reg = _registry(("SRC-A", False))
    assert validate_scvs(reg, []) == ()


def test_lint_scvs02_phantom_consumption() -> None:
    reg = _registry(("SRC-A", False))
    decls = [_decl("mod_x", "SRC-DOES-NOT-EXIST")]
    violations = validate_scvs(reg, decls)
    rules = {v.rule for v in violations}
    assert rules == {"SCVS-02"}
    assert any("mod_x" in v.detail for v in violations)


def test_lint_reports_both_rules_independently() -> None:
    reg = _registry(("SRC-A", True))  # enabled, no consumer -> SCVS-01
    decls = [_decl("mod_x", "SRC-MISSING")]  # phantom -> SCVS-02
    violations = validate_scvs(reg, decls)
    rules = {v.rule for v in violations}
    assert rules == {"SCVS-01", "SCVS-02"}


# ---------------------------------------------------------------------------
# tools/scvs_lint.py — CI entry point
# ---------------------------------------------------------------------------


def test_scvs_lint_canonical_repo_is_clean() -> None:
    """The committed registry + consumes.yaml tree must lint clean."""

    rc = scvs_lint.main(["scvs_lint", str(REPO_ROOT)])
    assert rc == 0


def test_scvs_lint_returns_nonzero_on_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Synthesise a broken repo: registry has one enabled source that no
    # consumes.yaml references.
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "data_source_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "v0.1.0",
                "sources": [
                    {
                        "id": "SRC-MARKET-X-001",
                        "name": "x",
                        "category": "market",
                        "provider": "x",
                        "endpoint": "https://x",
                        "schema": "x.X",
                        "auth": "none",
                        "enabled": True,
                    }
                ],
            }
        )
    )
    (tmp_path / "core").mkdir()  # discoverable root, no consumes.yaml inside

    rc = scvs_lint.main(["scvs_lint", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "SCVS-01" in err


def test_scvs_lint_returns_two_when_registry_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = scvs_lint.main(["scvs_lint", str(tmp_path)])
    assert rc == 2
    assert "registry not found" in capsys.readouterr().err
