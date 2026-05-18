"""C-4 / P2-2 — sensory perimeter registered in the SCVS source registry.

Pins the FULL_SYSTEM_ANALYSIS finding ("sensory/ is importable and
correct but none of its local data sources appear in
``registry/data_source_registry.yaml``; SCVS validation doesn't cover
sensory sources") by asserting the local feature-extractor rows are
present, parse, and do not trip SCVS-01 / SCVS-02.

The rows are intentionally ``enabled: false`` — they are declarative
placeholders so future ``consumes.yaml`` declarations can reference
them. SCVS-01 only fires on ``enabled: true`` rows, so a sensory row
with no consumer is legal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from system_engine.scvs import (
    discover_consumption_declarations,
    find_redundant_sources,
    load_source_registry,
    validate_scvs,
)
from system_engine.scvs.source_registry import SourceCategory

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "registry" / "data_source_registry.yaml"

# The five sensory rows added by C-4 / P2-2. Each must parse and be
# bound to ``category=sensory``. The list is the authoritative pin —
# any drift (rename / delete / category change) breaks this test.
_EXPECTED_SENSORY_IDS: tuple[str, ...] = (
    "SRC-SENSORY-SNN-INDIRA-001",
    "SRC-SENSORY-SNN-DYON-001",
    "SRC-SENSORY-SNN-GOVERNANCE-001",
    "SRC-SENSORY-WEB-AUTOLEARN-001",
    "SRC-SENSORY-INDICATORS-001",
)


@pytest.fixture(scope="module")
def registry():
    return load_source_registry(REGISTRY_PATH)


def test_sensory_category_enum_exists():
    """``SENSORY`` is part of the canonical category taxonomy."""

    assert SourceCategory.SENSORY.value == "sensory"
    # Defensive — the enum is closed; adding a category is a load-
    # bearing change. Pin the full set so accidental additions or
    # removals surface as a test failure.
    expected = {
        "market",
        "news",
        "social",
        "onchain",
        "macro",
        "regulatory",
        "dev",
        "alt",
        "trader",
        "ai",
        "synthetic",
        "sensory",
    }
    assert {c.value for c in SourceCategory} == expected


def test_all_five_sensory_rows_present(registry) -> None:
    ids = registry.ids
    missing = [sid for sid in _EXPECTED_SENSORY_IDS if sid not in ids]
    assert missing == [], f"sensory perimeter rows missing from registry: {missing}"


def test_sensory_rows_are_category_sensory(registry) -> None:
    for sid in _EXPECTED_SENSORY_IDS:
        decl = registry.by_id(sid)
        assert decl is not None, f"{sid} missing"
        assert decl.category is SourceCategory.SENSORY, (
            f"{sid} category={decl.category.value!r}; expected 'sensory'"
        )


def test_sensory_rows_use_local_provider_and_endpoint(registry) -> None:
    """C-4 / P2-2: local rows must NOT pretend to be external.

    The ``local://`` scheme is the convention — it marks the row as
    a local feature extractor (no network egress, no credential).
    The ``provider`` is ``local`` for the same reason.
    """

    for sid in _EXPECTED_SENSORY_IDS:
        decl = registry.by_id(sid)
        assert decl is not None
        assert decl.provider == "local", f"{sid} provider={decl.provider!r}; expected 'local'"
        assert decl.endpoint.startswith("local://"), (
            f"{sid} endpoint={decl.endpoint!r}; expected 'local://...'"
        )
        assert decl.auth == "none", f"{sid} auth={decl.auth!r}; expected 'none' for a local sensor"


def test_sensory_rows_are_disabled_by_default(registry) -> None:
    """Sensory rows ship ``enabled: false`` so SCVS-01 doesn't fire.

    Flipping to ``enabled: true`` is intentional and would require a
    matching ``consumes.yaml`` declaration somewhere — that closure
    check is what the SCVS lint enforces. This test pins the
    behaviour expected immediately after the C-4 / P2-2 PR.
    """

    for sid in _EXPECTED_SENSORY_IDS:
        decl = registry.by_id(sid)
        assert decl is not None
        assert decl.enabled is False, (
            f"{sid} enabled={decl.enabled!r}; expected False until a "
            f"consumer declares it in consumes.yaml"
        )


def test_sensory_rows_have_zero_liveness_threshold(registry) -> None:
    """Sensory rows inherit the ``sensory`` category default of 0.

    Local extractors run on demand from the harness, so they do not
    emit heartbeats. ``liveness_threshold_ms == 0`` is the canonical
    'not liveness-checked' sentinel (matches ``synthetic``).
    """

    for sid in _EXPECTED_SENSORY_IDS:
        decl = registry.by_id(sid)
        assert decl is not None
        assert decl.liveness_threshold_ms == 0, (
            f"{sid} liveness_threshold_ms={decl.liveness_threshold_ms}; "
            f"expected 0 (sensory category default)"
        )


def test_scvs_lint_still_green_after_sensory_rows(registry) -> None:
    """C-4 must not regress the bidirectional-closure lint.

    The new rows are ``enabled: false`` so SCVS-01 cannot fire on
    them, and no ``consumes.yaml`` references them yet so SCVS-02
    cannot fire either. The lint must remain at zero violations.
    """

    discover_roots = [
        REPO_ROOT / r
        for r in (
            "core",
            "execution_engine",
            "evolution_engine",
            "governance_engine",
            "intelligence_engine",
            "learning_engine",
            "sensory",
            "state",
            "system_engine",
            "ui",
        )
        if (REPO_ROOT / r).is_dir()
    ]
    declarations = discover_consumption_declarations(discover_roots)
    violations = validate_scvs(registry, declarations)
    assert violations == (), f"SCVS lint regressed after adding sensory rows: {violations}"


def test_no_redundant_sensory_rows(registry) -> None:
    """SCVS-08 — no two sensory rows share (category, provider, endpoint).

    Each sensory row points at a distinct local module path. If two
    rows collide the lint warns; the test treats that as a regression
    because sensory rows are intentionally one-per-module.
    """

    warnings = find_redundant_sources(registry)
    sensory_warns = [w for w in warnings if "category='sensory'" in w.detail]
    assert sensory_warns == [], (
        f"sensory rows triggered SCVS-08 redundancy warnings: {sensory_warns}"
    )
