"""PR-RT-5 — total_validation Phase 12 (topology drift) tests.

Pins the canonical invariant::

    declared_topology == active_topology ∪ DECLARED_BUT_DORMANT_ALLOWLIST

Phase 12 of ``tools/total_validation.py`` enforces that every declared
node in ``ui.harness.runtime_registrar._DECLARED_NODE_SPECS`` is either
statically wired in the canonical boot sources or explicitly listed on
the dormant allowlist. Any other state is silent runtime-topology drift
and trips the strict-mode CI gate.

These tests exercise the phase via three lanes:

* the registrar's data accessors (``declared_state_attrs`` /
  ``DECLARED_BUT_DORMANT_ALLOWLIST``);
* the AST scanner ``_collect_attr_assignments`` against synthetic
  inputs;
* the full Phase 12 entrypoint ``_phase_12_topology_drift`` against the
  real declared topology and against a monkey-patched ``boot_sources``
  set that simulates drift.

The phase must be byte-stable: two runs on the same tree produce the
same artifact contents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# registrar contract
# ---------------------------------------------------------------------------


def test_declared_state_attrs_returns_sorted_pairs() -> None:
    from ui.harness.runtime_registrar import declared_state_attrs

    pairs = declared_state_attrs()
    assert isinstance(pairs, tuple)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    node_ids = [p[0] for p in pairs]
    assert node_ids == sorted(node_ids), "pairs must be sorted by node_id"


def test_declared_state_attrs_matches_declared_node_ids() -> None:
    from ui.harness.runtime_registrar import (
        declared_node_ids,
        declared_state_attrs,
    )

    pair_node_ids = tuple(p[0] for p in declared_state_attrs())
    assert pair_node_ids == declared_node_ids()


def test_declared_state_attrs_is_byte_stable() -> None:
    """Two independent imports / calls return the byte-identical tuple."""
    from ui.harness.runtime_registrar import declared_state_attrs

    first = declared_state_attrs()
    second = declared_state_attrs()
    assert first == second
    # And the JSON projection of the pairs is also byte-stable.
    a = json.dumps(list(first), sort_keys=True)
    b = json.dumps(list(second), sort_keys=True)
    assert a == b


def test_declared_but_dormant_allowlist_is_frozen_empty() -> None:
    from ui.harness.runtime_registrar import DECLARED_BUT_DORMANT_ALLOWLIST

    assert isinstance(DECLARED_BUT_DORMANT_ALLOWLIST, frozenset)
    assert DECLARED_BUT_DORMANT_ALLOWLIST == frozenset()


def test_allowlist_is_immutable() -> None:
    """frozenset rejects mutation attempts."""
    from ui.harness.runtime_registrar import DECLARED_BUT_DORMANT_ALLOWLIST

    with pytest.raises(AttributeError):
        DECLARED_BUT_DORMANT_ALLOWLIST.add("never_added")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AST scanner
# ---------------------------------------------------------------------------


def test_collect_attr_assignments_picks_up_self_assigns(tmp_path: Path) -> None:
    from tools.total_validation import _collect_attr_assignments

    src = tmp_path / "sample.py"
    src.write_text(
        "class Foo:\n"
        "    def build(self):\n"
        "        self.alpha = 1\n"
        "        self.beta = 2\n"
        "        other.gamma = 3\n",
        encoding="utf-8",
    )
    found = _collect_attr_assignments(src)
    assert "alpha" in found
    assert "beta" in found
    assert "gamma" not in found, "only self.X / state.X assignments are counted"


def test_collect_attr_assignments_picks_up_state_assigns(tmp_path: Path) -> None:
    from tools.total_validation import _collect_attr_assignments

    src = tmp_path / "manager.py"
    src.write_text(
        "def populate(state):\n    state.engine = object()\n    state.loop = object()\n",
        encoding="utf-8",
    )
    found = _collect_attr_assignments(src)
    assert "engine" in found
    assert "loop" in found


def test_collect_attr_assignments_handles_ann_and_aug(tmp_path: Path) -> None:
    from tools.total_validation import _collect_attr_assignments

    src = tmp_path / "mixed.py"
    src.write_text(
        "class Bar:\n"
        "    def build(self):\n"
        "        self.counter: int = 0\n"
        "        self.counter += 1\n"
        "        self.lock = object()\n",
        encoding="utf-8",
    )
    found = _collect_attr_assignments(src)
    assert "counter" in found
    assert "lock" in found


def test_collect_attr_assignments_tolerates_missing_file(tmp_path: Path) -> None:
    from tools.total_validation import _collect_attr_assignments

    out = _collect_attr_assignments(tmp_path / "does_not_exist.py")
    assert out == set()


def test_collect_attr_assignments_tolerates_syntax_error(tmp_path: Path) -> None:
    from tools.total_validation import _collect_attr_assignments

    src = tmp_path / "broken.py"
    src.write_text("def broken(\n", encoding="utf-8")
    out = _collect_attr_assignments(src)
    assert out == set()


# ---------------------------------------------------------------------------
# Phase 12 against the real codebase
# ---------------------------------------------------------------------------


def test_phase_12_topology_drift_zero_drift_on_main(tmp_path: Path) -> None:
    """Under PR-RT-4 wiring every declared node is statically assigned."""
    from tools import total_validation as tv

    # Redirect artifact write to tmp so the live ``analysis/`` is not
    # touched by the test run.
    original_analysis = tv.ANALYSIS_DIR
    tv.ANALYSIS_DIR = tmp_path
    try:
        phase, ok, drift_count = tv._phase_12_topology_drift(advisory=True)
    finally:
        tv.ANALYSIS_DIR = original_analysis

    assert phase.phase_id == 12
    assert phase.name == "topology_drift"
    assert phase.artifact == "topology_drift.json"
    assert ok is True
    assert drift_count == 0
    assert phase.status == "ok"
    assert phase.details["declared_node_count"] >= 28
    assert phase.details["drift_count"] == 0
    assert phase.details["drift_nodes"] == []
    assert (tmp_path / "topology_drift.json").exists()


def test_phase_12_topology_drift_artifact_is_byte_stable(tmp_path: Path) -> None:
    from tools import total_validation as tv

    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    original_analysis = tv.ANALYSIS_DIR
    try:
        tv.ANALYSIS_DIR = a_dir
        tv._phase_12_topology_drift(advisory=True)
        tv.ANALYSIS_DIR = b_dir
        tv._phase_12_topology_drift(advisory=True)
    finally:
        tv.ANALYSIS_DIR = original_analysis

    a_bytes = (a_dir / "topology_drift.json").read_bytes()
    b_bytes = (b_dir / "topology_drift.json").read_bytes()
    assert a_bytes == b_bytes


def test_phase_12_topology_drift_flags_unwired_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the canonical boot sources do not assign a declared state_attr
    and the node is not on the allowlist, Phase 12 must flag drift."""
    from tools import total_validation as tv

    # Synthetic boot source: assigns nothing.
    empty_boot = tmp_path / "empty_boot.py"
    empty_boot.write_text("def populate(state):\n    return None\n", encoding="utf-8")

    monkeypatch.setattr(tv, "_TOPOLOGY_BOOT_SOURCES", (str(empty_boot),))
    # Make REPO_ROOT a no-op so the relative join in the phase resolves
    # to ``empty_boot`` itself.
    monkeypatch.setattr(tv, "REPO_ROOT", Path("/"))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(tv, "ANALYSIS_DIR", out_dir)

    phase, ok, drift_count = tv._phase_12_topology_drift(advisory=True)

    assert ok is False
    assert drift_count > 0
    assert phase.status == "warn"  # advisory mode -> warn (not fail)
    assert phase.details["drift_count"] == drift_count
    flagged = {entry["node_id"] for entry in phase.details["drift_nodes"]}
    # Every declared node should be in drift since nothing is wired.
    from ui.harness.runtime_registrar import declared_node_ids

    assert flagged == set(declared_node_ids())


def test_phase_12_strict_mode_returns_fail_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import total_validation as tv

    empty_boot = tmp_path / "empty_boot.py"
    empty_boot.write_text("def populate(state):\n    return None\n", encoding="utf-8")

    monkeypatch.setattr(tv, "_TOPOLOGY_BOOT_SOURCES", (str(empty_boot),))
    monkeypatch.setattr(tv, "REPO_ROOT", Path("/"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(tv, "ANALYSIS_DIR", out_dir)

    phase, ok, drift_count = tv._phase_12_topology_drift(advisory=False)

    assert ok is False
    assert drift_count > 0
    assert phase.status == "fail"


def test_phase_12_allowlist_covers_unwired_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Putting a declared node on the allowlist removes it from drift."""
    from tools import total_validation as tv

    empty_boot = tmp_path / "empty_boot.py"
    empty_boot.write_text("def populate(state):\n    return None\n", encoding="utf-8")
    monkeypatch.setattr(tv, "_TOPOLOGY_BOOT_SOURCES", (str(empty_boot),))
    monkeypatch.setattr(tv, "REPO_ROOT", Path("/"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(tv, "ANALYSIS_DIR", out_dir)

    # Reload the registrar with a synthetic allowlist that covers every
    # declared node. We cannot mutate the existing frozenset; instead we
    # monkey-patch the symbol Phase 12 imports lazily at call time.
    import ui.harness.runtime_registrar as registrar

    full_allowlist = frozenset(registrar.declared_node_ids())
    monkeypatch.setattr(registrar, "DECLARED_BUT_DORMANT_ALLOWLIST", full_allowlist)

    phase, ok, drift_count = tv._phase_12_topology_drift(advisory=False)

    assert ok is True
    assert drift_count == 0
    assert phase.status == "ok"
    assert set(phase.details["allowlisted_node_ids"]) == set(registrar.declared_node_ids())
    assert phase.details["wired_node_ids"] == []


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_run_emits_phase_12_and_summary_topology_fields(tmp_path: Path) -> None:
    """Running the full pipeline through ``tools.total_validation.run``
    appends a Phase 12 entry and emits the topology fields into
    ``coverage_summary.json``."""
    from tools import total_validation as tv

    original_analysis = tv.ANALYSIS_DIR
    tv.ANALYSIS_DIR = tmp_path
    try:
        summary = tv.run(advisory=True)
    finally:
        tv.ANALYSIS_DIR = original_analysis

    assert "topology_drift_valid" in summary
    assert "topology_drift_count" in summary
    assert summary["topology_drift_valid"] is True
    assert summary["topology_drift_count"] == 0

    phase_12 = [p for p in summary["phases"] if p["id"] == 12]
    assert len(phase_12) == 1
    assert phase_12[0]["name"] == "topology_drift"
    assert phase_12[0]["status"] == "ok"
    assert phase_12[0]["artifact"] == "topology_drift.json"


def test_total_validation_docstring_mentions_13_phases() -> None:
    import tools.total_validation as tv

    assert "13-phase" in (tv.__doc__ or "")
    assert "Phase 12" in (tv.__doc__ or "")
    assert "topology drift" in (tv.__doc__ or "")
