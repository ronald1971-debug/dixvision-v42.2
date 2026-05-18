# ADAPTED FROM: https://github.com/gorakhargosh/watchdog  (Apache-2.0)
"""Tests for the canonical hot-reload file watcher (I-10)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from system_engine import file_watcher
from system_engine.file_watcher import (
    DEFAULT_PATTERNS,
    FORBIDDEN_TIERS,
    NEW_PIP_DEPENDENCIES,
    WATCHER_VERSION,
    ChangeKind,
    DirectorySnapshot,
    FileChangeEvent,
    FileWatcherError,
    RegistryWatchSpec,
    diff_snapshots,
    enable_watchdog_observer_factory,
    match_patterns,
    scan_directory,
)

# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------


def test_version_tag() -> None:
    assert WATCHER_VERSION == "v1.0-I10"


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("watchdog",)


def test_default_patterns_are_yaml() -> None:
    assert DEFAULT_PATTERNS == ("*.yaml", "*.yml")


def test_forbidden_tiers_include_governance() -> None:
    assert "governance_engine" in FORBIDDEN_TIERS
    assert "execution_engine" in FORBIDDEN_TIERS
    assert "intelligence_engine" in FORBIDDEN_TIERS


def test_all_export_complete() -> None:
    expected = {
        "DEFAULT_PATTERNS",
        "FORBIDDEN_TIERS",
        "WATCHER_VERSION",
        "NEW_PIP_DEPENDENCIES",
        "ChangeKind",
        "DirectorySnapshot",
        "FileChangeEvent",
        "FileWatcherError",
        "RegistryWatchSpec",
        "diff_snapshots",
        "enable_watchdog_observer_factory",
        "match_patterns",
        "scan_directory",
    }
    assert set(file_watcher.__all__) == expected


# ---------------------------------------------------------------------------
# FileChangeEvent validation
# ---------------------------------------------------------------------------


def test_filechangeevent_created_requires_digest() -> None:
    with pytest.raises(FileWatcherError):
        FileChangeEvent(ChangeKind.CREATED, "a.yaml", "")


def test_filechangeevent_modified_requires_digest() -> None:
    with pytest.raises(FileWatcherError):
        FileChangeEvent(ChangeKind.MODIFIED, "a.yaml", "")


def test_filechangeevent_deleted_must_have_empty_digest() -> None:
    with pytest.raises(FileWatcherError):
        FileChangeEvent(ChangeKind.DELETED, "a.yaml", "nonempty")


def test_filechangeevent_rejects_empty_path() -> None:
    with pytest.raises(FileWatcherError):
        FileChangeEvent(ChangeKind.CREATED, "", "abc")


def test_filechangeevent_rejects_non_chankind() -> None:
    with pytest.raises(FileWatcherError):
        FileChangeEvent("CREATED", "a.yaml", "abc")  # type: ignore[arg-type]


def test_filechangeevent_frozen() -> None:
    e = FileChangeEvent(ChangeKind.CREATED, "a.yaml", "abc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.path = "b.yaml"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DirectorySnapshot validation
# ---------------------------------------------------------------------------


def test_directorysnapshot_rejects_empty_root() -> None:
    with pytest.raises(FileWatcherError):
        DirectorySnapshot("", {})


def test_directorysnapshot_rejects_non_string_value() -> None:
    with pytest.raises(FileWatcherError):
        DirectorySnapshot("r", {"a.yaml": 123})  # type: ignore[dict-item]


def test_directorysnapshot_canonical_items_sorted() -> None:
    snap = DirectorySnapshot("r", {"b": "1", "a": "2"})
    assert snap.canonical_items() == (("a", "2"), ("b", "1"))


# ---------------------------------------------------------------------------
# RegistryWatchSpec governance lock
# ---------------------------------------------------------------------------


def test_watchspec_accepts_registry_root() -> None:
    s = RegistryWatchSpec(root="registry")
    assert s.patterns == DEFAULT_PATTERNS


def test_watchspec_accepts_explicit_patterns() -> None:
    s = RegistryWatchSpec(root="registry", patterns=("*.yaml",))
    assert s.patterns == ("*.yaml",)


def test_watchspec_rejects_empty_patterns() -> None:
    with pytest.raises(FileWatcherError):
        RegistryWatchSpec(root="registry", patterns=())


def test_watchspec_rejects_governance_engine() -> None:
    with pytest.raises(FileWatcherError, match="governance-lock"):
        RegistryWatchSpec(root="governance_engine")


def test_watchspec_rejects_governance_engine_subdir() -> None:
    with pytest.raises(FileWatcherError, match="governance-lock"):
        RegistryWatchSpec(root="governance_engine/policies")


def test_watchspec_rejects_execution_engine() -> None:
    with pytest.raises(FileWatcherError, match="governance-lock"):
        RegistryWatchSpec(root="execution_engine/adapters")


def test_watchspec_rejects_intelligence_engine() -> None:
    with pytest.raises(FileWatcherError, match="governance-lock"):
        RegistryWatchSpec(root="intelligence_engine/cognitive")


# ---------------------------------------------------------------------------
# match_patterns
# ---------------------------------------------------------------------------


def test_match_patterns_yaml_hits() -> None:
    assert match_patterns("engines.yaml", DEFAULT_PATTERNS) is True


def test_match_patterns_yml_hits() -> None:
    assert match_patterns("confidence.yml", DEFAULT_PATTERNS) is True


def test_match_patterns_json_misses() -> None:
    assert match_patterns("engines.json", DEFAULT_PATTERNS) is False


def test_match_patterns_rejects_empty_pattern() -> None:
    with pytest.raises(FileWatcherError):
        match_patterns("x.yaml", ("",))


def test_match_patterns_rejects_non_string_name() -> None:
    with pytest.raises(FileWatcherError):
        match_patterns(123, DEFAULT_PATTERNS)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# scan_directory — pure mode with injected loaders
# ---------------------------------------------------------------------------


def _fake_lister(_root: str) -> list[str]:
    return ["a.yaml", "b.yaml", "c.txt"]


def _fake_loader(rel: str) -> bytes:
    return {"a.yaml": b"alpha", "b.yaml": b"beta", "c.txt": b"ignored"}[rel]


def test_scan_directory_filters_by_pattern() -> None:
    snap = scan_directory(
        "registry",
        file_loader=_fake_loader,
        file_lister=_fake_lister,
    )
    assert set(snap.entries.keys()) == {"a.yaml", "b.yaml"}


def test_scan_directory_digests_are_blake2b16() -> None:
    snap = scan_directory(
        "registry",
        file_loader=_fake_loader,
        file_lister=_fake_lister,
    )
    for digest in snap.entries.values():
        assert len(digest) == 32  # blake2b-16 → 16 bytes → 32 hex chars
        assert all(c in "0123456789abcdef" for c in digest)


def test_scan_directory_is_pure_three_runs_byte_identical() -> None:
    a = scan_directory("registry", file_loader=_fake_loader, file_lister=_fake_lister)
    b = scan_directory("registry", file_loader=_fake_loader, file_lister=_fake_lister)
    c = scan_directory("registry", file_loader=_fake_loader, file_lister=_fake_lister)
    assert a.entries == b.entries == c.entries
    assert a.canonical_items() == b.canonical_items() == c.canonical_items()


def test_scan_directory_rejects_governance() -> None:
    with pytest.raises(FileWatcherError, match="governance-lock"):
        scan_directory("governance_engine", file_loader=_fake_loader, file_lister=_fake_lister)


def test_scan_directory_rejects_non_bytes_loader() -> None:
    with pytest.raises(FileWatcherError):
        scan_directory(
            "registry",
            file_loader=lambda _: "not bytes",  # type: ignore[arg-type, return-value]
            file_lister=_fake_lister,
        )


def test_scan_directory_empty_lister() -> None:
    snap = scan_directory(
        "registry",
        file_loader=_fake_loader,
        file_lister=lambda _r: [],
    )
    assert snap.entries == {}


# ---------------------------------------------------------------------------
# diff_snapshots — pure function (INV-15)
# ---------------------------------------------------------------------------


def test_diff_detects_created() -> None:
    prev = DirectorySnapshot("r", {})
    curr = DirectorySnapshot("r", {"a.yaml": "d1"})
    events = diff_snapshots(prev, curr)
    assert len(events) == 1
    assert events[0].kind is ChangeKind.CREATED
    assert events[0].path == "a.yaml"
    assert events[0].digest == "d1"


def test_diff_detects_deleted() -> None:
    prev = DirectorySnapshot("r", {"a.yaml": "d1"})
    curr = DirectorySnapshot("r", {})
    events = diff_snapshots(prev, curr)
    assert len(events) == 1
    assert events[0].kind is ChangeKind.DELETED
    assert events[0].digest == ""


def test_diff_detects_modified() -> None:
    prev = DirectorySnapshot("r", {"a.yaml": "d1"})
    curr = DirectorySnapshot("r", {"a.yaml": "d2"})
    events = diff_snapshots(prev, curr)
    assert len(events) == 1
    assert events[0].kind is ChangeKind.MODIFIED
    assert events[0].digest == "d2"


def test_diff_ignores_unchanged() -> None:
    prev = DirectorySnapshot("r", {"a.yaml": "d1", "b.yaml": "d2"})
    curr = DirectorySnapshot("r", {"a.yaml": "d1", "b.yaml": "d2"})
    assert diff_snapshots(prev, curr) == ()


def test_diff_events_sorted_by_path() -> None:
    prev = DirectorySnapshot("r", {})
    curr = DirectorySnapshot("r", {"z.yaml": "d3", "a.yaml": "d1", "m.yaml": "d2"})
    events = diff_snapshots(prev, curr)
    paths = [e.path for e in events]
    assert paths == sorted(paths)


def test_diff_byte_identical_three_runs() -> None:
    prev = DirectorySnapshot("r", {"a.yaml": "d1"})
    curr = DirectorySnapshot("r", {"a.yaml": "d2", "b.yaml": "d3"})
    a = diff_snapshots(prev, curr)
    b = diff_snapshots(prev, curr)
    c = diff_snapshots(prev, curr)
    assert a == b == c


def test_diff_rejects_root_mismatch() -> None:
    prev = DirectorySnapshot("r1", {})
    curr = DirectorySnapshot("r2", {})
    with pytest.raises(FileWatcherError):
        diff_snapshots(prev, curr)


# ---------------------------------------------------------------------------
# End-to-end on registry/ directory (read-only).
# ---------------------------------------------------------------------------


def test_scan_real_registry_dir(tmp_path) -> None:
    (tmp_path / "engines.yaml").write_bytes(b"a: 1\n")
    (tmp_path / "ignored.json").write_bytes(b"{}")
    snap = scan_directory(str(tmp_path))
    # only yaml is kept
    assert any(k.endswith("engines.yaml") for k in snap.entries)
    assert not any(k.endswith("ignored.json") for k in snap.entries)


def test_end_to_end_diff(tmp_path) -> None:
    f = tmp_path / "policies.yaml"
    f.write_bytes(b"initial: 1\n")
    snap1 = scan_directory(str(tmp_path))
    f.write_bytes(b"updated: 2\n")
    snap2 = scan_directory(str(tmp_path))
    events = diff_snapshots(snap1, snap2)
    assert len(events) == 1
    assert events[0].kind is ChangeKind.MODIFIED


# ---------------------------------------------------------------------------
# Lazy seam — watchdog Observer factory
# ---------------------------------------------------------------------------


def test_enable_watchdog_observer_factory_lazy() -> None:
    try:
        build = enable_watchdog_observer_factory()
    except ImportError:
        pytest.skip("watchdog not installed")
    assert callable(build)
    obs = build()
    # Observer exposes start/stop/schedule per watchdog API
    assert hasattr(obs, "start")
    assert hasattr(obs, "stop")


# ---------------------------------------------------------------------------
# AST guardrails (B1 / INV-15 / B27 / B28 / INV-71)
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = (
    "watchdog",
    "time",
    "datetime",
    "random",
    "asyncio",
    "numpy",
    "torch",
    "polars",
    "requests",
)


def _module_source() -> str:
    return Path(inspect.getfile(file_watcher)).read_text(encoding="utf-8")


def _toplevel_imports() -> set[str]:
    tree = ast.parse(_module_source())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_top_level_imports() -> None:
    found = _toplevel_imports()
    for banned in _FORBIDDEN_TOP_LEVEL_IMPORTS:
        assert banned not in found, f"forbidden top-level import: {banned}"


def test_no_runtime_engine_imports_b1() -> None:
    forbidden_prefixes = (
        "execution_engine",
        "intelligence_engine",
        "governance_engine",
    )
    found = _toplevel_imports()
    for prefix in forbidden_prefixes:
        for name in found:
            assert not name.startswith(prefix), f"B1 violation: {name}"


def test_no_typed_event_constructors_b27_b28_inv71() -> None:
    src = _module_source()
    forbidden = (
        "SignalEvent(",
        "ExecutionEvent(",
        "HazardEvent(",
        "LearningUpdate(",
        "PatchProposal(",
    )
    for kind in forbidden:
        assert kind not in src, f"B27/B28/INV-71 violation: {kind} present"


def test_lazy_seam_uses_function_local_import() -> None:
    src = _module_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enable_watchdog_observer_factory":
            imports = [n for n in ast.walk(node) if isinstance(n, ast.Import)]
            modules = {alias.name for imp in imports for alias in imp.names}
            assert any(m.startswith("watchdog") for m in modules)
