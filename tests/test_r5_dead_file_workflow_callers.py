"""R-5 / Phase-6 audit fix — workflow/script callers of tools/*.py.

Pins the contract that ``tools/total_validation.py`` no longer flags a
``tools/<name>.py`` file as ``DEAD`` solely because no Python module
imports it. The five canonical CLI entry-points invoked from
``.github/workflows/*.yml`` must now appear as ``USED`` with the
workflow path listed in the ``referenced_by`` column.

Also exercises the new helper ``_build_external_caller_index`` in
isolation against synthetic inputs to lock the two reference syntaxes
the harvester recognises:

* path style — ``python tools/enforce.py --strict``
* module style — ``python -m tools.enforce``

The phase must remain deterministic — two invocations on the same tree
produce identical CSV rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# external-caller helper (unit)
# ---------------------------------------------------------------------------


def test_external_caller_index_picks_up_path_style(tmp_path, monkeypatch) -> None:
    from tools import total_validation as tv

    repo = tmp_path
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "tools").mkdir()
    (repo / "tools" / "enforce.py").write_text("# strict-mode gate\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  validate:\n    run: python tools/enforce.py --strict\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    callers = tv._build_external_caller_index()
    assert "tools/enforce.py" in callers
    assert ".github/workflows/ci.yml" in callers["tools/enforce.py"]


def test_external_caller_index_picks_up_module_style(tmp_path, monkeypatch) -> None:
    from tools import total_validation as tv

    repo = tmp_path
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "tools").mkdir()
    (repo / "tools" / "scvs_lint.py").write_text("# scvs lint\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  scvs:\n    run: python -m tools.scvs_lint .\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    callers = tv._build_external_caller_index()
    assert "tools/scvs_lint.py" in callers
    assert ".github/workflows/ci.yml" in callers["tools/scvs_lint.py"]


def test_external_caller_index_picks_up_scripts_dir(tmp_path, monkeypatch) -> None:
    from tools import total_validation as tv

    repo = tmp_path
    (repo / "scripts" / "windows").mkdir(parents=True)
    (repo / "tools").mkdir()
    (repo / "tools" / "rust_revival_reminder.py").write_text("# reminder\n", encoding="utf-8")
    (repo / "scripts" / "windows" / "reminder.bat").write_text(
        "python tools/rust_revival_reminder.py --open-issue\r\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    callers = tv._build_external_caller_index()
    assert "tools/rust_revival_reminder.py" in callers
    assert "scripts/windows/reminder.bat" in callers["tools/rust_revival_reminder.py"]


def test_external_caller_index_skips_binary_suffixes(tmp_path, monkeypatch) -> None:
    """Binary-suffix files must not be slurped — only allow-listed text exts."""
    from tools import total_validation as tv

    repo = tmp_path
    (repo / "scripts").mkdir()
    (repo / "scripts" / "icon.bmp").write_bytes(b"\x00\x01tools/enforce.py\x02\x03")

    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    callers = tv._build_external_caller_index()
    assert callers == {}


def test_external_caller_index_handles_missing_dirs(tmp_path, monkeypatch) -> None:
    from tools import total_validation as tv

    repo = tmp_path  # no .github/workflows, no scripts
    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    callers = tv._build_external_caller_index()
    assert callers == {}


def test_external_caller_index_is_deterministic(tmp_path, monkeypatch) -> None:
    from tools import total_validation as tv

    repo = tmp_path
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "tools").mkdir()
    (repo / "tools" / "a.py").write_text("# a\n", encoding="utf-8")
    (repo / "tools" / "b.py").write_text("# b\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text(
        "run: python tools/a.py && python -m tools.b\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tv, "REPO_ROOT", repo)
    one = {k: sorted(v) for k, v in tv._build_external_caller_index().items()}
    two = {k: sorted(v) for k, v in tv._build_external_caller_index().items()}
    assert one == two


# ---------------------------------------------------------------------------
# real-repo integration (live file_usage.csv)
# ---------------------------------------------------------------------------


_CANONICAL_CLI_ENTRYPOINTS: tuple[str, ...] = (
    "tools/enforce.py",
    "tools/authority_matrix_lint.py",
    "tools/scvs_lint.py",
    "tools/constraint_lint.py",
    "tools/rust_revival_reminder.py",
)


def _path_to_id_map(index_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with index_csv.open() as f:
        for row in csv.DictReader(f):
            out[row["file_path"]] = row["file_id"]
    return out


def _id_to_status_map(usage_csv: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with usage_csv.open() as f:
        for row in csv.DictReader(f):
            out[row["file_id"]] = (row["status"], row["referenced_by"])
    return out


@pytest.mark.parametrize("entrypoint", _CANONICAL_CLI_ENTRYPOINTS)
def test_canonical_cli_entrypoints_are_not_dead(entrypoint: str) -> None:
    """Every canonical CLI entry-point must report USED with a workflow caller."""
    repo_root = Path(__file__).resolve().parents[1]
    index = repo_root / "analysis" / "file_index.csv"
    usage = repo_root / "analysis" / "file_usage.csv"
    if not index.exists() or not usage.exists():
        pytest.skip("analysis artefacts not generated; run total_validation first")

    path_to_id = _path_to_id_map(index)
    id_to_status = _id_to_status_map(usage)

    file_id = path_to_id.get(entrypoint)
    assert file_id is not None, f"{entrypoint} missing from file_index.csv"
    status, callers = id_to_status[file_id]
    assert status == "USED", f"{entrypoint} regressed to DEAD: callers={callers!r}"
    assert ".github/workflows" in callers or "scripts/" in callers, (
        f"{entrypoint} USED but caller is not a workflow/script: {callers!r}"
    )
