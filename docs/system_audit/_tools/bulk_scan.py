"""Phase 3 — bulk static scan over the whole repo.

Runs every linter / analyser we have available and aggregates the
findings per file_id into ``bulk_findings.json``. Each finding has
shape::

    {
      "file_id": "F00123",
      "path": "core/foo.py",
      "tool": "ruff",
      "rule": "F401",
      "line": 12,
      "msg": "...",
    }

Tools executed (each is wrapped in try/except so a missing tool does
not abort the run):

* ``ruff check`` — style + simple bug detection.
* ``ruff format --check`` — formatting drift.
* ``vulture`` — dead-code candidates (low confidence first pass).
* ``python tools/authority_lint.py`` — repo-specific B-rules.
* ``python tools/authority_matrix_lint.py`` — authority matrix lint.
* ``python tools/constraint_lint.py`` — constraint-graph lint.
* ``python tools/scvs_lint.py`` — SCVS coverage lint.

Each finding is mapped to a file_id by joining the path with the
``file_index.csv``. Findings whose path is not in the index (e.g. a
generated file) are kept under ``unmapped_findings`` for visibility.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "docs" / "system_audit" / "file_index.csv"
OUT = REPO_ROOT / "docs" / "system_audit" / "bulk_findings.json"


def load_path_to_id() -> dict[str, str]:
    out: dict[str, str] = {}
    with INDEX.open() as fh:
        for row in csv.DictReader(fh):
            out[row["path"]] = row["file_id"]
    return out


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> tuple[int, str, str]:
    try:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return res.returncode, res.stdout, res.stderr


def parse_ruff_check(stdout: str) -> list[dict]:
    findings: list[dict] = []
    pat = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): (?P<rule>[A-Z0-9]+) (?P<msg>.+)$")
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            findings.append(
                {
                    "tool": "ruff",
                    "rule": m.group("rule"),
                    "path": m.group("path"),
                    "line": int(m.group("line")),
                    "msg": m.group("msg"),
                }
            )
    return findings


def parse_ruff_format(stdout: str) -> list[dict]:
    findings: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Would reformat:"):
            path = line.removeprefix("Would reformat:").strip()
            findings.append(
                {
                    "tool": "ruff-format",
                    "rule": "FORMAT",
                    "path": path,
                    "line": 0,
                    "msg": "would be reformatted",
                }
            )
    return findings


def parse_vulture(stdout: str) -> list[dict]:
    findings: list[dict] = []
    pat = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+): (?P<msg>.+)$")
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            findings.append(
                {
                    "tool": "vulture",
                    "rule": "DEAD",
                    "path": m.group("path"),
                    "line": int(m.group("line")),
                    "msg": m.group("msg"),
                }
            )
    return findings


def parse_simple_lint(tool: str, stdout: str, stderr: str) -> list[dict]:
    findings: list[dict] = []
    text = (stdout + "\n" + stderr).strip()
    if not text:
        return findings
    # Capture each line that mentions a path:line: msg
    pat = re.compile(r"^(?P<path>[^\s:]+\.py):(?P<line>\d+):.*?(?P<msg>.*)$")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            findings.append(
                {
                    "tool": tool,
                    "rule": "LINT",
                    "path": m.group("path"),
                    "line": int(m.group("line")),
                    "msg": m.group("msg").strip(),
                }
            )
    return findings


def main() -> int:
    path_to_id = load_path_to_id()
    findings: list[dict] = []
    summary: dict[str, dict] = {}

    # 1. ruff check (across whole repo)
    rc, stdout, _ = run(["python", "-m", "ruff", "check", ".", "--output-format=concise"])
    rf = parse_ruff_check(stdout)
    findings.extend(rf)
    summary["ruff"] = {"rc": rc, "n_findings": len(rf)}

    # 2. ruff format --check
    rc, stdout, _ = run(["python", "-m", "ruff", "format", "--check", "."])
    ff = parse_ruff_format(stdout)
    findings.extend(ff)
    summary["ruff-format"] = {"rc": rc, "n_findings": len(ff)}

    # 3. vulture (dead code, low-confidence first pass)
    rc, stdout, _ = run(
        [
            "python",
            "-m",
            "vulture",
            ".",
            "--exclude",
            "dashboard2026,dash_meme,docs,node_modules,.venv,.git",
            "--min-confidence",
            "80",
        ]
    )
    vf = parse_vulture(stdout)
    findings.extend(vf)
    summary["vulture"] = {"rc": rc, "n_findings": len(vf)}

    # 4. authority_lint (repo-specific)
    for tool in (
        "authority_lint",
        "authority_matrix_lint",
        "constraint_lint",
        "scvs_lint",
    ):
        script = REPO_ROOT / "tools" / f"{tool}.py"
        if not script.is_file():
            summary[tool] = {"rc": 127, "n_findings": 0, "missing": True}
            continue
        rc, stdout, stderr = run(["python", str(script)])
        # These linters mostly print pass/fail summaries; the regex
        # pulls out any concrete file:line lines.
        af = parse_simple_lint(tool, stdout, stderr)
        findings.extend(af)
        summary[tool] = {
            "rc": rc,
            "n_findings": len(af),
            "stdout_tail": stdout.strip().splitlines()[-5:],
            "stderr_tail": stderr.strip().splitlines()[-5:],
        }

    # Map paths -> file_ids
    mapped: list[dict] = []
    unmapped: list[dict] = []
    for f in findings:
        norm = f["path"].lstrip("./")
        f["path"] = norm
        fid = path_to_id.get(norm)
        if fid is None:
            unmapped.append(f)
            continue
        f["file_id"] = fid
        mapped.append(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(
            {
                "summary": summary,
                "mapped_findings": mapped,
                "unmapped_findings": unmapped,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(
        f"wrote {OUT.relative_to(REPO_ROOT)}: "
        f"{len(mapped)} mapped, {len(unmapped)} unmapped"
    )
    for tool, info in summary.items():
        print(f"  {tool}: rc={info['rc']} findings={info['n_findings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
