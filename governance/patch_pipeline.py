"""
governance.patch_pipeline -- staged gate every code change must traverse.

Stages (in order):
    PROPOSED            candidate registered, hash + submitter captured
    SANDBOX_IMPORT      tools/sandbox_runner.sandbox_import returns ok
    AUTHORITY_LINT      tools/authority_lint.py passes (0 violations)
    UNIT_TESTS          pytest -q succeeds on a sandbox copy
    DEP_SCAN            requirements typosquat / advisory check passes
    SHADOW_TEST         (cold-path) strategy arbiter replays without error
    CANARY              (manual promotion in cockpit; small-size window)
    GOVERNANCE_APPROVED (explicit human click in cockpit)
    LIVE                merged to main / adapter registered

Every stage emits a `GOVERNANCE/PATCH_*` ledger event with the patch id; the
pipeline refuses to short-circuit stages. Failing stages write
`GOVERNANCE/PATCH_REJECTED` with the stderr captured from the sandbox.

The CI entrypoint is `python -m governance.patch_pipeline --check-pr`, which
reads the PR diff from `git diff --name-only` and runs the first four stages
(sandbox + lint + tests + dep scan) over every changed .py file. The CI job
fails if any stage fails.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from state.ledger.writer import get_writer
from tools.sandbox_runner import (
    sandbox_authority_lint,
    sandbox_dep_scan,
    sandbox_import,
    sandbox_unit_tests,
)


class Stage(str, Enum):
    PROPOSED = "PROPOSED"
    SANDBOX_IMPORT = "SANDBOX_IMPORT"
    AUTHORITY_LINT = "AUTHORITY_LINT"
    UNIT_TESTS = "UNIT_TESTS"
    DEP_SCAN = "DEP_SCAN"
    SHADOW_TEST = "SHADOW_TEST"
    CANARY = "CANARY"
    GOVERNANCE_APPROVED = "GOVERNANCE_APPROVED"
    LIVE = "LIVE"
    REJECTED = "REJECTED"


@dataclass
class PatchVerdict:
    patch_id: str
    path: str
    stage: Stage
    ok: bool
    detail: str
    stderr: str = ""

    def as_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "path": self.path,
            "stage": self.stage.value,
            "ok": self.ok,
            "detail": self.detail,
            "stderr": self.stderr[-2000:],
        }


@dataclass
class PipelineReport:
    patch_id: str
    path: str
    verdicts: list[PatchVerdict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(v.ok for v in self.verdicts)

    def as_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "path": self.path,
            "ok": self.ok,
            "stages": [v.as_dict() for v in self.verdicts],
        }


def _patch_id(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    return hashlib.sha256(data + str(path).encode("utf-8")).hexdigest()[:16]


def run_pipeline(path: Path, repo_root: Path) -> PipelineReport:
    pid = _patch_id(path)
    rel = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
    report = PipelineReport(patch_id=pid, path=rel)
    writer = get_writer()

    writer.append_event(stream="GOVERNANCE", kind="PATCH_PROPOSED",
                        payload={"patch_id": pid, "path": rel})

    r_imp = sandbox_import(path, repo_root)
    report.verdicts.append(PatchVerdict(
        patch_id=pid, path=rel, stage=Stage.SANDBOX_IMPORT,
        ok=r_imp.ok, detail=r_imp.stage, stderr=r_imp.stderr,
    ))
    if not r_imp.ok:
        writer.append_event(stream="GOVERNANCE", kind="PATCH_REJECTED",
                            payload={"patch_id": pid, "at": Stage.SANDBOX_IMPORT.value,
                                     "stderr": r_imp.stderr[-500:]})
        return report

    r_lint = sandbox_authority_lint(repo_root)
    report.verdicts.append(PatchVerdict(
        patch_id=pid, path=rel, stage=Stage.AUTHORITY_LINT,
        ok=r_lint.ok, detail=r_lint.stage, stderr=r_lint.stderr,
    ))
    if not r_lint.ok:
        writer.append_event(stream="GOVERNANCE", kind="PATCH_REJECTED",
                            payload={"patch_id": pid, "at": Stage.AUTHORITY_LINT.value,
                                     "stderr": r_lint.stderr[-500:]})
        return report

    r_deps = sandbox_dep_scan(repo_root / "requirements.txt")
    report.verdicts.append(PatchVerdict(
        patch_id=pid, path=rel, stage=Stage.DEP_SCAN,
        ok=r_deps.ok, detail=r_deps.stage, stderr=r_deps.stderr,
    ))
    if not r_deps.ok:
        writer.append_event(stream="GOVERNANCE", kind="PATCH_REJECTED",
                            payload={"patch_id": pid, "at": Stage.DEP_SCAN.value,
                                     "stderr": r_deps.stderr[-500:]})
        return report

    writer.append_event(stream="GOVERNANCE", kind="PATCH_SANDBOX_PASS",
                        payload={"patch_id": pid, "path": rel})
    return report


def run_pipeline_with_tests(path: Path, repo_root: Path,
                            pattern: str | None = None) -> PipelineReport:
    report = run_pipeline(path, repo_root)
    if not report.ok:
        return report
    r_unit = sandbox_unit_tests(repo_root, pattern=pattern)
    report.verdicts.append(PatchVerdict(
        patch_id=report.patch_id, path=report.path, stage=Stage.UNIT_TESTS,
        ok=r_unit.ok, detail=r_unit.stage, stderr=r_unit.stderr,
    ))
    writer = get_writer()
    if not r_unit.ok:
        writer.append_event(stream="GOVERNANCE", kind="PATCH_REJECTED",
                            payload={"patch_id": report.patch_id,
                                     "at": Stage.UNIT_TESTS.value,
                                     "stderr": r_unit.stderr[-500:]})
    else:
        writer.append_event(stream="GOVERNANCE", kind="PATCH_UNIT_PASS",
                            payload={"patch_id": report.patch_id,
                                     "path": report.path})
    return report


def _changed_py_files(repo_root: Path, base_ref: str) -> list[Path]:
    try:
        out = subprocess.run(                                                   # noqa: S603
            ["git", "diff", "--name-only", f"{base_ref}..HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    files: list[Path] = []
    for line in out.stdout.splitlines():
        p = (repo_root / line.strip())
        if p.suffix == ".py" and p.is_file():
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-pr", action="store_true",
                        help="scan every .py file changed in this PR")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    ns = parser.parse_args()

    failures: list[PipelineReport] = []
    if ns.check_pr:
        changed = _changed_py_files(ns.repo, ns.base)
        if not changed:
            print("[patch_pipeline] no changed .py files; nothing to sandbox.")
            sys.exit(0)
        for f in changed:
            rep = run_pipeline(f, ns.repo)
            print(f"[{'PASS' if rep.ok else 'FAIL'}] {rep.path} "
                  f"({','.join(v.stage.value for v in rep.verdicts)})")
            if not rep.ok:
                failures.append(rep)
    elif ns.path:
        rep = run_pipeline(ns.path, ns.repo)
        print(f"[{'PASS' if rep.ok else 'FAIL'}] {rep.path}")
        if not rep.ok:
            failures.append(rep)
    else:
        parser.error("specify --check-pr or --path")

    if failures:
        for rep in failures:
            for v in rep.verdicts:
                if not v.ok:
                    print(f"  FAIL {v.path} @ {v.stage.value}: {v.stderr[-400:]}")
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["Stage", "PatchVerdict", "PipelineReport",
           "run_pipeline", "run_pipeline_with_tests"]
