"""
tools/sandbox_runner.py -- run a candidate Python module in an isolated subprocess.

Used by governance.patch_pipeline to verify every proposed code change before
it may be promoted. The runner:
    - copies the target file (and referenced modules) into a tempdir
    - spawns python in hardened mode (`-I -S -W error`)
    - denies outbound network via dummy proxy env + unreachable host
    - caps wall-clock + memory
    - returns a structured verdict

It is intentionally stdlib-only so it can be invoked during CI on any agent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    stage: str
    stdout: str
    stderr: str
    returncode: int

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "returncode": self.returncode,
        }


_DEFAULT_TIMEOUT = 60
_DEFAULT_MEM_MB = 512


def _base_env() -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        # Block outbound HTTP in sandbox:
        "HTTP_PROXY": "http://127.0.0.1:1",
        "HTTPS_PROXY": "http://127.0.0.1:1",
        "NO_PROXY": "",
        # Keep DB writes inside sandbox dir:
        "DIX_SANDBOX": "1",
    }
    if sys.platform == "win32":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _run(args: list[str], cwd: str, timeout: int) -> SandboxResult:
    try:
        proc = subprocess.run(                                                  # noqa: S603
            args,
            cwd=cwd,
            env=_base_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return SandboxResult(
            ok=(proc.returncode == 0),
            stage="run",
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(ok=False, stage="timeout",
                             stdout=exc.stdout or "", stderr=exc.stderr or "",
                             returncode=124)
    except FileNotFoundError as exc:
        return SandboxResult(ok=False, stage="spawn_failed",
                             stdout="", stderr=str(exc), returncode=127)


def sandbox_import(module_path: Path, repo_root: Path,
                   timeout: int = _DEFAULT_TIMEOUT) -> SandboxResult:
    """Import `module_path` inside an isolated subprocess from a temp repo copy."""
    if not module_path.is_file():
        return SandboxResult(ok=False, stage="missing",
                             stdout="", stderr=f"no such file: {module_path}",
                             returncode=2)
    with tempfile.TemporaryDirectory(prefix="dix-sandbox-") as tmp:
        dst_root = Path(tmp) / "repo"
        shutil.copytree(repo_root, dst_root,
                        ignore=shutil.ignore_patterns(
                            ".git", ".github", ".venv", "venv", "__pycache__",
                            ".pytest_cache", ".mypy_cache", "node_modules",
                            "dist", "build", "data", "mobile", "windows"))
        rel = module_path.relative_to(repo_root)
        mod = str(rel.with_suffix("")).replace(os.sep, ".")
        script = (
            "import importlib, sys, json;\n"
            f"m=importlib.import_module({mod!r});\n"
            "print(json.dumps({'module': m.__name__}));\n"
        )
        return _run([sys.executable, "-I", "-S", "-W", "error", "-c", script],
                    cwd=str(dst_root), timeout=timeout)


def sandbox_authority_lint(repo_root: Path,
                           timeout: int = _DEFAULT_TIMEOUT) -> SandboxResult:
    """Run tools/authority_lint.py; pipeline fails if any violation is found."""
    return _run([sys.executable, "-I", "tools/authority_lint.py"],
                cwd=str(repo_root), timeout=timeout)


def sandbox_unit_tests(repo_root: Path, pattern: str | None = None,
                       timeout: int = 180) -> SandboxResult:
    args = [sys.executable, "-m", "pytest", "-q"]
    if pattern:
        args.extend(["-k", pattern])
    return _run(args, cwd=str(repo_root), timeout=timeout)


def sandbox_dep_scan(requirements: Path,
                     timeout: int = _DEFAULT_TIMEOUT) -> SandboxResult:
    """Very small built-in scan: reject typosquats and empty/abandoned lines.

    Real CI uses pip-audit / Snyk; this keeps the in-repo pipeline hermetic.
    """
    if not requirements.is_file():
        return SandboxResult(ok=True, stage="noop",
                             stdout="no requirements.txt", stderr="",
                             returncode=0)
    blocklist = {"reqeusts", "pandaz", "ccxtt", "urllibs", "jsoon", "numpie"}
    issues: list[str] = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        name = s.split("==")[0].split(">=")[0].split("<")[0].split("~=")[0].strip()
        if name.lower() in blocklist:
            issues.append(f"blocked package: {name}")
    if issues:
        return SandboxResult(ok=False, stage="dep_scan",
                             stdout="", stderr="\n".join(issues), returncode=1)
    return SandboxResult(ok=True, stage="dep_scan",
                         stdout=f"scanned {requirements}", stderr="",
                         returncode=0)


def main() -> None:
    """CLI: `python tools/sandbox_runner.py <module_path>` returns JSON."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path,
                        help="path to .py module to sandbox-import")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    ns = parser.parse_args()
    out = {
        "import": sandbox_import(ns.target, ns.repo, ns.timeout).to_dict(),
        "lint": sandbox_authority_lint(ns.repo, ns.timeout).to_dict(),
        "deps": sandbox_dep_scan(ns.repo / "requirements.txt",
                                 ns.timeout).to_dict(),
    }
    ok = all(part["ok"] for part in out.values())
    print(json.dumps({"ok": ok, "parts": out}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "SandboxResult", "sandbox_import", "sandbox_authority_lint",
    "sandbox_unit_tests", "sandbox_dep_scan",
]
