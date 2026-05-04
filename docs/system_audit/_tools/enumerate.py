"""Phase 1 — enumerate every git-tracked file into file_index.csv.

Output schema: file_id,path,size_bytes,lang,sha256

* ``file_id`` is a stable zero-padded sequence number assigned in
  sorted-path order, so re-runs produce identical IDs as long as the
  file set is unchanged.
* ``lang`` is derived from the suffix and the directory (so a yaml
  under ``registry/`` is tagged ``registry-yaml`` to make filtering
  in tracking.csv more useful).
* ``sha256`` lets us detect drift between the snapshot and the live
  tree at validation time (Phase 4).

Pure stdlib; no external deps so it runs unconditionally on any box.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs" / "system_audit" / "file_index.csv"

LANG_BY_SUFFIX = {
    ".py": "python",
    ".tsx": "tsx",
    ".ts": "ts",
    ".js": "js",
    ".jsx": "jsx",
    ".css": "css",
    ".html": "html",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".bat": "batch",
    ".ps1": "powershell",
    ".proto": "protobuf",
    ".txt": "text",
    ".gitignore": "ignore",
    ".sh": "shell",
    ".cfg": "config",
    ".ini": "config",
}


def detect_lang(path: Path) -> str:
    suffix = path.suffix.lower()
    base = LANG_BY_SUFFIX.get(suffix, suffix.lstrip(".") or "unknown")
    parts = path.parts
    if parts and parts[0] == "registry" and base == "yaml":
        return "registry-yaml"
    if parts and parts[0] == "tests" and base == "python":
        return "test-python"
    if parts and parts[0] == "tools" and base == "python":
        return "lint-python"
    if parts and parts[0] == ".github":
        return "ci-config"
    if parts and parts[0] == "docs":
        return f"doc-{base}"
    return base


def main() -> int:
    paths = (
        subprocess.check_output(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True
        )
        .strip()
        .splitlines()
    )
    paths.sort()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "path", "size_bytes", "lang", "sha256"])
        for idx, rel in enumerate(paths):
            p = REPO_ROOT / rel
            if not p.is_file():
                # symlink targets / submodules — skip but flag.
                writer.writerow(
                    [
                        f"F{idx:05d}",
                        rel,
                        0,
                        "missing",
                        "",
                    ]
                )
                continue
            data = p.read_bytes()
            writer.writerow(
                [
                    f"F{idx:05d}",
                    rel,
                    len(data),
                    detect_lang(Path(rel)),
                    hashlib.sha256(data).hexdigest(),
                ]
            )
    print(f"wrote {OUT.relative_to(REPO_ROOT)} with {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
