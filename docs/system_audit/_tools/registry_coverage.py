"""Phase 3 — verify every YAML under registry/ is loaded somewhere.

A YAML file in ``registry/`` that no Python module references by path
is dead config — every adapter PR is supposed to land its config rows
alongside the consuming code, so any unreferenced row is a bug or a
phantom.

Output: ``registry_coverage.csv`` with ``[file_id, path, referenced_by,
status]``. ``status`` is ``ok`` (>=1 reference) or ``orphan``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "docs" / "system_audit" / "file_index.csv"
OUT = REPO_ROOT / "docs" / "system_audit" / "registry_coverage.csv"


def main() -> int:
    with INDEX.open() as fh:
        rows = list(csv.DictReader(fh))

    yaml_rows = [r for r in rows if r["lang"] == "registry-yaml"]
    py_rows = [r for r in rows if r["lang"] in {"python", "test-python", "lint-python"}]

    py_text: dict[str, str] = {}
    for r in py_rows:
        try:
            py_text[r["path"]] = (
                (REPO_ROOT / r["path"]).read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            py_text[r["path"]] = ""

    findings = []
    for r in yaml_rows:
        path = r["path"]
        # Match registry path with or without 'registry/' prefix.
        bare = path.removeprefix("registry/")
        # Normalize backslashes too (just in case).
        patterns = [
            re.escape(path),
            re.escape(bare),
            re.escape(Path(path).name),
        ]
        rx = re.compile("|".join(patterns))
        refs: list[str] = []
        for py_path, src in py_text.items():
            if rx.search(src):
                refs.append(py_path)
        status = "ok" if refs else "orphan"
        findings.append(
            {
                "file_id": r["file_id"],
                "path": path,
                "referenced_by": ";".join(sorted(refs)[:5]),
                "n_refs": len(refs),
                "status": status,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["file_id", "path", "n_refs", "status", "referenced_by_first5"]
        )
        for f in findings:
            writer.writerow(
                [
                    f["file_id"],
                    f["path"],
                    f["n_refs"],
                    f["status"],
                    f["referenced_by"],
                ]
            )

    n_orphan = sum(1 for f in findings if f["status"] == "orphan")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {n_orphan} orphan / {len(findings)}")
    for f in findings:
        if f["status"] == "orphan":
            print(f"  orphan: {f['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
