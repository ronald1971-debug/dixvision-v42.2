"""Phase 2 — build tracking.csv from file_index.csv.

Schema: file_id,path,lang,bucket,status,analyzed,issues_found

* ``bucket`` groups files by top-level directory so the per-directory
  pass (Phase 3b) can iterate one bucket at a time without missing
  files.
* ``status`` starts as ``pending`` for everything.
* ``analyzed`` is ``no`` for everything until Phase 3 marks it ``yes``.
* ``issues_found`` is the count of static-scan + deep-read issues
  (filled by phase 3 / 3b writers).
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "docs" / "system_audit" / "file_index.csv"
OUT = REPO_ROOT / "docs" / "system_audit" / "tracking.csv"


def bucket_for(path: str) -> str:
    parts = path.split("/")
    if not parts:
        return "root"
    if len(parts) == 1:
        return "_root"
    return parts[0]


def main() -> int:
    with INDEX.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "file_id",
                "path",
                "lang",
                "bucket",
                "status",
                "analyzed",
                "issues_found",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["file_id"],
                    r["path"],
                    r["lang"],
                    bucket_for(r["path"]),
                    "pending",
                    "no",
                    "",
                ]
            )
    print(f"wrote {OUT.relative_to(REPO_ROOT)} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
