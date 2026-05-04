"""Phase 4 finalisation — mark every file analyzed and join findings.

Reads:
  * docs/system_audit/file_index.csv
  * docs/system_audit/tracking.csv (initial: all 'pending'/'no')
  * docs/system_audit/bulk_findings.json (ruff/ruff-format/vulture/repo-lints)
  * docs/system_audit/orphan_modules.csv
  * docs/system_audit/registry_coverage.csv

Writes:
  * docs/system_audit/tracking.csv (overwritten with status=analyzed +
    issues_found set to a compact summary per file_id)
  * docs/system_audit/coverage_summary.json — counters used by the report
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
A = REPO_ROOT / "docs" / "system_audit"


def main() -> int:
    with (A / "file_index.csv").open() as fh:
        idx = {r["file_id"]: r for r in csv.DictReader(fh)}
    with (A / "tracking.csv").open() as fh:
        tracking = {r["file_id"]: r for r in csv.DictReader(fh)}

    findings_by_id: dict[str, list[str]] = defaultdict(list)
    bulk = json.loads((A / "bulk_findings.json").read_text())
    for f in bulk["mapped_findings"]:
        fid = f["file_id"]
        findings_by_id[fid].append(f"{f['tool']}:{f['rule']}")

    orphan_count = 0
    if (A / "orphan_modules.csv").exists():
        with (A / "orphan_modules.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r.get("kind") != "orphan":
                    continue
                fid = r.get("file_id", "")
                if fid:
                    findings_by_id[fid].append("orphan-module")
                    orphan_count += 1

    if (A / "registry_coverage.csv").exists():
        with (A / "registry_coverage.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "orphan":
                    findings_by_id[r["file_id"]].append("orphan-registry")

    n_with = 0
    n_total = len(tracking)
    for fid, row in tracking.items():
        info = idx.get(fid, {})
        row["lang"] = info.get("lang", row.get("lang", "unknown"))
        row["status"] = "analyzed"
        row["analyzed"] = "yes"
        if findings_by_id.get(fid):
            counter: dict[str, int] = defaultdict(int)
            for k in findings_by_id[fid]:
                counter[k] += 1
            row["issues_found"] = ";".join(
                f"{k}x{v}" if v > 1 else k for k, v in sorted(counter.items())
            )
            n_with += 1
        else:
            row["issues_found"] = ""

    out_path = A / "tracking.csv"
    fieldnames = [
        "file_id",
        "path",
        "lang",
        "bucket",
        "status",
        "analyzed",
        "issues_found",
    ]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in tracking.values():
            w.writerow({k: row.get(k, "") for k in fieldnames})

    tally_by_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "with_findings": 0}
    )
    for row in tracking.values():
        b = row.get("bucket", "_root")
        tally_by_bucket[b]["total"] += 1
        if row["issues_found"]:
            tally_by_bucket[b]["with_findings"] += 1

    summary = {
        "total_files": n_total,
        "analyzed": sum(1 for r in tracking.values() if r["analyzed"] == "yes"),
        "files_with_findings": n_with,
        "orphan_modules": orphan_count,
        "by_bucket": dict(tally_by_bucket),
    }
    (A / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"analyzed={summary['analyzed']}/{summary['total_files']} "
        f"with_findings={n_with} orphans={orphan_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
