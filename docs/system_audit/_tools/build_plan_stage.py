"""Reconcile docs/directory_tree.md vs current filesystem.

Parses the tree's path entries from the ``text``-fenced block, normalises
them, then for every entry checks whether the path exists on disk.

Outputs:
  * docs/system_audit/build_plan_stage.csv — [path, kind, exists]
  * docs/system_audit/build_plan_stage.json — coverage by top-level dir
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TREE = REPO_ROOT / "docs" / "directory_tree.md"
A = REPO_ROOT / "docs" / "system_audit"

# Collapse box-drawing prefix and split off the comment column.
LEAF_RE = re.compile(r"^[│├└─\s]*([\w./_\-]+/?)\s*(#.*)?$")


def parse_tree() -> list[tuple[str, str]]:
    """Return list of (path, kind) where kind ∈ {dir, file}."""
    in_block = False
    stack: list[tuple[int, str]] = []  # (indent_cols, segment)
    out: list[tuple[str, str]] = []
    for raw in TREE.read_text().splitlines():
        if raw.strip() == "```text":
            in_block = True
            continue
        if in_block and raw.strip() == "```":
            break
        if not in_block or not raw.strip():
            continue
        # Compute the leading-prefix length up to the first non-tree char.
        prefix_len = 0
        for ch in raw:
            if ch in "│├└─ ":
                prefix_len += 1
            else:
                break
        body = raw[prefix_len:]
        m = LEAF_RE.match(raw)
        if not m:
            continue
        token = m.group(1).strip()
        if not token:
            continue
        # Indent depth: roughly prefix_len // 4.
        depth = prefix_len // 4
        # Pop the stack to current depth.
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if token.endswith("/"):
            seg = token.rstrip("/")
            full = "/".join([s for _, s in stack] + [seg])
            stack.append((depth, seg))
            out.append((full + "/", "dir"))
        else:
            full = "/".join([s for _, s in stack] + [token])
            out.append((full, "file"))
    return out


def main() -> int:
    A.mkdir(parents=True, exist_ok=True)
    entries = parse_tree()
    # Keep unique, drop the synthetic 'dixvision-v42.2/' root.
    seen: set[str] = set()
    rows = []
    for path, kind in entries:
        clean = path
        if clean.startswith("dixvision-v42.2/"):
            clean = clean[len("dixvision-v42.2/"):]
        if not clean or clean in seen:
            continue
        seen.add(clean)
        on_disk = (REPO_ROOT / clean.rstrip("/")).exists()
        rows.append((clean, kind, on_disk))

    csv_path = A / "build_plan_stage.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "kind", "exists"])
        for path, kind, ok in rows:
            w.writerow([path, kind, "yes" if ok else "no"])

    by_pkg: dict[str, dict[str, int]] = {}
    for path, kind, ok in rows:
        if kind == "dir":
            continue
        top = path.split("/", 1)[0] if "/" in path else "_root"
        d = by_pkg.setdefault(top, {"total": 0, "exists": 0})
        d["total"] += 1
        if ok:
            d["exists"] += 1
    for d in by_pkg.values():
        d["coverage_pct"] = round(100 * d["exists"] / max(d["total"], 1), 1)

    total_files = sum(d["total"] for d in by_pkg.values())
    on_disk = sum(d["exists"] for d in by_pkg.values())
    summary = {
        "tree_total_files": total_files,
        "on_disk": on_disk,
        "coverage_pct": round(100 * on_disk / max(total_files, 1), 1),
        "by_package": dict(sorted(by_pkg.items())),
    }
    (A / "build_plan_stage.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"tree_files={total_files} on_disk={on_disk} "
        f"coverage={summary['coverage_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
