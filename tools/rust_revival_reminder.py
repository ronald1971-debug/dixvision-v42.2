#!/usr/bin/env python3
"""Rust revival reminder.

Reads ``docs/rust_revival_schedule.yaml`` and emits a CI signal as the
30-day shadow window after Rust deletion winds down. See the schedule
file's header comment for the rationale.

Behaviour, given today's date and the schedule's
``deletion_iso_date`` + ``shadow_window_days`` + ``warning_window_days``:

* days_remaining > warning_window_days
    -> exit 0, print ``"silent"``.
* 0 < days_remaining <= warning_window_days
    -> exit 0, print ``"::warning::Rust revival eligible in N days ..."``.
* days_remaining <= 0
    -> exit 0, print ``"::warning::Rust revival eligible -- open issue"``
       and (when ``--open-issue`` is passed and ``GH_TOKEN`` is set in
       the environment) open the reminder issue. The issue is opened
       at most once: if a closed issue with the same title already
       exists the script does nothing further (closed = operator
       acknowledged the reminder).

The script is intentionally a single self-contained file with no third-
party imports beyond ``yaml``; the daily CI workflow runs it without
installing the project's dev dependencies.

Run locally with ``--today YYYY-MM-DD`` to dry-run a future date.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = REPO_ROOT / "docs" / "rust_revival_schedule.yaml"


@dataclass(frozen=True, slots=True)
class Schedule:
    deletion_date: date
    shadow_window_days: int
    warning_window_days: int
    issue_title: str
    issue_labels: tuple[str, ...]
    issue_body: str

    @property
    def revival_date(self) -> date:
        from datetime import timedelta

        return self.deletion_date + timedelta(days=self.shadow_window_days)

    def days_remaining(self, today: date) -> int:
        return (self.revival_date - today).days


def load_schedule(path: Path = SCHEDULE_PATH) -> Schedule:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    deletion = raw["deletion_iso_date"]
    if isinstance(deletion, str):
        deletion_date = date.fromisoformat(deletion)
    elif isinstance(deletion, date):
        deletion_date = deletion
    else:
        raise TypeError(
            f"deletion_iso_date must be ISO string or date, got {type(deletion)!r}"
        )
    issue = raw["reminder_issue"]
    return Schedule(
        deletion_date=deletion_date,
        shadow_window_days=int(raw["shadow_window_days"]),
        warning_window_days=int(raw["warning_window_days"]),
        issue_title=str(issue["title"]),
        issue_labels=tuple(str(label) for label in issue.get("labels", ())),
        issue_body=str(issue["body"]),
    )


def classify(schedule: Schedule, today: date) -> str:
    """Return one of ``silent`` / ``warning`` / ``due``."""
    remaining = schedule.days_remaining(today)
    if remaining > schedule.warning_window_days:
        return "silent"
    if remaining > 0:
        return "warning"
    return "due"


def _gh_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - GitHub API only
        return json.loads(resp.read().decode("utf-8"))


def existing_issue_for(
    repo: str,
    title: str,
    token: str,
) -> tuple[bool, bool]:
    """Return ``(exists_open, exists_closed)`` for issues matching ``title``."""
    url = (
        "https://api.github.com/search/issues?q="
        + urllib.request.quote(f'repo:{repo} is:issue in:title "{title}"')
    )
    result = _gh_request("GET", url, token)
    items = result.get("items", []) if isinstance(result, dict) else []
    open_hit = any(item.get("state") == "open" for item in items)
    closed_hit = any(item.get("state") == "closed" for item in items)
    return open_hit, closed_hit


def open_reminder_issue(
    repo: str,
    schedule: Schedule,
    token: str,
) -> str:
    payload = {
        "title": schedule.issue_title,
        "body": schedule.issue_body,
        "labels": list(schedule.issue_labels),
    }
    url = f"https://api.github.com/repos/{repo}/issues"
    created = _gh_request("POST", url, token, payload)
    return str(created.get("html_url", ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),  # noqa: DTZ011 - reminder is calendar-day driven
        help="Override today's date for testing (ISO format).",
    )
    parser.add_argument(
        "--open-issue",
        action="store_true",
        help=(
            "When status is 'due', open the reminder issue via the GitHub "
            "API. Requires GH_TOKEN and GITHUB_REPOSITORY in env."
        ),
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=SCHEDULE_PATH,
        help="Path to the schedule YAML.",
    )
    args = parser.parse_args(argv)

    schedule = load_schedule(args.schedule)
    status = classify(schedule, args.today)
    remaining = schedule.days_remaining(args.today)

    if status == "silent":
        print(f"silent: {remaining} days until Rust revival eligible")
        return 0

    if status == "warning":
        print(
            f"::warning::Rust revival eligible in {remaining} day(s) "
            f"({schedule.revival_date.isoformat()}) -- review shadow-mode data"
        )
        return 0

    # due
    print(
        f"::warning::Rust revival eligible -- shadow window of "
        f"{schedule.shadow_window_days} days has elapsed"
    )

    if not args.open_issue:
        return 0

    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("::warning::GH_TOKEN/GITHUB_REPOSITORY missing -- skip issue open")
        return 0

    exists_open, exists_closed = existing_issue_for(repo, schedule.issue_title, token)
    if exists_open:
        print("reminder issue already open -- nothing to do")
        return 0
    if exists_closed:
        print("reminder issue previously closed -- operator acknowledged, idempotent skip")
        return 0

    url = open_reminder_issue(repo, schedule, token)
    print(f"opened reminder issue: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
