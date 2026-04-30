"""Tests for ``tools/rust_revival_reminder.py``.

These tests pin the calendar logic that decides whether the daily CI
reminder is silent / warns / opens an issue. The script's GitHub-API
side effects are exercised via stubs so the suite has zero network
dependencies.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from tools import rust_revival_reminder as rrr

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_SCHEDULE = REPO_ROOT / "docs" / "rust_revival_schedule.yaml"


def _write_schedule(
    tmp_path: Path,
    *,
    deletion: date,
    shadow_days: int = 30,
    warning_days: int = 5,
) -> Path:
    body = {
        "deletion_iso_date": deletion.isoformat(),
        "shadow_window_days": shadow_days,
        "warning_window_days": warning_days,
        "reminder_issue": {
            "title": "[REMINDER] test rust revival",
            "labels": ["rust-revival"],
            "body": "test body",
        },
    }
    path = tmp_path / "rust_revival_schedule.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_production_schedule_loads() -> None:
    """The committed schedule must parse without error."""
    schedule = rrr.load_schedule(PROD_SCHEDULE)
    assert schedule.shadow_window_days == 30
    assert 0 < schedule.warning_window_days < schedule.shadow_window_days
    assert schedule.issue_title.startswith("[REMINDER]")
    assert "checklist" in schedule.issue_body.lower() or "[ ]" in schedule.issue_body


def test_classify_silent_far_before_window(tmp_path: Path) -> None:
    schedule = rrr.load_schedule(
        _write_schedule(tmp_path, deletion=date(2026, 1, 1))
    )
    # 0 days after deletion -> 30 days remaining, silent.
    assert rrr.classify(schedule, date(2026, 1, 1)) == "silent"
    # day 24 -> 6 days remaining, still silent (warning starts at 5).
    assert rrr.classify(schedule, date(2026, 1, 25)) == "silent"


def test_classify_warning_inside_warning_window(tmp_path: Path) -> None:
    schedule = rrr.load_schedule(
        _write_schedule(tmp_path, deletion=date(2026, 1, 1))
    )
    # day 25 -> 5 days remaining
    assert rrr.classify(schedule, date(2026, 1, 26)) == "warning"
    # day 29 -> 1 day remaining
    assert rrr.classify(schedule, date(2026, 1, 30)) == "warning"


def test_classify_due_on_and_after_revival_date(tmp_path: Path) -> None:
    schedule = rrr.load_schedule(
        _write_schedule(tmp_path, deletion=date(2026, 1, 1))
    )
    # day 30 -> 0 remaining -> due
    assert rrr.classify(schedule, date(2026, 1, 31)) == "due"
    # day 31 -> -1 -> still due (idempotent past the cutoff)
    assert rrr.classify(schedule, date(2026, 2, 1)) == "due"
    # 100 days later -> still due
    assert rrr.classify(
        schedule,
        date(2026, 1, 1) + timedelta(days=100),
    ) == "due"


def test_main_silent_path_prints_silent_and_no_api_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be touched on silent path")

    monkeypatch.setattr(rrr, "_gh_request", _boom)
    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-01-01",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("silent:")


def test_main_warning_path_prints_warning_and_no_api_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be touched on warning path")

    monkeypatch.setattr(rrr, "_gh_request", _boom)
    # day 27 -> 3 remaining -> warning
    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-01-28",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "::warning::Rust revival eligible in 3 day" in out


def test_main_due_path_opens_issue_when_no_existing_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))
    monkeypatch.setenv("GH_TOKEN", "ghs_fake")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    calls: list[tuple[str, str]] = []

    def _fake_existing(_repo: str, _title: str, _token: str) -> tuple[bool, bool]:
        calls.append(("search", _title))
        return False, False

    def _fake_open(_repo: str, _schedule: rrr.Schedule, _token: str) -> str:
        calls.append(("open", _schedule.issue_title))
        return "https://example/issue/1"

    monkeypatch.setattr(rrr, "existing_issue_for", _fake_existing)
    monkeypatch.setattr(rrr, "open_reminder_issue", _fake_open)

    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-02-01",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "::warning::Rust revival eligible" in out
    assert "opened reminder issue: https://example/issue/1" in out
    assert calls == [
        ("search", "[REMINDER] test rust revival"),
        ("open", "[REMINDER] test rust revival"),
    ]


def test_main_due_path_skips_when_open_issue_already_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))
    monkeypatch.setenv("GH_TOKEN", "ghs_fake")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    monkeypatch.setattr(
        rrr,
        "existing_issue_for",
        lambda *_args, **_kwargs: (True, False),
    )

    def _boom_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not open another issue when one is open")

    monkeypatch.setattr(rrr, "open_reminder_issue", _boom_open)

    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-02-01",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "reminder issue already open" in out


def test_main_due_path_skips_when_closed_issue_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operator-acknowledged closure: never reopen automatically."""
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))
    monkeypatch.setenv("GH_TOKEN", "ghs_fake")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    monkeypatch.setattr(
        rrr,
        "existing_issue_for",
        lambda *_args, **_kwargs: (False, True),
    )

    def _boom_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not reopen after operator closes the reminder")

    monkeypatch.setattr(rrr, "open_reminder_issue", _boom_open)

    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-02-01",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "previously closed" in out


def test_main_due_path_no_open_issue_flag_does_not_call_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not call API without --open-issue")

    monkeypatch.setattr(rrr, "existing_issue_for", _boom)
    monkeypatch.setattr(rrr, "open_reminder_issue", _boom)

    # No --open-issue flag.
    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-02-01",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "::warning::Rust revival eligible" in out


def test_main_due_path_open_issue_without_token_warns_and_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schedule_path = _write_schedule(tmp_path, deletion=date(2030, 1, 1))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not call API without credentials")

    monkeypatch.setattr(rrr, "existing_issue_for", _boom)
    monkeypatch.setattr(rrr, "open_reminder_issue", _boom)

    rc = rrr.main(
        [
            "--schedule",
            str(schedule_path),
            "--today",
            "2030-02-01",
            "--open-issue",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "GH_TOKEN/GITHUB_REPOSITORY missing" in out


def test_revival_date_is_deletion_plus_shadow_window(tmp_path: Path) -> None:
    schedule = rrr.load_schedule(
        _write_schedule(
            tmp_path,
            deletion=date(2026, 1, 1),
            shadow_days=30,
        )
    )
    assert schedule.revival_date == date(2026, 1, 31)
    assert schedule.days_remaining(date(2026, 1, 31)) == 0
    assert schedule.days_remaining(date(2026, 1, 16)) == 15
