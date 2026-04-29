"""Tests for the credentials storage shim (read merge + write gate)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from system_engine.credentials import storage
from system_engine.credentials.storage import (
    StorageNotWritable,
    is_devin_session,
    resolve_env,
    write_credential,
)

# ----- is_devin_session ---------------------------------------------


def test_is_devin_session_true_with_session_id() -> None:
    assert is_devin_session({"DEVIN_SESSION_ID": "abc"}) is True


def test_is_devin_session_true_with_user_id() -> None:
    assert is_devin_session({"DEVIN_USER_ID": "user-x"}) is True


def test_is_devin_session_true_with_envrc(monkeypatch) -> None:
    # Make filesystem signal go away so we isolate the env signal.
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    assert is_devin_session({"ENVRC": "/opt/.devin/envrc"}) is True


def test_is_devin_session_false_with_clean_env(monkeypatch) -> None:
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    assert is_devin_session({}) is False


def test_is_devin_session_true_with_install_dir(monkeypatch) -> None:
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: True)
    assert is_devin_session({}) is True


# ----- resolve_env --------------------------------------------------


def test_resolve_env_os_environ_wins(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=from-file\nBAR=only-file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from-process")
    merged = resolve_env(dotenv_path=p)
    assert merged["FOO"] == "from-process"  # process wins
    assert merged["BAR"] == "only-file"  # filled from file


def test_resolve_env_with_no_dotenv_file(tmp_path: Path) -> None:
    merged = resolve_env(dotenv_path=tmp_path / "missing.env")
    # Should at least include the current process env unchanged.
    assert merged.get("PATH") == os.environ.get("PATH")


# ----- write_credential ---------------------------------------------


def test_write_credential_creates_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.delenv("DEVIN_USER_ID", raising=False)
    monkeypatch.delenv("ENVRC", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    p = tmp_path / ".env"
    written = write_credential(
        "MY_KEY", "value123", dotenv_path=p, refresh_process_env=False,
    )
    assert written == p
    assert p.read_text(encoding="utf-8") == "MY_KEY=value123\n"


def test_write_credential_refreshes_process_env(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.delenv("DEVIN_USER_ID", raising=False)
    monkeypatch.delenv("ENVRC", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    p = tmp_path / ".env"
    write_credential("MY_KEY", "v", dotenv_path=p)
    assert os.environ["MY_KEY"] == "v"


def test_write_credential_refuses_inside_devin_session(
    monkeypatch, tmp_path: Path,
) -> None:
    # Force Devin-session detection.
    monkeypatch.setenv("DEVIN_SESSION_ID", "abc")
    p = tmp_path / ".env"
    with pytest.raises(StorageNotWritable):
        write_credential(
            "MY_KEY", "v", dotenv_path=p, refresh_process_env=False,
        )
    assert not p.exists()


def test_write_credential_rejects_invalid_name(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.delenv("DEVIN_USER_ID", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    with pytest.raises(ValueError):
        write_credential(
            "1BAD", "v",
            dotenv_path=tmp_path / ".env",
            refresh_process_env=False,
        )


def test_write_credential_rejects_empty_value(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    with pytest.raises(ValueError):
        write_credential(
            "FOO", "",
            dotenv_path=tmp_path / ".env",
            refresh_process_env=False,
        )


def test_write_credential_rejects_newline_value(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    with pytest.raises(ValueError):
        write_credential(
            "FOO", "line1\nline2",
            dotenv_path=tmp_path / ".env",
            refresh_process_env=False,
        )
