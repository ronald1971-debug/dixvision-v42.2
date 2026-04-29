"""Tests for the minimal stdlib-only ``.env`` parser/writer."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from system_engine.credentials.dotenv_io import (
    is_valid_env_var_name,
    load_dotenv_file,
    parse_dotenv,
    update_dotenv_file,
)

# ----- name validation ----------------------------------------------


@pytest.mark.parametrize(
    "name,ok",
    [
        ("FOO", True),
        ("FOO_BAR", True),
        ("_FOO", True),
        ("FOO123", True),
        ("foo", True),
        ("123FOO", False),
        ("FOO BAR", False),
        ("FOO-BAR", False),
        ("", False),
        ("FOO=BAR", False),
    ],
)
def test_is_valid_env_var_name(name: str, ok: bool) -> None:
    assert is_valid_env_var_name(name) is ok


# ----- parse --------------------------------------------------------


def test_parse_simple() -> None:
    text = "FOO=bar\nBAZ=qux\n"
    assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_strips_quotes() -> None:
    text = 'FOO="hello world"\nBAR=\'single quoted\'\n'
    assert parse_dotenv(text) == {
        "FOO": "hello world",
        "BAR": "single quoted",
    }


def test_parse_skips_comments_and_blanks() -> None:
    text = "# header\n\nFOO=bar\n   # indented\nBAZ=qux\n"
    assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_handles_export_prefix() -> None:
    text = "export FOO=bar\nexport BAZ=qux\n"
    assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_later_value_wins() -> None:
    text = "FOO=first\nFOO=second\n"
    assert parse_dotenv(text) == {"FOO": "second"}


def test_parse_skips_garbage_lines() -> None:
    text = "FOO=bar\nthis is not a key\nBAZ=qux\n"
    assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}


def test_load_dotenv_file_missing(tmp_path: Path) -> None:
    assert load_dotenv_file(tmp_path / "no.env") == {}


def test_load_dotenv_file_round_trips(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
    assert load_dotenv_file(p) == {"FOO": "bar", "BAZ": "qux"}


# ----- update -------------------------------------------------------


def test_update_creates_file(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    final = update_dotenv_file(p, {"FOO": "bar"})
    assert final == {"FOO": "bar"}
    assert p.read_text(encoding="utf-8") == "FOO=bar\n"


def test_update_replaces_existing_key_in_place(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text(
        "# header\nFOO=old\n# spacer\nBAR=keep\n", encoding="utf-8"
    )
    update_dotenv_file(p, {"FOO": "new"})
    text = p.read_text(encoding="utf-8")
    # Order/comment preserved, only FOO line rewritten:
    assert text == "# header\nFOO=new\n# spacer\nBAR=keep\n"


def test_update_appends_new_keys_sorted(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n", encoding="utf-8")
    update_dotenv_file(p, {"ZED": "z", "ALPHA": "a"})
    text = p.read_text(encoding="utf-8")
    assert text == "FOO=bar\nALPHA=a\nZED=z\n"


def test_update_quotes_values_with_special_chars(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    update_dotenv_file(p, {"FOO": "hello world"})
    text = p.read_text(encoding="utf-8")
    assert text == 'FOO="hello world"\n'


def test_update_escapes_dollar_and_quotes(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    update_dotenv_file(p, {"FOO": 'a"b$c'})
    text = p.read_text(encoding="utf-8")
    assert text == 'FOO="a\\"b\\$c"\n'


def test_update_is_atomic_and_chmod_0600(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    update_dotenv_file(p, {"FOO": "bar"})
    mode = stat.S_IMODE(os.stat(p).st_mode)
    assert mode == 0o600


def test_update_rejects_invalid_name(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    with pytest.raises(ValueError):
        update_dotenv_file(p, {"123BAD": "x"})


def test_update_rejects_non_string_value(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    with pytest.raises(TypeError):
        update_dotenv_file(p, {"FOO": 42})  # type: ignore[dict-item]


def test_update_rejects_newline_in_value(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    with pytest.raises(ValueError):
        update_dotenv_file(p, {"FOO": "line1\nline2"})


# ----- round-trip ---------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "with space",
        'has"double"quote',
        "has$dollar",
        "has\\backslash",
        'all"three\\$mixed',
        "sk-abc$def",
        "key-with\"quote",
        "key-with\\backslash",
        "trailing space ",
        " leading space",
        "= equals = inside =",
        "#hash-not-comment",
        "",
    ],
)
def test_round_trip_through_file(tmp_path: Path, value: str) -> None:
    """``parse_dotenv(write_dotenv(value)) == value`` for every
    string the writer accepts (i.e. anything without a newline).
    """

    if value == "":
        # Empty values must round-trip through the bare writer
        # path (writer special-cases "" → ``""``).
        p = tmp_path / ".env"
        update_dotenv_file(p, {"FOO": value})
        assert load_dotenv_file(p) == {"FOO": value}
        return

    p = tmp_path / ".env"
    update_dotenv_file(p, {"FOO": value})
    assert load_dotenv_file(p) == {"FOO": value}


def test_round_trip_preserves_via_returned_mapping(tmp_path: Path) -> None:
    """:func:`update_dotenv_file` must return values matching what
    a fresh :func:`load_dotenv_file` would produce — otherwise an
    in-process write disagrees with a post-restart read.
    """

    p = tmp_path / ".env"
    val = 'mixed"$\\value'
    returned = update_dotenv_file(p, {"FOO": val})
    assert returned == {"FOO": val}
    assert load_dotenv_file(p) == {"FOO": val}


def test_update_dedupes_later_occurrences_of_same_key(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=a\nBAR=keep\nFOO=b\n", encoding="utf-8")
    update_dotenv_file(p, {"FOO": "new"})
    text = p.read_text(encoding="utf-8")
    # Single FOO line at the position of the first occurrence; the
    # second duplicate is dropped.
    assert text == "FOO=new\nBAR=keep\n"
