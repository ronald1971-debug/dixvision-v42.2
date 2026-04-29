"""Tests for authority_lint rule B23 (Dashboard-2026 wave-01).

B23 enforces "registry-driven AI providers" — chat widget files
must not contain string literals that name a specific AI vendor.
The rule scans:

* a hard-coded list of static files
  (``ui/static/chat_widget.js`` + the chat HTML pages), and
* Python modules under ``intelligence_engine.cognitive.chat.*`` /
  ``ui.cognitive.chat.*`` (wave-02 future scope, but covered here
  so the rule machinery is exercised today).

The shipping repo must lint clean (the chat widget code is already
registry-driven by construction); the synthetic violation cases
prove the rule actually fires on bad input.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import tools.authority_lint as al

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_passes_b23() -> None:
    """The shipping chat widget files must contain no forbidden tokens."""

    violations = al.lint_repo(REPO_ROOT)
    b23 = [v for v in violations if v.rule == "B23"]
    assert b23 == [], "shipping repo has B23 violations: " + "\n".join(
        v.format(REPO_ROOT) for v in b23
    )


def test_static_check_fires_on_forbidden_token(tmp_path: Path) -> None:
    """Synthetic chat widget JS that hard-codes a vendor must fail B23."""

    static_dir = tmp_path / "ui" / "static"
    static_dir.mkdir(parents=True)

    bad_js = static_dir / "chat_widget.js"
    # "openai" is in FORBIDDEN_AI_PROVIDER_TOKENS.
    bad_js.write_text(
        "const provider = 'openai';\nfetch('/api/openai/chat');\n",
        encoding="utf-8",
    )

    out = al._check_b23_static(tmp_path)
    rules = {v.rule for v in out}
    assert "B23" in rules
    targets = {v.imported for v in out}
    assert "openai" in targets


def test_static_check_is_silent_when_clean(tmp_path: Path) -> None:
    """A registry-driven JS file with no vendor tokens passes B23."""

    static_dir = tmp_path / "ui" / "static"
    static_dir.mkdir(parents=True)
    clean_js = static_dir / "chat_widget.js"
    clean_js.write_text(
        "const url = '/api/ai/providers';\nfetch(url);\n",
        encoding="utf-8",
    )
    out = al._check_b23_static(tmp_path)
    assert out == []


def test_static_check_is_case_insensitive(tmp_path: Path) -> None:
    static_dir = tmp_path / "ui" / "static"
    static_dir.mkdir(parents=True)
    bad = static_dir / "indira_chat.html"
    # Uppercase form must still trip the rule.
    bad.write_text("<p>Powered by GEMINI</p>\n", encoding="utf-8")
    out = al._check_b23_static(tmp_path)
    assert any(v.rule == "B23" for v in out)


def test_static_check_skips_missing_files(tmp_path: Path) -> None:
    """Missing static files don't crash the rule."""

    out = al._check_b23_static(tmp_path)
    assert out == []


def test_python_check_fires_on_chat_widget_module(tmp_path: Path) -> None:
    """A chat widget Python module hard-coding a vendor must fail B23."""

    src = textwrap.dedent(
        """
        '''Chat widget backend (wave-02 placeholder).'''


        DEFAULT_PROVIDER = "openai"


        def chat(prompt: str) -> str:
            return prompt
        """
    ).strip()
    fake_path = tmp_path / "chat.py"
    fake_path.write_text(src, encoding="utf-8")
    tree = al.ast.parse(src, filename=str(fake_path))
    importer = "intelligence_engine.cognitive.chat.indira"

    out = al._check_b23_python(importer, fake_path, tree)
    assert any(v.rule == "B23" and v.imported == "openai" for v in out)


def test_python_check_is_silent_for_non_chat_modules(tmp_path: Path) -> None:
    """Non-chat modules can mention vendors freely (they're routing data)."""

    # The cognitive router itself is NOT a chat widget — it's the
    # registry-driven routing logic. It must remain free to read
    # provider names from the registry.
    src = "PROVIDER = 'openai'\n"
    fake_path = tmp_path / "router.py"
    fake_path.write_text(src, encoding="utf-8")
    tree = al.ast.parse(src, filename=str(fake_path))
    importer = "core.cognitive_router.router"

    out = al._check_b23_python(importer, fake_path, tree)
    assert out == []


def test_python_check_is_silent_when_clean(tmp_path: Path) -> None:
    """A registry-driven chat widget Python module passes B23."""

    src = textwrap.dedent(
        """
        from core.cognitive_router import enabled_ai_providers


        def boot(registry):
            return enabled_ai_providers(registry)
        """
    ).strip()
    fake_path = tmp_path / "chat.py"
    fake_path.write_text(src, encoding="utf-8")
    tree = al.ast.parse(src, filename=str(fake_path))
    importer = "ui.cognitive.chat.dyon"

    out = al._check_b23_python(importer, fake_path, tree)
    assert out == []


def test_forbidden_token_table_is_non_empty() -> None:
    """Sanity: the forbidden token list isn't accidentally cleared."""

    assert len(al.FORBIDDEN_AI_PROVIDER_TOKENS) >= 5
    assert all(
        isinstance(t, str) and t == t.lower()
        for t in al.FORBIDDEN_AI_PROVIDER_TOKENS
    )


def test_chat_widget_static_relatives_are_documented() -> None:
    """The static-file list has stable, predictable paths."""

    assert al.CHAT_WIDGET_STATIC_RELATIVES == (
        ("ui", "static", "chat_widget.js"),
        ("ui", "static", "indira_chat.html"),
        ("ui", "static", "dyon_chat.html"),
    )
