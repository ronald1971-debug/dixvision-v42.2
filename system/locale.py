"""
system.locale \u2014 detect the host machine's main language.

Detection order:
    1. DIX_LOCALE environment variable (explicit operator override)
    2. Windows: GetUserDefaultLocaleName via ctypes
    3. POSIX:   LC_ALL / LC_MESSAGES / LANG / LANGUAGE
    4. Python:  locale.getdefaultlocale()
    5. Fallback: "en_US"

Exposes a normalized ISO-639-1 two-letter language code and a full locale
tag. All cockpit UI, chat voices, and hazard messages localize through
this module. Ledger entries remain English for audit stability.
"""
from __future__ import annotations

import locale as _stdlocale
import os
import sys
from dataclasses import dataclass

_SUPPORTED_UI: tuple[str, ...] = (
    "en", "nl", "de", "fr", "es", "it", "pt", "ja",
    "zh", "ko", "ru", "ar", "hi", "tr",
)


@dataclass(frozen=True)
class LocaleInfo:
    language: str          # "en"
    region: str            # "US"
    tag: str               # "en_US"
    source: str            # "env" / "windows" / "posix" / "python" / "fallback"
    ui_supported: bool     # True if language has an i18n.json pack


def _from_env() -> str | None:
    return os.environ.get("DIX_LOCALE") or None


def _from_windows() -> str | None:                    # pragma: no cover - Windows only
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(85)
        n = ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, len(buf))
        if n > 0:
            return buf.value.replace("-", "_")
    except Exception:
        pass
    return None


def _from_posix() -> str | None:
    for key in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        v = os.environ.get(key)
        if v and v.lower() not in ("c", "posix"):
            # "en_US.UTF-8" -> "en_US"
            return v.split(".")[0].split(":")[0]
    return None


def _from_python() -> str | None:
    try:
        lang, _enc = _stdlocale.getdefaultlocale()
        return lang
    except Exception:                                    # pragma: no cover
        return None


def _parse(tag: str | None) -> tuple[str, str] | None:
    if not tag:
        return None
    tag = tag.replace("-", "_")
    parts = tag.split("_")
    lang = parts[0].lower() if parts else ""
    region = parts[1].upper() if len(parts) > 1 else ""
    if len(lang) != 2:
        return None
    return lang, region


def detect() -> LocaleInfo:
    sources = (
        ("env", _from_env),
        ("windows", _from_windows),
        ("posix", _from_posix),
        ("python", _from_python),
    )
    for src, fn in sources:
        p = _parse(fn())
        if p:
            lang, region = p
            return LocaleInfo(
                language=lang, region=region,
                tag=f"{lang}_{region}" if region else lang,
                source=src, ui_supported=lang in _SUPPORTED_UI,
            )
    return LocaleInfo(language="en", region="US", tag="en_US",
                      source="fallback", ui_supported=True)


_cached: LocaleInfo | None = None


def current() -> LocaleInfo:
    global _cached
    if _cached is None:
        _cached = detect()
    return _cached


def set_override(tag: str) -> LocaleInfo:
    """Operator override from the cockpit; takes effect immediately."""
    global _cached
    p = _parse(tag)
    if not p:
        return current()
    lang, region = p
    _cached = LocaleInfo(
        language=lang, region=region,
        tag=f"{lang}_{region}" if region else lang,
        source="override", ui_supported=lang in _SUPPORTED_UI,
    )
    return _cached


def supported_ui_languages() -> tuple[str, ...]:
    return _SUPPORTED_UI


__all__ = ["LocaleInfo", "detect", "current", "set_override", "supported_ui_languages"]
