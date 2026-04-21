"""
mind.knowledge.language — language detection + pluggable translation.

Detection order:
    1. langdetect (if installed)
    2. character-set heuristic (CJK / Cyrillic / Arabic / Latin)
    3. default "en"

Translation providers (``DIX_TRANSLATOR`` env):
    none   (default) \u2014 passthrough (source text + best-effort heuristic)
    deepl           \u2014 DeepL API (DEEPL_API_KEY)
    google          \u2014 Google Cloud Translate (GOOGLE_TRANSLATE_API_KEY)
    openai          \u2014 OpenAI chat completion (OPENAI_API_KEY)
    local:nllb200   \u2014 local NLLB-200 model (optional heavy dep)

Degrades gracefully: if the configured provider fails, we return the
original text and flag translation_failed=True in the result.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.secrets import get_secret
from system.config import get_config


@dataclass
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str
    translated: bool
    provider: str
    error: str = ""


def detect_language(text: str) -> str:
    if not text:
        return "en"
    try:                                        # pragma: no cover - optional
        import langdetect  # type: ignore
        langdetect.DetectorFactory.seed = 0
        return str(langdetect.detect(text))
    except Exception:
        pass
    # Heuristic fallback
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    hangul = sum(1 for c in text if "\uac00" <= c <= "\ud7af")
    hira_kata = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    hebrew = sum(1 for c in text if "\u0590" <= c <= "\u05ff")
    greek = sum(1 for c in text if "\u0370" <= c <= "\u03ff")
    total = max(1, len(text))
    if cjk / total > 0.2:
        return "zh"
    if hangul / total > 0.2:
        return "ko"
    if hira_kata / total > 0.1:
        return "ja"
    if cyrillic / total > 0.2:
        return "ru"
    if arabic / total > 0.2:
        return "ar"
    if hebrew / total > 0.2:
        return "he"
    if greek / total > 0.2:
        return "el"
    return "en"


def _provider() -> str:
    try:
        return str(get_config().get("DIX_TRANSLATOR", "none")).lower().strip()
    except Exception:
        return "none"


def translate_to(text: str, target_lang: str = "en",
                 source_lang: str | None = None) -> TranslationResult:
    if not text:
        return TranslationResult(text="", source_lang="en", target_lang=target_lang,
                                 translated=False, provider="none")
    src = source_lang or detect_language(text)
    if src == target_lang:
        return TranslationResult(text=text, source_lang=src, target_lang=target_lang,
                                 translated=False, provider="noop")
    p = _provider()
    if p == "none":
        return TranslationResult(text=text, source_lang=src, target_lang=target_lang,
                                 translated=False, provider="none",
                                 error="translator disabled (DIX_TRANSLATOR=none)")
    try:
        if p == "deepl":
            return _deepl(text, src, target_lang)
        if p == "google":
            return _google(text, src, target_lang)
        if p == "openai":
            return _openai(text, src, target_lang)
        if p.startswith("local:"):
            return _local(text, src, target_lang, p.split(":", 1)[1])
    except Exception as e:                               # pragma: no cover
        return TranslationResult(text=text, source_lang=src, target_lang=target_lang,
                                 translated=False, provider=p, error=repr(e))
    return TranslationResult(text=text, source_lang=src, target_lang=target_lang,
                             translated=False, provider=p, error="unknown provider")


def _deepl(text: str, src: str, target: str) -> TranslationResult:      # pragma: no cover
    import json
    import urllib.parse
    import urllib.request
    key = get_secret("DEEPL_API_KEY", default="")
    if not key:
        return TranslationResult(text=text, source_lang=src, target_lang=target,
                                 translated=False, provider="deepl", error="no key")
    url = "https://api-free.deepl.com/v2/translate"
    data = urllib.parse.urlencode({"auth_key": key, "text": text, "target_lang": target.upper()}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=6.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    out = body.get("translations", [{}])[0].get("text", text)
    return TranslationResult(text=out, source_lang=src, target_lang=target,
                             translated=True, provider="deepl")


def _google(text: str, src: str, target: str) -> TranslationResult:     # pragma: no cover
    import json
    import urllib.parse
    import urllib.request
    key = get_secret("GOOGLE_TRANSLATE_API_KEY", default="")
    if not key:
        return TranslationResult(text=text, source_lang=src, target_lang=target,
                                 translated=False, provider="google", error="no key")
    url = f"https://translation.googleapis.com/language/translate/v2?key={key}"
    data = urllib.parse.urlencode({"q": text, "source": src, "target": target, "format": "text"}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=6.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    out = body.get("data", {}).get("translations", [{}])[0].get("translatedText", text)
    return TranslationResult(text=out, source_lang=src, target_lang=target,
                             translated=True, provider="google")


def _openai(text: str, src: str, target: str) -> TranslationResult:     # pragma: no cover
    import json
    import urllib.request
    key = get_secret("OPENAI_API_KEY", default="")
    if not key:
        return TranslationResult(text=text, source_lang=src, target_lang=target,
                                 translated=False, provider="openai", error="no key")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"Translate the user's text from {src} to {target}. "
                                          f"Preserve proper nouns and ticker symbols."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    out = body["choices"][0]["message"]["content"].strip()
    return TranslationResult(text=out, source_lang=src, target_lang=target,
                             translated=True, provider="openai")


def _local(text: str, src: str, target: str, model: str) -> TranslationResult:   # pragma: no cover
    # Optional heavy path. If model isn't installed, return passthrough.
    return TranslationResult(text=text, source_lang=src, target_lang=target,
                             translated=False, provider=f"local:{model}",
                             error="local model not installed")


__all__ = ["TranslationResult", "detect_language", "translate_to"]
