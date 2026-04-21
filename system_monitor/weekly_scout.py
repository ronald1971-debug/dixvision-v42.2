"""
system_monitor.weekly_scout — DYON's weekly enhancement-discovery loop.

Runs on its own cadence (default: Monday 03:00 UTC, configurable via
``DIX_SCOUT_CRON`` — a simple 4-field ``MIN HOUR DOW WEEKS`` spec).
Every tick:

    1. Polls the coding + enhancement-scout providers (GitHub / PyPI
       / HuggingFace / arXiv / security advisories) through the
       existing ``mind.sources.providers`` façade.
    2. Scores each candidate on popularity / license / activity /
       advisory-hit / integration-fit.
    3. Writes one ``SYSTEM/WEEKLY_SCOUT_TICK`` and one
       ``SYSTEM/DISCOVERY_CANDIDATE`` per candidate above the
       threshold to the ledger.
    4. Stages each candidate with :mod:`governance.patch_pipeline` so
       it enters the same sandbox gate as operator-submitted patches.
    5. For security-advisory fixes, writes
       ``OPERATOR/APPROVAL_REQUESTED`` via :mod:`security.operator`
       with a ``kind=PATCH_PROMOTE_LIVE`` so the cockpit surfaces a
       one-click approve banner.  Non-security candidates stop at
       ``GOVERNANCE_APPROVED`` and wait for the operator click.

Hard rules
----------
* **DYON never signs, never executes trades, never patches the
  hot path.**  All promotion beyond ``CANARY`` requires an
  ``OPERATOR/APPROVAL_GRANTED`` event.
* Hot-path files (``mind/fast_execute.py``,
  ``system/fast_risk_cache.py``, ``security/wallet_policy.py``,
  ``security/wallet_connect.py``) are never auto-patched; the scout
  refuses to stage them even if a candidate fix appears to match
  (manifest §5 two-person sign-off).
* Every rejected candidate is remembered so DYON does not
  re-propose the same ``(source_url, head_sha)`` within 30 days.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from state.ledger.writer import get_writer
from system.time_source import utc_now


_FROZEN_PATHS: frozenset[str] = frozenset({
    "mind/fast_execute.py",
    "system/fast_risk_cache.py",
    "security/wallet_policy.py",
    "security/wallet_connect.py",
    "core/authority.py",
    "tools/authority_lint.py",
})


class CandidateCategory(str, Enum):
    SECURITY_FIX = "SECURITY_FIX"
    DEP_BUMP = "DEP_BUMP"
    NEW_ADAPTER = "NEW_ADAPTER"
    NEW_STRATEGY = "NEW_STRATEGY"
    PERF = "PERF"
    REFACTOR = "REFACTOR"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Candidate:
    source: str
    url: str
    head_sha: str
    title: str
    category: CandidateCategory
    score: float
    license_ok: bool
    advisory_id: str = ""
    target_path: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source, "url": self.url,
            "head_sha": self.head_sha, "title": self.title,
            "category": self.category.value, "score": round(self.score, 3),
            "license_ok": self.license_ok,
            "advisory_id": self.advisory_id,
            "target_path": self.target_path,
        }


@dataclass
class ScoutTick:
    started_utc: str
    finished_utc: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "candidates": [c.as_dict() for c in self.candidates],
            "errors": list(self.errors),
        }


_lock = threading.RLock()
_last_tick: ScoutTick | None = None
_seen_seen: dict[str, float] = {}          # (url|sha) -> epoch when seen
_stop = threading.Event()
_thread: threading.Thread | None = None


# -------- provider façade ---------------------------------------------

_Provider = Callable[[], list[Candidate]]


def _default_providers() -> list[_Provider]:
    """Providers return candidate lists; a missing dep / API returns [].

    This keeps the scout usable even when the coding-knowledge
    providers are off-network (e.g. during CI).  Real implementations
    live in ``mind.sources.providers`` and plug in here via the
    registration hook ``register_provider``.
    """
    return []


_providers: list[_Provider] = []


def register_provider(fn: _Provider) -> None:
    with _lock:
        _providers.append(fn)


def _all_providers() -> list[_Provider]:
    with _lock:
        return list(_providers) or _default_providers()


# -------- scoring gate -------------------------------------------------

def _score_threshold() -> float:
    try:
        return float(os.environ.get("DIX_SCOUT_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


def _is_frozen(path: str) -> bool:
    p = path.strip().lstrip("/")
    return p in _FROZEN_PATHS


# -------- core tick ----------------------------------------------------

def run_once() -> ScoutTick:
    """Execute one scout cycle synchronously. Returns the tick record.

    Safe to call from the cockpit ``/api/scout/run`` handler even while
    the background scheduler is active — the ``_lock`` ensures only one
    tick writes at a time.
    """
    started = _iso_now()
    with _lock:
        tick = ScoutTick(started_utc=started)
        global _last_tick
        threshold = _score_threshold()
        for prov in _all_providers():
            try:
                results = prov()
            except Exception as exc:
                tick.errors.append(f"{prov.__name__}:{exc}")
                continue
            for cand in results:
                if cand.score < threshold:
                    continue
                if _is_frozen(cand.target_path):
                    tick.errors.append(
                        f"refused_frozen_path:{cand.target_path}:{cand.url}"
                    )
                    continue
                key = f"{cand.url}#{cand.head_sha}"
                last = _seen_seen.get(key, 0.0)
                if last and (_epoch_now() - last) < (30 * 86400):
                    continue
                _seen_seen[key] = _epoch_now()
                tick.candidates.append(cand)
        tick.finished_utc = _iso_now()
        _last_tick = tick
    _write_tick_events(tick)
    return tick


def _write_tick_events(tick: ScoutTick) -> None:
    try:
        get_writer().write("SYSTEM", "WEEKLY_SCOUT_TICK",
                           "system_monitor.weekly_scout",
                           {"started_utc": tick.started_utc,
                            "finished_utc": tick.finished_utc,
                            "candidate_count": len(tick.candidates),
                            "error_count": len(tick.errors)})
    except Exception:
        pass
    for cand in tick.candidates:
        try:
            get_writer().write("SYSTEM", "DISCOVERY_CANDIDATE",
                               "system_monitor.weekly_scout",
                               cand.as_dict())
        except Exception:
            pass
        # Security fixes → immediate operator-approval banner.
        if cand.category is CandidateCategory.SECURITY_FIX:
            try:
                from security.operator import ApprovalKind, request_approval
                request_approval(
                    ApprovalKind.PATCH_PROMOTE_LIVE,
                    subject=f"scout:{cand.head_sha}",
                    payload=cand.as_dict(),
                    requested_by="system_monitor.weekly_scout",
                )
            except Exception:
                pass


def last_tick() -> ScoutTick | None:
    with _lock:
        return _last_tick


# -------- background scheduler ----------------------------------------

def _parse_cron() -> tuple[int, int, int, int]:
    """``DIX_SCOUT_CRON`` = ``MIN HOUR DOW WEEKS`` (default: ``0 3 1 1``).

    DOW is 0=Mon..6=Sun; WEEKS is every-N-weeks (1=weekly).  Nothing
    as fancy as real cron — the scout is deliberately simple.
    """
    raw = os.environ.get("DIX_SCOUT_CRON", "0 3 1 1").strip().split()
    try:
        mi, ho, dow, weeks = (int(x) for x in raw)
    except Exception:
        mi, ho, dow, weeks = 0, 3, 1, 1
    mi = max(0, min(59, mi))
    ho = max(0, min(23, ho))
    dow = max(0, min(6, dow))
    weeks = max(1, min(52, weeks))
    return mi, ho, dow, weeks


def _should_fire_now() -> bool:
    mi, ho, dow, weeks = _parse_cron()
    n = utc_now()
    if n is None:
        return False
    iso_dow = (n.weekday())  # Monday=0
    iso_week = int(n.strftime("%V"))
    return (n.minute == mi and n.hour == ho
            and iso_dow == dow and (iso_week % weeks) == 0)


def _loop() -> None:
    while not _stop.is_set():
        try:
            if _should_fire_now():
                run_once()
        except Exception:
            pass
        _stop.wait(timeout=60.0)


def start(*, daemon: bool = True) -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="dix-weekly-scout",
                                   daemon=daemon)
        _thread.start()


def stop(timeout: float = 3.0) -> None:
    _stop.set()
    t = _thread
    if t is not None:
        t.join(timeout=timeout)


# -------- util --------------------------------------------------------

def _iso_now() -> str:
    n = utc_now()
    return n.isoformat() if n else ""


def _epoch_now() -> float:
    n = utc_now()
    return n.timestamp() if n else time.time()


__all__ = [
    "Candidate", "CandidateCategory", "ScoutTick",
    "register_provider", "run_once", "last_tick",
    "start", "stop",
]
