"""FastAPI harness for the engine bus (Phase E1).

Run::

    uvicorn ui.server:app --reload --port 8080

Endpoints:

* ``GET  /``                       — single-page UI (HTML).
* ``GET  /api/health``             — ``check_self()`` of all six engines.
* ``GET  /api/registry/engines``   — ``registry/engines.yaml`` parsed.
* ``GET  /api/registry/plugins``   — ``registry/plugins.yaml`` parsed.
* ``POST /api/tick``               — feed a ``MarketTick`` into the
                                     ExecutionEngine's mark cache.
* ``POST /api/signal``             — flow a SignalEvent through
                                     Intelligence -> Execution; returns
                                     the resulting ExecutionEvent(s).
* ``GET  /api/events?limit=N``     — recent events emitted by either engine
                                     (in-memory ring buffer; not durable).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.contracts.events import (
    Event,
    Side,
    SignalEvent,
)
from core.contracts.market import MarketTick
from evolution_engine.engine import EvolutionEngine
from execution_engine.engine import ExecutionEngine
from governance_engine.engine import GovernanceEngine
from intelligence_engine.engine import IntelligenceEngine
from intelligence_engine.plugins import MicrostructureV1
from learning_engine.engine import LearningEngine
from system_engine.engine import SystemEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REGISTRY_DIR = REPO_ROOT / "registry"


# ---------------------------------------------------------------------------
# State (in-process; harness only)
# ---------------------------------------------------------------------------


class _State:
    """Single-process holder for engines + ring buffer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.intelligence = IntelligenceEngine(
            microstructure_plugins=(MicrostructureV1(),),
        )
        self.execution = ExecutionEngine()
        self.system = SystemEngine()
        self.governance = GovernanceEngine()
        self.learning = LearningEngine()
        self.evolution = EvolutionEngine()
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.event_seq: int = 0

    def all_engines(self) -> dict[str, Any]:
        return {
            "intelligence": self.intelligence,
            "execution": self.execution,
            "system": self.system,
            "governance": self.governance,
            "learning": self.learning,
            "evolution": self.evolution,
        }

    def record(self, source: str, event: Event) -> None:
        self.event_seq += 1
        self.events.appendleft(
            {
                "seq": self.event_seq,
                "source": source,
                **_event_to_dict(event),
            }
        )


STATE = _State()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_to_dict(event: Event) -> dict[str, Any]:
    if not is_dataclass(event):
        raise TypeError(f"non-dataclass event: {type(event)}")
    raw = asdict(event)
    # Enums are JSON-serialisable as their str value (StrEnum).
    return json.loads(json.dumps(raw, default=str))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class TickIn(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    last: float = Field(..., gt=0)
    volume: float = Field(0.0, ge=0)
    venue: str = ""
    ts_ns: int | None = None


class SignalIn(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    side: str = Field(..., pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    ts_ns: int | None = None
    qty: float | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="DIX VISION — Phase E1 Harness",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(500, "static/index.html missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    out = {}
    with STATE.lock:
        for name, eng in STATE.all_engines().items():
            status = eng.check_self()
            out[name] = {
                "name": eng.name,
                "tier": str(eng.tier),
                "state": str(status.state),
                "detail": status.detail,
                "plugin_states": {
                    slot: {p: str(s) for p, s in d.items()}
                    for slot, d in status.plugin_states.items()
                },
            }
    return {"engines": out}


@app.get("/api/registry/engines")
def registry_engines() -> dict[str, Any]:
    return _read_yaml(REGISTRY_DIR / "engines.yaml")


@app.get("/api/registry/plugins")
def registry_plugins() -> dict[str, Any]:
    return _read_yaml(REGISTRY_DIR / "plugins.yaml")


@app.post("/api/tick")
def post_tick(body: TickIn) -> dict[str, Any]:
    ts = body.ts_ns if body.ts_ns is not None else _next_ts()
    tick = MarketTick(
        ts_ns=ts,
        symbol=body.symbol,
        bid=body.bid,
        ask=body.ask,
        last=body.last,
        volume=body.volume,
        venue=body.venue,
    )
    signals_out: list[dict[str, Any]] = []
    executions_out: list[dict[str, Any]] = []
    with STATE.lock:
        STATE.execution.on_market(tick)
        STATE.events.appendleft(
            {
                "seq": STATE.event_seq + 1,
                "source": "tick",
                "kind": "MARKET_TICK",
                "ts_ns": tick.ts_ns,
                "symbol": tick.symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
            }
        )
        STATE.event_seq += 1
        # Phase E2: drive intelligence plugins on every tick. Shadow
        # signals are tagged by the engine; Execution rejects them.
        for sig in STATE.intelligence.on_market(tick):
            STATE.record("intelligence", sig)
            signals_out.append(_event_to_dict(sig))
            for downstream in STATE.execution.process(sig):
                STATE.record("execution", downstream)
                executions_out.append(_event_to_dict(downstream))
    return {
        "accepted": True,
        "tick": _event_to_dict_tick(tick),
        "signals": signals_out,
        "executions": executions_out,
    }


@app.post("/api/signal")
def post_signal(body: SignalIn) -> dict[str, Any]:
    ts = body.ts_ns if body.ts_ns is not None else _next_ts()
    meta: dict[str, str] = {}
    if body.qty is not None:
        meta["qty"] = str(body.qty)
    sig = SignalEvent(
        ts_ns=ts,
        symbol=body.symbol,
        side=Side(body.side),
        confidence=body.confidence,
        plugin_chain=("ui_harness",),
        meta=meta,
    )
    out_events: list[dict[str, Any]] = []
    with STATE.lock:
        STATE.record("ui_harness", sig)
        for ev in STATE.intelligence.process(sig):
            STATE.record("intelligence", ev)
            for downstream in STATE.execution.process(ev):
                STATE.record("execution", downstream)
                out_events.append(_event_to_dict(downstream))
    return {"signal": _event_to_dict(sig), "executions": out_events}


@app.get("/api/events")
def get_events(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    with STATE.lock:
        items = list(STATE.events)[:limit]
    return {"events": items}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _event_to_dict_tick(tick: MarketTick) -> dict[str, Any]:
    return {
        "kind": "MARKET_TICK",
        "ts_ns": tick.ts_ns,
        "symbol": tick.symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "volume": tick.volume,
        "venue": tick.venue,
    }


_TS_COUNTER = {"v": 0}
_TS_LOCK = threading.Lock()


def _next_ts() -> int:
    """Monotonic timestamp counter — must be atomic across threads.

    FastAPI runs sync endpoint handlers in a thread pool, so concurrent
    ``POST /api/tick`` / ``POST /api/signal`` requests would otherwise
    race on the read-modify-write of ``_TS_COUNTER`` and emit duplicate
    ``ts_ns`` values, violating the monotonic-timestamp contract (INV-15
    / TimeAuthority T0-04). The dedicated lock keeps this function
    independently thread-safe regardless of where it is called from.
    """
    with _TS_LOCK:
        _TS_COUNTER["v"] += 1
        return _TS_COUNTER["v"]


__all__: Sequence[str] = ("app", "STATE")
