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
* ``POST /api/feeds/binance/start`` — start the read-only Binance public
                                     WebSocket pump (SRC-MARKET-BINANCE-001).
* ``POST /api/feeds/binance/stop``  — stop the pump.
* ``GET  /api/feeds/binance/status``— pump telemetry snapshot.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.cognitive_router import (
    TaskClass,
    enabled_ai_providers,
    select_providers,
)
from core.contracts.api.cognitive_chat import (
    ChatStatusResponse,
    ChatTurnRequest,
    ChatTurnResponse,
)
from core.contracts.api.cognitive_chat_approvals import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalsListResponse,
)
from core.contracts.api.credentials import (
    CredentialItem,
    CredentialsStatusResponse,
    CredentialsSummary,
    PresenceStateApi,
)
from core.contracts.api.operator import (
    OperatorActionResponse,
    OperatorEngineRow,
    OperatorKillRequest,
    OperatorMemecoinSnapshot,
    OperatorModeSnapshot,
    OperatorStrategyCounts,
    OperatorSummaryResponse,
)
from core.contracts.events import (
    Event,
    Side,
    SignalEvent,
)
from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
)
from core.contracts.market import MarketTick
from dashboard.control_plane.decision_trace import DecisionTracePanel
from dashboard.control_plane.engine_status_grid import EngineStatusGrid
from dashboard.control_plane.memecoin_control_panel import MemecoinControlPanel
from dashboard.control_plane.mode_control_bar import ModeControlBar
from dashboard.control_plane.router import ControlPlaneRouter
from dashboard.control_plane.strategy_lifecycle_panel import (
    StrategyLifecyclePanel,
)
from evolution_engine.engine import EvolutionEngine
from execution_engine.engine import ExecutionEngine
from governance_engine.engine import GovernanceEngine
from governance_engine.harness_approver import (
    approve_signal_for_execution,
)
from intelligence_engine.cognitive.approval_edge import (
    ApprovalAlreadyDecidedError,
    ApprovalEdge,
    ApprovalNotFoundError,
)
from intelligence_engine.cognitive.chat.http_chat_transport import (
    build_default_dispatch_transport,
)
from intelligence_engine.engine import IntelligenceEngine
from intelligence_engine.plugins import MicrostructureV1
from intelligence_engine.strategy_runtime.state_machine import (
    StrategyStateMachine,
)
from intelligence_engine.trader_modeling import (
    make_trader_observation,
    observation_as_system_event,
)
from learning_engine.engine import LearningEngine
from state.ledger.reader import LedgerReader
from system_engine.credentials import (
    DEFAULT_TIMEOUT_S as CREDENTIAL_VERIFY_TIMEOUT_S,
)
from system_engine.credentials import (
    StorageNotWritable,
    is_devin_session,
    presence_status,
    requirements_for_registry,
    resolve_env,
    verify_provider,
    write_credential,
)
from system_engine.engine import SystemEngine
from system_engine.scvs.source_registry import (
    SourceRegistry,
    load_source_registry,
)
from ui.cognitive_chat_runtime import (
    ChatTurnDisabled,
    ChatTurnNoProvider,
    ChatTurnTransportFailed,
    CognitiveChatRuntime,
)
from ui.cognitive_chat_runtime import (
    build_runtime as build_cognitive_chat_runtime,
)
from ui.dashboard_routes import build_dashboard_router
from ui.feeds.runner import FeedRunner
from ui.feeds.tradingview_ideas import (
    TRADINGVIEW_SOURCE_FEED,
    parse_tradingview_idea_payload,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REGISTRY_DIR = REPO_ROOT / "registry"
SOURCE_REGISTRY_PATH = REGISTRY_DIR / "data_source_registry.yaml"


# ---------------------------------------------------------------------------
# State (in-process; harness only)
# ---------------------------------------------------------------------------


class _State:
    """Single-process holder for engines + ring buffer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # SCVS source registry — single source of truth for AI
        # providers, market feeds, and every other external source.
        # Loaded once at process start; the cognitive router reads this
        # frozen projection (no hot-reload yet — wave-02).
        self.source_registry: SourceRegistry = load_source_registry(
            SOURCE_REGISTRY_PATH
        )
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

        # Phase 6 dashboard widgets (DASH-1).
        self.strategy_fsm = StrategyStateMachine()
        self.ledger_reader = LedgerReader()
        self.dashboard_router = ControlPlaneRouter(
            bridge=self.governance.operator,
        )
        self.mode_widget = ModeControlBar(
            state_transitions=self.governance.state_transitions,
            router=self.dashboard_router,
        )
        self.engines_widget = EngineStatusGrid(engines=self.all_engines())
        self.strategies_widget = StrategyLifecyclePanel(fsm=self.strategy_fsm)
        self.decisions_widget = DecisionTracePanel(ledger=self.ledger_reader)
        self.memecoin_widget = MemecoinControlPanel(
            router=self.dashboard_router,
        )

        # Wave-03 PR-4 — cognitive chat runtime.
        # Wave-03 PR-6 — the production process now wires the
        # registry-driven HTTP dispatch transport (OpenAI / xAI /
        # DeepSeek via the OpenAI-compat shape, Google Gemini via
        # generateContent, and Cognition / Devin via the session
        # API). Each backend reads its API key from ``os.environ``
        # on every turn, so adding a key via ``/credentials`` after
        # the runtime starts takes effect without a restart. A row
        # whose key is missing fails with
        # :class:`TransientProviderError`, which the chat model
        # translates into a clean fall-through to the next eligible
        # provider — matching the previous ``NotConfiguredTransport``
        # contract for un-credentialed deployments. Tests inject
        # fake transports via :func:`set_chat_runtime`.
        self.chat_runtime: CognitiveChatRuntime = build_cognitive_chat_runtime(
            registry=self.source_registry,
            ledger_writer=self.governance.ledger,
            transport=build_default_dispatch_transport(),
        )

        # Wave-03 PR-5 — operator-approval edge. Binds the chat
        # runtime's queue to the live intelligence → execution chain
        # and the audit ledger. Held here (not on the chat runtime
        # itself) so the cognitive package stays B1-clean: it never
        # touches IntelligenceEngine / ExecutionEngine / the ledger
        # writer directly. Constructed eagerly so the routes can
        # delegate without lazy-init guards.
        self.approval_edge = ApprovalEdge(
            queue=self.chat_runtime.approval_queue,
            signal_emitter=self._emit_cognitive_signal_locked,
            ledger_append=self._approval_ledger_append,
            ts_ns=self.next_ts,
        )

        # Live data feeds (SCVS-registered sources). The Binance public
        # WS pump is opt-in via ``POST /api/feeds/binance/start``; the
        # runner here is constructed but not yet started so the harness
        # boots quickly with no external network dependency.
        # Lambda defers the ``_next_ts`` lookup until the first tick
        # arrives — ``_next_ts`` is defined later in this module, so a
        # bare reference would NameError at import time when ``STATE``
        # is instantiated.
        self.binance_feed = FeedRunner(
            sink=self._ingest_market_tick_locked,
            clock_ns=lambda: _next_ts(),
        )

    def all_engines(self) -> dict[str, Any]:
        return {
            "intelligence": self.intelligence,
            "execution": self.execution,
            "system": self.system,
            "governance": self.governance,
            "learning": self.learning,
            "evolution": self.evolution,
        }

    @property
    def mode(self) -> ModeControlBar:
        return self.mode_widget

    @property
    def engines(self) -> EngineStatusGrid:
        return self.engines_widget

    @property
    def strategies(self) -> StrategyLifecyclePanel:
        return self.strategies_widget

    @property
    def decisions(self) -> DecisionTracePanel:
        return self.decisions_widget

    @property
    def memecoin(self) -> MemecoinControlPanel:
        return self.memecoin_widget

    def next_ts(self) -> int:
        return _next_ts()

    def record(self, source: str, event: Event) -> None:
        self.event_seq += 1
        self.events.appendleft(
            {
                "seq": self.event_seq,
                "source": source,
                **_event_to_dict(event),
            }
        )
        # Feed the ledger reader so DASH-1 ``/api/dashboard/decisions``
        # has a live trace. The Phase E0 reader stub uses
        # ``_seed_for_tests`` as its only ingestion API; that is fine
        # for the in-process harness.
        self.ledger_reader._seed_for_tests((event,))

    def _ingest_market_tick_locked(self, tick: MarketTick) -> None:
        """Sink callable used by ``ui/feeds/runner.FeedRunner``.

        Acquires :attr:`lock` and runs the same Intelligence -> Execution
        fan-out as ``POST /api/tick`` so a tick from the Binance public
        WS pump is byte-identical (per ``_event_to_dict_tick``) to a
        manually-posted one. Called from the pump's asyncio thread, so
        the lock is mandatory.
        """

        with self.lock:
            self.execution.on_market(tick)
            self.event_seq += 1
            self.events.appendleft(
                {
                    "seq": self.event_seq,
                    "source": "feed.binance",
                    "kind": "MARKET_TICK",
                    "ts_ns": tick.ts_ns,
                    "symbol": tick.symbol,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": tick.last,
                    "venue": tick.venue,
                }
            )
            for sig in self.intelligence.on_market(tick):
                self.record("intelligence", sig)
                intent = approve_signal_for_execution(sig, ts_ns=_next_ts())
                for downstream in self.execution.execute(intent):
                    self.record("execution", downstream)

    def _emit_cognitive_signal_locked(self, sig: SignalEvent) -> None:
        """``ApprovalEdge`` signal-emitter binding.

        Wave-03 PR-5 — the approval edge is the *only* path that
        constructs a ``SignalEvent`` carrying
        ``produced_by_engine="intelligence_engine.cognitive"``
        (B26 lint pins this). On approve, the edge calls back here
        with a fully-stamped event; the harness threads it through
        the same intelligence → execution fan-out as ``/api/signal``
        so the resulting trade flows through HARDEN-02's execute
        chokepoint and HARDEN-03's receiver provenance assertion.
        Holds :attr:`lock` for the duration so the ledger ring and
        execution state stay consistent with concurrent ticks.
        """

        with self.lock:
            self.record("cognitive_chat", sig)
            for ev in self.intelligence.process(sig):
                self.record("intelligence", ev)
                intent = approve_signal_for_execution(ev, ts_ns=_next_ts())
                for downstream in self.execution.execute(intent):
                    self.record("execution", downstream)

    def _approval_ledger_append(
        self, kind: str, payload: Mapping[str, str]
    ) -> None:
        """``ApprovalEdge`` ledger-append binding.

        Mirrors :func:`build_ledger_append` from the chat runtime
        (which already wraps ``LedgerAuthorityWriter``); kept here
        so the approval edge does not have to import the chat
        runtime's helper. Stamps ``ts_ns`` from the harness time
        source for replay determinism (INV-15)."""

        self.governance.ledger.append(
            ts_ns=_next_ts(),
            kind=kind,
            payload=dict(payload),
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


class TradingViewObservationIn(BaseModel):
    """Envelope for ``POST /api/feeds/tradingview/observation`` (Wave-04 PR-2).

    Schema mirrors :data:`ui.feeds.tradingview_ideas` module docstring.
    The endpoint is the operator-controlled ingest point for trader
    observations sourced from TradingView (webhook relay, alert push,
    or manual paste). The parser-only adapter pattern lets *any*
    upstream collector feed the same pipeline.
    """

    payload: dict[str, Any] = Field(
        ..., description="Decoded TradingView envelope (parser input)."
    )
    ts_ns: int | None = Field(
        default=None,
        description=(
            "Optional caller-supplied monotonic timestamp. Defaults to "
            "the harness's monotonic counter (TimeAuthority surrogate)."
        ),
    )


class BinanceFeedStartIn(BaseModel):
    """Optional override for ``POST /api/feeds/binance/start``.

    Empty body uses the runner's configured default symbol set
    (``ui.feeds.binance_public_ws.DEFAULT_SYMBOLS``: BTCUSDT + ETHUSDT).
    """

    symbols: list[str] | None = Field(
        default=None,
        description="Override symbol list, e.g. ['btcusdt', 'ethusdt', 'solusdt']",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="DIX VISION — Phase E1 Harness",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# DASH-1 — read-only widget projections for the operator dashboard.
app.include_router(build_dashboard_router(lambda: STATE))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(500, "static/index.html missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Wave-Live PR-2 — legacy operator surface retired. The vanilla HTML
# pages that lived under ``ui/static/`` (operator.html, indira_chat.html,
# dyon_chat.html, forms_grid.html, credentials.html and their .js/.css
# siblings) were the Dashboard-2026 wave-01 prototype. They were
# wholesale superseded by the React/Vite SPA in ``dashboard2026/`` which
# is mounted at ``/dash2/``. We keep the URLs alive as 307 redirects
# so any cached link, dashboard tile, or external integrator that still
# points at ``/operator`` etc. silently lands on the live SPA instead
# of a 404. ``HTMLResponse`` is intentionally no longer the response
# class — these endpoints never serve a body, only a Location header.
_LEGACY_REDIRECTS: tuple[tuple[str, str], ...] = (
    ("/operator", "/dash2/#/operator"),
    ("/indira-chat", "/dash2/#/chat"),
    ("/dyon-chat", "/dash2/#/chat"),
    ("/forms-grid", "/dash2/#/operator"),
    ("/credentials", "/dash2/#/credentials"),
)


def _make_legacy_redirect(target: str) -> Any:
    def _handler() -> RedirectResponse:
        return RedirectResponse(url=target, status_code=307)

    return _handler


for _legacy_path, _dash2_target in _LEGACY_REDIRECTS:
    app.add_api_route(
        _legacy_path,
        _make_legacy_redirect(_dash2_target),
        methods=["GET"],
        response_class=RedirectResponse,
        include_in_schema=False,
    )


if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# Wave-02 React build artefact, served under /dash2/* if present.
# Mount is conditional so operators without Node installed (and CI
# jobs that don't build the SPA) still get a fully-functional vanilla
# console at the legacy URLs.
_DASH2_DIST = Path(__file__).resolve().parent.parent / "dashboard2026" / "dist"
if _DASH2_DIST.exists():
    app.mount(
        "/dash2",
        StaticFiles(directory=str(_DASH2_DIST), html=True),
        name="dash2",
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


# ---------------------------------------------------------------------------
# Cognitive Router — registry-driven AI provider list (Dashboard-2026 wave-01)
# ---------------------------------------------------------------------------


@app.get("/api/ai/providers")
def ai_providers(task: str | None = None) -> dict[str, Any]:
    """Registry-driven list of enabled AI providers.

    Both Indira Chat and Dyon Chat fetch this at boot to populate
    their provider dropdown. The list is sourced from
    ``registry/data_source_registry.yaml`` (rows with
    ``category: ai`` and ``enabled: true``) — provider names are
    NEVER hard-coded in widget code (``tools/authority_lint.py`` rule
    B23 enforces this). Adding a new AI provider is a registry-only
    change; both widgets pick it up automatically on next boot.

    The optional ``task`` query parameter filters by
    :class:`TaskClass` value (e.g. ``indira_reasoning``,
    ``dyon_coding``). When omitted, every enabled AI provider is
    returned in registry order.
    """

    registry = STATE.source_registry
    if task is None:
        providers = enabled_ai_providers(registry)
        task_value: str | None = None
    else:
        try:
            task_class = TaskClass(task)
        except ValueError as exc:
            raise HTTPException(
                400,
                f"unknown task class {task!r};"
                f" expected one of {[t.value for t in TaskClass]}",
            ) from exc
        providers = select_providers(registry, task_class)
        task_value = task_class.value

    return {
        "task": task_value,
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "provider": p.provider,
                "endpoint": p.endpoint,
                "capabilities": list(p.capabilities),
            }
            for p in providers
        ],
        "task_classes": [t.value for t in TaskClass],
    }


# ---------------------------------------------------------------------------
# Credential discovery — registry-driven "what API keys do I need" matrix
# (Dashboard-2026 wave-01.5)
# ---------------------------------------------------------------------------


@app.get("/api/credentials/status", response_model=CredentialsStatusResponse)
def credentials_status() -> CredentialsStatusResponse:
    """Return the credential matrix for every ``auth: required`` row.

    Each entry tells the operator (a) which env var name(s) must be
    set for that source, (b) whether each is currently present in the
    process environment, (c) where to sign up for a key, and (d)
    whether a free tier exists.

    Verification (does the key actually authenticate?) is *not* in
    this endpoint — it lands separately in ``POST /api/credentials/verify``
    so the read path stays cheap and side-effect-free.
    """

    requirements = requirements_for_registry(STATE.source_registry)
    env = resolve_env()
    statuses = presence_status(requirements, env)

    items: list[CredentialItem] = []
    counts = {"present": 0, "partial": 0, "missing": 0}
    for st in statuses:
        req = st.requirement
        state_api = PresenceStateApi(st.state.value)
        items.append(
            CredentialItem(
                source_id=req.source_id,
                source_name=req.source_name,
                category=req.category,
                provider=req.provider,
                env_vars=list(req.env_vars),
                env_vars_present=list(st.env_vars_present),
                missing_env_vars=list(st.missing_env_vars),
                signup_url=req.signup_url,
                free_tier=req.free_tier,
                notes=req.notes,
                state=state_api,
            ),
        )
        counts[st.state.value] += 1

    return CredentialsStatusResponse(
        summary=CredentialsSummary(
            total=len(items),
            present=counts["present"],
            partial=counts["partial"],
            missing=counts["missing"],
        ),
        writable=not is_devin_session(),
        items=items,
    )


class CredentialVerifyIn(BaseModel):
    """Operator-initiated verification request body.

    The operator picks one row from ``GET /api/credentials/status``
    and asks the server to live-ping it. The secret value never
    travels — only the ``source_id`` does. The server reads the env
    var locally (same way the trading engine would), pings the
    provider's auth-cheap endpoint, and returns a tri-state-plus
    outcome. No retries, no caching: each click is one ping.
    """

    source_id: str = Field(..., min_length=1, max_length=128)


@app.post("/api/credentials/verify")
def credentials_verify(body: CredentialVerifyIn) -> dict[str, Any]:
    """Live auth-ping for one ``auth: required`` source.

    Returns ``{source_id, provider, outcome, http_status, detail}``.
    The ``outcome`` is one of:
    ``ok | unauthorized | rate_limited | not_found | server_error |
    timeout | network_error | no_verifier | missing_key``. ``detail``
    is operator-readable and **never** echoes the secret value
    (covered by ``test_credentials_verify_does_not_leak_value``).

    The endpoint is intentionally synchronous; FastAPI runs sync
    routes in its threadpool so a slow provider does not stall the
    event loop. Each verifier has a hard 5 s timeout.
    """

    requirements = requirements_for_registry(STATE.source_registry)
    matching = next(
        (r for r in requirements if r.source_id == body.source_id),
        None,
    )
    if matching is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown or non-auth-required source_id "
                f"'{body.source_id}'"
            ),
        )

    result = verify_provider(
        matching.provider,
        resolve_env(),
        timeout=CREDENTIAL_VERIFY_TIMEOUT_S,
    )
    return {
        "source_id": matching.source_id,
        "provider": matching.provider,
        "outcome": result.outcome.value,
        "http_status": result.http_status,
        "detail": result.detail,
    }


class CredentialSetIn(BaseModel):
    """Persist one credential (PR-C, local launcher only).

    The dashboard sends ``{source_id, env_var, value}``. The server
    refuses the request when (a) the row is not in the registry,
    (b) ``env_var`` is not declared by that row's blueprint, or
    (c) we are running inside a Devin session — Devin secrets are
    operator-set via the ``secrets`` tool, not via the dashboard.

    The secret value is never logged and never echoed back. The
    response only carries ``{ok, env_var, source_id}``.
    """

    source_id: str = Field(..., min_length=1, max_length=128)
    env_var: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., min_length=1, max_length=4096)


@app.post("/api/credentials/set")
def credentials_set(body: CredentialSetIn) -> dict[str, Any]:
    """Write one credential to the local ``.env`` file.

    Returns ``{ok: True, source_id, env_var}`` on success.
    Refuses with HTTP 409 inside a Devin session, HTTP 404 for an
    unknown ``source_id``, and HTTP 422 when ``env_var`` does not
    belong to the row's blueprint.
    """

    requirements = requirements_for_registry(STATE.source_registry)
    matching = next(
        (r for r in requirements if r.source_id == body.source_id),
        None,
    )
    if matching is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown or non-auth-required source_id "
                f"'{body.source_id}'"
            ),
        )
    if body.env_var not in matching.env_vars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"env_var '{body.env_var}' is not declared by "
                f"source '{body.source_id}' (expected one of "
                f"{list(matching.env_vars)})"
            ),
        )

    try:
        write_credential(body.env_var, body.value)
    except StorageNotWritable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except (ValueError, TypeError) as exc:
        # ``write_credential`` validates name/value shape; surface
        # those as 422 without leaking the value.
        raise HTTPException(
            status_code=422,
            detail=f"invalid credential payload: {type(exc).__name__}",
        ) from None

    return {
        "ok": True,
        "source_id": matching.source_id,
        "env_var": body.env_var,
    }


# ---------------------------------------------------------------------------
# Operator dashboard typed surface (Wave-02 PR-2)
# ---------------------------------------------------------------------------
#
# The vanilla operator dashboard at ``/operator`` continues to consume
# the loosely-typed ``/api/dashboard/*`` endpoints (read projections of
# frozen dataclass snapshots). The React port at ``/dash2/#/operator``
# consumes the typed parallel surface below, so the wave-02 codegen
# pipeline (Pydantic → TS) gives it byte-stable types.
#
# Both surfaces share the *same* Phase 6 widget instances on ``STATE``
# — so a kill submitted via either endpoint enters the same
# ``ControlPlaneRouter`` and is decided by the same Governance bridge
# (GOV-CP-07). The route handler never bypasses Governance.


_STRATEGY_STATE_KEYS = (
    ("PROPOSED", "proposed"),
    ("SHADOW", "shadow"),
    ("CANARY", "canary"),
    ("LIVE", "live"),
    ("RETIRED", "retired"),
    ("FAILED", "failed"),
)


@app.get("/api/operator/summary", response_model=OperatorSummaryResponse)
def operator_summary() -> OperatorSummaryResponse:
    """Typed read projection of mode + engines + strategies + memecoin."""

    with STATE.lock:
        mode_snap = STATE.mode_widget.snapshot()
        engine_rows = STATE.engines_widget.snapshot()
        strategies_by_state = STATE.strategies_widget.by_state()
        memecoin_snap = STATE.memecoin_widget.status()
        chain_count = len(STATE.decisions_widget.chains(limit=200))

    counts: dict[str, int] = {field: 0 for _, field in _STRATEGY_STATE_KEYS}
    for state_key, field in _STRATEGY_STATE_KEYS:
        counts[field] = len(strategies_by_state.get(state_key, ()))

    return OperatorSummaryResponse(
        mode=OperatorModeSnapshot(
            current_mode=mode_snap.current_mode,
            legal_targets=list(mode_snap.legal_targets),
            is_locked=mode_snap.is_locked,
        ),
        engines=[
            OperatorEngineRow(
                engine_name=row.engine_name,
                bucket=row.bucket,
                detail=row.detail,
                plugin_count=len(row.plugin_states),
            )
            for row in engine_rows
        ],
        strategies=OperatorStrategyCounts(**counts),
        memecoin=OperatorMemecoinSnapshot(
            enabled=memecoin_snap.enabled,
            killed=memecoin_snap.killed,
            summary=memecoin_snap.summary,
        ),
        decision_chain_count=chain_count,
    )


@app.post(
    "/api/operator/action/kill",
    response_model=OperatorActionResponse,
)
def operator_action_kill(body: OperatorKillRequest) -> OperatorActionResponse:
    """Submit an operator KILL request through the governance bridge.

    The route handler constructs the typed ``OperatorRequest`` and
    submits it through ``ControlPlaneRouter`` (DASH-CP-01) →
    ``OperatorInterfaceBridge`` (GOV-CP-07). The decision returned by
    Governance is forwarded verbatim — both approvals and rejections
    are visible in the UI. The dashboard never writes the ledger and
    never bypasses Governance (B7 lint, INV-37).
    """

    request = OperatorRequest(
        ts_ns=_next_ts(),
        requestor=body.requestor,
        action=OperatorAction.REQUEST_KILL,
        payload={"reason": body.reason},
    )
    with STATE.lock:
        outcome = STATE.dashboard_router.submit(request)
    decision_dict = _decision_to_dict(outcome.decision)
    return OperatorActionResponse(
        approved=outcome.approved,
        summary=outcome.summary,
        decision=decision_dict,
    )


def _decision_to_dict(decision: Any) -> dict[str, Any]:
    """JSON-friendly conversion for governance decisions.

    Mirrors ``ui.dashboard_routes._to_dict`` semantics — frozen
    dataclasses are walked by ``asdict``, enums become their string
    value, and tuples become lists. Used only by the typed operator
    routes; the legacy dashboard surface keeps its own copy in
    ``ui.dashboard_routes`` to stay decoupled.
    """

    if is_dataclass(decision) and not isinstance(decision, type):
        return _decision_to_dict(asdict(decision))  # type: ignore[no-any-return]
    if isinstance(decision, dict):
        return {str(k): _decision_to_dict(v) for k, v in decision.items()}
    if isinstance(decision, (list, tuple)):
        return [_decision_to_dict(item) for item in decision]  # type: ignore[return-value]
    if isinstance(decision, (str, int, float, bool)) or decision is None:
        return decision  # type: ignore[return-value]
    return str(decision)  # type: ignore[return-value]


@app.get("/api/cognitive/chat/status", response_model=ChatStatusResponse)
def cognitive_chat_status() -> ChatStatusResponse:
    """Wave-03 PR-4 — feature-flag + provider availability snapshot.

    Polled by the operator chat page on mount so the UI can decide
    whether to render the input box or a "feature disabled" notice.
    Read-only; never writes the ledger.
    """

    with STATE.lock:
        return STATE.chat_runtime.status()


@app.post("/api/cognitive/chat/turn", response_model=ChatTurnResponse)
def cognitive_chat_turn(body: ChatTurnRequest) -> ChatTurnResponse:
    """Wave-03 PR-4 — drive one turn of the cognitive chat graph.

    Honors ``DIX_COGNITIVE_CHAT_ENABLED`` (off by default — 503 in
    that case). Dispatches through the registry-driven chat model
    from PR-1 so no vendor name appears on the wire. State is
    persisted to the audit ledger via PR-2's saver. Operator-
    approval edges that gate ``SignalEvent`` proposal emission are
    deferred to PR-5.
    """

    # Snapshot the runtime under the process-wide lock, then drop
    # it before calling ``turn`` — the LLM round-trip can take
    # seconds, and holding ``STATE.lock`` across it would block
    # every other endpoint (health, ticks, operator summary, …).
    # ``CognitiveChatRuntime`` has its own lock guarding the
    # bundle lazy-init path; the graph itself is invocation-safe
    # under concurrent calls because LangGraph keys state by
    # ``thread_id``.
    with STATE.lock:
        runtime = STATE.chat_runtime
    try:
        return runtime.turn(body)
    except ChatTurnDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ChatTurnNoProvider as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ChatTurnTransportFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        # Bad request shape (empty messages / wrong tail role /
        # SYSTEM message before PR-5 lands).
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/cognitive/chat/approvals",
    response_model=ApprovalsListResponse,
)
def cognitive_chat_approvals_list(
    include_decided: bool = False,
) -> ApprovalsListResponse:
    """Wave-03 PR-5 — snapshot the operator-approval queue.

    Default returns ``PENDING``-only rows so the dashboard panel
    only shows what still needs an operator click. Pass
    ``?include_decided=true`` to also surface the recent
    ``APPROVED`` / ``REJECTED`` history (used by the audit panel).
    Read-only; never writes the ledger.
    """

    with STATE.lock:
        rows = STATE.chat_runtime.approval_queue.list(
            include_decided=include_decided,
        )
    return ApprovalsListResponse(requests=list(rows))


@app.post(
    "/api/cognitive/chat/approvals/{request_id}/approve",
    response_model=ApprovalDecisionResponse,
)
def cognitive_chat_approval_approve(
    request_id: str,
    body: ApprovalDecisionRequest | None = None,
) -> ApprovalDecisionResponse:
    """Wave-03 PR-5 — operator approves a queued cognitive proposal.

    The approval edge stamps ``produced_by_engine=
    "intelligence_engine.cognitive"`` on the resulting
    ``SignalEvent`` (B26 / HARDEN-03), routes it through the
    intelligence → execution chain (HARDEN-02 chokepoint), and
    writes an ``OPERATOR_APPROVED_SIGNAL`` ledger row. Returns
    the decided request and the new event's audit-ledger id.
    """

    decision = body if body is not None else ApprovalDecisionRequest()
    try:
        decided, sig = STATE.approval_edge.approve(
            request_id=request_id,
            decision=decision,
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalAlreadyDecidedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalDecisionResponse(
        request=decided,
        emitted_signal_id=f"{sig.symbol}:{sig.side.value}:{sig.ts_ns}",
    )


@app.post(
    "/api/cognitive/chat/approvals/{request_id}/reject",
    response_model=ApprovalDecisionResponse,
)
def cognitive_chat_approval_reject(
    request_id: str,
    body: ApprovalDecisionRequest | None = None,
) -> ApprovalDecisionResponse:
    """Wave-03 PR-5 — operator rejects a queued cognitive proposal.

    No event hits the bus; an ``OPERATOR_REJECTED_SIGNAL`` row is
    written to the ledger so the audit chain captures every
    decision (not just the approvals). Returns the decided
    request with ``emitted_signal_id`` left empty.
    """

    decision = body if body is not None else ApprovalDecisionRequest()
    try:
        decided = STATE.approval_edge.reject(
            request_id=request_id,
            decision=decision,
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalAlreadyDecidedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ApprovalDecisionResponse(request=decided, emitted_signal_id="")


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
            intent = approve_signal_for_execution(sig, ts_ns=_next_ts())
            for downstream in STATE.execution.execute(intent):
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
            intent = approve_signal_for_execution(ev, ts_ns=_next_ts())
            for downstream in STATE.execution.execute(intent):
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
# Live data feeds (SCVS-registered sources)
# ---------------------------------------------------------------------------


def _feed_status_dict(source_id: str) -> dict[str, Any]:
    status = STATE.binance_feed.status()
    return {
        "source_id": source_id,
        "running": status.running,
        "url": status.url,
        "symbols": list(status.symbols),
        "ticks_received": status.ticks_received,
        "errors": status.errors,
        "last_tick_ts_ns": status.last_tick_ts_ns,
    }


@app.post("/api/feeds/binance/start")
def post_binance_feed_start(
    body: BinanceFeedStartIn | None = None,
) -> dict[str, Any]:
    """Start the read-only Binance public WS pump (SRC-MARKET-BINANCE-001).

    Idempotent — returns the current status if already running. Pass
    ``{"symbols": ["btcusdt", "ethusdt", "solusdt"]}`` to override the
    default symbol set for this run.
    """
    symbols = body.symbols if body is not None else None
    try:
        STATE.binance_feed.start(symbols=symbols)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "started": True,
        "feed": _feed_status_dict("SRC-MARKET-BINANCE-001"),
    }


@app.post("/api/feeds/binance/stop")
def post_binance_feed_stop() -> dict[str, Any]:
    """Stop the Binance public WS pump.

    Idempotent — returns the current status if not running.
    """
    STATE.binance_feed.stop()
    return {
        "stopped": True,
        "feed": _feed_status_dict("SRC-MARKET-BINANCE-001"),
    }


@app.get("/api/feeds/binance/status")
def get_binance_feed_status() -> dict[str, Any]:
    """Return a telemetry snapshot of the Binance public WS pump."""
    return {"feed": _feed_status_dict("SRC-MARKET-BINANCE-001")}


@app.post("/api/feeds/tradingview/observation")
def post_tradingview_observation(body: TradingViewObservationIn) -> dict[str, Any]:
    """Ingest one TradingView trader observation (SRC-TRADER-TRADINGVIEW-001).

    Wave-04 PR-2 — operator-controlled trader-feed ingest. The body's
    ``payload`` is fed to :func:`ui.feeds.tradingview_ideas.parse_tradingview_idea_payload`,
    which never constructs a :class:`TraderObservation` directly (B29
    forbids it). Construction happens inside
    :func:`intelligence_engine.trader_modeling.aggregator.make_trader_observation`,
    the only B29-allowed runtime location, and the resulting record is
    projected into a :class:`SystemEvent` for the audit ledger via
    :func:`intelligence_engine.trader_modeling.observation.observation_as_system_event`.

    Returns ``{"accepted": False, "reason": "..."}`` (HTTP 200) for
    malformed payloads so a webhook relay can keep streaming without
    fragile error handling. Returns ``{"accepted": True, "event": ...}``
    on success.
    """
    ts = body.ts_ns if body.ts_ns is not None else _next_ts()
    parsed = parse_tradingview_idea_payload(
        body.payload,
        ts_ns=ts,
        source_feed=TRADINGVIEW_SOURCE_FEED,
    )
    if parsed is None:
        return {
            "accepted": False,
            "reason": "payload rejected by tradingview parser",
            "source_feed": TRADINGVIEW_SOURCE_FEED,
        }
    model, observation_kind, meta = parsed
    observation = make_trader_observation(
        ts_ns=ts,
        model=model,
        observation_kind=observation_kind,
        meta=meta,
    )
    event = observation_as_system_event(observation)
    with STATE.lock:
        STATE.record("feed.tradingview", event)
    return {
        "accepted": True,
        "source_feed": TRADINGVIEW_SOURCE_FEED,
        "trader_id": observation.trader_id,
        "observation_kind": observation.observation_kind,
        "ts_ns": observation.ts_ns,
    }


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
