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
import os
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
    HazardEvent,
    Side,
    SignalEvent,
)
from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
)
from core.contracts.market import MarketTick
from core.contracts.risk import RiskSnapshot
from dashboard_backend.control_plane.decision_trace import DecisionTracePanel
from dashboard_backend.control_plane.engine_status_grid import EngineStatusGrid
from dashboard_backend.control_plane.memecoin_control_panel import MemecoinControlPanel
from dashboard_backend.control_plane.mode_control_bar import ModeControlBar
from dashboard_backend.control_plane.router import ControlPlaneRouter
from dashboard_backend.control_plane.strategy_lifecycle_panel import (
    StrategyLifecyclePanel,
)
from evolution_engine.engine import EvolutionEngine
from execution_engine.engine import ExecutionEngine
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_hash_anchor import (
    PolicyHashAnchor,
)
from governance_engine.engine import GovernanceEngine
from governance_engine.harness_approver import (
    HARNESS_APPROVER_ENV_VAR,
    approve_signal_for_execution,
)

# Hardening-S1 item 1 — explicit opt-in for the harness approval shim.
# ``ui.server`` is the harness, by definition. Setting the env var at
# import time means *every* call to ``approve_signal_for_execution``
# from within this process passes the gate without each call site
# needing ``enabled=True``. Engines, adapters, and dashboard surfaces
# do NOT set this env var; if any of them imports the shim the
# authority lint B33 rule fires at CI time and at runtime the call
# raises :class:`HarnessApproverDisabledError`.
os.environ.setdefault(HARNESS_APPROVER_ENV_VAR, "1")
from intelligence_engine.cognitive.approval_edge import (
    ApprovalAlreadyDecidedError,
    ApprovalEdge,
    ApprovalNotFoundError,
)
from intelligence_engine.cognitive.chat import (
    FEATURE_FLAG_ENV_VAR as COGNITIVE_CHAT_FEATURE_FLAG_ENV_VAR,
)
from intelligence_engine.cognitive.chat import (
    CognitiveChatFeatureFlag,
)
from intelligence_engine.cognitive.chat.http_chat_transport import (
    build_default_dispatch_transport,
)
from intelligence_engine.engine import IntelligenceEngine
from intelligence_engine.knowledge import NewsKnowledgeIndex
from intelligence_engine.mcp import OpenNewsServer
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
from system.time_source import wall_ns
from system_engine.coupling import HazardThrottleAdapter
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
from ui._ledger_boot import resolve_ledger_path
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
from ui.execution_routes import build_execution_router
from ui.feeds.news_runner import CoinDeskRSSFeedRunner
from ui.feeds.pumpfun_runner import PumpFunFeedRunner
from ui.feeds.raydium_runner import RaydiumPoolFeedRunner
from ui.feeds.runner import FeedRunner
from ui.feeds.tradingview_ideas import (
    TRADINGVIEW_SOURCE_FEED,
    parse_tradingview_idea_payload,
)
from ui.governance_routes import build_governance_router
from ui.plugin_routes import (
    PluginRegistry,
    PluginToggleState,
    build_plugin_router,
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
        # AUDIT-WIRE.1 / P0-2 — close the BEHAVIOR-P3 hazard throttle
        # chain. Until this wiring landed the harness constructed a
        # bare ``ExecutionEngine()`` with ``throttle_adapter=None``,
        # so observed :class:`HazardEvent` rows never tightened the
        # hot-path :class:`RiskSnapshot` and the ``apply_throttle``
        # primitive built in PR #139 was inert in production. Hazards
        # delivered to ``self.execution.on_hazard`` now feed the
        # adapter's observer ring, and every subsequent
        # :meth:`ExecutionEngine.execute` projects the active throttle
        # decision onto the configured baseline.
        self.hazard_throttle = HazardThrottleAdapter()
        # The baseline RiskSnapshot is the un-throttled FastRiskCache
        # view; the harness has no live cache yet (Phase 7 wave) so
        # the seed below is a deterministic, permissive baseline that
        # the throttle adapter narrows in place. ``halted`` flips to
        # True the moment a CRITICAL hazard is observed inside the
        # active window, which short-circuits dispatch to a single
        # REJECTED ExecutionEvent with reason ``hazard_throttled``.
        self.execution = ExecutionEngine(
            throttle_adapter=self.hazard_throttle,
            risk_baseline=RiskSnapshot(version=0, ts_ns=wall_ns()),
        )
        self.system = SystemEngine()
        # Sprint-1 / Class-B "Trust the Ledger" — if the operator sets
        # ``DIXVISION_LEDGER_PATH`` the harness opens a SQLite-backed
        # authority ledger; every governance decision (mode
        # transition, strategy lifecycle, operator approval) is then
        # persisted before the in-memory chain is mutated, so a
        # crash / Ctrl+C / kill -9 between rows is survivable. The
        # writer replays existing rows on construction and runs a
        # boot-time hash-chain verification gate so a tampered file
        # aborts startup loudly.
        #
        # AUDIT-P0.3 — the silent fallback to an in-memory writer when
        # the env var was unset meant default operator deployments
        # could run for hours believing every decision was persisted
        # while losing the entire chain on restart. ``resolve_ledger_path``
        # now refuses to boot without persistence unless the operator
        # has explicitly set ``DIXVISION_PERMIT_EPHEMERAL_LEDGER=1``
        # (the test suite sets this at session start in
        # ``tests/conftest.py``).
        ledger_path = resolve_ledger_path()
        ledger = (
            LedgerAuthorityWriter(db_path=ledger_path)
            if ledger_path
            else LedgerAuthorityWriter()
        )
        self.ledger_writer = ledger
        self.governance = GovernanceEngine(ledger=ledger)
        # Hardening-S1 item 4-ext -- bind the SHA-256 of every
        # canonical policy YAML to the authority ledger at boot. The
        # anchor turns the policy set into "the document of record":
        # any mid-session edit is detected by ``verify_no_drift`` and
        # surfaced as a CRITICAL ``HAZ-POLICY-DRIFT`` hazard, which
        # ``GovernanceEngine.process`` routes through the single FSM
        # mutator so the system downgrades to SAFE through the same
        # audited chain every other hazard takes (B32 / GOV-CP-03).
        self.policy_hash_anchor = PolicyHashAnchor(ledger=ledger)
        self.policy_hash_anchor.bind_session(
            ts_ns=wall_ns(), requestor="ui_harness_boot"
        )
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
        # Plugin-manager toggle state — holds in-process overrides
        # for env-gated plugins (cognitive chat today; future:
        # learning-engine, evolution-engine). Constructed before
        # the chat runtime so we can hand its custom getter into
        # ``CognitiveChatFeatureFlag``: the override wins when set,
        # otherwise the env var is consulted (which itself defaults
        # ON after the cognitive_chat_graph default-on flip).
        self.plugin_toggle_state = PluginToggleState()

        def _cognitive_chat_flag_getter(name: str, default: str) -> str:
            if name != COGNITIVE_CHAT_FEATURE_FLAG_ENV_VAR:
                return os.getenv(name, default)
            override = self.plugin_toggle_state.cognitive_chat
            if override is True:
                return "1"
            if override is False:
                return "0"
            return os.getenv(name, default)

        self.chat_runtime: CognitiveChatRuntime = build_cognitive_chat_runtime(
            registry=self.source_registry,
            ledger_writer=self.governance.ledger,
            transport=build_default_dispatch_transport(),
            feature_flag=CognitiveChatFeatureFlag(
                getter=_cognitive_chat_flag_getter
            ),
        )

        # Plugin manager registry — references the live plugin
        # objects so a lifecycle mutation through the dashboard is
        # observed immediately by the engines on the next tick.
        self.plugin_registry = PluginRegistry(
            microstructure_plugins=tuple(
                self.intelligence.microstructure_plugins
            ),
            toggle_state=self.plugin_toggle_state,
            cognitive_chat_env_enabled=lambda: (
                CognitiveChatFeatureFlag().enabled
            ),
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
        # Sprint-1 / Class-B — ``wall_ns`` is the canonical TimeAuthority
        # hot-path API (``system/time_source.py``); replaces the
        # legacy ``_TS_COUNTER`` integer counter so ledger rows carry
        # real wall-clock nanoseconds across process restarts
        # (architectural-review P0-3 / INV-15).
        self.binance_feed = FeedRunner(
            sink=self._ingest_market_tick_locked,
            clock_ns=wall_ns,
        )

        # P0-5 — close the news loop. Wave-news-fusion shipped the
        # NewsItem -> SignalEvent projection (PR #118), the
        # NewsShockSensor (PR #119) and the NewsFanout composer
        # (PR #120), but no caller ran them in the live process. The
        # runner here wraps a CoinDeskRSSPump with a NewsFanout whose
        # signal/hazard sinks hand each emitted event back into the
        # in-process intelligence -> execution and governance fan-outs
        # used by the Binance pump and POST /api/signal. Idle by
        # default; opt-in via POST /api/feeds/coindesk/start so the
        # harness boots without an external network dependency.
        # D4 — deterministic in-memory news similarity index. Every
        # NewsItem flowing through ``NewsFanout`` is appended here so
        # downstream learners (slow-loop) and the OpenNews MCP server
        # share one source of truth. Bounded; no clocks; no PRNG.
        self.news_index = NewsKnowledgeIndex()
        self.opennews_server = OpenNewsServer(self.news_index)
        self.coindesk_feed = CoinDeskRSSFeedRunner(
            signal_sink=self._ingest_news_signal_locked,
            hazard_sink=self._ingest_news_hazard_locked,
            index_sink=self.news_index.add,
            clock_ns=wall_ns,
        )

        # D2 — Pump.fun launches + Raydium pool snapshots. Both runners
        # are constructed idle; opt-in via
        # ``POST /api/feeds/{pumpfun,raydium}/start`` so the harness
        # boots without an external network dependency. The sinks
        # publish into bounded ring buffers exposed by
        # ``GET /api/feeds/{pumpfun,raydium}/recent``; downstream
        # memecoin-tier consumers (LaunchFirehose, BundleDetector,
        # PoolSnapshotPanel, etc.) read from the same buffers.
        self.recent_launches: deque[dict[str, Any]] = deque(maxlen=200)
        self.recent_pool_snapshots: deque[dict[str, Any]] = deque(
            maxlen=500
        )
        self.pumpfun_feed = PumpFunFeedRunner(
            sink=self._ingest_pumpfun_launch_locked,
            clock_ns=wall_ns,
        )
        self.raydium_feed = RaydiumPoolFeedRunner(
            sink=self._ingest_raydium_snapshot_locked,
            clock_ns=wall_ns,
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
        return wall_ns()

    def current_ts(self) -> int:
        """Read the monotonic counter WITHOUT incrementing it.

        Read-only diagnostic surfaces (e.g. governance JSON GETs) need a
        ``now_ns`` value to compute gaps without consuming a sequence
        number. ``next_ts()`` is reserved for write paths that emit a
        sequenced event into the ledger.
        """
        return wall_ns()

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
                intent = approve_signal_for_execution(sig, ts_ns=wall_ns())
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
                intent = approve_signal_for_execution(ev, ts_ns=wall_ns())
                for downstream in self.execution.execute(intent):
                    self.record("execution", downstream)

    def _ingest_news_signal_locked(self, sig: SignalEvent) -> None:
        """``NewsFanout`` signal-sink binding.

        P0-5 — runs the same intelligence -> execution fan-out the
        Binance pump uses (``_ingest_market_tick_locked``) and the
        cognitive approval edge uses (``_emit_cognitive_signal_locked``)
        so a news-projected ``SignalEvent`` flows through HARDEN-02's
        execute chokepoint without bypassing AuthorityGuard. Holds
        :attr:`lock` for the duration so the ledger ring and execution
        state stay consistent with concurrent ticks. Called from the
        runner's asyncio thread, so the lock is mandatory.
        """
        with self.lock:
            self.record("news.coindesk", sig)
            for ev in self.intelligence.process(sig):
                self.record("intelligence", ev)
                intent = approve_signal_for_execution(ev, ts_ns=wall_ns())
                for downstream in self.execution.execute(intent):
                    self.record("execution", downstream)

    def _ingest_news_hazard_locked(self, hazard: HazardEvent) -> None:
        """``NewsFanout`` hazard-sink binding.

        P0-5 — feeds the news-shock hazard event into the live
        ``GovernanceEngine`` so the throttle / mode-FSM machinery
        registered by the wave-news-fusion shock sensor (HAZ-NEWS-SHOCK)
        actually fires in production. Mirrors the
        ``_ingest_news_signal_locked`` lock discipline; called from the
        runner's asyncio thread.
        """
        with self.lock:
            self.record("hazard.news", hazard)
            # AUDIT-WIRE.1 — every hazard the harness sees must reach
            # the execution-engine throttle adapter, not just the
            # governance FSM. Without this branch the ``apply_throttle``
            # chain stayed dark for hazards that never crossed the
            # mode-FSM CRITICAL/HIGH gate (e.g. WARN-tier news shocks).
            self.execution.on_hazard(hazard)
            for downstream in self.governance.process(hazard):
                self.record("governance", downstream)

    def _ingest_pumpfun_launch_locked(self, ev: Any) -> None:
        """``PumpFunFeedRunner`` sink — D2.

        Pushes one row per :class:`LaunchEvent` into the
        ``recent_launches`` ring (bounded by ``maxlen=200``). Called
        from the runner's asyncio thread, so we acquire the harness
        lock to keep writes serialized with the event dict reads on
        the FastAPI sync handlers.
        """

        with self.lock:
            self.recent_launches.appendleft(
                {
                    "ts_ns": int(ev.ts_ns),
                    "venue": str(ev.venue),
                    "chain": str(ev.chain),
                    "mint": str(ev.mint),
                    "symbol": str(ev.symbol),
                    "name": str(ev.name),
                    "creator": str(ev.creator),
                    "market_cap_usd": float(ev.market_cap_usd),
                    "liquidity_usd": float(ev.liquidity_usd),
                }
            )

    def _ingest_raydium_snapshot_locked(self, snap: Any) -> None:
        """``RaydiumPoolFeedRunner`` sink — D2.

        Pushes one row per :class:`PoolSnapshot` into the
        ``recent_pool_snapshots`` ring (bounded by ``maxlen=500``).
        Called from the runner's asyncio thread.
        """

        with self.lock:
            self.recent_pool_snapshots.appendleft(
                {
                    "ts_ns": int(snap.ts_ns),
                    "venue": str(snap.venue),
                    "chain": str(snap.chain),
                    "pool_id": str(snap.pool_id),
                    "base_mint": str(snap.base_mint),
                    "quote_mint": str(snap.quote_mint),
                    "base_symbol": str(snap.base_symbol),
                    "quote_symbol": str(snap.quote_symbol),
                    "price": float(snap.price),
                    "liquidity_usd": float(snap.liquidity_usd),
                    "volume_24h_usd": float(snap.volume_24h_usd),
                }
            )

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
            ts_ns=wall_ns(),
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

# Plugin manager — operator-toggleable plugin lifecycles. Powers the
# /dash2 "Plugins" page and writes a PLUGIN_LIFECYCLE row to the
# authority ledger on every successful toggle.
app.include_router(
    build_plugin_router(
        registry_provider=lambda: STATE.plugin_registry,
        ledger_provider=lambda: STATE.governance.ledger,
        ts_provider=lambda: STATE.next_ts(),
    )
)

# Tier-1 governance widgets — promotion gates, drift oracle, SCVS source
# liveness, hazard monitor. Read-only JSON projections consumed by the
# /dash2 governance page.
app.include_router(build_governance_router(lambda: STATE))

# D1 / EXEC-ADAPTERS — operator dashboard surface for live execution
# adapters (Hummingbot, Pump.fun, UniswapX, …). Read-only JSON.
app.include_router(build_execution_router())


# Wave-Live PR-4 — root URL routes operators to the live SPA. PR #105
# redirected the named legacy paths (``/operator``, ``/indira-chat`` etc.)
# but missed ``/`` itself, so the Windows launcher (which opens
# ``http://127.0.0.1:8080/``) was still landing on the Phase E1 stub.
# We redirect to ``/dash2/`` when the React build artefact is present;
# otherwise we fall back to the stub so operators are not left staring at
# a 404 if the SPA was not built. The ``dashboard2026/dist`` location is
# resolved here once at handler-registration time — the actual
# ``StaticFiles`` mount happens further below.
_DASH2_DIST = Path(__file__).resolve().parent.parent / "dashboard2026" / "dist"
_DASH2_INDEX = _DASH2_DIST / "index.html"
# Freeze availability at module-load time so the ``GET /`` handler's redirect
# decision and the conditional ``StaticFiles`` mount below stay in lock-step.
# A per-request ``_DASH2_INDEX.exists()`` check would silently diverge if a
# developer ran ``npm run build`` after ``uvicorn --reload`` had already
# imported the module: ``--reload`` only watches ``.py`` files, so the SPA
# build wouldn't restart the server, the handler would 307 to ``/dash2/``,
# and the mount that was never registered would 404. Devin Review BUG_0001
# on PR #123 caught this.
_DASH2_AVAILABLE: bool = _DASH2_DIST.exists() and _DASH2_INDEX.exists()

# DIX MEME — DEXtools-styled memecoin dashboard. Same conditional-mount
# pattern as ``/dash2/``: separate React app, separate launcher, but a
# *viewer* on the same harness — every execution intent it submits goes
# through the same ``/api/dashboard/action/intent`` chokepoint and the
# same Governance FSM as ``/dash2/``. Closing the browser does not stop
# the harness; the learning loop, sensors, and audit ledger keep
# running independent of which (or no) dashboard is open.
_MEME_DIST = Path(__file__).resolve().parent.parent / "dash_meme" / "dist"
_MEME_INDEX = _MEME_DIST / "index.html"
_MEME_AVAILABLE: bool = _MEME_DIST.exists() and _MEME_INDEX.exists()


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    if _DASH2_AVAILABLE:
        return RedirectResponse(url="/dash2/", status_code=307)
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
# console at the legacy URLs. ``_DASH2_AVAILABLE`` is the same module-load
# boolean the ``/`` handler reads — keeping them on one snapshot prevents
# the redirect-without-mount race documented above.
if _DASH2_AVAILABLE:
    app.mount(
        "/dash2",
        StaticFiles(directory=str(_DASH2_DIST), html=True),
        name="dash2",
    )


# DIX MEME — same StaticFiles pattern, mounted at ``/meme/``. Operators
# launch it via ``start_dixvision_meme.bat`` (separate from the cockpit
# launcher), but it talks to the *same* API surface as ``/dash2/``. The
# mount is conditional on a built artefact under ``dash_meme/dist`` so
# the harness still boots when the app hasn't been built (e.g. CI jobs
# that skip the npm build, or a fresh clone where Node isn't installed).
if _MEME_AVAILABLE:
    app.mount(
        "/meme",
        StaticFiles(directory=str(_MEME_DIST), html=True),
        name="meme",
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
        ts_ns=wall_ns(),
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
    ts = body.ts_ns if body.ts_ns is not None else wall_ns()
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
            intent = approve_signal_for_execution(sig, ts_ns=wall_ns())
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
    ts = body.ts_ns if body.ts_ns is not None else wall_ns()
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
            intent = approve_signal_for_execution(ev, ts_ns=wall_ns())
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


def _coindesk_feed_status_dict() -> dict[str, Any]:
    status = STATE.coindesk_feed.status()
    return {
        "source_id": status.source,
        "running": status.running,
        "url": status.url,
        "items_received": status.items_received,
        "polls": status.polls,
        "errors": status.errors,
        "last_poll_ts_ns": status.last_poll_ts_ns,
        "last_item_ts_ns": status.last_item_ts_ns,
    }


@app.post("/api/feeds/coindesk/start")
def post_coindesk_feed_start() -> dict[str, Any]:
    """Start the read-only CoinDesk RSS pump (SRC-NEWS-COINDESK-001).

    P0-5 — closes the news loop. Each polled :class:`NewsItem` flows
    through :class:`NewsFanout`, fanning out to a projected
    :class:`SignalEvent` (intelligence -> execution chain) and any
    emitted :class:`HazardEvent` (governance throttle / mode FSM).
    Idempotent.
    """
    STATE.coindesk_feed.start()
    return {
        "started": True,
        "feed": _coindesk_feed_status_dict(),
    }


@app.post("/api/feeds/coindesk/stop")
def post_coindesk_feed_stop() -> dict[str, Any]:
    """Stop the CoinDesk RSS pump. Idempotent."""
    STATE.coindesk_feed.stop()
    return {
        "stopped": True,
        "feed": _coindesk_feed_status_dict(),
    }


@app.get("/api/feeds/coindesk/status")
def get_coindesk_feed_status() -> dict[str, Any]:
    """Return a telemetry snapshot of the CoinDesk RSS pump."""
    return {"feed": _coindesk_feed_status_dict()}


def _pumpfun_feed_status_dict() -> dict[str, Any]:
    status = STATE.pumpfun_feed.status()
    return {
        "source_id": "SRC-LAUNCH-PUMPFUN-001",
        "running": status.running,
        "url": status.url,
        "launches_received": status.launches_received,
        "errors": status.errors,
        "last_launch_ts_ns": status.last_launch_ts_ns,
    }


@app.post("/api/feeds/pumpfun/start")
def post_pumpfun_feed_start() -> dict[str, Any]:
    """Start the read-only Pump.fun PumpPortal WS pump (D2).

    Streams new-token mint events from
    ``wss://pumpportal.fun/api/data`` into the ``recent_launches``
    ring exposed by ``GET /api/feeds/pumpfun/recent``.
    """
    STATE.pumpfun_feed.start()
    return {
        "started": True,
        "feed": _pumpfun_feed_status_dict(),
    }


@app.post("/api/feeds/pumpfun/stop")
def post_pumpfun_feed_stop() -> dict[str, Any]:
    """Stop the Pump.fun WS pump. Idempotent."""
    STATE.pumpfun_feed.stop()
    return {
        "stopped": True,
        "feed": _pumpfun_feed_status_dict(),
    }


@app.get("/api/feeds/pumpfun/status")
def get_pumpfun_feed_status() -> dict[str, Any]:
    """Return a telemetry snapshot of the Pump.fun WS pump."""
    return {"feed": _pumpfun_feed_status_dict()}


@app.get("/api/feeds/pumpfun/recent")
def get_pumpfun_recent(limit: int = 50) -> dict[str, Any]:
    """Return the most recent Pump.fun launches (newest first)."""
    cap = max(1, min(int(limit), 200))
    with STATE.lock:
        launches = list(STATE.recent_launches)[:cap]
    return {
        "launches": launches,
        "count": len(launches),
        "feed": _pumpfun_feed_status_dict(),
    }


def _raydium_feed_status_dict() -> dict[str, Any]:
    status = STATE.raydium_feed.status()
    return {
        "source_id": "SRC-POOL-RAYDIUM-001",
        "running": status.running,
        "url": status.url,
        "snapshots_emitted": status.snapshots_emitted,
        "errors": status.errors,
        "last_poll_ts_ns": status.last_poll_ts_ns,
    }


@app.post("/api/feeds/raydium/start")
def post_raydium_feed_start() -> dict[str, Any]:
    """Start the read-only Raydium AMM pool poller (D2).

    Polls ``https://api.raydium.io/v2/main/pairs`` on a fixed
    interval and emits one :class:`PoolSnapshot` per pair into the
    ``recent_pool_snapshots`` ring exposed by
    ``GET /api/feeds/raydium/recent``.
    """
    STATE.raydium_feed.start()
    return {
        "started": True,
        "feed": _raydium_feed_status_dict(),
    }


@app.post("/api/feeds/raydium/stop")
def post_raydium_feed_stop() -> dict[str, Any]:
    """Stop the Raydium pool poller. Idempotent."""
    STATE.raydium_feed.stop()
    return {
        "stopped": True,
        "feed": _raydium_feed_status_dict(),
    }


@app.get("/api/feeds/raydium/status")
def get_raydium_feed_status() -> dict[str, Any]:
    """Return a telemetry snapshot of the Raydium pool poller."""
    return {"feed": _raydium_feed_status_dict()}


@app.get("/api/feeds/raydium/recent")
def get_raydium_recent(limit: int = 100) -> dict[str, Any]:
    """Return the most recent Raydium pool snapshots (newest first)."""
    cap = max(1, min(int(limit), 500))
    with STATE.lock:
        snaps = list(STATE.recent_pool_snapshots)[:cap]
    return {
        "snapshots": snaps,
        "count": len(snaps),
        "feed": _raydium_feed_status_dict(),
    }


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
    ts = body.ts_ns if body.ts_ns is not None else wall_ns()
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


__all__: Sequence[str] = ("app", "STATE")
