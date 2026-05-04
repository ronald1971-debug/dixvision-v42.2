"""HTTP plugin manager — operator-facing toggle surface.

The dashboard "Plugins" page consumes this router to enumerate every
hot-toggleable plugin in the runtime and to flip its lifecycle without
restarting the process. Two distinct shapes are exposed under one
uniform contract:

* **Microstructure plugins** (and any other concrete
  :class:`MicrostructurePlugin` slot) carry a binary lifecycle —
  ``DISABLED`` / ``ACTIVE``. The route mutates the dataclass
  attribute in place and writes a ``PLUGIN_LIFECYCLE`` row to the
  authority ledger. (Plugin-level SHADOW was demolished by
  SHADOW-DEMOLITION-01.)

* **Cognitive chat** is gated by the ``DIX_COGNITIVE_CHAT_ENABLED``
  env flag — but the dashboard needs an in-process override so the
  operator can switch it off (or back on) without restarting uvicorn.
  The toggle is stored on a small mutable ``PluginToggleState`` and
  read by the chat runtime's feature flag via a getter closure
  installed at startup.

Both shapes return the same JSON record shape so the frontend can
render them in one grid:

.. code-block:: json

    {
      "id": "microstructure_v1",
      "category": "intelligence",
      "version": "0.1.0",
      "lifecycle": "ACTIVE",
      "lifecycle_options": ["DISABLED", "ACTIVE"],
      "description": "...",
      "ledger_kind": "PLUGIN_LIFECYCLE"
    }

The router writes through whichever ``LedgerAuthorityWriter`` the
caller hands in, so the audit trail joins the existing governance
hash chain (hash-chain integrity is preserved transparently).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.contracts.engine import MicrostructurePlugin, PluginLifecycle
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)


class _StartStopFeed(Protocol):
    """Minimal feed-runner shape consumed by the plugin manager.

    Every ``ui.feeds.*`` runner satisfies this protocol; the registry
    holds them by id so ``set_lifecycle`` can map ``ACTIVE`` to
    ``start()`` and ``DISABLED`` to ``stop()`` without import-time
    coupling to the concrete runner classes.
    """

    def start(self, *args: Any, **kwargs: Any) -> Any: ...
    def stop(self, *args: Any, **kwargs: Any) -> Any: ...
    def status(self) -> Any: ...


class _AdapterRegistryShape(Protocol):
    """Minimal :class:`AdapterRegistry` shape (introspection only)."""

    def snapshot(self) -> Any: ...


class _SensorArrayShape(Protocol):
    """Minimal :class:`SensorArray` shape (introspection only)."""

    @property
    def sensors(self) -> tuple[Any, ...]: ...

__all__ = [
    "PluginRecord",
    "PluginToggleState",
    "PluginRegistry",
    "build_plugin_router",
]


_LIFECYCLE_VALUES: tuple[str, ...] = (
    PluginLifecycle.DISABLED.value,
    PluginLifecycle.ACTIVE.value,
)


class PluginRecord(BaseModel):
    """One plugin's snapshot in the dashboard manager grid."""

    id: str
    category: str
    version: str
    lifecycle: str
    lifecycle_options: list[str]
    description: str = ""
    ledger_kind: str = "PLUGIN_LIFECYCLE"


class PluginListResponse(BaseModel):
    plugins: list[PluginRecord]


class PluginLifecycleRequest(BaseModel):
    lifecycle: str = Field(
        ...,
        description=(
            "Target lifecycle. All plugins accept DISABLED (off) "
            "and ACTIVE (on); plugin-level SHADOW was demolished."
        ),
    )
    requestor: str = "dashboard"
    reason: str = ""


@dataclass(slots=True)
class PluginToggleState:
    """In-process override for env-gated plugins.

    Today this is just the cognitive-chat flag, but the same shape
    naturally extends to any other plugin that reads its enabled
    status from the environment at startup.

    A toggle of ``None`` means "no override — defer to the env".
    Once the operator flips the dashboard switch the override is
    pinned to ``True`` or ``False`` and the env stops mattering for
    that flag until the process restarts.
    """

    cognitive_chat: bool | None = None
    _listeners: list[Callable[[str, bool | None], None]] = field(
        default_factory=list
    )

    def set_cognitive_chat(self, enabled: bool | None) -> None:
        self.cognitive_chat = enabled
        for listener in tuple(self._listeners):
            listener("cognitive_chat", enabled)

    def add_listener(
        self, listener: Callable[[str, bool | None], None]
    ) -> None:
        self._listeners.append(listener)


_FEED_DESCRIPTIONS: Mapping[str, str] = {
    "binance_public_ws": (
        "Binance public WebSocket pump (BTC/ETH/SOL aggTrades). "
        "ACTIVE connects the WS pump and emits MarketTick events "
        "into the intelligence pipeline; DISABLED stops the loop."
    ),
    "coindesk_rss": (
        "CoinDesk RSS news pump → NewsFanout (signal projection + "
        "HAZ-NEWS-SHOCK sensor). ACTIVE polls the RSS feed; "
        "DISABLED stops the runner."
    ),
    "pumpfun_ws": (
        "Pump.fun WebSocket launch firehose. ACTIVE connects to the "
        "live launch stream; DISABLED stops the loop."
    ),
    "raydium_pools": (
        "Raydium pool snapshot poller. ACTIVE polls the live pool "
        "REST endpoint; DISABLED stops the runner."
    ),
}


_ADAPTER_DESCRIPTIONS: Mapping[str, str] = {
    "hummingbot": (
        "Hummingbot Gateway adapter (centralised + DEX connectors). "
        "Read-only here — connect via the Operator → Adapters page."
    ),
    "pumpfun": (
        "Pump.fun execution adapter (Solana memecoin launches). "
        "Read-only here — DISCONNECTED until credentials are wired."
    ),
    "uniswapx": (
        "UniswapX EIP-712 signer + REST quote/order client. "
        "Read-only here — requires eth-account + signer key."
    ),
}


@dataclass(slots=True)
class PluginRegistry:
    """Façade over the runtime's hot-toggleable plugins.

    Constructed once in ``ui.server`` and handed to the router via
    ``build_plugin_router``. The registry holds *references* to live
    runtime objects (not copies), so a lifecycle mutation through
    this registry is observed immediately by the engines that
    consume those plugins on the next tick.

    The plugin grid surfaces three additional categories beyond the
    historical microstructure + cognitive_chat pair so the operator
    has a single place to see every hot-toggleable runtime
    component:

    * ``feed`` — ``ui.feeds.*`` runners (Binance / CoinDesk /
      Pump.fun / Raydium). ``ACTIVE`` calls ``start()``;
      ``DISABLED`` calls ``stop()``.
    * ``adapter`` — ``execution_engine.adapters.AdapterRegistry``
      entries. Read-only; the lifecycle column reflects the live
      ``AdapterStatus.connected`` flag.
    * ``sensor`` — every ``HazardSensor`` registered on the system
      ``SensorArray``. Read-only; lifecycle reflects whether the
      sensor is registered (``ACTIVE``) or absent.

    The new fields are optional with empty defaults so existing
    callers that constructed ``PluginRegistry`` with only the
    microstructure + cognitive arguments continue to work.
    """

    microstructure_plugins: tuple[MicrostructurePlugin, ...]
    toggle_state: PluginToggleState
    cognitive_chat_env_enabled: Callable[[], bool]
    cognitive_chat_version: str = "0.1.0"
    feed_runners: Mapping[str, _StartStopFeed] = field(default_factory=dict)
    adapter_registry: _AdapterRegistryShape | None = None
    sensor_array: _SensorArrayShape | None = None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list(self) -> list[PluginRecord]:
        out: list[PluginRecord] = []
        for plugin in self.microstructure_plugins:
            out.append(
                PluginRecord(
                    id=plugin.name,
                    category="intelligence",
                    version=plugin.version,
                    lifecycle=plugin.lifecycle.value,
                    lifecycle_options=list(_LIFECYCLE_VALUES),
                    description=(
                        "Tick-driven microstructure signal generator. "
                        "ACTIVE emits signals into the conflict "
                        "resolver; DISABLED skips the plugin entirely."
                    ),
                )
            )

        out.append(
            PluginRecord(
                id="cognitive_chat",
                category="cognitive",
                version=self.cognitive_chat_version,
                lifecycle=self._cognitive_chat_lifecycle(),
                lifecycle_options=[
                    PluginLifecycle.DISABLED.value,
                    PluginLifecycle.ACTIVE.value,
                ],
                description=(
                    "LangGraph-driven operator chat surface. ACTIVE "
                    "exposes /api/cognitive/chat/{status,turn} and "
                    "the /dash2 chat page; DISABLED returns 503 from "
                    "those endpoints. Default-on once the runtime "
                    "boots; the dashboard switch persists the "
                    "override in-process until a restart."
                ),
            )
        )

        for feed_id, runner in self.feed_runners.items():
            out.append(
                PluginRecord(
                    id=f"feed:{feed_id}",
                    category="feed",
                    version="1.0.0",
                    lifecycle=self._feed_lifecycle(runner),
                    lifecycle_options=list(_LIFECYCLE_VALUES),
                    description=_FEED_DESCRIPTIONS.get(
                        feed_id,
                        f"{feed_id} feed runner. ACTIVE starts the "
                        f"runner; DISABLED stops it.",
                    ),
                )
            )

        if self.adapter_registry is not None:
            try:
                statuses = self.adapter_registry.snapshot()
            except Exception:  # pragma: no cover - defensive
                statuses = ()
            for status in statuses:
                name = getattr(status, "name", "")
                if not name:
                    continue
                connected = bool(getattr(status, "connected", False))
                lifecycle = (
                    PluginLifecycle.ACTIVE.value
                    if connected
                    else PluginLifecycle.DISABLED.value
                )
                out.append(
                    PluginRecord(
                        id=f"adapter:{name}",
                        category="adapter",
                        version=str(getattr(status, "version", "1.0.0")),
                        lifecycle=lifecycle,
                        lifecycle_options=[lifecycle],
                        description=_ADAPTER_DESCRIPTIONS.get(
                            name,
                            f"{name} execution adapter. Read-only "
                            f"here — manage from the Operator → "
                            f"Adapters page.",
                        ),
                    )
                )

        if self.sensor_array is not None:
            for sensor in self.sensor_array.sensors:
                sensor_name = getattr(sensor, "name", "")
                if not sensor_name:
                    continue
                code = getattr(sensor, "code", "") or sensor_name
                out.append(
                    PluginRecord(
                        id=f"sensor:{sensor_name}",
                        category="sensor",
                        version="1.0.0",
                        lifecycle=PluginLifecycle.ACTIVE.value,
                        lifecycle_options=[PluginLifecycle.ACTIVE.value],
                        description=(
                            f"Hazard sensor {code} ({sensor_name}). "
                            f"Registered on SystemEngine.sensor_array "
                            f"and polled every tick. Read-only here."
                        ),
                    )
                )

        return out

    @staticmethod
    def _feed_lifecycle(runner: _StartStopFeed) -> str:
        try:
            status = runner.status()
        except Exception:  # pragma: no cover - defensive
            return PluginLifecycle.DISABLED.value
        running = bool(getattr(status, "running", False))
        return (
            PluginLifecycle.ACTIVE.value
            if running
            else PluginLifecycle.DISABLED.value
        )

    def _cognitive_chat_lifecycle(self) -> str:
        if self.toggle_state.cognitive_chat is True:
            return PluginLifecycle.ACTIVE.value
        if self.toggle_state.cognitive_chat is False:
            return PluginLifecycle.DISABLED.value
        # No override — defer to env (which itself defaults on after
        # the cognitive_chat_graph flag flip).
        return (
            PluginLifecycle.ACTIVE.value
            if self.cognitive_chat_env_enabled()
            else PluginLifecycle.DISABLED.value
        )

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------

    def set_lifecycle(
        self,
        plugin_id: str,
        lifecycle: str,
    ) -> PluginRecord:
        normalized = lifecycle.strip().upper()
        if normalized not in _LIFECYCLE_VALUES:
            raise ValueError(
                f"unknown lifecycle '{lifecycle}'. expected one of "
                f"{_LIFECYCLE_VALUES}"
            )

        if plugin_id == "cognitive_chat":
            self.toggle_state.set_cognitive_chat(
                normalized == PluginLifecycle.ACTIVE.value
            )
            return self._record_for("cognitive_chat")

        if plugin_id.startswith("feed:"):
            feed_id = plugin_id[len("feed:") :]
            runner = self.feed_runners.get(feed_id)
            if runner is None:
                raise KeyError(plugin_id)
            try:
                if normalized == PluginLifecycle.ACTIVE.value:
                    runner.start()
                else:
                    runner.stop()
            except Exception as exc:  # pragma: no cover - propagated as 500
                raise RuntimeError(
                    f"feed lifecycle change failed for {feed_id}: {exc}"
                ) from exc
            return self._record_for(plugin_id)

        if plugin_id.startswith("adapter:") or plugin_id.startswith("sensor:"):
            # Adapters and sensors are read-only on this surface; the
            # operator manages adapter connectivity from the dedicated
            # Adapters page (which calls adapter.connect() / disconnect()
            # against credentials), and sensors are wired at boot. The
            # registry refuses lifecycle writes here so the dashboard
            # cannot half-disable a sensor and silently mute a hazard.
            raise PermissionError(
                f"{plugin_id} is read-only on the plugin manager"
            )

        for plugin in self.microstructure_plugins:
            if plugin.name == plugin_id:
                plugin.lifecycle = PluginLifecycle(normalized)
                return self._record_for(plugin_id)

        raise KeyError(plugin_id)

    def _record_for(self, plugin_id: str) -> PluginRecord:
        for record in self.list():
            if record.id == plugin_id:
                return record
        raise KeyError(plugin_id)


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------


def build_plugin_router(
    *,
    registry_provider: Callable[[], PluginRegistry],
    ledger_provider: Callable[[], LedgerAuthorityWriter] | None = None,
    ts_provider: Callable[[], int] | None = None,
) -> APIRouter:
    """Build the ``/api/plugins`` FastAPI router.

    ``registry_provider`` returns the live registry (so tests can
    swap in fakes without a running ``STATE`` object).
    ``ledger_provider`` is optional; when present every successful
    lifecycle change writes a ``PLUGIN_LIFECYCLE`` audit row.
    ``ts_provider`` (also optional) provides the wall-time
    ``ts_ns`` for ledger rows; defaults to ``0`` when absent so
    fake-ledger tests stay clock-free.
    """

    router = APIRouter(prefix="/api/plugins", tags=["plugins"])

    def _ts() -> int:
        if ts_provider is None:
            return 0
        return ts_provider()

    def _audit(plugin_id: str, lifecycle: str, body: PluginLifecycleRequest) -> None:
        if ledger_provider is None:
            return
        writer = ledger_provider()
        payload: Mapping[str, str] = {
            "requestor": body.requestor,
            "plugin_path": plugin_id,
            "target_status": lifecycle,
            "reason": body.reason,
            "source": "dashboard.plugin_manager",
        }
        writer.append(
            ts_ns=_ts(),
            kind="PLUGIN_LIFECYCLE",
            payload=payload,
        )

    @router.get("", response_model=PluginListResponse)
    def list_plugins() -> PluginListResponse:
        registry = registry_provider()
        return PluginListResponse(plugins=registry.list())

    @router.post("/{plugin_id}/lifecycle", response_model=PluginRecord)
    def set_lifecycle(
        plugin_id: str, body: PluginLifecycleRequest
    ) -> PluginRecord:
        registry = registry_provider()
        try:
            record = registry.set_lifecycle(plugin_id, body.lifecycle)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown plugin id: {plugin_id}"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        try:
            _audit(plugin_id, record.lifecycle, body)
        except Exception as exc:  # pragma: no cover — surfaced as 500
            # Mutation already happened; audit failure should not
            # silently roll back the toggle, but the operator must
            # see the failure.
            raise HTTPException(
                status_code=500,
                detail=f"plugin toggled but ledger audit failed: {exc!r}",
            ) from exc

        return record

    return router


# Re-exported helpers ---------------------------------------------------

def lifecycle_to_str(lifecycle: PluginLifecycle) -> str:
    return lifecycle.value


def _record_to_dict(record: PluginRecord) -> dict[str, Any]:
    """Stable JSON-friendly form for tests."""
    return record.model_dump()
