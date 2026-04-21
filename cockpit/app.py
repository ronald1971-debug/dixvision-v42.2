"""
cockpit.app \u2014 FastAPI operator dashboard.

Loopback-only by default, bearer-token gated, surfaces:
    - /health                     public liveness probe
    - /api/status                 overall status
    - /api/locale                 current locale + supported UI langs
    - /api/charters               all voice charters
    - /api/providers              25 data-source providers + enabled state
    - /api/ai                     AI router provider status + roles
    - /api/traders/count          trader KB counts
    - /api/traders/search?q=...   search traders by name/style/region
    - /api/chat                   POST {message, voice?, locale?} -> voice answer
    - /api/chat/history?limit=..  chat transcript
    - /api/risk                   current fast_risk_cache snapshot
    - /static/                    SPA assets (HTML + JS + i18n.json)
"""
from __future__ import annotations

import os
from pathlib import Path

try:                                                                            # pragma: no cover
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    _FASTAPI_OK = True
except Exception:                                                               # pragma: no cover
    _FASTAPI_OK = False

from cockpit import pairing as _pairing
from cockpit.auth import TokenAuthMiddleware, get_or_create_token
from cockpit.chat import get_chat
from cockpit.llm import get_router as get_llm_router
from cockpit.qr import qr_png_bytes
from core.charter import Voice, all_charters
from mind.knowledge.trader_knowledge import get_trader_knowledge
from mind.sources.providers import bootstrap_all_providers, provider_summary
from mind.strategy_arbiter import get_arbiter
from security import wallet_connect as _wc
from security import wallet_policy as _wp
from state.episodic_memory import get_episodic_memory
from state.ledger.writer import get_writer
from system.fast_risk_cache import get_risk_cache
from system.locale import current as current_locale
from system.locale import set_override, supported_ui_languages
from system_monitor.dead_man import get_dead_man
from system_monitor.latency_guard import get_latency_guard

if _FASTAPI_OK:
    class ChatIn(BaseModel):
        message: str
        voice: str | None = None
        locale: str | None = None

    class LocaleIn(BaseModel):
        tag: str

    class WalletIn(BaseModel):
        label: str
        chain: str
        backend: str = "watch_only"
        address: str
        notes: str = ""

    class WalletApproveIn(BaseModel):
        chain: str
        address: str
        approved_by: str
        expires_utc: str

    class PairingIssueIn(BaseModel):
        label: str
        ttl_sec: int = 900

    class PairingClaimIn(BaseModel):
        token: str
        device: str = "unknown"


def _charters_payload() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for voice, c in all_charters().items():
        out.append({
            "voice": voice.value,
            "domain": c.domain.value,
            "what": c.what,
            "how": list(c.how),
            "why": list(c.why),
            "not_do": list(c.not_do),
            "accountability": list(c.accountability),
            "tools": list(c.tools),
            "peers_readable": c.peers_readable,
        })
    return out


def _risk_payload() -> dict[str, object]:
    rc = get_risk_cache().get()
    return {
        "max_order_size_usd": rc.max_order_size_usd,
        "max_position_pct": rc.max_position_pct,
        "circuit_breaker_drawdown": rc.circuit_breaker_drawdown,
        "circuit_breaker_loss_pct": rc.circuit_breaker_loss_pct,
        "trading_allowed": rc.trading_allowed,
        "safe_mode": rc.safe_mode,
        "last_updated_utc": rc.last_updated_utc,
    }


def _ai_payload() -> dict[str, object]:
    rows = get_llm_router().status()
    return {
        "providers": [
            {
                "name": s.name,
                "role": s.role,
                "model": s.model,
                "enabled": s.enabled,
                "has_key": s.has_key,
                "capabilities": s.capabilities,
                "cost_per_1k_tokens_usd": s.cost_per_1k_tokens_usd,
                "local": s.local,
                "total_calls": s.total_calls,
                "total_cost_usd": round(s.total_cost_usd, 6),
                "last_error": s.last_error,
            }
            for s in rows
        ],
    }


def _providers_payload() -> dict[str, object]:
    bootstrap_all_providers()
    return {"providers": provider_summary()}


def _traders_search(q: str = "", style: str = "", region: str = "",
                    limit: int = 50) -> dict[str, object]:
    kb = get_trader_knowledge()
    rows = kb.find_traders(q=q, style=style, region=region, limit=min(limit, 200))
    return {
        "count": len(rows),
        "results": [
            {"id": t.id, "name": t.name, "era": t.era, "region": t.region,
             "style_tags": t.style_tags.split(",") if t.style_tags else [],
             "cautionary": bool(t.cautionary),
             "bio_summary": t.bio_summary, "bio_lang": t.bio_lang,
             "source_url": t.source_url}
            for t in rows
        ],
    }


def create_app() -> FastAPI:
    if not _FASTAPI_OK:                                                         # pragma: no cover
        raise RuntimeError("fastapi is not installed; cockpit unavailable")

    app = FastAPI(title="DIX VISION Cockpit", version="42.2.0", docs_url=None, redoc_url=None)
    token = get_or_create_token()
    app.add_middleware(TokenAuthMiddleware, token=token)

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index() -> Response:
        idx = static_dir / "index.html"
        if idx.is_file():
            return FileResponse(str(idx))
        return JSONResponse({"service": "DIX VISION Cockpit", "version": "42.2.0"})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "42.2.0"})

    @app.get("/pair")
    async def pair_page() -> Response:
        p = static_dir / "pair.html"
        if p.is_file():
            return FileResponse(str(p))
        return JSONResponse({"error": "pair_page_missing"}, status_code=404)

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse({
            "version": "42.2.0",
            "locale": current_locale().__dict__,
            "risk": _risk_payload(),
            "charters": len(all_charters()),
            "ai_providers": len(_ai_payload()["providers"]),
        })

    @app.get("/api/locale")
    async def locale() -> JSONResponse:
        return JSONResponse({
            "current": current_locale().__dict__,
            "supported_ui": list(supported_ui_languages()),
        })

    @app.post("/api/locale")
    async def set_locale(body: LocaleIn) -> JSONResponse:
        info = set_override(body.tag)
        return JSONResponse(info.__dict__)

    @app.get("/api/charters")
    async def charters() -> JSONResponse:
        return JSONResponse({"charters": _charters_payload()})

    @app.get("/api/providers")
    async def providers() -> JSONResponse:
        return JSONResponse(_providers_payload())

    @app.get("/api/ai")
    async def ai() -> JSONResponse:
        return JSONResponse(_ai_payload())

    @app.get("/api/risk")
    async def risk() -> JSONResponse:
        return JSONResponse(_risk_payload())

    @app.get("/api/traders/count")
    async def traders_count() -> JSONResponse:
        return JSONResponse({"count": get_trader_knowledge().count()})

    @app.get("/api/traders/search")
    async def traders_search(q: str = "", style: str = "", region: str = "",
                             limit: int = 50) -> JSONResponse:
        return JSONResponse(_traders_search(q=q, style=style, region=region, limit=limit))

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> JSONResponse:
        fv: Voice | None = None
        if body.voice:
            try:
                fv = Voice(body.voice.strip().upper())
            except Exception:
                raise HTTPException(status_code=400, detail="unknown_voice")
        turn = get_chat().send(body.message, forced_voice=fv,
                               locale_tag=body.locale or "")
        return JSONResponse({
            "voice": turn.voice.value,
            "language": turn.language,
            "answer": turn.answer,
            "model": turn.model_used,
            "ledger_refs": turn.ledger_refs,
        })

    @app.get("/api/wallets")
    async def wallets() -> JSONResponse:
        rows = _wc.list_wallets()
        return JSONResponse({"count": len(rows), "wallets": [
            {"id": w.id, "label": w.label, "chain": w.chain.value,
             "backend": w.backend.value, "address_masked": w.mask(),
             "live_signing_allowed": w.live_signing_allowed,
             "approval_expires_utc": w.approval_expires_utc,
             "notes": w.notes}
            for w in rows
        ]})

    @app.post("/api/wallets")
    async def wallet_connect(body: WalletIn) -> JSONResponse:
        try:
            chain = _wc.Chain(body.chain.lower())
            backend = _wc.Backend(body.backend.lower())
        except Exception:
            raise HTTPException(status_code=400, detail="unknown_chain_or_backend")
        w = _wc.connect_wallet(label=body.label, chain=chain,
                               backend=backend, address=body.address,
                               notes=body.notes)
        return JSONResponse({"id": w.id, "chain": w.chain.value,
                             "backend": w.backend.value,
                             "address_masked": w.mask()})

    @app.post("/api/wallets/approve")
    async def wallet_approve(body: WalletApproveIn) -> JSONResponse:
        try:
            chain = _wc.Chain(body.chain.lower())
        except Exception:
            raise HTTPException(status_code=400, detail="unknown_chain")
        try:
            w = _wc.approve_live_signing(chain, body.address,
                                         approved_by=body.approved_by,
                                         expires_utc=body.expires_utc)
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if w is None:
            raise HTTPException(status_code=404, detail="wallet_not_found")
        return JSONResponse({"live_signing_allowed": w.live_signing_allowed,
                             "approval_expires_utc": w.approval_expires_utc})

    @app.get("/api/wallet/policy")
    async def wallet_policy() -> JSONResponse:
        return JSONResponse(_wp.snapshot().as_dict())

    @app.get("/api/strategies")
    async def strategies() -> JSONResponse:
        arb = get_arbiter()
        arb.refresh_decay()
        return JSONResponse({"strategies": arb.state()})

    @app.get("/api/episodic/count")
    async def episodic_count() -> JSONResponse:
        return JSONResponse({"count": get_episodic_memory().count()})

    @app.get("/api/safety")
    async def safety() -> JSONResponse:
        return JSONResponse({
            "dead_man": get_dead_man().status().as_dict(),
            "latency_guard": get_latency_guard().snapshot().as_dict(),
        })

    @app.post("/api/safety/heartbeat")
    async def safety_heartbeat() -> JSONResponse:
        get_dead_man().heartbeat(source="cockpit")
        return JSONResponse(get_dead_man().status().as_dict())

    @app.post("/api/pair/new")
    async def pair_new(body: PairingIssueIn) -> JSONResponse:
        p = _pairing.issue_pairing(label=body.label, ttl_sec=body.ttl_sec)
        base = os.environ.get("DIX_PUBLIC_URL", "").rstrip("/")
        if not base:
            host = os.environ.get("DIX_BIND_HOST", "127.0.0.1")
            port = os.environ.get("DIX_PORT", "8765")
            base = f"http://{host}:{port}"
        claim_url = f"{base}/pair?t={p.token}"
        return JSONResponse({"token": p.token, "label": p.label,
                             "expires_utc": p.expires_utc,
                             "claim_url": claim_url})

    @app.get("/api/pair/list")
    async def pair_list() -> JSONResponse:
        rows = _pairing.list_pairings()
        return JSONResponse({"pairings": [
            {"token_prefix": r.token[:6], "label": r.label,
             "created_utc": r.created_utc, "expires_utc": r.expires_utc,
             "consumed": bool(r.consumed_utc), "revoked": bool(r.revoked_utc),
             "device": r.device_fingerprint or ""}
            for r in rows
        ]})

    @app.post("/api/pair/revoke")
    async def pair_revoke(body: PairingClaimIn) -> JSONResponse:
        ok = _pairing.revoke_pairing(body.token)
        return JSONResponse({"revoked": ok})

    @app.post("/api/pair/claim")
    async def pair_claim(body: PairingClaimIn) -> JSONResponse:
        tok = _pairing.claim_pairing(body.token,
                                     bearer_token=token,
                                     device_fingerprint=body.device)
        if tok is None:
            raise HTTPException(status_code=404, detail="pairing_invalid_or_expired")
        return JSONResponse({"token": tok})

    @app.get("/api/pair/qr")
    async def pair_qr(t: str) -> Response:
        base = os.environ.get("DIX_PUBLIC_URL", "").rstrip("/")
        if not base:
            host = os.environ.get("DIX_BIND_HOST", "127.0.0.1")
            port = os.environ.get("DIX_PORT", "8765")
            base = f"http://{host}:{port}"
        payload = f"{base}/pair?t={t}"
        return Response(content=qr_png_bytes(payload, module_px=8),
                        media_type="image/png")

    @app.get("/api/chat/history")
    async def chat_history(limit: int = 50) -> JSONResponse:
        turns = get_chat().history(limit=limit)
        return JSONResponse({
            "history": [
                {"voice": t.voice.value, "message": t.operator_message,
                 "answer": t.answer, "language": t.language,
                 "model": t.model_used}
                for t in turns
            ],
        })

    # warm-start
    get_writer()
    bootstrap_all_providers()
    get_chat()
    return app


app: FastAPI | None = None
try:                                                                            # pragma: no cover
    app = create_app()
except Exception:
    app = None


__all__ = ["create_app", "app"]
