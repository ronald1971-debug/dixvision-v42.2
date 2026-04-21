"""
cockpit.launcher — single entrypoint for desktop / cloud / worker modes.

    python -m cockpit                     # auto (DIX_MODE env, defaults to desktop)
    python -m cockpit --mode desktop      # loopback only, 127.0.0.1:8765
    python -m cockpit --mode cloud        # 0.0.0.0 bind, CORS from env
    python -m cockpit --mode worker       # no HTTP; background ingestion only

Relevant env vars:
    DIX_MODE               desktop | cloud | worker
    DIX_BIND_HOST          override bind address (default: loopback in desktop,
                           0.0.0.0 in cloud)
    DIX_PORT               8765 default
    DIX_PUBLIC_URL         https://cockpit.example.com   (for pairing QR)
    DIX_ALLOWED_ORIGINS    comma-separated list used by the CORS middleware
    DIX_COCKPIT_TOKEN      bearer token (auto-generated on first boot)

The worker mode is for headless 24/7 learning/sourcing containers: no HTTP
surface, just the bootstrap providers + background refresh loops.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def _resolve_mode(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(prog="dix-cockpit", add_help=False)
    parser.add_argument("--mode", default=os.environ.get("DIX_MODE", "desktop"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("-h", "--help", action="store_true")
    ns, _ = parser.parse_known_args(argv)
    if ns.help:
        print(__doc__)
        sys.exit(0)
    mode = (ns.mode or "desktop").lower().strip()
    if mode not in {"desktop", "cloud", "worker"}:
        print(f"unknown --mode {mode!r}; choose desktop|cloud|worker", file=sys.stderr)
        sys.exit(2)
    if ns.host:
        os.environ["DIX_BIND_HOST"] = ns.host
    if ns.port:
        os.environ["DIX_PORT"] = str(ns.port)
    return mode


def _default_host(mode: str) -> str:
    env = os.environ.get("DIX_BIND_HOST", "").strip()
    if env:
        return env
    return "0.0.0.0" if mode == "cloud" else "127.0.0.1"


def _default_port() -> int:
    try:
        return int(os.environ.get("DIX_PORT", "8765"))
    except ValueError:
        return 8765


def run_http(mode: str) -> None:
    try:
        import uvicorn
    except Exception:
        print("uvicorn is required for http modes; install: pip install 'uvicorn[standard]' fastapi",
              file=sys.stderr)
        sys.exit(3)
    from cockpit.app import create_app
    app = create_app()
    host = _default_host(mode)
    port = _default_port()
    print(f"[cockpit] mode={mode} bind={host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


def run_worker() -> None:
    """Headless 24/7 ingestion loop for cloud worker containers."""
    from mind.sources.providers import bootstrap_all_providers
    from mind.strategy_arbiter import get_arbiter
    from state.ledger.writer import get_writer
    from system_monitor.dead_man import get_dead_man

    writer = get_writer()
    bootstrap_all_providers()
    arb = get_arbiter()
    dm = get_dead_man()
    interval = int(os.environ.get("DIX_WORKER_INTERVAL_SEC", "60"))
    print(f"[worker] starting; refresh every {interval}s", flush=True)

    stopping = {"flag": False}

    def _sig(_signum, _frame) -> None:  # noqa: ANN001
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    while not stopping["flag"]:
        try:
            arb.refresh_decay()
            dm.heartbeat(source="worker")
            writer.append_event(stream="SYSTEM", kind="WORKER_TICK",
                                payload={"interval_sec": interval})
        except Exception as exc:                                                # noqa: BLE001
            print(f"[worker] tick error: {exc}", file=sys.stderr, flush=True)
        for _ in range(interval):
            if stopping["flag"]:
                break
            time.sleep(1)
    print("[worker] shutdown", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = _resolve_mode(argv)
    if mode == "worker":
        run_worker()
    else:
        run_http(mode)


if __name__ == "__main__":
    main()
