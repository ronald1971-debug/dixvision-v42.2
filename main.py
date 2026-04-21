"""
main.py
DIX VISION v42.2 — Runtime Launcher
"""
from __future__ import annotations

import signal
import sys
import time


def main() -> None:
    from bootstrap_kernel import run
    from execution.engine import get_dyon_engine
    from mind.engine import IndiraEngine
    from system.health_monitor import get_health_monitor
    from system.logger import get_logger
    from system.state import get_state_manager

    env = "dev" if "--dev" in sys.argv else "prod"
    verify_only = "--verify" in sys.argv

    run(env=env, verify_only=verify_only)
    if verify_only:
        return

    log = get_logger("main")
    state_mgr = get_state_manager()
    health = get_health_monitor()

    # Build Indira fast-path engine
    indira = IndiraEngine()

    log.info("[MAIN] Entering trading loop")
    print("\n[DIX VISION v42.2] System ONLINE. Press Ctrl+C to stop.\n")

    # Graceful shutdown handler
    def _shutdown(sig, frame):
        print("\n[DIX VISION] Shutdown signal received...")
        state_mgr.set_mode("HALTED")
        dyon = get_dyon_engine()
        dyon.stop()
        from enforcement.runtime_guardian import get_runtime_guardian
        get_runtime_guardian().stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Simulated market data loop (replace with live feed in Phase 8)
    tick = 0
    while True:
        state_mgr.heartbeat()  # Update heartbeat — guardian monitors this
        tick += 1

        # Simulate market tick
        import math
        signal_val = math.sin(tick * 0.1) * 0.8
        market_data = {
            "signal": signal_val,
            "asset": "BTCUSDT",
            "price": 65_000.0 + (signal_val * 500),
            "data_quality": 0.95,
            "execution_confidence": 0.90,
            "strategy": "regime_adaptive",
        }

        # Indira fast path
        ev = indira.process_tick(market_data)
        if ev.event_type != "HOLD" and tick % 20 == 0:
            log.info(f"Indira: {ev.event_type} {ev.asset} side={ev.side} "
                     f"size_usd={ev.size_usd:.0f} latency_ms={ev.latency_ns/1e6:.2f}")

        # Health status every 60 ticks
        if tick % 60 == 0:
            health.print_status()

        time.sleep(0.1)  # 10Hz loop — replace with WebSocket push in Phase 8

if __name__ == "__main__":
    main()
