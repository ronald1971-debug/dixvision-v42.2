"""
dix.py
DIX VISION v42.2 — CLI Entrypoint

Commands:
  python dix.py start          # start the system
  python dix.py start --dev    # development mode (lax integrity)
  python dix.py verify         # verify integrity and exit
  python dix.py status         # print system status
  python dix.py ledger check   # verify event chain integrity
  python dix.py stop           # graceful shutdown
"""
from __future__ import annotations

import sys


def cmd_start(args) -> None:
    from main import main
    if "--dev" in args:
        sys.argv.append("--dev")
    main()

def cmd_verify(args) -> None:
    from bootstrap_kernel import run
    run(env="dev", verify_only=True)
    print("\n✅ System verification passed.")

def cmd_status(args) -> None:
    from system.fast_risk_cache import get_risk_cache
    from system.state import get_state
    state = get_state()
    print(f"\nMode: {state.mode}")
    print(f"Health: {state.health:.2f}")
    print(f"Trading allowed: {state.trading_allowed}")
    print(f"Active hazards: {state.active_hazards}")
    rc = get_risk_cache().get()
    print(f"Max order USD: {rc.max_order_size_usd:.0f}")
    print(f"Safe mode: {rc.safe_mode}")

def cmd_ledger_check(args) -> None:
    from state.ledger.hash_chain import verify_full_chain
    ok, msg = verify_full_chain()
    icon = "✅" if ok else "❌"
    print(f"{icon} Ledger chain: {msg}")

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python dix.py [start|verify|status|ledger]")
        return
    cmd = args[0].lower()
    rest = args[1:]
    commands = {
        "start": cmd_start,
        "verify": cmd_verify,
        "status": cmd_status,
        "ledger": cmd_ledger_check,
    }
    fn = commands.get(cmd)
    if fn:
        fn(rest)
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")

if __name__ == "__main__":
    main()
