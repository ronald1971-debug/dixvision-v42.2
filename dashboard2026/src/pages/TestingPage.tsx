import { Backtester } from "@/widgets/testing/Backtester";

/**
 * Testing & evaluation surface — the operator's "lab" for running
 * strategies against historical bars before they enter the strategy
 * lifecycle (PAPER → SHADOW → CANARY → LIVE → DECAY).
 *
 * Currently mounts the deterministic Backtester (PR-#2 spec §5.1).
 * Forward tester, walk-forward, replay harness, regime-shift fixtures,
 * promotion-gates dashboard, and drift-oracle dashboard land in
 * follow-up commits and will all live on this surface.
 */
export function TestingPage() {
  return (
    <section className="flex h-full flex-col">
      <header className="mb-3 flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            Testing &amp; Evaluation{" "}
            <span className="ml-2 rounded border border-border bg-bg px-2 py-0.5 font-mono text-[11px] uppercase tracking-widest text-slate-400">
              LAB
            </span>
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Backtest, forward-test, walk-forward, replay, and regime-shift
            harnesses. Every run is governed by the same audit ledger and
            promotion gates as live trading — there is no untracked
            evaluation surface.
          </p>
        </div>
      </header>
      <div className="flex-1 overflow-auto pb-6">
        <Backtester />
      </div>
    </section>
  );
}
