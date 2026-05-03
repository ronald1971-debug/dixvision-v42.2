import { Backtester } from "@/widgets/testing/Backtester";
import { ForwardTester } from "@/widgets/testing/ForwardTester";
import { RegimeShiftBoard } from "@/widgets/testing/RegimeShiftBoard";
import { ReplayHarness } from "@/widgets/testing/ReplayHarness";
import { WalkForwardHarness } from "@/widgets/testing/WalkForwardHarness";

/**
 * Testing & evaluation surface — the operator's "lab" for running
 * strategies against historical bars before they enter the strategy
 * lifecycle (PAPER → SHADOW → CANARY → LIVE → DECAY).
 *
 * Mounts the full Tier-8 testing harness: Backtester (deterministic
 * historical run), ForwardTester (30-day SHADOW gate), Walk-forward
 * (IS/OOS robustness), Replay harness (ledger re-run with variant
 * comparison), Regime-shift fixtures (canonical stress windows).
 *
 * Every harness is gated by the same audit ledger and promotion
 * gates as live trading — there is no untracked evaluation surface.
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
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="lg:col-span-2">
            <Backtester />
          </div>
          <ForwardTester />
          <WalkForwardHarness />
          <ReplayHarness />
          <RegimeShiftBoard />
        </div>
      </div>
    </section>
  );
}
