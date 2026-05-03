import { CorrelationMatrix } from "@/widgets/risk/CorrelationMatrix";
import { GreeksPanel } from "@/widgets/risk/GreeksPanel";
import { LiqCalc } from "@/widgets/risk/LiqCalc";
import { OptionsChain } from "@/widgets/risk/OptionsChain";
import { ScenarioBook } from "@/widgets/risk/ScenarioBook";

/**
 * Risk surface — Tier-6 of the 2026 cockpit.
 *
 * Cross-asset risk analytics: options chain, portfolio Greeks,
 * isolated-margin liquidation calculator, scenario book (price-shock
 * × IV-shock grid + canned regime fixtures), and rolling 30d
 * correlation matrix.
 *
 * All five widgets render against deterministic mocks today; live
 * wiring runs through the Tier-6 backend (position aggregator +
 * options venue adapters: Deribit / OKX-options / CME index options).
 */
export function RiskPage() {
  return (
    <section className="flex h-full flex-col">
      <header className="mb-3 flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            Risk &amp; Greeks{" "}
            <span className="ml-2 rounded border border-border bg-bg px-2 py-0.5 font-mono text-[11px] uppercase tracking-widest text-slate-400">
              CROSS-ASSET
            </span>
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Option chain, portfolio Greeks, liquidation distance,
            scenario book, and correlation matrix. All widgets feed the
            governance promotion gates and are subject to the same
            kill-switch as the execution path.
          </p>
        </div>
      </header>
      <div className="grid flex-1 grid-cols-1 gap-3 overflow-auto pb-6 md:grid-cols-2 xl:grid-cols-6">
        <div className="md:col-span-2 xl:col-span-3 xl:row-span-2">
          <OptionsChain />
        </div>
        <div className="xl:col-span-3">
          <GreeksPanel />
        </div>
        <div className="xl:col-span-3">
          <LiqCalc />
        </div>
        <div className="md:col-span-2 xl:col-span-3">
          <ScenarioBook />
        </div>
        <div className="md:col-span-2 xl:col-span-3">
          <CorrelationMatrix />
        </div>
      </div>
    </section>
  );
}
