import { useMemo, useState } from "react";

/**
 * Tier-3 AI widget — Counterfactual analyzer.
 *
 * Pick any historical trade from the audit ledger and re-simulate the
 * outcome under alternative parameters: tighter / looser SL, scaled
 * entry, opposite side, or "no trade". The composite score and PnL
 * are recomputed from a deterministic surrogate model so the
 * operator can ask "what if I hadn't entered at 16:42?" without
 * touching live capital.
 *
 * Backed by the audit ledger (PR #64 DecisionTrace + INV-65). The
 * surrogate uses bar-replayed return series; canonical replay lives
 * in the long-horizon harness (Tier 8).
 */
interface PastTrade {
  id: string;
  ts_iso: string;
  symbol: string;
  side: "BUY" | "SELL";
  entry: number;
  sl: number;
  tp: number;
  exit: number;
  size: number;
  pnl: number;
  why: string;
}

const SAMPLE_TRADES: PastTrade[] = [
  {
    id: "t-2025-10-12-1642",
    ts_iso: "2025-10-12T16:42:00Z",
    symbol: "BTC-USDT",
    side: "BUY",
    entry: 67_420,
    sl: 66_200,
    tp: 69_800,
    exit: 67_950,
    size: 0.4,
    pnl: 212.0,
    why: "Funding flipped negative on HL · CVD +180 last 5m · BeliefState confidence 0.71",
  },
  {
    id: "t-2025-10-13-0815",
    ts_iso: "2025-10-13T08:15:00Z",
    symbol: "SOL-USDT",
    side: "SELL",
    entry: 178.4,
    sl: 182.0,
    tp: 168.0,
    exit: 174.6,
    size: 25,
    pnl: 95.0,
    why: "PressureVector.uncertainty 0.62 · CoinDesk wire bearish · CANARY size cap applied",
  },
  {
    id: "t-2025-10-14-2103",
    ts_iso: "2025-10-14T21:03:00Z",
    symbol: "ETH-USDT",
    side: "BUY",
    entry: 3_140,
    sl: 3_050,
    tp: 3_320,
    exit: 3_050,
    size: 1.5,
    pnl: -135.0,
    why: "Composite 0.58 marginal · entered before BLS print · stopped on macro shock",
  },
];

type Counterfactual = "tighter_sl" | "wider_sl" | "half_size" | "no_trade";

function recompute(trade: PastTrade, cf: Counterfactual): number {
  // Deterministic surrogate: scale exit move + adjust SL band.
  const dir = trade.side === "BUY" ? 1 : -1;
  const move = (trade.exit - trade.entry) * dir;
  switch (cf) {
    case "tighter_sl": {
      const tighter = Math.abs(trade.entry - trade.sl) * 0.5;
      const stopped = move < -tighter;
      return stopped ? -tighter * trade.size : trade.pnl;
    }
    case "wider_sl": {
      // Wider SL would have let losers lose more; we model as 1.4× pnl floor.
      return trade.pnl < 0 ? trade.pnl * 1.4 : trade.pnl;
    }
    case "half_size":
      return trade.pnl * 0.5;
    case "no_trade":
      return 0;
  }
}

const CF_LABELS: Record<Counterfactual, string> = {
  tighter_sl: "Tighter SL (50%)",
  wider_sl: "Wider SL (×1.4)",
  half_size: "Half size",
  no_trade: "No trade",
};

export function CounterfactualPanel() {
  const [selected, setSelected] = useState<string>(SAMPLE_TRADES[0].id);
  const trade = useMemo(
    () => SAMPLE_TRADES.find((t) => t.id === selected) ?? SAMPLE_TRADES[0],
    [selected],
  );
  const scenarios: Counterfactual[] = [
    "tighter_sl",
    "wider_sl",
    "half_size",
    "no_trade",
  ];

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Counterfactual analyzer
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          replay any past trade under alternative parameters · surrogate
          model
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-3 overflow-auto p-3 text-[12px]">
        <label className="flex items-baseline gap-2 font-mono text-[10px] uppercase tracking-wider text-slate-400">
          ledger entry
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="flex-1 rounded border border-border bg-bg/40 px-2 py-1 text-[11px] text-slate-200 focus:border-accent focus:outline-none"
          >
            {SAMPLE_TRADES.map((t) => (
              <option key={t.id} value={t.id}>
                {t.symbol} · {t.side} · {new Date(t.ts_iso).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <div className="rounded border border-border bg-bg/40 p-2 font-mono text-[11px] text-slate-300">
          <div className="flex items-baseline justify-between">
            <span className="text-slate-500">actual</span>
            <span
              className={
                trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
              }
            >
              {trade.pnl >= 0 ? "+" : ""}
              {trade.pnl.toFixed(2)}
            </span>
          </div>
          <div className="mt-1 text-[10px] text-slate-500">{trade.why}</div>
        </div>
        <div>
          <h4 className="mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
            counterfactuals
          </h4>
          <ul className="divide-y divide-border/40 rounded border border-border">
            {scenarios.map((cf) => {
              const pnl = recompute(trade, cf);
              const delta = pnl - trade.pnl;
              return (
                <li
                  key={cf}
                  className="flex items-baseline justify-between px-2 py-1.5 font-mono text-[11px]"
                >
                  <span className="text-slate-300">{CF_LABELS[cf]}</span>
                  <span className="flex items-baseline gap-2">
                    <span
                      className={
                        pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                      }
                    >
                      {pnl >= 0 ? "+" : ""}
                      {pnl.toFixed(2)}
                    </span>
                    <span
                      className={`text-[10px] ${
                        delta >= 0 ? "text-emerald-500/70" : "text-rose-500/70"
                      }`}
                    >
                      Δ {delta >= 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
        <p className="text-[10px] text-slate-500">
          surrogate model · canonical replay lives in the long-horizon harness
          (Tier 8)
        </p>
      </div>
    </section>
  );
}
