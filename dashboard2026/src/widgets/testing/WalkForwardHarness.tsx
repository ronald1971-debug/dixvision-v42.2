import { useMemo, useState } from "react";

import { LineChart } from "lucide-react";

/**
 * Tier-8 testing widget — walk-forward harness.
 *
 * Rolling train/test windows. Each fold tunes parameters on the
 * in-sample (IS) window and evaluates *only* on the next out-of-sample
 * (OOS) window. The fold table + IS/OOS curve answer the single
 * question: "is the strategy curve-fit, or does the edge persist?"
 *
 * Live wiring source: `evaluation.walk_forward.run(strategy, ranges)`
 * — currently rendered from a deterministic seed of (strategy, n_folds,
 * is_days, oos_days) so the panel is interactive before the backend
 * route lands.
 */
type Strategy =
  | "ema_cross_20_50"
  | "rsi_2_meanrev"
  | "vwap_reversion"
  | "breakout_channel"
  | "microstructure_v1"
  | "memecoin_copy";

const STRATEGIES: ReadonlyArray<{ key: Strategy; label: string }> = [
  { key: "ema_cross_20_50", label: "EMA cross 20/50" },
  { key: "rsi_2_meanrev", label: "RSI(2) mean-reversion" },
  { key: "vwap_reversion", label: "VWAP reversion" },
  { key: "breakout_channel", label: "Breakout channel" },
  { key: "microstructure_v1", label: "Microstructure v1" },
  { key: "memecoin_copy", label: "Memecoin copy-trader" },
];

interface Fold {
  index: number;
  is_sharpe: number;
  oos_sharpe: number;
  is_pct: number;
  oos_pct: number;
}

function hashSeed(parts: ReadonlyArray<string | number>): number {
  let h = 2166136261;
  for (const p of parts) {
    const s = String(p);
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
  }
  return h >>> 0;
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function runFolds(
  strategy: Strategy,
  nFolds: number,
  isDays: number,
  oosDays: number,
): Fold[] {
  const r = rng(hashSeed([strategy, nFolds, isDays, oosDays]));
  const folds: Fold[] = [];
  for (let i = 0; i < nFolds; i += 1) {
    const isSharpe = 0.7 + r() * 1.6;
    const decay = strategy === "memecoin_copy" ? 0.55 : 0.78;
    const oosSharpe = Math.max(-0.4, isSharpe * decay + (r() - 0.5) * 0.6);
    folds.push({
      index: i + 1,
      is_sharpe: isSharpe,
      oos_sharpe: oosSharpe,
      is_pct: (r() - 0.2) * 14,
      oos_pct: (r() - 0.35) * 10,
    });
  }
  return folds;
}

export function WalkForwardHarness() {
  const [strategy, setStrategy] = useState<Strategy>("ema_cross_20_50");
  const [nFolds, setNFolds] = useState(8);
  const [isDays, setIsDays] = useState(180);
  const [oosDays, setOosDays] = useState(30);

  const folds = useMemo(
    () => runFolds(strategy, nFolds, isDays, oosDays),
    [strategy, nFolds, isDays, oosDays],
  );

  const isAvg =
    folds.reduce((a, f) => a + f.is_sharpe, 0) / Math.max(1, folds.length);
  const oosAvg =
    folds.reduce((a, f) => a + f.oos_sharpe, 0) / Math.max(1, folds.length);
  const decay = isAvg > 0 ? 1 - oosAvg / isAvg : 0;
  const overfit = decay > 0.5;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <LineChart className="h-3.5 w-3.5 text-accent" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Walk-forward harness
          </h3>
        </div>
        <span
          className={`rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest ${
            overfit
              ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
              : "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
          }`}
        >
          {overfit ? "overfit risk" : "edge persists"}
        </span>
      </header>

      <div className="grid grid-cols-2 gap-2 border-b border-border px-3 py-2 text-[11px] md:grid-cols-4">
        <Field label="Strategy">
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as Strategy)}
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          >
            {STRATEGIES.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Folds">
          <input
            type="number"
            min={3}
            max={20}
            value={nFolds}
            onChange={(e) =>
              setNFolds(Math.max(3, Math.min(20, Number(e.target.value) || 0)))
            }
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          />
        </Field>
        <Field label="IS days">
          <input
            type="number"
            min={30}
            max={720}
            value={isDays}
            onChange={(e) =>
              setIsDays(Math.max(30, Math.min(720, Number(e.target.value) || 0)))
            }
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          />
        </Field>
        <Field label="OOS days">
          <input
            type="number"
            min={5}
            max={180}
            value={oosDays}
            onChange={(e) =>
              setOosDays(Math.max(5, Math.min(180, Number(e.target.value) || 0)))
            }
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          />
        </Field>
      </div>

      <div className="grid grid-cols-3 gap-2 border-b border-border px-3 py-2 text-xs">
        <Metric label="IS Sharpe (avg)" value={isAvg.toFixed(2)} />
        <Metric label="OOS Sharpe (avg)" value={oosAvg.toFixed(2)} />
        <Metric
          label="IS→OOS decay"
          value={`${(decay * 100).toFixed(0)}%`}
          tone={overfit ? "rose" : "emerald"}
        />
      </div>

      <div className="flex-1 overflow-auto p-3">
        <table className="w-full text-[11px]">
          <thead className="text-[10px] uppercase tracking-widest text-slate-500">
            <tr>
              <th className="pb-1 text-left">Fold</th>
              <th className="pb-1 text-right">IS Sharpe</th>
              <th className="pb-1 text-right">OOS Sharpe</th>
              <th className="pb-1 text-right">IS %</th>
              <th className="pb-1 text-right">OOS %</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {folds.map((f) => (
              <tr key={f.index} className="border-t border-border">
                <td className="py-1 text-slate-400">#{f.index}</td>
                <td className="py-1 text-right text-slate-200">
                  {f.is_sharpe.toFixed(2)}
                </td>
                <td
                  className={`py-1 text-right ${
                    f.oos_sharpe < 0 ? "text-rose-300" : "text-emerald-300"
                  }`}
                >
                  {f.oos_sharpe.toFixed(2)}
                </td>
                <td className="py-1 text-right text-slate-300">
                  {f.is_pct.toFixed(1)}
                </td>
                <td
                  className={`py-1 text-right ${
                    f.oos_pct < 0 ? "text-rose-300" : "text-emerald-300"
                  }`}
                >
                  {f.oos_pct.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "emerald" | "rose";
}) {
  const tc =
    tone === "emerald"
      ? "text-emerald-300"
      : tone === "rose"
        ? "text-rose-300"
        : "text-slate-100";
  return (
    <div className="rounded border border-border bg-bg px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </div>
      <div className={`font-mono text-base ${tc}`}>{value}</div>
    </div>
  );
}
