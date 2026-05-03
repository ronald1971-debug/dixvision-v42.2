import { useMemo, useState } from "react";

/**
 * G-track widget — Risk-parity allocator.
 *
 * Inverse-volatility allocator: each leg's weight is 1/σ_i / Σ(1/σ_j).
 * Operator can edit symbols + σ; output reweights live. Stages a
 * BasketIntent through the operator-approval edge (INV-72).
 *
 * Backend hook: ``POST /api/portfolio/rebalance/risk_parity`` will
 * route to ``portfolio_engine.allocator.risk_parity`` once the
 * adapter lands. Today the math is computed deterministically
 * client-side.
 */
interface Asset {
  symbol: string;
  sigma_pct: number;
}

const DEFAULTS: Asset[] = [
  { symbol: "BTC-USDT", sigma_pct: 35 },
  { symbol: "ETH-USDT", sigma_pct: 48 },
  { symbol: "SOL-USDT", sigma_pct: 78 },
  { symbol: "AVAX-USDT", sigma_pct: 92 },
  { symbol: "WIF-USDT", sigma_pct: 220 },
];

let SEQ = DEFAULTS.length;

export function RiskParityAllocator() {
  const [assets, setAssets] = useState<Asset[]>(DEFAULTS);
  const [notional, setNotional] = useState(500_000);
  const [staged, setStaged] = useState(false);

  const rows = useMemo(() => {
    const inv = assets.map((a) => 1 / Math.max(a.sigma_pct, 0.01));
    const sum = inv.reduce((acc, v) => acc + v, 0);
    return assets.map((a, i) => {
      const w = inv[i] / sum;
      return {
        ...a,
        weight: w,
        notional: w * notional,
      };
    });
  }, [assets, notional]);

  const update = (i: number, patch: Partial<Asset>) =>
    setAssets((prev) =>
      prev.map((a, idx) => (idx === i ? { ...a, ...patch } : a)),
    );

  const remove = (i: number) =>
    setAssets((prev) => prev.filter((_, idx) => idx !== i));

  const add = () => {
    SEQ += 1;
    setAssets((prev) => [
      ...prev,
      { symbol: `NEW${SEQ}-USDT`, sigma_pct: 60 },
    ]);
  };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Risk-parity allocator
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          inverse-volatility weights · stages basket via approval edge
        </p>
      </header>
      <div className="border-b border-border bg-bg/40 px-3 py-2">
        <label className="flex items-center gap-2 font-mono text-[11px] text-slate-300">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            target notional USD
          </span>
          <input
            type="number"
            value={notional}
            onChange={(e) => setNotional(Math.max(0, Number(e.target.value) || 0))}
            className="w-32 rounded border border-border bg-bg/60 px-2 py-1 text-slate-200 focus:border-accent focus:outline-none"
          />
        </label>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-right">σ %</th>
              <th className="px-3 py-1.5 text-right">weight</th>
              <th className="px-3 py-1.5">distribution</th>
              <th className="px-3 py-1.5 text-right">notional</th>
              <th className="px-3 py-1.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r, i) => {
              const pct = r.weight * 100;
              return (
                <tr key={i}>
                  <td className="px-3 py-1">
                    <input
                      value={r.symbol}
                      onChange={(e) =>
                        update(i, { symbol: e.target.value.toUpperCase() })
                      }
                      className="w-full rounded border border-border bg-bg/60 px-2 py-0.5 text-slate-200 focus:border-accent focus:outline-none"
                    />
                  </td>
                  <td className="px-3 py-1 text-right">
                    <input
                      type="number"
                      value={r.sigma_pct}
                      onChange={(e) =>
                        update(i, {
                          sigma_pct: Math.max(0.01, Number(e.target.value) || 0.01),
                        })
                      }
                      className="w-16 rounded border border-border bg-bg/60 px-2 py-0.5 text-right text-slate-200 focus:border-accent focus:outline-none"
                    />
                  </td>
                  <td className="px-3 py-1 text-right text-slate-200">
                    {pct.toFixed(2)}%
                  </td>
                  <td className="px-3 py-1">
                    <div className="h-2 rounded bg-bg/40">
                      <div
                        className="h-full rounded bg-accent/60"
                        style={{ width: `${Math.min(pct * 2, 100)}%` }}
                      />
                    </div>
                  </td>
                  <td className="px-3 py-1 text-right">
                    {Math.round(r.notional).toLocaleString()}
                  </td>
                  <td className="px-3 py-1 text-right">
                    <button
                      type="button"
                      onClick={() => remove(i)}
                      className="rounded border border-border bg-bg/40 px-1.5 py-0.5 text-[10px] text-slate-500 hover:border-rose-500/40 hover:text-rose-400"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <footer className="flex items-center gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        <span>{rows.length} legs</span>
        <button
          type="button"
          onClick={add}
          className="rounded border border-border bg-bg/40 px-2 py-0.5 uppercase tracking-wider text-slate-400 hover:border-accent hover:text-accent"
        >
          + leg
        </button>
        <button
          type="button"
          onClick={() => setStaged((s) => !s)}
          className={`ml-auto rounded border px-2 py-0.5 uppercase tracking-wider ${
            staged
              ? "border-accent/40 bg-accent/10 text-accent"
              : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
          }`}
        >
          {staged ? "staged" : "stage rebalance"}
        </button>
      </footer>
    </section>
  );
}
