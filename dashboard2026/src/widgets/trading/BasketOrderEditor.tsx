import { useState } from "react";

/**
 * F-track widget — Basket order editor.
 *
 * Edits a multi-symbol basket with target weights. Submitting the
 * basket stages N child intents (one per leg) into the
 * operator-approval queue (INV-72). Sum of weights is normalized
 * client-side so the operator sees deltas, not absolute drift.
 */
type Side = "BUY" | "SELL";

interface BasketLeg {
  id: string;
  symbol: string;
  weight: number;
  side: Side;
}

const SEED: BasketLeg[] = [
  { id: "1", symbol: "BTC-USDT", weight: 40, side: "BUY" },
  { id: "2", symbol: "ETH-USDT", weight: 30, side: "BUY" },
  { id: "3", symbol: "SOL-USDT", weight: 20, side: "BUY" },
  { id: "4", symbol: "AVAX-USDT", weight: 10, side: "BUY" },
];

let SEQ = SEED.length;

export function BasketOrderEditor() {
  const [legs, setLegs] = useState<BasketLeg[]>(SEED);
  const [notional, setNotional] = useState(250_000);
  const [staged, setStaged] = useState(false);

  const total = legs.reduce((acc, l) => acc + (Number.isFinite(l.weight) ? l.weight : 0), 0);

  const update = (id: string, patch: Partial<BasketLeg>) =>
    setLegs((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const remove = (id: string) =>
    setLegs((prev) => prev.filter((l) => l.id !== id));

  const add = () => {
    SEQ += 1;
    setLegs((prev) => [
      ...prev,
      { id: String(SEQ), symbol: "NEW-USDT", weight: 0, side: "BUY" },
    ]);
  };

  const normalized = legs.map((l) => ({
    ...l,
    pct: total > 0 ? (l.weight / total) * 100 : 0,
    notional: total > 0 ? (l.weight / total) * notional : 0,
  }));

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Basket order editor
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          multi-leg · target weights · stages one child intent per leg
        </p>
      </header>
      <div className="border-b border-border bg-bg/40 px-3 py-2">
        <label className="flex items-center gap-2 font-mono text-[11px] text-slate-300">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            basket notional USD
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
              <th className="px-3 py-1.5 text-left">side</th>
              <th className="px-3 py-1.5 text-right">weight</th>
              <th className="px-3 py-1.5 text-right">% basket</th>
              <th className="px-3 py-1.5 text-right">notional</th>
              <th className="px-3 py-1.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {normalized.map((l) => (
              <tr key={l.id}>
                <td className="px-3 py-1">
                  <input
                    value={l.symbol}
                    onChange={(e) =>
                      update(l.id, { symbol: e.target.value.toUpperCase() })
                    }
                    className="w-full rounded border border-border bg-bg/60 px-2 py-0.5 text-slate-200 focus:border-accent focus:outline-none"
                  />
                </td>
                <td className="px-3 py-1">
                  <select
                    value={l.side}
                    onChange={(e) =>
                      update(l.id, { side: e.target.value as Side })
                    }
                    className="rounded border border-border bg-bg/60 px-2 py-0.5 text-slate-200 focus:border-accent focus:outline-none"
                  >
                    <option value="BUY">BUY</option>
                    <option value="SELL">SELL</option>
                  </select>
                </td>
                <td className="px-3 py-1 text-right">
                  <input
                    type="number"
                    value={l.weight}
                    onChange={(e) =>
                      update(l.id, {
                        weight: Math.max(0, Number(e.target.value) || 0),
                      })
                    }
                    className="w-16 rounded border border-border bg-bg/60 px-2 py-0.5 text-right text-slate-200 focus:border-accent focus:outline-none"
                  />
                </td>
                <td className="px-3 py-1 text-right text-slate-400">
                  {l.pct.toFixed(1)}%
                </td>
                <td className="px-3 py-1 text-right">
                  {Math.round(l.notional).toLocaleString()}
                </td>
                <td className="px-3 py-1 text-right">
                  <button
                    type="button"
                    onClick={() => remove(l.id)}
                    className="rounded border border-border bg-bg/40 px-1.5 py-0.5 text-[10px] text-slate-500 hover:border-rose-500/40 hover:text-rose-400"
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <footer className="flex items-center gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        <span>
          {legs.length} legs · weights sum {total.toFixed(0)}
        </span>
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
          {staged ? "staged" : "stage basket"}
        </button>
      </footer>
    </section>
  );
}
