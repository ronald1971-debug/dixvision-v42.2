interface SignalRow {
  pair: string;
  composite: number;
  momentum: number;
  on_chain_vol: number;
  social: number;
  holder_growth: number;
  rug_score: number;
  state: "tracking" | "armed" | "entered" | "exited";
}

const MOCK: SignalRow[] = [
  {
    pair: "WIFCAT / SOL",
    composite: 0.78,
    momentum: 0.84,
    on_chain_vol: 0.66,
    social: 0.72,
    holder_growth: 0.68,
    rug_score: 0.84,
    state: "armed",
  },
  {
    pair: "$JELLY / SOL",
    composite: 0.58,
    momentum: 0.61,
    on_chain_vol: 0.42,
    social: 0.58,
    holder_growth: 0.51,
    rug_score: 0.74,
    state: "tracking",
  },
  {
    pair: "BIGFI / WETH",
    composite: 0.39,
    momentum: 0.31,
    on_chain_vol: 0.22,
    social: 0.46,
    holder_growth: 0.33,
    rug_score: 0.55,
    state: "tracking",
  },
];

export function SignalTracker() {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Signal Tracker
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            DexScreener + Birdeye + Helius + X/TG + holders + rug-score
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">pair</th>
              <th className="px-2 py-1 text-right">composite</th>
              <th className="px-2 py-1 text-right">momo</th>
              <th className="px-2 py-1 text-right">on-chain</th>
              <th className="px-2 py-1 text-right">social</th>
              <th className="px-2 py-1 text-right">holders</th>
              <th className="px-2 py-1 text-right">rug</th>
              <th className="px-2 py-1 text-right">state</th>
            </tr>
          </thead>
          <tbody>
            {MOCK.map((r, i) => (
              <tr key={i} className="border-t border-border">
                <td className="px-2 py-0.5 text-slate-200">{r.pair}</td>
                <td className="px-2 py-0.5 text-right text-emerald-300">
                  {r.composite.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {r.momentum.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {r.on_chain_vol.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {r.social.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {r.holder_growth.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {r.rug_score.toFixed(2)}
                </td>
                <td className="px-2 py-0.5 text-right">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${
                      r.state === "armed"
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                        : r.state === "entered"
                          ? "border-accent/40 bg-accent/10 text-accent"
                          : "border-border bg-bg text-slate-500"
                    }`}
                  >
                    {r.state}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
