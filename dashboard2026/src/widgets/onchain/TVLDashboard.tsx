import { useEffect, useState } from "react";

/**
 * Tier-5 on-chain widget — TVL dashboard.
 *
 * Total Value Locked across the top-N protocols, with chain
 * affinity and 24h / 7d deltas. Real wiring through the
 * DefiLlama adapter (filed). Mock drifts deterministically.
 */
interface Protocol {
  name: string;
  category: "DEX" | "Lend" | "LST" | "Bridge" | "Perp";
  chain: string;
  tvl_usd: number;
  d24h_pct: number;
  d7d_pct: number;
}

const SEED: Protocol[] = [
  { name: "Lido",        category: "LST",   chain: "ETH", tvl_usd: 38_400_000_000, d24h_pct: 0.0021,  d7d_pct: 0.018 },
  { name: "EigenLayer",  category: "LST",   chain: "ETH", tvl_usd: 18_900_000_000, d24h_pct: 0.0048,  d7d_pct: 0.041 },
  { name: "Aave v3",     category: "Lend",  chain: "ETH", tvl_usd: 14_200_000_000, d24h_pct: -0.0011, d7d_pct: 0.012 },
  { name: "Maker",       category: "Lend",  chain: "ETH", tvl_usd: 9_300_000_000,  d24h_pct: 0.0008,  d7d_pct: -0.004 },
  { name: "Uniswap v3",  category: "DEX",   chain: "ETH", tvl_usd: 6_200_000_000,  d24h_pct: 0.0017,  d7d_pct: 0.022 },
  { name: "Hyperliquid", category: "Perp",  chain: "Arb", tvl_usd: 4_400_000_000,  d24h_pct: 0.0091,  d7d_pct: 0.073 },
  { name: "Pendle",      category: "Lend",  chain: "ETH", tvl_usd: 3_100_000_000,  d24h_pct: 0.0033,  d7d_pct: 0.028 },
  { name: "Jito",        category: "LST",   chain: "SOL", tvl_usd: 2_800_000_000,  d24h_pct: 0.0061,  d7d_pct: 0.034 },
  { name: "GMX",         category: "Perp",  chain: "Arb", tvl_usd: 1_800_000_000,  d24h_pct: -0.0024, d7d_pct: -0.012 },
  { name: "Across",      category: "Bridge",chain: "ETH", tvl_usd: 1_100_000_000,  d24h_pct: 0.0042,  d7d_pct: 0.024 },
];

const CAT_COLOR: Record<Protocol["category"], string> = {
  DEX: "border-sky-500/40 bg-sky-500/10 text-sky-300",
  Lend: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  LST: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  Bridge: "border-violet-500/40 bg-violet-500/10 text-violet-300",
  Perp: "border-rose-500/40 bg-rose-500/10 text-rose-300",
};

export function TVLDashboard() {
  const [rows, setRows] = useState<Protocol[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((p) => {
          const drift = Math.sin(Date.now() / 8_000 + p.name.length) * 0.0008;
          return {
            ...p,
            d24h_pct: p.d24h_pct + drift,
            tvl_usd: p.tvl_usd * (1 + drift),
          };
        }),
      );
    }, 5_000);
    return () => clearInterval(id);
  }, []);

  const total = rows.reduce((s, r) => s + r.tvl_usd, 0);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <div className="flex items-baseline justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            TVL · top protocols
          </h3>
          <span className="text-[10px] text-slate-500">
            Σ ${(total / 1_000_000_000).toFixed(0)}B
          </span>
        </div>
        <p className="mt-0.5 text-[11px] text-slate-500">
          locked value across DeFi · 24h drift · 7d trend
        </p>
      </header>
      <table className="w-full flex-1 overflow-auto text-[11px]">
        <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
          <tr className="text-left">
            <th className="px-3 py-1">protocol</th>
            <th className="px-2 py-1">cat</th>
            <th className="px-2 py-1">chain</th>
            <th className="px-2 py-1 text-right">tvl</th>
            <th className="px-2 py-1 text-right">24h</th>
            <th className="px-3 py-1 text-right">7d</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40 font-mono">
          {rows.map((p) => (
            <tr key={p.name}>
              <td className="px-3 py-1 text-slate-200">{p.name}</td>
              <td className="px-2 py-1">
                <span
                  className={`rounded border px-1 py-0.5 text-[9px] uppercase ${CAT_COLOR[p.category]}`}
                >
                  {p.category}
                </span>
              </td>
              <td className="px-2 py-1 text-slate-400">{p.chain}</td>
              <td className="px-2 py-1 text-right text-slate-300">
                ${(p.tvl_usd / 1_000_000_000).toFixed(2)}B
              </td>
              <td
                className={`px-2 py-1 text-right ${p.d24h_pct >= 0 ? "text-emerald-300" : "text-rose-300"}`}
              >
                {(p.d24h_pct * 100).toFixed(2)}%
              </td>
              <td
                className={`px-3 py-1 text-right ${p.d7d_pct >= 0 ? "text-emerald-300" : "text-rose-300"}`}
              >
                {(p.d7d_pct * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
