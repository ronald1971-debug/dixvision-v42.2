/**
 * Copy-trading leaderboard (PR-#2 spec §3.4.1).
 *
 * Up to 500 leader wallets across Solana + Ethereum mempools + Jito
 * shred-stream. Per-leader settings: min mirror size, max slippage,
 * daily cap, auto-exit on leader exit, PnL cutoff. Mirrors run
 * through our order engine, so autonomy mode + wallet policy +
 * kill-switch + dead-man + SL/TP all apply.
 */
interface Leader {
  wallet: string;
  pnl_30d_pct: number;
  win_rate_pct: number;
  trades_24h: number;
  copying: boolean;
  mirror_cap_usd: number;
  median_latency_ms: number;
}

const MOCK: Leader[] = [
  {
    wallet: "ANSEm…q4kE",
    pnl_30d_pct: 412.3,
    win_rate_pct: 71,
    trades_24h: 38,
    copying: true,
    mirror_cap_usd: 250,
    median_latency_ms: 380,
  },
  {
    wallet: "5kPnX…H3Lr",
    pnl_30d_pct: 268.1,
    win_rate_pct: 64,
    trades_24h: 22,
    copying: true,
    mirror_cap_usd: 150,
    median_latency_ms: 410,
  },
  {
    wallet: "9XfGH…Y7Bn",
    pnl_30d_pct: 144.7,
    win_rate_pct: 58,
    trades_24h: 14,
    copying: false,
    mirror_cap_usd: 0,
    median_latency_ms: 0,
  },
  {
    wallet: "0xDEAD…Beef",
    pnl_30d_pct: 86.2,
    win_rate_pct: 52,
    trades_24h: 9,
    copying: false,
    mirror_cap_usd: 0,
    median_latency_ms: 0,
  },
];

export function CopyLeaderboard() {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Copy Leaderboard
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            up to 500 wallets · sub-0.4 s mirror · governed by autonomy
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
              <th className="px-2 py-1 text-left">wallet</th>
              <th className="px-2 py-1 text-right">30d %</th>
              <th className="px-2 py-1 text-right">win %</th>
              <th className="px-2 py-1 text-right">trades 24h</th>
              <th className="px-2 py-1 text-right">cap</th>
              <th className="px-2 py-1 text-right">lat</th>
              <th className="px-2 py-1 text-right">mirror</th>
            </tr>
          </thead>
          <tbody>
            {MOCK.map((l) => (
              <tr key={l.wallet} className="border-t border-border">
                <td className="px-2 py-0.5 text-slate-200">{l.wallet}</td>
                <td
                  className={`px-2 py-0.5 text-right ${
                    l.pnl_30d_pct >= 0 ? "text-emerald-300" : "text-red-300"
                  }`}
                >
                  {l.pnl_30d_pct >= 0 ? "+" : ""}
                  {l.pnl_30d_pct.toFixed(1)}%
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {l.win_rate_pct}%
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {l.trades_24h}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {l.copying ? `$${l.mirror_cap_usd}` : "—"}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-400">
                  {l.copying ? `${l.median_latency_ms} ms` : "—"}
                </td>
                <td className="px-2 py-0.5 text-right">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${
                      l.copying
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                        : "border-border bg-bg text-slate-500"
                    }`}
                  >
                    {l.copying ? "ON" : "off"}
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
