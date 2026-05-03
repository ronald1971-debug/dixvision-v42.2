import { useEffect, useState } from "react";

/**
 * Tier-3 AI widget — Smart-money tracker.
 *
 * Live leaderboard of top wallets by 30d PnL with their latest
 * trades. Mirrors the Nansen / Arkham smart-money panel: surface
 * who's buying / selling what, when, and how their copy-edge has
 * decayed. The "follow" toggle is approval-edge gated — it stages
 * a copy directive that the operator-approval edge (INV-72) must
 * accept before any execution.
 *
 * Data source delegated to on-chain adapters (Tier 5). Mock feed
 * here so the surface renders today.
 */
interface Wallet {
  addr: string;
  label: string;
  pnl_30d: number;
  win_rate: number;
  last_trade: { ts: number; symbol: string; side: "BUY" | "SELL"; size: number };
  follow: boolean;
}

const SEED: Wallet[] = [
  {
    addr: "0x7a1f...d4c2",
    label: "MarsWalker",
    pnl_30d: 142_300,
    win_rate: 0.71,
    last_trade: { ts: Date.now() - 60_000, symbol: "WIF", side: "BUY", size: 4_200_000 },
    follow: false,
  },
  {
    addr: "0x9b3e...8a17",
    label: "QuietWhale",
    pnl_30d: 88_700,
    win_rate: 0.64,
    last_trade: { ts: Date.now() - 240_000, symbol: "ETH", side: "SELL", size: 12 },
    follow: true,
  },
  {
    addr: "5fHt...kQ8",
    label: "SolSniper.sol",
    pnl_30d: 56_400,
    win_rate: 0.58,
    last_trade: { ts: Date.now() - 90_000, symbol: "BONK", side: "BUY", size: 80_000_000 },
    follow: false,
  },
  {
    addr: "0xdeaf...f00d",
    label: "AlphaDelta",
    pnl_30d: 42_100,
    win_rate: 0.62,
    last_trade: { ts: Date.now() - 180_000, symbol: "BTC", side: "BUY", size: 0.8 },
    follow: false,
  },
  {
    addr: "0xfa11...beef",
    label: "ContrarianHL",
    pnl_30d: -18_400,
    win_rate: 0.41,
    last_trade: { ts: Date.now() - 300_000, symbol: "DOGE", side: "SELL", size: 250_000 },
    follow: false,
  },
];

export function SmartMoneyTracker() {
  const [wallets, setWallets] = useState<Wallet[]>(SEED);

  useEffect(() => {
    // Drift mock pnls deterministically so the panel feels alive
    // without depending on the SSE bridge.
    const id = setInterval(() => {
      setWallets((prev) =>
        prev.map((w) => {
          const drift = (Math.sin(Date.now() / 5_000 + w.addr.length) - 0.5) * 60;
          return { ...w, pnl_30d: w.pnl_30d + drift };
        }),
      );
    }, 4_000);
    return () => clearInterval(id);
  }, []);

  const toggle = (addr: string) =>
    setWallets((prev) =>
      prev.map((w) => (w.addr === addr ? { ...w, follow: !w.follow } : w)),
    );

  const sorted = [...wallets].sort((a, b) => b.pnl_30d - a.pnl_30d);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Smart-money tracker
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          top wallets · 30d PnL · last trade · follow stages a copy directive
        </p>
      </header>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {sorted.map((w) => {
          const ago = Math.max(1, Math.floor((Date.now() - w.last_trade.ts) / 1000));
          return (
            <li
              key={w.addr}
              className="grid grid-cols-[1fr_auto] items-baseline gap-2 px-3 py-2 font-mono text-[11px] text-slate-300"
            >
              <div className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="truncate font-semibold text-slate-200">
                    {w.label}
                  </span>
                  <span className="truncate text-[10px] text-slate-500">
                    {w.addr}
                  </span>
                </div>
                <div className="mt-0.5 flex items-baseline gap-3 text-[10px] text-slate-500">
                  <span>
                    pnl 30d{" "}
                    <span
                      className={
                        w.pnl_30d >= 0 ? "text-emerald-400" : "text-rose-400"
                      }
                    >
                      {w.pnl_30d >= 0 ? "+" : ""}
                      {Math.round(w.pnl_30d).toLocaleString()}
                    </span>
                  </span>
                  <span>win {(w.win_rate * 100).toFixed(0)}%</span>
                  <span>
                    {w.last_trade.side} {w.last_trade.size.toLocaleString()}{" "}
                    {w.last_trade.symbol} · {ago}s ago
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => toggle(w.addr)}
                className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wider ${
                  w.follow
                    ? "border-accent/40 bg-accent/10 text-accent"
                    : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
                }`}
                aria-pressed={w.follow}
              >
                {w.follow ? "following" : "follow"}
              </button>
            </li>
          );
        })}
      </ul>
      <footer className="border-t border-border px-3 py-1 text-[10px] text-slate-500">
        copy directives are approval-edge gated · canonical data via Tier-5
        on-chain adapters
      </footer>
    </section>
  );
}
