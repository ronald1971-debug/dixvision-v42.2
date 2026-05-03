import { useEffect, useState } from "react";

/**
 * Tier-4 memecoin widget — Dev-dump watchdog.
 *
 * Tracks the deployer wallet's outflow vs initial allocation. A dev
 * who starts moving the supply they minted toward an exchange or
 * splitting it across fresh wallets is the single biggest rug
 * indicator in the memecoin alpha space. The widget surfaces:
 *
 *   - dev_share        : % of supply still in deployer wallet
 *   - last_outflow_s   : seconds since last outflow event
 *   - lp_lock_pct      : % of LP that is verifiably locked
 *   - dump_score       : composite 0..1 (higher = more dangerous)
 *
 * `dump_score >= 0.7` triggers `HAZ-DEV-DUMP` (filed); operator sees
 * a 🛑 chip and SLTP engine auto-applies the rug-trip stop loss
 * (per PR-#2 spec §4 DEX/memecoin SL rules).
 */
interface Watch {
  ticker: string;
  dev_share: number;
  last_outflow_s: number;
  lp_lock_pct: number;
  dump_score: number;
  trip: boolean;
}

const SEED: Watch[] = [
  {
    ticker: "WIFCAT",
    dev_share: 0.04,
    last_outflow_s: 480,
    lp_lock_pct: 0.92,
    dump_score: 0.18,
    trip: false,
  },
  {
    ticker: "BONKER",
    dev_share: 0.21,
    last_outflow_s: 18,
    lp_lock_pct: 0.0,
    dump_score: 0.78,
    trip: true,
  },
  {
    ticker: "TURBOX",
    dev_share: 0.09,
    last_outflow_s: 240,
    lp_lock_pct: 0.55,
    dump_score: 0.41,
    trip: false,
  },
  {
    ticker: "GIGAFROG",
    dev_share: 0.16,
    last_outflow_s: 75,
    lp_lock_pct: 0.71,
    dump_score: 0.52,
    trip: false,
  },
  {
    ticker: "MOONOPUS",
    dev_share: 0.31,
    last_outflow_s: 6,
    lp_lock_pct: 0.0,
    dump_score: 0.91,
    trip: true,
  },
];

export function DevDumpWatchdog() {
  const [rows, setRows] = useState<Watch[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((r) => {
          const tripped = r.dump_score >= 0.7;
          const next_dump = Math.max(
            0,
            Math.min(1, r.dump_score + (tripped ? 0.01 : -0.005)),
          );
          return {
            ...r,
            last_outflow_s: Math.max(
              0,
              r.last_outflow_s + 5 - (tripped ? 10 : 0),
            ),
            dump_score: next_dump,
            trip: next_dump >= 0.7,
          };
        }),
      );
    }, 3_000);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Dev-dump watchdog
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          deployer outflow · LP lock · composite rug score
        </p>
      </header>
      <div className="flex-1 overflow-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
          <tr className="text-left">
            <th className="px-3 py-1">ticker</th>
            <th className="px-2 py-1 text-right">dev %</th>
            <th className="px-2 py-1 text-right">lp lock</th>
            <th className="px-2 py-1 text-right">last out</th>
            <th className="px-3 py-1 text-right">score</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40 font-mono">
          {rows.map((r) => {
            const tripped = r.dump_score >= 0.7;
            return (
            <tr key={r.ticker}>
              <td className="px-3 py-1 text-slate-200">
                {r.ticker}
                {tripped && (
                  <span className="ml-2 rounded bg-rose-500/20 px-1 py-0.5 text-[9px] uppercase text-rose-300">
                    HAZ-DEV-DUMP
                  </span>
                )}
              </td>
              <td className="px-2 py-1 text-right text-slate-300">
                {(r.dev_share * 100).toFixed(0)}%
              </td>
              <td className="px-2 py-1 text-right">
                <span
                  className={
                    r.lp_lock_pct > 0.5 ? "text-emerald-300" : "text-rose-300"
                  }
                >
                  {(r.lp_lock_pct * 100).toFixed(0)}%
                </span>
              </td>
              <td className="px-2 py-1 text-right text-slate-400">
                {r.last_outflow_s}s
              </td>
              <td className="px-3 py-1 text-right">
                <ScorePill score={r.dump_score} />
              </td>
            </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <footer className="border-t border-border px-3 py-1 text-[10px] text-slate-500">
        score ≥ 0.7 trips rug-stop SL via SLTP engine
      </footer>
    </section>
  );
}

function ScorePill({ score }: { score: number }) {
  const tone =
    score >= 0.7
      ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
      : score >= 0.4
        ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
        : "border-emerald-500/40 bg-emerald-500/10 text-emerald-300";
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[10px] ${tone}`}>
      {score.toFixed(2)}
    </span>
  );
}
