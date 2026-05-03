import { useState } from "react";

/**
 * I-track widget — Champion / Challenger A/B board.
 *
 * Side-by-side comparison between the live "champion" strategy and a
 * "challenger" running in SHADOW or CANARY. Metrics are pulled from
 * the strategy registry (PR #113) ledger replay; promotion is gated
 * by the hash-anchored gate from PR #124.
 *
 * Backend hook: ``GET /api/testing/champion-challenger?challenger=<id>``
 */
type Stat = {
  name: string;
  champion: number;
  challenger: number;
  unit: "pct" | "ratio" | "usd" | "count";
  better: "higher" | "lower";
};

const STATS: Stat[] = [
  { name: "Sharpe (90d)", champion: 1.82, challenger: 2.14, unit: "ratio", better: "higher" },
  { name: "Sortino (90d)", champion: 2.41, challenger: 2.76, unit: "ratio", better: "higher" },
  { name: "Calmar", champion: 1.05, challenger: 1.31, unit: "ratio", better: "higher" },
  { name: "Max DD", champion: -8.4, challenger: -6.2, unit: "pct", better: "higher" },
  { name: "Win rate", champion: 54.1, challenger: 57.8, unit: "pct", better: "higher" },
  { name: "PnL (90d)", champion: 142_300, challenger: 168_900, unit: "usd", better: "higher" },
  { name: "Avg slippage (bps)", champion: 4.2, challenger: 3.6, unit: "ratio", better: "lower" },
  { name: "Trades / day", champion: 38, challenger: 52, unit: "count", better: "higher" },
];

function fmt(v: number, unit: Stat["unit"]) {
  if (unit === "pct") return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
  if (unit === "usd") return `$${(v / 1000).toFixed(1)}k`;
  if (unit === "count") return v.toLocaleString();
  return v.toFixed(2);
}

function winner(s: Stat): "champion" | "challenger" | "tie" {
  if (s.champion === s.challenger) return "tie";
  if (s.better === "higher") return s.champion > s.challenger ? "champion" : "challenger";
  return s.champion < s.challenger ? "champion" : "challenger";
}

export function ChampionChallenger() {
  const [challenger, setChallenger] = useState("strat_meanrev_v3");
  const challengerWins = STATS.filter((s) => winner(s) === "challenger").length;
  const verdict = challengerWins >= 6 ? "promote" : challengerWins >= 4 ? "extend canary" : "reject";
  const verdictTone =
    verdict === "promote" ? "emerald-400" : verdict === "extend canary" ? "amber-400" : "rose-400";

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Champion / Challenger
        </h3>
        <div className="mt-1 flex items-center gap-2 text-[11px]">
          <label className="text-slate-500">challenger:</label>
          <select
            value={challenger}
            onChange={(e) => setChallenger(e.target.value)}
            className="rounded border border-border bg-bg/50 px-2 py-0.5 font-mono text-[11px] text-slate-200 outline-none"
          >
            <option value="strat_meanrev_v3">strat_meanrev_v3 (CANARY)</option>
            <option value="strat_breakout_v2">strat_breakout_v2 (SHADOW)</option>
            <option value="strat_funding_v1">strat_funding_v1 (SHADOW)</option>
          </select>
          <span className={`ml-auto rounded border border-${verdictTone}/40 bg-${verdictTone}/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-${verdictTone}`}>
            verdict: {verdict}
          </span>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">metric</th>
              <th className="px-3 py-1.5 text-right">champion</th>
              <th className="px-3 py-1.5 text-right">challenger</th>
              <th className="px-3 py-1.5 text-center">winner</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {STATS.map((s) => {
              const w = winner(s);
              return (
                <tr key={s.name}>
                  <td className="px-3 py-1 text-slate-200">{s.name}</td>
                  <td className={`px-3 py-1 text-right ${w === "champion" ? "text-emerald-400" : "text-slate-300"}`}>
                    {fmt(s.champion, s.unit)}
                  </td>
                  <td className={`px-3 py-1 text-right ${w === "challenger" ? "text-emerald-400" : "text-slate-300"}`}>
                    {fmt(s.challenger, s.unit)}
                  </td>
                  <td className="px-3 py-1 text-center">
                    {w === "tie" ? (
                      <span className="text-slate-500">—</span>
                    ) : (
                      <span className={`rounded px-1.5 py-0.5 text-[9px] uppercase ${w === "champion" ? "bg-slate-500/20 text-slate-300" : "bg-emerald-500/20 text-emerald-400"}`}>
                        {w}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <footer className="border-t border-border px-3 py-1.5 font-mono text-[10px] text-slate-500">
        challenger wins {challengerWins} / {STATS.length} · gate threshold ≥6
      </footer>
    </section>
  );
}
