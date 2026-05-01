/**
 * Rug Score widget (PR-#2 spec §3.4 + §4 trip rules).
 *
 * Composite score 0–100 derived from:
 *   - mint authority revoked
 *   - freeze authority revoked
 *   - LP locked / burned
 *   - top-10 holder concentration
 *   - dev-wallet share
 *   - sell-tax / honeypot simulation
 *   - bundle / dev-dump watchdog flags
 *
 * Trip rules feed the SL/TP engine: rug-trip SL fires immediately
 * when score crosses the operator-set threshold (default 60 → 50
 * over 5 minutes triggers exit).
 */
interface RugFactor {
  key: string;
  label: string;
  score: number; // 0–10
  weight: number;
}

const FACTORS: readonly RugFactor[] = [
  { key: "mint", label: "Mint authority revoked", score: 10, weight: 1.5 },
  { key: "freeze", label: "Freeze authority revoked", score: 10, weight: 1.2 },
  { key: "lp", label: "LP locked / burned", score: 9, weight: 1.5 },
  { key: "top10", label: "Top-10 holders < 25%", score: 7, weight: 1.0 },
  { key: "dev", label: "Dev wallet < 5%", score: 8, weight: 1.0 },
  { key: "honeypot", label: "Honeypot sim · sell ok", score: 10, weight: 1.5 },
  { key: "tax", label: "Buy/sell tax under 5%", score: 8, weight: 1.0 },
  { key: "bundle", label: "Bundle ratio under 8%", score: 6, weight: 0.8 },
];

export function RugScore() {
  const composite = computeScore(FACTORS);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Rug Score
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            composite · trips into rug-SL via §4 engine
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${
            composite >= 75
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              : composite >= 50
                ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                : "border-red-500/40 bg-red-500/10 text-red-300"
          }`}
        >
          {composite.toFixed(0)} / 100
        </span>
      </header>
      <div className="flex-1 space-y-1.5 overflow-auto p-3 text-[12px]">
        {FACTORS.map((f) => {
          const pct = (f.score / 10) * 100;
          return (
            <div key={f.key} className="flex items-center gap-2 font-mono">
              <span className="w-44 text-[11px] text-slate-300">{f.label}</span>
              <div className="relative flex-1 overflow-hidden rounded bg-bg/60">
                <div
                  className={`h-1.5 ${
                    f.score >= 7
                      ? "bg-emerald-500"
                      : f.score >= 4
                        ? "bg-amber-500"
                        : "bg-red-500"
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-10 text-right text-[11px] text-slate-400">
                {f.score}/10
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function computeScore(factors: readonly RugFactor[]): number {
  const weight = factors.reduce((acc, f) => acc + f.weight, 0);
  const value = factors.reduce((acc, f) => acc + f.score * f.weight, 0);
  return weight === 0 ? 0 : (value / weight) * 10;
}
