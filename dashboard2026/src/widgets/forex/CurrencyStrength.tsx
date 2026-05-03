interface StrengthRow {
  ccy: string;
  score: number; // -100..+100
}

const MOCK: StrengthRow[] = [
  { ccy: "USD", score: 64 },
  { ccy: "GBP", score: 28 },
  { ccy: "CHF", score: 12 },
  { ccy: "CAD", score: 5 },
  { ccy: "EUR", score: -8 },
  { ccy: "AUD", score: -22 },
  { ccy: "NZD", score: -34 },
  { ccy: "JPY", score: -71 },
];

export function CurrencyStrength() {
  const rows = [...MOCK].sort((a, b) => b.score - a.score);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Currency Strength
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            8-currency relative strength · 4h rolling
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
          {rows[0].ccy} lead
        </span>
      </header>
      <div className="flex-1 overflow-auto px-3 py-2">
        <div className="space-y-1.5">
          {rows.map((r) => {
            const positive = r.score >= 0;
            const w = Math.min(Math.abs(r.score), 100);
            return (
              <div key={r.ccy} className="flex items-center gap-2 text-[11px]">
                <span className="w-10 font-mono text-slate-200">{r.ccy}</span>
                <div className="relative flex-1">
                  <div className="absolute left-1/2 top-0 h-3 w-px -translate-x-1/2 bg-slate-700/60" />
                  <div className="flex h-3 overflow-hidden rounded bg-slate-800/40">
                    <div className="flex-1 flex justify-end">
                      {!positive && (
                        <div
                          className="h-full bg-rose-500/60"
                          style={{ width: `${w}%` }}
                        />
                      )}
                    </div>
                    <div className="flex-1">
                      {positive && (
                        <div
                          className="h-full bg-emerald-500/60"
                          style={{ width: `${w}%` }}
                        />
                      )}
                    </div>
                  </div>
                </div>
                <span
                  className={`w-10 text-right font-mono ${
                    positive ? "text-emerald-300" : "text-rose-300"
                  }`}
                >
                  {positive ? "+" : ""}
                  {r.score}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
