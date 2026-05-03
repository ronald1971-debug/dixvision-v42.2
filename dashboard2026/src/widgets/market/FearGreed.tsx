/**
 * H-track widget — Crypto Fear & Greed (alternative.me-style index).
 *
 * Backend hook: ``GET /api/market/fear-greed`` proxies the alternative.me
 * crypto F&G feed; SCVS-registered as SRC-MACRO-FEARGREED-001 (PR #56).
 */
const TODAY = 64;
const HIST = [42, 48, 51, 55, 58, 61, 64];
const LABELS = ["7d", "6d", "5d", "4d", "3d", "2d", "now"];

function tone(v: number) {
  if (v >= 75) return { name: "extreme greed", cls: "emerald-400" };
  if (v >= 55) return { name: "greed", cls: "lime-400" };
  if (v >= 45) return { name: "neutral", cls: "amber-400" };
  if (v >= 25) return { name: "fear", cls: "orange-400" };
  return { name: "extreme fear", cls: "rose-400" };
}

export function FearGreed() {
  const t = tone(TODAY);
  const angle = (TODAY / 100) * 180 - 90;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Fear &amp; Greed
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          alternative.me · crypto
        </p>
      </header>
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-3 py-3">
        <div className="relative h-24 w-40">
          <svg viewBox="0 0 200 110" className="absolute inset-0">
            <defs>
              <linearGradient id="fg-arc" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f43f5e" />
                <stop offset="35%" stopColor="#f59e0b" />
                <stop offset="65%" stopColor="#a3e635" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
            <path
              d="M 10 100 A 90 90 0 0 1 190 100"
              stroke="url(#fg-arc)"
              strokeWidth={14}
              fill="none"
              strokeLinecap="round"
            />
            <g transform={`translate(100 100) rotate(${angle})`}>
              <line x1={0} y1={0} x2={0} y2={-78} stroke="#e2e8f0" strokeWidth={2} />
              <circle cx={0} cy={0} r={4} fill="#e2e8f0" />
            </g>
          </svg>
        </div>
        <div className="text-center">
          <div className={`font-mono text-2xl text-${t.cls}`}>{TODAY}</div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            {t.name}
          </div>
        </div>
        <div className="w-full">
          <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            7-day trend
          </div>
          <div className="mt-1 grid grid-cols-7 gap-1">
            {HIST.map((v, i) => {
              const cls = tone(v).cls;
              return (
                <div
                  key={LABELS[i]}
                  className="flex flex-col items-center gap-0.5"
                >
                  <div
                    className={`h-2 w-full rounded bg-${cls}/40`}
                    title={`${LABELS[i]}: ${v}`}
                  />
                  <div className="font-mono text-[9px] text-slate-500">
                    {v}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
