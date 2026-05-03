/**
 * H-track widget — Sentiment gauge.
 *
 * Composite of 4 streams (news polarity / social / funding skew / spot premium).
 * Backend hook: ``GET /api/market/sentiment`` reads the NewsFanout
 * polarity emitter (PR #120) blended with on-chain funding from PR #131.
 */
const COMPONENTS = [
  { name: "News polarity", value: 0.62, weight: 0.30 },
  { name: "Social", value: 0.71, weight: 0.20 },
  { name: "Funding skew", value: 0.55, weight: 0.30 },
  { name: "Spot premium", value: 0.48, weight: 0.20 },
];

export function SentimentGauge() {
  const composite = COMPONENTS.reduce((s, c) => s + c.value * c.weight, 0);
  const angle = (composite - 0.5) * 180; // -90..+90
  const t =
    composite >= 0.65
      ? { name: "very bullish", textCls: "text-emerald-400" }
      : composite >= 0.55
        ? { name: "bullish", textCls: "text-lime-400" }
        : composite >= 0.45
          ? { name: "neutral", textCls: "text-amber-400" }
          : composite >= 0.35
            ? { name: "bearish", textCls: "text-orange-400" }
            : { name: "very bearish", textCls: "text-rose-400" };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Sentiment composite
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          news · social · funding · premium
        </p>
      </header>
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-3 py-3">
        <div className="relative h-24 w-44">
          <svg viewBox="0 0 200 110" className="absolute inset-0">
            <defs>
              <linearGradient id="sent-arc" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f43f5e" />
                <stop offset="50%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
            <path
              d="M 10 100 A 90 90 0 0 1 190 100"
              stroke="url(#sent-arc)"
              strokeWidth={14}
              fill="none"
              strokeLinecap="round"
            />
            <g transform={`translate(100 100) rotate(${angle})`}>
              <line
                x1={0}
                y1={0}
                x2={0}
                y2={-78}
                stroke="#e2e8f0"
                strokeWidth={2}
              />
              <circle cx={0} cy={0} r={4} fill="#e2e8f0" />
            </g>
          </svg>
        </div>
        <div className="text-center">
          <div
            className={`font-mono text-2xl ${t.textCls}`}
            data-testid="sentiment-score"
          >
            {(composite * 100).toFixed(0)}
          </div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            {t.name}
          </div>
        </div>
        <table className="w-full font-mono text-[10px] text-slate-400">
          <tbody className="divide-y divide-border/40">
            {COMPONENTS.map((c) => (
              <tr key={c.name}>
                <td className="py-0.5">{c.name}</td>
                <td className="py-0.5 text-right text-slate-300">
                  {(c.value * 100).toFixed(0)}
                </td>
                <td className="py-0.5 text-right text-slate-500">
                  ×{c.weight.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
