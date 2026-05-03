/**
 * I-track widget — INV-53 calibration / reliability plot.
 *
 * For each predicted-probability bucket the model emits a confidence in,
 * we measure the *actual* hit rate. A perfectly calibrated model lies on
 * the y=x diagonal; over-confidence sags below, under-confidence rises
 * above.
 *
 * Backend hook: ``GET /api/testing/calibration?strategy=<id>&window=90d``
 * reads the audit ledger (PR #64 DecisionTrace) and bins observed
 * outcomes by predicted probability.
 */
interface Bin {
  predicted: number; // bucket center, 0..1
  actual: number; // observed hit rate, 0..1
  count: number;
}

const BINS: Bin[] = [
  { predicted: 0.05, actual: 0.04, count: 1240 },
  { predicted: 0.15, actual: 0.18, count: 980 },
  { predicted: 0.25, actual: 0.27, count: 870 },
  { predicted: 0.35, actual: 0.34, count: 720 },
  { predicted: 0.45, actual: 0.41, count: 540 },
  { predicted: 0.55, actual: 0.51, count: 470 },
  { predicted: 0.65, actual: 0.58, count: 380 },
  { predicted: 0.75, actual: 0.69, count: 290 },
  { predicted: 0.85, actual: 0.74, count: 180 },
  { predicted: 0.95, actual: 0.81, count: 110 },
];

function brierScore(bins: Bin[]): number {
  let total = 0;
  let n = 0;
  for (const b of bins) {
    total += b.count * Math.pow(b.predicted - b.actual, 2);
    n += b.count;
  }
  return total / n;
}

function eceScore(bins: Bin[]): number {
  let total = 0;
  let n = 0;
  for (const b of bins) {
    total += b.count * Math.abs(b.predicted - b.actual);
    n += b.count;
  }
  return total / n;
}

const W = 320;
const H = 220;
const PAD = 32;

export function CalibrationReliability() {
  const brier = brierScore(BINS);
  const ece = eceScore(BINS);
  const totalCount = BINS.reduce((s, b) => s + b.count, 0);

  const x = (p: number) => PAD + p * (W - PAD * 2);
  const y = (p: number) => H - PAD - p * (H - PAD * 2);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2 flex items-baseline justify-between">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Calibration / Reliability
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            INV-53 · 90d window · {totalCount.toLocaleString()} predictions
          </p>
        </div>
        <div className="font-mono text-[10px] uppercase tracking-wider">
          <span className="text-slate-500">Brier</span>{" "}
          <span className="text-slate-200">{brier.toFixed(4)}</span>{" "}
          <span className="ml-2 text-slate-500">ECE</span>{" "}
          <span
            className={ece <= 0.05 ? "text-emerald-400" : ece <= 0.1 ? "text-amber-400" : "text-rose-400"}
          >
            {ece.toFixed(3)}
          </span>
        </div>
      </header>
      <div className="flex flex-1 items-center justify-center px-3 py-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md">
          <rect x={PAD} y={PAD} width={W - PAD * 2} height={H - PAD * 2} fill="none" stroke="#334155" strokeWidth={0.6} />
          {[0.25, 0.5, 0.75].map((g) => (
            <g key={g}>
              <line x1={x(g)} y1={PAD} x2={x(g)} y2={H - PAD} stroke="#1e293b" strokeWidth={0.4} />
              <line x1={PAD} y1={y(g)} x2={W - PAD} y2={y(g)} stroke="#1e293b" strokeWidth={0.4} />
            </g>
          ))}
          <line
            x1={x(0)}
            y1={y(0)}
            x2={x(1)}
            y2={y(1)}
            stroke="#475569"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth={1.4}
            points={BINS.map((b) => `${x(b.predicted)},${y(b.actual)}`).join(" ")}
          />
          {BINS.map((b) => {
            const r = 2 + Math.sqrt(b.count / 100);
            return (
              <circle
                key={b.predicted}
                cx={x(b.predicted)}
                cy={y(b.actual)}
                r={r}
                fill={
                  Math.abs(b.predicted - b.actual) <= 0.05
                    ? "#10b981"
                    : Math.abs(b.predicted - b.actual) <= 0.1
                      ? "#f59e0b"
                      : "#f43f5e"
                }
                opacity={0.85}
              >
                <title>
                  pred {b.predicted.toFixed(2)} → actual {b.actual.toFixed(2)} (n={b.count})
                </title>
              </circle>
            );
          })}
          {[0, 0.5, 1].map((g) => (
            <g key={g}>
              <text x={x(g)} y={H - PAD + 12} textAnchor="middle" fontSize={9} fill="#64748b">
                {g.toFixed(1)}
              </text>
              <text x={PAD - 6} y={y(g) + 3} textAnchor="end" fontSize={9} fill="#64748b">
                {g.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">
            predicted
          </text>
          <text x={10} y={H / 2} textAnchor="middle" fontSize={9} fill="#94a3b8" transform={`rotate(-90 10 ${H / 2})`}>
            actual
          </text>
        </svg>
      </div>
      <footer className="border-t border-border px-3 py-1.5 font-mono text-[10px] text-slate-500">
        diagonal = perfect calibration · sag below = overconfident · point size ∝ bin count
      </footer>
    </section>
  );
}
