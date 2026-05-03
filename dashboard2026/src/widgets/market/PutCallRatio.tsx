/**
 * H-track widget — Put/Call ratio (Deribit options).
 *
 * Backend hook: ``GET /api/market/put-call?symbol=BTC`` reads the
 * Deribit options open-interest snapshot and computes:
 *   - puts/calls volume ratio (24h)
 *   - puts/calls OI ratio (instantaneous)
 *   - 7-day trend
 */
const VOL = { puts: 4_120, calls: 5_340 };
const OI = { puts: 11_800, calls: 14_900 };
const HIST = [0.65, 0.71, 0.78, 0.74, 0.79, 0.81, 0.77];

interface Tone {
  textCls: string;
  bgCls: string;
}

function tone(r: number): Tone {
  if (r >= 0.9) return { textCls: "text-rose-400", bgCls: "bg-rose-400/50" };
  if (r >= 0.7) return { textCls: "text-amber-400", bgCls: "bg-amber-400/50" };
  return { textCls: "text-emerald-400", bgCls: "bg-emerald-400/50" };
}

export function PutCallRatio() {
  const volRatio = VOL.puts / VOL.calls;
  const oiRatio = OI.puts / OI.calls;
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Put/Call ratio
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          Deribit · BTC · 24h volume + spot OI
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-3 px-3 py-3">
        <div className="grid grid-cols-2 gap-2">
          <Card label="vol" ratio={volRatio} puts={VOL.puts} calls={VOL.calls} />
          <Card label="OI" ratio={oiRatio} puts={OI.puts} calls={OI.calls} />
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            7-day vol p/c
          </div>
          <div className="mt-1 grid grid-cols-7 gap-1">
            {HIST.map((v, i) => (
              <div key={i} className="flex flex-col items-center gap-0.5">
                <div
                  className={`h-2 w-full rounded ${tone(v).bgCls}`}
                  title={`${v.toFixed(2)}`}
                />
                <div className="font-mono text-[9px] text-slate-500">
                  {v.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function Card({
  label,
  ratio,
  puts,
  calls,
}: {
  label: string;
  ratio: number;
  puts: number;
  calls: number;
}) {
  const t = tone(ratio);
  return (
    <div className="rounded border border-border/60 bg-bg/30 p-2">
      <div className="font-mono text-[9px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`mt-1 font-mono text-xl ${t.textCls}`}>
        {ratio.toFixed(2)}
      </div>
      <div className="font-mono text-[10px] text-slate-400">
        P {puts.toLocaleString()} / C {calls.toLocaleString()}
      </div>
    </div>
  );
}
