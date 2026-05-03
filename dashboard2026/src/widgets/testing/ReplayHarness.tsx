import { useEffect, useMemo, useRef, useState } from "react";

import { Pause, Play, Rewind, SkipForward } from "lucide-react";

/**
 * Tier-8 testing widget — replay harness.
 *
 * Pick any window from the canonical audit ledger and re-run it
 * deterministically against one or more strategy variants. Same
 * intelligence + governance + execution path as live; only the
 * source of bars is the cold-tier ledger replay store. Outputs the
 * scrubbing playhead + a side-by-side equity comparison for variants
 * A and B.
 *
 * Live wiring source: `evaluation.replay_engine.run(start, end, [A,B])`
 * — currently mocked from a deterministic seed of (start_iso, end_iso,
 * strategy_a, strategy_b).
 */
type Strategy = "champion" | "challenger_v2" | "challenger_v3" | "memecoin_v1";

const STRATEGY_LABEL: Record<Strategy, string> = {
  champion: "Champion (LIVE)",
  challenger_v2: "Challenger v2",
  challenger_v3: "Challenger v3",
  memecoin_v1: "Memecoin v1",
};

function hashSeed(parts: ReadonlyArray<string | number>): number {
  let h = 2166136261;
  for (const p of parts) {
    const s = String(p);
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
  }
  return h >>> 0;
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function buildCurve(
  strategy: Strategy,
  startIso: string,
  endIso: string,
  n: number,
): number[] {
  const r = rng(hashSeed([strategy, startIso, endIso]));
  const drift =
    strategy === "champion"
      ? 0.04
      : strategy === "memecoin_v1"
        ? 0.18
        : 0.07;
  const vol = strategy === "memecoin_v1" ? 1.2 : 0.5;
  const out: number[] = [100];
  for (let i = 1; i < n; i += 1) {
    const step = (r() - 0.5) * vol + drift / 100;
    out.push(out[i - 1] * (1 + step / 100));
  }
  return out;
}

function curvePath(curve: number[], w: number, h: number): string {
  if (curve.length === 0) return "";
  const min = Math.min(...curve);
  const max = Math.max(...curve);
  const range = max - min || 1;
  return curve
    .map((v, i) => {
      const x = (i / Math.max(1, curve.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function ReplayHarness() {
  const [start, setStart] = useState("2024-11-01");
  const [end, setEnd] = useState("2024-12-01");
  const [a, setA] = useState<Strategy>("champion");
  const [b, setB] = useState<Strategy>("challenger_v2");
  const [playing, setPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const tickRef = useRef<number | null>(null);

  const N = 200;
  const curveA = useMemo(() => buildCurve(a, start, end, N), [a, start, end]);
  const curveB = useMemo(() => buildCurve(b, start, end, N), [b, start, end]);

  useEffect(() => {
    if (!playing) {
      if (tickRef.current !== null) {
        window.clearInterval(tickRef.current);
        tickRef.current = null;
      }
      return;
    }
    tickRef.current = window.setInterval(() => {
      setPlayhead((p) => (p + 1 >= N ? 0 : p + 1));
    }, 80);
    return () => {
      if (tickRef.current !== null) {
        window.clearInterval(tickRef.current);
        tickRef.current = null;
      }
    };
  }, [playing]);

  const W = 320;
  const H = 100;
  const playheadX = (playhead / Math.max(1, N - 1)) * W;
  const finalA = curveA[Math.min(playhead, curveA.length - 1)];
  const finalB = curveB[Math.min(playhead, curveB.length - 1)];

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Replay harness
        </h3>
        <span className="rounded border border-border bg-bg px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-slate-400">
          ledger replay
        </span>
      </header>

      <div className="grid grid-cols-2 gap-2 border-b border-border px-3 py-2 text-[11px] md:grid-cols-4">
        <Field label="Start">
          <input
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          />
        </Field>
        <Field label="End">
          <input
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          />
        </Field>
        <Field label="Variant A">
          <select
            value={a}
            onChange={(e) => setA(e.target.value as Strategy)}
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          >
            {(Object.keys(STRATEGY_LABEL) as Strategy[]).map((s) => (
              <option key={s} value={s}>
                {STRATEGY_LABEL[s]}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Variant B">
          <select
            value={b}
            onChange={(e) => setB(e.target.value as Strategy)}
            className="w-full rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
          >
            {(Object.keys(STRATEGY_LABEL) as Strategy[]).map((s) => (
              <option key={s} value={s}>
                {STRATEGY_LABEL[s]}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <button
          type="button"
          onClick={() => setPlayhead(0)}
          className="rounded border border-border bg-bg px-2 py-1 text-slate-300 hover:text-slate-100"
          aria-label="rewind"
        >
          <Rewind className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded border border-accent/40 bg-accent/15 px-2 py-1 text-accent hover:bg-accent/25"
          aria-label={playing ? "pause" : "play"}
        >
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </button>
        <button
          type="button"
          onClick={() => setPlayhead(N - 1)}
          className="rounded border border-border bg-bg px-2 py-1 text-slate-300 hover:text-slate-100"
          aria-label="end"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </button>
        <input
          type="range"
          min={0}
          max={N - 1}
          value={playhead}
          onChange={(e) => setPlayhead(Number(e.target.value))}
          className="flex-1 accent-emerald-500"
        />
        <span className="font-mono text-[10px] text-slate-400">
          {playhead}/{N - 1}
        </span>
      </div>

      <div className="flex-1 overflow-hidden p-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full">
          <path
            d={curvePath(curveA, W, H)}
            fill="none"
            stroke="rgb(45 212 191)"
            strokeWidth={1.4}
          />
          <path
            d={curvePath(curveB, W, H)}
            fill="none"
            stroke="rgb(244 114 182)"
            strokeWidth={1.4}
          />
          <line
            x1={playheadX}
            x2={playheadX}
            y1={0}
            y2={H}
            stroke="rgb(148 163 184)"
            strokeWidth={0.6}
            strokeDasharray="2,3"
          />
        </svg>

        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
          <div className="rounded border border-emerald-500/40 bg-emerald-500/5 p-2">
            <div className="text-[10px] uppercase tracking-widest text-emerald-300/80">
              {STRATEGY_LABEL[a]}
            </div>
            <div className="font-mono text-base text-emerald-300">
              {finalA.toFixed(2)}
            </div>
            <div className="font-mono text-[10px] text-slate-500">
              {((finalA - 100) >= 0 ? "+" : "")}
              {(finalA - 100).toFixed(2)}%
            </div>
          </div>
          <div className="rounded border border-pink-500/40 bg-pink-500/5 p-2">
            <div className="text-[10px] uppercase tracking-widest text-pink-300/80">
              {STRATEGY_LABEL[b]}
            </div>
            <div className="font-mono text-base text-pink-300">
              {finalB.toFixed(2)}
            </div>
            <div className="font-mono text-[10px] text-slate-500">
              {((finalB - 100) >= 0 ? "+" : "")}
              {(finalB - 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </span>
      {children}
    </label>
  );
}
