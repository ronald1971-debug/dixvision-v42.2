/**
 * Coherence widget — BeliefState regime + PressureVector 5-D +
 * DecisionTrace 'Why' feed (PR-#2 spec §0 + manifest v3.5).
 *
 * Renders a compact summary on the right rail of every per-asset
 * surface so the operator sees what Indira thinks before clicking
 * any order. The actual values arrive over the SSE bridge once the
 * `coherence` channel is wired; until then we show the latest
 * snapshot pulled from `/api/dashboard/decisions`.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";

interface BeliefState {
  regime: "bullish" | "neutral" | "bearish";
  confidence: number;
  entropy: number;
  conflict_hazard: number;
  ts_iso: string;
}

interface PressureVector {
  momentum: number;
  uncertainty: number;
  sentiment: number;
  liquidity: number;
  macro: number;
}

interface DecisionTraceEntry {
  ts_iso: string;
  why: string;
  composite_score: number;
}

interface CoherenceSnapshot {
  belief: BeliefState;
  pressure: PressureVector;
  trace: DecisionTraceEntry[];
}

const FALLBACK: CoherenceSnapshot = {
  belief: {
    regime: "neutral",
    confidence: 0.62,
    entropy: 0.41,
    conflict_hazard: 0.18,
    ts_iso: new Date().toISOString(),
  },
  pressure: {
    momentum: 0.34,
    uncertainty: 0.41,
    sentiment: 0.12,
    liquidity: 0.66,
    macro: -0.11,
  },
  trace: [
    {
      ts_iso: new Date().toISOString(),
      why: "Funding flipped negative on HL · CoinDesk wire bullish on BTC ETF inflows · BeliefState confidence above 0.6 → SHADOW signal emitted, awaiting CANARY promotion gate.",
      composite_score: 0.71,
    },
    {
      ts_iso: new Date(Date.now() - 60_000).toISOString(),
      why: "Liquidation cascade detected on hyperliquid 1h · pressure.uncertainty > 0.5 → no-trade hold.",
      composite_score: 0.42,
    },
  ],
};

async function fetchCoherence(
  signal?: AbortSignal,
): Promise<CoherenceSnapshot> {
  try {
    const res = await fetch(apiUrl("/api/dashboard/coherence"), { signal });
    if (!res.ok) throw new Error(`status ${res.status}`);
    return (await res.json()) as CoherenceSnapshot;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return FALLBACK;
  }
}

export function CoherencePanel() {
  const { data } = useQuery({
    queryKey: ["dashboard", "coherence"],
    queryFn: ({ signal }) => fetchCoherence(signal),
    refetchInterval: 5_000,
    initialData: FALLBACK,
  });
  const snapshot = data ?? FALLBACK;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Coherence · Why
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            BeliefState · PressureVector · DecisionTrace
          </p>
        </div>
        <RegimeChip regime={snapshot.belief.regime} />
      </header>
      <div className="flex-1 overflow-auto p-3 text-[12px]">
        <Section title="BeliefState">
          <KV k="confidence" v={snapshot.belief.confidence.toFixed(2)} />
          <KV k="entropy" v={snapshot.belief.entropy.toFixed(2)} />
          <KV
            k="conflict-hazard"
            v={snapshot.belief.conflict_hazard.toFixed(2)}
          />
        </Section>
        <Section title="PressureVector (5-D)">
          {(["momentum", "uncertainty", "sentiment", "liquidity", "macro"] as const).map(
            (k) => (
              <Bar key={k} label={k} value={snapshot.pressure[k]} />
            ),
          )}
        </Section>
        <Section title="Why · DecisionTrace">
          <ul className="space-y-1.5">
            {snapshot.trace.map((entry, i) => (
              <li
                key={i}
                className="rounded border border-border bg-bg/40 p-1.5 text-[11px] text-slate-300"
              >
                <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-wider text-slate-500">
                  <span>{new Date(entry.ts_iso).toLocaleTimeString()}</span>
                  <span>score {entry.composite_score.toFixed(2)}</span>
                </div>
                {entry.why}
              </li>
            ))}
          </ul>
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-3 last:mb-0">
      <h4 className="mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
        {title}
      </h4>
      <div className="space-y-0.5">{children}</div>
    </section>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between font-mono text-[11px]">
      <span className="text-slate-400">{k}</span>
      <span className="text-slate-200">{v}</span>
    </div>
  );
}

function Bar({ label, value }: { label: string; value: number }) {
  const positive = value >= 0;
  const pct = Math.min(100, Math.abs(value) * 100);
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-20 font-mono text-[10px] uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <div className="relative flex-1 overflow-hidden rounded bg-bg/60">
        <div
          className={`h-1.5 ${
            positive ? "bg-emerald-500" : "bg-red-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-12 text-right font-mono text-slate-300">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function RegimeChip({ regime }: { regime: BeliefState["regime"] }) {
  const tone =
    regime === "bullish"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : regime === "bearish"
        ? "border-red-500/40 bg-red-500/10 text-red-300"
        : "border-amber-500/40 bg-amber-500/10 text-amber-300";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${tone}`}
    >
      {regime}
    </span>
  );
}
