import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Tier-3 AI widget — ASKB-style agentic orchestrator.
 *
 * Bloomberg ASKB-style coordinator: a single operator question is
 * dispatched to multiple sub-agents in parallel (data / news /
 * research / analytics / risk). Each agent surfaces a partial
 * answer with its source, confidence, and elapsed time. The
 * orchestrator synthesizes the partials into a final composite
 * once all agents respond — or marks the answer as "partial" if
 * one stalls.
 *
 * Real agent dispatch goes through the registry-driven chat
 * adapter (PR #82) and the Why Layer (PR #100). This widget shows
 * the parallelism and citation surface; canonical wiring is filed
 * for Wave-06 multi-agent.
 */
type AgentKey = "data" | "news" | "research" | "analytics" | "risk";

interface AgentState {
  key: AgentKey;
  label: string;
  status: "idle" | "running" | "done" | "stalled";
  elapsed_ms: number;
  partial?: string;
  source?: string;
  confidence?: number;
}

const INITIAL: Record<AgentKey, AgentState> = {
  data: { key: "data", label: "Data", status: "idle", elapsed_ms: 0 },
  news: { key: "news", label: "News", status: "idle", elapsed_ms: 0 },
  research: {
    key: "research",
    label: "Research",
    status: "idle",
    elapsed_ms: 0,
  },
  analytics: {
    key: "analytics",
    label: "Analytics",
    status: "idle",
    elapsed_ms: 0,
  },
  risk: { key: "risk", label: "Risk", status: "idle", elapsed_ms: 0 },
};

// Deterministic mock partials by question hash and agent key, so
// repeated submits feel real but stay reproducible.
function mockPartial(q: string, key: AgentKey): AgentState {
  const lc = q.toLowerCase();
  const hit = (kw: string) => lc.includes(kw);
  switch (key) {
    case "data":
      return {
        key,
        label: "Data",
        status: "done",
        elapsed_ms: 240,
        partial: hit("btc")
          ? "BTC last $67,420 · 24h vol $42B · 30d vol +18%"
          : "Top movers: NVDA +3.2%, TSLA -1.8%, AAPL +0.4%",
        source: "binance.spot · tradingview",
        confidence: 0.86,
      };
    case "news":
      return {
        key,
        label: "News",
        status: "done",
        elapsed_ms: 410,
        partial: hit("etf")
          ? "BlackRock IBIT inflows +$420M; weekly net +$1.1B."
          : "CoinDesk: macro print due Thursday; positioning skewed long.",
        source: "coindesk.rss · ap.wire",
        confidence: 0.74,
      };
    case "research":
      return {
        key,
        label: "Research",
        status: "done",
        elapsed_ms: 720,
        partial: hit("perps")
          ? "HL funding flipped negative on majors; basis -3bps annualized."
          : "Smart-money 7d net: BUY · top wallets long majors.",
        source: "askb.research-mock",
        confidence: 0.66,
      };
    case "analytics":
      return {
        key,
        label: "Analytics",
        status: "done",
        elapsed_ms: 320,
        partial: "BeliefState confidence 0.71 · PressureVector momentum +0.34",
        source: "intelligence_engine",
        confidence: 0.82,
      };
    case "risk":
      return {
        key,
        label: "Risk",
        status: "done",
        elapsed_ms: 180,
        partial: "VaR 1d 0.9% · DD floor 4.2% · CANARY size cap 1% notional",
        source: "padlock_floors",
        confidence: 0.91,
      };
  }
}

function synthesize(agents: AgentState[]): string {
  const completed = agents.filter((a) => a.status === "done");
  if (completed.length === 0) return "";
  const avgConf =
    completed.reduce((s, a) => s + (a.confidence ?? 0), 0) / completed.length;
  const allDone = completed.length === agents.length;
  const tag = allDone ? "synthesis" : "partial synthesis";
  const stitched = completed.map((a) => `[${a.label}] ${a.partial}`).join(" · ");
  return `${tag} (avg conf ${avgConf.toFixed(2)}) — ${stitched}`;
}

export function ASKBOrchestrator() {
  const [q, setQ] = useState("");
  const [agents, setAgents] = useState<AgentState[]>(Object.values(INITIAL));
  const [composite, setComposite] = useState("");
  const timers = useRef<number[]>([]);

  // Cleanup any pending dispatch timers on unmount.
  useEffect(() => {
    return () => {
      timers.current.forEach((t) => window.clearTimeout(t));
      timers.current = [];
    };
  }, []);

  const summary = useMemo(() => synthesize(agents), [agents]);
  useEffect(() => setComposite(summary), [summary]);

  const ask = () => {
    if (!q.trim()) return;
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
    setAgents(
      Object.values(INITIAL).map((a) => ({
        ...a,
        status: "running" as const,
      })),
    );
    const keys: AgentKey[] = ["data", "news", "research", "analytics", "risk"];
    keys.forEach((k, idx) => {
      const t = window.setTimeout(
        () => {
          setAgents((prev) =>
            prev.map((a) => (a.key === k ? mockPartial(q, k) : a)),
          );
        },
        180 + idx * 140,
      );
      timers.current.push(t);
    });
  };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          ASKB orchestrator
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          parallel agents · cited partials · synthesized answer
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-2 overflow-auto p-3 text-[12px]">
        <div className="flex gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") ask();
            }}
            placeholder="ask: BTC ETF flows · funding regime · macro print risk"
            className="flex-1 rounded border border-border bg-bg/40 px-2 py-1 font-mono text-[11px] text-slate-200 focus:border-accent focus:outline-none"
          />
          <button
            type="button"
            onClick={ask}
            className="rounded border border-accent/40 bg-accent/10 px-2 py-1 text-[11px] uppercase tracking-wider text-accent hover:bg-accent/20"
          >
            Dispatch
          </button>
        </div>
        <ul className="grid grid-cols-1 gap-1 sm:grid-cols-2 xl:grid-cols-5">
          {agents.map((a) => (
            <li
              key={a.key}
              className="rounded border border-border bg-bg/40 p-2 text-[11px]"
            >
              <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-wider">
                <span className="text-slate-300">{a.label}</span>
                <span
                  className={
                    a.status === "running"
                      ? "text-amber-300"
                      : a.status === "done"
                        ? "text-emerald-300"
                        : a.status === "stalled"
                          ? "text-rose-300"
                          : "text-slate-500"
                  }
                >
                  {a.status}
                </span>
              </div>
              {a.status === "done" ? (
                <>
                  <p className="mt-1 text-slate-300">{a.partial}</p>
                  <div className="mt-1 flex items-baseline justify-between text-[10px] text-slate-500">
                    <span>{a.source}</span>
                    <span>conf {a.confidence?.toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {a.elapsed_ms} ms
                  </div>
                </>
              ) : (
                <p className="mt-1 text-[10px] text-slate-500">
                  {a.status === "running" ? "dispatched · awaiting" : "idle"}
                </p>
              )}
            </li>
          ))}
        </ul>
        <div className="rounded border border-border bg-bg/40 p-2 text-[11px] text-slate-300">
          <h4 className="mb-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
            composite
          </h4>
          {composite ? (
            <p>{composite}</p>
          ) : (
            <p className="text-slate-500">no answer yet · ask a question</p>
          )}
        </div>
      </div>
    </section>
  );
}
