import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  fetchDecisionChains,
  type DecisionChain,
  type DecisionChainStep,
} from "@/api/governance";

/**
 * Tier-1 governance widget — Audit ledger / DecisionTrace browser.
 *
 * Reads `/api/dashboard/decisions` (DASH-1, PR #53) which projects
 * the per-decision DecisionTrace contract from the audit ledger
 * (PR #64). Each chain is a causal sequence: signal -> intent -> risk
 * -> exec -> result. Operator can drill into any chain to see the
 * full step-by-step payload.
 */
export function AuditLedgerViewer({ limit = 50 }: { limit?: number }) {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "decisions", limit],
    queryFn: ({ signal }) => fetchDecisionChains(limit, signal),
    refetchInterval: 6_000,
  });

  const [selected, setSelected] = useState<string | null>(null);
  const selectedChain =
    data?.find((c) => (c.trace_id ?? "") === selected) ?? null;

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Audit ledger
          </h2>
          <p className="text-[11px] text-slate-500">
            DecisionTrace browser (PR #53 / #64) — last {limit} chains
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="rounded border border-border bg-bg px-2 py-1 text-[11px] hover:border-accent disabled:opacity-50"
        >
          {isFetching ? "…" : "refresh"}
        </button>
      </header>
      <div className="flex flex-1 overflow-hidden text-xs">
        <ul className="w-1/2 overflow-auto border-r border-border">
          {isPending && <li className="px-3 py-2 text-slate-500">loading…</li>}
          {isError && (
            <li className="px-3 py-2 text-rose-400">
              failed: {(error as Error).message}
            </li>
          )}
          {data && data.length === 0 && (
            <li className="px-3 py-2 text-slate-500">no decision chains</li>
          )}
          {data?.map((c, i) => (
            <ChainRow
              key={c.trace_id ?? i}
              chain={c}
              active={selected === c.trace_id}
              onClick={() => setSelected(c.trace_id ?? null)}
            />
          ))}
        </ul>
        <div className="w-1/2 overflow-auto p-3">
          {selectedChain ? (
            <ChainDetail chain={selectedChain} />
          ) : (
            <p className="text-slate-500">select a chain to view trace</p>
          )}
        </div>
      </div>
    </section>
  );
}

function ChainRow({
  chain,
  active,
  onClick,
}: {
  chain: DecisionChain;
  active: boolean;
  onClick: () => void;
}) {
  const steps = chain.steps ?? [];
  const last = steps.length > 0 ? steps[steps.length - 1] : null;
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={`flex w-full items-baseline justify-between gap-2 px-3 py-1.5 text-left hover:bg-bg/40 ${
          active ? "bg-accent/10" : ""
        }`}
      >
        <span className="font-mono text-[10px] text-slate-400">
          {(chain.trace_id ?? "—").toString().slice(0, 16)}…
        </span>
        <span className="text-[10px] text-slate-500">{steps.length} step</span>
        <span className="text-[10px] text-slate-300">{last?.kind ?? ""}</span>
      </button>
    </li>
  );
}

function ChainDetail({ chain }: { chain: DecisionChain }) {
  const steps = chain.steps ?? [];
  return (
    <div className="space-y-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        trace
      </div>
      <div className="font-mono text-[11px] text-slate-300 break-all">
        {chain.trace_id}
      </div>
      <ol className="mt-2 flex flex-col gap-1.5">
        {steps.map((s, i) => (
          <StepRow key={i} step={s} index={i} />
        ))}
      </ol>
    </div>
  );
}

function StepRow({
  step,
  index,
}: {
  step: DecisionChainStep;
  index: number;
}) {
  return (
    <li className="rounded border border-border bg-bg/40 p-2">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[10px] text-slate-400">
          step {index}
        </span>
        <span className="rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-accent">
          {step.kind}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-slate-500">
        ts_ns {step.ts_ns}
      </div>
      <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-[10px] text-slate-300">
        {JSON.stringify(step.payload, null, 2)}
      </pre>
    </li>
  );
}
