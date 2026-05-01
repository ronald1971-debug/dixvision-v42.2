import { ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { fetchMode } from "@/api/dashboard";

/**
 * Promote-chain widget (PR-#2 spec §5.7).
 *
 * Renders the full promote chain
 *   `[Backtest] → [Paper] → [Shadow] → [Canary] → [Live] → [Auto]`
 * with the chip matching the current System Mode highlighted.
 *
 * The "promote" buttons here are display-only: actual promotion runs
 * the sandbox patch pipeline + operator click + ledger event in the
 * backend — this widget simply makes the chain visible on every
 * surface so the operator never loses track of where a strategy sits.
 */
interface PromoteStage {
  id: string;
  label: string;
  mode: string | null;
}

const STAGES: readonly PromoteStage[] = [
  { id: "backtest", label: "Backtest", mode: null },
  { id: "paper", label: "Paper", mode: "PAPER" },
  { id: "shadow", label: "Shadow", mode: "SHADOW" },
  { id: "canary", label: "Canary", mode: "CANARY" },
  { id: "live", label: "Live", mode: "LIVE" },
  { id: "auto", label: "Auto", mode: "AUTO" },
] as const;

export function PromoteChain() {
  const { data } = useQuery({
    queryKey: ["dashboard", "mode"],
    queryFn: ({ signal }) => fetchMode(signal),
    refetchInterval: 2_000,
  });
  // Sentinel that no STAGES.mode equals — prevents the Backtest chip
  // (which has mode === null) from looking active before the API resolves.
  const current = data?.current_mode ?? "__loading__";
  return (
    <div
      className="flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider"
      role="list"
      aria-label="promote chain"
      data-testid="promote-chain"
    >
      {STAGES.map((stage, idx) => {
        const isActive = stage.mode === current;
        const cls = `rounded border px-2 py-1 leading-none ${
          isActive
            ? "border-accent bg-accent text-bg"
            : "border-border bg-bg text-slate-400"
        }`;
        return (
          <span key={stage.id} role="listitem" className="flex items-center gap-1">
            <span className={cls} data-active={isActive ? "true" : "false"}>
              {stage.label}
            </span>
            {idx < STAGES.length - 1 && (
              <ChevronRight className="h-3 w-3 text-slate-600" />
            )}
          </span>
        );
      })}
    </div>
  );
}
