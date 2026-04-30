import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  approveApproval,
  fetchApprovals,
  rejectApproval,
} from "@/api/cognitive_chat";
import type { ApprovalRequestApi } from "@/types/generated/api";

const APPROVALS_QUERY_KEY = ["cognitive", "chat", "approvals"] as const;

function fmtConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function sideClass(side: ApprovalRequestApi["proposal"]["side"]): string {
  if (side === "BUY") return "text-emerald-300";
  if (side === "SELL") return "text-red-300";
  return "text-slate-400";
}

export interface ApprovalPanelProps {
  /**
   * Wave-03 PR-5 — when a chat reply contains a structured ``propose``
   * fence the runtime queues an :class:`ApprovalRequest` and returns its
   * id on the chat turn response.  The chat page keeps the most-recent
   * id here so we can scroll/highlight the matching row when the
   * operator switches focus to the panel.
   */
  highlightProposalId?: string;
}

export function ApprovalPanel(props: ApprovalPanelProps) {
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const list = useQuery({
    queryKey: APPROVALS_QUERY_KEY,
    queryFn: ({ signal }) => fetchApprovals({ signal }),
    refetchInterval: 2_000,
  });

  const approveMut = useMutation({
    mutationFn: (id: string) => approveApproval(id),
    onSuccess: () => {
      setErrorDetail(null);
      void queryClient.invalidateQueries({ queryKey: APPROVALS_QUERY_KEY });
    },
    onError: (err: Error) => setErrorDetail(err.message),
  });

  const rejectMut = useMutation({
    mutationFn: (id: string) => rejectApproval(id),
    onSuccess: () => {
      setErrorDetail(null);
      void queryClient.invalidateQueries({ queryKey: APPROVALS_QUERY_KEY });
    },
    onError: (err: Error) => setErrorDetail(err.message),
  });

  const isPending = approveMut.isPending || rejectMut.isPending;

  const rows = list.data?.requests ?? [];
  const banner = (() => {
    if (list.isPending) {
      return (
        <p className="text-xs text-slate-500 font-mono">
          loading approval queue…
        </p>
      );
    }
    if (list.isError) {
      return (
        <p className="text-xs text-red-400 font-mono">
          approval-queue fetch failed: {(list.error as Error).message}
        </p>
      );
    }
    if (rows.length === 0) {
      return (
        <p className="text-xs text-slate-500 italic font-mono">
          no pending proposals — chat replies that contain a structured
          ``propose`` block will queue here for operator approval before
          they emit a SignalEvent on the bus.
        </p>
      );
    }
    return null;
  })();

  return (
    <section
      className="rounded border border-border bg-surface p-3 space-y-3"
      data-testid="approval-panel"
    >
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">
          Pending proposals
        </h2>
        <span className="text-xs text-slate-500 font-mono">
          {rows.length} pending
        </span>
      </header>

      {banner}

      {errorDetail ? (
        <p
          className="text-xs text-red-400 font-mono"
          data-testid="approval-error"
        >
          {errorDetail}
        </p>
      ) : null}

      <ul className="space-y-2">
        {rows.map((row) => {
          const highlighted =
            props.highlightProposalId !== undefined &&
            props.highlightProposalId === row.request_id;
          return (
            <li
              key={row.request_id}
              data-testid={`approval-row-${row.request_id}`}
              className={
                "rounded border bg-bg p-3 text-xs font-mono space-y-1 " +
                (highlighted ? "border-accent" : "border-border")
              }
            >
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                <span>
                  symbol:{" "}
                  <span className="text-slate-200">{row.proposal.symbol}</span>
                </span>
                <span>
                  side:{" "}
                  <span className={sideClass(row.proposal.side)}>
                    {row.proposal.side}
                  </span>
                </span>
                <span>
                  confidence:{" "}
                  <span className="text-slate-200">
                    {fmtConfidence(row.proposal.confidence)}
                  </span>
                </span>
              </div>
              {row.proposal.rationale ? (
                <p className="text-slate-400 whitespace-pre-wrap">
                  {row.proposal.rationale}
                </p>
              ) : null}
              <div className="flex gap-2 pt-1">
                <button
                  type="button"
                  className="rounded border border-emerald-400/50 bg-emerald-500/10 px-3 py-1 text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-50"
                  disabled={isPending}
                  onClick={() => approveMut.mutate(row.request_id)}
                  data-testid={`approval-approve-${row.request_id}`}
                >
                  approve
                </button>
                <button
                  type="button"
                  className="rounded border border-red-400/50 bg-red-500/10 px-3 py-1 text-red-300 hover:bg-red-500/20 disabled:opacity-50"
                  disabled={isPending}
                  onClick={() => rejectMut.mutate(row.request_id)}
                  data-testid={`approval-reject-${row.request_id}`}
                >
                  reject
                </button>
                <span className="ml-auto text-slate-500">
                  id: {row.request_id.slice(0, 8)}…
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
