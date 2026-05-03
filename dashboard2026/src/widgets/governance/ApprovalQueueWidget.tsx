import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  approveApproval,
  fetchApprovals,
  rejectApproval,
} from "@/api/cognitive_chat";
import type {
  ApprovalRequestApi,
  ApprovalsListResponse,
} from "@/types/generated/api";

/**
 * Tier-1 governance widget — Operator approval queue (INV-72).
 *
 * Reads the ledger-backed projection of pending cognitive proposals
 * (PR #87 + PR #90). Each row carries the proposal payload and
 * approve/reject buttons that POST through
 * `/api/cognitive/chat/approvals/{id}/{verb}` so the operator's
 * decision is itself a ledger row.
 */
export function ApprovalQueueWidget() {
  const qc = useQueryClient();
  const [includeDecided, setIncludeDecided] = useState(false);
  const [decidedBy, setDecidedBy] = useState("operator");
  const [note, setNote] = useState("");

  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "approvals", { includeDecided }],
    queryFn: ({ signal }) => fetchApprovals({ includeDecided, signal }),
    refetchInterval: 4_000,
  });

  const approve = useMutation({
    mutationFn: (id: string) =>
      approveApproval(id, { decided_by: decidedBy, note }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["governance", "approvals"] }),
  });
  const reject = useMutation({
    mutationFn: (id: string) =>
      rejectApproval(id, { decided_by: decidedBy, note }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["governance", "approvals"] }),
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Approval queue
          </h2>
          <p className="text-[11px] text-slate-500">
            Operator-approval edge (INV-72 — PR #87 / #90)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-[10px] text-slate-400">
            <input
              type="checkbox"
              className="h-3 w-3"
              checked={includeDecided}
              onChange={(e) => setIncludeDecided(e.target.checked)}
            />
            include decided
          </label>
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="rounded border border-border bg-bg px-2 py-1 text-[11px] hover:border-accent disabled:opacity-50"
          >
            {isFetching ? "…" : "refresh"}
          </button>
        </div>
      </header>

      <div className="flex flex-col gap-2 border-b border-border px-3 py-2">
        <div className="flex gap-2">
          <input
            value={decidedBy}
            onChange={(e) => setDecidedBy(e.target.value)}
            placeholder="decided_by"
            className="w-32 rounded border border-border bg-bg px-2 py-1 text-[11px] outline-none focus:border-accent"
          />
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="optional note (recorded with the decision)"
            className="flex-1 rounded border border-border bg-bg px-2 py-1 text-[11px] outline-none focus:border-accent"
          />
        </div>
      </div>

      <div className="flex-1 overflow-auto p-2 text-xs">
        {isPending && <p className="px-2 py-1 text-slate-500">loading…</p>}
        {isError && (
          <p className="px-2 py-1 text-rose-400">
            failed: {(error as Error).message}
          </p>
        )}
        {data && data.requests.length === 0 && (
          <p className="px-2 py-1 text-slate-500">
            no {includeDecided ? "" : "pending "}approvals
          </p>
        )}
        {data && data.requests.length > 0 && (
          <ul className="flex flex-col gap-2">
            {data.requests.map((a) => (
              <ApprovalRow
                key={a.request_id}
                a={a}
                onApprove={() => approve.mutate(a.request_id)}
                onReject={() => reject.mutate(a.request_id)}
                pending={
                  approve.isPending && approve.variables === a.request_id
                    ? "approve"
                    : reject.isPending && reject.variables === a.request_id
                      ? "reject"
                      : null
                }
              />
            ))}
          </ul>
        )}
      </div>
      <Footer data={data} />
    </section>
  );
}

function ApprovalRow({
  a,
  onApprove,
  onReject,
  pending,
}: {
  a: ApprovalRequestApi;
  onApprove: () => void;
  onReject: () => void;
  pending: "approve" | "reject" | null;
}) {
  const status = String(a.status);
  return (
    <li className="rounded border border-border bg-bg/40 p-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[10px] text-slate-400" title={a.request_id}>
          {a.request_id.slice(0, 14)}…
        </span>
        <StatusChip status={status} />
      </div>
      <div className="mt-1 flex items-baseline gap-2 text-[11px]">
        <span className="rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[9px] text-slate-300">
          {a.proposal.symbol}
        </span>
        <span
          className={`font-mono text-[10px] ${
            String(a.proposal.side).toUpperCase() === "BUY"
              ? "text-emerald-400"
              : "text-rose-400"
          }`}
        >
          {String(a.proposal.side).toUpperCase()}
        </span>
        <span className="text-[10px] text-slate-400">
          conf {(a.proposal.confidence * 100).toFixed(1)}%
        </span>
      </div>
      {a.proposal.rationale && (
        <p className="mt-1 text-[11px] leading-snug text-slate-300">
          {a.proposal.rationale}
        </p>
      )}
      {status === "pending" && (
        <div className="mt-2 flex gap-1.5">
          <button
            type="button"
            onClick={onApprove}
            disabled={pending !== null}
            className="rounded border border-emerald-600/60 bg-emerald-600/10 px-2 py-1 text-[10px] text-emerald-300 hover:bg-emerald-600/20 disabled:opacity-50"
          >
            {pending === "approve" ? "…" : "approve"}
          </button>
          <button
            type="button"
            onClick={onReject}
            disabled={pending !== null}
            className="rounded border border-rose-600/60 bg-rose-600/10 px-2 py-1 text-[10px] text-rose-300 hover:bg-rose-600/20 disabled:opacity-50"
          >
            {pending === "reject" ? "…" : "reject"}
          </button>
        </div>
      )}
      {status !== "pending" && a.decided_by && (
        <div className="mt-1 text-[10px] text-slate-500">
          decided by {a.decided_by}
        </div>
      )}
    </li>
  );
}

function StatusChip({ status }: { status: string }) {
  const tone =
    status === "pending"
      ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
      : status === "approved"
        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
        : status === "rejected"
          ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
          : "border-slate-600/40 bg-slate-800/40 text-slate-400";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${tone}`}
    >
      {status}
    </span>
  );
}

function Footer({ data }: { data: ApprovalsListResponse | undefined }) {
  if (!data) return null;
  const pending = data.requests.filter(
    (a) => String(a.status) === "pending",
  ).length;
  return (
    <div className="flex items-center justify-between border-t border-border px-3 py-1 text-[10px] text-slate-500">
      <span>total: {data.requests.length}</span>
      <span>pending: {pending}</span>
    </div>
  );
}
