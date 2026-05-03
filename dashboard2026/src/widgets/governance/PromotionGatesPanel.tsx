import { useQuery } from "@tanstack/react-query";

import { fetchPromotionGates } from "@/api/governance";

/**
 * Tier-1 governance widget — Hash-anchored Promotion Gates panel.
 *
 * Surfaces:
 *   - the SHA-256 of the live `docs/promotion_gates.yaml`
 *   - the bound hash recorded at SHADOW entry (via the ledger)
 *   - whether the live hash matches the bound hash (governance gate)
 *   - which forward modes (CANARY/LIVE/AUTO) are currently gated
 *
 * Mechanism (PR #124): once SHADOW is entered the promotion-gates
 * yaml is hash-locked. Mid-window edits are visible — Governance
 * refuses CANARY/LIVE/AUTO with `PROMOTION_GATES_HASH_MISMATCH` until
 * the operator de-escalates to PAPER, edits the file, and restarts
 * the SHADOW clock with a new bound hash.
 */
export function PromotionGatesPanel() {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "promotion_gates"],
    queryFn: ({ signal }) => fetchPromotionGates(signal),
    refetchInterval: 10_000,
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Promotion gates
          </h2>
          <p className="text-[11px] text-slate-500">
            SHA-256 hash anchor — pre-committed thresholds (PR #124)
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

      <div className="flex-1 overflow-auto p-3 text-xs">
        {isPending && <p className="text-slate-500">loading gates…</p>}
        {isError && (
          <p className="text-rose-400">
            failed: {(error as Error).message}
          </p>
        )}
        {data && (
          <div className="space-y-3">
            <Row label="file" value={data.path} mono />
            <Row
              label="file present"
              value={data.file_present ? "yes" : "MISSING"}
              tone={data.file_present ? "ok" : "err"}
            />
            <Row label="file hash (live)" value={shortHash(data.file_hash)} mono />
            <Row label="bound hash" value={shortHash(data.bound_hash)} mono />
            <MatchPill
              backendWired={data.backend_wired}
              matches={data.matches}
            />
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
                gated forward targets
              </div>
              <div className="flex gap-1.5">
                {data.gated_targets.map((t) => (
                  <span
                    key={t}
                    className="rounded border border-border bg-bg px-2 py-0.5 font-mono text-[10px] text-slate-300"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <a
              href={data.doc_url}
              target="_blank"
              rel="noreferrer"
              className="block text-[11px] text-accent hover:underline"
            >
              view promotion_gates.yaml on GitHub →
            </a>
          </div>
        )}
      </div>
    </section>
  );
}

function shortHash(h: string | null): string {
  if (!h) return "—";
  return `${h.slice(0, 12)}…${h.slice(-8)}`;
}

function Row({
  label,
  value,
  mono,
  tone,
}: {
  label: string;
  value: string;
  mono?: boolean;
  tone?: "ok" | "err" | "warn";
}) {
  const toneClass =
    tone === "ok"
      ? "text-emerald-400"
      : tone === "err"
        ? "text-rose-400"
        : tone === "warn"
          ? "text-amber-400"
          : "text-slate-200";
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </span>
      <span
        className={`truncate ${toneClass} ${mono ? "font-mono text-[11px]" : ""}`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

function MatchPill({
  backendWired,
  matches,
}: {
  backendWired: boolean;
  matches: boolean | null;
}) {
  if (!backendWired) {
    return (
      <div className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-300">
        <strong>backend not wired</strong> — PromotionGates instance not
        attached to ui.server.STATE. The hash anchor refuses CANARY/LIVE/AUTO
        in the runtime FSM, but the dashboard cannot read its bound state
        until P0-7 wiring completes.
      </div>
    );
  }
  if (matches === null) {
    return (
      <div className="rounded border border-slate-600/40 bg-slate-800/40 px-2 py-1.5 text-[11px] text-slate-400">
        no bound hash yet — SHADOW window has not started in this process
      </div>
    );
  }
  if (matches) {
    return (
      <div className="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-1.5 text-[11px] text-emerald-300">
        <strong>HASH MATCHES</strong> — live yaml is identical to the
        SHADOW-bound document of record. CANARY/LIVE/AUTO promotions are
        permitted (subject to other gates).
      </div>
    );
  }
  return (
    <div className="rounded border border-rose-500/40 bg-rose-500/10 px-2 py-1.5 text-[11px] text-rose-300">
      <strong>HASH MISMATCH</strong> — live yaml diverges from the bound
      document. Governance refuses every forward gated transition until the
      operator de-escalates to PAPER and restarts the SHADOW clock.
    </div>
  );
}
