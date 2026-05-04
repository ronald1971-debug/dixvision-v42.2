import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { fetchMode } from "@/api/dashboard";
import { postOperatorUnlock } from "@/api/operator";

/**
 * 7-state Mode FSM ribbon — the single most important indicator on
 * the dashboard.
 *
 * Per the system manifest the legal modes are
 * LOCKED · SAFE · PAPER · SHADOW · CANARY · LIVE · AUTO.
 * The ribbon renders one chip per mode, colour-coded by the role of
 * that mode in the safety pipeline, and highlights the chip the
 * Governance FSM currently reports as `current_mode`. A tooltip on
 * each chip explains the operator-visible meaning of that mode.
 *
 * Source of truth: `/api/dashboard/mode`, polled every 2 s. The chip
 * highlighting reflects the *Governance ledger* state, never any UI
 * setting.
 *
 * When the FSM reports `is_locked === true` the ribbon also exposes
 * an inline **Unlock** button that POSTs `/api/operator/action/unlock`
 * (mediated by the OperatorInterfaceBridge → REQUEST_UNLOCK). On
 * approval the bridge transitions the mode to SAFE and the next 2 s
 * poll redraws the ribbon.
 */

type Tone = "neutral" | "info" | "warn" | "danger" | "ok" | "alert";

interface ModeChipSpec {
  name: string;
  tone: Tone;
  tooltip: string;
}

const CHIPS: readonly ModeChipSpec[] = [
  {
    name: "LOCKED",
    tone: "danger",
    tooltip:
      "LOCKED — system frozen. No order entry, no plugin lifecycle, no learning updates. Operator must explicitly unlock.",
  },
  {
    name: "SAFE",
    tone: "warn",
    tooltip:
      "SAFE — system idle. Engines on, execution off. The default starting state of every cold launch.",
  },
  {
    name: "PAPER",
    tone: "info",
    tooltip:
      "PAPER — paper broker only. Real signals routed to a fake fill engine; ledger and learning loop fully active.",
  },
  {
    name: "SHADOW",
    tone: "info",
    tooltip:
      "SHADOW — signals on, execution off. Strategies emit candidate trades that are scored and logged but never sent.",
  },
  {
    name: "CANARY",
    tone: "alert",
    tooltip:
      "CANARY — bounded live exposure. Per-trade notional capped to 1% (mode-effect table); promotion-gates enforce window.",
  },
  {
    name: "LIVE",
    tone: "ok",
    tooltip:
      "LIVE — full live execution under operator oversight. Every trade requires the AuthorityGuard chokepoint.",
  },
  {
    name: "AUTO",
    tone: "ok",
    tooltip:
      "AUTO — operator-attention relaxed to exception-only. Drift oracle and promotion-gates enforce continuous safety.",
  },
];

const TONE_BG: Record<Tone, string> = {
  neutral: "bg-bg border-border text-slate-500",
  info: "bg-accent/10 border-accent/40 text-accent",
  warn: "bg-amber-500/10 border-amber-500/40 text-amber-400",
  danger: "bg-red-500/10 border-red-500/40 text-red-400",
  ok: "bg-emerald-500/10 border-emerald-500/40 text-emerald-400",
  alert: "bg-orange-500/10 border-orange-500/40 text-orange-400",
};

const TONE_ACTIVE: Record<Tone, string> = {
  neutral: "bg-slate-700 border-slate-300 text-white",
  info: "bg-accent border-accent text-bg",
  warn: "bg-amber-500 border-amber-300 text-bg",
  danger: "bg-red-500 border-red-300 text-white",
  ok: "bg-emerald-500 border-emerald-300 text-bg",
  alert: "bg-orange-500 border-orange-300 text-bg",
};

export function ModeRibbon() {
  const queryClient = useQueryClient();
  const [feedback, setFeedback] = useState<{
    tone: "ok" | "danger";
    text: string;
  } | null>(null);

  const { data, isError } = useQuery({
    queryKey: ["dashboard", "mode"],
    queryFn: ({ signal }) => fetchMode(signal),
    refetchInterval: 2_000,
  });

  const unlockMutation = useMutation({
    mutationFn: () =>
      postOperatorUnlock({
        reason: "operator unlock from dashboard ribbon",
        requestor: "operator",
      }),
    onSuccess: (response) => {
      const decisionReason =
        response.decision &&
        typeof response.decision === "object" &&
        typeof (response.decision as { reason?: unknown }).reason === "string"
          ? ((response.decision as { reason: string }).reason)
          : null;
      if (response.approved) {
        setFeedback({
          tone: "ok",
          text: response.summary || "unlock approved",
        });
      } else {
        setFeedback({
          tone: "danger",
          text:
            decisionReason ??
            response.summary ??
            "unlock refused by governance",
        });
      }
      // Force the polling query to refetch immediately so the chip
      // flips off LOCKED without waiting the full 2 s window.
      void queryClient.invalidateQueries({ queryKey: ["dashboard", "mode"] });
    },
    onError: (err: unknown) => {
      const text =
        err instanceof Error ? err.message : "unlock request failed";
      setFeedback({ tone: "danger", text });
    },
  });

  const currentMode = data?.current_mode ?? null;
  const isLocked = data?.is_locked ?? false;
  const legalTargets = new Set(data?.legal_targets ?? []);

  return (
    <div
      className="flex flex-col gap-1 font-mono text-[11px] uppercase tracking-wider"
      role="status"
      aria-label="mode ribbon"
      data-testid="mode-ribbon"
    >
      <div className="flex items-center gap-1">
        {CHIPS.map((chip) => {
          const isActive = currentMode === chip.name;
          const isLegalTarget = legalTargets.has(chip.name);
          const baseClass =
            "rounded border px-2 py-1 leading-none transition-colors duration-150";
          let cls: string;
          if (isError) {
            cls = `${baseClass} ${TONE_BG.neutral} opacity-50`;
          } else if (isActive) {
            cls = `${baseClass} ${TONE_ACTIVE[chip.tone]} shadow-sm`;
          } else if (isLegalTarget) {
            cls = `${baseClass} ${TONE_BG[chip.tone]}`;
          } else {
            cls = `${baseClass} ${TONE_BG.neutral} opacity-60`;
          }
          return (
            <span
              key={chip.name}
              className={cls}
              title={chip.tooltip}
              data-active={isActive ? "true" : "false"}
              data-legal-target={isLegalTarget ? "true" : "false"}
            >
              {chip.name}
              {isActive && isLocked ? " · 🔒" : ""}
            </span>
          );
        })}
        {isLocked && (
          <button
            type="button"
            onClick={() => {
              setFeedback(null);
              unlockMutation.mutate();
            }}
            disabled={unlockMutation.isPending}
            data-testid="mode-ribbon-unlock"
            className="ml-2 rounded border border-emerald-500/60 bg-emerald-500/10 px-2 py-1 leading-none text-emerald-400 transition-colors duration-150 hover:bg-emerald-500 hover:text-bg disabled:cursor-not-allowed disabled:opacity-50"
            title="Request LOCKED → SAFE transition through the OperatorInterfaceBridge."
          >
            {unlockMutation.isPending ? "UNLOCKING…" : "UNLOCK"}
          </button>
        )}
      </div>
      {feedback && (
        <div
          className={
            feedback.tone === "ok"
              ? "text-[10px] normal-case tracking-normal text-emerald-400"
              : "text-[10px] normal-case tracking-normal text-red-400"
          }
          role="alert"
          data-testid="mode-ribbon-unlock-feedback"
        >
          {feedback.text}
        </div>
      )}
    </div>
  );
}
