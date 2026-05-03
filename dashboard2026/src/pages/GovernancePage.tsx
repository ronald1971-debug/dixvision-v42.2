import { ApprovalQueueWidget } from "@/widgets/governance/ApprovalQueueWidget";
import { AuditLedgerViewer } from "@/widgets/governance/AuditLedgerViewer";
import { DriftOraclePanel } from "@/widgets/governance/DriftOraclePanel";
import { HazardMonitorGrid } from "@/widgets/governance/HazardMonitorGrid";
import { PromotionGatesPanel } from "@/widgets/governance/PromotionGatesPanel";
import { SCVSLivenessGrid } from "@/widgets/governance/SCVSLivenessGrid";
import { StrategyRegistryFSM } from "@/widgets/governance/StrategyRegistryFSM";

/**
 * Tier-1 governance page (#/governance).
 *
 * Mounts the six Tier-1 governance widgets called out in the dash2
 * spec:
 *
 *   1. PromotionGatesPanel  (PR #124 — hash-anchored gates)
 *   2. DriftOraclePanel     (PR #125 — continuous AUTO-mode gate)
 *   3. ApprovalQueueWidget  (INV-72 — operator-approval edge)
 *   4. AuditLedgerViewer    (PR #64 — DecisionTrace browser)
 *   5. StrategyRegistryFSM  (PR #113 — lifecycle FSM panel)
 *   6. SCVSLivenessGrid + HazardMonitorGrid (PR #57 + HAZ-01..13)
 *
 * The mode + autonomy ribbons remain visible in the global header
 * so both orthogonal axes are controllable while inspecting
 * governance state.
 */
export function GovernancePage() {
  return (
    <section className="flex h-full flex-col">
      <header className="mb-3">
        <h1 className="text-lg font-semibold tracking-tight">
          Governance
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Six Tier-1 surfaces: promotion gates, drift oracle, operator
          approval queue, audit ledger / DecisionTrace browser,
          strategy lifecycle FSM, SCVS source liveness + hazard
          monitor. All read directly from the canonical ledger; the
          decision buttons in the approval queue route back through
          the ledger so every operator action is itself recorded.
        </p>
      </header>
      <div className="grid flex-1 grid-cols-1 gap-3 overflow-auto pb-6 lg:grid-cols-2 xl:grid-cols-3">
        <div className="min-h-[320px]">
          <PromotionGatesPanel />
        </div>
        <div className="min-h-[320px]">
          <DriftOraclePanel />
        </div>
        <div className="min-h-[320px]">
          <StrategyRegistryFSM />
        </div>
        <div className="min-h-[320px]">
          <ApprovalQueueWidget />
        </div>
        <div className="min-h-[320px] lg:col-span-2 xl:col-span-2">
          <AuditLedgerViewer />
        </div>
        <div className="min-h-[320px] lg:col-span-2 xl:col-span-2">
          <SCVSLivenessGrid />
        </div>
        <div className="min-h-[320px]">
          <HazardMonitorGrid />
        </div>
      </div>
    </section>
  );
}
