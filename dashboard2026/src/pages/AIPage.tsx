import { AltSignalDashboard } from "@/widgets/ai/AltSignalDashboard";
import { ASKBOrchestrator } from "@/widgets/ai/ASKBOrchestrator";
import { CausalRiskAttribution } from "@/widgets/ai/CausalRiskAttribution";
import { CounterfactualPanel } from "@/widgets/ai/CounterfactualPanel";
import { EarningsRAG } from "@/widgets/ai/EarningsRAG";
import { IntentExecutionPanel } from "@/widgets/ai/IntentExecutionPanel";
import { MultilingualNewsFusion } from "@/widgets/ai/MultilingualNewsFusion";
import { NLQConsole } from "@/widgets/ai/NLQConsole";
import { SmartMoneyTracker } from "@/widgets/ai/SmartMoneyTracker";

/**
 * Tier-3 + E-track AI surface.
 *
 * Tier-3 (PR #129): counterfactual / NLQ / earnings RAG / smart-money /
 * ASKB orchestrator.
 *
 * E-track (this PR): multilingual news fusion / alt-signal dashboard /
 * causal risk attribution / intent execution router.
 *
 * Each widget is independently scrollable so a long earnings transcript
 * or a noisy NLQ history doesn't push other widgets off-screen.
 */
export function AIPage() {
  return (
    <div className="grid h-full grid-cols-1 gap-3 overflow-auto p-3 lg:grid-cols-2 xl:grid-cols-3">
      <div className="min-h-[320px] xl:col-span-2">
        <ASKBOrchestrator />
      </div>
      <div className="min-h-[320px]">
        <CounterfactualPanel />
      </div>
      <div className="min-h-[320px]">
        <NLQConsole />
      </div>
      <div className="min-h-[320px]">
        <EarningsRAG />
      </div>
      <div className="min-h-[320px]">
        <MultilingualNewsFusion />
      </div>
      <div className="min-h-[320px]">
        <AltSignalDashboard />
      </div>
      <div className="min-h-[320px]">
        <CausalRiskAttribution />
      </div>
      <div className="min-h-[320px]">
        <IntentExecutionPanel />
      </div>
      <div className="min-h-[320px] xl:col-span-3">
        <SmartMoneyTracker />
      </div>
    </div>
  );
}
