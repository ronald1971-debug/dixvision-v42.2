import { ASKBOrchestrator } from "@/widgets/ai/ASKBOrchestrator";
import { CounterfactualPanel } from "@/widgets/ai/CounterfactualPanel";
import { EarningsRAG } from "@/widgets/ai/EarningsRAG";
import { NLQConsole } from "@/widgets/ai/NLQConsole";
import { SmartMoneyTracker } from "@/widgets/ai/SmartMoneyTracker";

/**
 * Tier-3 AI surface — counterfactual / NLQ / earnings RAG /
 * smart-money / ASKB orchestrator.
 *
 * Mounts the five Tier-3 AI widgets in a responsive grid. Each
 * widget is independently scrollable so a long earnings transcript
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
      <div className="min-h-[320px] xl:col-span-3">
        <SmartMoneyTracker />
      </div>
    </div>
  );
}
