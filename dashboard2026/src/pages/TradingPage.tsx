import { AlgoOrderBuilder } from "@/widgets/trading/AlgoOrderBuilder";
import { BasketOrderEditor } from "@/widgets/trading/BasketOrderEditor";
import { ConditionalBracketBuilder } from "@/widgets/trading/ConditionalBracketBuilder";
import { OrderHotkeysPanel } from "@/widgets/trading/OrderHotkeysPanel";
import { PreTradeSlippageSim } from "@/widgets/trading/PreTradeSlippageSim";

/**
 * F-track surface — Order entry depth.
 *
 * Five widgets covering the prop-trader cockpit baseline that was
 * missing from the v42.2 dashboard:
 *   - AlgoOrderBuilder        (TWAP / VWAP / Iceberg / POV)
 *   - ConditionalBracketBuilder (if-then triggers + TP/SL/trail/OCO)
 *   - BasketOrderEditor       (multi-leg basket with target weights)
 *   - PreTradeSlippageSim     (square-root impact + Almgren-Chriss)
 *   - OrderHotkeysPanel       (configurable chords)
 *
 * Every widget stages intents through the operator-approval edge
 * (INV-72); none of them auto-execute.
 */
export function TradingPage() {
  return (
    <div className="grid h-full grid-cols-1 gap-3 overflow-auto p-3 lg:grid-cols-2 xl:grid-cols-3">
      <div className="min-h-[360px] xl:col-span-2">
        <AlgoOrderBuilder />
      </div>
      <div className="min-h-[360px]">
        <ConditionalBracketBuilder />
      </div>
      <div className="min-h-[360px] xl:col-span-2">
        <BasketOrderEditor />
      </div>
      <div className="min-h-[360px]">
        <PreTradeSlippageSim />
      </div>
      <div className="min-h-[360px] xl:col-span-3">
        <OrderHotkeysPanel />
      </div>
    </div>
  );
}
