import { useEffect, useState } from "react";

import { AutonomyRibbon } from "@/components/AutonomyRibbon";
import { CommandPalette } from "@/components/CommandPalette";
import { ModeRibbon } from "@/components/ModeRibbon";
import { LiveStatusPill } from "@/components/LiveStatusPill";
import { PadlockFloors } from "@/components/PadlockFloors";
import { PreferencesBar } from "@/components/PreferencesBar";
import { PromoteChain } from "@/components/PromoteChain";
import { Sidebar } from "@/components/Sidebar";
import { AIPage } from "@/pages/AIPage";
import { ChartingPage } from "@/pages/ChartingPage";
import { CognitiveChatPage } from "@/pages/CognitiveChatPage";
import { CredentialsPage } from "@/pages/CredentialsPage";
import { DyonLearningPage } from "@/pages/DyonLearningPage";
import { GovernancePage } from "@/pages/GovernancePage";
import { IndiraLearningPage } from "@/pages/IndiraLearningPage";
import { MarketContextPage } from "@/pages/MarketContextPage";
import { OnChainPage } from "@/pages/OnChainPage";
import { OperatorPage } from "@/pages/OperatorPage";
import { OrderFlowPage } from "@/pages/OrderFlowPage";
import { RiskPage } from "@/pages/RiskPage";
import { TestingPage } from "@/pages/TestingPage";
import { TradingPage } from "@/pages/TradingPage";
import { DexPage } from "@/pages/asset/DexPage";
import { ForexPage } from "@/pages/asset/ForexPage";
import { MemecoinPage } from "@/pages/asset/MemecoinPage";
import { NftPage } from "@/pages/asset/NftPage";
import { PerpsPage } from "@/pages/asset/PerpsPage";
import { SpotPage } from "@/pages/asset/SpotPage";
import { StocksPage } from "@/pages/asset/StocksPage";
import { useApplyPreferences } from "@/preferences/store";
import { useHashRoute, type Route } from "@/router";

function renderRoute(route: Route) {
  switch (route) {
    case "spot":
      return <SpotPage />;
    case "perps":
      return <PerpsPage />;
    case "dex":
      return <DexPage />;
    case "memecoin":
      return <MemecoinPage />;
    case "forex":
      return <ForexPage />;
    case "stocks":
      return <StocksPage />;
    case "nft":
      return <NftPage />;
    case "operator":
      return <OperatorPage />;
    case "credentials":
      return <CredentialsPage />;
    case "chat":
      return <CognitiveChatPage />;
    case "indira":
      return <IndiraLearningPage />;
    case "dyon":
      return <DyonLearningPage />;
    case "testing":
      return <TestingPage />;
    case "onchain":
      return <OnChainPage />;
    case "ai":
      return <AIPage />;
    case "orderflow":
      return <OrderFlowPage />;
    case "governance":
      return <GovernancePage />;
    case "risk":
      return <RiskPage />;
    case "charting":
      return <ChartingPage />;
    case "market":
      return <MarketContextPage />;
    case "trading":
      return <TradingPage />;
  }
}

export function App() {
  const route = useHashRoute();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  useApplyPreferences();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-col gap-2 border-b border-border bg-surface px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="text-base font-semibold tracking-tight">
            DIX VISION
          </span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
            /{route}
          </span>
          <div className="ml-4 flex-1 overflow-x-auto">
            <ModeRibbon />
          </div>
          <AutonomyRibbon />
          <LiveStatusPill />
          <PreferencesBar />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <PromoteChain />
          <div className="flex-1" />
          <PadlockFloors />
        </div>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          active={route}
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((c) => !c)}
        />
        <main className="flex-1 overflow-auto px-4 py-3">
          {renderRoute(route)}
        </main>
      </div>
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onNavigate={(r) => {
          window.location.hash = `#/${r}`;
        }}
        extraActions={[
          {
            id: "sys:toggle-sidebar",
            group: "System",
            label: sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar",
            hint: "\u2318B",
            run: () => setSidebarCollapsed((c) => !c),
          },
        ]}
      />
    </div>
  );
}
