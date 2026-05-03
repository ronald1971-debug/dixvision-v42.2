import { useState } from "react";

import { AutonomyRibbon } from "@/components/AutonomyRibbon";
import { ModeRibbon } from "@/components/ModeRibbon";
import { LiveStatusPill } from "@/components/LiveStatusPill";
import { PadlockFloors } from "@/components/PadlockFloors";
import { PromoteChain } from "@/components/PromoteChain";
import { Sidebar } from "@/components/Sidebar";
import { CognitiveChatPage } from "@/pages/CognitiveChatPage";
import { CredentialsPage } from "@/pages/CredentialsPage";
import { DyonLearningPage } from "@/pages/DyonLearningPage";
import { GovernancePage } from "@/pages/GovernancePage";
import { IndiraLearningPage } from "@/pages/IndiraLearningPage";
import { OperatorPage } from "@/pages/OperatorPage";
import { TestingPage } from "@/pages/TestingPage";
import { DexPage } from "@/pages/asset/DexPage";
import { ForexPage } from "@/pages/asset/ForexPage";
import { MemecoinPage } from "@/pages/asset/MemecoinPage";
import { NftPage } from "@/pages/asset/NftPage";
import { PerpsPage } from "@/pages/asset/PerpsPage";
import { SpotPage } from "@/pages/asset/SpotPage";
import { StocksPage } from "@/pages/asset/StocksPage";
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
    case "governance":
      return <GovernancePage />;
  }
}

export function App() {
  const route = useHashRoute();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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
    </div>
  );
}
