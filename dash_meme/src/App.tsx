import { useState } from "react";

import { HotPairsTicker } from "@/components/HotPairsTicker";
import { Sidebar } from "@/components/Sidebar";
import { ToastHost } from "@/components/ToastHost";
import { TopBar } from "@/components/TopBar";
import { BigSwapPage } from "@/pages/BigSwapPage";
import { CopyTradingPage } from "@/pages/CopyTradingPage";
import { MultichartPage } from "@/pages/MultichartPage";
import { MultiswapPage } from "@/pages/MultiswapPage";
import { PairExplorerPage } from "@/pages/PairExplorerPage";
import { PoolExplorerPage } from "@/pages/PoolExplorerPage";
import { SniperPage } from "@/pages/SniperPage";
import { StatsPage } from "@/pages/StatsPage";
import { TradePage } from "@/pages/TradePage";
import { WalletInfoPage } from "@/pages/WalletInfoPage";
import { useHashRoute, type Route } from "@/router";

function renderRoute(route: Route) {
  switch (route) {
    case "explorer":
      return <PairExplorerPage />;
    case "pools":
      return <PoolExplorerPage />;
    case "bigswap":
      return <BigSwapPage />;
    case "multichart":
      return <MultichartPage />;
    case "trade":
      return <TradePage />;
    case "copy":
      return <CopyTradingPage />;
    case "sniper":
      return <SniperPage />;
    case "multiswap":
      return <MultiswapPage />;
    case "wallet":
      return <WalletInfoPage />;
    case "stats":
      return <StatsPage />;
  }
}

export function App() {
  const route = useHashRoute();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className="flex h-screen w-screen flex-col overflow-hidden bg-bg text-text-primary"
      data-theme="default"
    >
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active={route} collapsed={collapsed} />
        <div className="flex flex-1 flex-col overflow-hidden">
          <TopBar
            sidebarCollapsed={collapsed}
            onToggleSidebar={() => setCollapsed((c) => !c)}
          />
          <HotPairsTicker />
          <main className="flex-1 overflow-hidden">{renderRoute(route)}</main>
        </div>
      </div>
      <ToastHost />
    </div>
  );
}
