import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// DIX MEME — DEXtools-styled memecoin dashboard.
//
// The FastAPI harness mounts the build artefact under `/meme/`, so `base`
// MUST match or the bundle's asset references break. This is a *separate*
// React app from `/dash2/`; both apps talk to the SAME backend `/api/*`
// surface. There is no parallel authority path: every execution intent
// (manual order, sniper hit, copy-trade mirror) flows through the same
// `/api/intent` chokepoint and the same Governance engine that the
// operator console uses.
export default defineConfig({
  base: "/meme/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    emptyOutDir: true,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("lightweight-charts")) return "charts";
            if (id.includes("lucide-react")) return "icons";
            return "vendor";
          }
          // One chunk per top-level page (PairExplorer / CopyTrading /
          // Sniper / Trade / Multiswap …) so first paint stays small
          // even as the surface grows.
          const pageMatch = id.match(/\/pages\/(\w+)Page\.tsx?$/);
          if (pageMatch) return `page-${pageMatch[1].toLowerCase()}`;
        },
      },
    },
  },
  server: {
    // Different port than dashboard2026 so both dev servers can run
    // side-by-side without colliding (5173 = /dash2/, 5174 = /meme/).
    port: 5174,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.VITE_DEV_PROXY_PORT ?? "8080"}`,
        changeOrigin: true,
      },
    },
  },
});
