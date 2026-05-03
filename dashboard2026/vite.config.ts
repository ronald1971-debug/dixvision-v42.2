import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Build output is mounted under `/dash2/` by the FastAPI harness so
// the existing vanilla pages keep their canonical URLs untouched.
// `base` MUST match the StaticFiles mount path or the bundle's
// asset references break.
export default defineConfig({
  base: "/dash2/",
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
          // J-track: keep the main entry small. Heavy vendor libs and
          // each top-level page are routed to their own chunks so first
          // paint isn't gated on a 800-kB monolith and so route-switch
          // payloads stay route-local.
          if (id.includes("node_modules")) {
            if (id.includes("react-grid-layout")) return "grid";
            if (id.includes("lucide-react")) return "icons";
            // react + react-dom + scheduler share an internal closure
            // graph; keep them in one chunk to avoid a circular split.
            return "vendor";
          }
          // One chunk per asset-class page (each pulls in its own widget tree).
          const assetMatch = id.match(/\/pages\/asset\/(\w+)Page\.tsx?$/);
          if (assetMatch) return `page-asset-${assetMatch[1].toLowerCase()}`;
          // One chunk per top-level system page.
          const sysMatch = id.match(/\/pages\/(\w+)Page\.tsx?$/);
          if (sysMatch) return `page-${sysMatch[1].toLowerCase()}`;
          // Group widget folders together so they're not each a 5-kB chunk.
          const widgetMatch = id.match(/\/widgets\/([\w-]+)\//);
          if (widgetMatch) return `widgets-${widgetMatch[1]}`;
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Dev server: proxy `/api/*` to the FastAPI harness so the
      // React app talks to the real backend without CORS shims.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
