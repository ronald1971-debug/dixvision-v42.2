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
