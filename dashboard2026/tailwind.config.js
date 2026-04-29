/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Match the operator-dashboard palette so wave-02 visually
        // continues from the vanilla pages (high-contrast dark UI).
        bg: "#0b0d12",
        surface: "#11141b",
        border: "#1f2330",
        accent: "#3aa0ff",
        ok: "#3ddc84",
        warn: "#ffaa3b",
        danger: "#ff5a5a",
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
