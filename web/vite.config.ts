import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The gateway serves the built assets itself from web/dist, so there is no
// CORS story in production. In development, proxy the API to a local gateway
// so the browser still talks to a single origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/v1": { target: "http://localhost:8080", changeOrigin: true },
      "/metrics": "http://localhost:8080",
    },
  },
});
