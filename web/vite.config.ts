import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The built bundle is served by FastAPI (api/main.py): index.html at "/" and
// everything else under "/assets". Nothing may be emitted outside those two
// places, which is why there is no `public/` directory — a file copied to the
// dist root would 404 in the container.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    assetsDir: "assets",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    // Same-origin in development too, so the app never needs to know where the
    // API lives. (LABEL_VERIFY_DEV_CORS=1 on the backend is then optional.)
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
