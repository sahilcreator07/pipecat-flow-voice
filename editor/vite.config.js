import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  base: process.env.NODE_ENV === "production" ? "/pipecat-flows/" : "/",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
  },
  publicDir: resolve(__dirname, "public"),
});
