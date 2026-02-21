import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  integrations: [react()],
  vite: {
    plugins: [tailwindcss()],
    server: {
      proxy: {
        "/diagnose": "http://localhost:8080",
      },
    },
  },
  outDir: "../static",  // output directly to backend/static
});
