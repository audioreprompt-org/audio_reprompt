import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true, // permite acceso desde Docker
    port: 5173,
  },
  assetsInclude: ["**/*.svg"],
});
