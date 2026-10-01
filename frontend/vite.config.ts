import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const API_URL =
  process.env.VITE_DEV_API_URL ?? "http://localhost:8000"

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],

  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },

  server: {
    proxy: {
      "/api": {
        target: API_URL,
        changeOrigin: true,
        secure: false,
      },
      "/health": {
        target: API_URL,
        changeOrigin: true,
        secure: false,
      },
      "/ready": {
        target: API_URL,
        changeOrigin: true,
        secure: false,
      },
    },
  },
})