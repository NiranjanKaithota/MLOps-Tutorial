import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Needed for Docker
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // Redirect /api -> Python Backend (use 127.0.0.1 to avoid IPv6 issues)
        changeOrigin: true,
        secure: false,
      },
    },
  },
})