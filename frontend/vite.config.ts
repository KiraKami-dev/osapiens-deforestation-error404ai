import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/analyze': { target: 'http://127.0.0.1:8001', changeOrigin: true },
      '/vegetation-timeseries': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/tiles': { target: 'http://127.0.0.1:8001', changeOrigin: true },
      '/submission-geojson': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/tile-sentinel-frames': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/tile-sentinel-preview': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/tile-sentinel1-frames': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/tile-sentinel1-preview': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
})
