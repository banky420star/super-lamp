import { defineConfig } from 'vite'

// Attempt to load React plugin if available; fall back gracefully if not.
let plugins: any[] = []
try {
  // Dynamically require to avoid hard failure when plugin isn't installed
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const reactPlugin = require('@vitejs/plugin-react')
  plugins = [typeof reactPlugin === 'function' ? reactPlugin() : reactPlugin?.default?.() ?? reactPlugin]
} catch {
  // Plugin not available; continue with a minimal config
  plugins = []
}

export default defineConfig({
  base: '/',
  plugins: plugins,
  server: {
    port: 4180,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5051',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:5051',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
