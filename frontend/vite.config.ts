import { defineConfig, loadEnv } from 'vite'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backend = env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    optimizeDeps: {
      entries: ['index.html'],
      noDiscovery: true,
      include: [
        'react', 'react-dom/client', 'react/jsx-runtime', 'react/jsx-dev-runtime',
        '@radix-ui/react-progress', '@radix-ui/react-select', '@radix-ui/react-separator', '@radix-ui/react-slot',
        'class-variance-authority', 'clsx', 'lucide-react', 'tailwind-merge',
      ],
    },
    server: {
      host: '127.0.0.1',
      port: 5173,
      fs: { allow: [process.cwd()] },
      proxy: {
        '/api': { target: backend, changeOrigin: true },
        '/health': { target: backend, changeOrigin: true },
      },
    },
  }
})
