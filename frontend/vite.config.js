import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development, /api goes to the FastAPI server on :8000. On Vercel, both live on one domain.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
