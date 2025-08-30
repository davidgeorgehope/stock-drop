// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000';

// Dev feature flags (Vite env)
const viteFlag = (import.meta?.env?.VITE_DEV_SIGNALS ?? '').toString().toLowerCase();
// Show in all non-production builds; in production require explicit flag
const DEV_SIGNALS_ENABLED = import.meta.env.PROD
  ? (viteFlag === '1' || viteFlag === 'true')
  : true;

export { API_BASE_URL, DEV_SIGNALS_ENABLED };