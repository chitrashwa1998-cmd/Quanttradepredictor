// Use relative URL for API requests
export const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? '/api'  // Development (proxied to port 8080)
  : '/api';  // Production (proxied)