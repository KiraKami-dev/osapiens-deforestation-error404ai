/** Origin for geospatial API (no trailing slash). Empty = same origin (Vite dev proxy). */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * Build URL for backend routes (`/vegetation-timeseries`, …).
 * Leaves `http(s):` URLs unchanged. Static assets under the site stay on same origin.
 */
export function apiUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    return pathOrUrl
  }
  const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`
  if (!API_BASE) return path
  return `${API_BASE}${path}`
}
