import { apiUrl } from './apiUrl'

/** Ngrok free tier interstitial: programmatic requests must send this header. */
const NGROK_SKIP_HEADER = 'ngrok-skip-browser-warning'

function shouldSendNgrokBypass(targetUrl: string): boolean {
  try {
    return new URL(targetUrl).hostname.includes('ngrok')
  } catch {
    return false
  }
}

function resolveUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    return pathOrUrl
  }
  return apiUrl(pathOrUrl)
}

/**
 * Same as `fetch` but targets the configured API origin and adds ngrok bypass headers when needed.
 */
export function apiFetch(pathOrUrl: string, init?: RequestInit): Promise<Response> {
  const url = resolveUrl(pathOrUrl)
  const headers = new Headers(init?.headers)
  if (shouldSendNgrokBypass(url)) {
    headers.set(NGROK_SKIP_HEADER, 'true')
  }
  return fetch(url, { ...init, headers })
}

/** True when `<img src>` cannot skip ngrok’s browser page (use blob fetch + object URL instead). */
export function previewUrlNeedsBlobFetch(url: string): boolean {
  return shouldSendNgrokBypass(url)
}
