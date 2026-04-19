import { useEffect, useState } from 'react'
import { apiFetch } from './apiFetch'
import { apiUrl } from './apiUrl'
import type { Region } from '../types/region'

export type MonitoringFrame = {
  month: string
  year: number
  month_num: number
  preview_url: string
  source: 'raster' | 'synthetic'
}

export type MonitoringFramesResponse = {
  tile_id: string
  center: { lat: number; lng: number }
  detail: string
  frames: MonitoringFrame[]
}

type MonitoringFramesState = {
  key: string
  status: 'loading' | 'ok' | 'error'
  error: string | null
  data: MonitoringFramesResponse | null
}

export function useMonitoringFrames(
  endpoint: '/tile-sentinel-frames' | '/tile-sentinel1-frames',
  region: Region,
  maxFrames = 20,
): MonitoringFramesState {
  const requestKey = `${endpoint}:${region.lat}:${region.lng}:${maxFrames}`
  const [state, setState] = useState<MonitoringFramesState>({
    key: '',
    status: 'ok',
    error: null,
    data: null,
  })

  useEffect(() => {
    let cancelled = false
    const q = new URLSearchParams({
      lat: String(region.lat),
      lng: String(region.lng),
      max_frames: String(maxFrames),
    })

    void apiFetch(`${endpoint}?${q}`)
      .then(async (res) => {
        if (!res.ok) {
          const fallback =
            endpoint === '/tile-sentinel1-frames'
              ? `Could not load SAR previews (${res.status})`
              : `Could not load previews (${res.status})`
          let msg = fallback
          try {
            const j = (await res.json()) as { detail?: unknown }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            /* ignore */
          }
          throw new Error(msg)
        }
        return res.json() as Promise<MonitoringFramesResponse>
      })
      .then((json) => {
        if (cancelled) return
        const data: MonitoringFramesResponse = {
          ...json,
          frames: json.frames.map((f) => ({
            ...f,
            preview_url: apiUrl(f.preview_url),
          })),
        }
        setState({
          key: requestKey,
          status: 'ok',
          error: null,
          data,
        })
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setState({
          key: requestKey,
          status: 'error',
          error:
            e instanceof Error
              ? e.message
              : endpoint === '/tile-sentinel1-frames'
                ? 'Failed to load Sentinel-1 previews.'
                : 'Failed to load Sentinel previews.',
          data: null,
        })
      })

    return () => {
      cancelled = true
    }
  }, [endpoint, maxFrames, region.lat, region.lng, requestKey])

  if (state.key !== requestKey) {
    return {
      key: requestKey,
      status: 'loading',
      error: null,
      data: null,
    }
  }

  return { key: requestKey, status: state.status, error: state.error, data: state.data }
}
