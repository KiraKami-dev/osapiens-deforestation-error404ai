import type { Region } from '../types/region'

export type LngLat = readonly [number, number]

export type SubmissionPolygon = {
  key: string
  id: string | null
  timeStep: string | null
  rings: LngLat[][]
}

export type SubmissionOverlayData = {
  polygons: SubmissionPolygon[]
  featureCount: number
  timeStep: string | null
}

export type GeoBounds = {
  minLng: number
  maxLng: number
  minLat: number
  maxLat: number
}

const TILE_SIZE_KM = 10
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function extractSubmissionOverlay(data: unknown): SubmissionOverlayData {
  if (!data || typeof data !== 'object') {
    return { polygons: [], featureCount: 0, timeStep: null }
  }

  const collection = data as { type?: string; features?: unknown[] }
  if (collection.type !== 'FeatureCollection' || !Array.isArray(collection.features)) {
    return { polygons: [], featureCount: 0, timeStep: null }
  }

  const polygons: SubmissionPolygon[] = []
  let timeStep: string | null = null

  collection.features.forEach((feature, featureIndex) => {
    if (!feature || typeof feature !== 'object') return
    const feat = feature as {
      id?: unknown
      geometry?: { type?: string; coordinates?: unknown }
      properties?: { time_step?: unknown }
    }
    const nextTimeStep =
      typeof feat.properties?.time_step === 'string' ? feat.properties.time_step : null
    if (!timeStep && nextTimeStep) timeStep = nextTimeStep

    const geometry = feat.geometry
    if (!geometry || typeof geometry !== 'object') return
    const base = {
      id: feat.id == null ? null : String(feat.id),
      timeStep: nextTimeStep,
    }

    if (geometry.type === 'Polygon' && Array.isArray(geometry.coordinates)) {
      const rings = extractRings(geometry.coordinates)
      if (rings.length) {
        polygons.push({ ...base, key: `poly-${featureIndex}`, rings })
      }
      return
    }

    if (geometry.type === 'MultiPolygon' && Array.isArray(geometry.coordinates)) {
      geometry.coordinates.forEach((poly, polyIndex) => {
        const rings = extractRings(poly)
        if (rings.length) {
          polygons.push({ ...base, key: `multi-${featureIndex}-${polyIndex}`, rings })
        }
      })
    }
  })

  return {
    polygons,
    featureCount: collection.features.length,
    timeStep,
  }
}

function extractRings(value: unknown): LngLat[][] {
  if (!Array.isArray(value)) return []
  const outer = value[0]
  if (!Array.isArray(outer)) return []

  const ring: LngLat[] = []
  for (const point of outer) {
    if (!Array.isArray(point) || point.length < 2) continue
    const lng = point[0]
    const lat = point[1]
    if (typeof lng !== 'number' || typeof lat !== 'number') continue
    ring.push([lng, lat])
  }
  return ring.length >= 3 ? [ring] : []
}

export function estimateTileBounds(region: Region, tileSizeKm = TILE_SIZE_KM): GeoBounds {
  const half = tileSizeKm / 2
  const latHalfSpan = half / 110.574
  const cosLat = Math.cos((region.lat * Math.PI) / 180)
  const lngHalfSpan = half / Math.max(111.32 * Math.abs(cosLat), 1e-6)
  return {
    minLng: region.lng - lngHalfSpan,
    maxLng: region.lng + lngHalfSpan,
    minLat: region.lat - latHalfSpan,
    maxLat: region.lat + latHalfSpan,
  }
}

export function filterSubmissionToBounds(
  polygons: SubmissionPolygon[],
  bounds: GeoBounds,
): SubmissionPolygon[] {
  return polygons.filter((polygon) => polygon.rings.some((ring) => ringIntersectsBounds(ring, bounds)))
}

function ringIntersectsBounds(ring: LngLat[], bounds: GeoBounds): boolean {
  let minLng = Number.POSITIVE_INFINITY
  let maxLng = Number.NEGATIVE_INFINITY
  let minLat = Number.POSITIVE_INFINITY
  let maxLat = Number.NEGATIVE_INFINITY

  for (const [lng, lat] of ring) {
    if (lng < minLng) minLng = lng
    if (lng > maxLng) maxLng = lng
    if (lat < minLat) minLat = lat
    if (lat > maxLat) maxLat = lat
  }

  return !(
    maxLng < bounds.minLng ||
    minLng > bounds.maxLng ||
    maxLat < bounds.minLat ||
    minLat > bounds.maxLat
  )
}

export function formatTimeStep(timeStep: string | null | undefined): string | null {
  if (!timeStep || !/^\d{4}$/.test(timeStep)) return null
  const monthIndex = Number(timeStep.slice(2, 4)) - 1
  if (monthIndex < 0 || monthIndex >= MONTHS.length) return null
  const year = 2000 + Number(timeStep.slice(0, 2))
  return `${MONTHS[monthIndex]} ${year}`
}

/**
 * Compact label from challenge `time_step` (YYMM): e.g. `2106` → `Jun '21`
 * (YY = year-in-century, MM = month 01–12).
 */
export function formatTimeStepShort(timeStep: string | null | undefined): string | null {
  if (!timeStep || !/^\d{4}$/.test(timeStep)) return null
  const yy = timeStep.slice(0, 2)
  const mm = timeStep.slice(2, 4)
  const monthIndex = Number(mm) - 1
  if (monthIndex < 0 || monthIndex >= MONTHS.length) return null
  return `${MONTHS[monthIndex]} '${yy}`
}

/** e.g. `2106` → `06 · 2021` (calendar month + full year). */
export function formatTimeStepMonthYearNumeric(timeStep: string | null | undefined): string | null {
  if (!timeStep || !/^\d{4}$/.test(timeStep)) return null
  const yy = timeStep.slice(0, 2)
  const mm = timeStep.slice(2, 4)
  const monthNum = Number(mm)
  if (monthNum < 1 || monthNum > 12) return null
  const year = 2000 + Number(yy)
  if (!Number.isFinite(year)) return null
  return `${mm} · ${year}`
}

/** Smallest legend/chip label: `2506` → `06'25` (MM + YY). */
export function formatTimeStepMmYy(timeStep: string | null | undefined): string | null {
  if (!timeStep || !/^\d{4}$/.test(timeStep)) return null
  const yy = timeStep.slice(0, 2)
  const mm = timeStep.slice(2, 4)
  const monthNum = Number(mm)
  if (monthNum < 1 || monthNum > 12) return null
  return `${mm}'${yy}`
}

export function projectRingToPercentPoints(ring: LngLat[], bounds: GeoBounds): string {
  const width = bounds.maxLng - bounds.minLng
  const height = bounds.maxLat - bounds.minLat
  if (width <= 0 || height <= 0) return ''

  return ring
    .map(([lng, lat]) => {
      const x = ((lng - bounds.minLng) / width) * 100
      const y = ((bounds.maxLat - lat) / height) * 100
      return `${clamp(x, 0, 100).toFixed(3)},${clamp(y, 0, 100).toFixed(3)}`
    })
    .join(' ')
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}
