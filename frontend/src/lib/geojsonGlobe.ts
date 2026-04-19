import { latLngToSurfacePosition } from './geo'

/** Closed ring of 3D points on the globe (last vertex may duplicate first). */
export type Vec3Ring = readonly [number, number, number][]

export type GeoJsonGlobeRings = {
  rings: Vec3Ring[]
  featureCount: number
}

function lngLatRingToVec3(ring: number[][], radius: number): Vec3Ring {
  const pts: [number, number, number][] = []
  for (const coord of ring) {
    const lng = coord[0]
    const lat = coord[1]
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) continue
    const p = latLngToSurfacePosition(lat, lng, radius)
    pts.push([p[0], p[1], p[2]])
  }
  if (pts.length >= 2) {
    const a = pts[0]
    const b = pts[pts.length - 1]
    if (a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]) {
      pts.push([a[0], a[1], a[2]])
    }
  }
  return pts
}

/** Extract outer rings from GeoJSON polygons; ignores holes (v1). */
export function geoJsonToGlobeRings(data: unknown, radius: number): GeoJsonGlobeRings {
  if (!data || typeof data !== 'object') {
    return { rings: [], featureCount: 0 }
  }
  const fc = data as { type?: string; features?: unknown[] }
  if (fc.type !== 'FeatureCollection' || !Array.isArray(fc.features)) {
    return { rings: [], featureCount: 0 }
  }

  const rings: Vec3Ring[] = []
  for (const feat of fc.features) {
    if (!feat || typeof feat !== 'object') continue
    const g = (feat as { geometry?: unknown }).geometry
    if (!g || typeof g !== 'object') continue
    const geom = g as { type?: string; coordinates?: unknown }

    if (geom.type === 'Polygon' && Array.isArray(geom.coordinates)) {
      const poly = geom.coordinates as number[][][]
      const outer = poly[0]
      if (outer?.length) rings.push(lngLatRingToVec3(outer, radius))
    } else if (geom.type === 'MultiPolygon' && Array.isArray(geom.coordinates)) {
      const multi = geom.coordinates as number[][][][]
      for (const poly of multi) {
        const outer = poly[0]
        if (outer?.length) rings.push(lngLatRingToVec3(outer, radius))
      }
    }
  }

  return { rings, featureCount: fc.features.length }
}
