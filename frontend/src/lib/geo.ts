import type { Mesh } from 'three'
import { Vector3 } from 'three'

/** Unit sphere position for the Earth mesh used in GlobeCanvas (same axis convention). */
export function latLngToSurfacePosition(
  lat: number,
  lng: number,
  radius: number,
): readonly [number, number, number] {
  const latRad = (lat * Math.PI) / 180
  const lngRad = (lng * Math.PI) / 180
  const x = radius * Math.cos(latRad) * Math.cos(lngRad)
  const y = radius * Math.sin(latRad)
  const z = -radius * Math.cos(latRad) * Math.sin(lngRad)
  return [x, y, z] as const
}

export function worldPointToLatLng(mesh: Mesh, worldPoint: Vector3) {
  const local = worldPoint.clone()
  mesh.worldToLocal(local)
  const v = local.normalize()
  const lat =
    Math.asin(Math.max(-1, Math.min(1, v.y))) * (180 / Math.PI)
  const lng = Math.atan2(v.x, v.z) * (180 / Math.PI)
  return { lat, lng }
}
