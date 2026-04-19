import * as THREE from 'three'
import { latLngToSurfacePosition } from './geo'

/**
 * Yaw (radians) to apply to the Earth group so a lat/lng on the surface faces the camera.
 * Camera is assumed to sit on +Z (default R3F view toward origin).
 */
export function earthYawToFaceLatLngTowardCamera(
  lat: number,
  lng: number,
  cameraPosition: readonly [number, number, number],
  radius: number,
): number {
  const p = new THREE.Vector3(...latLngToSurfacePosition(lat, lng, radius))
  const cam = new THREE.Vector3(
    cameraPosition[0],
    cameraPosition[1],
    cameraPosition[2],
  )
  const towardCam = cam.clone().normalize()

  const pXZ = new THREE.Vector3(p.x, 0, p.z)
  const tXZ = new THREE.Vector3(towardCam.x, 0, towardCam.z)
  if (pXZ.lengthSq() < 1e-10 || tXZ.lengthSq() < 1e-10) return 0
  pXZ.normalize()
  tXZ.normalize()

  const sin = pXZ.x * tXZ.z - pXZ.z * tXZ.x
  const cos = pXZ.dot(tXZ)
  return Math.atan2(sin, cos)
}
