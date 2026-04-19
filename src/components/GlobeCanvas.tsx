import { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { Line, OrbitControls, Stars, useTexture } from '@react-three/drei'
import * as THREE from 'three'
import { apiFetch } from '../lib/apiFetch'
import { latLngToSurfacePosition, worldPointToLatLng } from '../lib/geo'
import { earthYawToFaceLatLngTowardCamera } from '../lib/globeFocus'
import type { Vec3Ring } from '../lib/geojsonGlobe'
import { isLikelyOcean } from '../lib/oceanHeuristic'
import type { Region } from '../types/region'

export const EARTH_RADIUS = 2
/** Slightly above tile markers so outline is visible. */
export const POLYGON_OVERLAY_RADIUS = EARTH_RADIUS + 0.042
const TILE_POINT_RADIUS = 0.014

/** Served from `/public/textures` (NASA Blue Marble–style assets via three.js examples). */
const TEXTURES = {
  map: '/textures/earth_atmos_2048.jpg',
  normalMap: '/textures/earth_normal_2048.jpg',
  clouds: '/textures/earth_clouds_1024.png',
} as const

/** Default camera matches `<Canvas camera={{ position: [0, 0.35, 5.2], fov: 45 }} />` */
const DEFAULT_CAMERA_POS = [0, 0.35, 5.2] as const

type EarthProps = {
  onSelect: (region: Region) => void
  tiles: TilePoint[]
  onTileClick: (tile: TilePoint) => void
  polygonRings: Vec3Ring[]
  /** When set (e.g. monitoring), Earth rotates so this point faces the camera; use with pauseSpin */
  focusLatLng: { lat: number; lng: number } | null
  pauseSpin: boolean
}

type TilePoint = {
  tile_id: string
  lat: number
  lng: number
}

function Earth({
  onSelect,
  tiles,
  onTileClick,
  polygonRings,
  focusLatLng,
  pauseSpin,
}: EarthProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const cloudsRef = useRef<THREE.Mesh>(null)
  const groupRef = useRef<THREE.Group>(null)
  const sampleCtxRef = useRef<CanvasRenderingContext2D | null>(null)
  const [isTileHovered, setIsTileHovered] = useState(false)

  const { map: loadedMap, normalMap: loadedNormal, clouds: loadedClouds } =
    useTexture(TEXTURES)

  const map = useMemo(() => {
    const t = loadedMap.clone()
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = 16
    t.needsUpdate = true
    return t
  }, [loadedMap])

  const normalMap = useMemo(() => {
    const t = loadedNormal.clone()
    t.colorSpace = THREE.NoColorSpace
    t.anisotropy = 16
    t.needsUpdate = true
    return t
  }, [loadedNormal])

  const clouds = useMemo(() => {
    const t = loadedClouds.clone()
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = 8
    t.needsUpdate = true
    return t
  }, [loadedClouds])

  useLayoutEffect(() => {
    const c = cloudsRef.current
    if (!c) return
    c.raycast = () => {}
  }, [])

  useLayoutEffect(() => {
    const img = loadedMap.image as CanvasImageSource & { width?: number }
    if (!img || !img.width) {
      sampleCtxRef.current = null
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = img.width as number
    canvas.height = (img as HTMLImageElement).height ?? (img as ImageBitmap).height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) {
      sampleCtxRef.current = null
      return
    }
    try {
      ctx.drawImage(img, 0, 0)
      sampleCtxRef.current = ctx
    } catch {
      sampleCtxRef.current = null
    }
  }, [loadedMap])

  useLayoutEffect(() => {
    const g = groupRef.current
    if (!g || !focusLatLng) return
    g.rotation.y = earthYawToFaceLatLngTowardCamera(
      focusLatLng.lat,
      focusLatLng.lng,
      DEFAULT_CAMERA_POS,
      EARTH_RADIUS,
    )
  }, [focusLatLng])

  useFrame((_, delta) => {
    if (pauseSpin) return
    if (groupRef.current && !isTileHovered) {
      groupRef.current.rotation.y += delta * 0.08
    }
  })

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    if (!meshRef.current) return

    const uv = e.uv
    const ctx = sampleCtxRef.current
    if (uv && ctx) {
      const w = ctx.canvas.width
      const h = ctx.canvas.height
      const tx = Math.min(w - 1, Math.max(0, Math.floor(uv.x * w)))
      const ty = Math.min(h - 1, Math.max(0, Math.floor((1 - uv.y) * h)))
      try {
        const { data } = ctx.getImageData(tx, ty, 1, 1)
        if (isLikelyOcean(data[0], data[1], data[2])) return
      } catch {
        /* tainted canvas etc. — fall through and allow region pick */
      }
    }

    const { lat, lng } = worldPointToLatLng(meshRef.current, e.point)
    onSelect({ lat, lng })
  }

  const handleTilePointClick = (tile: TilePoint) => {
    onSelect({ lat: tile.lat, lng: tile.lng })
    void onTileClick(tile)
  }

  return (
    <group ref={groupRef}>
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerOver={() => {
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto'
        }}
      >
        <sphereGeometry args={[EARTH_RADIUS, 96, 96]} />
        <meshStandardMaterial
          map={map}
          normalMap={normalMap}
          roughness={0.48}
          metalness={0}
        />
      </mesh>
      <TilePoints
        tiles={tiles}
        onTileClick={handleTilePointClick}
        onHoverChange={setIsTileHovered}
      />
      <GlobePolygonOverlays rings={polygonRings} />
      <mesh ref={cloudsRef} scale={1.005} renderOrder={1}>
        <sphereGeometry args={[EARTH_RADIUS, 80, 80]} />
        <meshStandardMaterial
          map={clouds}
          transparent
          opacity={0.78}
          depthWrite={false}
          roughness={1}
          metalness={0}
        />
      </mesh>
      <mesh scale={1.04}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        <meshBasicMaterial
          color="#b8e8ff"
          transparent
          opacity={0.09}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <mesh scale={1.12}>
        <sphereGeometry args={[EARTH_RADIUS, 48, 48]} />
        <meshBasicMaterial
          color="#dbeafe"
          transparent
          opacity={0.025}
          side={THREE.FrontSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}

function TilePoints({
  tiles,
  onTileClick,
  onHoverChange,
}: {
  tiles: TilePoint[]
  onTileClick: (tile: TilePoint) => void
  onHoverChange: (isHovered: boolean) => void
}) {
  return (
    <group>
      {tiles.map((tile) => (
        <mesh
          key={tile.tile_id}
          position={latLngToSurfacePosition(tile.lat, tile.lng, EARTH_RADIUS + 0.028)}
          onClick={(e) => {
            e.stopPropagation()
            void onTileClick(tile)
          }}
          onPointerOver={() => {
            document.body.style.cursor = 'pointer'
            onHoverChange(true)
          }}
          onPointerOut={() => {
            document.body.style.cursor = 'auto'
            onHoverChange(false)
          }}
        >
          <sphereGeometry args={[TILE_POINT_RADIUS, 10, 10]} />
          <meshStandardMaterial
            color="#22c55e"
            emissive="#16a34a"
            emissiveIntensity={0.55}
            roughness={0.35}
            metalness={0.1}
          />
        </mesh>
      ))}
    </group>
  )
}

function GlobePolygonRing({ ring }: { ring: Vec3Ring }) {
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = []
    for (const p of ring) {
      pts.push(new THREE.Vector3(p[0], p[1], p[2]))
    }
    return pts
  }, [ring])

  if (ring.length < 3) return null

  return (
    <Line
      points={points}
      color="#ef4444"
      lineWidth={2.8}
      depthTest
      renderOrder={2}
    />
  )
}

function GlobePolygonOverlays({ rings }: { rings: Vec3Ring[] }) {
  return (
    <group>
      {rings.map((ring, i) => (
        <GlobePolygonRing key={i} ring={ring} />
      ))}
    </group>
  )
}

function Scene({
  onSelectRegion,
  tiles,
  onTileClick,
  polygonRings,
  focusLatLng,
  pauseSpin,
}: {
  onSelectRegion: (r: Region) => void
  tiles: TilePoint[]
  onTileClick: (tile: TilePoint) => void
  polygonRings: Vec3Ring[]
  focusLatLng: { lat: number; lng: number } | null
  pauseSpin: boolean
}) {
  return (
    <>
      <color attach="background" args={['#0e1f36']} />
      <hemisphereLight args={['#e8f2ff', '#9aab8a', 1.05]} />
      <ambientLight intensity={0.52} color="#f2f7ff" />
      <directionalLight
        position={[4.5, 7, 5]}
        intensity={3.15}
        color="#fff6e6"
      />
      <directionalLight
        position={[-7, 3.5, -5]}
        intensity={1.05}
        color="#d9ecff"
      />
      <directionalLight
        position={[0, -5, 7]}
        intensity={0.65}
        color="#eef3e4"
      />
      <directionalLight
        position={[7, 1, -2]}
        intensity={0.55}
        color="#ffe8d4"
      />
      <Stars
        radius={120}
        depth={60}
        count={2200}
        factor={2.2}
        saturation={0}
        fade
        speed={0.08}
      />
      <Earth
        onSelect={onSelectRegion}
        tiles={tiles}
        onTileClick={onTileClick}
        polygonRings={polygonRings}
        focusLatLng={focusLatLng}
        pauseSpin={pauseSpin}
      />
      <OrbitControls
        enablePan={false}
        enableZoom
        zoomSpeed={0.9}
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.45}
        minDistance={2.35}
        maxDistance={9}
        minPolarAngle={Math.PI * 0.25}
        maxPolarAngle={Math.PI * 0.78}
      />
    </>
  )
}

type GlobeCanvasProps = {
  onSelectRegion: (region: Region) => void
  polygonRings?: Vec3Ring[]
  /** `contained` = fill parent (Monitoring panel). Default = full-viewport fixed canvas. */
  layout?: 'viewport' | 'contained'
  /** Center this lat/lng toward the camera (monitoring + demo overlays). */
  focusLatLng?: { lat: number; lng: number } | null
  /** Stop idle spin (recommended when focusLatLng is set). */
  pauseSpin?: boolean
}

export function GlobeCanvas({
  onSelectRegion,
  polygonRings = [],
  layout = 'viewport',
  focusLatLng = null,
  pauseSpin = false,
}: GlobeCanvasProps) {
  const [tiles, setTiles] = useState<TilePoint[]>([])

  useEffect(() => {
    let active = true
    const tilesUrl = new URL('../assets/tiles_clickable.json', import.meta.url).href
    void fetch(tilesUrl)
      .then((res) => res.json() as Promise<TilePoint[]>)
      .then((data) => {
        if (active) setTiles(data)
      })
      .catch((err) => {
        console.error('Failed to load tiles_clickable.json', err)
      })
    return () => {
      active = false
    }
  }, [])

  const handleTileClick = async (tile: TilePoint) => {
    try {
      const res = await apiFetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: tile.lat, lng: tile.lng }),
      })
      const data = await res.json()
      console.log('Analyze response:', data)
    } catch (err) {
      console.error('Failed to analyze tile', err)
    }
  }

  return (
    <div
      className={
        layout === 'contained'
          ? 'globe-canvas-wrap globe-canvas-wrap--contained'
          : 'globe-canvas-wrap'
      }
      aria-hidden="false"
    >
      <Canvas
        camera={{ position: [0, 0.35, 5.2], fov: 45 }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        }}
        dpr={[1, 2]}
        onCreated={({ gl }) => {
          gl.outputColorSpace = THREE.SRGBColorSpace
          gl.toneMapping = THREE.ACESFilmicToneMapping
          gl.toneMappingExposure = 1.38
        }}
      >
        <Suspense fallback={null}>
          <Scene
            onSelectRegion={onSelectRegion}
            tiles={tiles}
            onTileClick={handleTileClick}
            polygonRings={polygonRings}
            focusLatLng={focusLatLng}
            pauseSpin={pauseSpin}
          />
        </Suspense>
      </Canvas>
    </div>
  )
}
