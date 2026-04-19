import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl, {
  type GeoJSONSource,
  type MapMouseEvent,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Region } from '../types/region'
import './SubmissionGlobeMap.css'

type TilePoint = {
  tile_id: string
  lat: number
  lng: number
}

type SubmissionGlobeMapProps = {
  onSelectRegion: (region: Region, tile?: TilePoint) => void
}

const TILES_URL = new URL('../assets/tiles_clickable.json', import.meta.url).href
const SUBMISSION_URL = `${import.meta.env.BASE_URL}data/submission.geojson`
const AUTO_ROTATE_DEG_PER_SEC = 1.8
const AUTO_ROTATE_RESUME_DELAY_MS = 900

const GLOBE_STYLE = {
  version: 8 as const,
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    satellite: {
      type: 'raster' as const,
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      maxzoom: 19,
      attribution:
        'Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community',
    },
    labels: {
      type: 'raster' as const,
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: 'sky',
      type: 'background' as const,
      paint: { 'background-color': '#020617' } as Record<string, unknown>,
    },
    { id: 'satellite', type: 'raster' as const, source: 'satellite' },
    {
      id: 'labels',
      type: 'raster' as const,
      source: 'labels',
      paint: { 'raster-opacity': 0.85 } as Record<string, unknown>,
    },
  ],
}

export function SubmissionGlobeMap({ onSelectRegion }: SubmissionGlobeMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const tilesGeoJSONRef = useRef<GeoJSON.FeatureCollection>({
    type: 'FeatureCollection',
    features: [],
  })
  const isTileHoveredRef = useRef(false)
  const isUserInteractingRef = useRef(false)
  const lastInteractionAtRef = useRef(0)
  const [tiles, setTiles] = useState<TilePoint[]>([])
  const [hoveredTile, setHoveredTile] = useState<TilePoint | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch(TILES_URL)
      .then((r) => r.json() as Promise<TilePoint[]>)
      .then((d) => {
        if (!cancelled) setTiles(Array.isArray(d) ? d : [])
      })
      .catch((err) => console.error('Failed to load tiles_clickable.json', err))
    return () => {
      cancelled = true
    }
  }, [])

  const tilesGeoJSON = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: 'FeatureCollection',
      features: tiles.map((t, i) => ({
        type: 'Feature',
        id: i,
        geometry: { type: 'Point', coordinates: [t.lng, t.lat] },
        properties: { tile_id: t.tile_id, lat: t.lat, lng: t.lng },
      })),
    }),
    [tiles],
  )

  useEffect(() => {
    tilesGeoJSONRef.current = tilesGeoJSON
  }, [tilesGeoJSON])

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: GLOBE_STYLE as unknown as maplibregl.StyleSpecification,
      // Start centered near India on first paint
      center: [78.96, 22.59],
      zoom: 2.85,
      attributionControl: { compact: true },
      maxZoom: 16,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-right')
    mapRef.current = map

    const canvas = map.getCanvas()
    const onPointerDown = () => {
      isUserInteractingRef.current = true
      lastInteractionAtRef.current = Date.now()
    }
    const onPointerUp = () => {
      isUserInteractingRef.current = false
      lastInteractionAtRef.current = Date.now()
    }
    const onPointerLeave = (e: PointerEvent) => {
      if (e.buttons !== 0) return
      isUserInteractingRef.current = false
      lastInteractionAtRef.current = Date.now()
    }
    const onWindowBlur = () => {
      isUserInteractingRef.current = false
      lastInteractionAtRef.current = Date.now()
    }
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointerleave', onPointerLeave)
    window.addEventListener('pointerup', onPointerUp)
    window.addEventListener('blur', onWindowBlur)

    let rafId = 0
    let autoRotateStarted = false
    let prevTs = performance.now()
    const tick = (ts: number) => {
      const dtSec = (ts - prevTs) / 1000
      prevTs = ts
      const canAutoRotate =
        !isTileHoveredRef.current &&
        !isUserInteractingRef.current &&
        Date.now() - lastInteractionAtRef.current >= AUTO_ROTATE_RESUME_DELAY_MS
      if (canAutoRotate) {
        const center = map.getCenter()
        const nextLng =
          ((((center.lng + dtSec * AUTO_ROTATE_DEG_PER_SEC) % 360) + 540) % 360) - 180
        map.jumpTo({ center: [nextLng, center.lat] })
      }
      rafId = window.requestAnimationFrame(tick)
    }
    const startAutoRotate = () => {
      if (autoRotateStarted) return
      autoRotateStarted = true
      prevTs = performance.now()
      rafId = window.requestAnimationFrame(tick)
    }

    const initLayers = () => {
      if (map.getSource('tiles-globe')) return
      try {
        map.setProjection({ type: 'globe' })
      } catch {
        /* projection not supported in this MapLibre build — falls back to mercator */
      }

      map.addSource('submission-globe', {
        type: 'geojson',
        data: SUBMISSION_URL,
      })
      map.addLayer({
        id: 'submission-globe-fill',
        type: 'fill',
        source: 'submission-globe',
        paint: {
          'fill-color': '#f43f5e',
          'fill-opacity': 0.35,
        },
      })
      map.addLayer({
        id: 'submission-globe-outline',
        type: 'line',
        source: 'submission-globe',
        paint: {
          'line-color': '#fb7185',
          'line-width': 1,
          'line-opacity': 0.9,
        },
      })

      map.addSource('tiles-globe', {
        type: 'geojson',
        data: tilesGeoJSONRef.current as GeoJSON.GeoJSON,
      })
      map.addLayer({
        id: 'tiles-glow',
        type: 'circle',
        source: 'tiles-globe',
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            1,
            14,
            4,
            22,
            8,
            34,
          ],
          'circle-color': '#22c55e',
          'circle-opacity': 0.22,
          'circle-blur': 0.8,
        },
      })
      map.addLayer({
        id: 'tiles-dot',
        type: 'circle',
        source: 'tiles-globe',
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            1,
            6,
            4,
            8,
            8,
            10,
          ],
          'circle-color': '#22c55e',
          'circle-stroke-color': '#052e16',
          'circle-stroke-width': 1.6,
          'circle-opacity': 1,
        },
      })

      map.on('mouseenter', 'tiles-dot', (e: MapMouseEvent & { features?: GeoJSON.Feature[] }) => {
        map.getCanvas().style.cursor = 'pointer'
        isTileHoveredRef.current = true
        const f = e.features?.[0]
        if (!f) return
        const props = (f.properties ?? {}) as Record<string, unknown>
        setHoveredTile({
          tile_id: String(props.tile_id ?? ''),
          lat: Number(props.lat ?? 0),
          lng: Number(props.lng ?? 0),
        })
      })
      map.on('mouseleave', 'tiles-dot', () => {
        map.getCanvas().style.cursor = ''
        isTileHoveredRef.current = false
        setHoveredTile(null)
      })

      map.on('click', 'tiles-dot', (e: MapMouseEvent & { features?: GeoJSON.Feature[] }) => {
        const f = e.features?.[0]
        if (!f) return
        const props = (f.properties ?? {}) as Record<string, unknown>
        const tile: TilePoint = {
          tile_id: String(props.tile_id ?? ''),
          lat: Number(props.lat ?? 0),
          lng: Number(props.lng ?? 0),
        }
        onSelectRegion({ lat: tile.lat, lng: tile.lng }, tile)
      })
      startAutoRotate()
    }

    if (map.isStyleLoaded()) initLayers()
    else map.on('load', initLayers)
    map.on('style.load', initLayers)

    return () => {
      window.cancelAnimationFrame(rafId)
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointerleave', onPointerLeave)
      window.removeEventListener('pointerup', onPointerUp)
      window.removeEventListener('blur', onWindowBlur)
      map.off('load', initLayers)
      map.off('style.load', initLayers)
      map.remove()
      mapRef.current = null
      isTileHoveredRef.current = false
      isUserInteractingRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const apply = () => {
      const src = map.getSource('tiles-globe') as GeoJSONSource | undefined
      if (src) src.setData(tilesGeoJSON as GeoJSON.GeoJSON)
    }
    apply()
    map.once('idle', apply)
  }, [tilesGeoJSON])

  return (
    <div className="submission-globe">
      <div ref={containerRef} className="submission-globe__canvas" />

      {hoveredTile && (
        <div className="submission-globe__tooltip">
          <div className="submission-globe__tooltip-title">{hoveredTile.tile_id}</div>
          <div className="submission-globe__tooltip-sub">
            {hoveredTile.lat.toFixed(3)}°, {hoveredTile.lng.toFixed(3)}°
          </div>
        </div>
      )}
    </div>
  )
}

export default SubmissionGlobeMap
