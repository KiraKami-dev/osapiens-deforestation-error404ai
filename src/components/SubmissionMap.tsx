import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl, {
  type ExpressionSpecification,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type MapMouseEvent,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Region } from '../types/region'
import {
  formatTimeStep,
  formatTimeStepMmYy,
  formatTimeStepMonthYearNumeric,
  formatTimeStepShort,
} from '../lib/submissionOverlay'
import './SubmissionMap.css'

type SubmissionMapProps = {
  /** GeoJSON URL relative to the public root. Defaults to the bundled submission file. */
  geojsonUrl?: string
  /** Optional center; the map prefers to fit to polygons near this point. */
  region?: Region | null
  /** Search radius (km) around `region` used to find nearby polygons before falling back to data extent. */
  searchRadiusKm?: number
  /** Render mode: full-bleed analyze screen or embedded card. */
  variant?: 'fullscreen' | 'embedded'
  /** Optional className for the outer container. */
  className?: string
}

const DEFAULT_GEOJSON_URL = `${import.meta.env.BASE_URL}data/submission.geojson`

const TIME_STEP_COLORS: Record<string, string> = {
  '2106': '#fde047',
  '2206': '#fb923c',
  '2306': '#f43f5e',
  '2406': '#a855f7',
  '2506': '#38bdf8',
}
const FALLBACK_POLYGON_COLOR = '#22d3ee'

const FILL_COLOR_EXPR = [
  'match',
  ['get', 'time_step'],
  ...Object.entries(TIME_STEP_COLORS).flatMap(([k, v]) => [k, v]),
  FALLBACK_POLYGON_COLOR,
] as unknown as ExpressionSpecification

const SATELLITE_STYLE = {
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
    { id: 'satellite', type: 'raster' as const, source: 'satellite' },
    {
      id: 'labels',
      type: 'raster' as const,
      source: 'labels',
      paint: { 'raster-opacity': 0.85 } as Record<string, unknown>,
    },
  ],
}

type FeatureCollection = {
  type: 'FeatureCollection'
  features: GeoJSON.Feature[]
}

const EMPTY_FC: FeatureCollection = { type: 'FeatureCollection', features: [] }

export function SubmissionMap({
  geojsonUrl = DEFAULT_GEOJSON_URL,
  region = null,
  searchRadiusKm = 60,
  variant = 'fullscreen',
  className,
}: SubmissionMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const popupRef = useRef<maplibregl.Popup | null>(null)
  const regionMarkerRef = useRef<maplibregl.Marker | null>(null)
  const [data, setData] = useState<FeatureCollection>(EMPTY_FC)
  const [status, setStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [selectedTimeStep, setSelectedTimeStep] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await fetch(geojsonUrl)
        if (!res.ok) throw new Error(`HTTP ${res.status} loading ${geojsonUrl}`)
        const fc = (await res.json()) as FeatureCollection
        if (cancelled) return
        if (!fc || fc.type !== 'FeatureCollection') {
          setStatus('error')
          setError('Invalid GeoJSON: expected FeatureCollection.')
          return
        }
        setData(fc)
        setError(null)
        setStatus('ok')
      } catch (e: unknown) {
        if (cancelled) return
        setStatus('error')
        setError(e instanceof Error ? e.message : 'Failed to load GeoJSON.')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [geojsonUrl])

  const timeSteps = useMemo(() => {
    const set = new Set<string>()
    for (const f of data.features) {
      const ts = (f.properties as Record<string, unknown> | null)?.time_step
      if (typeof ts === 'string') set.add(ts)
    }
    return Array.from(set).sort()
  }, [data])

  const missingTimeStepCount = useMemo(() => {
    let n = 0
    for (const f of data.features) {
      const ts = (f.properties as Record<string, unknown> | null)?.time_step
      if (typeof ts !== 'string' || !ts.trim()) n += 1
    }
    return n
  }, [data])

  const polygonCentroids = useMemo<Array<[number, number]>>(() => {
    const out: Array<[number, number]> = []
    const centroidOfRing = (ring: GeoJSON.Position[]): [number, number] | null => {
      if (!ring.length) return null
      let sx = 0
      let sy = 0
      for (const p of ring) {
        sx += p[0]
        sy += p[1]
      }
      return [sx / ring.length, sy / ring.length]
    }
    for (const f of data.features) {
      const g = f.geometry
      if (!g) continue
      if (g.type === 'Polygon') {
        const c = centroidOfRing(g.coordinates[0] ?? [])
        if (c) out.push(c)
      } else if (g.type === 'MultiPolygon') {
        for (const poly of g.coordinates) {
          const c = centroidOfRing(poly[0] ?? [])
          if (c) out.push(c)
        }
      }
    }
    return out
  }, [data])

  const dataBounds = useMemo<LngLatBoundsLike | null>(() => {
    if (!polygonCentroids.length) return null
    let minLng = Infinity,
      maxLng = -Infinity,
      minLat = Infinity,
      maxLat = -Infinity
    for (const [lng, lat] of polygonCentroids) {
      if (lng < minLng) minLng = lng
      if (lng > maxLng) maxLng = lng
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }
    return [
      [minLng, minLat],
      [maxLng, maxLat],
    ]
  }, [polygonCentroids])

  const nearbyBounds = useMemo<LngLatBoundsLike | null>(() => {
    if (!region || !polygonCentroids.length) return null
    const R = 6371
    const toRad = (d: number) => (d * Math.PI) / 180
    const haversine = (lat1: number, lng1: number, lat2: number, lng2: number) => {
      const dLat = toRad(lat2 - lat1)
      const dLng = toRad(lng2 - lng1)
      const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2
      return 2 * R * Math.asin(Math.sqrt(a))
    }
    const near: Array<[number, number]> = []
    for (const [lng, lat] of polygonCentroids) {
      if (haversine(region.lat, region.lng, lat, lng) <= searchRadiusKm) near.push([lng, lat])
    }
    if (!near.length) return null
    let minLng = Infinity,
      maxLng = -Infinity,
      minLat = Infinity,
      maxLat = -Infinity
    for (const [lng, lat] of near) {
      if (lng < minLng) minLng = lng
      if (lng > maxLng) maxLng = lng
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }
    return [
      [minLng, minLat],
      [maxLng, maxLat],
    ]
  }, [region, polygonCentroids, searchRadiusKm])

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: SATELLITE_STYLE as maplibregl.StyleSpecification,
      // Default (no region): Vietnam / Thailand area; with region, tile center until fitBounds runs
      center: region ? [region.lng, region.lat] : [102.5, 14.2],
      zoom: region ? 12 : 5.2,
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
    map.addControl(
      new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }),
      'bottom-left',
    )
    mapRef.current = map
    return () => {
      popupRef.current?.remove()
      popupRef.current = null
      map.remove()
      mapRef.current = null
    }
    // Intentionally only on mount — region/data changes are handled in their own effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const apply = () => {
      const src = map.getSource('submission') as GeoJSONSource | undefined
      if (src) {
        src.setData(data as GeoJSON.GeoJSON)
        return
      }
      map.addSource('submission', {
        type: 'geojson',
        data: data as GeoJSON.GeoJSON,
        promoteId: undefined,
      })
      map.addLayer({
        id: 'submission-fill',
        type: 'fill',
        source: 'submission',
        paint: {
          'fill-color': FILL_COLOR_EXPR,
          'fill-opacity': [
            'case',
            ['boolean', ['feature-state', 'hover'], false],
            0.55,
            0.32,
          ],
        },
      })
      map.addLayer({
        id: 'submission-outline',
        type: 'line',
        source: 'submission',
        paint: {
          'line-color': FILL_COLOR_EXPR,
          'line-width': [
            'interpolate',
            ['linear'],
            ['zoom'],
            6,
            0.6,
            12,
            1.4,
            16,
            2.4,
          ],
          'line-opacity': 0.95,
        },
      })

      let hoveredId: number | string | undefined
      map.on('mousemove', 'submission-fill', (e: MapMouseEvent & { features?: GeoJSON.Feature[] }) => {
        map.getCanvas().style.cursor = 'pointer'
        const f = e.features?.[0]
        if (!f) return
        if (hoveredId !== undefined) {
          map.setFeatureState({ source: 'submission', id: hoveredId }, { hover: false })
        }
        hoveredId = f.id as number | string | undefined
        if (hoveredId !== undefined) {
          map.setFeatureState({ source: 'submission', id: hoveredId }, { hover: true })
        }
      })
      map.on('mouseleave', 'submission-fill', () => {
        map.getCanvas().style.cursor = ''
        if (hoveredId !== undefined) {
          map.setFeatureState({ source: 'submission', id: hoveredId }, { hover: false })
          hoveredId = undefined
        }
      })

      map.on('click', 'submission-fill', (e: MapMouseEvent & { features?: GeoJSON.Feature[] }) => {
        const f = e.features?.[0]
        if (!f) return
        const props = (f.properties ?? {}) as Record<string, unknown>
        const ts = typeof props.time_step === 'string' ? props.time_step : '—'
        const human = formatTimeStepShort(ts) ?? ts
        const numeric = formatTimeStepMonthYearNumeric(ts)
        const tile = typeof props.tile_id === 'string' ? props.tile_id : null
        const html = `
          <div class="submission-map-popup">
            <div class="submission-map-popup__row"><span>Period</span><strong>${human}</strong></div>
            ${
              numeric && ts !== '—'
                ? `<div class="submission-map-popup__row"><span>Month · year</span><strong>${numeric}</strong></div>`
                : ''
            }
            <div class="submission-map-popup__row"><span>time_step</span><strong>${ts}</strong></div>
            ${tile ? `<div class="submission-map-popup__row"><span>tile_id</span><strong>${tile}</strong></div>` : ''}
            <div class="submission-map-popup__row"><span>lng,lat</span><strong>${e.lngLat.lng.toFixed(5)}, ${e.lngLat.lat.toFixed(5)}</strong></div>
          </div>`
        popupRef.current?.remove()
        popupRef.current = new maplibregl.Popup({ closeButton: true, maxWidth: '260px' })
          .setLngLat(e.lngLat)
          .setHTML(html)
          .addTo(map)
      })
    }

    if (map.isStyleLoaded()) apply()
    else map.once('load', apply)
  }, [data])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const filter: ExpressionSpecification | null = selectedTimeStep
      ? ['==', ['get', 'time_step'], selectedTimeStep]
      : null
    const setFilters = () => {
      if (map.getLayer('submission-fill')) map.setFilter('submission-fill', filter)
      if (map.getLayer('submission-outline')) map.setFilter('submission-outline', filter)
    }
    if (map.isStyleLoaded()) setFilters()
    else map.once('load', setFilters)
  }, [selectedTimeStep])

  useEffect(() => {
    const map = mapRef.current
    if (!map || status !== 'ok') return
    const fit = () => {
      if (nearbyBounds) {
        map.fitBounds(nearbyBounds, { padding: 50, maxZoom: 15, duration: 700 })
        return
      }
      if (dataBounds) {
        map.fitBounds(dataBounds, { padding: 40, maxZoom: 10, duration: 700 })
      }
    }
    if (map.isStyleLoaded()) fit()
    else map.once('load', fit)
  }, [region, nearbyBounds, dataBounds, status])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (regionMarkerRef.current) {
      regionMarkerRef.current.remove()
      regionMarkerRef.current = null
    }
    if (!region) return
    const el = document.createElement('div')
    el.className = 'submission-map__region-marker'
    el.title = `Tile center ${region.lat.toFixed(4)}°, ${region.lng.toFixed(4)}°`
    regionMarkerRef.current = new maplibregl.Marker({ element: el, anchor: 'center' })
      .setLngLat([region.lng, region.lat])
      .addTo(map)
  }, [region])

  return (
    <div
      className={`submission-map submission-map--${variant}${className ? ` ${className}` : ''}`}
    >
      <div ref={containerRef} className="submission-map__canvas" />

      <div className="submission-map__legend" role="region" aria-label="Map filters and legend">
        <div className="submission-map__legend-compact">
          <p
            className="submission-map__legend-meta"
            title={
              region
                ? `${data.features.length.toLocaleString()} polygons · focus ${region.lat.toFixed(4)}°, ${region.lng.toFixed(4)}°${nearbyBounds === null ? ` · no polygons within ${searchRadiusKm} km (full extent)` : ''}`
                : `${data.features.length.toLocaleString()} polygons`
            }
          >
            <span className="submission-map__legend-meta-strong">
              {data.features.length.toLocaleString()} polygons
            </span>
            {region && (
              <>
                <span className="submission-map__legend-meta-sep" aria-hidden>
                  ·
                </span>
                <span className="submission-map__legend-meta-coords">
                  {region.lat.toFixed(2)}°, {region.lng.toFixed(2)}°
                </span>
                {nearbyBounds === null && (
                  <span className="submission-map__legend-meta-hint" title="No polygons near this tile">
                    {' '}
                    (wide view)
                  </span>
                )}
              </>
            )}
          </p>
          <div className="submission-map__chips-scroll">
            <div className="submission-map__chips">
              <button
                type="button"
                className={`submission-map__chip${selectedTimeStep === null ? ' is-active' : ''}`}
                onClick={() => setSelectedTimeStep(null)}
              >
                All
              </button>
              {timeSteps.map((ts) => (
                <button
                  type="button"
                  key={ts}
                  className={`submission-map__chip${selectedTimeStep === ts ? ' is-active' : ''}`}
                  onClick={() => setSelectedTimeStep((cur) => (cur === ts ? null : ts))}
                  style={{
                    ['--chip-color' as string]: TIME_STEP_COLORS[ts] ?? FALLBACK_POLYGON_COLOR,
                  }}
                  title={
                    formatTimeStepMmYy(ts)
                      ? `${formatTimeStep(ts) ?? formatTimeStepShort(ts)} — ${formatTimeStepMonthYearNumeric(ts)} — code ${ts}`
                      : ts
                  }
                >
                  <span className="submission-map__chip-dot" />
                  {formatTimeStepMmYy(ts) ?? ts}
                </button>
              ))}
            </div>
          </div>
        </div>

        <details className="submission-map__legend-details">
          <summary className="submission-map__legend-summary">Color key · time_step</summary>
          <div className="submission-map__legend-details-body">
            <p className="submission-map__color-legend-copy">
              Fill/outline = <code>time_step</code> as <code>YYMM</code> (YY = year in century, MM = month).
              Labels show month + year; raw code in tooltip on chips.
            </p>
            {timeSteps.length > 0 ? (
              <ul className="submission-map__color-grid">
                {timeSteps.map((ts) => {
                  const color = TIME_STEP_COLORS[ts] ?? FALLBACK_POLYGON_COLOR
                  const short = formatTimeStepShort(ts)
                  const numeric = formatTimeStepMonthYearNumeric(ts)
                  const paletteNote =
                    ts in TIME_STEP_COLORS ? '' : ' · fallback'
                  return (
                    <li key={ts} className="submission-map__color-grid-item">
                      <span
                        className="submission-map__color-swatch"
                        style={{ background: color }}
                        aria-hidden
                      />
                      <span className="submission-map__color-grid-text">
                        <span className="submission-map__color-primary">
                          {formatTimeStepMmYy(ts) ?? ts}
                          {paletteNote}
                        </span>
                        <span className="submission-map__color-secondary">
                          {short ?? '—'} · {numeric ?? `code ${ts}`}
                        </span>
                      </span>
                    </li>
                  )
                })}
              </ul>
            ) : (
              <p className="submission-map__color-legend-empty">No <code>time_step</code> in file.</p>
            )}
            {missingTimeStepCount > 0 && (
              <p className="submission-map__color-legend-foot">
                {missingTimeStepCount} without <code>time_step</code> → cyan.
              </p>
            )}
          </div>
        </details>
      </div>

      {status === 'loading' && (
        <div className="submission-map__overlay">Loading polygons…</div>
      )}
      {status === 'error' && error && (
        <div className="submission-map__overlay submission-map__overlay--error">{error}</div>
      )}
    </div>
  )
}

export default SubmissionMap
