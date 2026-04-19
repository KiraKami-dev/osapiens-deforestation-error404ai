import { useEffect, useState } from 'react'
// import { MonitoringS1Strip } from './MonitoringS1Strip'
import { MonitoringSentinelStrip } from './MonitoringSentinelStrip'
import { SubmissionMap } from './SubmissionMap'
// import { VegetationTimeseriesChart } from './VegetationTimeseriesChart'
import { apiFetch } from '../lib/apiFetch'
import { extractSubmissionOverlay, type SubmissionPolygon } from '../lib/submissionOverlay'
import type { Region } from '../types/region'
import './MonitoringView.css'

export type MonitoringViewProps = {
  onBack: () => void
  /** Tile center; defaults to demo tile if omitted (e.g. tests). */
  region?: Region
}

const DEMO_GEOJSON_URL = `${import.meta.env.BASE_URL}data/demo_predictions.geojson`

/** Must fall inside a challenge tile in `aks/tiles_metadata.json` (submission uses 18NVJ_1_6). */
const MONITORING_REGION: Region = {
  lat: 3.030549839353614,
  lng: -75.7649152945593,
}

type SubmissionMeta = {
  preview: boolean
  totalFeatures: number
  returnedFeatures: number
  label: string
  timeStep: string | null
}

type SubmissionState = {
  key: string
  status: 'ok' | 'error'
  error: string | null
  polygons: SubmissionPolygon[]
  meta: SubmissionMeta | null
}

export function MonitoringView({ onBack, region: regionProp }: MonitoringViewProps) {
  const region = regionProp ?? MONITORING_REGION
  const requestKey = `${region.lat}:${region.lng}`
  const [submission, setSubmission] = useState<SubmissionState>({
    key: '',
    status: 'ok',
    error: null,
    polygons: [],
    meta: null,
  })

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      const trySubmission = await apiFetch('/submission-geojson?max_features=150').catch(
        () => null,
      )
      if (
        trySubmission?.ok &&
        trySubmission.headers.get('content-type')?.includes('json')
      ) {
        const data = (await trySubmission.json()) as {
          type?: string
          features?: unknown[]
          _preview?: boolean
          _total_features?: number
          _returned_features?: number
        }
        if (cancelled) return
        const overlay = extractSubmissionOverlay({
          type: 'FeatureCollection',
          features: data.features ?? [],
        })
        setSubmission({
          key: requestKey,
          status: overlay.polygons.length ? 'ok' : 'error',
          error: overlay.polygons.length ? null : 'No polygon geometry in submission GeoJSON.',
          polygons: overlay.polygons,
          meta: {
            preview: Boolean(data._preview),
            totalFeatures: data._total_features ?? overlay.featureCount,
            returnedFeatures: data._returned_features ?? overlay.featureCount,
            label: 'submission_version_0.geojson',
            timeStep: overlay.timeStep,
          },
        })
        return
      }

      const demo = await fetch(DEMO_GEOJSON_URL)
      if (!demo.ok) throw new Error(`Demo GeoJSON HTTP ${demo.status}`)
      const raw = (await demo.json()) as unknown
      if (cancelled) return
      const overlay = extractSubmissionOverlay(raw)
      setSubmission({
        key: requestKey,
        status: overlay.polygons.length ? 'ok' : 'error',
        error: overlay.polygons.length ? null : 'No polygon geometry found in GeoJSON.',
        polygons: overlay.polygons,
        meta: {
          preview: false,
          totalFeatures: overlay.featureCount,
          returnedFeatures: overlay.featureCount,
          label: 'demo_predictions.geojson',
          timeStep: overlay.timeStep,
        },
      })
    }

    void run().catch((e: unknown) => {
      if (cancelled) return
      setSubmission({
        key: requestKey,
        status: 'error',
        polygons: [],
        meta: null,
        error: e instanceof Error ? e.message : 'Failed to load GeoJSON.',
      })
    })

    return () => {
      cancelled = true
    }
  }, [requestKey])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onBack()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onBack])

  const geoStatus = submission.key === requestKey ? submission.status : 'loading'
  const geoError = submission.key === requestKey ? submission.error : null
  const polygons = submission.key === requestKey ? submission.polygons : []
  const meta = submission.key === requestKey ? submission.meta : null

  return (
    <div className="monitoring-view">
      <header className="monitoring-topbar">
        <button type="button" className="monitoring-back" onClick={onBack}>
          <span aria-hidden="true">←</span> Back to Globe
        </button>
        <div className="monitoring-topbar-text">
          <h1 className="monitoring-title">Tile analysis</h1>
          <p className="monitoring-lead">
            Explore your selected area in two views: mapped submission areas and a monthly image
            timeline. Use this page to check location, boundaries, and how the landscape changes over
            time.
          </p>
        </div>
      </header>

      <main className="monitoring-chart-main">
        {geoStatus === 'loading' && (
          <section className="monitoring-imagery-card glass-panel">
            <h2 className="monitoring-section-title">Your submitted areas on the map</h2>
            <p className="monitoring-chart-caption">Loading your submitted areas…</p>
          </section>
        )}

        {geoStatus === 'error' && geoError && (
          <section className="monitoring-imagery-card glass-panel">
            <h2 className="monitoring-section-title">Your submitted areas on the map</h2>
            <p className="monitoring-chart-caption monitoring-chart-caption--error">{geoError}</p>
          </section>
        )}

        {meta && geoStatus !== 'error' && (
          <section className="monitoring-imagery-card glass-panel monitoring-map-card">
            <div className="monitoring-map-card__head">
              <div>
                <h2 className="monitoring-section-title">Your submitted areas on the map</h2>
                <p className="monitoring-chart-caption">
                  {polygons.length} areas mapped. Click any area to view details, then use the
                  time-step filters to focus on a specific phase.
                </p>
              </div>
            </div>
            <div className="monitoring-map-card__map">
              <SubmissionMap region={region} variant="embedded" />
            </div>
          </section>
        )}

        {/* Vegetation index / change over time
        <section className="monitoring-chart-card glass-panel">
          <h2 className="monitoring-section-title">Vegetation index / change over time</h2>
          <p className="monitoring-chart-caption">
            API <code>POST /vegetation-timeseries</code> — NDVI from Sentinel-2 stacks only (not the full
            feature set at inference). Tile center ({region.lat.toFixed(4)}°, {region.lng.toFixed(4)}°)
          </p>
          <VegetationTimeseriesChart
            region={region}
            showBrush
            chartHeight={360}
            className="monitoring-vega-chart"
          />
        </section>
        */}

        <section className="monitoring-imagery-card glass-panel">
          <h2 className="monitoring-section-title">Monthly image timeline for this tile</h2>
          <p className="monitoring-chart-caption">
            Scroll through monthly snapshots of the same area to quickly compare visual changes over
            time.
          </p>
          <MonitoringSentinelStrip region={region} maxFrames={20} />
        </section>

        {/* Sentinel-1 SAR (same months as optical)
        <section className="monitoring-imagery-card glass-panel">
          <h2 className="monitoring-section-title">Sentinel-1 SAR (same months as optical)</h2>
          <p className="monitoring-chart-caption">
            API <code>GET /tile-sentinel1-frames</code> · RTC backscatter as grayscale (
            <code>GET /tile-sentinel1-preview/…</code> PNG). Months match the strip above (from the
            Sentinel-2 timeline); one orbit per month (descending preferred). Complements optical under
            clouds.
          </p>
          <MonitoringS1Strip region={region} maxFrames={20} />
        </section>
        */}
      </main>
    </div>
  )
}
