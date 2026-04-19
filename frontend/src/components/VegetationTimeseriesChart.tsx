import { useEffect, useId, useState } from 'react'
import {
  Area,
  AreaChart,
  Brush,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiFetch } from '../lib/apiFetch'
import type { Region } from '../types/region'

type SeriesRow = { month: string; ndvi: number }

type VegetationApiResponse = {
  tile_id: string
  source: 'raster' | 'synthetic'
  detail: string
  series: SeriesRow[]
}

type VegetationTimeseriesChartProps = {
  region: Region
  /** Taller chart + range slider to scrub across months/years */
  showBrush?: boolean
  chartHeight?: number
  className?: string
}

export function VegetationTimeseriesChart({
  region,
  showBrush = false,
  chartHeight = 260,
  className,
}: VegetationTimeseriesChartProps) {
  const gradId = `ndviFill-${useId().replace(/:/g, '')}`
  const [rows, setRows] = useState<SeriesRow[]>([])
  const [meta, setMeta] = useState<{
    tile_id: string
    source: string
    detail: string
  } | null>(null)
  const [status, setStatus] = useState<'loading' | 'ok' | 'error'>('loading')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    setErrorMessage(null)

    void apiFetch('/vegetation-timeseries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: region.lat, lng: region.lng }),
    })
      .then(async (res) => {
        if (!res.ok) {
          let msg = `Server error (${res.status})`
          try {
            const errBody = (await res.json()) as { detail?: unknown }
            if (typeof errBody.detail === 'string') {
              msg = errBody.detail
            }
          } catch {
            /* keep msg */
          }
          if (res.status === 404 && msg.startsWith('Server error')) {
            msg =
              'No tile for this location — pick a point inside a challenge tile (GET /tiles lists centers).'
          }
          throw new Error(msg)
        }
        return res.json() as Promise<VegetationApiResponse>
      })
      .then((data) => {
        if (cancelled) return
        setRows(data.series ?? [])
        setMeta({
          tile_id: data.tile_id,
          source: data.source,
          detail: data.detail,
        })
        setStatus('ok')
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setStatus('error')
        setRows([])
        setMeta(null)
        setErrorMessage(e instanceof Error ? e.message : 'Failed to load vegetation series.')
      })

    return () => {
      cancelled = true
    }
  }, [region.lat, region.lng])

  return (
    <div className={className ?? 'analysis-vega-chart'}>
      {status === 'loading' && (
        <p className="analysis-vega-status">Loading vegetation timeline…</p>
      )}
      {status === 'error' && errorMessage && (
        <p className="analysis-vega-status analysis-vega-status--error">{errorMessage}</p>
      )}
      {status === 'ok' && meta && (
        <>
          <p className="analysis-vega-meta">
            Tile <code>{meta.tile_id}</code>
            <span className="analysis-vega-pill" data-source={meta.source}>
              {meta.source === 'raster' ? 'NDVI from GeoTIFF' : 'NDVI proxy (demo)'}
            </span>
          </p>
          <p className="analysis-vega-detail">{meta.detail}</p>
        </>
      )}
      {status === 'ok' && rows.length > 0 && (
        <div className="analysis-vega-plot" role="img" aria-label="NDVI over time chart">
          <ResponsiveContainer width="100%" height={chartHeight + (showBrush ? 48 : 0)}>
            <AreaChart
              data={rows}
              margin={{
                top: 8,
                right: 12,
                left: 4,
                bottom: showBrush ? 8 : 0,
              }}
            >
              <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
              <XAxis
                dataKey="month"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                interval={5}
                angle={-32}
                textAnchor="end"
                height={56}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(v) => v.toFixed(1)}
                width={36}
                label={{
                  value: 'NDVI',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#94a3b8',
                  fontSize: 11,
                }}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(71, 85, 105, 0.6)',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(value: unknown) => {
                  const v =
                    typeof value === 'number'
                      ? value.toFixed(3)
                      : String(value ?? '—')
                  return [v, 'NDVI']
                }}
              />
              <Area
                type="monotone"
                dataKey="ndvi"
                stroke="#4ade80"
                strokeWidth={2}
                fill={`url(#${gradId})`}
                dot={false}
                activeDot={{ r: 4, fill: '#86efac' }}
              />
              {showBrush && (
                <Brush
                  dataKey="month"
                  height={40}
                  stroke="#64748b"
                  fill="rgba(15, 23, 42, 0.75)"
                  travellerWidth={10}
                  tickFormatter={(v) => {
                    const s = String(v)
                    if (s.length >= 7) return s.slice(0, 7)
                    return s
                  }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      {status === 'ok' && rows.length === 0 && !errorMessage && (
        <p className="analysis-vega-status">
          No Sentinel-2 month entries in metadata for this tile’s NDVI layer.
        </p>
      )}
    </div>
  )
}
