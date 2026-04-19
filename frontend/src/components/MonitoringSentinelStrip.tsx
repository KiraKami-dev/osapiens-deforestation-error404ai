import type { Region } from '../types/region'
import { ApiBackedImg } from './ApiBackedImg'
import { useMonitoringFrames } from '../lib/useMonitoringFrames'
import './MonitoringSentinelStrip.css'

type MonitoringSentinelStripProps = {
  region: Region
  maxFrames?: number
}

export function MonitoringSentinelStrip({
  region,
  maxFrames = 20,
}: MonitoringSentinelStripProps) {
  const { status, error, data } = useMonitoringFrames('/tile-sentinel-frames', region, maxFrames)

  if (status === 'loading') {
    return <p className="monitoring-sentinel-status">Loading monthly preview images…</p>
  }
  if (status === 'error' && error) {
    return <p className="monitoring-sentinel-status monitoring-sentinel-status--error">{error}</p>
  }
  if (!data?.frames.length) {
    return (
      <p className="monitoring-sentinel-status">No preview frames returned for this tile.</p>
    )
  }

  return (
    <div className="monitoring-sentinel">
      <p className="monitoring-sentinel-detail">Monthly previews for the selected tile.</p>
      <ul className="monitoring-sentinel-strip" aria-label="Monthly previews for selected tile">
        {data.frames.map((f) => (
          <li key={`${f.year}-${f.month_num}`} className="monitoring-sentinel-item">
            <figure className="monitoring-sentinel-figure">
              <div className="monitoring-sentinel-img-wrap">
                <ApiBackedImg
                  src={f.preview_url}
                  alt={`True-color preview for ${f.month}`}
                  loading="lazy"
                  decoding="async"
                />
                <span
                  className={`monitoring-sentinel-badge monitoring-sentinel-badge--${f.source}`}
                  title={
                    f.source === 'raster'
                      ? 'Built from full-resolution source imagery'
                      : 'Estimated preview generated for this month'
                  }
                >
                  {f.source === 'raster' ? 'Source image' : 'Estimated'}
                </span>
              </div>
              <figcaption className="monitoring-sentinel-cap">{f.month}</figcaption>
            </figure>
          </li>
        ))}
      </ul>
    </div>
  )
}
