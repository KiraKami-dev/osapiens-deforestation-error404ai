import type { Region } from '../types/region'
import { ApiBackedImg } from './ApiBackedImg'
import { useMonitoringFrames } from '../lib/useMonitoringFrames'
import './MonitoringSentinelStrip.css'

type MonitoringS1StripProps = {
  region: Region
  maxFrames?: number
}

export function MonitoringS1Strip({ region, maxFrames = 20 }: MonitoringS1StripProps) {
  const { status, error, data } = useMonitoringFrames('/tile-sentinel1-frames', region, maxFrames)

  if (status === 'loading') {
    return <p className="monitoring-sentinel-status">Loading Sentinel-1 SAR previews…</p>
  }
  if (status === 'error' && error) {
    return <p className="monitoring-sentinel-status monitoring-sentinel-status--error">{error}</p>
  }
  if (!data?.frames.length) {
    return (
      <p className="monitoring-sentinel-status">No SAR preview frames returned for this tile.</p>
    )
  }

  return (
    <div className="monitoring-sentinel">
      <p className="monitoring-sentinel-detail">{data.detail}</p>
      <ul className="monitoring-sentinel-strip" aria-label="Sentinel-1 monthly SAR previews">
        {data.frames.map((f) => (
          <li key={`s1-${f.year}-${f.month_num}`} className="monitoring-sentinel-item">
            <figure className="monitoring-sentinel-figure">
              <div className="monitoring-sentinel-img-wrap">
                <ApiBackedImg
                  src={f.preview_url}
                  alt={`Sentinel-1 RTC backscatter preview for ${f.month}`}
                  loading="lazy"
                  decoding="async"
                />
                <span
                  className={`monitoring-sentinel-badge monitoring-sentinel-badge--${f.source}`}
                  title={
                    f.source === 'raster'
                      ? 'RTC GeoTIFF on disk (single-band backstretch)'
                      : 'Synthetic SAR-like preview (no local file)'
                  }
                >
                  {f.source === 'raster' ? 'GeoTIFF' : 'Demo'}
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
