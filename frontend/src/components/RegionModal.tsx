import { useEffect } from 'react'
import type { Region } from '../types/region'

type RegionModalProps = {
  region: Region
  onClose: () => void
  onConfirm: () => void
}

function formatCoord(n: number, pos: string, neg: string) {
  const abs = Math.abs(n).toFixed(2)
  const hemi = n >= 0 ? pos : neg
  return `${abs}° ${hemi}`
}

export function RegionModal({ region, onClose, onConfirm }: RegionModalProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="modal glass-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="region-modal-title"
      >
        <div className="modal-region-preview" aria-hidden="true">
          <div className="modal-region-preview-grid" />
          <div className="modal-region-preview-hot" />
        </div>
        <h2 id="region-modal-title" className="modal-title">
          Run detection for this region?
        </h2>
        <p className="modal-copy">
          Do you want to run deforestation detection for this region?
        </p>
        <dl className="coord-block">
          <div>
            <dt>Latitude</dt>
            <dd>{formatCoord(region.lat, 'N', 'S')}</dd>
          </div>
          <div>
            <dt>Longitude</dt>
            <dd>{formatCoord(region.lng, 'E', 'W')}</dd>
          </div>
        </dl>
        <div className="modal-actions">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={onConfirm}>
            Run analysis
          </button>
        </div>
      </div>
    </div>
  )
}
