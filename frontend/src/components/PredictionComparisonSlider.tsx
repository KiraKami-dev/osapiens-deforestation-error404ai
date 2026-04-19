import { useMemo, useState } from 'react'
import type { Region } from '../types/region'
import { ApiBackedImg } from './ApiBackedImg'
import { useMonitoringFrames, type MonitoringFrame } from '../lib/useMonitoringFrames'
import {
  estimateTileBounds,
  filterSubmissionToBounds,
  formatTimeStep,
  projectRingToPercentPoints,
  type SubmissionPolygon,
} from '../lib/submissionOverlay'
import './PredictionComparisonSlider.css'

type PredictionComparisonSliderProps = {
  region: Region
  polygons: SubmissionPolygon[]
  submissionLabel: string
  totalFeatures: number
  returnedFeatures: number
  preview: boolean
  timeStep: string | null
}

type ImageryMode = 'optical' | 'sar'

type FrameSelection = Record<ImageryMode, string>

const FRAME_LIMIT = 20

function frameKey(frame: MonitoringFrame): string {
  return `${frame.year}-${String(frame.month_num).padStart(2, '0')}`
}

export function PredictionComparisonSlider({
  region,
  polygons,
  submissionLabel,
  totalFeatures,
  returnedFeatures,
  preview,
  timeStep,
}: PredictionComparisonSliderProps) {
  const optical = useMonitoringFrames('/tile-sentinel-frames', region, FRAME_LIMIT)
  const sar = useMonitoringFrames('/tile-sentinel1-frames', region, FRAME_LIMIT)
  const [mode, setMode] = useState<ImageryMode>('optical')
  const [reveal, setReveal] = useState(56)
  const [selectedFrames, setSelectedFrames] = useState<FrameSelection>({
    optical: '',
    sar: '',
  })

  const tileBounds = useMemo(() => estimateTileBounds(region), [region])
  const visiblePolygons = useMemo(
    () => filterSubmissionToBounds(polygons, tileBounds),
    [polygons, tileBounds],
  )
  const formattedTimeStep = formatTimeStep(timeStep)
  const activeFrames = mode === 'optical' ? optical : sar
  const frames = useMemo(() => activeFrames.data?.frames ?? [], [activeFrames.data])
  const selectedKey = selectedFrames[mode]
  const selectedFrame = useMemo(() => {
    if (!frames.length) return null
    return frames.find((frame) => frameKey(frame) === selectedKey) ?? frames[frames.length - 1]
  }, [frames, selectedKey])

  const overlayPolygons = useMemo(
    () =>
      visiblePolygons
        .flatMap((polygon) => polygon.rings)
        .map((ring) => projectRingToPercentPoints(ring, tileBounds))
        .filter(Boolean),
    [tileBounds, visiblePolygons],
  )

  const hasFrame = Boolean(selectedFrame)
  const statusMessage =
    activeFrames.status === 'loading'
      ? mode === 'optical'
        ? 'Loading optical frame…'
        : 'Loading SAR frame…'
      : activeFrames.status === 'error'
        ? activeFrames.error
        : !frames.length
          ? mode === 'optical'
            ? 'No optical preview frames returned for this tile.'
            : 'No SAR preview frames returned for this tile.'
          : null

  return (
    <section className="prediction-compare glass-panel" aria-label="Prediction comparison">
      <div className="prediction-compare__head">
        <div>
          <h2 className="monitoring-section-title">Prediction overlay comparison</h2>
          <p className="monitoring-chart-caption prediction-compare__caption">
            Swipe between raw imagery and the same frame with submission polygons overlaid to
            inspect how model output lines up with visual evidence.
          </p>
        </div>
        <div className="prediction-compare__meta">
          <span className="prediction-compare__pill">{submissionLabel}</span>
          <span className="prediction-compare__pill">
            {preview ? `${returnedFeatures}/${totalFeatures} features` : `${returnedFeatures} features`}
          </span>
          <span className="prediction-compare__pill">
            {visiblePolygons.length} in tile
          </span>
          {formattedTimeStep && (
            <span className="prediction-compare__pill prediction-compare__pill--accent">
              Model period {formattedTimeStep}
            </span>
          )}
        </div>
      </div>

      <div className="prediction-compare__controls">
        <div className="prediction-compare__toggle" role="tablist" aria-label="Imagery source">
          <button
            type="button"
            className={mode === 'optical' ? 'is-active' : ''}
            onClick={() => setMode('optical')}
          >
            Optical
          </button>
          <button
            type="button"
            className={mode === 'sar' ? 'is-active' : ''}
            onClick={() => setMode('sar')}
          >
            SAR
          </button>
        </div>

        <label className="prediction-compare__field">
          <span>Month</span>
          <select
            value={selectedFrame ? frameKey(selectedFrame) : ''}
            onChange={(event) =>
              setSelectedFrames((current) => ({ ...current, [mode]: event.target.value }))
            }
            disabled={!frames.length}
          >
            {frames.map((frame) => (
              <option key={frameKey(frame)} value={frameKey(frame)}>
                {frame.month}
              </option>
            ))}
          </select>
        </label>

        <label className="prediction-compare__field prediction-compare__field--slider">
          <span>Swipe</span>
          <input
            type="range"
            min={0}
            max={100}
            value={reveal}
            onChange={(event) => setReveal(Number(event.target.value))}
            disabled={!hasFrame}
          />
        </label>
      </div>

      <div className="prediction-compare__stage">
        {statusMessage ? (
          <p
            className={`prediction-compare__status${
              activeFrames.status === 'error' ? ' prediction-compare__status--error' : ''
            }`}
          >
            {statusMessage}
          </p>
        ) : selectedFrame ? (
          <div className="prediction-compare__viewer">
            <div className="prediction-compare__label prediction-compare__label--left">
              Prediction overlay
            </div>
            <div className="prediction-compare__label prediction-compare__label--right">
              Raw {mode === 'optical' ? 'optical' : 'SAR'}
            </div>
            <ApiBackedImg
              className="prediction-compare__image"
              src={selectedFrame.preview_url}
              alt={`${mode === 'optical' ? 'Optical' : 'SAR'} preview for ${selectedFrame.month}`}
            />
            <div className="prediction-compare__overlay" style={{ width: `${reveal}%` }}>
              <ApiBackedImg
                className="prediction-compare__image"
                src={selectedFrame.preview_url}
                alt=""
                aria-hidden
              />
              <svg
                className="prediction-compare__svg"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {overlayPolygons.map((points, index) => (
                  <polygon key={`${index}-${points.slice(0, 24)}`} points={points} />
                ))}
              </svg>
            </div>
            <div className="prediction-compare__divider" style={{ left: `${reveal}%` }} />
          </div>
        ) : null}
      </div>

      <div className="prediction-compare__footer">
        <p className="prediction-compare__detail">
          {activeFrames.data?.detail ??
            'Preview imagery uses the same tile crop as the monthly monitoring strips.'}
        </p>
        {!visiblePolygons.length && (
          <p className="prediction-compare__detail prediction-compare__detail--warning">
            No submission polygons intersect the selected tile after filtering the global
            submission file to a 10 km tile footprint.
          </p>
        )}
      </div>
    </section>
  )
}
