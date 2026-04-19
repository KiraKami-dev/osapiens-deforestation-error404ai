import { useCallback, useEffect, useRef, useState } from 'react'

type UploadPanelProps = {
  onClose: () => void
}

type Phase = 'idle' | 'uploading' | 'done'

const ACCEPT =
  '.tif,.tiff,.geotiff,.gtiff,.jp2,.j2k,.png,.jpg,.jpeg,.webp,.nc'

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export function UploadPanel({ onClose }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const progressTimerRef = useRef<number | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState(0)
  const [fileName, setFileName] = useState<string | null>(null)
  const [fileSize, setFileSize] = useState<number | null>(null)
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    return () => {
      if (progressTimerRef.current != null) {
        window.clearInterval(progressTimerRef.current)
      }
    }
  }, [])

  const simulateUpload = useCallback((file: File) => {
    if (progressTimerRef.current != null) {
      window.clearInterval(progressTimerRef.current)
    }
    setPhase('uploading')
    setProgress(0)
    setFileName(file.name)
    setFileSize(file.size)

    const start = performance.now()
    const duration = 1400 + Math.min(file.size / (1024 * 1024), 6) * 400

    progressTimerRef.current = window.setInterval(() => {
      const t = Math.min(1, (performance.now() - start) / duration)
      setProgress(Math.round(t * 100))
      if (t >= 1 && progressTimerRef.current != null) {
        window.clearInterval(progressTimerRef.current)
        progressTimerRef.current = null
        setPhase('done')
      }
    }, 40)
  }, [])

  const onFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (!file) return
      simulateUpload(file)
    },
    [simulateUpload],
  )

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="modal modal-wide glass-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-modal-title"
      >
        <div className="modal-head-row">
          <h2 id="upload-modal-title" className="modal-title">
            Upload satellite imagery
          </h2>
          <button
            type="button"
            className="icon-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <p className="modal-copy">
          GeoTIFF, Cloud-Optimized GeoTIFF, JPEG 2000, or high-resolution raster
          exports. Files stay in your browser for this demo.
        </p>

        <div
          className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${
            phase === 'uploading' ? 'dropzone-busy' : ''
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            if (phase === 'uploading') return
            onFiles(e.dataTransfer.files)
          }}
          onClick={() => {
            if (phase === 'idle') inputRef.current?.click()
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              if (phase === 'idle') inputRef.current?.click()
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Drop satellite image files here or browse"
        >
          <input
            ref={inputRef}
            type="file"
            className="sr-only"
            accept={ACCEPT}
            onChange={(e) => {
              onFiles(e.target.files)
              e.target.value = ''
            }}
          />
          {phase === 'idle' && (
            <>
              <span className="dropzone-icon" aria-hidden="true">
                <svg viewBox="0 0 48 48" width="40" height="40" fill="none">
                  <path
                    d="M14 30l10-14 10 14M24 16v20"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <rect
                    x="6"
                    y="36"
                    width="36"
                    height="6"
                    rx="2"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                </svg>
              </span>
              <p className="dropzone-title">Drag & drop your image</p>
              <p className="dropzone-sub">or click to browse supported formats</p>
            </>
          )}
          {phase === 'uploading' && (
            <div className="upload-status">
              <p className="upload-file-name">{fileName}</p>
              {fileSize != null && (
                <p className="upload-file-meta">{formatBytes(fileSize)}</p>
              )}
              <div className="progress-track" aria-hidden="true">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="upload-progress-label">Uploading… {progress}%</p>
            </div>
          )}
          {phase === 'done' && (
            <div className="upload-status">
              <p className="upload-done">Upload complete</p>
              <p className="upload-file-name">{fileName}</p>
              <p className="modal-copy">
                Imagery is staged for analysis. Connect a backend to process
                tiles.
              </p>
            </div>
          )}
        </div>

        <div className="modal-actions modal-actions-end">
          {phase === 'done' ? (
            <button type="button" className="btn-primary" onClick={onClose}>
              Done
            </button>
          ) : (
            <button
              type="button"
              className="btn-secondary"
              onClick={onClose}
              disabled={phase === 'uploading'}
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
