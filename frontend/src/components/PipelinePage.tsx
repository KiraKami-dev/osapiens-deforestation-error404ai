import { useEffect, useMemo, useRef, useState } from 'react'

type PipelinePageProps = {
  onBack: () => void
}

type PipelineStage = 'idle' | 'uploading' | 'training' | 'complete'

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function progressLabel(stage: PipelineStage) {
  if (stage === 'idle') return 'Waiting for analysis'
  if (stage === 'uploading') return 'Uploading dataset'
  if (stage === 'training') return 'Training and geolocating'
  return 'Pipeline complete'
}

export function PipelinePage({ onBack }: PipelinePageProps) {
  const uploadTimerRef = useRef<number | null>(null)
  const timeoutRefs = useRef<number[]>([])
  const [showSplash, setShowSplash] = useState(true)
  const [stage, setStage] = useState<PipelineStage>('idle')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadNotice, setUploadNotice] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const datasetName = selectedFile?.name ?? null
  const datasetSize = selectedFile?.size ?? null
  const analyzing = stage === 'uploading' || stage === 'training'

  const timeline = useMemo(
    () => [
      {
        id: 'ingest',
        title: '1) Data Ingestion',
        desc: 'User uploads satellite tiles and geospatial datasets.',
        done: stage !== 'idle',
      },
      {
        id: 'train',
        title: '2) Model Training',
        desc: 'Pipeline cleans data, trains model, and checks deforestation signals.',
        done: stage === 'training' || stage === 'complete',
      },
      {
        id: 'map',
        title: '3) Globe Mapping',
        desc: 'A geospatial point is added on the Earth globe.',
        done: stage === 'complete',
      },
    ],
    [stage],
  )

  const clearPipelineTimers = () => {
    if (uploadTimerRef.current != null) {
      window.clearInterval(uploadTimerRef.current)
      uploadTimerRef.current = null
    }
    for (const id of timeoutRefs.current) {
      window.clearTimeout(id)
    }
    timeoutRefs.current = []
  }

  useEffect(() => {
    return () => clearPipelineTimers()
  }, [])

  const runDemoPipeline = (file: File) => {
    clearPipelineTimers()
    setSelectedFile(file)
    setStage('uploading')
    setProgress(0)

    uploadTimerRef.current = window.setInterval(() => {
      setProgress((prev) => {
        if (prev >= 45) {
          if (uploadTimerRef.current != null) {
            window.clearInterval(uploadTimerRef.current)
            uploadTimerRef.current = null
          }
          setStage('training')
          return 45
        }
        return prev + 5
      })
    }, 120)

    timeoutRefs.current.push(window.setTimeout(() => {
      setProgress(70)
    }, 1800))

    timeoutRefs.current.push(window.setTimeout(() => {
      setProgress(88)
    }, 2600))

    timeoutRefs.current.push(window.setTimeout(() => {
      setProgress(100)
      setStage('complete')
    }, 3500))
  }

  const handleUploadClick = () => {
    setUploadNotice('Upload dataset is coming in a future release.')
  }

  const handleAnalyze = () => {
    if (analyzing) return
    const demoFile = selectedFile ?? new File(['demo'], 'demo_dataset.geojson', { type: 'application/geo+json' })
    setUploadNotice(null)
    runDemoPipeline(demoFile)
  }

  if (showSplash) {
    return (
      <main className="pipeline-page">
        <section className="pipeline-splash glass-panel">
          <span className="pipeline-badge">Deforestation Pipeline</span>
          <h1>Welcome to Forest Analysis</h1>
          <p>
            Upload your dataset and run one-click analysis to detect high-risk
            deforestation areas and map them to the globe.
          </p>
          <div className="pipeline-splash-actions">
            <button type="button" className="btn-primary" onClick={() => setShowSplash(false)}>
              Start Workflow
            </button>
            <button type="button" className="glass-btn" onClick={onBack}>
              Back to Globe
            </button>
          </div>
        </section>
      </main>
    )
  }

  return (
    <main className="pipeline-page">
      <div className="pipeline-shell glass-panel">
        <header className="pipeline-head">
          <button type="button" className="glass-btn" onClick={onBack}>
            Back to Globe
          </button>
          <span className="pipeline-badge">Deforestation Pipeline</span>
        </header>

        <section className="pipeline-hero">
          <h1>Upload Forest Data, Train Model, Map Deforestation</h1>
          <p>
            This page turns your platform into an end-to-end deforestation
            workflow. Anyone can upload satellite or tabular data, trigger the
            model pipeline, and map results globally.
          </p>
          <p className="pipeline-highlight">
            Automatically place results on the globe.
          </p>
        </section>

        <section className="pipeline-grid">
          <article className="pipeline-card">
            <h2>Upload Data</h2>
            <p>Dataset uploads will be available soon.</p>
            <button
              type="button"
              className="btn-primary pipeline-upload-btn"
              onClick={handleUploadClick}
            >
              Upload Dataset
            </button>
            <button
              type="button"
              className="btn-secondary pipeline-analyze-btn"
              onClick={handleAnalyze}
              disabled={analyzing}
            >
              {analyzing ? 'Analyzing...' : 'Analyze'}
            </button>
            {uploadNotice && <p className="pipeline-meta">{uploadNotice}</p>}
            {datasetName && (
              <p className="pipeline-meta">
                {datasetName}
                {datasetSize != null ? ` - ${formatBytes(datasetSize)}` : ''}
              </p>
            )}
          </article>

          <article className="pipeline-card">
            <h2>Pipeline Status</h2>
            <p>{progressLabel(stage)}</p>
            <div className="progress-track" aria-hidden="true">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="pipeline-meta">{progress}% complete</p>
          </article>
        </section>

        <section className="pipeline-card pipeline-steps">
          <h2>Automation Flow</h2>
          {timeline.map((item) => (
            <div
              key={item.id}
              className={`pipeline-step ${item.done ? 'pipeline-step-done' : ''}`}
            >
              <p className="pipeline-step-title">{item.title}</p>
              <p className="pipeline-step-desc">{item.desc}</p>
            </div>
          ))}
        </section>

        <section className="pipeline-card">
          <h2>Predicted Globe Output</h2>
          {stage === 'complete' ? (
            <p>
              New deforestation marker generated (lat: -3.12, lng: -60.02) with
              high-risk confidence. You can now show this point directly on the
              Earth globe layer.
            </p>
          ) : (
            <p>
              Run a dataset through the pipeline to generate a new geolocated
              deforestation point.
            </p>
          )}
        </section>
        {analyzing && (
          <section className="pipeline-card pipeline-loading-card" aria-live="polite">
            <div className="loading-pulse" aria-hidden="true" />
            <p>
              Running analysis: {progressLabel(stage)} ({progress}%)
            </p>
          </section>
        )}
      </div>
    </main>
  )
}
