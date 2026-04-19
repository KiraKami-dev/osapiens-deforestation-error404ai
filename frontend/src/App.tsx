import { useCallback, useState } from 'react'
import { CanvasErrorBoundary } from './components/CanvasErrorBoundary'
import { SubmissionGlobeMap } from './components/SubmissionGlobeMap'
import { Header } from './components/Header'
import { FooterHint } from './components/FooterHint'
import { RegionModal } from './components/RegionModal'
import { Toast } from './components/Toast'
import { MonitoringView } from './components/MonitoringView'
import { PipelinePage } from './components/PipelinePage'
import type { Region } from './types/region'
import './App.css'

type Screen = 'globe' | 'analyze' | 'pipeline'

function App() {
  const [screen, setScreen] = useState<Screen>('globe')
  const [analyzeRegion, setAnalyzeRegion] = useState<Region | null>(null)
  const [region, setRegion] = useState<Region | null>(null)
  const [regionModalOpen, setRegionModalOpen] = useState(false)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  const handleGlobeSelect = useCallback((r: Region) => {
    setRegion(r)
    setRegionModalOpen(true)
  }, [])

  const closeRegionModal = useCallback(() => {
    setRegionModalOpen(false)
    setRegion(null)
  }, [])

  const handleRegionConfirm = useCallback(() => {
    if (!region) return
    const snapshot = { lat: region.lat, lng: region.lng }
    closeRegionModal()
    setAnalyzeRegion(snapshot)
    setScreen('analyze')
  }, [region, closeRegionModal])

  const handleBackFromAnalyze = useCallback(() => {
    setScreen('globe')
    setAnalyzeRegion(null)
  }, [])

  return (
    <div className="app-root">
      {screen === 'globe' && (
        <>
          <div className="app-bg-gradient" aria-hidden="true" />
          <CanvasErrorBoundary
            fallback={
              <div className="globe-fallback glass-panel">
                <p className="globe-fallback-title">Globe unavailable</p>
                <p className="globe-fallback-copy">
                  WebGL failed to initialize MapLibre. Try refreshing, updating your
                  browser, or disabling browser extensions that block WebGL.
                </p>
              </div>
            }
          >
            <SubmissionGlobeMap onSelectRegion={handleGlobeSelect} />
          </CanvasErrorBoundary>

          <div className="app-ui">
            <Header onUpload={() => setScreen('pipeline')} />
          </div>

          <FooterHint />

          {regionModalOpen && region && (
            <RegionModal
              region={region}
              onClose={closeRegionModal}
              onConfirm={handleRegionConfirm}
            />
          )}

          {toastMessage && (
            <Toast
              message={toastMessage}
              onDismiss={() => setToastMessage(null)}
            />
          )}

          <button
            type="button"
            className="fab-upload glass-panel"
            onClick={() => setScreen('pipeline')}
            aria-label="Upload satellite image"
            title="Upload satellite image"
          >
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden>
              <path
                d="M12 15V4m0 0l4 4m-4-4L8 8M5 19h14"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </>
      )}

      {screen === 'analyze' && analyzeRegion && (
        <MonitoringView onBack={handleBackFromAnalyze} region={analyzeRegion} />
      )}

      {screen === 'pipeline' && <PipelinePage onBack={() => setScreen('globe')} />}
    </div>
  )
}

export default App
