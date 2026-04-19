import smallLogo from '../assets/small_logo.png'

type HeaderProps = {
  onUpload: () => void
}

export function Header({ onUpload }: HeaderProps) {
  return (
    <header className="site-header glass-panel">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <img src={smallLogo} alt="" className="brand-logo" />
        </span>
        <div className="brand-text">
          <span className="brand-title">ForestWatch AI</span>
          <span className="brand-sub">Global Deforestation Monitoring</span>
        </div>
      </div>
      <nav className="header-actions" aria-label="Primary">
        <button type="button" className="glass-btn" onClick={onUpload}>
          <span className="glass-btn-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
              <path
                d="M12 15V4m0 0l4 4m-4-4L8 8M5 19h14"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          Upload Image
        </button>
      </nav>
    </header>
  )
}
