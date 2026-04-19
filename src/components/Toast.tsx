import { useEffect } from 'react'

type ToastProps = {
  message: string
  onDismiss: () => void
}

export function Toast({ message, onDismiss }: ToastProps) {
  useEffect(() => {
    const t = window.setTimeout(onDismiss, 4200)
    return () => window.clearTimeout(t)
  }, [onDismiss])

  return (
    <div className="toast glass-panel" role="status">
      {message}
    </div>
  )
}
