import { useEffect, useState, type ImgHTMLAttributes } from 'react'
import { apiFetch, previewUrlNeedsBlobFetch } from '../lib/apiFetch'

type ApiBackedImgProps = {
  src: string
  alt: string
  className?: string
  loading?: ImgHTMLAttributes<HTMLImageElement>['loading']
  decoding?: ImgHTMLAttributes<HTMLImageElement>['decoding']
  'aria-hidden'?: ImgHTMLAttributes<HTMLImageElement>['aria-hidden']
}

/**
 * Ngrok free cannot send custom headers on `<img src>`. When the preview host is ngrok,
 * load bytes via `fetch` (with bypass header) and show a blob URL.
 */
export function ApiBackedImg({
  src,
  alt,
  className,
  loading,
  decoding,
  'aria-hidden': ariaHidden,
}: ApiBackedImgProps) {
  const [blobSrc, setBlobSrc] = useState<string | null>(() =>
    previewUrlNeedsBlobFetch(src) ? null : src,
  )

  useEffect(() => {
    if (!previewUrlNeedsBlobFetch(src)) {
      setBlobSrc(src)
      return
    }

    let cancelled = false
    let objectUrl: string | null = null
    setBlobSrc(null)

    void apiFetch(src)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.blob()
      })
      .then((blob) => {
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setBlobSrc(objectUrl)
      })
      .catch(() => {
        if (!cancelled) setBlobSrc('')
      })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [src])

  if (blobSrc === '') {
    return (
      <span className={className} role="img" aria-label={alt}>
        Preview failed to load (check API / ngrok).
      </span>
    )
  }

  if (!blobSrc) {
    return <span className={className} aria-busy="true" aria-label={`Loading ${alt}`} />
  }

  return (
    <img
      src={blobSrc}
      alt={alt}
      className={className}
      loading={loading}
      decoding={decoding}
      aria-hidden={ariaHidden}
    />
  )
}
