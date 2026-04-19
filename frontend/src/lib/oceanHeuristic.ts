/**
 * Classify a Blue Marble–style RGB sample as open water vs land / shallow / ice.
 * Tuned for NASA-style true-color equirectangular imagery (not a GIS guarantee).
 */
export function isLikelyOcean(r255: number, g255: number, b255: number): boolean {
  const r = r255 / 255
  const g = g255 / 255
  const b = b255 / 255
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

  // Bright clouds, snow, sun glint — treat as non-ocean so we never silently block ambiguous picks.
  if (lum > 0.78 && r > 0.65 && g > 0.65 && b > 0.65) return false

  // Deep / mid-latitude ocean: blue dominates.
  if (b > r + 0.07 && b > g + 0.045 && b > 0.3 && lum < 0.62) return true

  // Tropical / greener water.
  if (b > 0.24 && g > 0.28 && b > r + 0.02 && g > r + 0.05 && r < 0.42 && lum < 0.58)
    return true

  // Darker water bodies (lakes, shadows on water) — still clearly water-biased.
  if (lum < 0.38 && b >= r && b >= g && b > 0.18) return true

  return false
}
