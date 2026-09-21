/** Small, display-only formatting helpers. */

/** "1.8 s". Always seconds: agents compare against the old tool's 30–40 s. */
export function formatSeconds(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const seconds = ms / 1000;
  return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds).toString()} s`;
}

/** 0.873 -> "87%". */
export function formatPercent(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return "—";
  return `${Math.round(fraction * 100)}%`;
}

/** "contrast_ratio" -> "Contrast ratio". */
export function humanizeKey(key: string): string {
  const words = key.replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"] as const;
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

/** Checked before upload so the agent hears about a wrong file immediately. */
export function checkFile(file: File | { type: string; size: number }): string | null {
  if (!(ACCEPTED_TYPES as readonly string[]).includes(file.type)) {
    return "That file type is not supported. Use a JPEG, PNG, WebP, TIFF or BMP image.";
  }
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_UPLOAD_BYTES) return "That image is larger than 20 MB. Try a smaller photo.";
  return null;
}
