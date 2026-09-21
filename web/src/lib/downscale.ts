/**
 * Client-side downscale before a batch upload (batch mode only).
 *
 * Why: phone photos are often 4000+ px and 5–10 MB, and the server shrinks
 * them to 1600 px before reading anyway. Shrinking the big ones to 2600 px in
 * the browser cuts a 300-label upload by most of its bytes while leaving
 * headroom above what the server reads. Smaller images go up untouched, byte
 * for byte, so nothing is re-encoded that did not need to be.
 *
 * The decision logic is pure and unit-tested; the browser-specific decode and
 * encode steps are injected so tests do not need a canvas.
 */

export const MAX_EDGE_PX = 2600;
export const JPEG_QUALITY = 0.92;

export interface Size {
  width: number;
  height: number;
}

/**
 * The size to shrink to, or null when the image should go up untouched.
 * Dimensions are AFTER EXIF orientation (a portrait phone photo stored as
 * landscape pixels is measured as portrait), which only swaps the axes — the
 * longest edge is the same either way.
 */
export function targetSize(width: number, height: number, maxEdge = MAX_EDGE_PX): Size | null {
  if (!(width > 0 && height > 0)) return null;
  const longest = Math.max(width, height);
  if (longest <= maxEdge) return null;
  const scale = maxEdge / longest;
  // The longest edge is pinned to exactly maxEdge; the other is rounded and
  // never allowed to reach 0 for extreme panoramas.
  return width >= height
    ? { width: maxEdge, height: Math.max(1, Math.round(height * scale)) }
    : { width: Math.max(1, Math.round(width * scale)), height: maxEdge };
}

/** A decoded image, already in display orientation. */
export interface Decoded {
  width: number;
  height: number;
  close?: () => void;
}

export interface Codec<D extends Decoded = Decoded> {
  /** Decode with EXIF orientation applied. Reject if the browser can't decode it. */
  decode: (file: File) => Promise<D>;
  /** Draw at the given size and encode as JPEG. */
  encode: (decoded: D, size: Size, quality: number) => Promise<Blob>;
}

export type PrepareOutcome =
  | { kind: "untouched"; file: File; reason: "small" | "undecodable" | "not-smaller" }
  | { kind: "resized"; file: File; from: Size; to: Size };

/**
 * Prepare one image for upload.
 *
 * - Small enough: the original File object is returned as-is.
 * - Browser cannot decode it (TIFF outside Safari): sent untouched; the server
 *   decodes it and applies its own downscale, so nothing is lost but bytes.
 * - Otherwise: re-encoded as JPEG under the ORIGINAL filename, so the manifest
 *   pairing — which matches on filename — still lines up. The type becomes
 *   image/jpeg, which the server accepts regardless of the extension.
 */
export async function prepareImage<D extends Decoded>(file: File, codec: Codec<D>): Promise<PrepareOutcome> {
  let decoded: D;
  try {
    decoded = await codec.decode(file);
  } catch {
    return { kind: "untouched", file, reason: "undecodable" };
  }
  try {
    const from = { width: decoded.width, height: decoded.height };
    const to = targetSize(from.width, from.height);
    if (!to) return { kind: "untouched", file, reason: "small" };
    const blob = await codec.encode(decoded, to, JPEG_QUALITY);
    // Pathological case (already heavily compressed): keep the original.
    if (blob.size >= file.size) return { kind: "untouched", file, reason: "not-smaller" };
    const resized = new File([blob], file.name, { type: "image/jpeg", lastModified: file.lastModified });
    return { kind: "resized", file: resized, from, to };
  } catch {
    return { kind: "untouched", file, reason: "undecodable" };
  } finally {
    decoded.close?.();
  }
}

/**
 * The real browser codec. `imageOrientation: "from-image"` makes the bitmap
 * upright per EXIF, matching what the server does, and the re-encoded JPEG
 * carries no EXIF — so the server cannot rotate it a second time.
 */
export const browserCodec: Codec<ImageBitmap> = {
  decode: (file) => createImageBitmap(file, { imageOrientation: "from-image" }),
  encode: async (bitmap, size, quality) => {
    if (typeof OffscreenCanvas !== "undefined") {
      const canvas = new OffscreenCanvas(size.width, size.height);
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("no 2d context");
      paint(ctx, bitmap, size);
      return canvas.convertToBlob({ type: "image/jpeg", quality });
    }
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    paint(ctx, bitmap, size);
    return new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode failed"))), "image/jpeg", quality),
    );
  },
};

function paint(
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
  bitmap: ImageBitmap,
  size: Size,
) {
  // JPEG has no transparency; a transparent PNG would otherwise turn black.
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size.width, size.height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, size.width, size.height);
}
