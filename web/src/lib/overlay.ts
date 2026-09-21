/**
 * Bounding-box overlay geometry.
 *
 * The server downscales every upload (longest edge <= 1600 px) and applies
 * EXIF orientation before reading it, so a bbox is in the PREPROCESSED image's
 * pixel space, not the original file's. The browser shows the original file.
 * Both are the same picture at different sizes, so the only coordinate system
 * they share is a fraction of the image: we convert to percentages of
 * image_width / image_height from the response and let CSS do the rest. That
 * also keeps the highlight correct at any zoom level or window width.
 */

import type { BoxModel } from "../api/types";

export interface PercentBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

/** Convert a server bbox to percentages of the displayed image, clamped to it. */
export function bboxToPercent(box: BoxModel, imageWidth: number, imageHeight: number): PercentBox | null {
  if (!(imageWidth > 0) || !(imageHeight > 0)) return null;
  const left = clamp((box.left / imageWidth) * 100, 0, 100);
  const top = clamp((box.top / imageHeight) * 100, 0, 100);
  const right = clamp(((box.left + box.width) / imageWidth) * 100, 0, 100);
  const bottom = clamp(((box.top + box.height) / imageHeight) * 100, 0, 100);
  if (right <= left || bottom <= top) return null;
  return { left, top, width: right - left, height: bottom - top };
}

/** Percentages as inline CSS for an absolutely-positioned overlay element. */
export function percentStyle(p: PercentBox): { left: string; top: string; width: string; height: string } {
  const pct = (n: number) => `${n.toFixed(3)}%`;
  return { left: pct(p.left), top: pct(p.top), width: pct(p.width), height: pct(p.height) };
}

/**
 * True when the picture the browser shows is not the same shape as the one the
 * server read — e.g. a browser that ignored EXIF rotation. Percent positions
 * would then point at the wrong place, and an evidence highlight that points
 * at the wrong place is worse than none, so the UI warns instead of pretending.
 */
export function aspectMismatch(
  naturalWidth: number,
  naturalHeight: number,
  serverWidth: number,
  serverHeight: number,
  tolerance = 0.02,
): boolean {
  if (!(naturalWidth > 0 && naturalHeight > 0 && serverWidth > 0 && serverHeight > 0)) return false;
  const shown = naturalWidth / naturalHeight;
  const read = serverWidth / serverHeight;
  return Math.abs(shown - read) / read > tolerance;
}

export interface CropView {
  /** Width / height of the cropped region, for CSS aspect-ratio. */
  aspect: number;
  /** Width of the cropped region in server pixels, to cap magnification. */
  sourceWidth: number;
  /** Inline styles for the full image inside the crop window, in percent. */
  img: { width: string; left: string; top: string };
}

/**
 * Geometry for showing just the region a value was read from, enlarged, next
 * to the reading itself. The full image sits inside a window the shape of the
 * (padded) box, scaled and shifted so only that region shows. Everything is
 * relative, so it too survives any zoom level.
 *
 * `top` in CSS is a percentage of the containing block's HEIGHT and `left` of
 * its WIDTH, which is exactly what makes the offsets below independent of the
 * rendered size.
 */
export function cropView(box: BoxModel, imageWidth: number, imageHeight: number): CropView | null {
  if (!(imageWidth > 0) || !(imageHeight > 0)) return null;
  // Pad by half the box height (6–20 px) so the reading keeps a little
  // context without a large block drowning in its surroundings.
  const pad = Math.max(6, Math.min(box.height, 40) * 0.5);
  const x0 = Math.max(0, box.left - pad);
  const y0 = Math.max(0, box.top - pad);
  const x1 = Math.min(imageWidth, box.left + box.width + pad);
  const y1 = Math.min(imageHeight, box.top + box.height + pad);
  const cw = x1 - x0;
  const ch = y1 - y0;
  if (cw <= 0 || ch <= 0) return null;
  const pct = (n: number) => `${n.toFixed(3)}%`;
  return {
    aspect: cw / ch,
    sourceWidth: cw,
    img: {
      width: pct((imageWidth / cw) * 100),
      left: pct((-x0 / cw) * 100),
      top: pct((-y0 / ch) * 100),
    },
  };
}
