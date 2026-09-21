/**
 * Batch endpoints (contract: api/batch_models.py docstring). Same-origin
 * only; the access token goes in the `X-Access-Token` header on every call,
 * including the XHR upload and the two CSV downloads.
 */

import type { ErrorDetail } from "../lib/errors";
import { ApiError, TOKEN_HEADER, downloadWithToken, parseRetryAfter, send } from "./client";
import type { BatchItem, BatchStatus, PairingReport } from "./types";

const batchPath = (id: string) => `/api/batch/${encodeURIComponent(id)}`;

/** Saved from script so the token can travel as a header, not in the link. */
export const downloadTemplate = (token: string) =>
  downloadWithToken("/api/batch/template.csv", token, "label-manifest-template.csv");
export const downloadResults = (batchId: string, token: string) =>
  downloadWithToken(`${batchPath(batchId)}/results.csv`, token, `batch-${batchId.slice(0, 8)}-results.csv`);

/**
 * Pair a manifest with filenames only — no image bytes — so a typo is found
 * before a gigabyte upload, not after. Fields: `manifest` (file) and
 * `filenames` (repeated).
 */
export function checkPairing(manifest: File, filenames: string[], token: string): Promise<PairingReport> {
  const body = new FormData();
  body.append("manifest", manifest, manifest.name);
  for (const name of filenames) body.append("filenames", name);
  return send<PairingReport>("/api/batch/check", token, { method: "POST", body });
}

/** Summaries only: every item's `result` is null on a status poll. */
export function getBatch(batchId: string, token: string): Promise<BatchStatus> {
  return send<BatchStatus>(batchPath(batchId), token);
}

/** One label WITH its full result (null while it is still pending). */
export function getItem(batchId: string, rowNumber: number, token: string): Promise<BatchItem> {
  return send<BatchItem>(`${batchPath(batchId)}/items/${rowNumber}`, token);
}

/** Cancel: nothing new starts; finished results and the CSV are kept. */
export function cancelBatch(batchId: string, token: string): Promise<BatchStatus> {
  return send<BatchStatus>(batchPath(batchId), token, { method: "DELETE" });
}

export interface UploadHandle {
  promise: Promise<BatchStatus>;
  abort: () => void;
}

/**
 * Upload with real byte progress. XHR rather than fetch because fetch still
 * has no portable upload-progress event, and 300 photos over a government
 * network can take minutes — a frozen screen reads as broken.
 */
export function submitBatch(
  manifest: File,
  images: File[],
  token: string,
  onProgress: (sent: number, total: number) => void,
): UploadHandle {
  const body = new FormData();
  body.append("manifest", manifest, manifest.name);
  // The third argument keeps the original filename even for a downscaled
  // Blob, so pairing with the manifest still matches.
  for (const image of images) body.append("images", image, image.name);

  const xhr = new XMLHttpRequest();
  const promise = new Promise<BatchStatus>((resolve, reject) => {
    xhr.open("POST", "/api/batch");
    if (token) xhr.setRequestHeader(TOKEN_HEADER, token);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      const response = xhr.response as (BatchStatus & { detail?: ErrorDetail }) | null;
      if (xhr.status >= 200 && xhr.status < 300 && response) resolve(response);
      else
        reject(
          new ApiError(xhr.status, response?.detail ?? null, parseRetryAfter(xhr.getResponseHeader("Retry-After"))),
        );
    };
    xhr.onerror = () => reject(new ApiError(null, null));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    xhr.send(body);
  });
  return { promise, abort: () => xhr.abort() };
}

/** The submit endpoint's 422 carries the pairing report inside `detail`. */
export function pairingFromDetail(detail: ErrorDetail): PairingReport | null {
  if (detail && typeof detail === "object" && !Array.isArray(detail) && "pairing" in detail && detail.pairing) {
    return detail.pairing as PairingReport;
  }
  return null;
}

/**
 * The batch calls as one object, so the dev-only mock (src/dev/batchMock.ts)
 * can stand in for the server without the components knowing.
 */
export interface BatchApi {
  check: typeof checkPairing;
  submit: typeof submitBatch;
  get: typeof getBatch;
  getItem: typeof getItem;
  cancel: typeof cancelBatch;
  downloadTemplate: typeof downloadTemplate;
  downloadResults: typeof downloadResults;
}

export const realBatchApi: BatchApi = {
  check: checkPairing,
  submit: submitBatch,
  get: getBatch,
  getItem,
  cancel: cancelBatch,
  downloadTemplate,
  downloadResults,
};
