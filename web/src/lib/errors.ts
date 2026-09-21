/**
 * Turn an HTTP failure into something an agent can act on.
 *
 * Requirement: errors are shown in plain language next to the upload, never as
 * raw JSON. Each message says what happened and what to do next. The server's
 * own `detail` is kept only where it is already written for a person (400) or
 * adds specifics the agent can use (the reason an image could not be read).
 */

export interface FriendlyError {
  /** Short, plain headline. */
  title: string;
  /** What to do about it. */
  message: string;
  /** 401: show the access-code field. */
  needsToken: boolean;
}

/**
 * FastAPI puts a string or a list of pydantic errors in `detail`. The batch
 * endpoints also use objects: 503 is `{ code, message }` and the submit 422
 * is `{ message, pairing }` (api/batch_models.py docstring).
 */
export type ErrorDetail =
  | string
  | Array<{ loc?: unknown[]; msg?: string }>
  | { code?: string; message?: string; pairing?: unknown }
  | null
  | undefined;

/** Batch 503 codes, from the contract. */
export type BatchErrorCode = "extraction_unavailable" | "batch_capacity" | "batch_unavailable";

function objectDetail(detail: ErrorDetail): { code?: string; message?: string } | null {
  return detail && typeof detail === "object" && !Array.isArray(detail) ? detail : null;
}

/** The machine-readable code, when the server sent one. */
export function errorCode(detail: ErrorDetail): string | null {
  const code = objectDetail(detail)?.code;
  return typeof code === "string" ? code : null;
}

/** The server's own plain-language message, from either detail shape. */
export function errorMessage(detail: ErrorDetail): string | null {
  if (typeof detail === "string") return detail;
  const message = objectDetail(detail)?.message;
  return typeof message === "string" ? message : null;
}

/**
 * Whether a batch failure is the "service is full, retry later" case. The
 * code decides; message text is only a fallback for a server that predates
 * the codes.
 */
export function isCapacityError(status: number | null, detail: ErrorDetail): boolean {
  if (status !== 503) return false;
  const code = errorCode(detail);
  if (code) return code === "batch_capacity";
  const text = errorMessage(detail);
  return !!text && !text.startsWith("Extraction is unavailable") && /batch/i.test(text);
}

const FIELD_NAMES: Record<string, string> = {
  brand_name: "Brand name",
  class_type: "Class/type",
  alcohol_content: "Alcohol content",
  net_contents: "Net contents",
  beverage_class: "Beverage class",
  label_width_mm: "Label width",
  image: "Label image",
};

function describeValidation(detail: ErrorDetail): string | null {
  if (!Array.isArray(detail) || detail.length === 0) return null;
  const parts = detail.map((item) => {
    const loc = Array.isArray(item.loc) ? item.loc : [];
    const key = String(loc[loc.length - 1] ?? "");
    const name = FIELD_NAMES[key] ?? "A form value";
    return `${name}: ${item.msg ?? "is not valid"}`;
  });
  return parts.join(". ");
}

/** Which screen the error belongs to; the advice differs. */
export type ErrorContext = "single" | "batch";

export function friendlyError(
  status: number | null,
  detail?: ErrorDetail,
  context: ErrorContext = "single",
): FriendlyError {
  const text = typeof detail === "string" ? detail : null;
  const plain = (title: string, message: string): FriendlyError => ({
    title,
    message,
    needsToken: false,
  });

  if (context === "batch") {
    const batch = batchError(status, detail, errorMessage(detail), plain);
    if (batch) return batch;
  }

  switch (status) {
    case null:
      return plain(
        "Could not reach the server",
        "Check your network connection and try again. Your entries have been kept.",
      );
    case 401:
      return {
        title: "An access code is needed",
        message: "Enter the access code you were given, then press Check label again.",
        needsToken: true,
      };
    case 400:
      return plain("The upload could not be used", text ?? "Please check the image and form, then try again.");
    case 413:
      return plain(
        "This image file is too large",
        "The limit is 20 MB. Try a smaller photo or export the image at a lower resolution.",
      );
    case 415:
      return plain(
        "This file type is not supported",
        "Use a JPEG, PNG, WebP, TIFF or BMP image. PDF artwork is not handled by this prototype.",
      );
    case 422: {
      const validation = describeValidation(detail);
      if (validation) return plain("Please check the form", validation);
      return plain(
        "The tool could not read this image",
        "Try a sharper, well-lit photo taken straight on." + (text ? ` (Details: ${text})` : ""),
      );
    }
    case 503:
      return plain(
        "Automatic label reading is unavailable",
        "The server is running without its text-reading engine, so labels cannot be checked right now. " +
          "This is not a problem with your label. Please tell your administrator.",
      );
    default:
      return plain(
        "Something went wrong on the server",
        `Please try again. If it keeps happening, tell your administrator (error ${status}).`,
      );
  }
}

/**
 * Batch-specific wording. The batch endpoints write their `detail` strings for
 * people ("Split the submission into smaller batches."), so those are shown
 * as the advice; only the headline is ours. Returns null to fall through to
 * the shared mapping (401, network failure, 415, unknown statuses).
 */
function batchError(
  status: number | null,
  detail: ErrorDetail,
  text: string | null,
  plain: (title: string, message: string) => FriendlyError,
): FriendlyError | null {
  switch (status) {
    case 404:
      return plain(
        "This batch has expired or was stopped",
        "The server no longer holds it. If you downloaded the results CSV, that is your record. " +
          "Otherwise choose the files again and start a new batch.",
      );
    case 400:
      return plain("The batch could not be started", text ?? "Check the manifest and images, then try again.");
    case 413:
      return plain(
        "This batch is too large",
        text ?? "Split the submission into smaller batches and try again.",
      );
    case 422:
      if (Array.isArray(detail)) return null; // pydantic validation: shared wording
      return plain(
        "Nothing in this batch could be checked",
        text ?? "The manifest and images do not line up. See the issues listed.",
      );
    case 503: {
      // Three 503s, told apart by `code` (message text only as a fallback).
      if (isCapacityError(status, detail)) {
        return plain(
          "The server is busy with other batches",
          "Your files are still chosen. The tool will try again automatically; you do not need to do anything.",
        );
      }
      if (errorCode(detail) === "batch_unavailable") {
        return plain(
          "Checking many labels is not available on this server",
          "Batch checking is not set up here. Single labels may still work. Please tell your administrator.",
        );
      }
      return plain(
        "Automatic label reading is unavailable",
        "The server is running without its text-reading engine, so labels cannot be checked right now. " +
          "Nothing was lost — your files are still selected. Please tell your administrator.",
      );
    }
    default:
      return null;
  }
}
