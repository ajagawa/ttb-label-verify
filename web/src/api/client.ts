/**
 * API calls. Same-origin paths only: the page is served by the API process in
 * production and proxied by Vite in development. No other host is ever
 * contacted — the target network blocks outbound domains.
 *
 * The access token travels in the `X-Access-Token` header, never in a query
 * string: query strings end up in server and proxy logs. It is still READ from
 * the page URL (`?token=`), which is how demo links hand it out.
 */

import type { ErrorDetail } from "../lib/errors";
import type { ApplicationForm, HealthResponse, RulesetMetadata, VerificationResult } from "./types";

export const TOKEN_HEADER = "X-Access-Token";

export class ApiError extends Error {
  /** null when the request never got a response (network failure). */
  readonly status: number | null;
  readonly detail: ErrorDetail;
  /** Seconds from a `Retry-After` header, when the server sent one. */
  readonly retryAfter: number | null;

  constructor(status: number | null, detail: ErrorDetail, retryAfter: number | null = null) {
    super(typeof detail === "string" ? detail : `HTTP ${status ?? "network error"}`);
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

/** The shared-secret token, if the page was opened with `?token=...`. */
export function tokenFromLocation(search: string = window.location.search): string {
  return new URLSearchParams(search).get("token") ?? "";
}

const TOKEN_STORAGE_KEY = "label-verify.access-code";

/**
 * The code to start with: from the link if it carries one, otherwise the one
 * last entered in this browser. Asked for once per
 * browser, not once per visit — re-typing a code on every reload was the
 * first thing a real user hit. It is a shared demo code, not a credential
 * tied to a person; production sits behind the agency's own sign-in.
 * Storage can be unavailable (private windows, blocked site data), so every
 * access is guarded and the app works without it.
 */
export function initialToken(search: string = window.location.search): string {
  const fromLink = tokenFromLocation(search);
  if (fromLink) {
    rememberToken(fromLink);
    dropTokenFromAddressBar();
    return fromLink;
  }
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

/**
 * Take `?token=` out of the address bar once it has been read. Left there, the
 * code sits in browser history and bookmarks, and in any link copied from the
 * page — including the `?batch=` reload link (security review).
 */
function dropTokenFromAddressBar(): void {
  try {
    const url = new URL(window.location.href);
    if (!url.searchParams.has("token")) return;
    url.searchParams.delete("token");
    window.history.replaceState(window.history.state, "", url.toString());
  } catch {
    /* no history API (tests, very old browsers): nothing to tidy */
  }
}

export function rememberToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* storage unavailable: the code lasts for this page only */
  }
}

export function authHeaders(token: string): Record<string, string> {
  return token ? { [TOKEN_HEADER]: token } : {};
}

/** `Retry-After` in seconds; the HTTP-date form is accepted too. */
export function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;
  const seconds = Number(value.trim());
  if (Number.isFinite(seconds) && seconds >= 0) return Math.round(seconds);
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, Math.round((date - Date.now()) / 1000));
}

async function errorFrom(response: Response): Promise<ApiError> {
  let detail: ErrorDetail = null;
  try {
    detail = ((await response.json()) as { detail?: ErrorDetail }).detail;
  } catch {
    // Non-JSON error body (e.g. a proxy page). The status alone is enough.
  }
  return new ApiError(response.status, detail, parseRetryAfter(response.headers.get("Retry-After")));
}

async function request(input: string, token: string, init: RequestInit = {}): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(input, { ...init, headers: { ...authHeaders(token), ...(init.headers ?? {}) } });
  } catch {
    throw new ApiError(null, null);
  }
  if (!response.ok) throw await errorFrom(response);
  return response;
}

export async function send<T>(input: string, token: string, init?: RequestInit): Promise<T> {
  return (await (await request(input, token, init)).json()) as T;
}

/**
 * Download a file that needs the access header. A plain <a href> cannot send
 * headers, and putting the token in the link would leak it into logs, so the
 * file is fetched as a blob and saved from script.
 */
export async function downloadWithToken(path: string, token: string, fallbackName: string): Promise<void> {
  const response = await request(path, token);
  const blob = await response.blob();
  const name = filenameFromDisposition(response.headers.get("Content-Disposition")) ?? fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a moment to start the save before releasing the blob.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header);
  if (star?.[1]) return decodeURIComponent(star[1].trim().replace(/^"|"$/g, ""));
  const plain = /filename="?([^";]+)"?/i.exec(header);
  return plain?.[1]?.trim() ?? null;
}

export function getRuleset(): Promise<RulesetMetadata> {
  return send<RulesetMetadata>("/api/ruleset", "");
}

export function getHealth(): Promise<HealthResponse> {
  return send<HealthResponse>("/api/health", "");
}

export function verifyLabel(form: ApplicationForm, image: File, token: string): Promise<VerificationResult> {
  const body = new FormData();
  body.append("image", image);
  body.append("brand_name", form.brand_name.trim());
  body.append("class_type", form.class_type.trim());
  body.append("alcohol_content", form.alcohol_content.trim());
  body.append("net_contents", form.net_contents.trim());
  body.append("beverage_class", form.beverage_class);
  // Omitted entirely when blank: the server treats absence as "not supplied"
  // and reports the type-size check as not evaluable.
  if (form.label_width_mm.trim()) body.append("label_width_mm", form.label_width_mm.trim());
  return send<VerificationResult>("/api/verify", token, { method: "POST", body });
}
