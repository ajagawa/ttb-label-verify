/**
 * DEV ONLY — never part of the production bundle.
 *
 * A stand-in for the batch endpoints, loaded only behind
 * `import.meta.env.DEV && MOCK_MODE` (see src/lib/devMode.ts). It lets the
 * whole batch flow — pairing issues, upload progress, a status that advances
 * poll by poll, stop, reload-and-reattach, expiry — be exercised without the
 * backend. Progress is derived from wall-clock time since submit and kept in
 * sessionStorage, so a page reload reattaches exactly as with the server.
 */

import type { BatchApi, UploadHandle } from "../api/batch";
import { ApiError } from "../api/client";
import type { BatchItem, BatchStatus, FieldResult, PairingReport, VerificationResult } from "../api/types";
import { MOCK_RESULT } from "./mock";
import sampleLabelUrl from "./sample-label.svg?url";

const MS_PER_LABEL = 450;
const STORE_KEY = "label-verify-dev-batch";

interface Stored {
  id: string;
  names: string[];
  startedAt: number;
  cancelledAt: number | null;
}

function load(id: string): Stored | null {
  try {
    const raw = sessionStorage.getItem(STORE_KEY);
    const stored = raw ? (JSON.parse(raw) as Stored) : null;
    return stored && stored.id === id ? stored : null;
  } catch {
    return null;
  }
}

function save(stored: Stored | null) {
  try {
    if (stored) sessionStorage.setItem(STORE_KEY, JSON.stringify(stored));
    else sessionStorage.removeItem(STORE_KEY);
  } catch {
    /* dev only */
  }
}

const accepted = (n: string) => /\.(jpe?g|png|webp|tiff?|bmp)$/i.test(n);

/** The last image is always "not in the manifest", and one row names a missing file. */
function pairingFor(names: string[]): PairingReport {
  const images = names.filter(accepted);
  const unmatched = images[images.length - 1];
  const matched = Math.max(0, images.length - 1);
  return {
    matched,
    total_images: images.length,
    total_rows: matched + 1,
    blocking: matched === 0,
    issues: [
      {
        kind: "row_without_image",
        filename: "old_tom_reserve.jpg",
        row_number: matched + 2,
        message: `Row ${matched + 2} names old_tom_reserve.jpg, but no image with that name was selected.`,
      },
      ...(unmatched
        ? [
            {
              kind: "image_without_row" as const,
              filename: unmatched,
              row_number: null,
              message: `${unmatched} was selected, but no row in the manifest names it. It will be left out.`,
            },
          ]
        : []),
    ],
  };
}

// ---------------------------------------------------------------- results

const allMatch = (f: FieldResult): FieldResult => ({
  ...f,
  verdict: "MATCH",
  automated: true,
  found: f.field_id === "net_contents" ? "750 mL" : (f.found ?? f.expected),
  reason: `${f.label} appears to agree with the application.`,
  checks: f.checks.map((c) => (c.outcome === "fail" ? { ...c, outcome: "pass" as const } : c)),
  differences: [],
  bbox: f.bbox ?? { left: 120, top: 262, width: 560, height: 38 },
  strategy: f.strategy === "not_found" ? "vocabulary" : f.strategy,
});
const withFields = (fields: FieldResult[]): VerificationResult => ({ ...MOCK_RESULT, fields });

const CLEAR = withFields(MOCK_RESULT.fields.map(allMatch));
const ASSERTED = withFields(MOCK_RESULT.fields.map((f) => (f.field_id === "net_contents" ? f : allMatch(f))));
const REVIEW = withFields(MOCK_RESULT.fields.map((f) => (f.field_id === "government_warning" ? f : allMatch(f))));
const UNCHECKED = withFields(MOCK_RESULT.fields.map((f) => (f.field_id === "class_type" ? f : allMatch(f))));
const MISSING = withFields(
  MOCK_RESULT.fields.map((f) =>
    f.field_id === "government_warning"
      ? {
          ...f,
          verdict: "REVIEW" as const,
          strategy: "not_found" as const,
          elevated: true,
          found: null,
          bbox: null,
          reason: "No government warning statement was found, although the photo is clear.",
          checks: [],
          differences: [],
        }
      : allMatch(f),
  ),
);

type Variant = { tier: number; label: string; result: VerificationResult | null; error?: string };
const VARIANTS: Variant[] = [
  { tier: 5, label: "All clear", result: CLEAR },
  { tier: 0, label: "Differs from the application", result: ASSERTED },
  { tier: 5, label: "All clear", result: CLEAR },
  { tier: 3, label: "Needs a look", result: REVIEW },
  { tier: 1, label: "Mandatory statement not found", result: MISSING },
  { tier: 5, label: "All clear", result: CLEAR },
  {
    tier: 2,
    label: "Could not read this image",
    result: null,
    error: "Could not read this image: the photo is too blurred to read any text.",
  },
  { tier: 4, label: "Some items not checked automatically", result: UNCHECKED },
];

const BRANDS = ["OLD TOM DISTILLERY", "STONE'S THROW", "RIVER BEND", "COPPER HOLLOW", "NORTH FORK"];

function itemFor(name: string, index: number, finished: boolean): BatchItem {
  const base = {
    filename: name,
    row_number: index + 2,
    brand_name: BRANDS[index % BRANDS.length] ?? "OLD TOM DISTILLERY",
    mismatch_count: 0,
    elevated_count: 0,
    review_count: 0,
    unchecked_count: 0,
    headline: null,
    result: null,
    error: null,
  };
  if (!finished) return { ...base, state: "pending", tier: null, tier_label: null, needs_attention: null };
  const v = VARIANTS[index % VARIANTS.length]!;
  const r = v.result;
  const count = (pred: (f: FieldResult) => boolean) => (r ? r.fields.filter(pred).length : 0);
  const lead = r?.fields.find((f) => f.verdict !== "MATCH");
  return {
    ...base,
    state: r ? "done" : "error",
    tier: v.tier,
    tier_label: v.label,
    needs_attention: v.tier !== 5,
    mismatch_count: count((f) => f.verdict === "MISMATCH"),
    elevated_count: count((f) => f.verdict === "REVIEW" && f.elevated),
    review_count: count((f) => f.verdict === "REVIEW" && !f.elevated && f.automated),
    unchecked_count: count((f) => !f.automated),
    headline: r ? (lead ? `${lead.label}: ${lead.reason}` : null) : (v.error ?? null),
    result: r,
    error: v.error ?? null,
  };
}

/** Summaries only, as the real status endpoint now returns. */
function statusOf(stored: Stored): BatchStatus {
  const full = fullStatusOf(stored);
  return { ...full, items: full.items.map((item) => ({ ...item, result: null })) };
}

/** Labels read at the same time, as with the server's worker threads. */
const SLOTS = 2;

function fullStatusOf(stored: Stored): BatchStatus {
  const now = Date.now();
  const elapsed = Math.max(0, now - stored.startedAt);
  const total = stored.names.length;
  // Everything up to `limit` gets read. Running normally, that is every label;
  // after a stop, only those finished or already inside a slot at the moment
  // of the stop — they finish (and are kept), the rest become SKIPPED.
  const limit = stored.cancelledAt
    ? Math.min(total, Math.floor((stored.cancelledAt - stored.startedAt) / MS_PER_LABEL) + SLOTS)
    : total;
  const done = Math.min(limit, Math.floor(elapsed / MS_PER_LABEL));
  const runningEnd = Math.min(limit, done + SLOTS);
  const finished = stored.names.slice(0, done).map((n, i) => itemFor(n, i, true));
  const running = stored.names
    .slice(done, runningEnd)
    .map((n, i) => ({ ...itemFor(n, done + i, false), state: "running" as const }));
  const pending = stored.names.slice(runningEnd, limit).map((n, i) => itemFor(n, runningEnd + i, false));
  const skipped = stored.names
    .slice(limit)
    .map((n, i) => ({ ...itemFor(n, limit + i, false), state: "skipped" as const, needs_attention: true }));
  // Same ordering rule as rules/triage.sort_key (tier, then more problems, then filename).
  finished.sort((a, b) => {
    const problems = (x: BatchItem) => x.mismatch_count + x.elevated_count + x.review_count + x.unchecked_count;
    return (a.tier ?? 9) - (b.tier ?? 9) || problems(b) - problems(a) || a.filename.localeCompare(b.filename);
  });
  const state = stored.cancelledAt ? "cancelled" : done >= total ? "done" : "running";
  const rate = done > 0 ? elapsed / done : null;
  return {
    batch_id: stored.id,
    state,
    total,
    completed: done,
    failed: finished.filter((i) => i.state === "error").length,
    needs_attention: finished.filter((i) => i.needs_attention).length + skipped.length,
    created_at: new Date(stored.startedAt).toISOString(),
    started_at: new Date(stored.startedAt).toISOString(),
    finished_at:
      state === "done" ? new Date(stored.startedAt + total * MS_PER_LABEL).toISOString() : stored.cancelledAt
        ? new Date(stored.cancelledAt).toISOString()
        : null,
    elapsed_ms: state === "done" ? total * MS_PER_LABEL : elapsed,
    estimated_remaining_ms: state === "running" && rate != null ? rate * (total - done) : null,
    expires_at: new Date(stored.startedAt + 4 * 3600_000).toISOString(),
    pairing: pairingFor([...stored.names, "unmatched.jpg"]),
    ruleset_version: "ttb-v1",
    // Server order: finished worst-first, then running, pending, skipped.
    items: [...finished, ...running, ...pending, ...skipped],
  };
}

const delay = <T>(value: T, ms = 300) => new Promise<T>((r) => setTimeout(() => r(value), ms));

let busyServed = false;
/** Rows fetched through getItem, exposed for the Playwright cache/prefetch check. */
const fetchLog: number[] = [];
(window as unknown as { __devItemFetches?: number[] }).__devItemFetches = fetchLog;

function saveBlob(text: string, name: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export const mockBatchApi: BatchApi = {
  check: (_manifest, filenames) => delay(pairingFor(filenames)),

  submit: (_manifest, images, _token, onProgress): UploadHandle => {
    // ?mockbusy=1: the first submit is refused as "busy", with Retry-After 3.
    if (new URLSearchParams(location.search).has("mockbusy") && !busyServed) {
      busyServed = true;
      return {
        promise: delay(null, 200).then(() => {
          throw new ApiError(
            503,
            { code: "batch_capacity", message: "Other batches are still being processed." },
            3,
          );
        }),
        abort: () => {},
      };
    }
    const total = images.reduce((s, f) => s + f.size, 0) || 1;
    // Exposed for the Playwright check of the client-side downscale.
    (window as unknown as { __devUploaded?: unknown }).__devUploaded = images.map((f) => ({
      name: f.name,
      type: f.type,
      size: f.size,
    }));
    let timer = 0;
    let rejectFn: (e: unknown) => void = () => {};
    const promise = new Promise<BatchStatus>((resolve, reject) => {
      rejectFn = reject;
      let sent = 0;
      timer = window.setInterval(() => {
        sent = Math.min(total, sent + total / 8);
        onProgress(sent, total);
        if (sent >= total) {
          window.clearInterval(timer);
          const names = images.filter((f) => accepted(f.name)).map((f) => f.name);
          const stored: Stored = {
            id: `dev-${Date.now().toString(36)}`,
            names: names.slice(0, Math.max(1, names.length - 1)), // last one "unmatched"
            startedAt: Date.now(),
            cancelledAt: null,
          };
          save(stored);
          resolve(statusOf(stored));
        }
      }, 150);
    });
    return {
      promise,
      abort: () => {
        window.clearInterval(timer);
        rejectFn(new DOMException("Upload cancelled", "AbortError"));
      },
    };
  },

  get: async (id) => {
    const w = window as unknown as { __devPolls?: number };
    w.__devPolls = (w.__devPolls ?? 0) + 1; // for the Playwright "polling stops" check
    const stored = load(id);
    if (!stored) throw new ApiError(404, "This batch could not be found.");
    return delay(statusOf(stored), 80);
  },

  getItem: async (id, row) => {
    const stored = load(id);
    if (!stored) throw new ApiError(404, "This batch could not be found.");
    const item = fullStatusOf(stored).items.find((i) => i.row_number === row);
    if (!item) throw new ApiError(404, `Row ${row} of the manifest is not part of this batch.`);
    fetchLog.push(row);
    return delay(item, 400); // slow enough to see the loading state
  },

  // Cancel keeps the batch: finished items and the CSV stay available.
  cancel: async (id) => {
    const stored = load(id);
    if (!stored) throw new ApiError(404, "This batch could not be found.");
    if (!stored.cancelledAt && Date.now() - stored.startedAt < stored.names.length * MS_PER_LABEL) {
      stored.cancelledAt = Date.now();
      save(stored);
    }
    return delay(statusOf(stored), 150);
  },

  downloadTemplate: async () =>
    saveBlob(
      "filename,brand_name,class_type,alcohol_content,net_contents,beverage_class,label_width_mm\r\n" +
        "old_tom.jpg,OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol. (90 Proof),750 mL,distilled_spirits,\r\n",
      "label-manifest-template.csv",
    ),

  downloadResults: async (id) => {
    const stored = load(id);
    if (!stored) throw new ApiError(404, "This batch could not be found.");
    const rows = fullStatusOf(stored).items.map((i) => `${i.filename},${i.row_number},${i.state},${i.tier_label ?? "not checked"}`);
    saveBlob(["filename,manifest_row,status,tier", ...rows].join("\r\n"), `batch-${id}-results.csv`);
  },
};

/** DEV: twelve copies of the sample label plus a manifest, as if chosen by the agent. */
export async function sampleBatchFiles(): Promise<File[]> {
  const svg = await (await fetch(sampleLabelUrl)).blob();
  const images = Array.from({ length: 12 }, (_, i) => {
    const name = `label_${String(i + 1).padStart(2, "0")}.png`;
    // SVG bytes under a .png name: fine for the mock, which only pairs names.
    return new File([svg], name, { type: "image/svg+xml" });
  });
  const rows = images.map((f) => `${f.name},OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL`);
  const csv = new File(
    [["filename,brand_name,class_type,alcohol_content,net_contents", ...rows].join("\r\n")],
    "labels.csv",
    { type: "text/csv" },
  );
  return [csv, ...images];
}
