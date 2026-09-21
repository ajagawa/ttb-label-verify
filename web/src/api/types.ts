/**
 * API types. Mirrors rules/results.py (the output contract) and the metadata
 * endpoints in api/main.py. Keep this file in step with those two; it is the
 * only place the frontend describes the server's shapes.
 */

export type Verdict = "MATCH" | "REVIEW" | "MISMATCH";

export type Strategy = "anchored" | "pattern" | "positional" | "vocabulary" | "not_found";

export type CheckOutcome = "pass" | "fail" | "advisory" | "not_evaluable";

/** Region of the label, in the PREPROCESSED image's pixel space. */
export interface BoxModel {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface CheckResult {
  id: string;
  outcome: CheckOutcome;
  citation: string;
  summary: string;
  detail: string | null;
  measurement: Record<string, number>;
  confidence: number | null;
}

export interface TextDifference {
  position: number;
  /** null where the label has an extra token */
  expected: string | null;
  /** null where a required token is missing */
  found: string | null;
  kind: "substituted" | "missing" | "extra" | string;
}

export interface FieldResult {
  field_id: string;
  label: string;
  verdict: Verdict;
  strategy: Strategy;
  expected: string | null;
  found: string | null;
  bbox: BoxModel | null;
  score: number;
  ocr_confidence: number;
  reason: string;
  citation: string;
  citation_url: string | null;
  elevated: boolean;
  /**
   * False when the tool has no check for this field and never looked at it.
   * Distinct from strategy "not_found" on an automated field, which means it
   * looked and could not find the value. See rules/results.py.
   */
  automated: boolean;
  checks: CheckResult[];
  differences: TextDifference[];
}

export interface ImageQuality {
  usable: boolean;
  score: number;
  issues: string[];
  guidance: string[];
  measurements: Record<string, number>;
  /**
   * False when no quality assessment was made. An unassessed quality is not a
   * good one: read `usable`/`score` only when this is true.
   */
  assessed: boolean;
}

export interface VerificationResult {
  verified_at: string;
  ruleset_version: string;
  provider_name: string;
  quality: ImageQuality;
  /** Already in the agents' printed-checklist order. Never re-sort. */
  fields: FieldResult[];
  image_width: number;
  image_height: number;
  elapsed_ms: number;
  extraction_ms: number;
  diagnostics?: Record<string, unknown> | null;
}

/** GET /api/ruleset */
export interface FieldDescriptor {
  id: string;
  label: string;
  citation: string;
  citation_url: string | null;
  required: boolean;
  implemented: boolean;
}

export interface RulesetMetadata {
  version: string;
  effective_date: string;
  description: string;
  beverage_classes: Record<string, string>;
  fields: FieldDescriptor[];
}

/** GET /api/health */
export interface HealthResponse {
  status: "ok" | "degraded" | string;
  ruleset_version: string;
  provider: string | null;
  provider_error: string | null;
  diagnostics_enabled: boolean;
}

/** The application record as entered in the form. */
export interface ApplicationForm {
  brand_name: string;
  class_type: string;
  alcohol_content: string;
  net_contents: string;
  beverage_class: string;
  /** Kept as the raw input string; empty means "not supplied". */
  label_width_mm: string;
}

// ---------------------------------------------------------------------------
// Batch — mirrors api/batch_models.py and rules/triage.py
// ---------------------------------------------------------------------------

/** Manifest columns, in template order. The first five are required. */
export const MANIFEST_REQUIRED_COLUMNS = [
  "filename",
  "brand_name",
  "class_type",
  "alcohol_content",
  "net_contents",
] as const;
export const MANIFEST_OPTIONAL_COLUMNS = ["beverage_class", "label_width_mm"] as const;
export const MANIFEST_COLUMNS = [...MANIFEST_REQUIRED_COLUMNS, ...MANIFEST_OPTIONAL_COLUMNS] as const;

/** Largest batch the server accepts. */
export const MAX_BATCH_SIZE = 500;

export interface ManifestRow {
  row_number: number;
  filename: string;
  brand_name: string;
  class_type: string;
  alcohol_content: string;
  net_contents: string;
  beverage_class: string;
  label_width_mm: number | null;
}

export type PairingIssueKind =
  | "image_without_row"
  | "row_without_image"
  | "duplicate_row"
  | "duplicate_image"
  | "invalid_row"
  | "missing_column"
  | "unsupported_file";

export interface PairingIssue {
  kind: PairingIssueKind;
  filename: string | null;
  row_number: number | null;
  /** Written for the agent; shown verbatim. */
  message: string;
}

export interface PairingReport {
  matched: number;
  total_images: number;
  total_rows: number;
  issues: PairingIssue[];
  /** True only when nothing at all could be checked. */
  blocking: boolean;
}

export type BatchState = "queued" | "running" | "done" | "cancelled";
/**
 * pending: not started yet, will run · running: being read now ·
 * done: verified · error: could not be processed ·
 * skipped: never checked, because the batch was stopped first.
 */
export type ItemState = "pending" | "running" | "done" | "error" | "skipped";

/**
 * rules/triage.py `Tier`, 0 = most urgent. The server sends the int and its
 * label; the frontend displays both and never re-ranks.
 */
export const Tier = {
  ASSERTED: 0,
  MISSING_MANDATORY: 1,
  UNREADABLE: 2,
  REVIEW: 3,
  UNCHECKED: 4,
  CLEAR: 5,
} as const;

export interface BatchItem {
  filename: string;
  row_number: number;
  brand_name: string;
  state: ItemState;
  /** null while pending */
  tier: number | null;
  tier_label: string | null;
  needs_attention: boolean | null;
  mismatch_count: number;
  elevated_count: number;
  review_count: number;
  unchecked_count: number;
  headline: string | null;
  result: VerificationResult | null;
  error: string | null;
}

export interface BatchStatus {
  batch_id: string;
  state: BatchState;
  total: number;
  /** Finished items, DONE and ERROR. Excludes SKIPPED. */
  completed: number;
  failed: number;
  needs_attention: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_ms: number;
  /** null until at least one label has finished — never invent one. */
  estimated_remaining_ms: number | null;
  expires_at: string;
  pairing: PairingReport;
  ruleset_version: string;
  /**
   * Server order — finished worst-first, then running, then pending, then
   * skipped (api/batch_models.py). Never re-sort.
   */
  items: BatchItem[];
}
