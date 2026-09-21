/**
 * DEV ONLY — never part of the production bundle.
 *
 * Loaded only through a dynamic import guarded by `import.meta.env.DEV` in
 * App.tsx, which Vite replaces with `false` in production builds so this
 * module and the sample image are dropped entirely. Enabled with `?mock=1`
 * while running `npm run dev`. It lets the results view and the bbox overlay
 * be developed without the OCR engine, which the degraded local backend lacks.
 *
 * Coordinates match sample-label.svg (800 x 1100).
 */

import type { VerificationResult } from "../api/types";
import sampleLabelUrl from "./sample-label.svg?url";

export async function loadSampleLabel(): Promise<File> {
  const blob = await (await fetch(sampleLabelUrl)).blob();
  return new File([blob], "dev-sample-label.svg", { type: "image/svg+xml" });
}

export function mockVerify(): Promise<VerificationResult> {
  return new Promise((resolve) => setTimeout(() => resolve(MOCK_RESULT), 900));
}

export const MOCK_RESULT: VerificationResult = {
  verified_at: "2026-09-20T12:00:00Z",
  ruleset_version: "ttb-v1",
  provider_name: "dev-mock",
  image_width: 800,
  image_height: 1100,
  elapsed_ms: 1840,
  extraction_ms: 1310,
  quality: {
    usable: true,
    score: 0.81,
    issues: ["glare"],
    guidance: ["Slight glare near the lower edge of the label. If the warning text looks unclear, ask for a photo taken without flash."],
    measurements: {},
    assessed: true,
  },
  fields: [
    {
      field_id: "brand_name",
      label: "Brand Name",
      verdict: "MATCH",
      strategy: "anchored",
      expected: "OLD TOM DISTILLERY",
      found: "OLD TOM DISTILLERY",
      bbox: { left: 98, top: 150, width: 605, height: 64 },
      score: 0.98,
      ocr_confidence: 0.96,
      reason: "The brand name on the label matches the application.",
      citation: "27 CFR 5.63",
      citation_url: "https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-5/subpart-D/section-5.63",
      elevated: false,
      automated: true,
      checks: [],
      differences: [],
    },
    {
      field_id: "alcohol_content",
      label: "Alcohol Content",
      verdict: "MATCH",
      strategy: "pattern",
      expected: "45% Alc./Vol. (90 Proof)",
      found: "45% Alc./Vol. (90 Proof)",
      bbox: { left: 80, top: 696, width: 375, height: 36 },
      score: 1,
      ocr_confidence: 0.93,
      reason: "45% ABV on the label equals 45% in the application; 90 proof is consistent.",
      citation: "27 CFR 5.65",
      citation_url: "https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-5/subpart-D/section-5.65",
      elevated: false,
      automated: true,
      checks: [],
      differences: [],
    },
    {
      field_id: "net_contents",
      label: "Net Contents",
      verdict: "MISMATCH",
      strategy: "pattern",
      expected: "750 mL",
      found: "700 mL",
      bbox: { left: 610, top: 696, width: 110, height: 36 },
      score: 0.2,
      ocr_confidence: 0.94,
      reason: "The label states 700 mL; the application states 750 mL.",
      citation: "27 CFR 5.68",
      citation_url: "https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-5/subpart-D/section-5.68",
      elevated: false,
      automated: true,
      checks: [],
      differences: [],
    },
    {
      field_id: "class_type",
      label: "Class/Type Designation",
      verdict: "REVIEW",
      strategy: "not_found",
      expected: "Kentucky Straight Bourbon Whiskey",
      found: null,
      bbox: null,
      score: 0,
      ocr_confidence: 0,
      reason: "This prototype does not yet check class/type designation automatically. Verify manually.",
      citation: "27 CFR 5.63",
      citation_url: "https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-5/subpart-D/section-5.63",
      elevated: false,
      automated: false,
      checks: [],
      differences: [],
    },
    {
      field_id: "government_warning",
      label: "Government Warning Statement",
      verdict: "REVIEW",
      strategy: "anchored",
      expected: null,
      found:
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and can cause health problems.",
      bbox: { left: 72, top: 866, width: 656, height: 170 },
      score: 0.9,
      ocr_confidence: 0.88,
      reason: "The warning statement is present, but one word differs from the text required by § 16.21.",
      citation: "27 CFR 16.21",
      citation_url: "https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-16/subpart-C/section-16.21",
      elevated: false,
      automated: true,
      checks: [
        {
          id: "header_capitalisation",
          outcome: "pass",
          citation: "27 CFR 16.22(a)(2)",
          summary: "“GOVERNMENT WARNING” appears in capital letters.",
          detail: null,
          measurement: {},
          confidence: 0.95,
        },
        {
          id: "text_accuracy",
          outcome: "fail",
          citation: "27 CFR 16.21",
          summary: "The statement is not word-for-word as prescribed.",
          detail: "1 word differs. See the wording differences below.",
          measurement: {},
          confidence: 0.88,
        },
        {
          id: "header_bold",
          outcome: "advisory",
          citation: "27 CFR 16.22(a)(2)",
          summary: "The header appears heavier than the body text.",
          detail: "Estimated from stroke width. A naturally heavy typeface can read as bold.",
          measurement: { stroke_ratio: 1.46 },
          confidence: 0.62,
        },
        {
          id: "contrast",
          outcome: "advisory",
          citation: "27 CFR 16.22(a)(1)",
          summary: "Text-to-background contrast measured at 12.8 : 1.",
          detail: null,
          measurement: { contrast_ratio: 12.8 },
          confidence: 0.8,
        },
        {
          id: "type_size",
          outcome: "not_evaluable",
          citation: "27 CFR 16.22(b)",
          summary: "Minimum type size could not be checked.",
          detail: "Enter the label width in millimetres to enable this check.",
          measurement: {},
          confidence: null,
        },
      ],
      differences: [{ position: 40, expected: "may", found: "can", kind: "substituted" }],
    },
  ],
};
