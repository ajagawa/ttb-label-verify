import { describe, expect, it } from "vitest";
import type { FieldResult } from "../api/types";
import { checkPresentation, countTones, verdictPresentation } from "./presentation";

function field(overrides: Partial<FieldResult> = {}): FieldResult {
  return {
    field_id: "brand_name",
    label: "Brand Name",
    verdict: "MATCH",
    strategy: "anchored",
    expected: "OLD TOM DISTILLERY",
    found: "OLD TOM DISTILLERY",
    bbox: { left: 0, top: 0, width: 10, height: 10 },
    score: 1,
    ocr_confidence: 1,
    reason: "",
    citation: "27 CFR 5.63",
    citation_url: null,
    elevated: false,
    automated: true,
    checks: [],
    differences: [],
    ...overrides,
  };
}

describe("verdictPresentation", () => {
  it("never relies on colour alone: every state has an icon and a text label", () => {
    for (const f of [
      field(),
      field({ verdict: "REVIEW" }),
      field({ verdict: "REVIEW", elevated: true }),
      field({ verdict: "MISMATCH" }),
      field({ verdict: "REVIEW", automated: false }),
    ]) {
      const p = verdictPresentation(f);
      expect(p.icon).toBeTruthy();
      expect(p.label.length).toBeGreaterThan(3);
    }
  });

  it("maps the three verdicts", () => {
    expect(verdictPresentation(field()).tone).toBe("match");
    expect(verdictPresentation(field({ verdict: "REVIEW" })).tone).toBe("review");
    expect(verdictPresentation(field({ verdict: "MISMATCH" })).tone).toBe("mismatch");
  });

  it("raises elevated REVIEW above review without making it a mismatch", () => {
    const p = verdictPresentation(field({ verdict: "REVIEW", elevated: true, strategy: "not_found" }));
    expect(p.tone).toBe("elevated");
    expect(p.icon).not.toBe("cross");
    expect(p.label).not.toMatch(/mismatch/i);
  });

  it("shows unimplemented fields as not checked, from the ruleset flag", () => {
    const p = verdictPresentation(field({ field_id: "class_type", verdict: "REVIEW" }), { class_type: false });
    expect(p.tone).toBe("unchecked");
    expect(p.label).toBe("Not checked automatically — verify manually");
  });

  it("shows unimplemented fields as not checked, from the result's own automated flag", () => {
    const p = verdictPresentation(field({ verdict: "REVIEW", automated: false }));
    expect(p.tone).toBe("unchecked");
  });

  it("never lets an unimplemented field read as passing, whatever its verdict", () => {
    expect(verdictPresentation(field({ verdict: "MATCH" }), { brand_name: false }).tone).toBe("unchecked");
  });

  it("counts tones for the summary", () => {
    const counts = countTones(
      [field(), field({ verdict: "MISMATCH" }), field({ verdict: "REVIEW", elevated: true })],
      {},
    );
    expect(counts).toEqual({ match: 1, review: 0, elevated: 1, mismatch: 1, unchecked: 0 });
  });
});

describe("checkPresentation", () => {
  it("never presents advisory or not-evaluable as pass/fail", () => {
    expect(checkPresentation("advisory").label).not.toMatch(/pass/i);
    expect(checkPresentation("not_evaluable").label).toBe("Couldn't check");
    expect(checkPresentation("pass").tone).toBe("match");
    expect(checkPresentation("fail").tone).toBe("mismatch");
  });
});

