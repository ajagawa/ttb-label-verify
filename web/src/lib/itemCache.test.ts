import { describe, expect, it } from "vitest";
import type { BatchItem, VerificationResult } from "../api/types";
import { displayItem, needsFetch, prefetchTarget, ResultCache } from "./itemCache";
import { itemKey } from "./worklist";

const RESULT = { fields: [] } as unknown as VerificationResult;

function summary(row: number, state: BatchItem["state"] = "done"): BatchItem {
  return {
    filename: `label_${row}.jpg`,
    row_number: row,
    brand_name: "OLD TOM",
    state,
    tier: state === "pending" ? null : 3,
    tier_label: state === "pending" ? null : "Needs a look",
    needs_attention: state === "pending" ? null : true,
    mismatch_count: 0,
    elevated_count: 0,
    review_count: 1,
    unchecked_count: 0,
    headline: null,
    result: null,
    error: state === "error" ? "Could not read this image" : null,
  };
}
const full = (row: number): BatchItem => ({ ...summary(row), result: RESULT });

describe("ResultCache", () => {
  it("fetches a finished label once, then serves it from the cache", () => {
    const cache = new ResultCache();
    const s = summary(4);
    expect(needsFetch(s, cache)).toBe(true);
    cache.begin(4);
    expect(needsFetch(s, cache)).toBe(false); // in flight: no duplicate request
    cache.put(full(4));
    cache.end(4);
    expect(needsFetch(s, cache)).toBe(false); // Previous/Next back to it: no refetch
    expect(displayItem(s, cache).result).toBe(RESULT);
  });

  it("never caches a pending, running or skipped item — only finished outcomes are final", () => {
    const cache = new ResultCache();
    for (const state of ["pending", "running", "skipped"] as const) {
      expect(cache.put(summary(5, state))).toBe(false);
      expect(needsFetch(summary(5, state), cache)).toBe(false);
    }
    expect(cache.has(5)).toBe(false);
  });

  it("does not fetch a pending item, and does once a poll shows it done", () => {
    const cache = new ResultCache();
    expect(needsFetch(summary(6, "pending"), cache)).toBe(false);
    expect(needsFetch(summary(6, "done"), cache)).toBe(true);
  });

  it("does not fetch an unreadable item: its summary already carries the reason", () => {
    expect(needsFetch(summary(7, "error"), new ResultCache())).toBe(false);
  });

  it("shows the latest summary until the full item is cached", () => {
    const cache = new ResultCache();
    const s = summary(8);
    expect(displayItem(s, cache)).toBe(s);
  });
});

describe("prefetchTarget", () => {
  const queue = [summary(2), summary(3), summary(4, "pending"), summary(5)];
  const key = (i: number) => itemKey(queue[i]!);

  it("prefetches exactly the next label in queue order", () => {
    expect(prefetchTarget(queue, key(0), new ResultCache())?.row_number).toBe(3);
  });

  it("does nothing when the next label is already cached or loading", () => {
    const cache = new ResultCache();
    cache.put(full(3));
    expect(prefetchTarget(queue, key(0), cache)).toBeNull();
    const loading = new ResultCache();
    loading.begin(3);
    expect(prefetchTarget(queue, key(0), loading)).toBeNull();
  });

  it("does not prefetch a pending next label, nor past the end of the queue", () => {
    expect(prefetchTarget(queue, key(1), new ResultCache())).toBeNull();
    expect(prefetchTarget(queue, key(3), new ResultCache())).toBeNull();
  });

  it("does nothing without a selection", () => {
    expect(prefetchTarget(queue, null, new ResultCache())).toBeNull();
  });
});
