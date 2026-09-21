import { describe, expect, it } from "vitest";
import type { BatchItem, BatchStatus } from "../api/types";
import { tierPresentation } from "./presentation";
import {
  categoryOf,
  shouldKeepPolling,
  countByCategory,
  describeCounts,
  filterItems,
  findByKey,
  itemKey,
  positionOf,
} from "./worklist";

function item(filename: string, overrides: Partial<BatchItem> = {}): BatchItem {
  return {
    filename,
    row_number: 2,
    brand_name: "OLD TOM",
    state: "done",
    tier: 5,
    tier_label: "All clear",
    needs_attention: false,
    mismatch_count: 0,
    elevated_count: 0,
    review_count: 0,
    unchecked_count: 0,
    headline: null,
    result: null,
    error: null,
    ...overrides,
  };
}
const bad = (name: string, row: number, tier = 0) =>
  item(name, { row_number: row, tier, tier_label: "Differs from the application", needs_attention: true });
const clear = (name: string, row: number) => item(name, { row_number: row });
const pending = (name: string, row: number) =>
  item(name, { row_number: row, state: "pending", tier: null, tier_label: null, needs_attention: null });
const running = (name: string, row: number) => ({ ...pending(name, row), state: "running" as const });
// SKIPPED carries needs_attention=true and no tier (api/batch_models.py).
const skipped = (name: string, row: number) => ({ ...pending(name, row), state: "skipped" as const, needs_attention: true });

describe("filtering and counts", () => {
  const items = [bad("c.jpg", 4), bad("a.jpg", 2, 3), clear("b.jpg", 3), pending("d.jpg", 5)];

  it("counts each category", () => {
    expect(countByCategory(items)).toEqual({ attention: 2, clear: 1, pending: 1, unchecked: 0 });
  });

  it("filters without re-sorting: the server's order is kept exactly", () => {
    expect(filterItems(items, "attention").map((i) => i.filename)).toEqual(["c.jpg", "a.jpg"]);
  });

  it("treats a finished item with unknown needs_attention as needing attention", () => {
    expect(categoryOf(item("x.jpg", { needs_attention: null }))).toBe("attention");
  });

  it("files unreadable (error) items under needs attention", () => {
    expect(categoryOf(item("x.jpg", { state: "error", tier: 2, needs_attention: true }))).toBe("attention");
  });

  it("counts SKIPPED items under Not checked, and RUNNING with the waiting ones", () => {
    const stopped = [...items.slice(0, 3), running("r.jpg", 6), skipped("s.jpg", 7), skipped("t.jpg", 8)];
    expect(countByCategory(stopped)).toEqual({ attention: 2, clear: 1, pending: 1, unchecked: 2 });
    expect(filterItems(stopped, "unchecked").map((i) => i.filename)).toEqual(["s.jpg", "t.jpg"]);
    // Not inferred from the batch being stopped: a PENDING item stays "pending".
    expect(categoryOf(pending("d.jpg", 5))).toBe("pending");
  });
});

describe("selection stays put while the list refreshes", () => {
  it("finds the same label by key after new items arrive above it", () => {
    const before = [bad("m.jpg", 7)];
    const key = itemKey(before[0]!);
    const after = [bad("a.jpg", 2), bad("b.jpg", 3), bad("m.jpg", 7)];
    expect(findByKey(after, key)?.filename).toBe("m.jpg");
    const pos = positionOf(after, key);
    expect(pos.index).toBe(2);
    expect(pos.prev).toBe(itemKey(after[1]!));
  });

  it("distinguishes two rows naming the same file", () => {
    expect(itemKey(bad("x.jpg", 2))).not.toBe(itemKey(bad("x.jpg", 3)));
  });
});

describe("previous / next boundaries", () => {
  const queue = [bad("a.jpg", 2), bad("b.jpg", 3), bad("c.jpg", 4)];
  const k = (i: number) => itemKey(queue[i]!);

  it("has no Previous at the first label", () => {
    expect(positionOf(queue, k(0))).toEqual({ index: 0, total: 3, prev: null, next: k(1) });
  });
  it("has no Next at the last label", () => {
    expect(positionOf(queue, k(2))).toEqual({ index: 2, total: 3, prev: k(1), next: null });
  });
  it("offers the top of the list when the selection is outside it", () => {
    expect(positionOf(queue, "nope")).toEqual({ index: -1, total: 3, prev: null, next: k(0) });
  });
  it("handles an empty queue", () => {
    expect(positionOf([], null)).toEqual({ index: -1, total: 0, prev: null, next: null });
  });
});

describe("row presentation", () => {
  it("describes counts in words", () => {
    expect(describeCounts(item("x", { mismatch_count: 1, review_count: 2, unchecked_count: 1 }))).toBe(
      "1 differs · 2 need a look · 1 to check by hand",
    );
  });

  it("uses the server's tier label verbatim, with an icon for every tier", () => {
    for (let tier = 0; tier <= 5; tier++) {
      const p = tierPresentation({ state: "done", tier, tier_label: `label ${tier}` });
      expect(p.label).toBe(`label ${tier}`);
      expect(p.icon).toBeTruthy();
    }
    expect(tierPresentation({ state: "done", tier: 0, tier_label: "x" }).tone).toBe("mismatch");
    expect(tierPresentation({ state: "done", tier: 1, tier_label: "x" }).tone).toBe("elevated");
    expect(tierPresentation({ state: "error", tier: 2, tier_label: "x" }).icon).toBe("no-image");
  });

  it("maps all five item states to a distinct icon + text + tone", () => {
    const pres = (state: BatchItem["state"], tier: number | null = null) =>
      tierPresentation({ state, tier, tier_label: tier == null ? null : "Server label" });
    expect(pres("pending")).toEqual({ tone: "pending", icon: "clock", label: "Waiting to be checked" });
    expect(pres("running")).toEqual({ tone: "running", icon: "reading", label: "Reading now" });
    expect(pres("done", 3)).toEqual({ tone: "review", icon: "alert", label: "Server label" });
    expect(pres("error", 2)).toEqual({ tone: "unreadable", icon: "no-image", label: "Server label" });
    expect(pres("skipped")).toEqual({
      tone: "unchecked",
      icon: "dash",
      label: "Not checked — the batch was stopped",
    });
    // SKIPPED must never read as waiting or as fine.
    expect(pres("skipped").label).not.toMatch(/wait|clear/i);
    const icons = ["pending", "running", "skipped"].map((s) => pres(s as BatchItem["state"]).icon);
    expect(new Set(icons).size).toBe(3);
  });

  it("does not guess a severity for a finished item missing its tier", () => {
    expect(tierPresentation({ state: "done", tier: null, tier_label: null }).tone).toBe("review");
  });

  it("shows an unknown future tier as a neutral look", () => {
    expect(tierPresentation({ state: "pending", tier: null, tier_label: null }).label).toBe("Waiting to be checked");
    expect(tierPresentation({ state: "done", tier: 9, tier_label: "New thing" })).toMatchObject({
      tone: "review",
      label: "New thing",
    });
  });
});

describe("stop-then-settle polling", () => {
  const status = (state: BatchStatus["state"], states: BatchItem["state"][]) => ({
    state,
    items: states.map((st, i) => ({ ...pending(`f${i}.jpg`, i + 2), state: st })),
  });

  it("keeps polling while the batch is queued or running", () => {
    expect(shouldKeepPolling(status("queued", ["pending"]))).toBe(true);
    expect(shouldKeepPolling(status("running", ["done", "running", "pending"]))).toBe(true);
  });

  it("after a stop, keeps polling only while a label is still being read", () => {
    expect(shouldKeepPolling(status("cancelled", ["done", "running", "skipped"]))).toBe(true);
    expect(shouldKeepPolling(status("cancelled", ["done", "done", "skipped"]))).toBe(false);
  });

  it("stops once the batch is done", () => {
    expect(shouldKeepPolling(status("done", ["done", "error"]))).toBe(false);
  });
});

describe("empty list messages", () => {
  const counts = (attention: number, clear: number) => ({ attention, clear, pending: 0, unchecked: 0 });

  it("points to the passing labels when nothing needs attention", async () => {
    const { emptyMessage } = await import("./worklist");
    const m = emptyMessage("attention", counts(0, 2), false);
    expect(m.text).toBe("No labels need attention.");
    expect(m.action).toEqual({ label: "Show the 2 all-clear labels", filter: "clear" });
  });

  it("uses the singular for one label", async () => {
    const { emptyMessage } = await import("./worklist");
    expect(emptyMessage("attention", counts(0, 1), false).action?.label).toBe("Show the 1 all-clear label");
  });

  it("offers nothing when the other list is empty too", async () => {
    const { emptyMessage } = await import("./worklist");
    expect(emptyMessage("attention", counts(0, 0), true)).toEqual({ text: "No labels need attention so far." });
  });

  it("points the other way from an empty all-clear list", async () => {
    const { emptyMessage } = await import("./worklist");
    const m = emptyMessage("clear", counts(3, 0), false);
    expect(m.text).toBe("No labels are all clear.");
    expect(m.action).toEqual({ label: "Show the 3 labels that need attention", filter: "attention" });
  });
});
