/**
 * Worklist logic for batch mode. Pure, so it is unit-tested.
 *
 * The server's item order IS "worst first" (rules/triage.py is the single
 * definition, and the evaluator measures that exact order). Nothing here
 * sorts: filtering keeps relative order, and navigation walks it as given.
 */

import type { BatchItem, BatchStatus } from "../api/types";

/**
 * "pending" = will be checked (waiting, or being read now); "unchecked" =
 * SKIPPED, never checked because the batch was stopped. They must never read
 * the same: one result is coming, the other is not.
 */
export type WorklistFilter = "attention" | "clear" | "pending" | "unchecked";

/**
 * Stable identity for an item across polls. Filenames alone are not unique in
 * principle, but a (manifest row, filename) pair is: each row pairs with at
 * most one image.
 */
export function itemKey(item: Pick<BatchItem, "row_number" | "filename">): string {
  return `${item.row_number}\u0000${item.filename}`;
}

export function categoryOf(item: BatchItem): WorklistFilter {
  if (item.state === "skipped") return "unchecked";
  if (item.state === "pending" || item.state === "running") return "pending";
  // Unknown (null) on a finished item is treated as needing attention: the
  // safe mistake is showing an agent something, not hiding it.
  return item.needs_attention === false ? "clear" : "attention";
}

export function countByCategory(items: BatchItem[]): Record<WorklistFilter, number> {
  const counts: Record<WorklistFilter, number> = { attention: 0, clear: 0, pending: 0, unchecked: 0 };
  for (const item of items) counts[categoryOf(item)] += 1;
  return counts;
}

export function filterItems(items: BatchItem[], filter: WorklistFilter): BatchItem[] {
  return items.filter((item) => categoryOf(item) === filter);
}

/**
 * Whether the status is worth polling again. A running or queued batch, yes.
 * A finished one, no. A stopped one only while a label that was already being
 * read when the agent pressed Stop is still RUNNING — it will finish and its
 * result is kept. No fixed "a few more polls": the server says when it is done.
 */
export function shouldKeepPolling(status: Pick<BatchStatus, "state" | "items">): boolean {
  if (status.state === "queued" || status.state === "running") return true;
  if (status.state === "cancelled") return status.items.some((item) => item.state === "running");
  return false;
}

export function findByKey(items: BatchItem[], key: string | null): BatchItem | null {
  if (key == null) return null;
  return items.find((item) => itemKey(item) === key) ?? null;
}

export interface Position {
  /** 0-based index of the selection in the list, or -1 when not in it. */
  index: number;
  total: number;
  prev: string | null;
  next: string | null;
}

/**
 * Where the selection sits in the (filtered, openable) queue, and what
 * Previous/Next lead to. Computed from keys on every refresh, so items
 * arriving above the selection during a run shift its number but never move
 * the agent off the label they are reading.
 *
 * If the selection is not in this list (e.g. the filter changed), Next starts
 * at the top and Previous is unavailable.
 */
export function positionOf(queue: BatchItem[], key: string | null): Position {
  const index = key == null ? -1 : queue.findIndex((item) => itemKey(item) === key);
  const at = (i: number) => {
    const item = queue[i];
    return item ? itemKey(item) : null;
  };
  if (index < 0) return { index, total: queue.length, prev: null, next: at(0) };
  return { index, total: queue.length, prev: at(index - 1), next: at(index + 1) };
}

/** Plain-language counts for a row: "1 differs · 2 need a look". */
export function describeCounts(item: BatchItem): string {
  const parts: string[] = [];
  if (item.mismatch_count) parts.push(`${item.mismatch_count} ${item.mismatch_count === 1 ? "differs" : "differ"}`);
  if (item.elevated_count) parts.push(`${item.elevated_count} not found`);
  if (item.review_count) parts.push(`${item.review_count} ${item.review_count === 1 ? "needs" : "need"} a look`);
  if (item.unchecked_count) parts.push(`${item.unchecked_count} to check by hand`);
  return parts.join(" · ");
}

export interface EmptyMessage {
  text: string;
  /** Where the labels the agent may be looking for actually are. */
  action?: { label: string; filter: WorklistFilter };
}

const labelsWord = (n: number) => (n === 1 ? "label" : "labels");

/**
 * What an empty filter says. An empty "Needs attention" list is the good
 * outcome, but on its own it reads as "nothing happened": a reviewer who ran
 * two passing labels saw no rows and asked where they went. So the message
 * says how many are in the other list and offers to show them.
 */
export function emptyMessage(
  filter: WorklistFilter,
  counts: Record<WorklistFilter, number>,
  running: boolean,
): EmptyMessage {
  const soFar = running ? " so far" : "";
  switch (filter) {
    case "attention": {
      const n = counts.clear;
      return {
        text: `No labels need attention${soFar}.`,
        action: n > 0 ? { label: `Show the ${n} all-clear ${labelsWord(n)}`, filter: "clear" } : undefined,
      };
    }
    case "clear": {
      const n = counts.attention;
      return {
        text: `No labels are all clear${soFar}.`,
        action: n > 0 ? { label: `Show the ${n} ${labelsWord(n)} that need attention`, filter: "attention" } : undefined,
      };
    }
    case "pending":
      return { text: "Nothing is waiting." };
    case "unchecked":
      return { text: "Every label was checked." };
  }
}
