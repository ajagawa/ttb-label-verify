/**
 * Full results for batch items, fetched one at a time.
 *
 * Status polls carry summaries only (the full results were ~2 MB per poll at
 * 300 labels). A label's full result is fetched when the agent opens it,
 * cached by manifest row for the life of the batch view, and the NEXT label
 * in the queue is prefetched so walking with → feels instant.
 *
 * Only finished items are cached: a finished result cannot change, a pending
 * one will. Pure apart from the Map, so it is unit-tested.
 */

import type { BatchItem } from "../api/types";
import { itemKey } from "./worklist";

export class ResultCache {
  private readonly items = new Map<number, BatchItem>();
  private readonly inflight = new Set<number>();

  get(row: number): BatchItem | undefined {
    return this.items.get(row);
  }

  has(row: number): boolean {
    return this.items.has(row);
  }

  /** Stores a fetched item if it is final. Returns whether it was stored. */
  put(item: BatchItem): boolean {
    // Only finished outcomes are final. A pending/running label will change;
    // a skipped one has nothing to fetch.
    if (item.state !== "done" && item.state !== "error") return false;
    this.items.set(item.row_number, item);
    return true;
  }

  isLoading(row: number): boolean {
    return this.inflight.has(row);
  }

  begin(row: number): void {
    this.inflight.add(row);
  }

  end(row: number): void {
    this.inflight.delete(row);
  }

  get size(): number {
    return this.items.size;
  }
}

/**
 * Whether opening/showing this summary needs a fetch. Only DONE items have a
 * result to fetch: a pending or running item has none yet (it is picked up when a later
 * poll shows it done), and an ERROR item's summary already says everything —
 * the reason the image could not be read.
 */
export function needsFetch(summary: BatchItem, cache: ResultCache): boolean {
  return summary.state === "done" && !cache.has(summary.row_number) && !cache.isLoading(summary.row_number);
}

/** The one item worth prefetching: the next in queue order, if it needs a fetch. */
export function prefetchTarget(queue: BatchItem[], selectedKey: string | null, cache: ResultCache): BatchItem | null {
  if (selectedKey == null) return null;
  const index = queue.findIndex((item) => itemKey(item) === selectedKey);
  const next = index >= 0 ? queue[index + 1] : undefined;
  return next && needsFetch(next, cache) ? next : null;
}

/**
 * What the detail view shows: the cached full item when there is one,
 * otherwise the latest summary (result null). The summary is preferred for
 * state while the item is not final, so a pending item flips to done as soon
 * as a poll says so.
 */
export function displayItem(summary: BatchItem, cache: ResultCache): BatchItem {
  return cache.get(summary.row_number) ?? summary;
}
