/**
 * Progress and time formatting for batch mode.
 *
 * Rule: never invent an estimate. The server sends `estimated_remaining_ms`
 * as null until at least one label has finished; that is shown as
 * "estimating…", not as a number.
 */

/** "9 min 42 s", "42 s", "1 h 5 min". */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return m > 0 ? `${h} h ${m} min` : `${h} h`;
  if (m > 0) return s > 0 ? `${m} min ${s} s` : `${m} min`;
  return `${s} s`;
}

/** "about 5 minutes left" — or "estimating…" when there is no basis yet. */
export function formatRemaining(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "estimating…";
  if (ms < 60_000) return "less than a minute left";
  const minutes = Math.ceil(ms / 60_000);
  if (minutes < 60) return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"} left`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `about ${h} h${m ? ` ${m} min` : ""} left`;
}

/** "340 MB", "1.1 GB", "820 KB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Whole percent, clamped; 0 when the total is unknown. */
export function percent(done: number, total: number): number {
  if (!(total > 0)) return 0;
  return Math.max(0, Math.min(100, Math.floor((done / total) * 100)));
}

/** Poll every 2 s while the tab is visible; back off to 15 s when hidden. */
export function pollDelay(hidden: boolean): number {
  return hidden ? 15_000 : 2_000;
}
