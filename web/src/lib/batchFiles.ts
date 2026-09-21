/**
 * Which of the files an agent picked are candidate label images.
 *
 * Mirrors api/manifest.py (`is_os_clutter`, `ACCEPTED_EXTENSIONS`). Choosing a
 * folder picks up .DS_Store, Thumbs.db and macOS "._" files; those are dropped
 * silently, as the server does. Unsupported files (a stray PDF) are NOT
 * dropped from the pairing check — the server reports them in plain language
 * — but they are not uploaded, since the server would only discard them.
 */

const CLUTTER = new Set([".ds_store", "thumbs.db", "desktop.ini", "ehthumbs.db"]);
export const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"];

export function isOsClutter(name: string): boolean {
  const base = (name.split(/[\\/]/).pop() ?? name).toLowerCase();
  return CLUTTER.has(base) || base.startsWith("._");
}

export function hasAcceptedExtension(name: string): boolean {
  const base = name.toLowerCase();
  const dot = base.lastIndexOf(".");
  return dot > 0 && ACCEPTED_EXTENSIONS.includes(base.slice(dot));
}

export function isCsv(file: { name: string; type: string }): boolean {
  return file.name.toLowerCase().endsWith(".csv") || file.type === "text/csv";
}

/**
 * Split a mixed drop/folder selection into manifest candidates and images.
 * Duplicate File objects (same name, size, date — the same file picked twice)
 * are collapsed; genuinely different files with the same name are kept, so
 * the server can report the duplicate rather than the tool silently choosing.
 */
export function sortSelection(files: File[]): { csvs: File[]; images: File[] } {
  const csvs: File[] = [];
  const images: File[] = [];
  const seen = new Set<string>();
  for (const file of files) {
    if (isOsClutter(file.name)) continue;
    const identity = `${file.name}\u0000${file.size}\u0000${file.lastModified}`;
    if (seen.has(identity)) continue;
    seen.add(identity);
    (isCsv(file) ? csvs : images).push(file);
  }
  return { csvs, images };
}

/**
 * The locally chosen file for a worklist item, for the evidence overlay.
 * Matches like the server pairs: exact name first, then a case-insensitive
 * match only when exactly one file fits. If that is ambiguous, no image is
 * shown — a highlight drawn on the wrong photo would be worse than none.
 */
export function findLocalFile<T extends { name: string }>(files: T[], filename: string): T | null {
  const exact = files.filter((f) => f.name === filename);
  if (exact.length === 1) return exact[0] ?? null;
  if (exact.length > 1) return null;
  const folded = filename.toLowerCase();
  const loose = files.filter((f) => f.name.toLowerCase() === folded);
  return loose.length === 1 ? (loose[0] ?? null) : null;
}
