import { describe, expect, it } from "vitest";
import { findLocalFile, hasAcceptedExtension, isOsClutter, sortSelection } from "./batchFiles";
import { friendlyError } from "./errors";
import { formatBytes, formatDuration, formatRemaining, percent, pollDelay } from "./progress";

describe("progress and estimates", () => {
  it("never fabricates an estimate", () => {
    expect(formatRemaining(null)).toBe("estimating…");
    expect(formatRemaining(undefined)).toBe("estimating…");
    expect(formatRemaining(Number.NaN)).toBe("estimating…");
  });

  it("phrases real estimates plainly", () => {
    expect(formatRemaining(30_000)).toBe("less than a minute left");
    expect(formatRemaining(61_000)).toBe("about 2 minutes left");
    expect(formatRemaining(5 * 60_000)).toBe("about 5 minutes left");
    expect(formatRemaining(90 * 60_000)).toBe("about 1 h 30 min left");
  });

  it("formats durations", () => {
    expect(formatDuration(582_000)).toBe("9 min 42 s");
    expect(formatDuration(42_000)).toBe("42 s");
    expect(formatDuration(3_900_000)).toBe("1 h 5 min");
  });

  it("formats bytes and percent", () => {
    expect(formatBytes(1.1 * 1024 ** 3)).toBe("1.1 GB");
    expect(formatBytes(340 * 1024 ** 2)).toBe("340 MB");
    expect(percent(50, 200)).toBe(25);
    expect(percent(5, 0)).toBe(0);
  });

  it("backs off polling while the tab is hidden", () => {
    expect(pollDelay(false)).toBe(2000);
    expect(pollDelay(true)).toBeGreaterThan(2000);
  });
});

describe("file selection", () => {
  it("drops operating-system clutter from folder picks", () => {
    expect(isOsClutter(".DS_Store")).toBe(true);
    expect(isOsClutter("labels/._IMG_1.jpg")).toBe(true);
    expect(isOsClutter("Thumbs.db")).toBe(true);
    expect(isOsClutter("IMG_1.jpg")).toBe(false);
  });

  it("recognises accepted image extensions case-insensitively", () => {
    expect(hasAcceptedExtension("IMG_0412.JPG")).toBe(true);
    expect(hasAcceptedExtension("scan.tif")).toBe(true);
    expect(hasAcceptedExtension("artwork.pdf")).toBe(false);
  });

  it("separates the manifest from images and ignores a file picked twice", () => {
    const a = new File(["x"], "a.jpg", { lastModified: 1 });
    const csv = new File(["h"], "labels.csv", { type: "text/csv" });
    const { csvs, images } = sortSelection([a, csv, a, new File([""], ".DS_Store")]);
    expect(csvs.map((f) => f.name)).toEqual(["labels.csv"]);
    expect(images).toHaveLength(1);
  });

  it("finds the local image for a worklist item like the server pairs", () => {
    const files = [{ name: "IMG_1.JPG" }, { name: "b.jpg" }];
    expect(findLocalFile(files, "b.jpg")?.name).toBe("b.jpg");
    expect(findLocalFile(files, "img_1.jpg")?.name).toBe("IMG_1.JPG");
    // Ambiguous by case only: show no image rather than possibly the wrong one.
    expect(findLocalFile([{ name: "a.jpg" }, { name: "A.JPG" }], "a.Jpg")).toBeNull();
  });
});

describe("batch error wording", () => {
  it("explains an expired or stopped batch", () => {
    expect(friendlyError(404, "whatever", "batch").title).toBe("This batch has expired or was stopped");
  });

  it("uses the server's readable size-limit advice", () => {
    const e = friendlyError(413, "A batch can hold at most 500 images. Split the submission.", "batch");
    expect(e.title).toMatch(/too large/);
    expect(e.message).toMatch(/at most 500/);
  });

  it("tells 'busy' apart from 'OCR engine missing' on 503", () => {
    expect(friendlyError(503, "Other batches are still being processed…", "batch").title).toBe("The server is busy with other batches");
    expect(friendlyError(503, "Extraction is unavailable: no engine", "batch").title).toMatch(/unavailable/);
  });

  it("reads the message out of the submit endpoint's object-shaped 422", () => {
    const e = friendlyError(422, { message: "Nothing in this batch could be checked.", pairing: {} }, "batch");
    expect(e.message).toBe("Nothing in this batch could be checked.");
    expect(e.message).not.toMatch(/[{}]/);
  });

  it("still asks for the access code on 401 in batch mode", () => {
    expect(friendlyError(401, null, "batch").needsToken).toBe(true);
  });

  it("leaves single-label wording unchanged", () => {
    expect(friendlyError(413).message).toMatch(/20 MB/);
  });
});

import { filenameFromDisposition, parseRetryAfter } from "../api/client";
import { errorCode, isCapacityError } from "./errors";

describe("503 codes", () => {
  const capacity = { code: "batch_capacity", message: "Other batches are still being processed." };

  it("decides by code, not message text", () => {
    expect(isCapacityError(503, capacity)).toBe(true);
    expect(isCapacityError(503, { code: "extraction_unavailable", message: "batch batch batch" })).toBe(false);
    expect(errorCode(capacity)).toBe("batch_capacity");
    expect(friendlyError(503, capacity, "batch").title).toBe("The server is busy with other batches");
    expect(friendlyError(503, { code: "extraction_unavailable", message: "x" }, "batch").title).toMatch(
      /reading is unavailable/,
    );
    expect(friendlyError(503, { code: "batch_unavailable", message: "x" }, "batch").title).toMatch(
      /not available on this server/,
    );
  });

  it("falls back to the text when an older server sends no code", () => {
    expect(isCapacityError(503, "Other batches are still being processed.")).toBe(true);
    expect(isCapacityError(503, "Extraction is unavailable: no engine")).toBe(false);
    expect(isCapacityError(413, capacity)).toBe(false);
  });

  it("reads Retry-After in seconds or as a date", () => {
    expect(parseRetryAfter("45")).toBe(45);
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter("soon")).toBeNull();
    expect(parseRetryAfter(new Date(Date.now() + 60_000).toUTCString())).toBeGreaterThan(50);
  });

  it("takes the download's filename from Content-Disposition", () => {
    expect(filenameFromDisposition('attachment; filename="batch-1a2b-results.csv"')).toBe("batch-1a2b-results.csv");
    expect(filenameFromDisposition(null)).toBeNull();
  });
});
