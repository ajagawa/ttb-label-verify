import { describe, expect, it } from "vitest";
import { friendlyError } from "./errors";
import { checkFile, formatSeconds } from "./format";

describe("friendlyError", () => {
  it.each([401, 413, 415, 422, 503, 500, null])("gives a plain message for %s, never raw JSON", (status) => {
    const e = friendlyError(status as number | null, "some server detail");
    expect(e.title).toBeTruthy();
    expect(e.message).not.toMatch(/[{}[\]]/);
  });

  it("asks for the access code on 401", () => {
    expect(friendlyError(401).needsToken).toBe(true);
    expect(friendlyError(503).needsToken).toBe(false);
  });

  it("explains unsupported types and the size limit", () => {
    expect(friendlyError(415).message).toMatch(/JPEG/);
    expect(friendlyError(413).message).toMatch(/20 MB/);
  });

  it("separates an unreadable image from a form validation error on 422", () => {
    expect(friendlyError(422, "Could not read this image: too dark").title).toMatch(/could not read/i);
    const validation = friendlyError(422, [
      { loc: ["body", "label_width_mm"], msg: "Input should be greater than 0" },
    ]);
    expect(validation.title).toMatch(/check the form/i);
    expect(validation.message).toContain("Label width");
  });

  it("tells the agent a 503 is not their label's fault", () => {
    expect(friendlyError(503).message).toMatch(/not a problem with your label/i);
  });

  it("passes through the server's readable 400 detail", () => {
    expect(friendlyError(400, "The uploaded file is empty.").message).toBe("The uploaded file is empty.");
  });
});

describe("formatting helpers", () => {
  it("shows elapsed time in seconds", () => {
    expect(formatSeconds(1840)).toBe("1.8 s");
    expect(formatSeconds(420)).toBe("0.4 s");
    expect(formatSeconds(34000)).toBe("34 s");
  });

  it("checks files before upload", () => {
    expect(checkFile({ type: "application/pdf", size: 10 })).toMatch(/not supported/);
    expect(checkFile({ type: "image/png", size: 0 })).toMatch(/empty/);
    expect(checkFile({ type: "image/jpeg", size: 21 * 1024 * 1024 })).toMatch(/20 MB/);
    expect(checkFile({ type: "image/tiff", size: 1000 })).toBeNull();
  });
});
