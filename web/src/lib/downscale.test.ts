import { describe, expect, it } from "vitest";
import { JPEG_QUALITY, MAX_EDGE_PX, prepareImage, targetSize, type Codec, type Decoded } from "./downscale";

describe("targetSize", () => {
  it("leaves images at or under 2600 px alone", () => {
    expect(targetSize(2600, 1200)).toBeNull();
    expect(targetSize(1600, 1200)).toBeNull();
  });

  it("pins the longest edge to 2600 and keeps the aspect ratio", () => {
    expect(targetSize(4032, 3024)).toEqual({ width: 2600, height: 1950 });
    expect(targetSize(3024, 4032)).toEqual({ width: 1950, height: 2600 });
  });

  it("never produces a zero-pixel edge", () => {
    expect(targetSize(100000, 10)).toEqual({ width: MAX_EDGE_PX, height: 1 });
  });

  it("rejects nonsense dimensions", () => {
    expect(targetSize(0, 100)).toBeNull();
  });
});

function fakeCodec(width: number, height: number, encodedBytes = 1000, fail = false) {
  const calls: Array<{ size: { width: number; height: number }; quality: number }> = [];
  let closed = false;
  const codec: Codec<Decoded> = {
    decode: async () => {
      if (fail) throw new Error("cannot decode");
      return { width, height, close: () => (closed = true) };
    },
    encode: async (_d, size, quality) => {
      calls.push({ size, quality });
      return new Blob([new Uint8Array(encodedBytes)], { type: "image/jpeg" });
    },
  };
  return { codec, calls, closed: () => closed };
}

const photo = (name: string, bytes: number, type = "image/jpeg") =>
  new File([new Uint8Array(bytes)], name, { type, lastModified: 1234 });

describe("prepareImage", () => {
  it("uploads small images untouched — the very same File", async () => {
    const file = photo("small.png", 5000, "image/png");
    const out = await prepareImage(file, fakeCodec(1600, 1200).codec);
    expect(out).toMatchObject({ kind: "untouched", reason: "small" });
    expect(out.file).toBe(file);
  });

  it("resizes large images to JPEG, keeping the original filename for pairing", async () => {
    const fake = fakeCodec(4032, 3024, 2000);
    const out = await prepareImage(photo("IMG_0412.PNG", 9_000_000, "image/png"), fake.codec);
    expect(out.kind).toBe("resized");
    expect(out.file.name).toBe("IMG_0412.PNG");
    expect(out.file.type).toBe("image/jpeg");
    expect(out.file.size).toBe(2000);
    expect(fake.calls).toEqual([{ size: { width: 2600, height: 1950 }, quality: JPEG_QUALITY }]);
    expect(fake.closed()).toBe(true);
  });

  it("measures dimensions after orientation (portrait photo stays portrait)", async () => {
    // The codec decodes with EXIF applied; a rotated phone photo reports as portrait.
    const fake = fakeCodec(3024, 4032);
    const out = await prepareImage(photo("portrait.jpg", 8_000_000), fake.codec);
    expect(out.kind === "resized" && out.to).toEqual({ width: 1950, height: 2600 });
  });

  it("sends files the browser cannot decode (e.g. TIFF) untouched", async () => {
    const file = photo("scan.tiff", 30_000, "image/tiff");
    const out = await prepareImage(file, fakeCodec(0, 0, 0, true).codec);
    expect(out).toMatchObject({ kind: "untouched", reason: "undecodable" });
    expect(out.file).toBe(file);
  });

  it("keeps the original if re-encoding would not make it smaller", async () => {
    const file = photo("tiny-but-huge.jpg", 500);
    const out = await prepareImage(file, fakeCodec(5000, 5000, 900).codec);
    expect(out).toMatchObject({ kind: "untouched", reason: "not-smaller" });
    expect(out.file).toBe(file);
  });
});
