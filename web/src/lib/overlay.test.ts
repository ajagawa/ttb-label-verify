import { describe, expect, it } from "vitest";
import { aspectMismatch, bboxToPercent, cropView, percentStyle } from "./overlay";

describe("bboxToPercent", () => {
  it("scales by the server's preprocessed image size, not raw pixels", () => {
    // 1600 x 1200 is what the server read; the browser may show a 4000 x 3000 original.
    expect(bboxToPercent({ left: 400, top: 300, width: 800, height: 120 }, 1600, 1200)).toEqual({
      left: 25,
      top: 25,
      width: 50,
      height: 10,
    });
  });

  it("clamps boxes that spill past the image edge", () => {
    const p = bboxToPercent({ left: -10, top: 1150, width: 200, height: 100 }, 1000, 1200);
    expect(p).not.toBeNull();
    expect(p!.left).toBe(0);
    expect(p!.width).toBeCloseTo(19);
    expect(p!.top + p!.height).toBeCloseTo(100);
  });

  it("refuses to draw with a missing or zero image size", () => {
    expect(bboxToPercent({ left: 1, top: 1, width: 1, height: 1 }, 0, 100)).toBeNull();
  });

  it("drops degenerate boxes", () => {
    expect(bboxToPercent({ left: 10, top: 10, width: 0, height: 5 }, 100, 100)).toBeNull();
  });

  it("formats as CSS percentages", () => {
    expect(percentStyle({ left: 25, top: 12.5, width: 50, height: 10 })).toEqual({
      left: "25.000%",
      top: "12.500%",
      width: "50.000%",
      height: "10.000%",
    });
  });
});

describe("aspectMismatch", () => {
  it("accepts a downscaled copy of the same picture", () => {
    expect(aspectMismatch(4000, 3000, 1600, 1200)).toBe(false);
  });
  it("flags a picture displayed rotated relative to what the server read", () => {
    expect(aspectMismatch(3000, 4000, 1600, 1200)).toBe(true);
  });
});

describe("cropView", () => {
  it("positions the full image so only the padded box shows", () => {
    const view = cropView({ left: 100, top: 100, width: 200, height: 20 }, 1000, 500)!;
    // pad = max(6, 10) = 10 -> region 90..310 x 90..130 (220 x 40)
    expect(view.aspect).toBeCloseTo(220 / 40);
    expect(view.img.width).toBe(`${((1000 / 220) * 100).toFixed(3)}%`);
    expect(view.img.left).toBe(`${((-90 / 220) * 100).toFixed(3)}%`);
    expect(view.img.top).toBe(`${((-90 / 40) * 100).toFixed(3)}%`);
  });
});
