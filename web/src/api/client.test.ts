import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { initialToken, rememberToken } from "./client";

// Tests run in node: provide the one browser API this code touches.
beforeAll(() => {
  const store = new Map<string, string>();
  (globalThis as unknown as { window: unknown }).window = {
    localStorage: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
    },
  };
});

describe("access code persistence", () => {
  afterEach(() => window.localStorage.clear());

  it("is asked for once per browser, not once per visit", () => {
    rememberToken("abc");
    expect(initialToken("")).toBe("abc");
  });

  it("prefers a code in the link, and remembers it", () => {
    rememberToken("old");
    expect(initialToken("?token=new")).toBe("new");
    expect(initialToken("")).toBe("new");
  });

  it("still works when storage is blocked", () => {
    const w = window as unknown as { localStorage: unknown };
    const saved = w.localStorage;
    w.localStorage = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    };
    expect(() => rememberToken("abc")).not.toThrow();
    expect(initialToken("")).toBe("");
    expect(initialToken("?token=x")).toBe("x");
    w.localStorage = saved;
  });

  it("removes the code from the address bar after reading it", () => {
    const w = window as unknown as Record<string, unknown>;
    let replaced: string | null = null;
    w.location = { href: "https://example.test/?token=abc&batch=42" };
    w.history = { state: null, replaceState: (_s: unknown, _t: string, url: string) => (replaced = url) };
    expect(initialToken("?token=abc&batch=42")).toBe("abc");
    expect(replaced).toBe("https://example.test/?batch=42");
    delete w.location;
    delete w.history;
  });

  it("forgets the code when cleared", () => {
    rememberToken("abc");
    rememberToken("");
    expect(initialToken("")).toBe("");
  });
});
