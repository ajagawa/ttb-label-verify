import type { ReactElement } from "react";
import type { IconName } from "../lib/presentation";

/**
 * Inline SVG icons: no icon font, no network request. Always decorative
 * (aria-hidden) — the adjacent text label carries the meaning, the icon
 * reinforces it for people who cannot tell the colours apart.
 */
const PATHS: Record<IconName, ReactElement> = {
  check: <path d="M5 12.5l4.5 4.5L19 7.5" fill="none" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />,
  cross: <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" fill="none" strokeWidth="3" strokeLinecap="round" />,
  alert: (
    <>
      <path d="M12 3.5L22 20.5H2z" fill="none" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M12 10v4.5" fill="none" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="12" cy="17.4" r="1.3" stroke="none" fill="currentColor" />
    </>
  ),
  "alert-strong": (
    <>
      <path d="M12 3.5L22 20.5H2z" fill="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M12 10v4.5" fill="none" stroke="var(--icon-knockout)" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="12" cy="17.4" r="1.3" stroke="none" fill="var(--icon-knockout)" />
    </>
  ),
  dash: (
    <>
      <circle cx="12" cy="12" r="9" fill="none" strokeWidth="2.2" strokeDasharray="3 2.5" />
      <path d="M8 12h8" fill="none" strokeWidth="2.6" strokeLinecap="round" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" fill="none" strokeWidth="2.2" />
      <path d="M12 11v6" fill="none" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="12" cy="7.6" r="1.3" stroke="none" fill="currentColor" />
    </>
  ),
  "no-image": (
    <>
      <rect x="3.5" y="5" width="17" height="14" rx="1.5" fill="none" strokeWidth="2.2" />
      <path d="M3 3l18 18" fill="none" strokeWidth="2.4" strokeLinecap="round" />
    </>
  ),
  // Magnifier: "being read". Static on purpose — no animation that could
  // outlive a dropped connection.
  reading: (
    <>
      <circle cx="10.5" cy="10.5" r="6" fill="none" strokeWidth="2.4" />
      <path d="M15 15l5.5 5.5" fill="none" strokeWidth="2.8" strokeLinecap="round" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" fill="none" strokeWidth="2.2" />
      <path d="M12 7v5.5l3.5 2" fill="none" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </>
  ),
};

export function Icon({ name, className }: { name: IconName; className?: string }) {
  return (
    <svg
      className={`icon ${className ?? ""}`}
      viewBox="0 0 24 24"
      width="24"
      height="24"
      stroke="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  );
}
