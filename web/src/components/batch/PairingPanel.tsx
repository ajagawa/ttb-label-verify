import { useState, type ReactNode } from "react";
import type { PairingReport } from "../../api/types";
import type { FriendlyError } from "../../lib/errors";
import { Icon } from "../Icon";

export type PairingState =
  | { kind: "waiting"; missing: "manifest" | "images" | "both" }
  | { kind: "checking" }
  | { kind: "ready"; report: PairingReport }
  | { kind: "error"; error: FriendlyError };

/** Long issue lists (a wrong manifest can produce hundreds) start collapsed. */
const FIRST_ISSUES = 8;

/**
 * Step 2: does the manifest line up with the images — BEFORE anything is
 * uploaded. Only filenames go to the server for this, so a typo in row 14 is
 * found in a second, not after a gigabyte upload.
 *
 * The start button lives here, directly under the explanation, so that when
 * it is disabled the reason is the thing right above it.
 */
export function PairingPanel({ state, onStart, busy }: { state: PairingState; onStart: () => void; busy: boolean }) {
  const [showAll, setShowAll] = useState(false);

  let body: ReactNode = null;
  let startLabel = "Check labels";
  let blockedReason: string | null = null;

  switch (state.kind) {
    case "waiting":
      blockedReason =
        state.missing === "both"
          ? "Choose a manifest and the label images to begin."
          : state.missing === "manifest"
            ? "Choose the manifest (.csv) to begin."
            : "Choose the label images to begin.";
      break;
    case "checking":
      blockedReason = "Checking that the manifest and images line up…";
      break;
    case "error":
      blockedReason = "The files could not be checked. See the message above.";
      body = (
        <div className="notice notice-error" role="alert">
          <p className="notice-title">
            <Icon name="cross" /> {state.error.title}
          </p>
          <p>{state.error.message}</p>
        </div>
      );
      break;
    case "ready": {
      const { report } = state;
      const issues = report.issues;
      const shown = showAll ? issues : issues.slice(0, FIRST_ISSUES);
      startLabel = `Check ${report.matched} ${report.matched === 1 ? "label" : "labels"}`;
      if (report.blocking) {
        blockedReason = "Nothing can be checked until the problems below are fixed in the manifest or the files.";
      }
      body = (
        <div
          className={`notice ${report.blocking ? "notice-error" : issues.length ? "notice-review" : "notice-ok"}`}
          role="status"
        >
          <p className="notice-title">
            <Icon name={report.blocking ? "cross" : issues.length ? "alert" : "check"} />
            <span>
              {report.blocking ? (
                "These files cannot be checked yet."
              ) : (
                <>
                  {report.matched} {report.matched === 1 ? "label" : "labels"} ready.
                  {issues.length > 0 && <> {issues.length} {issues.length === 1 ? "needs" : "need"} attention before you start:</>}
                </>
              )}
            </span>
          </p>
          {issues.length > 0 && (
            <>
              <ul className="issue-list">
                {shown.map((issue, i) => (
                  <li key={`${issue.kind}-${issue.row_number ?? ""}-${issue.filename ?? ""}-${i}`}>{issue.message}</li>
                ))}
              </ul>
              {issues.length > FIRST_ISSUES && (
                <button type="button" className="button button-quiet button-small" onClick={() => setShowAll((v) => !v)}>
                  {showAll ? "Show fewer" : `Show all ${issues.length}`}
                </button>
              )}
              {!report.blocking && (
                <p className="small">
                  You can start now: the {report.matched} matched {report.matched === 1 ? "label is" : "labels are"}{" "}
                  checked and the items above are left out. Or fix them first and choose the files again.
                </p>
              )}
            </>
          )}
          <p className="muted small">
            Manifest rows: {report.total_rows} · Images: {report.total_images}
          </p>
        </div>
      );
      break;
    }
  }

  const disabled = busy || blockedReason !== null;
  return (
    <div className="pairing">
      {body}
      <div className="start-row">
        <button
          type="button"
          className="button button-primary"
          disabled={disabled}
          aria-describedby={blockedReason ? "start-blocked" : undefined}
          onClick={onStart}
        >
          {startLabel}
        </button>
        {blockedReason && (
          <p id="start-blocked" className="muted">
            {blockedReason}
          </p>
        )}
      </div>
    </div>
  );
}
