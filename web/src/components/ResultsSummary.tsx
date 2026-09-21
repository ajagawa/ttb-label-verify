import type { ImageQuality, VerificationResult } from "../api/types";
import { formatSeconds } from "../lib/format";
import type { VerdictCounts } from "../lib/presentation";
import { Icon } from "./Icon";

/**
 * Top of the results: how long it took, and the tally.
 *
 * Requirement (Sarah Chen): the previous vendor tool failed at 30–40 s per
 * label, so the elapsed time is shown prominently on every result — the
 * number is proved, not promised. It is the server's wall-clock figure for the
 * whole request, not just the model time.
 */
export function ResultsSummary({ result, counts }: { result: VerificationResult; counts: VerdictCounts }) {
  const items: Array<{ n: number; text: string; tone: string; icon: Parameters<typeof Icon>[0]["name"] }> = [
    { n: counts.mismatch, text: counts.mismatch === 1 ? "mismatch" : "mismatches", tone: "mismatch", icon: "cross" },
    { n: counts.elevated, text: "to look at first", tone: "elevated", icon: "alert-strong" },
    { n: counts.review, text: counts.review === 1 ? "needs review" : "need review", tone: "review", icon: "alert" },
    { n: counts.unchecked, text: "not checked automatically", tone: "unchecked", icon: "dash" },
    { n: counts.match, text: counts.match === 1 ? "match" : "matches", tone: "match", icon: "check" },
  ];

  return (
    <div className="summary">
      <p className="elapsed">
        Checked in <strong>{formatSeconds(result.elapsed_ms)}</strong>
      </p>
      <ul className="tally" aria-label="Summary of results">
        {items
          .filter((item) => item.n > 0)
          .map((item) => (
            <li key={item.tone} className={`badge tone-${item.tone}`}>
              <Icon name={item.icon} />
              <span>
                {item.n} {item.text}
              </span>
            </li>
          ))}
      </ul>
      <p className="muted small">
        These are the tool's recommendations. You make the decision. · Rule set {result.ruleset_version} · Text
        reading took {formatSeconds(result.extraction_ms)}
      </p>
    </div>
  );
}

/**
 * The quality gate's guidance, in its own words. It says what is wrong with
 * the photo (glare, blur, angle) so the agent can ask the submitter for one
 * specific fix instead of a generic "please resubmit".
 */
export function QualityNotice({ quality }: { quality: ImageQuality }) {
  // Unassessed is not the same as good: say so rather than stay silent.
  if (quality.assessed === false) {
    return (
      <div className="notice notice-info">
        <p className="notice-title">
          <Icon name="info" />
          Photo quality was not assessed for this label
        </p>
        <p>Judge for yourself whether the photo is clear enough to rely on.</p>
      </div>
    );
  }
  if (quality.usable && quality.guidance.length === 0) return null;
  // Guidance is written for people; raw issue codes are only a fallback.
  const lines = quality.guidance.length > 0 ? quality.guidance : quality.issues;
  return (
    <div className={`notice ${quality.usable ? "notice-info" : "notice-review"}`}>
      <p className="notice-title">
        <Icon name={quality.usable ? "info" : "alert"} />
        {quality.usable ? "About this photo" : "This photo may be too poor to read reliably"}
      </p>
      {!quality.usable && (
        <p>Results below may be incomplete. A better photo will give a more dependable check.</p>
      )}
      {lines.length > 0 && (
        <ul>
          {lines.map((g) => (
            <li key={g}>{g}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
