/**
 * How each verdict, check outcome and diff entry is presented.
 *
 * Requirement (WCAG 2.1 AA / discovery): status is never carried by colour
 * alone. Every state here has an icon AND a text label AND a tone; the tone
 * only picks colours in CSS. Keeping this mapping pure and in one place means
 * it is unit-tested and every component says the same words for the same state.
 *
 * Copy is deliberately hedged ("appears to", "please look"): the tool
 * recommends and the agent decides, so no label claims more certainty than
 * the tool has.
 */

import { Tier, type BatchItem, type CheckOutcome, type FieldResult, type TextDifference } from "../api/types";

export type Tone = "match" | "review" | "elevated" | "mismatch" | "unchecked";
export type IconName =
  | "check"
  | "alert"
  | "alert-strong"
  | "cross"
  | "dash"
  | "info"
  | "no-image"
  | "clock"
  | "reading";

export interface Presentation {
  tone: Tone;
  icon: IconName;
  /** Short status text shown in the badge. */
  label: string;
  /** One line explaining what the status means for the agent. */
  hint: string;
}

/**
 * Whether the tool never looked at this field.
 *
 * `automated` on the result is the authority. The ruleset flag is consulted
 * too, so that a field the ruleset declares unimplemented is never shown as
 * checked even if a result somehow claimed otherwise — when the two sources
 * disagree, the display errs toward "verify manually".
 */
export function isUnimplemented(field: FieldResult, implementedById?: Record<string, boolean>): boolean {
  if (field.automated === false) return true;
  return Boolean(implementedById && implementedById[field.field_id] === false);
}

export function verdictPresentation(field: FieldResult, implementedById?: Record<string, boolean>): Presentation {
  // Checked before the verdict: an unimplemented field comes back as REVIEW,
  // and presenting it as an ordinary "needs review" would imply the tool
  // looked. It never looked — and it must never read as passing either.
  if (isUnimplemented(field, implementedById)) {
    return {
      tone: "unchecked",
      icon: "dash",
      label: "Not checked automatically — verify manually",
      hint: "This prototype does not check this item yet. Check it on the label yourself.",
    };
  }
  switch (field.verdict) {
    case "MATCH":
      return {
        tone: "match",
        icon: "check",
        label: "Match",
        hint: "The label appears to agree with the application.",
      };
    case "MISMATCH":
      return {
        tone: "mismatch",
        icon: "cross",
        label: "Mismatch",
        hint: "The label appears to differ from the application. Compare the highlighted area.",
      };
    case "REVIEW":
    default:
      // `elevated` raises prominence without escalating the verdict: it is
      // still the agent's call, so it must not look like a MISMATCH.
      if (field.elevated) {
        return {
          tone: "elevated",
          icon: "alert-strong",
          label: "Needs review — look at this first",
          hint: "A mandatory item was not found even though the photo is clear.",
        };
      }
      return {
        tone: "review",
        icon: "alert",
        label: "Needs review",
        hint: "The tool is not sure. Please look at the label.",
      };
  }
}

export interface CheckPresentation {
  tone: "match" | "mismatch" | "info" | "unchecked";
  icon: IconName;
  label: string;
}

/**
 * Advisory and not-evaluable checks are shown honestly: an advisory is a
 * measurement for the agent's judgement, and not-evaluable is "couldn't
 * check" — neither is ever shown as pass or fail.
 */
export function checkPresentation(outcome: CheckOutcome): CheckPresentation {
  switch (outcome) {
    case "pass":
      return { tone: "match", icon: "check", label: "Passed" };
    case "fail":
      return { tone: "mismatch", icon: "cross", label: "Did not pass" };
    case "advisory":
      return { tone: "info", icon: "info", label: "Measurement — your judgement" };
    case "not_evaluable":
    default:
      return { tone: "unchecked", icon: "dash", label: "Couldn't check" };
  }
}

export function differenceLabel(diff: TextDifference): string {
  switch (diff.kind) {
    case "missing":
      return "Missing from label";
    case "extra":
      return "Extra on label";
    case "substituted":
      return "Different wording";
    default:
      return "Difference";
  }
}

export interface VerdictCounts {
  match: number;
  review: number;
  elevated: number;
  mismatch: number;
  unchecked: number;
}

export function countTones(fields: FieldResult[], implementedById?: Record<string, boolean>): VerdictCounts {
  const counts: VerdictCounts = { match: 0, review: 0, elevated: 0, mismatch: 0, unchecked: 0 };
  for (const field of fields) counts[verdictPresentation(field, implementedById).tone] += 1;
  return counts;
}

export type TierTone = Tone | "unreadable" | "pending" | "running";

export interface TierPresentation {
  tone: TierTone;
  icon: IconName;
  /** The server's `tier_label`, verbatim — rules/triage.py owns the wording. */
  label: string;
}

/**
 * A worklist row's badge. The words come from the server (`tier_label`); only
 * the icon and colour are chosen here, and they reuse the single-label tones
 * so "differs" looks the same on both screens. Unreadable gets its own
 * no-image icon: it is not a finding about the label, it is a missing photo.
 */
export function tierPresentation(item: Pick<BatchItem, "state" | "tier" | "tier_label">): TierPresentation {
  switch (item.state) {
    case "skipped":
      // Never checked because the batch was stopped: "not checked", in the
      // same dashed style as a field the tool does not check — never
      // "waiting", which would promise a result that is not coming.
      return { tone: "unchecked", icon: "dash", label: "Not checked — the batch was stopped" };
    case "running":
      // A static icon, deliberately: a spinner keeps spinning if the
      // connection drops, claiming progress nobody can see.
      return { tone: "running", icon: "reading", label: "Reading now" };
    case "pending":
      return { tone: "pending", icon: "clock", label: "Waiting to be checked" };
  }
  if (item.tier == null) {
    // A finished item without a tier should not happen; say "look" rather than guess.
    return { tone: "review", icon: "alert", label: item.tier_label ?? "Needs a look" };
  }
  const label = item.tier_label ?? "Needs a look";
  switch (item.tier) {
    case Tier.ASSERTED:
      return { tone: "mismatch", icon: "cross", label };
    case Tier.MISSING_MANDATORY:
      return { tone: "elevated", icon: "alert-strong", label };
    case Tier.UNREADABLE:
      return { tone: "unreadable", icon: "no-image", label };
    case Tier.REVIEW:
      return { tone: "review", icon: "alert", label };
    case Tier.UNCHECKED:
      return { tone: "unchecked", icon: "dash", label };
    case Tier.CLEAR:
      return { tone: "match", icon: "check", label };
    default:
      // A tier this build does not know (a newer server): show the server's
      // words with a neutral "look" style rather than guessing at severity.
      return { tone: "review", icon: "alert", label };
  }
}
