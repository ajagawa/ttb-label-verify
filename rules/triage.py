"""Worst-first ordering for a batch worklist.

Batch is throughput-bound, not latency-bound: nobody is watching a progress bar
for 300 labels. Its value is triage — putting an agent's attention on the labels
that need it, in the order they need it, instead of in upload order. This module
is the single definition of that order.

It is deliberately one module used everywhere. The API sorts the worklist with
it, the evaluator measures ranking quality with it, and the frontend displays
the tier it assigns. If each computed its own order, the harness could report a
ranking the agent never sees.

Deterministic and pure: same results in, same order out, ties broken by
filename. An order that shuffled between two page loads would make "work down
the list" impossible.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from rules.results import FieldResult, Verdict, VerificationResult


class Tier(enum.IntEnum):
    """Why a label needs attention, most urgent first.

    The order is a judgement, recorded here so it can be argued with:

    *   **ASSERTED** first. The tool located a value and it clearly differs — a
        violation with evidence attached. The most actionable thing in the list.
    *   **MISSING_MANDATORY** next. A mandatory field (the warning statement)
        was not found on an image the quality gate judged good. Rejection-grade
        in practice, but not asserted, because the tool could not see it.
    *   **UNREADABLE** next. The image could not be processed at all, so
        *nothing* on the label has been checked. Needs a person, and usually a
        new photograph — which is a round trip, so it should start early.
    *   **REVIEW** next. Something is uncertain: a near-miss, a low-confidence
        read, a residual character difference. Most will turn out fine.
    *   **UNCHECKED** next. Every automated check passed, but a field has no
        automated check and must be verified by hand.
    *   **CLEAR** last. Every field matched.
    """

    ASSERTED = 0
    MISSING_MANDATORY = 1
    UNREADABLE = 2
    REVIEW = 3
    UNCHECKED = 4
    CLEAR = 5

    @property
    def label(self) -> str:
        """Plain-language name, as shown to the agent."""
        return {
            Tier.ASSERTED: "Differs from the application",
            Tier.MISSING_MANDATORY: "Mandatory statement not found",
            Tier.UNREADABLE: "Could not read this image",
            Tier.REVIEW: "Needs a look",
            Tier.UNCHECKED: "Some items not checked automatically",
            Tier.CLEAR: "All clear",
        }[self]


@dataclass(frozen=True, slots=True)
class TriageSummary:
    """Everything the worklist needs to place and describe one label."""

    tier: Tier
    mismatch_count: int
    elevated_count: int
    review_count: int
    unchecked_count: int
    lowest_score: float
    #: The first thing to look at, in one line. Taken from the most severe
    #: field so that an agent scanning the list knows why each row is there
    #: without opening it.
    headline: str | None

    @property
    def needs_attention(self) -> bool:
        return self.tier is not Tier.CLEAR


def summarise(result: VerificationResult | None, error: str | None = None) -> TriageSummary:
    """Place one label in the worklist.

    Args:
        result: The verification result, or None when the image failed.
        error: Why the image failed, when it did.

    Returns:
        Its tier, counts and headline.

    Raises:
        ValueError: when neither a result nor an error is supplied — a label
            with no outcome has no defensible place in the list, and silently
            ranking it as clear would be the worst possible guess.
    """
    if result is None:
        if not error:
            raise ValueError("a label needs either a result or an error to be triaged")
        return TriageSummary(
            tier=Tier.UNREADABLE,
            mismatch_count=0,
            elevated_count=0,
            review_count=0,
            unchecked_count=0,
            lowest_score=0.0,
            headline=error,
        )

    fields = result.fields
    mismatches = [f for f in fields if f.verdict is Verdict.MISMATCH]
    elevated = [f for f in fields if f.verdict is Verdict.REVIEW and f.elevated]
    # An unchecked field carries REVIEW too; it is counted separately so that a
    # label whose only flag is "not automated" is not mistaken for a label the
    # tool was uncertain about.
    unchecked = [f for f in fields if not f.automated]
    reviews = [f for f in fields if f.verdict is Verdict.REVIEW and not f.elevated and f.automated]

    if mismatches:
        tier, lead = Tier.ASSERTED, mismatches[0]
    elif elevated:
        tier, lead = Tier.MISSING_MANDATORY, elevated[0]
    elif reviews:
        tier, lead = Tier.REVIEW, reviews[0]
    elif unchecked:
        tier, lead = Tier.UNCHECKED, unchecked[0]
    else:
        tier, lead = Tier.CLEAR, None

    return TriageSummary(
        tier=tier,
        mismatch_count=len(mismatches),
        elevated_count=len(elevated),
        review_count=len(reviews),
        unchecked_count=len(unchecked),
        lowest_score=result.lowest_score,
        headline=_headline(lead),
    )


def _headline(field: FieldResult | None) -> str | None:
    """ "Brand Name: Label reads …" — the field's own reason, labelled."""
    if field is None:
        return None
    return f"{field.label}: {field.reason}"


def sort_key(summary: TriageSummary, filename: str) -> tuple:
    """Total order over labels, most urgent first.

    Within a tier, more problems before fewer, then the weakest single field
    before stronger ones, then filename. The filename tie-break is what makes
    the order stable across page loads; without it, labels with identical
    outcomes could swap places every time the list is fetched.
    """
    problems = (
        summary.mismatch_count
        + summary.elevated_count
        + summary.review_count
        + summary.unchecked_count
    )
    return (int(summary.tier), -problems, summary.lowest_score, filename.casefold(), filename)
