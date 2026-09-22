"""Verification result types — the output contract.

These models are what the API returns and what the frontend renders. They are
also where the system's central safety invariant is enforced structurally
rather than by convention:

    A field that was not located can never carry a MISMATCH verdict.

MISMATCH asserts that the label is wrong. Absence of a match asserts only that
the tool did not find something, which on a stylized label usually means OCR
failed. Asserting a violation because extraction failed is the worst error this
system can make — worse than missing a violation, because it actively misleads
an agent who is trusting the tool. `FieldResult` refuses to be constructed that
way, so no future strategy implementation can reintroduce the failure by
accident.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(enum.StrEnum):
    """Three states, not two.

    Binary pass/fail forces every ambiguity into one of two wrong answers. The
    discovery notes gave the canonical example: "STONE'S THROW" on the label
    against "Stone's Throw" in the application is technically a mismatch and
    obviously the same thing. REVIEW is where judgment belongs.
    """

    MATCH = "MATCH"
    REVIEW = "REVIEW"
    MISMATCH = "MISMATCH"


class Strategy(enum.StrEnum):
    """How a field's value was located. Recorded for the audit trail.

    Worth surfacing to the agent as well as to diagnostics: a value found by
    `vocabulary` (matched against known designations rather than against the
    application record) means something different from one found by `anchored`
    search, and an agent reading the result should be able to tell.
    """

    ANCHORED = "anchored"  # fuzzy match against the expected value
    PATTERN = "pattern"  # syntactic match (ABV, volume)
    POSITIONAL = "positional"  # located by spatial prior
    VOCABULARY = "vocabulary"  # matched against a controlled vocabulary
    NOT_FOUND = "not_found"  # nothing cleared the threshold


class CheckOutcome(enum.StrEnum):
    """Result of an individual formatting check within a field."""

    PASS = "pass"  # nosec B105 - a verdict value, not a credential
    FAIL = "fail"
    ADVISORY = "advisory"  # measured, reported, not adjudicated
    NOT_EVALUABLE = "not_evaluable"  # prerequisite absent — e.g. no label dimensions


class BoxModel(BaseModel):
    """A region of the label image, in CSS-friendly form for the overlay."""

    model_config = ConfigDict(frozen=True)

    left: float
    top: float
    width: float
    height: float


class CheckResult(BaseModel):
    """One formatting check — chiefly the § 16.22 warning checks.

    `advisory` checks are measurements. They are shown to the agent with their
    numbers and never change a field's verdict on their own, because they are
    derived from pixels and a confident claim from an estimate is misleading.
    NOT_EVALUABLE is reported explicitly rather than silently skipped, so that
    "we could not check this" never reads as "this passed".
    """

    model_config = ConfigDict(frozen=True)

    id: str
    outcome: CheckOutcome
    citation: str
    summary: str
    detail: str | None = None
    measurement: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TextDifference(BaseModel):
    """One token-level difference in the warning statement diff.

    The word-level diff is the feature that makes a warning-text finding
    actionable: "the statement differs" sends an agent hunting, whereas
    "'may cause health problems' appears as 'can cause health problems'" is
    something they can act on immediately.
    """

    model_config = ConfigDict(frozen=True)

    position: int
    expected: str | None  # None where the label has an extra token
    found: str | None  # None where a required token is missing
    kind: str  # "substituted" | "missing" | "extra"


class FieldResult(BaseModel):
    """The verdict for one checked field, with its supporting evidence.

    Everything an agent needs to act is here: what was expected, what was
    found, where on the label it was found, how confident the tool is, why it
    reached its verdict, and which rule governs.
    """

    model_config = ConfigDict(frozen=True)

    field_id: str
    label: str
    verdict: Verdict
    strategy: Strategy

    expected: str | None  # from the application record
    found: str | None  # as printed, raw — never the canonical form
    bbox: BoxModel | None  # region to highlight; None when not located

    score: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float = Field(ge=0.0, le=1.0)

    #: Plain-language explanation. Shown verbatim to the agent, so it is
    #: written for a reader rather than for a log.
    reason: str
    citation: str
    citation_url: str | None = None

    #: Raised prominence in the UI without escalating the verdict. Used for the
    #: one case that pulls against "not-found is never a mismatch": a mandatory
    #: warning statement genuinely absent from a good-quality image. It stays
    #: REVIEW — the agent looks, the tool does not assert — but it must not be
    #: presented at the same weight as an ordinary low-confidence read.
    elevated: bool = False

    #: False when no strategy exists for this field and the tool did not look
    #: at it at all. Distinct from NOT_FOUND on an automated field, which means
    #: the tool looked and could not locate the value. Both carry REVIEW and a
    #: zero score, so without this flag the two are indistinguishable — and
    #: telling an agent "no warning statement located" when the check never ran
    #: is a false statement about what was verified.
    automated: bool = True

    checks: tuple[CheckResult, ...] = ()
    differences: tuple[TextDifference, ...] = ()

    @model_validator(mode="after")
    def not_found_is_never_a_mismatch(self) -> FieldResult:
        """Enforce the system's central safety invariant structurally.

        A convention documented in prose survives exactly as long as everyone
        who adds a field strategy reads the prose. A validator survives
        indefinitely. Any strategy that tries to construct a NOT_FOUND
        MISMATCH raises at construction, in that strategy's own tests, rather
        than shipping a confident false accusation.
        """
        if self.strategy is Strategy.NOT_FOUND and self.verdict is Verdict.MISMATCH:
            raise ValueError(
                f"field {self.field_id!r}: a field that was not located cannot be a MISMATCH. "
                "MISMATCH asserts the label is wrong; failing to find text asserts only that "
                "extraction failed. Use REVIEW (with elevated=True if the field is mandatory "
                "and the image quality was good)."
            )
        if self.verdict is Verdict.MISMATCH and self.bbox is None:
            raise ValueError(
                f"field {self.field_id!r}: a MISMATCH must carry the region it was read from. "
                "An assertion the agent cannot visually verify is not actionable."
            )
        return self


class ImageQuality(BaseModel):
    """What the quality gate concluded, in the form the agent sees.

    `guidance` is the part that matters operationally. The discovery notes
    described agents rejecting unreadable images and asking for better ones —
    a round trip. Telling the submitter what specifically was wrong turns that
    into a single fix.
    """

    model_config = ConfigDict(frozen=True)

    usable: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: tuple[str, ...] = ()
    guidance: tuple[str, ...] = ()
    measurements: dict[str, float] = Field(default_factory=dict)

    #: False when no assessment was made — an extraction that did not come
    #: through the preprocessing pipeline. Consumers that would otherwise read
    #: `score` must check this first: an unassessed quality is not a good one,
    #: and treating it as good is how a guard silently stops guarding.
    assessed: bool = True


class ApplicationRecord(BaseModel):
    """The application-side values a label is checked against.

    No COLA lookup: this arrives with the request. That was the scope boundary
    set during discovery, and it is also what keeps extraction honest — a
    provider never sees these values and so cannot bias its reading toward
    them.
    """

    model_config = ConfigDict(frozen=True)

    brand_name: str
    class_type: str
    alcohol_content: str  # as written on the form, e.g. "45% Alc./Vol."
    net_contents: str  # e.g. "750 mL"
    beverage_class: str = "distilled_spirits"

    #: Physical label width in millimetres. Optional, and its absence is the
    #: reason the § 16.22(b) type-size check reports NOT_EVALUABLE: a
    #: photograph carries no scale reference, so millimetres cannot be derived
    #: from pixels without it.
    label_width_mm: float | None = Field(default=None, gt=0)


class VerificationResult(BaseModel):
    """The complete result for one label.

    `ruleset_version` is not bookkeeping. TTB requirements are under active
    rulemaking, and a verdict recorded today has to be reproducible against the
    rules that were in force when it was made.
    """

    model_config = ConfigDict(frozen=True)

    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ruleset_version: str
    provider_name: str

    quality: ImageQuality
    fields: tuple[FieldResult, ...]

    image_width: int
    image_height: int

    #: Wall-clock milliseconds for the whole request. Displayed on every
    #: result: the previous vendor pilot failed on latency, and these users
    #: have reason to want the number proved rather than promised.
    elapsed_ms: float
    extraction_ms: float

    #: Verbose per-verification record — every candidate, every score, which
    #: strategy won, what was masked. Gated behind a flag; absent in normal
    #: responses.
    diagnostics: dict | None = None

    @property
    def flagged_count(self) -> int:
        return sum(1 for f in self.fields if f.verdict is not Verdict.MATCH)

    @property
    def lowest_score(self) -> float:
        """Used to order a batch worklist worst-first.

        Batch is throughput-bound rather than latency-bound, so the value it
        adds is triage: an agent's attention should land on the labels that
        need it, not on whatever arrived first.
        """
        return min((f.score for f in self.fields), default=0.0)

    def field(self, field_id: str) -> FieldResult:
        for result in self.fields:
            if result.field_id == field_id:
                return result
        raise KeyError(f"no result for field {field_id!r}")
