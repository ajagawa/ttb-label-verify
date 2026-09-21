"""Field comparison strategies.

This is the deterministic half of the system. Nothing here performs I/O, calls
a model, or touches the network: given the same spans and the same application
record, it produces the same verdict every time. That is what makes a finding
auditable, and it is why the test suite runs in milliseconds with no OCR engine
installed.

One strategy — `anchored` — is implemented here in full. It is the reference
pattern every other strategy follows: score the candidate pool, apply the
field's thresholds, and return a `FieldResult` that carries its own evidence.
The remaining strategies are declared with their contracts and raise
`StrategyNotImplemented` until they are written, so that an unfinished field
fails loudly in its own tests rather than silently returning a passing verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from extraction.geometry import BBox
from extraction.normalize import canonicalize, fold_confusable
from extraction.spans import Span
from rules.results import BoxModel, FieldResult, ImageQuality, Strategy, Verdict
from rules.schema import BeverageClass, FieldRule, RuleSet


class StrategyNotImplemented(NotImplementedError):
    """A declared field strategy that has not been written yet.

    Raised rather than returning a neutral result. A field that quietly
    resolves to MATCH because nobody implemented its comparator is the single
    most dangerous kind of gap in a compliance tool: the output looks complete
    and is not.
    """


@dataclass(frozen=True, slots=True)
class ScoredSpan:
    """A candidate with its composite score, retained for the diagnostic record."""

    span: Span
    score: float
    similarity: float
    length_penalty: float
    confidence_factor: float


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to see.

    Deliberately narrow. A strategy gets the candidate pool, the expected value
    and its own rules — not the image, not the other fields' results, not the
    provider. Widening this is how a deterministic layer stops being
    deterministic.
    """

    field_rule: FieldRule
    ruleset: RuleSet
    beverage: BeverageClass
    spans: list[Span]
    expected: str
    #: Region of an already-resolved field, for strategies that use a spatial
    #: prior (class/type sits adjacent to the brand name).
    anchor_bbox: BBox | None = None

    #: The engine's image-quality summary — `usable`, `score`, `issues`. Not
    #: the image: a deterministic summary computed once before the resolution
    #: loop, so passing it keeps this layer reproducible.
    #:
    #: It may only be used to WITHDRAW an assertion, never to add one. A
    #: strategy that becomes *more* willing to assert a violation because an
    #: image scored well would be inferring compliance from photography. On a
    #: low-contrast photograph, "GOVERNMENT WARNiNG" is overwhelmingly more
    #: likely to be a misread than a label genuinely printed with one
    #: lowercase letter, so the finding is reported rather than asserted.
    quality: ImageQuality | None = None


def length_penalty_factor(span: Span, expected: str, penalty_weight: float) -> float:
    """Discount candidates substantially longer than the expected value.

    A five-word span scoring 0.90 against a two-word brand is usually the
    bottler statement ("Bottled by OLD TOM DISTILLERY"), not the brand display.
    Both genuinely contain the brand name, so neither is wrong — but the
    display instance is the one an agent wants highlighted, and without this
    the longer span frequently wins on raw similarity.

    Shorter-than-expected spans are *not* penalised here: a partial match
    already scores lower on similarity, and penalising it twice pushes
    legitimate fragments below the threshold.

    Args:
        span: The candidate.
        expected: The application record's value.
        penalty_weight: Per-excess-word discount, from the field rule. Zero
            disables the penalty entirely.

    Returns:
        A multiplier in (0, 1].
    """
    if penalty_weight <= 0:
        return 1.0

    expected_words = len(canonicalize(expected).split())
    if expected_words == 0:
        return 1.0

    excess = max(0, span.word_count - expected_words)
    return max(0.1, 1.0 - (excess * penalty_weight))


def confidence_factor(span: Span, weight: float) -> float:
    """Blend OCR confidence into the score without letting it dominate.

    Text similarity is the primary signal; confidence modulates it. At the
    default weight of 0.30, a perfectly matching span read at 50% confidence
    still scores 0.85 — flagged for review, not discarded — while a
    high-confidence read is not inflated above its similarity.

    Args:
        span: The candidate.
        weight: Share of the score attributable to confidence, from the field
            rule.

    Returns:
        A multiplier in (0, 1].
    """
    weight = max(0.0, min(1.0, weight))
    return (1.0 - weight) + (weight * span.mean_confidence)


def text_similarity(candidate_canonical: str, expected_canonical: str) -> float:
    """Similarity between two canonical strings, insensitive to word spacing.

    The better of two measures is taken:

    *   `token_sort_ratio`, so a candidate whose words arrived in a different
        order than expected is not punished for OCR line-ordering quirks.
    *   The plain ratio of both strings with **all spaces removed**.

    The second exists because of a failure found only when the real engine ran
    against real pixels. PP-OCR reads letterspaced display type as a single
    token: "OLD TOM DISTILLERY" comes back as "OLDTOMDISTILLERY", and
    "Kentucky Straight Bourbon Whiskey" as "KentuckyStraightBourbonWhiskey".
    Against a token-based measure that is three or four tokens versus one, and
    the brand name — the largest, clearest text on the label — scored below the
    match threshold and was reported as "could not locate". Brand-name accuracy
    was 24% against a corpus where every brand was plainly legible.

    Removing spaces from both sides is the right fix rather than a patch,
    because word spacing carries no meaning for these fields. "STONE'S THROW"
    and "STONES THROW" already converge in canonicalisation for the same
    reason. What spacing *would* matter for — the warning statement's wording —
    does not come through here: that comparison is token-level and lives in
    rules/warning.py.

    Args:
        candidate_canonical: The span's canonical text.
        expected_canonical: The application value, canonicalised the same way.

    Returns:
        Similarity in [0, 1].
    """
    token_sort = fuzz.token_sort_ratio(candidate_canonical, expected_canonical)
    despaced = fuzz.ratio(candidate_canonical.replace(" ", ""), expected_canonical.replace(" ", ""))
    return max(token_sort, despaced) / 100.0


def score_span(span: Span, expected: str, rule: FieldRule) -> ScoredSpan:
    """Composite score for one candidate against an expected value.

    score = similarity x confidence_factor x length_penalty
    """
    similarity = text_similarity(span.canonical_text, canonicalize(expected))
    penalty = length_penalty_factor(span, expected, rule.length_penalty)
    confidence = confidence_factor(span, rule.confidence_weight)

    return ScoredSpan(
        span=span,
        score=similarity * penalty * confidence,
        similarity=similarity,
        length_penalty=penalty,
        confidence_factor=confidence,
    )


def rank_candidates(context: StrategyContext) -> list[ScoredSpan]:
    """Score the whole pool, best first.

    The full ranking is returned rather than just the winner because the
    diagnostic record needs the runners-up: tuning a threshold without seeing
    what almost won is guesswork.
    """
    scored = [score_span(span, context.expected, context.field_rule) for span in context.spans]
    return sorted(scored, key=lambda s: s.score, reverse=True)


def verdict_for(score: float, ocr_confidence: float, field_id: str, ruleset: RuleSet) -> Verdict:
    """Map a score to one of the three verdicts.

    Low OCR confidence forces REVIEW regardless of text score: a perfect match
    against text the engine is unsure it read correctly is not evidence of
    anything.
    """
    thresholds = ruleset.thresholds_for(field_id)

    if ocr_confidence < ruleset.defaults.min_ocr_confidence:
        return Verdict.REVIEW
    if score >= thresholds.match:
        return Verdict.MATCH
    if score <= thresholds.mismatch:
        return Verdict.MISMATCH
    return Verdict.REVIEW


def not_found_result(context: StrategyContext, *, elevated: bool = False) -> FieldResult:
    """The result for a field nothing matched.

    Always REVIEW, never MISMATCH — the invariant enforced in
    `rules.results.FieldResult`. `elevated` raises the prominence for a
    mandatory field absent from a good-quality image without escalating the
    verdict: the agent looks, the tool does not assert.
    """
    rule = context.field_rule
    return FieldResult(
        field_id=rule.id,
        label=rule.label,
        verdict=Verdict.REVIEW,
        strategy=Strategy.NOT_FOUND,
        expected=context.expected,
        found=None,
        bbox=None,
        score=0.0,
        ocr_confidence=0.0,
        reason=(
            f"Could not locate the {rule.label.lower()} on this image. "
            "This may mean the text is stylized or the image is unclear rather than "
            "that the label is non-compliant — verify manually."
        ),
        citation=rule.citation,
        citation_url=rule.citation_url,
        elevated=elevated,
    )


@dataclass(frozen=True, slots=True)
class ResidualDifference:
    """What still differs between two name-like values after normalisation.

    `count` drives the verdict. `described` is for the agent, and lists only
    differences that `count` also saw — see `fold_confusable` for why that
    matters.
    """

    count: int
    described: tuple[str, ...]

    @property
    def any(self) -> bool:
        return self.count > 0


def residual_difference(found: str, expected: str) -> ResidualDifference:
    """Character edits separating two values once meaningless variation is gone.

    Both sides are canonicalised (case, punctuation, accents and the OCR
    confusion classes removed) and stripped of spaces, then compared by
    Levenshtein distance — an absolute count of inserted, deleted and
    substituted characters.

    The count is what fixes the long-name false negative. Similarity ratios
    divide by length, so one wrong letter in "OLD TOM DISTILLERY" cost less than
    one wrong letter in "OLD TOM", and the longer brand cleared the match
    threshold. A count does not divide by anything: one surviving difference is
    one difference, on a six-letter name or a thirty-letter designation.

    Args:
        found: The label's reading, raw.
        expected: The application's value, raw.

    Returns:
        The count, plus up to three plain-language descriptions of what differs.

    Examples:
        >>> residual_difference("OLD TON DISTILLERY", "OLD TOM DISTILLERY").count
        1
        >>> residual_difference("STONE'S THROW", "Stone's Throw").count
        0
        >>> residual_difference("OLD T0M DISTILLERY", "OLD TOM DISTILLERY").count
        0
    """
    folded_found = canonicalize(found).replace(" ", "")
    folded_expected = canonicalize(expected).replace(" ", "")
    count = Levenshtein.distance(folded_expected, folded_found)
    if count == 0:
        return ResidualDifference(count=0, described=())

    # Describe from the UNfolded forms, so the agent reads the letters actually
    # printed ("M" and "N"), not their confusion-class representatives. Pairs
    # that fold together are skipped: the count above did not include them, and
    # listing a difference the verdict ignored would misstate why it was raised.
    plain_found = canonicalize(found, fold_ocr_confusions=False).replace(" ", "")
    plain_expected = canonicalize(expected, fold_ocr_confusions=False).replace(" ", "")

    described: list[str] = []
    for op in Levenshtein.editops(plain_expected, plain_found):
        if op.tag == "replace":
            was, now = plain_expected[op.src_pos], plain_found[op.dest_pos]
            if fold_confusable(was) == fold_confusable(now):
                continue
            described.append(
                f"\u201c{was}\u201d in the application reads \u201c{now}\u201d on the label"
            )
        elif op.tag == "delete":
            described.append(f"\u201c{plain_expected[op.src_pos]}\u201d is missing from the label")
        elif op.tag == "insert":
            described.append(f"the label has an extra \u201c{plain_found[op.dest_pos]}\u201d")
        if len(described) == 3:
            break

    return ResidualDifference(count=count, described=tuple(described))


# ---------------------------------------------------------------------------
# Strategy: anchored  (reference implementation)
# ---------------------------------------------------------------------------


def anchored_strategy(context: StrategyContext) -> FieldResult:
    """Fuzzy search for a known expected value. Used for the brand name.

    This is the framing that makes local OCR viable: the task is not "what does
    this label say?" but "does OLD TOM DISTILLERY appear on it, and where?".
    A misread of `OID TOM DISTILERY` scores about 0.95 against the known target,
    so the value is *located* confidently where open-ended transcription would
    have failed.

    Locating is not certifying, though. `OID` for `OLD` is an OCR confusion the
    fold table absorbs, but the missing `L` in `DISTILERY` is not, and from the
    text alone it is indistinguishable from a label genuinely misspelled. So
    that reading resolves to REVIEW with the missing letter named, not to
    MATCH — see `residual_difference` for why a count, not the score, decides.

    When nothing clears the mismatch floor, the result is not_found rather than
    a MISMATCH built from the best of a bad pool. Stylized and script lettering
    is where this field fails, and failing to read display type is not evidence
    that the brand name is wrong.
    """
    if not context.spans:
        return not_found_result(context)

    ranked = rank_candidates(context)
    best = ranked[0]

    thresholds = context.ruleset.thresholds_for(context.field_rule.id)
    if best.score <= thresholds.mismatch:
        # Nothing plausible. Reporting the best of a bad pool as a MISMATCH
        # would assert the label is wrong on the strength of a poor match.
        return not_found_result(context)

    verdict = verdict_for(
        best.score, best.span.mean_confidence, context.field_rule.id, context.ruleset
    )

    # A score high enough to MATCH is not enough on its own. If any character
    # difference survives normalisation, the tool cannot tell a misread from a
    # different name, so it reports rather than certifies. Only MATCH is
    # affected: this can move a verdict to REVIEW, never to MISMATCH.
    residual = residual_difference(best.span.raw_text, context.expected)
    budget = context.ruleset.tuning.names.max_residual_character_edits
    reason = _explain_anchored(context, best, verdict)
    if verdict is Verdict.MATCH and residual.count > budget:
        verdict = Verdict.REVIEW
        reason = explain_residual(best.span.raw_text, context.expected, residual)

    return FieldResult(
        field_id=context.field_rule.id,
        label=context.field_rule.label,
        verdict=verdict,
        strategy=Strategy.ANCHORED,
        expected=context.expected,
        found=best.span.raw_text,
        bbox=BoxModel(**best.span.bbox.to_dict()),
        score=round(best.score, 4),
        ocr_confidence=round(best.span.mean_confidence, 4),
        reason=reason,
        citation=context.field_rule.citation,
        citation_url=context.field_rule.citation_url,
    )


def explain_residual(found: str, expected: str, residual: ResidualDifference) -> str:
    """Tell the agent exactly which characters differ, and why it matters.

    Public so that other name-like strategies (class/type) phrase the same
    finding the same way. An agent reading two differently worded explanations
    for one kind of finding would reasonably wonder whether they mean different
    things.
    """
    letters = "character" if residual.count == 1 else "characters"
    detail = "; ".join(residual.described)
    detail = f" ({detail})" if detail else ""
    return (
        f"Label reads \u201c{found}\u201d; the application states \u201c{expected}\u201d. "
        f"They differ by {residual.count} {letters}{detail}. This may be a misread, or the "
        "label may genuinely differ \u2014 confirm against the artwork."
    )


def _explain_anchored(context: StrategyContext, best: ScoredSpan, verdict: Verdict) -> str:
    """Plain-language reasoning, written for an agent rather than a log.

    Every flag an agent sees costs them attention, so the explanation has to
    say what to look at. "Score 0.83" does not; "differs only in
    capitalisation" does.
    """
    found, expected = best.span.raw_text, context.expected
    label = context.field_rule.label.lower()

    if verdict is Verdict.MATCH:
        if found == expected:
            return "Label matches the application record exactly."
        if found.casefold() == expected.casefold():
            return (
                f"Matches the application record, differing only in capitalisation "
                f"(label reads “{found}”)."
            )
        return f"Matches the application record (label reads “{found}”)."

    if verdict is Verdict.MISMATCH:
        return (
            f"Label reads “{found}”; the application states "
            f"“{expected}”. Verify against the submitted artwork."
        )

    if best.span.mean_confidence < context.ruleset.defaults.min_ocr_confidence:
        return (
            f"Read the {label} as “{found}” but with low confidence "
            f"({best.span.mean_confidence:.0%}). The text may be stylized or unclear — "
            "confirm visually."
        )

    return (
        f"Label reads “{found}” against “{expected}” in the application. "
        "Close but not an exact match — confirm this is the same product."
    )


# ---------------------------------------------------------------------------
# Strategies still to be written
# ---------------------------------------------------------------------------
#
# Each raises rather than returning a neutral result. See StrategyNotImplemented.


def alcohol_content_strategy(context: StrategyContext) -> FieldResult:
    """Pattern search for ABV and proof, compared numerically.

    Contract:

    *   Match against `Span.numeric_text`, never `canonical_text`. The
        canonical form folds digits onto letters and would turn a genuine
        46%/45% discrepancy into an apparent match — the precise failure this
        tool exists to prevent.
    *   Collect every pattern hit. `45% Alc./Vol. (90 Proof)` matches three.
    *   Cross-validate: proof is exactly twice ABV. Disagreement between the
        two is an internal inconsistency on the label and is reportable in its
        own right, even when the ABV matches the application.
    *   Never fuzzy-forgive a numeric difference. Annotate it instead:
        "label reads 46%, application states 45%; single-digit OCR confusion is
        possible, verify visually."
    *   Tolerance comes from `context.beverage.alcohol_content`.
    """
    raise StrategyNotImplemented(
        "alcohol_content strategy is not implemented. See docs/field-assignment.md, "
        "'Alcohol content' and 'Numeric fields are never fuzzy-forgiven'."
    )


def net_contents_strategy(context: StrategyContext) -> FieldResult:
    """Pattern search for a volume, normalised to millilitres before comparison.

    Contract:

    *   `750 mL`, `75 cL`, `0.75 L` and `25.4 fl oz` are the same value.
        Normalise, then compare numerically.
    *   Compare against the application record *and*, separately, against the
        standards of fill in `context.beverage.net_contents`. A volume that
        matches the application but is not a standard of fill is a finding.
    *   `tolerance_ml` covers unit-conversion rounding, not discrepancy.
    """
    raise StrategyNotImplemented(
        "net_contents strategy is not implemented. See docs/field-assignment.md, 'Net contents'."
    )


def class_type_strategy(context: StrategyContext) -> FieldResult:
    """Anchored search, then a spatial prior, then a controlled vocabulary.

    The weakest field, and the one whose expectations need setting. Contract:

    1.  Anchored search against the expected value, as `anchored_strategy`.
    2.  If that fails, restrict candidates to spans within roughly 2x the brand
        name's glyph height of `context.anchor_bbox` and re-score. Class/type
        sits adjacent to the brand, usually directly below and smaller.
    3.  If that fails, match against `context.beverage.class_type_vocabulary`
        so the tool can report what it *did* find: "found Kentucky Straight
        Bourbon Whiskey, application states Straight Bourbon Whiskey" is
        actionable where a bare not-found is not.

    Step 3 is what makes this field worth including. Without it a genuine
    class/type mismatch is indistinguishable from an OCR failure.
    Apply `context.beverage.abbreviations` before comparison.
    """
    raise StrategyNotImplemented(
        "class_type strategy is not implemented. See docs/field-assignment.md, 'Class/type'."
    )


#: Strategy name -> implementation. The warning strategy lives in
#: rules/warning.py because its checks are substantial enough to warrant their
#: own module; it is registered by rules/engine.py at import time.
STRATEGIES = {
    "anchored": anchored_strategy,
    "alcohol_content": alcohol_content_strategy,
    "net_contents": net_contents_strategy,
    "class_type": class_type_strategy,
}
