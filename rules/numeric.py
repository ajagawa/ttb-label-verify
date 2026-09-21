"""Numeric field strategies: alcohol content and net contents.

These are the two *pattern* fields from docs/field-assignment.md. Unlike the
brand name, their expected value is not searched for by similarity — it is
found by shape ("a number followed by % ALC/VOL", "a number followed by a
volume unit"), read as a number, and compared as a number.

That changes the failure mode. Anchored search fails by finding nothing.
Pattern search fails by finding the *wrong instance*: an unrelated percentage
("51% corn"), a numeral inside the warning statement, or a bare "750" that is
a batch number rather than a volume. Most of the machinery below exists to
stop the wrong instance from winning, and to make sure that when two
instances disagree the agent is told rather than one being quietly chosen.

Three properties are non-negotiable, and each is a decision recorded in
docs/field-assignment.md:

1.  **Patterns read ``Span.numeric_text``, never ``canonical_text``.**
    The canonical form folds digits onto letters (5->S, 0->O) for fuzzy name
    matching. Reading a number out of it would at best fail and at worst
    compare two folded strings and call them equal. This is trap 2 in
    extraction/normalize.py.

2.  **A numeric difference is never fuzzy-forgiven.**
    A label reading 46% against an application stating 45% is exactly the
    one-character discrepancy an OCR engine can produce *and* exactly the
    violation this tool exists to catch. It is never resolved to MATCH. The
    result states both readings plainly and says that a single-digit OCR
    confusion is possible, so the agent verifies visually. The tool reports;
    it does not decide which reading is true.

3.  **Not-found is never a mismatch.**
    Nothing located means `not_found_result`, which is REVIEW.
    `FieldResult` enforces this structurally as well.

Verdicts here are *not* derived from the score bands in the rule set. Those
bands (0.92 / 0.60) describe fuzzy similarity, and a number either equals
another number or it does not. The verdict comes from the explicit decision
table in each strategy's docstring; the score is still computed, in [0, 1],
because the batch worklist sorts on it (`VerificationResult.lowest_score`)
and a clear numeric mismatch must sort to the top.

Everything here is deterministic: no I/O, no randomness, and ties between
candidates are broken by position on the label, never by iteration order.
"""

from __future__ import annotations

import enum
import functools
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from extraction.normalize import canonicalize_numeric, collapse_for_anchor
from extraction.spans import Span
from rules.comparators import StrategyContext, not_found_result
from rules.engine import register_strategy
from rules.results import BoxModel, CheckOutcome, CheckResult, FieldResult, Strategy, Verdict

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Two thresholds decide whether this module ASSERTS a violation rather than
# merely reporting one, and both live in the versioned rule set as
# `ruleset.tuning.numeric`: `mismatch_min_ocr_confidence` (a second, higher
# confidence bar for MISMATCH, because mid-range confidence on a digit is
# exactly what a 6-read-as-5 looks like) and `wording_min_similarity`.
#
# They are read at the point of use and have no module-level default on
# purpose. A constant shadowing the rule set lets a verdict change without the
# recorded rule-set version changing, and a fallback default is
# indistinguishable, in the result, from the rule set having been ignored. A
# rule set missing them fails to load, which is the intended failure.

#: Two decimal places is finer than anything printed on a label (ABV to 0.1,
#: volumes to the millilitre) and coarse enough to absorb float noise from
#: unit conversion (0.75 L * 1000 = 750.0000000000001).
_EXACT_PLACES = 2

#: 1 US fluid ounce in millilitres (21 CFR 101.105 / NIST Handbook 130).
#: Distinct from the imperial fluid ounce (28.4131 mL); US labels mean the US one.
ML_PER_US_FL_OZ = 29.5735

# Score components. See `_score`. Kept as named constants so the diagnostic
# record can be read against them rather than against magic numbers.
_AGREE_EXACT = 1.0
_AGREE_WITHIN_TOLERANCE = 0.75
_AGREE_UNKNOWN = 0.5  # application value unreadable, or label value low-confidence
_AGREE_DIFFERS = 0.0
_PENALTY_WEAK_FORM = 0.85  # ABV only as proof — correct value, deficient statement
_PENALTY_BARE = 0.6  # a percentage not labelled as alcohol content
_PENALTY_INCONSISTENT = 0.6  # two statements on the label disagree
_PENALTY_NONSTANDARD_FILL = 0.8
_PENALTY_APPROX_WORDING = 0.95  # unit wording recognised fuzzily
_CORROBORATION_BONUS = 0.05  # proof agrees with ABV


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


class Tier(enum.IntEnum):
    """Pattern precedence. Lower wins.

    The reported hit is chosen by tier first, because tier encodes how sure we
    are that the text is *the field* at all — "45% Alc./Vol." is certainly an
    alcohol statement; a bare "45%" might be "45% rye in the mash bill".
    """

    LABELLED = 0  # ABV with an ALC/VOL qualifier; a metric volume
    SECONDARY = 1  # proof; a US fluid-ounce volume
    BARE = 2  # a percentage with no qualifier (ABV only)


@dataclass(frozen=True, slots=True)
class Hit:
    """One pattern match on the label, tied to the span it was read from.

    ``value`` is already in the comparison unit (percent ABV, or millilitres),
    so every hit on a field can be compared against every other regardless of
    how it was printed. ``printed`` keeps the number as written for the
    explanation the agent reads.
    """

    tier: Tier
    value: float
    printed: str
    unit: str
    span: Span

    #: False when the *unit wording* beside the number ("Alc./Vol.", "Proof")
    #: had to be recognised fuzzily because OCR garbled a character in it. The
    #: number itself is never fuzzy — see `_fuzzy_wording`.
    wording_exact: bool = True

    @property
    def position(self) -> tuple[float, float]:
        """Reading-order sort key: top, then left."""
        return (self.span.bbox.top, self.span.bbox.left)


# Wraps a field pattern so it must account for the *whole* span, allowing only
# surrounding brackets and trailing punctuation: "(90 Proof)" and "45% Alc./Vol.,"
# are accepted; "Batch 750 mL-2" is not.
#
# Full-matching is how one physical statement maps to one span. The span index
# contains every contiguous run of 1-6 words on a line, so the run that is
# *exactly* the statement is always present; a search()-based approach would
# instead hit the same statement in dozens of overlapping spans and need
# de-duplication by geometry, with a bounding box that includes neighbouring
# text. Full-matching gives the tightest box for free — which is the box the
# overlay should highlight.
_LEAD = r"[\s(\[{]*"
_TRAIL = r"[\s)\]}.,;:*]*"


@dataclass(frozen=True, slots=True)
class _FieldPattern:
    """One field pattern, compiled for both of the places it is used.

    ``whole`` is full-matched against a label span (see `_LEAD` / `_TRAIL`);
    ``search`` finds the same shape inside the application's free-text value.
    Sharing one source string is what guarantees the label side and the
    application side agree on what counts as, say, an ABV statement.
    """

    tier: Tier
    source: str

    @property
    def whole(self) -> re.Pattern[str]:
        return _compile(rf"{_LEAD}(?:{self.source}){_TRAIL}")

    @property
    def search(self) -> re.Pattern[str]:
        # The lookarounds stop "1750 mL" yielding "750 mL" and "12 LBS" yielding "12 L".
        return _compile(rf"(?<![\w.,])(?:{self.source})(?![A-Z])")


@functools.cache
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _to_float(number: str) -> float:
    """Parse a printed number, accepting a comma as thousands or decimal mark.

    "1,000" is one thousand (comma followed by exactly three digits); "1,75"
    is one and three-quarters, as printed on European-origin labels. Anything
    else is a plain decimal.
    """
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", number):
        return float(number.replace(",", ""))
    return float(number.replace(",", "."))


def _fmt(value: float) -> str:
    """45.0 -> '45', 40.5 -> '40.5', 751.1735 -> '751.2'. For explanations only."""
    return f"{round(value, 1):g}"


def _same(a: float, b: float) -> bool:
    return round(a, _EXACT_PLACES) == round(b, _EXACT_PLACES)


def _warning_line_indices(context: StrategyContext) -> frozenset[int]:
    """Lines that are recognisably part of the government warning.

    The engine resolves the warning first and masks its region, so normally
    nothing from it reaches this module. This is the second line of defence,
    for when the warning was not resolved (its strategy unimplemented, or its
    header too mangled to anchor): the warning's numerals — "(1)", "(2)" —
    are exactly what a loose numeric pattern latches onto.

    The patterns below already require a unit or an ALC/PROOF qualifier, so
    the warning's numerals cannot match them as written. This guard exists
    because "already cannot match" is a property of today's patterns, and a
    future pattern change should not silently reopen the hole.

    Detection is by the warning's own anchor phrases from the rule set, over
    whole-line spans only.
    """
    rule = context.ruleset.government_warning
    phrases = [collapse_for_anchor(p) for p in (rule.header_text, *rule.anchor.tail_phrases)]
    flagged: set[int] = set()
    for span in context.spans:
        if span.word_range is not None or len(span.line_indices) != 1:
            continue
        collapsed = collapse_for_anchor(span.raw_text)
        if any(phrase and phrase in collapsed for phrase in phrases):
            flagged.add(span.line_indices[0])
    return frozenset(flagged)


#: Bracketed text is a statement boundary: labels print "45% Alc./Vol.
#: (90 Proof)", and the bracket is what separates the two statements.
_BRACKET_SPLIT = re.compile(r"[()\[\]{}]")


def _span_fragments(span: Span) -> list[str]:
    """Statements inside a span the OCR engine failed to separate with spaces.

    Normally one physical statement is one span, because the span index
    enumerates every contiguous run of words and a strategy full-matches
    against it. That breaks when the engine returns no spaces at all: on the
    degraded fixtures PP-OCR reads a compliant

        45% Alc./Vol. (90 Proof)

    as the single token ``45%Alc.Vol.(90Proof)``. There is no word boundary
    left for the index to split on, so full-matching finds nothing and the
    strategy reports "could not locate the alcohol content" about a label
    whose alcohol content is right there and correct.

    A single OCR token is atomic — nothing downstream can subdivide it — so
    this is the one place where looking *inside* a candidate is the only way
    to read it. It is deliberately narrow:

    *   Only for spans with no whitespace, and only when nothing full-matched
        the span as a whole. A multi-word span is still full-matched, so the
        "wrong instance" protection is untouched for every ordinary label.
    *   Fragments are split on brackets only, which is how labels delimit a
        secondary statement. Each fragment then goes through exactly the same
        full-match and wording checks as an ordinary span, with the same
        strict reading of digits.

    The resulting hits carry the whole token's bounding box, since that is
    the smallest region the engine actually located.
    """
    if " " in span.numeric_text:
        return []
    return [fragment for fragment in _BRACKET_SPLIT.split(span.numeric_text) if fragment.strip()]


def _eligible_spans(context: StrategyContext) -> list[Span]:
    excluded = _warning_line_indices(context)
    return [s for s in context.spans if not excluded.intersection(s.line_indices)]


def _score(agreement: float, confidence: float, *penalties: float, bonus: float = 0.0) -> float:
    """Composite score in [0, 1], for triage ordering only — not for the verdict.

        score = agreement x (0.7 + 0.3 x ocr_confidence) x penalties (+ bonus)

    The confidence blend mirrors `rules.comparators.confidence_factor` at its
    default weight, so a numeric score and a name score sit on comparable
    scales in the batch worklist. A clear numeric difference has agreement 0
    and therefore sorts first, which is the intent: it is the finding most
    worth an agent's attention.
    """
    value = agreement * (0.7 + 0.3 * max(0.0, min(1.0, confidence)))
    for penalty in penalties:
        value *= penalty
    return round(max(0.0, min(1.0, value + bonus)), 4)


def _result(
    context: StrategyContext,
    hit: Hit,
    verdict: Verdict,
    score: float,
    reason: str,
    checks: list[CheckResult],
) -> FieldResult:
    """Assemble a located result. The evidence is always the reported hit's span."""
    rule = context.field_rule
    return FieldResult(
        field_id=rule.id,
        label=rule.label,
        verdict=verdict,
        strategy=Strategy.PATTERN,
        expected=context.expected or None,
        found=hit.span.raw_text,
        bbox=BoxModel(**hit.span.bbox.to_dict()),
        score=score,
        ocr_confidence=round(max(0.0, min(1.0, hit.span.mean_confidence)), 4),
        reason=reason,
        citation=rule.citation,
        citation_url=rule.citation_url,
        checks=tuple(checks),
    )


def _difference_verdict(
    context: StrategyContext, confidence: float, *, wording_exact: bool = True
) -> Verdict:
    """Verdict for a located value that differs beyond tolerance.

    Three conditions, each of which can only *withdraw* the assertion and
    never create one:

    *   **OCR confidence** below `tuning.numeric.mismatch_min_ocr_confidence`
        caps it at REVIEW. Mid-range confidence on a digit is what a
        6-read-as-5 looks like.
    *   **Fuzzily recognised unit wording** caps it at REVIEW. The digits were
        still read strictly, so the difference is real as read — but a garbled
        character next to them is direct evidence that this region did not
        come through cleanly.
    *   **Poor image quality** caps it at REVIEW. The quality gate is the
        engine's own summary of whether the photograph was worth trusting;
        on an image it judged unusable or below `defaults.good_image_quality`,
        the more likely explanation for a differing number is the photograph.

    The quality test is used in one direction only. Good quality never turns a
    REVIEW into a MISMATCH — a clean photograph of an unreadable digit is
    still an unreadable digit — it only declines to assert on a bad one.
    """
    if not wording_exact:
        return Verdict.REVIEW
    if confidence < context.ruleset.tuning.numeric.mismatch_min_ocr_confidence:
        return Verdict.REVIEW
    quality = context.quality
    if quality is not None and (
        not quality.usable or quality.score < context.ruleset.defaults.good_image_quality
    ):
        return Verdict.REVIEW
    return Verdict.MISMATCH


_OCR_CAVEAT = (
    "A single-digit OCR misread could produce this difference, so verify the "
    "highlighted region visually before acting on it."
)


# ---------------------------------------------------------------------------
# Alcohol content
# ---------------------------------------------------------------------------

# Numbers: ABV is at most two integer digits (a 100% statement is not a
# distilled spirit label); a comma decimal is accepted.
_ABV_NUM = r"(?P<n>\d{1,2}(?:[.,]\d{1,2})?)"
_ALC = r"ALC(?:OHOL|OH)?\.?"
_VOL = r"VOL(?:UME)?\.?"
# "ALC./VOL.", "ALC/VOL", "ALC. / VOL.", "ALC BY VOL", "ALCOHOL BY VOLUME", "ALC. VOL."
_ALC_VOL = rf"{_ALC}\s*(?:/|BY|-)?\s*{_VOL}"

#: (tier, pattern) in precedence order. Each is compiled twice by
#: `_FieldPattern`: full-matched against a label span, searched in the
#: application's value.
_ABV_PATTERNS: tuple[_FieldPattern, ...] = (
    # 45% ALC./VOL.   45 % ALC BY VOL   45%ABV   45% ALCOHOL BY VOLUME
    _FieldPattern(Tier.LABELLED, rf"{_ABV_NUM}\s*%\s*(?:{_ALC_VOL}|{_ALC}|ABV)"),
    # ALC./VOL. 45%   ALCOHOL BY VOLUME: 45%   ABV 45%
    _FieldPattern(Tier.LABELLED, rf"(?:{_ALC_VOL}|ABV)\s*:?\s*{_ABV_NUM}\s*%"),
    # ALC. 45% BY VOL.   ALCOHOL 45% BY VOLUME
    _FieldPattern(Tier.LABELLED, rf"{_ALC}\s*{_ABV_NUM}\s*%\s*(?:BY\s*)?{_VOL}"),
    # 90 PROOF   90° PROOF   (90 PROOF)
    _FieldPattern(Tier.SECONDARY, r"(?P<n>\d{1,3}(?:[.,]\d)?)\s*°?\s*PROOF"),
    # 45%  — last resort. Guarded in `alcohol_content_strategy`: discarded
    # whenever any better hit exists, and never allowed to produce MATCH or
    # MISMATCH on its own.
    _FieldPattern(Tier.BARE, rf"{_ABV_NUM}\s*%"),
)


# ---------------------------------------------------------------------------
# Fuzzy unit wording
# ---------------------------------------------------------------------------
#
# Real OCR (PP-OCR on fixtures/labels/brand_twice_bottler.png) reads a
# compliant "45% Alc./Vol. (90 Proof)" as "45% Alc./Nol. (90 Proof)" — one
# character wrong, in the *wording*, not in the number. Against the strict
# patterns above that statement produces no labelled hit at all, so the only
# surviving hit is the proof, and the strategy told an agent there was no
# alcohol-by-volume statement on a label that plainly carries a correct one.
# A false flag on a compliant label is how a tool loses its reader.
#
# So the unit wording — and only the unit wording — tolerates noise. This is
# not a softening of "numeric fields are never fuzzy-forgiven": that rule is
# about the *value*, and the value is still read by the strict patterns, digit
# for digit, with no folding. The wording carries no quantity. Misreading
# "Vol." as "Nol." changes nothing about what the label claims; misreading
# "45" as "46" changes everything.
#
# The safeguards that keep this from becoming a loose match:
#
# *   The wording is compared **as a whole phrase**, never token by token. A
#     single substitution in a three-letter token scores 0.67 against its
#     neighbours ("ALL" against "ALC", "OIL" against "VOL"), which is
#     indistinguishable from coincidence; the same substitution inside
#     "ALC VOL" scores 0.86, because the rest of the phrase corroborates it.
# *   The captured wording may contain no digits, so a number can never be
#     absorbed into it.
# *   A minimum wording length, so short accidental tokens cannot reach the
#     threshold at all.
# *   The resulting hit is marked `wording_exact=False`, which makes it lose
#     to a cleanly read statement during selection and caps a value difference
#     at REVIEW rather than MISMATCH.

#: The similarity at or above which a garbled wording is accepted as the real
#: thing is `ruleset.tuning.numeric.wording_min_similarity`. At its current
#: 0.80 it accepts one substitution in "ALC VOL" (0.857) and in "PROOF"
#: (0.80), and rejects two.

#: Shortest normalised wording considered at all. "ABV" and "ALC" fall below
#: it: they are matched exactly by the strict patterns, and three letters are
#: too little context to correct safely.
_WORDING_MIN_LENGTH = 5

_ABV_WORDINGS = ("ALC VOL", "ALC BY VOL", "ALCOHOL BY VOLUME", "ALCOHOL VOLUME", "ALC VOLUME")
_PROOF_WORDINGS = ("PROOF",)

#: Letters, spaces and the punctuation that separates them — no digits.
_WORDING_CHARS = r"[A-Z][A-Z./\s-]{2,24}"

#: (tier, acceptable wordings, pattern). Tried only when no strict pattern
#: matched the span, so a cleanly printed label never reaches them.
_FUZZY_ABV_PATTERNS: tuple[tuple[Tier, tuple[str, ...], re.Pattern[str]], ...] = (
    # 45% Alc./Nol.
    (
        Tier.LABELLED,
        _ABV_WORDINGS,
        _compile(rf"{_LEAD}{_ABV_NUM}\s*%\s*(?P<w>{_WORDING_CHARS}){_TRAIL}"),
    ),
    # Alc./Nol. 45%
    (
        Tier.LABELLED,
        _ABV_WORDINGS,
        _compile(rf"{_LEAD}(?P<w>{_WORDING_CHARS})\s*:?\s*{_ABV_NUM}\s*%{_TRAIL}"),
    ),
    # 90 Prool
    (
        Tier.SECONDARY,
        _PROOF_WORDINGS,
        _compile(rf"{_LEAD}(?P<n>\d{{1,3}}(?:[.,]\d)?)\s*°?\s*(?P<w>{_WORDING_CHARS}){_TRAIL}"),
    ),
)


def _normalise_wording(wording: str) -> str:
    """Letters only, single-spaced: "ALC./NOL." -> "ALC NOL"."""
    return " ".join(re.sub(r"[^A-Z]+", " ", wording.upper()).split())


def _fuzzy_wording(wording: str, candidates: tuple[str, ...], min_similarity: float) -> bool:
    """Whether a garbled unit wording is close enough to a real one.

    Compared as a whole phrase with `rapidfuzz.fuzz.ratio` — plain edit
    similarity rather than `token_sort_ratio`, because the word order in
    "ALC BY VOL" is part of what makes it that phrase rather than a
    coincidence of letters.
    """
    normalised = _normalise_wording(wording)
    if len(normalised) < _WORDING_MIN_LENGTH:
        return False
    return any(
        fuzz.ratio(normalised, candidate) / 100.0 >= min_similarity for candidate in candidates
    )


def _abv_from(tier: Tier, number: str) -> float | None:
    """Percent ABV from a matched number, or None if it is not a plausible ABV."""
    value = _to_float(number)
    if tier is Tier.SECONDARY:
        # Proof is defined as twice ABV in US usage (27 CFR 5.65). A proof
        # above 200 is not a proof.
        return value / 2.0 if 0 < value <= 200 else None
    return value if 0 < value <= 100 else None


def parse_abv(text: str) -> tuple[float, Tier] | None:
    """Read an ABV from free text such as the application record's value.

    Accepts "45% Alc./Vol.", "45%", "Alc/Vol 45%", "90 Proof" and a bare "45".
    Searches (rather than full-matches) because the application field is the
    whole of its own text. Runs the same patterns as the label side, in the
    same precedence, over the same numeric normalisation — so the two sides
    cannot drift apart in what they consider a number.

    Returns:
        (percent ABV, tier of the pattern that matched), or None when no
        number can be read. A bare "45" is taken as percent: on the
        application's alcohol-content field that is the only reading.
    """
    numeric = canonicalize_numeric(text)
    for pattern in _ABV_PATTERNS:
        match = pattern.search.search(numeric)
        if match:
            value = _abv_from(pattern.tier, match.group("n"))
            if value is not None:
                return value, pattern.tier
    bare = re.fullmatch(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*", numeric)
    if bare:
        value = _abv_from(Tier.LABELLED, bare.group(1))
        if value is not None:
            return value, Tier.BARE
    return None


def find_abv_hits(context: StrategyContext) -> list[Hit]:
    """Every ABV statement on the label, one hit per physical statement.

    A span is tested against the patterns in precedence order and yields at
    most one hit (its best-tier interpretation), so "45%" inside
    "45% Alc./Vol." is not double-counted from the same span. It does still
    appear from the shorter span "45%" as a BARE hit — which `alcohol_content_strategy`
    discards whenever a better hit exists.
    """
    similarity = context.ruleset.tuning.numeric.wording_min_similarity
    hits: list[Hit] = []
    for span in _eligible_spans(context):
        hit = _strict_abv_hit(span, span.numeric_text) or _fuzzy_abv_hit(
            span, span.numeric_text, similarity
        )
        if hit is not None:
            hits.append(hit)
            continue
        # Nothing matched the span whole. If it is one unsegmented token, read
        # the statements inside it — see `_span_fragments`.
        for fragment in _span_fragments(span):
            inner = _strict_abv_hit(span, fragment) or _fuzzy_abv_hit(span, fragment, similarity)
            if inner is not None:
                hits.append(inner)
    return hits


def _strict_abv_hit(span: Span, text: str) -> Hit | None:
    """Best strict-pattern reading of ``text``, attributed to ``span``."""
    for pattern in _ABV_PATTERNS:
        match = pattern.whole.fullmatch(text)
        if not match:
            continue
        value = _abv_from(pattern.tier, match.group("n"))
        if value is None:
            continue
        unit = "proof" if pattern.tier is Tier.SECONDARY else "%"
        return Hit(pattern.tier, value, match.group("n"), unit, span)
    return None


def _fuzzy_abv_hit(span: Span, text: str, min_similarity: float) -> Hit | None:
    """Reading of one span whose unit wording had to be recognised fuzzily.

    Only reached when no strict pattern matched the span, so this can add a
    statement the strict pass missed but can never override one it found. The
    number is still parsed by the same strict expression; only the wording
    beside it is allowed to be noisy. See the commentary above
    `_fuzzy_wording`, and the threshold in the rule set.

    Note the BARE tier has no fuzzy equivalent: a bare percentage has no
    wording to be noisy about.
    """
    for tier, wordings, pattern in _FUZZY_ABV_PATTERNS:
        match = pattern.fullmatch(text)
        if not match or not _fuzzy_wording(match.group("w"), wordings, min_similarity):
            continue
        value = _abv_from(tier, match.group("n"))
        if value is None:
            continue
        unit = "proof" if tier is Tier.SECONDARY else "%"
        return Hit(tier, value, match.group("n"), unit, span, wording_exact=False)
    return None


def _select(hits: list[Hit], expected: float | None) -> Hit:
    """Choose the hit to report: best tier, then closest to expectation, then position.

    Closeness to the expected value is used here, as docs/field-assignment.md
    permits, *only to pick which hit to show*. It cannot decide the verdict,
    because whenever two same-field hits on the label disagree the strategy
    reports an internal inconsistency and the verdict is capped at REVIEW
    regardless of which one was shown. Picking the closer one simply means the
    agent is shown the statement most likely to be the one the applicant
    meant, next to the one that contradicts it.
    """

    def key(hit: Hit) -> tuple:
        distance = abs(hit.value - expected) if expected is not None else 0.0
        return (
            hit.tier,
            not hit.wording_exact,  # a cleanly read statement outranks a garbled one
            round(distance, _EXACT_PLACES),
            *hit.position,
            hit.span.raw_text,
        )

    return min(hits, key=key)


def _abv_checks(
    context: StrategyContext, reported: Hit, hits: list[Hit]
) -> tuple[list[CheckResult], bool, bool]:
    """Internal-consistency checks among the label's own ABV statements.

    Returns (checks, inconsistent, corroborated).

    Two checks, both reported whether they pass or fail so the agent can see
    what was verified:

    *   ``proof_consistency`` — proof must be twice ABV, within the rule set's
        `proof_tolerance`. Emitted only when both appear.
    *   ``abv_statement_consistency`` — two ABV statements (front and back
        label, say) must state the same value. Emitted only when there are two.

    A failure in either is a finding about the *label*, independent of the
    application: a label reading "45% Alc./Vol. (80 Proof)" is wrong even when
    the application says 45%. So it prevents a clean MATCH.
    """
    rule = context.beverage.alcohol_content
    citation = context.field_rule.citation
    checks: list[CheckResult] = []
    inconsistent = corroborated = False

    labelled = [h for h in hits if h.tier is Tier.LABELLED]
    proofs = [h for h in hits if h.tier is Tier.SECONDARY]

    if len({round(h.value, _EXACT_PLACES) for h in labelled}) > 1:
        inconsistent = True
        readings = ", ".join(f"“{h.span.raw_text}”" for h in sorted(labelled, key=_pos))
        checks.append(
            CheckResult(
                id="abv_statement_consistency",
                outcome=CheckOutcome.FAIL,
                citation=citation,
                summary="The label carries more than one alcohol content statement, and they disagree.",
                detail=f"Statements read: {readings}.",
            )
        )

    if rule.proof_is_double_abv and proofs and labelled:
        abv = reported.value if reported.tier is Tier.LABELLED else labelled[0].value
        bad = [p for p in proofs if abs(p.value * 2 - abv * 2) > rule.proof_tolerance]
        if bad:
            inconsistent = True
            proof = bad[0]
            checks.append(
                CheckResult(
                    id="proof_consistency",
                    outcome=CheckOutcome.FAIL,
                    citation=citation,
                    summary=(
                        f"Proof does not agree with alcohol by volume: the label states "
                        f"{_fmt(abv)}% and {proof.printed} proof, but proof must be twice "
                        f"ABV ({_fmt(abv * 2)} proof)."
                    ),
                    detail=f"Proof statement read as “{proof.span.raw_text}”.",
                    measurement={"abv": abv, "proof": proof.value * 2},
                    confidence=round(min(1.0, proof.span.mean_confidence), 4),
                )
            )
        else:
            corroborated = True
            checks.append(
                CheckResult(
                    id="proof_consistency",
                    outcome=CheckOutcome.PASS,
                    citation=citation,
                    summary=f"Proof agrees with alcohol by volume ({_fmt(abv * 2)} proof).",
                    measurement={"abv": abv, "proof": proofs[0].value * 2},
                )
            )

    return checks, inconsistent, corroborated


def _pos(hit: Hit) -> tuple[float, float]:
    return hit.position


def alcohol_content_strategy(context: StrategyContext) -> FieldResult:
    """Locate the alcohol content statement and compare it numerically.

    Procedure:

    1.  Collect every hit of every pattern over ``numeric_text``
        (`find_abv_hits`). "45% Alc./Vol. (90 Proof)" yields a labelled hit,
        a proof hit, and a bare "45%" hit from its sub-span.
    2.  Discard BARE hits if any better hit exists. A bare percentage is the
        weakest evidence there is — "51% corn", "100% agave" — and must never
        outrank or contradict a statement that says what it is.
    3.  Select the hit to report (`_select`).
    4.  Run the internal-consistency checks (`_abv_checks`).
    5.  Compare against the application's value, parsed with the same
        patterns (`parse_abv`).

    Decision table, first matching row wins:

    ============================================  ==========  ================
    Condition                                     Verdict     Why
    ============================================  ==========  ================
    nothing located                               REVIEW      not-found is
                                                  (not found) never a mismatch
    OCR confidence < rule set min                 REVIEW      the read itself
                                                              is untrustworthy
    application value unreadable                  REVIEW      nothing to
                                                              compare against
    reported hit is a bare percentage             REVIEW      may not be the
                                                              alcohol statement
                                                              at all
    label's own statements disagree               REVIEW      see below
    value differs beyond `tolerance_abv`          MISMATCH if confidence
                                                  >= the rule set's
                                                  mismatch_min_ocr_confidence
                                                  and the image quality is
                                                  good,
                                                  else REVIEW
    value differs within `tolerance_abv`          REVIEW      any printed
                                                              difference is
                                                              reported
    stated only as proof, value agrees            REVIEW      27 CFR 5.65 wants
                                                              % by volume;
                                                              proof is optional
                                                              extra
    otherwise (exact agreement)                   MATCH
    ============================================  ==========  ================

    **Why an internal inconsistency is REVIEW, not MISMATCH.** When the label
    reads "46% Alc./Vol. (90 Proof)" against an application of 45%, one of the
    two printed numbers is wrong, and a misread digit in either would produce
    exactly this picture. The tool cannot tell which, so it asserts neither;
    the failed check is attached and the verdict is a flag the agent resolves.
    It is never MATCH, even when the ABV agrees with the application, because
    the label is wrong on its face.

    **Why within-tolerance is REVIEW, not MATCH.** `tolerance_abv` (0.3 points,
    27 CFR 5.65) is a tolerance on the *product* against its label. Applied to
    label-against-application, it marks the line between "the numbers differ
    slightly" and "the numbers clearly differ" — it decides REVIEW versus
    MISMATCH, not whether a difference is reported. "Never fuzzy-forgive a
    numeric difference" is taken literally: 45.2% against 45% is shown to the
    agent.

    **Why a clear difference can be MISMATCH at all.** "Not found is never a
    mismatch" is about the absence of evidence. A clearly read, well-located,
    high-confidence "46% Alc./Vol." is evidence, with a bounding box the agent
    can check in one glance, and the reason text still names both readings
    and the OCR caveat.
    """
    hits = find_abv_hits(context)
    if not hits:
        return not_found_result(context)

    if any(h.tier is not Tier.BARE for h in hits):
        hits = [h for h in hits if h.tier is not Tier.BARE]

    parsed = parse_abv(context.expected) if context.expected else None
    expected = parsed[0] if parsed else None

    reported = _select(hits, expected)
    checks, inconsistent, corroborated = _abv_checks(context, reported, hits)
    confidence = reported.span.mean_confidence
    min_conf = context.ruleset.defaults.min_ocr_confidence
    tolerance = context.beverage.alcohol_content.tolerance_abv

    found_desc = _describe_abv(reported)
    penalties: list[float] = []
    if not reported.wording_exact:
        penalties.append(_PENALTY_APPROX_WORDING)
    if inconsistent:
        penalties.append(_PENALTY_INCONSISTENT)
    if reported.tier is Tier.SECONDARY:
        penalties.append(_PENALTY_WEAK_FORM)
    if reported.tier is Tier.BARE:
        penalties.append(_PENALTY_BARE)
        checks.append(
            CheckResult(
                id="abv_statement_form",
                outcome=CheckOutcome.ADVISORY,
                citation=context.field_rule.citation,
                summary="Found a percentage with no “alcohol by volume” wording beside it.",
            )
        )
    if reported.tier is Tier.SECONDARY:
        checks.append(
            CheckResult(
                id="abv_statement_form",
                outcome=CheckOutcome.ADVISORY,
                citation=context.field_rule.citation,
                summary=(
                    "Alcohol content appears only as proof. A statement of alcohol by "
                    "volume is required; proof may appear only in addition to it."
                ),
            )
        )
    bonus = _CORROBORATION_BONUS if corroborated else 0.0

    def done(verdict: Verdict, agreement: float, reason: str) -> FieldResult:
        score = _score(agreement, confidence, *penalties, bonus=bonus if agreement else 0.0)
        return _result(context, reported, verdict, score, reason, checks)

    consistency_note = (
        " The label's own alcohol statements also disagree with each other — see the checks below."
        if inconsistent
        else ""
    )

    # --- low confidence: the read itself cannot be trusted ----------------
    if confidence < min_conf:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN,
            f"Read the alcohol content as {found_desc}, but with low OCR confidence "
            f"({confidence:.0%}); the application states “{context.expected}”. "
            "The digits may not be what is printed — confirm visually." + consistency_note,
        )

    # --- nothing to compare against ----------------------------------------
    if expected is None:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN,
            f"Label reads {found_desc}, but the application's alcohol content "
            f"(“{context.expected}”) could not be read as a number. "
            "Compare the two manually." + consistency_note,
        )

    difference = abs(reported.value - expected)
    exact = _same(reported.value, expected)
    beyond = not exact and difference > tolerance + 1e-9
    expected_desc = f"{_fmt(expected)}%"

    # --- a bare percentage: may not be the alcohol statement at all -------
    if reported.tier is Tier.BARE:
        agreement = (
            "matches"
            if exact
            else f"does not match the application's {expected_desc}. {_OCR_CAVEAT}"
        )
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN,
            f"No alcohol content statement with “Alc./Vol.” or “proof” was found. "
            f"The only candidate is a bare percentage, “{reported.span.raw_text}”, which "
            f"{agreement} Confirm it is the alcohol content and that the statement "
            "is worded as required.",
        )

    # --- the label contradicts itself --------------------------------------
    if inconsistent:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN if not beyond else _AGREE_DIFFERS,
            f"Label reads {found_desc}; the application states {expected_desc}. "
            "The label's alcohol statements disagree with each other, so at least one "
            "of them is wrong (or was misread). Check each highlighted statement "
            "against the artwork.",
        )

    # --- clear difference ---------------------------------------------------
    if beyond:
        return done(
            _difference_verdict(context, confidence, wording_exact=reported.wording_exact),
            _AGREE_DIFFERS,
            f"Label reads {found_desc}; the application states {expected_desc}. "
            f"These differ by {_fmt(difference)} percentage "
            f"{'point' if _same(difference, 1) else 'points'}, more than the "
            f"{_fmt(tolerance)}-point tolerance. {_OCR_CAVEAT}",
        )

    # --- small difference ---------------------------------------------------
    if not exact:
        return done(
            Verdict.REVIEW,
            _AGREE_WITHIN_TOLERANCE,
            f"Label reads {found_desc}; the application states {expected_desc}. "
            f"The difference ({_fmt(difference)} points) is within the "
            f"{_fmt(tolerance)}-point tolerance but the numbers are not the same — "
            f"confirm which is intended. {_OCR_CAVEAT}",
        )

    # --- agrees in value, deficient in form -------------------------------
    if reported.tier is Tier.SECONDARY:
        return done(
            Verdict.REVIEW,
            _AGREE_EXACT,
            f"Label states {found_desc}, which agrees with the application's "
            f"{expected_desc}, but no alcohol-by-volume statement was found. Proof may "
            "appear only in addition to a percentage-by-volume statement — check "
            "whether the label carries one.",
        )

    corroboration = " Proof on the label agrees." if corroborated else ""
    return done(
        Verdict.MATCH,
        _AGREE_EXACT,
        f"Label reads {found_desc}, matching the application's {expected_desc}."
        + corroboration
        + _wording_note(reported),
    )


def _wording_note(hit: Hit) -> str:
    """Disclose a fuzzily recognised unit wording in the agent's own reason.

    The number matched exactly, so the verdict is MATCH — but the agent is
    told that the wording beside it came through garbled, because the printed
    wording is itself a labelling requirement and this is the tool's only
    chance to say so.
    """
    if hit.wording_exact:
        return ""
    return (
        f" The wording beside the number was recognised as “{hit.span.raw_text}”, which "
        "appears to be an OCR misreading rather than what is printed; the number itself "
        "was read exactly. Confirm the wording on the artwork."
    )


def _describe_abv(hit: Hit) -> str:
    if hit.tier is Tier.SECONDARY:
        return f"{hit.printed} proof ({_fmt(hit.value)}% alcohol by volume)"
    return f"{_fmt(hit.value)}% (“{hit.span.raw_text}”)"


# ---------------------------------------------------------------------------
# Net contents
# ---------------------------------------------------------------------------

# Integer part up to four digits with optional thousands groups, optional
# decimal part with either mark: 750, 1,000, 1.75, 1,75, 0.75.
_VOL_NUM = r"(?P<n>\d{1,4}(?:,\d{3})*(?:[.,]\d+)?)"

#: Unit spelling -> millilitres per unit. Matched case-insensitively via
#: numeric_text, which is uppercase. The unit is *required*: a bare "750" on a
#: label is as likely a batch number, a bottle number or a year fragment as it
#: is a volume, and reading it as millilitres would be a guess dressed as a
#: finding.
_METRIC_UNITS = {
    "ML": 1.0,
    "MILLILITER": 1.0,
    "MILLILITERS": 1.0,
    "MILLILITRE": 1.0,
    "MILLILITRES": 1.0,
    "CL": 10.0,
    "CENTILITER": 10.0,
    "CENTILITERS": 10.0,
    "CENTILITRE": 10.0,
    "CENTILITRES": 10.0,
    "L": 1000.0,
    "LITER": 1000.0,
    "LITERS": 1000.0,
    "LITRE": 1000.0,
    "LITRES": 1000.0,
}

_METRIC_ALT = "|".join(sorted(_METRIC_UNITS, key=len, reverse=True))
# "FL OZ", "FL. OZ.", "FL.OZ", "FLOZ", "FLUID OUNCES", "U.S. FL. OZ." — but not
# a bare "OZ", which on a label is at least as likely to be a weight.
_FLOZ = r"(?:U\.?\s*S\.?\s*)?(?:FL(?:UID)?\.?\s*(?:OZ|OUNCES?)\.?)"

_VOLUME_PATTERNS: tuple[_FieldPattern, ...] = (
    _FieldPattern(Tier.LABELLED, rf"{_VOL_NUM}\s*(?P<u>{_METRIC_ALT})\.?"),
    _FieldPattern(Tier.SECONDARY, rf"{_VOL_NUM}\s*(?P<u>{_FLOZ})"),
)


def _to_ml(tier: Tier, number: str, unit: str) -> float | None:
    value = _to_float(number)
    if value <= 0:
        return None
    if tier is Tier.SECONDARY:
        return value * ML_PER_US_FL_OZ
    return value * _METRIC_UNITS[unit]


def _is_floz(hit: Hit) -> bool:
    return hit.tier is Tier.SECONDARY


def _unit_label(tier: Tier, unit: str) -> str:
    return "fl oz" if tier is Tier.SECONDARY else unit


def parse_volume_ml(text: str) -> tuple[float, Tier] | None:
    """Read a volume in millilitres from free text, such as the application value.

    Requires a unit, for the same reason as on the label: "750" with no unit
    is not read as 750 mL. On the application side that yields REVIEW ("could
    not be read") rather than a guess.
    """
    numeric = canonicalize_numeric(text)
    for pattern in _VOLUME_PATTERNS:
        match = pattern.search.search(numeric)
        if match:
            value = _to_ml(pattern.tier, match.group("n"), match.group("u"))
            if value is not None:
                return value, pattern.tier
    return None


def find_volume_hits(context: StrategyContext) -> list[Hit]:
    """Every net contents statement on the label, one hit per physical statement."""
    hits: list[Hit] = []
    for span in _eligible_spans(context):
        hit = _volume_hit(span, span.numeric_text)
        if hit is not None:
            hits.append(hit)
            continue
        for fragment in _span_fragments(span):
            inner = _volume_hit(span, fragment)
            if inner is not None:
                hits.append(inner)
    return hits


def _volume_hit(span: Span, text: str) -> Hit | None:
    """Best volume reading of ``text``, attributed to ``span``.

    No fuzzy-wording equivalent here, deliberately: volume units are one or
    two characters ("mL", "L", "cL"), so there is no surrounding phrase to
    corroborate a garbled one and a fuzzy match would be a coin flip between
    units that differ by a factor of a thousand.
    """
    for pattern in _VOLUME_PATTERNS:
        match = pattern.whole.fullmatch(text)
        if not match:
            continue
        value = _to_ml(pattern.tier, match.group("n"), match.group("u"))
        if value is None:
            continue
        unit = _unit_label(pattern.tier, match.group("u").strip())
        return Hit(pattern.tier, value, match.group("n"), unit, span)
    return None


def _volumes_agree(a_ml: float, b_ml: float, *, converted: bool, tol: float) -> bool:
    """Whether two volumes are the same, applying tolerance only to conversions.

    `tolerance_ml` is documented in the rule set as covering *unit-conversion
    rounding* (25.4 fl oz is 751.2 mL), "not a genuine discrepancy". So it is
    applied only when a US fluid-ounce figure is involved. Two metric figures
    convert exactly (75 cL is 750 mL, no rounding), and between them any
    difference is a printed difference: 749 mL against 750 mL is reported,
    not absorbed.
    """
    if converted:
        return abs(a_ml - b_ml) <= tol + 1e-9
    return _same(a_ml, b_ml)


def net_contents_strategy(context: StrategyContext) -> FieldResult:
    """Locate the net contents statement, normalise to mL, and compare.

    Procedure:

    1.  Collect every volume statement (`find_volume_hits`). A unit is
        required; a bare number is never a volume.
    2.  Select the one to report: metric before US fluid ounces (US
        standards of fill for spirits are metric, so the metric statement is
        the primary one), then closest to the application, then position.
    3.  Check the label's volume statements agree with each other (a label
        carrying "750 mL" and "25.4 fl oz" should — they are the same volume
        within conversion rounding).
    4.  Compare against the application, and separately against the
        standards of fill.

    Decision table, first matching row wins:

    ============================================  ==========================
    Condition                                     Verdict
    ============================================  ==========================
    nothing located                               REVIEW (not found)
    OCR confidence < rule set min                 REVIEW
    application value unreadable / no unit        REVIEW
    label's own volume statements disagree        REVIEW
    value differs from the application            MISMATCH if the read and
                                                  the image are clean (see
                                                  `_difference_verdict`),
                                                  else REVIEW
    agrees, but not a standard of fill            REVIEW
    agrees and is a standard of fill              MATCH
    ============================================  ==========================

    **Why a non-standard fill is REVIEW, not MISMATCH.** MISMATCH in this
    tool means "the label disagrees with the application record". A 740 mL
    label on a 740 mL application agrees with it; what is wrong (if anything)
    is the product, and the check that says so is attached as a FAIL. There
    are also legitimate reasons the tool cannot adjudicate — the standards of
    fill have been amended repeatedly (TTB added sizes in 2020 and again
    since), and the list in the rule set may trail the regulation. It is
    never MATCH, because an agent must see it.

    **Why "differs" has no within-tolerance band here** (unlike ABV):
    `tolerance_ml` exists only to absorb fl-oz conversion rounding, so within
    it the two values *are* the same volume; outside it they are not.
    """
    hits = find_volume_hits(context)
    if not hits:
        return not_found_result(context)

    parsed = parse_volume_ml(context.expected) if context.expected else None
    expected = parsed[0] if parsed else None
    expected_is_floz = bool(parsed and parsed[1] is Tier.SECONDARY)

    reported = _select(hits, expected)
    net_rule = context.beverage.net_contents
    tol = net_rule.tolerance_ml
    confidence = reported.span.mean_confidence
    min_conf = context.ruleset.defaults.min_ocr_confidence
    checks: list[CheckResult] = []

    # --- internal consistency among the label's volume statements --------
    disagreeing = [
        h
        for h in hits
        if not _volumes_agree(
            h.value, reported.value, converted=_is_floz(h) or _is_floz(reported), tol=tol
        )
    ]
    inconsistent = bool(disagreeing)
    if inconsistent:
        readings = ", ".join(f"“{h.span.raw_text}”" for h in sorted(hits, key=_pos))
        checks.append(
            CheckResult(
                id="volume_statement_consistency",
                outcome=CheckOutcome.FAIL,
                citation=context.field_rule.citation,
                summary="The label carries more than one net contents statement, and they disagree.",
                detail=f"Statements read: {readings}.",
            )
        )

    # --- standard of fill, independent of the application ----------------
    standard = any(
        _volumes_agree(reported.value, s, converted=_is_floz(reported), tol=tol)
        for s in net_rule.standards_of_fill_ml
    )
    checks.append(
        CheckResult(
            id="standard_of_fill",
            outcome=CheckOutcome.PASS if standard else CheckOutcome.FAIL,
            citation="27 CFR 5.203",
            summary=(
                f"{_fmt(reported.value)} mL is an authorised standard of fill for distilled spirits."
                if standard
                else f"{_fmt(reported.value)} mL is not on the standards of fill for "
                "distilled spirits in this rule set."
            ),
            measurement={"volume_ml": round(reported.value, 3)},
        )
    )

    found_desc = _describe_volume(reported)
    penalties = [_PENALTY_INCONSISTENT] if inconsistent else []
    if not standard:
        penalties.append(_PENALTY_NONSTANDARD_FILL)

    def done(verdict: Verdict, agreement: float, reason: str) -> FieldResult:
        return _result(
            context, reported, verdict, _score(agreement, confidence, *penalties), reason, checks
        )

    fill_note = (
        ""
        if standard
        else f" Separately, {_fmt(reported.value)} mL is not a standard of fill for "
        "distilled spirits (27 CFR 5.203) — confirm the container size is permitted."
    )

    if confidence < min_conf:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN,
            f"Read the net contents as {found_desc}, but with low OCR confidence "
            f"({confidence:.0%}); the application states “{context.expected}”. "
            "The digits may not be what is printed — confirm visually.",
        )

    if expected is None:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN,
            f"Label reads {found_desc}, but the application's net contents "
            f"(“{context.expected}”) could not be read as a volume with a unit. "
            "Compare the two manually." + fill_note,
        )

    agrees = _volumes_agree(
        reported.value, expected, converted=expected_is_floz or _is_floz(reported), tol=tol
    )
    expected_desc = f"“{context.expected}”" + (
        "" if re.search(r"\d\s*ML\b", context.expected.upper()) else f" ({_fmt(expected)} mL)"
    )

    if inconsistent:
        return done(
            Verdict.REVIEW,
            _AGREE_UNKNOWN if agrees else _AGREE_DIFFERS,
            f"Label reads {found_desc}; the application states {expected_desc}. "
            "The label's own net contents statements disagree with each other, so at "
            "least one is wrong (or was misread). Check each against the artwork." + fill_note,
        )

    if not agrees:
        return done(
            _difference_verdict(context, confidence),
            _AGREE_DIFFERS,
            f"Label reads {found_desc}; the application states {expected_desc}. "
            f"{_OCR_CAVEAT}" + fill_note,
        )

    if not standard:
        return done(
            Verdict.REVIEW,
            _AGREE_EXACT,
            f"Label reads {found_desc}, matching the application, but "
            f"{_fmt(reported.value)} mL is not a standard of fill for distilled spirits "
            "(27 CFR 5.203). Confirm the container size is permitted.",
        )

    return done(
        Verdict.MATCH,
        _AGREE_EXACT,
        f"Label reads {found_desc}, matching the application's {expected_desc}.",
    )


def _describe_volume(hit: Hit) -> str:
    raw = f"“{hit.span.raw_text}”"
    if hit.tier is Tier.LABELLED and hit.unit == "ML":
        return raw
    return f"{raw} ({_fmt(hit.value)} mL)"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
#
# Replaces the placeholders declared in rules/comparators.py. The engine's
# `load_strategies()` imports this module; importing it directly (as the tests
# do) registers the same way. No cycle: rules.engine imports rules.comparators
# and never this module at top level.

register_strategy("alcohol_content", alcohol_content_strategy)
register_strategy("net_contents", net_contents_strategy)
