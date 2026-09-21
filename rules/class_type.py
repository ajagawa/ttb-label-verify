"""Class/type designation strategy — anchored, then positional, then vocabulary.

Class/type is the weakest field on the label, and the one whose expectations
need setting. The brand name is large and distinctive; the alcohol content and
net contents have a syntactic shape a pattern can find. The class/type
designation has neither. It is ordinary words ("Kentucky Straight Bourbon
Whiskey") in mid-size type, frequently abbreviated ("Ky."), frequently spelled
two legal ways ("Whisky" / "Whiskey"), and it shares vocabulary with the
bottler statement, the fine print, and — when the brand is something like
"Old Rye Whiskey Co." — with the brand name itself.

So resolution runs three stages, each a fallback for the one before, and the
result records which stage produced it (``FieldResult.strategy``) so that an
agent can tell "the label says what the application says" from "the tool
found *a* designation and it is not the one on the application":

1.  **Anchored** (``Strategy.ANCHORED``). Fuzzy search for the expected value,
    exactly as ``rules.comparators.anchored_strategy`` does for the brand name,
    except that both sides are first passed through the rule set's abbreviation
    table. Most labels resolve here.

2.  **Positional** (``Strategy.POSITIONAL``). If nothing clears the match
    threshold, restrict the pool to spans adjacent to the resolved brand name
    (``StrategyContext.anchor_bbox``) and re-score them on text alone. The
    spatial prior stands in for the OCR confidence the read lacks: a garbled,
    low-confidence line directly under the brand is far more likely to be the
    class/type than a clean fine-print mention at the bottom of the panel. A
    positional result is always REVIEW — position is evidence of *where* the
    designation is, never of *what* it says.

3.  **Vocabulary** (``Strategy.VOCABULARY``). Match the pool against the known
    designations in ``beverage_classes.<class>.class_type_vocabulary``. If the
    label clearly carries a designation that *differs* from the application,
    say so: "label reads Gin; application states Vodka" is actionable, where a
    bare not-found is indistinguishable from an OCR failure. This stage is the
    reason the field is worth checking at all.

If all three come up empty, the result is ``not_found_result`` — REVIEW, never
MISMATCH. See docs/field-assignment.md, "Not-found is never a mismatch".

REVIEW versus MISMATCH for a vocabulary finding
-----------------------------------------------

MISMATCH asserts the label is wrong. The tool's stated policy is to over-flag
to REVIEW and to reserve MISMATCH for a value that was located with high
confidence *and* clearly differs. For a class/type that means distinguishing
three situations the fuzzy score alone cannot tell apart:

*   **Refinement** — the label is *more specific* than the application
    ("Kentucky Straight Bourbon Whiskey" against "Straight Bourbon Whiskey"),
    or less specific ("Bourbon Whiskey" against "Straight Bourbon Whiskey").
    One designation's words are a subset of the other's. This is plausibly not
    a violation at all — a state of origin or an age statement folded into the
    designation — but the label and the application disagree, and deciding
    whether that is acceptable is a judgment the tool cannot make. **REVIEW.**

*   **Same family, different designation** — "Straight Rye Whiskey" against
    "Straight Bourbon Whiskey", "Cognac" against "Brandy", "Cordial" against
    "Liqueur". Some of these are genuine violations and some are a class name
    against one of its own types (27 CFR 5.143 is hierarchical: Cognac *is* a
    brandy, and "liqueur" and "cordial" are the same class). The vocabulary in
    the rule set is flat and cannot express that hierarchy, so the tool cannot
    tell which case it is looking at. **REVIEW**, with both readings in the
    reason so the agent can decide in seconds.

*   **Different family** — "Gin" against "Vodka", "Rum" against "Tequila".
    There is no reading of the regulation under which one is an acceptable
    rendering of the other. **MISMATCH**, but only when every one of the
    corroborating conditions in ``_mismatch_blockers`` holds: a near-exact
    read, high OCR confidence, located in the class/type position next to the
    brand name, a usable image, and no competing plausible read of the
    expected value elsewhere on the label. If any fails, the finding is REVIEW
    with the same text — the agent still sees "label reads Gin", just without
    the tool asserting it.

    None of the numbers behind those conditions are written in this module.
    How near-exact the read must be, how confident the engine must be, how far
    from the brand name still counts as "in position", and the positional
    stage's own floor are
    ``tuning.class_type.vocabulary_mismatch_min_similarity``,
    ``.mismatch_min_confidence``, ``.positional_reach_brand_heights`` and
    ``.positional_min_similarity`` in the versioned rule set. Every recorded
    verdict carries the rule-set version, so the thresholds that produced it
    stay recoverable; a copy of them here, even as a default, would break that.

    The field's own score bands keep a veto over all of this. A designation in
    a different family whose text still scores inside the REVIEW band is
    reported, not asserted (`close_to_expected`): a strategy that asserted a
    violation on a score the rule set calls reviewable would be contradicting
    the thresholds it was handed.

Image quality enters at exactly one point: a degraded photograph adds a
blocker, withdrawing an assertion that would otherwise have been made. It
never makes the tool more willing to assert. Inferring compliance from good
photography is not a thing this module does.

The families come from ``beverage_classes.<class>.class_type_families`` in the
rule set, not from this module, for the same reason the thresholds do: which
designations conflict is a regulatory judgment, and it belongs in the
versioned file a verdict can be reproduced against. An empty mapping means no
class/type MISMATCH is ever asserted, which is the safe reading for a beverage
class added later.

Abbreviations versus canonical folding
--------------------------------------

``canonicalize()`` folds OCR confusion classes: ``L``→``I``, ``1``→``I``,
``0``→``O``, ``5``→``S`` and so on. The abbreviation table in the rule set is
written in plain English ("KY" → "KENTUCKY"). If expansion ran on raw text,
OCR-confused variants of an abbreviation ("WH1SKY") would never expand; if it
ran on canonical text with *un*-canonicalised keys, any key or value containing
a folded character would silently never match ("SINGLE" is "SINGIE" in
canonical space).

So the table is itself canonicalised — keys and values both — and expansion
runs token-wise on canonical text, on both the candidate and the expected
value. Every comparison therefore happens in one space. The consequence to be
aware of: a key is matched *after* folding, so a key that folds onto a common
token (a hypothetical ``"L": "LITRE"`` would become ``"I"``) expands every
occurrence of that token. The current table has no such key; a schema
validator for it is recommended rather than enforced here.

Nothing here performs I/O. Same spans, same record, same rule set → same
result.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from extraction.geometry import BBox
from extraction.normalize import canonicalize
from extraction.spans import Span
from rules.comparators import (
    ResidualDifference,
    ScoredSpan,
    StrategyContext,
    explain_residual,
    not_found_result,
    residual_difference,
    score_span,
    text_similarity,
    verdict_for,
)
from rules.engine import register_strategy
from rules.results import BoxModel, FieldResult, Strategy, Verdict
from rules.schema import FieldRule, Thresholds

# --- Constants -------------------------------------------------------------
#
# What is here and what is not is a deliberate split.
#
# The three thresholds that decide whether the tool *asserts* a violation —
# the positional similarity floor, the near-exact vocabulary read, and the OCR
# confidence required to assert — live in the versioned rule set, at
# `ruleset.tuning.class_type`, and are read through `context.ruleset`. They are
# not duplicated here, not even as defaults: a module constant shadowing the
# rule set would mean the recorded `ruleset_version` no longer describes the
# thresholds that actually ran, and a fallback default is indistinguishable in
# behaviour from the rule set having been ignored.
#
# The reach of the spatial prior went the same way, at
# `tuning.class_type.positional_reach_brand_heights`. "Is this text beside the
# brand name" reads as a layout question, but its answer is one of the MISMATCH
# conditions (`not_in_position`), so widening it makes the tool marginally more
# willing to assert — which makes it policy, and policy belongs in the
# versioned file.
#
# What remains below describes layout or bounds a search: widening
# `VOCABULARY_PREFILTER_LIMIT` or `BRAND_OVERLAP_LIMIT` changes which
# candidates are looked at, and the rule set's thresholds then decide what to
# say about them. Spatial constants are multiples of the brand name's
# bounding-box height, not pixels, so they hold for a phone photo and a
# flatbed scan alike.

#: A span with at least this fraction of its own area inside the brand name's
#: bbox is excluded from the class/type pool. The engine already masks the
#: brand region at 0.5 before class/type runs; this stricter second guard
#: catches the multi-line span that straddles the brand and the line beneath
#: it, which survives the engine's mask and would otherwise let brand text
#: ("Old Rye Whiskey Co.") be read as the designation.
BRAND_OVERLAP_LIMIT = 0.25

#: Spans longer than the longest vocabulary designation by more than this many
#: words are not scored against the vocabulary. With the class/type length
#: penalty of 0.10 per excess word they could not clear the match threshold
#: anyway; skipping them keeps the vocabulary stage cheap on a pool of
#: thousands of spans.
VOCABULARY_EXCESS_WORDS = 1

#: How many vocabulary entries to fully score per span, after a fast C-level
#: prefilter. More than one because the length penalty can reorder the top
#: few; the full vocabulary is only ~35 entries, so three is ample.
VOCABULARY_PREFILTER_LIMIT = 3


# ---------------------------------------------------------------------------
# Abbreviation handling — in canonical space, on both sides
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Abbreviation:
    """One abbreviation, in both the form the agent reads and the form scored."""

    #: As written in the rule set, for the reason text ("KY" → "KENTUCKY").
    display_key: str
    display_value: str
    #: Canonical tokens, for matching and replacement.
    value_tokens: tuple[str, ...]
    #: The value canonicalised *without* OCR folding ("SINGLE", not "SINGIE"),
    #: for the one place that must show the agent real letters: the residual
    #: character difference. See `_readable_expansion`.
    plain_value_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Expansion:
    """Canonical text after abbreviation expansion, plus what was expanded."""

    text: str
    applied: tuple[_Abbreviation, ...]


@functools.lru_cache(maxsize=16)
def _abbreviation_table(
    items: tuple[tuple[str, str], ...],
) -> dict[tuple[str, ...], _Abbreviation]:
    """Canonicalise the rule set's abbreviation table, keys *and* values.

    This is the step that puts expansion in the same space as scoring. See the
    module docstring, "Abbreviations versus canonical folding": a key or value
    left in plain English would silently never match once folding has turned
    L into I, and an OCR-confused abbreviation ("WH1SKY") would never expand if
    the key were matched against raw text.

    Keys that canonicalise to nothing, or to their own value, are dropped —
    they could only ever be no-ops or loops.

    Cached on the table's items: the rule set is immutable and this runs once
    per verification otherwise.
    """
    table: dict[tuple[str, ...], _Abbreviation] = {}
    for key, value in items:
        key_tokens = tuple(canonicalize(key).split())
        value_tokens = tuple(canonicalize(value).split())
        if not key_tokens or key_tokens == value_tokens:
            continue
        table[key_tokens] = _Abbreviation(
            display_key=key,
            display_value=value,
            value_tokens=value_tokens,
            plain_value_tokens=tuple(canonicalize(value, fold_ocr_confusions=False).split()),
        )
    return table


def _table_for(abbreviations: dict[str, str]) -> dict[tuple[str, ...], _Abbreviation]:
    # Sorted so that the cache key, and therefore behaviour, does not depend on
    # dict insertion order.
    return _abbreviation_table(tuple(sorted(abbreviations.items())))


def _expand(canonical_text: str, table: dict[tuple[str, ...], _Abbreviation]) -> _Expansion:
    """Replace abbreviation tokens with their expansions, one pass, longest first.

    Token-level, never substring: "KY" must expand in "KY STRAIGHT BOURBON" and
    must not expand inside "KYOTO". Multi-token keys are supported and the
    longest key wins at each position, so a future "STR BRBN" entry would beat
    "STR" alone.

    Single pass on purpose: an expansion's output is not re-expanded. A table
    that chains ("A"→"B", "B"→"C") would otherwise make the result depend on
    iteration order, and one that cycles would never terminate.

    Args:
        canonical_text: Text already passed through `canonicalize()`.
        table: From `_abbreviation_table`.

    Returns:
        The expanded canonical text and the abbreviations that fired.
    """
    tokens = canonical_text.split()
    if not table or not tokens:
        return _Expansion(text=canonical_text, applied=())

    longest_key = max(len(k) for k in table)
    out: list[str] = []
    applied: list[_Abbreviation] = []
    position = 0

    while position < len(tokens):
        for size in range(min(longest_key, len(tokens) - position), 0, -1):
            entry = table.get(tuple(tokens[position : position + size]))
            if entry is not None:
                out.extend(entry.value_tokens)
                applied.append(entry)
                position += size
                break
        else:
            out.append(tokens[position])
            position += 1

    return _Expansion(text=" ".join(out), applied=tuple(applied))


def _readable_expansion(raw: str, table: dict[tuple[str, ...], _Abbreviation]) -> str:
    """Abbreviation-expanded text with real letters, for the residual count.

    The residual rule (`rules.comparators.residual_difference`) must see the
    two values *after* expansion, or "Ky." against "Kentucky" counts as six
    differing characters and every abbreviated label becomes a false flag. It
    also describes the differences to the agent, so it needs letters as
    printed — "“O” is missing", never a folded "“I” is missing" standing for an
    L the label actually carries.

    Those two requirements pull in different directions: keys must be matched
    in *folded* space (so "WH1SKY" still expands), but the output must be
    *unfolded*. So expansion runs in parallel over both canonical forms of the
    same text. Folding is a per-character translation applied after
    tokenisation, so the two forms always have the same tokens in the same
    positions; a key is matched on the folded tokens and the replacement is
    written from the entry's unfolded value.

    `residual_difference` canonicalises its inputs again, folding included.
    Folding the output here yields exactly `_expand(...).text`, so the count
    it computes is the count in the space everything else is scored in.

    Args:
        raw: Raw text — a span's `raw_text` or the application value.
        table: From `_abbreviation_table`.

    Returns:
        Canonical, expanded, unfolded text.
    """
    folded = canonicalize(raw).split()
    plain = canonicalize(raw, fold_ocr_confusions=False).split()
    if not table or len(folded) != len(plain):
        # The length check is defensive: tokenisation is shared, so it cannot
        # differ today. If canonicalisation ever changed that, falling back to
        # unexpanded text over-flags (an abbreviation reads as a difference)
        # rather than under-flags, which is the right way to fail.
        return " ".join(plain)

    longest_key = max(len(k) for k in table)
    out: list[str] = []
    position = 0
    while position < len(folded):
        for size in range(min(longest_key, len(folded) - position), 0, -1):
            entry = table.get(tuple(folded[position : position + size]))
            if entry is not None:
                out.extend(entry.plain_value_tokens)
                position += size
                break
        else:
            out.append(plain[position])
            position += 1
    return " ".join(out)


def _residual(found_raw: str, expected_raw: str, table) -> ResidualDifference:
    """The residual character difference, measured after abbreviation expansion."""
    return residual_difference(
        _readable_expansion(found_raw, table), _readable_expansion(expected_raw, table)
    )


# ---------------------------------------------------------------------------
# Scoring helpers — thin wrappers over rules.comparators so the composite
# score is identical to the brand name's, just computed on expanded text
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A pool span with its expanded canonical text, computed once."""

    span: Span
    expanded: _Expansion
    near_brand: bool


def _score_expanded(candidate: _Candidate, target: str, rule: FieldRule) -> ScoredSpan:
    """`score_span`, but on abbreviation-expanded text.

    Reuses the comparator's composite (similarity x confidence x length
    penalty) rather than reimplementing it, by scoring a copy of the span whose
    canonical text has been expanded. The returned `ScoredSpan` carries the
    *original* span, so the raw text and bbox reported to the agent are
    exactly what was recognised.

    `score_span` re-canonicalises `target`. That is safe because
    `canonicalize` is idempotent on its own output (asserted in the tests):
    folding maps onto letters that are not themselves folded.
    """
    proxy = dataclasses.replace(candidate.span, canonical_text=candidate.expanded.text)
    scored = score_span(proxy, target, rule)
    return dataclasses.replace(scored, span=candidate.span)


def _text_score(scored: ScoredSpan) -> float:
    """Similarity x length penalty — the composite without the confidence factor.

    Used by the positional stage, where the spatial prior takes the place of
    OCR confidence as corroboration. The result of that stage is always
    REVIEW, so dropping the confidence factor can never promote a low-quality
    read to a MATCH; it only lets the tool point at the region.
    """
    return scored.similarity * scored.length_penalty


def _is_near_brand(box: BBox, anchor: BBox | None, reach_brand_heights: float) -> bool:
    """Whether a span sits in the class/type position relative to the brand.

    Class/type usually sits directly below the brand in smaller type; it is
    sometimes a banner directly above. Both are accepted. "Near" means within
    `reach_brand_heights` brand-heights vertically *and* sharing the brand's
    column (horizontally overlapping it, widened by the same reach), so a line
    of fine print at the same height on the far side of the panel does not
    qualify.

    Args:
        box: The candidate span's region.
        anchor: The resolved brand name's region, or None.
        reach_brand_heights:
            `tuning.class_type.positional_reach_brand_heights` from the rule
            set. It gates one of the MISMATCH conditions, which is why it is
            rule-set data rather than a constant here. Measured against the
            brand's *bounding box*, so a brand wrapped onto two lines has an
            effective reach of about four glyph heights.
    """
    if anchor is None or anchor.height <= 0:
        return False
    reach = anchor.height * reach_brand_heights
    if box.vertical_gap(anchor) > reach:
        return False
    column = BBox(
        left=anchor.left - reach, top=anchor.top, right=anchor.right + reach, bottom=anchor.bottom
    )
    return box.horizontal_overlap_ratio(column) > 0.0


def _overlaps_brand(span: Span, anchor: BBox | None) -> bool:
    """The brand guard. See `BRAND_OVERLAP_LIMIT`."""
    return anchor is not None and span.bbox.overlaps(anchor, min_fraction=BRAND_OVERLAP_LIMIT)


def _contains(outer: BBox, inner: BBox) -> bool:
    """`inner` lies (almost) wholly inside `outer`."""
    return inner.overlaps(outer, min_fraction=0.9)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Designation:
    """A vocabulary entry, as displayed and as scored."""

    name: str  # as written in the rule set
    expanded: str  # canonical + abbreviation-expanded


@dataclass(frozen=True, slots=True)
class _VocabularyHit:
    """A span that strongly matches a known designation differing from expected."""

    designation: _Designation
    scored: ScoredSpan  # against the designation
    agreement: ScoredSpan  # the same span scored against the *expected* value
    near_brand: bool


def _designations(
    vocabulary: Iterable[str], table: dict[tuple[str, ...], _Abbreviation]
) -> list[_Designation]:
    """Expand the vocabulary once, collapsing entries that become identical.

    "Canadian Whisky" and a hypothetical "Canadian Whiskey" would both expand
    to CANADIAN WHISKEY; keeping both would only double the work and make the
    reported name depend on list order. The first-listed spelling is kept.
    """
    seen: dict[str, _Designation] = {}
    for name in vocabulary:
        expanded = _expand(canonicalize(name), table).text
        if expanded and expanded not in seen:
            seen[expanded] = _Designation(name=name, expanded=expanded)
    return list(seen.values())


def _same_designation(a: str, b: str, match_threshold: float) -> bool:
    """Whether two expanded designations are the same designation.

    Fuzzy rather than exact, so that an application typed "Straight Burbon
    Whiskey" is not treated as a *different* designation from the vocabulary's
    "Straight Bourbon Whiskey" and escalated on the strength of a typo.

    Uses the comparator's own `text_similarity`, so that "what counts as the
    same designation" here is the same question the scorer answers — including
    its insensitivity to word spacing, which matters because PP-OCR returns
    letterspaced display type as one token.
    """
    return text_similarity(a, b) >= match_threshold


def _families(
    expanded: str,
    families: Mapping[str, tuple[str, ...]],
    table: dict[tuple[str, ...], _Abbreviation],
) -> frozenset[str]:
    """Families a designation belongs to, from `beverage.class_type_families`.

    Membership is by whole-word phrase in the same canonical,
    abbreviation-expanded space everything else here is compared in, so
    "Kentucky Straight Bourbon Whiskey" is in `whisky` by virtue of WHISKEY.
    The terms are expanded too, not just canonicalised: a family listing only
    "Whisky" must still match a designation whose WHISKY the table has already
    rewritten to WHISKEY, or the rule set's two spellings would silently
    disagree with each other.

    A designation matching no family can never produce a MISMATCH — an
    incomplete or empty mapping under-asserts rather than over-asserts.
    """
    padded = f" {expanded} "
    return frozenset(
        family
        for family, terms in families.items()
        if any(f" {_expand(canonicalize(term), table).text} " in padded for term in terms)
    )


def _vocabulary_hits(
    candidates: list[_Candidate],
    designations: list[_Designation],
    target: str,
    rule: FieldRule,
    match_threshold: float,
) -> list[_VocabularyHit]:
    """Every span that strongly matches a designation differing from expected.

    One hit per span (its best designation). Prefiltered with rapidfuzz's
    C-level `process.extract` because the pool runs to thousands of spans and
    the vocabulary to dozens of entries; only the top few per span get the
    full composite score. The similarity cutoff is safe as a prefilter because
    the composite can never exceed the similarity.

    The prefilter runs twice, mirroring the two measures in the comparator's
    `text_similarity`: once token-wise, and once on both strings with spaces
    removed. The second pass is not optional — PP-OCR returns letterspaced
    display type as a single token ("KentuckyStraightBourbonWhiskey"), which a
    token-based prefilter scores near zero against a four-word designation and
    would drop before it was ever scored properly.
    """
    if not designations:
        return []

    choices = [d.expanded for d in designations]
    despaced_choices = [c.replace(" ", "") for c in choices]
    max_words = max(len(c.split()) for c in choices) + VOCABULARY_EXCESS_WORDS
    cutoff = match_threshold * 100.0
    hits: list[_VocabularyHit] = []

    for candidate in candidates:
        text = candidate.expanded.text
        if len(text.split()) > max_words:
            continue
        shortlist = {
            index
            for _choice, _similarity, index in process.extract(
                text,
                choices,
                scorer=fuzz.token_sort_ratio,
                limit=VOCABULARY_PREFILTER_LIMIT,
                score_cutoff=cutoff,
            )
        } | {
            index
            for _choice, _similarity, index in process.extract(
                text.replace(" ", ""),
                despaced_choices,
                scorer=fuzz.ratio,
                limit=VOCABULARY_PREFILTER_LIMIT,
                score_cutoff=cutoff,
            )
        }

        best: tuple[_Designation, ScoredSpan] | None = None
        for index in sorted(shortlist):
            designation = designations[index]
            scored = _score_expanded(candidate, designation.expanded, rule)
            # Text score, not the composite: a clearly printed "Gin" read at
            # 60% confidence is still a "Gin" worth reporting. Confidence is
            # applied where it belongs — as a MISMATCH blocker — so a doubtful
            # read surfaces as REVIEW instead of vanishing into not-found.
            if _text_score(scored) < match_threshold:
                continue
            if best is None or _text_score(scored) > _text_score(best[1]):
                best = (designation, scored)

        if best is None:
            continue
        designation, scored = best
        if _same_designation(designation.expanded, target, match_threshold):
            # The label carries the expected designation; that is stage 1's
            # business, not a finding.
            continue
        hits.append(
            _VocabularyHit(
                designation=designation,
                scored=scored,
                agreement=_score_expanded(candidate, target, rule),
                near_brand=candidate.near_brand,
            )
        )

    return hits


def _most_specific(hits: list[_VocabularyHit], chosen: _VocabularyHit) -> _VocabularyHit:
    """Grow a hit to the longest hit containing it.

    The line "Kentucky Straight Bourbon Whiskey" produces sub-line spans that
    are themselves exact vocabulary hits — "Straight Bourbon Whiskey",
    "Bourbon Whiskey". If a sub-span happened to read slightly cleaner, it
    would win on score and the agent would be told the label says less than
    it does. Reporting the most specific designation physically containing
    the chosen one is always the more accurate statement.
    """
    containing = [
        h
        for h in hits
        if _contains(h.scored.span.bbox, chosen.scored.span.bbox)
        and h.scored.span.word_count > chosen.scored.span.word_count
    ]
    if not containing:
        return chosen
    return max(containing, key=lambda h: (h.scored.span.word_count, h.scored.score))


def _best_vocabulary_hit(hits: list[_VocabularyHit]) -> _VocabularyHit | None:
    """Prefer a hit in the class/type position, then the strongest, then the longest.

    Python's sort is stable, so ties fall back to pool order, which is
    deterministic — the same label always reports the same designation.
    """
    if not hits:
        return None
    ranked = sorted(
        hits,
        key=lambda h: (h.near_brand, round(h.scored.score, 4), h.scored.span.word_count),
        reverse=True,
    )
    return _most_specific(hits, ranked[0])


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


def class_type_strategy(context: StrategyContext) -> FieldResult:
    """Anchored search, then a spatial prior, then a controlled vocabulary.

    See the module docstring for the design and the REVIEW-versus-MISMATCH
    policy. In outline:

    1.  Score the pool (brand region excluded) against the expected value on
        abbreviation-expanded canonical text. A score at or above the match
        threshold resolves the field — unless the matched text is part of a
        longer, *different* known designation on the label, in which case the
        label is more specific than the application and that is reported
        (REVIEW, vocabulary).
    2.  Otherwise, look beside the brand name for the best text-only read.
    3.  Otherwise — or in preference to a weak read, when the label clearly
        carries a different known designation — report that designation.
    4.  Otherwise, a weak anchored read in the REVIEW band, or not-found.

    Args:
        context: The strategy context. `anchor_bbox` is the resolved brand
            name's region, or None when the brand was not located (stage 2 is
            then skipped and a vocabulary finding cannot be a MISMATCH).

    Returns:
        A `FieldResult` whose `strategy` records which stage produced it.
    """
    rule = context.field_rule
    expected_canonical = canonicalize(context.expected)
    if not expected_canonical:
        return _no_expected_value_result(context)

    table = _table_for(context.beverage.abbreviations)
    target = _expand(expected_canonical, table)
    anchor = context.anchor_bbox
    thresholds = context.ruleset.thresholds_for(rule.id)
    # Assertion thresholds come from the versioned rule set, never from this
    # module — see the note above the constants.
    tuning = context.ruleset.tuning.class_type

    candidates = [
        _Candidate(
            span=span,
            expanded=_expand(span.canonical_text, table),
            near_brand=_is_near_brand(span.bbox, anchor, tuning.positional_reach_brand_heights),
        )
        for span in context.spans
        if not _overlaps_brand(span, anchor)
    ]
    if not candidates:
        return not_found_result(context)

    by_candidate = {id(c.span): c for c in candidates}
    ranked = sorted(
        (_score_expanded(c, target.text, rule) for c in candidates),
        key=lambda s: s.score,
        reverse=True,
    )
    best = ranked[0]
    designations = _designations(context.beverage.class_type_vocabulary, table)

    # --- Stage 1: anchored ------------------------------------------------
    if best.score >= thresholds.match:
        # Several spans can clear the threshold — the display line, a wrapped
        # two-line span over the same words, a fine-print mention. Prefer the
        # highest-scoring one whose residual character difference is within
        # the rule set's budget: if the label carries the expected designation
        # exactly anywhere, that is the reading to report, and a misread copy
        # elsewhere that happened to score higher (a cleaner OCR confidence,
        # say) must not turn a compliant label into a REVIEW. Only when every
        # clearing span has a residual is the top one reported with it.
        budget = context.ruleset.tuning.names.max_residual_character_edits
        clearing = [s for s in ranked if s.score >= thresholds.match]
        residuals = [_residual(s.span.raw_text, context.expected, table) for s in clearing]
        within = [i for i, r in enumerate(residuals) if r.count <= budget]
        chosen_index = within[0] if within else 0
        best, residual = clearing[chosen_index], residuals[chosen_index]

        # The expected text is on the label. Before calling it a match, check
        # that it is not merely a *part* of a longer, different designation:
        # "Straight Bourbon Whiskey" is a sub-span of the line "Kentucky
        # Straight Bourbon Whiskey", and scores 1.0 against an application
        # stating the shorter form. This runs before the residual verdict and
        # takes precedence over it: a vocabulary finding names the designation
        # the label actually carries, which says more than a character count.
        containing = [
            c
            for c in candidates
            if _contains(c.span.bbox, best.span.bbox) and c.span.word_count > best.span.word_count
        ]
        refinements = _vocabulary_hits(
            containing, designations, target.text, rule, thresholds.match
        )
        if refinements:
            chosen = _best_vocabulary_hit(refinements)
            assert chosen is not None  # non-empty input
            return _vocabulary_result(context, chosen, target, blockers=("refinement",))
        return _anchored_result(context, best, target, table, residual)

    # --- Stage 2: positional prior ---------------------------------------
    positional = _positional_candidate(ranked, by_candidate, tuning.positional_min_similarity)

    # --- Stage 3: controlled vocabulary ----------------------------------
    hits = _vocabulary_hits(candidates, designations, target.text, rule, thresholds.match)
    hit = _best_vocabulary_hit(_without_fragments(hits, target, ranked, positional, thresholds))
    if hit is not None:
        blockers = _mismatch_blockers(context, hit, target, ranked, positional, table)
        return _vocabulary_result(context, hit, target, blockers=blockers)

    # --- Weak reads --------------------------------------------------------
    best_is_plausible = best.score > thresholds.mismatch
    best_is_near = by_candidate[id(best.span)].near_brand

    if best_is_plausible and (best_is_near or positional is None):
        return _review_result(context, best, Strategy.ANCHORED)
    if positional is not None:
        return _review_result(context, positional, Strategy.POSITIONAL)

    return not_found_result(context)


def _positional_candidate(
    ranked: list[ScoredSpan], by_candidate: dict[int, _Candidate], min_similarity: float
) -> ScoredSpan | None:
    """Stage 2: the best text-only read in the class/type position.

    Returns None when no brand anchor is available (every candidate is then
    `near_brand=False`) or when nothing beside the brand reads plausibly.
    Ties on text score go to the higher composite score (i.e. the more
    confident read), then to ranking order.

    Args:
        ranked: The scored pool.
        by_candidate: Candidate metadata, keyed by `id()` of the span.
        min_similarity: `tuning.class_type.positional_min_similarity` from the
            rule set. Below it, decoration that sits beside a brand name
            ("Small Batch", "Bottled in Bond") starts to qualify as a reading
            of the designation.
    """
    near = [s for s in ranked if by_candidate[id(s.span)].near_brand]
    if not near:
        return None
    chosen = max(near, key=lambda s: (round(_text_score(s), 4), s.score))
    if _text_score(chosen) < min_similarity:
        return None
    return chosen


def _without_fragments(
    hits: list[_VocabularyHit],
    target: _Expansion,
    ranked: list[ScoredSpan],
    positional: ScoredSpan | None,
    thresholds: Thresholds,
) -> list[_VocabularyHit]:
    """Drop hits that are only a fragment of a weak read of the expected value.

    The mirror image of the refinement check in stage 1. A poor read of
    "Kentucky Straight Bourbon Whiskey" — "Kntcky Strght Brbn Whsky" — has a
    sub-span "Brbn Whsky" that expands to an exact "Bourbon Whiskey". Reported
    as a vocabulary finding, that would tell the agent the label says *less*
    than the application, when the line as a whole is plainly an attempt at the
    application's designation. The hit is dropped when its designation is a
    subset of the expected one *and* a longer span containing it reads
    plausibly as the expected value; the weak read is then reported as such.
    """
    expected_tokens = set(target.text.split())
    plausible = [s for s in ranked if s.score > thresholds.mismatch]
    if positional is not None:
        plausible.append(positional)

    def is_fragment(hit: _VocabularyHit) -> bool:
        if not set(hit.designation.expanded.split()) <= expected_tokens:
            return False
        span = hit.scored.span
        return any(
            _contains(s.span.bbox, span.bbox) and s.span.word_count > span.word_count
            for s in plausible
        )

    return [h for h in hits if not is_fragment(h)]


def _mismatch_blockers(
    context: StrategyContext,
    hit: _VocabularyHit,
    target: _Expansion,
    ranked: list[ScoredSpan],
    positional: ScoredSpan | None,
    table: dict[tuple[str, ...], _Abbreviation],
) -> tuple[str, ...]:
    """Every reason a vocabulary finding must stay REVIEW. Empty → MISMATCH.

    Each condition is one piece of corroboration that the tool has located the
    class/type and that it is clearly a different designation. All must hold;
    the default is REVIEW. Returned as a tuple of tags rather than a bool so
    the reason text can tell the agent *why* the tool is not asserting.

    Args:
        context: The strategy context; the thresholds, the beverage class's
            designation families and the image quality all come from it.
        hit: The vocabulary finding under consideration.
        target: The expected value, canonical and expanded.
        ranked: The scored pool, for the competing-read check.
        positional: Stage 2's candidate, if any.
        table: The abbreviation table, for expanding the family terms.
    """
    thresholds = context.ruleset.thresholds_for(context.field_rule.id)
    tuning = context.ruleset.tuning.class_type
    span = hit.scored.span
    blockers: list[str] = []

    found_tokens = set(hit.designation.expanded.split())
    expected_tokens = set(target.text.split())
    if found_tokens <= expected_tokens or expected_tokens <= found_tokens:
        blockers.append("refinement")
    else:
        families = context.beverage.class_type_families
        found_families = _families(hit.designation.expanded, families, table)
        expected_families = _families(target.text, families, table)
        if not found_families or not expected_families:
            blockers.append("unknown_family")
        elif found_families & expected_families:
            blockers.append("same_family")

    if hit.scored.similarity < tuning.vocabulary_mismatch_min_similarity:
        blockers.append("weak_read")
    if span.mean_confidence < max(
        tuning.mismatch_min_confidence, context.ruleset.defaults.min_ocr_confidence
    ):
        blockers.append("low_confidence")
    if context.anchor_bbox is None:
        blockers.append("no_brand_anchor")
    elif not hit.near_brand:
        blockers.append("not_in_position")
    if context.quality is not None and (
        not context.quality.usable
        or context.quality.score < context.ruleset.defaults.good_image_quality
    ):
        # Quality may only withdraw an assertion, never add one, so it appears
        # here and nowhere else in this module. On a poor photograph the more
        # likely explanation for an unexpected designation is the photograph.
        blockers.append("degraded_image")
    if hit.agreement.score > thresholds.mismatch:
        # The found text is still in the REVIEW band against the application.
        # The families above *permit* an assertion; they do not force one, and
        # this is where the field's own score bands keep their veto. Asserting
        # a violation on a score the rule set calls reviewable would be the
        # strategy contradicting the thresholds it was handed — so "Straight
        # Bourbon Whiskey" against "Straight Rye Whiskey" (0.81) is reported
        # even if a rule set puts bourbon and rye in different families.
        blockers.append("close_to_expected")

    # A plausible read of the expected value somewhere *else* on the label
    # means the label may carry both — the designation the application states
    # and another one. The tool cannot tell which is the class/type statement.
    competing = [
        s
        for s in ranked
        if s.score > thresholds.mismatch and not s.span.bbox.overlaps(span.bbox, 0.25)
    ]
    if positional is not None and not positional.span.bbox.overlaps(span.bbox, 0.25):
        competing.append(positional)
    if competing:
        blockers.append("competing_read")

    return tuple(blockers)


# ---------------------------------------------------------------------------
# Result builders and plain-language reasons
# ---------------------------------------------------------------------------
#
# Reasons are shown verbatim to a compliance agent. Every flag costs them
# attention, so each reason says what to look at and, for REVIEW, why the tool
# is not deciding. Canonical text is never shown: it is uppercased and folded
# ("SINGIE MAIT") and would read as an OCR error the tool introduced.


def _field_result(
    context: StrategyContext,
    scored: ScoredSpan,
    *,
    verdict: Verdict,
    strategy: Strategy,
    score: float,
    reason: str,
) -> FieldResult:
    rule = context.field_rule
    return FieldResult(
        field_id=rule.id,
        label=rule.label,
        verdict=verdict,
        strategy=strategy,
        expected=context.expected,
        found=scored.span.raw_text,
        bbox=BoxModel(**scored.span.bbox.to_dict()),
        score=round(max(0.0, min(1.0, score)), 4),
        ocr_confidence=round(scored.span.mean_confidence, 4),
        reason=reason,
        citation=rule.citation,
        citation_url=rule.citation_url,
    )


def _anchored_result(
    context: StrategyContext,
    best: ScoredSpan,
    target: _Expansion,
    table: dict[tuple[str, ...], _Abbreviation],
    residual: ResidualDifference,
) -> FieldResult:
    """Stage 1 success: MATCH, or REVIEW for a low-confidence read or a residual.

    The residual rule is the same one the brand name applies, for the same
    reason, and on this field its effect is at its largest. A similarity ratio
    divides by length, so one wrong letter in a thirty-character designation
    barely moves it: "Kentucky Straight Burbon Whiskey" scores 0.97 against
    the application's "Bourbon" and clears the match threshold. After
    canonicalisation, the OCR folds and abbreviation expansion have removed
    every difference that carries no meaning, a character that still differs
    is either a misread outside those classes or a genuinely different word,
    and the text alone cannot say which. So it goes to a person.

    The budget is `tuning.names.max_residual_character_edits`, shared with the
    brand name so that both name-like fields hold the same line. It only ever
    moves MATCH to REVIEW, never to MISMATCH: a misread and a misprint look
    identical from here. The reason text comes from `explain_residual`, so an
    agent reads one wording for one kind of finding whichever field raised it.

    Args:
        context: The strategy context.
        best: The stage-1 winner.
        target: The expected value, expanded.
        table: The abbreviation table, for the MATCH explanation.
        residual: Character edits between `best` and the expected value,
            measured after abbreviation expansion (`_residual`).
    """
    verdict = verdict_for(
        best.score, best.span.mean_confidence, context.field_rule.id, context.ruleset
    )
    found, expected = best.span.raw_text, context.expected

    if (
        verdict is Verdict.MATCH
        and residual.count > context.ruleset.tuning.names.max_residual_character_edits
    ):
        return _field_result(
            context,
            best,
            verdict=Verdict.REVIEW,
            strategy=Strategy.ANCHORED,
            score=best.score,
            reason=explain_residual(found, expected, residual),
        )

    if verdict is not Verdict.MATCH:
        # The only way past the match threshold to a non-MATCH is low OCR
        # confidence (verdict_for); the text agrees but cannot be trusted.
        reason = (
            f"Read the class/type as “{found}”, which agrees with the application, "
            f"but with low confidence ({best.span.mean_confidence:.0%}). "
            "The text may be small or unclear — confirm visually."
        )
    elif found == expected:
        reason = "Label matches the application record exactly."
    elif found.casefold() == expected.casefold():
        reason = (
            "Matches the application record, differing only in capitalisation "
            f"(label reads “{found}”)."
        )
    else:
        found_expansion = _expand(best.span.canonical_text, table)
        applied = _describe_abbreviations(found_expansion.applied + target.applied)
        if applied and found_expansion.text == target.text:
            reason = (
                f"Matches the application record once standard abbreviations and "
                f"spellings are read as equivalent ({applied}). Label reads “{found}”."
            )
        else:
            reason = f"Matches the application record (label reads “{found}”)."

    return _field_result(
        context, best, verdict=verdict, strategy=Strategy.ANCHORED, score=best.score, reason=reason
    )


def _describe_abbreviations(applied: Iterable[_Abbreviation]) -> str:
    """ "KY” read as “KENTUCKY”; …" — deduplicated, in first-seen order."""
    seen: dict[tuple[str, str], None] = {}
    for entry in applied:
        seen.setdefault((entry.display_key, entry.display_value), None)
    return "; ".join(f"“{k}” read as “{v}”" for k, v in seen)


def _review_result(context: StrategyContext, scored: ScoredSpan, strategy: Strategy) -> FieldResult:
    """A weak read: REVIEW, from stage 1's REVIEW band or from stage 2."""
    found, expected = scored.span.raw_text, context.expected
    low_confidence = scored.span.mean_confidence < context.ruleset.defaults.min_ocr_confidence
    confidence_note = (
        f" The text was read with low confidence ({scored.span.mean_confidence:.0%})."
        if low_confidence
        else ""
    )

    if strategy is Strategy.POSITIONAL:
        reason = (
            f"The text beside the brand name, where the class/type designation normally "
            f"appears, reads “{found}” — not a clear match for “{expected}” in the "
            f"application.{confidence_note} It may be stylized or unclear; confirm "
            "visually that it states the application's designation."
        )
    elif (
        _text_score(scored) >= context.ruleset.tuning.class_type.vocabulary_mismatch_min_similarity
    ):
        # The words agree near-exactly; only the confidence factor held the
        # composite below the match threshold. Saying "close but not an exact
        # match" here would send the agent hunting for a textual difference
        # that does not exist. The bar is the rule set's near-exact reading
        # threshold, not the match threshold, so a genuinely garbled read at
        # 0.90 is not described as agreeing.
        reason = (
            f"Read the class/type as “{found}”, which agrees with the application, but "
            f"with limited confidence ({scored.span.mean_confidence:.0%}). The text may "
            "be small or unclear — confirm visually."
        )
    else:
        reason = (
            f"Label reads “{found}” against “{expected}” in the application. Close but "
            f"not an exact match.{confidence_note} Confirm the designation is the same."
        )

    return _field_result(
        context,
        scored,
        verdict=Verdict.REVIEW,
        strategy=strategy,
        score=scored.score,
        reason=reason,
    )


#: Plain-language explanation for each MISMATCH blocker, in the order they are
#: most useful to an agent. Only the first applicable one is shown: one clear
#: reason is more useful than a list of caveats.
_BLOCKER_EXPLANATIONS: dict[str, str] = {
    "refinement": (
        "One designation is a more or less specific form of the other. That is "
        "often acceptable, but the label and the application should agree — "
        "confirm which is correct."
    ),
    "same_family": (
        "Both are designations of the same general class, and the tool cannot "
        "tell whether one is an acceptable form of the other — confirm against "
        "27 CFR 5.143."
    ),
    "unknown_family": (
        "The tool cannot classify one of the two designations, so it does not "
        "decide whether they conflict — confirm manually."
    ),
    "competing_read": (
        "Text resembling the application's designation also appears elsewhere on "
        "the label, so it is unclear which is the class/type statement — confirm "
        "visually."
    ),
    "degraded_image": (
        "The image quality is poor, so the difference is as likely to be the "
        "photograph as the label — confirm visually, or re-photograph the label."
    ),
    "low_confidence": (
        "The text was read with limited confidence, so the tool does not assert a "
        "discrepancy — confirm visually."
    ),
    "weak_read": (
        "The reading is close to, but not exactly, the known designation — confirm visually."
    ),
    "no_brand_anchor": (
        "The brand name was not located, so the tool cannot confirm this text is in "
        "the class/type position — confirm visually."
    ),
    "not_in_position": (
        "It is not beside the brand name where the class/type normally appears, and "
        "may be incidental text — confirm visually."
    ),
    "close_to_expected": (
        "It is close enough to the application's wording that the difference may be "
        "a misread — confirm visually."
    ),
}


def _vocabulary_result(
    context: StrategyContext,
    hit: _VocabularyHit,
    target: _Expansion,
    *,
    blockers: tuple[str, ...],
) -> FieldResult:
    """Stage 3: the label carries a known designation other than the expected one.

    `score` is the found text's agreement with the *application* value, not
    its similarity to the vocabulary entry. The score field means "how well
    does the label agree with the record" everywhere else, and batch triage
    orders worst-first on it; reporting the 1.0 vocabulary similarity here
    would sort a clear discrepancy to the bottom of the worklist.
    """
    found, expected = hit.scored.span.raw_text, context.expected
    verdict = Verdict.REVIEW if blockers else Verdict.MISMATCH

    if verdict is Verdict.MISMATCH:
        reason = (
            f"Label reads “{found}”, which is a different class/type designation from "
            f"“{expected}” in the application. It appears in the class/type position "
            "beside the brand name and was read clearly. Verify against the submitted "
            "artwork."
        )
    else:
        detail = next(
            (_BLOCKER_EXPLANATIONS[b] for b in _BLOCKER_EXPLANATIONS if b in blockers),
            "Confirm visually.",
        )
        reason = (
            f"Label reads “{found}” (a recognised designation); the application states "
            f"“{expected}”. {detail}"
        )

    return _field_result(
        context,
        hit.scored,
        verdict=verdict,
        strategy=Strategy.VOCABULARY,
        score=hit.agreement.score,
        reason=reason,
    )


def _no_expected_value_result(context: StrategyContext) -> FieldResult:
    """The application record gave nothing to check against.

    Not the generic not-found text: "could not locate the class/type" would be
    a false statement about what the tool did. It never looked, because there
    was nothing to look for.
    """
    rule = context.field_rule
    return FieldResult(
        field_id=rule.id,
        label=rule.label,
        verdict=Verdict.REVIEW,
        strategy=Strategy.NOT_FOUND,
        expected=context.expected or None,
        found=None,
        bbox=None,
        score=0.0,
        ocr_confidence=0.0,
        reason=(
            "The application record gives no class/type designation to check the label "
            "against — verify manually."
        ),
        citation=rule.citation,
        citation_url=rule.citation_url,
    )


register_strategy("class_type", class_type_strategy)
