"""Candidate span generation — stage 1 of field assignment.

OCR returns lines. Fields do not respect line boundaries in either direction:

*   A brand name wraps: ``OLD TOM`` / ``DISTILLERY`` on two lines.
*   One line carries two fields: ``750 mL  •  45% Alc./Vol.``

Matching against raw OCR lines therefore fails on ordinary labels, in both
directions at once. This module builds an exhaustive index of *candidate spans*
— every contiguous run of words within a line, and every spatially coherent run
of consecutive lines — so that whichever grouping happens to correspond to a
field is present in the pool and can be scored.

The index is generated once per verification and shared by every field
strategy. For a typical label of 30-60 lines it produces a few thousand
candidates, which is cheap to score exhaustively and much simpler than trying
to segment correctly up front.

See docs/field-assignment.md for the design reasoning.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from extraction.geometry import BBox
from extraction.normalize import canonicalize, canonicalize_numeric
from extraction.providers.base import Line

# --- Tuning constants ------------------------------------------------------
#
# Every spatial threshold is expressed as a multiple of the image's median line
# height rather than in absolute pixels, so that the same values hold for a
# 900px phone photo and a 4000px flatbed scan. Absolute pixel thresholds are a
# bug waiting for a differently sized upload.

#: Longest run of consecutive lines considered as one span. Four covers a
#: wrapped brand name (2), a multi-line class/type designation (2-3), and a
#: wrapped address block. The warning statement is longer than this and is
#: handled by anchored expansion instead, not by the generic index.
MAX_LINES_PER_SPAN = 4

#: Longest run of words within a line considered as one span. Six covers every
#: field value in scope; the warning statement again gets its own treatment.
MAX_WORDS_PER_SPAN = 6

#: Two lines join a span only if the whitespace between them is less than this
#: multiple of the mean line height. Wider than this and they are separate
#: blocks that happen to be vertically adjacent.
MAX_VERTICAL_GAP_RATIO = 1.5

#: Two lines join a span only if they overlap horizontally by at least this
#: fraction of the narrower line's width. Distinguishes stacked text from two
#: columns sitting at the same height.
MIN_HORIZONTAL_OVERLAP = 0.3

#: Two lines join a span only if their apparent type sizes are within this
#: ratio of each other. This is what stops a display-size brand name from
#: merging with the 6pt legal text beneath it — the single most useful of the
#: three coherence tests in practice.
MIN_GLYPH_HEIGHT_RATIO = 0.6
MAX_GLYPH_HEIGHT_RATIO = 1 / MIN_GLYPH_HEIGHT_RATIO

#: Lines whose vertical centres fall within this multiple of the median line
#: height are treated as one visual row and ordered left-to-right within it.
ROW_GROUPING_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous run of words or lines, treated as one match candidate.

    Carries both text forms deliberately. See extraction/normalize.py: scoring
    reads ``canonical_text``, formatting checks read ``raw_text``, and confusing
    the two produces silent false negatives on the capitalisation requirement.
    """

    #: Indices of the source lines, in reading order.
    line_indices: tuple[int, ...]

    #: Word slice within the line, for sub-line spans. None for whole-line and
    #: multi-line spans.
    word_range: tuple[int, int] | None

    #: Exactly as recognised. Display and formatting checks only.
    raw_text: str

    #: Normalised for fuzzy scoring. Never for formatting checks.
    canonical_text: str

    #: Normalised without character folding, for numeric pattern matching.
    numeric_text: str

    bbox: BBox
    mean_confidence: float

    #: Median glyph height across constituent words — a proxy for type size.
    glyph_height: float

    #: Free-form notes for the diagnostic record.
    notes: dict[str, float] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.canonical_text.split()) if self.canonical_text else 0

    @property
    def is_multiline(self) -> bool:
        return len(self.line_indices) > 1


def _order_lines(lines: tuple[Line, ...]) -> list[Line]:
    """Sort lines into reading order: top-to-bottom, left-to-right within a row.

    OCR engines return lines in their own detection order, which is usually but
    not reliably reading order — a two-column layout or a rotated block can
    scramble it. Multi-line span generation assumes consecutive indices are
    vertically adjacent, so the ordering has to be imposed here rather than
    trusted.

    Lines whose vertical centres fall within ``ROW_GROUPING_RATIO`` of the
    median line height are considered one visual row and ordered by their left
    edge. This keeps ``750 mL`` and ``45% Alc./Vol.`` in the order a reader
    would see them even when the engine emitted them the other way round.
    """
    if not lines:
        return []

    heights = [ln.glyph_height for ln in lines if ln.glyph_height > 0]
    median_height = statistics.median(heights) if heights else 1.0
    row_tolerance = median_height * ROW_GROUPING_RATIO

    by_vertical = sorted(lines, key=lambda ln: ln.bbox.center_y)

    ordered: list[Line] = []
    row: list[Line] = []
    row_anchor = by_vertical[0].bbox.center_y

    for line in by_vertical:
        if abs(line.bbox.center_y - row_anchor) <= row_tolerance:
            row.append(line)
        else:
            ordered.extend(sorted(row, key=lambda ln: ln.bbox.left))
            row = [line]
            row_anchor = line.bbox.center_y

    ordered.extend(sorted(row, key=lambda ln: ln.bbox.left))
    return ordered


def _lines_are_coherent(upper: Line, lower: Line, median_height: float) -> bool:
    """Whether two consecutive lines plausibly belong to the same field.

    All three tests must pass. Each rules out a distinct false grouping:

    *   **Vertical gap** rules out unrelated blocks that happen to be adjacent.
    *   **Horizontal overlap** rules out side-by-side columns.
    *   **Glyph height ratio** rules out a display-size brand name absorbing the
        fine print beneath it — the most valuable of the three, because that
        particular false grouping otherwise produces a long span that scores
        respectably against the brand name and wins with the wrong region.
    """
    reference = median_height if median_height > 0 else 1.0

    if upper.bbox.vertical_gap(lower.bbox) > reference * MAX_VERTICAL_GAP_RATIO:
        return False

    if upper.bbox.horizontal_overlap_ratio(lower.bbox) < MIN_HORIZONTAL_OVERLAP:
        return False

    upper_h, lower_h = upper.glyph_height, lower.glyph_height
    if upper_h <= 0 or lower_h <= 0:
        return False

    ratio = upper_h / lower_h
    return MIN_GLYPH_HEIGHT_RATIO <= ratio <= MAX_GLYPH_HEIGHT_RATIO


def _make_span(
    source_lines: list[Line],
    line_indices: tuple[int, ...],
    word_range: tuple[int, int] | None,
    words: list,
) -> Span | None:
    """Build a Span from constituent words, or None if there is nothing to build."""
    if not words:
        return None

    raw_text = " ".join(w.text for w in words).strip()
    if not raw_text:
        return None

    canonical = canonicalize(raw_text)
    if not canonical:
        # Punctuation-only spans ("•", "---") survive raw but canonicalise to
        # nothing. They can never match a field and only add scoring noise.
        return None

    return Span(
        line_indices=line_indices,
        word_range=word_range,
        raw_text=raw_text,
        canonical_text=canonical,
        numeric_text=canonicalize_numeric(raw_text),
        bbox=BBox.union([w.bbox for w in words]),
        mean_confidence=statistics.fmean(w.confidence for w in words),
        glyph_height=statistics.median([w.bbox.height for w in words]),
        notes={"line_count": float(len(line_indices)), "word_count": float(len(words))},
    )


def build_span_index(
    lines: tuple[Line, ...],
    *,
    max_lines: int = MAX_LINES_PER_SPAN,
    max_words: int = MAX_WORDS_PER_SPAN,
) -> list[Span]:
    """Generate every plausible match candidate from OCR output.

    Produces three families of span, all in one pool:

    1. **Whole lines** — the common case, one field per line.
    2. **Sub-line runs** of up to ``max_words`` consecutive words, which recover
       ``750 mL`` from a line that also carries the ABV.
    3. **Multi-line runs** of up to ``max_lines`` consecutive lines that pass
       the coherence test, which recover a wrapped brand name.

    Duplicates are collapsed: a single-word line generates the same span as a
    one-word sub-line run, and a one-line multi-line run is just the line. The
    de-duplication key is the span's source extent rather than its text,
    because two different regions with identical text (a brand name appearing
    both in display type and in the bottler statement) are genuinely different
    candidates and the scorer needs to see both to pick the right region.

    Args:
        lines: OCR lines, in any order. Reading order is imposed internally.
        max_lines: Longest run of consecutive lines to consider as one span.
        max_words: Longest run of consecutive words within a line.

    Returns:
        Candidate spans, unordered. Field strategies score the whole pool.

    Examples:
        A label whose brand name wraps produces, among others, a span covering
        both lines::

            spans = build_span_index(lines)
            assert any(s.canonical_text == "OLD TOM DISTILLERY" for s in spans)
    """
    ordered = _order_lines(lines)
    if not ordered:
        return []

    heights = [ln.glyph_height for ln in ordered if ln.glyph_height > 0]
    median_height = statistics.median(heights) if heights else 1.0

    spans: list[Span] = []
    seen: set[tuple] = set()

    def add(span: Span | None) -> None:
        if span is None:
            return
        key = (span.line_indices, span.word_range)
        if key in seen:
            return
        seen.add(key)
        spans.append(span)

    # --- families 1 and 2: within a single line ---------------------------
    for position, line in enumerate(ordered):
        words = list(line.words)
        if not words:
            continue

        add(_make_span(ordered, (position,), None, words))

        if len(words) > 1:
            for start in range(len(words)):
                for end in range(start + 1, min(start + max_words, len(words)) + 1):
                    if end - start == len(words):
                        continue  # identical to the whole-line span above
                    add(_make_span(ordered, (position,), (start, end), words[start:end]))

    # --- family 3: runs of consecutive lines ------------------------------
    for start in range(len(ordered)):
        run = [ordered[start]]
        for offset in range(1, max_lines):
            nxt = start + offset
            if nxt >= len(ordered):
                break
            if not _lines_are_coherent(run[-1], ordered[nxt], median_height):
                break
            run.append(ordered[nxt])

            add(
                _make_span(
                    ordered,
                    tuple(range(start, nxt + 1)),
                    None,
                    [w for ln in run for w in ln.words],
                )
            )

    return spans


def mask_spans(spans: list[Span], mask: BBox | None, *, min_fraction: float = 0.5) -> list[Span]:
    """Remove spans lying mostly inside an already-resolved region.

    This is the masking step from docs/field-assignment.md stage 3. Fields are
    resolved in descending order of precision and each one's region is removed
    from the pool before the next runs.

    It prevents a concrete failure: the warning statement contains the words
    "alcoholic beverages", which score respectably against a class/type value
    containing "alcohol", and it contains numerals that a loose net-contents
    pattern will latch onto. Resolving and removing the warning first
    eliminates most cross-field contention in a single step.

    Args:
        spans: Current candidate pool.
        mask: Region to exclude. None returns the pool unchanged, which is the
            normal case for the first field resolved.
        min_fraction: Fraction of a span's own area that must lie inside the
            mask for it to be removed. Note the asymmetry — see
            `BBox.overlaps`.

    Returns:
        A new list; the input is not mutated, so the full pool stays available
        for the diagnostic record.
    """
    if mask is None:
        return list(spans)
    return [s for s in spans if not s.bbox.overlaps(mask, min_fraction=min_fraction)]
