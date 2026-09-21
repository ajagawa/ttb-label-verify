"""Shared test fixtures and factories.

Building `Line` objects by hand is verbose enough that every test module would
otherwise grow its own slightly different helper, and small divergences in how
test geometry is constructed make failures hard to compare across modules.
These factories lay text out on a synthetic label with plausible geometry, so
tests can describe a label in terms of what it says and where, rather than in
pixel arithmetic.
"""

from __future__ import annotations

import pytest

from extraction.geometry import BBox
from extraction.providers.base import ExtractionResult, Line, QualityAssessment, Word

#: Nominal glyph height for body text in the synthetic layouts below. Chosen so
#: that display type (2-3x) and fine print (0.5x) are both expressible in round
#: numbers.
BODY_HEIGHT = 20.0

#: Approximate advance width per character at BODY_HEIGHT, used to give words
#: plausible widths. Real proportional type varies; the span logic only cares
#: about relative sizes and overlaps, so a constant is sufficient and keeps the
#: expected geometry in tests readable.
CHAR_WIDTH_RATIO = 0.55


def make_line(
    text: str,
    *,
    index: int = 0,
    left: float = 100.0,
    top: float = 100.0,
    height: float = BODY_HEIGHT,
    confidence: float = 0.95,
    word_gap: float = 8.0,
) -> Line:
    """Lay a line of text out left-to-right at a given position and type size.

    Args:
        text: The line's words, space-separated.
        index: Line index, as an OCR engine would assign.
        left: X coordinate of the first word's left edge.
        top: Y coordinate of the line's top edge.
        height: Glyph height — the knob for display type vs fine print.
        confidence: Applied uniformly to every word in the line.
        word_gap: Horizontal whitespace between words.

    Returns:
        A `Line` whose word boxes tile left to right without overlapping.
    """
    words: list[Word] = []
    cursor = left
    char_width = height * CHAR_WIDTH_RATIO

    for token in text.split():
        width = max(char_width, len(token) * char_width)
        words.append(
            Word(
                text=token,
                bbox=BBox(left=cursor, top=top, right=cursor + width, bottom=top + height),
                confidence=confidence,
            )
        )
        cursor += width + word_gap

    return Line(index=index, words=tuple(words), bbox=BBox.union([w.bbox for w in words]))


def make_lines(
    specs: list[str | tuple[str, dict]],
    *,
    left: float = 100.0,
    start_top: float = 100.0,
    leading: float = 10.0,
) -> tuple[Line, ...]:
    """Stack several lines vertically, one below the next.

    Each spec is either a string (body text at the default size) or a
    ``(text, overrides)`` pair whose overrides are passed to `make_line` — most
    usefully ``height`` for display type, or ``left`` and ``top`` to place a
    line somewhere other than the flowing position.

    Args:
        specs: Line specifications in top-to-bottom order.
        left: Default left edge for all lines.
        start_top: Y coordinate of the first line.
        leading: Vertical whitespace between consecutive lines.

    Returns:
        Lines with sequential indices and non-overlapping vertical positions.
    """
    lines: list[Line] = []
    cursor = start_top

    for index, spec in enumerate(specs):
        text, overrides = (spec, {}) if isinstance(spec, str) else spec
        height = overrides.get("height", BODY_HEIGHT)
        line = make_line(
            text,
            index=index,
            left=overrides.get("left", left),
            top=overrides.get("top", cursor),
            height=height,
            confidence=overrides.get("confidence", 0.95),
        )
        lines.append(line)
        cursor = line.bbox.bottom + overrides.get("leading", leading)

    return tuple(lines)


def make_extraction(
    lines: tuple[Line, ...],
    *,
    width: int = 1000,
    height: int = 1400,
    provider: str = "test",
    usable: bool = True,
    quality_score: float = 0.9,
) -> ExtractionResult:
    """Wrap lines in an `ExtractionResult` with a passing quality assessment."""
    return ExtractionResult(
        lines=lines,
        image_width=width,
        image_height=height,
        quality=QualityAssessment(usable=usable, score=quality_score),
        provider_name=provider,
        extraction_ms=0.0,
    )


#: The statutory warning text from 27 CFR 16.21, as it must appear on a label.
#: Defined here as well as in the rule set so that tests can construct a
#: compliant label without importing the rules package — the extraction-side
#: tests must stay runnable independently of the verification side.
STATUTORY_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)


@pytest.fixture
def sample_label_lines() -> tuple[Line, ...]:
    """A compliant distilled-spirits label matching the brief's example.

    Deliberately includes the two layout awkwardnesses the span index exists to
    handle: the brand name wraps onto a second line in display type, and the
    net contents shares a line with the alcohol content.
    """
    return make_lines(
        [
            ("OLD TOM", {"height": 48.0}),
            ("DISTILLERY", {"height": 48.0}),
            ("Kentucky Straight Bourbon Whiskey", {"height": 22.0}),
            ("750 mL 45% Alc./Vol. (90 Proof)", {"height": 18.0, "leading": 40.0}),
            ("GOVERNMENT WARNING: (1) According to the Surgeon General, women", {"height": 12.0}),
            ("should not drink alcoholic beverages during pregnancy because of", {"height": 12.0}),
            ("the risk of birth defects. (2) Consumption of alcoholic beverages", {"height": 12.0}),
            ("impairs your ability to drive a car or operate machinery, and may", {"height": 12.0}),
            ("cause health problems.", {"height": 12.0}),
        ]
    )
