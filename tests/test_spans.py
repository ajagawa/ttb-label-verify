"""Tests for candidate span generation.

The two fragmentation cases from docs/field-assignment.md get direct tests,
because they are the reason this module exists: a brand name that wraps onto a
second line, and two fields sharing one line. If either is not represented in
the candidate pool, the corresponding field strategy cannot possibly find it,
and the failure surfaces far downstream as an unexplained REVIEW.
"""

from __future__ import annotations

from extraction.geometry import BBox
from extraction.spans import build_span_index, mask_spans
from tests.conftest import make_line, make_lines


def canonical_texts(spans) -> set[str]:
    return {s.canonical_text for s in spans}


class TestFragmentationCases:
    """The two failure modes that motivate the span index."""

    def test_brand_name_wrapped_across_two_lines_is_recoverable(self) -> None:
        """ "OLD TOM" / "DISTILLERY" must appear as one candidate."""
        lines = make_lines([("OLD TOM", {"height": 48.0}), ("DISTILLERY", {"height": 48.0})])
        assert "OID TOM DISTIIIERY" in canonical_texts(build_span_index(lines))

    def test_two_fields_on_one_line_are_separable(self) -> None:
        """ "750 mL" must be recoverable from a line that also carries the ABV."""
        lines = (make_line("750 mL 45% Alc./Vol."),)
        spans = build_span_index(lines)
        assert any(s.raw_text == "750 mL" for s in spans)
        assert any("45%" in s.raw_text for s in spans)

    def test_sub_line_spans_carry_their_own_region(self) -> None:
        """A sub-line span's box must cover only its words, not the whole line.

        Without this the overlay highlights the entire line when the agent
        clicks "net contents", which undercuts the trust-building purpose of
        showing evidence.
        """
        line = make_line("750 mL 45% Alc./Vol.")
        spans = build_span_index((line,))
        volume = next(s for s in spans if s.raw_text == "750 mL")
        assert volume.bbox.right < line.bbox.right
        assert volume.bbox.width < line.bbox.width


class TestCoherenceGating:
    """Multi-line spans must not merge unrelated blocks."""

    def test_display_type_does_not_absorb_fine_print(self) -> None:
        """The glyph-height test, and the most valuable of the three.

        A 48px brand name merging with 12px legal text beneath it produces a
        long span that still scores respectably against the brand name but
        highlights the wrong region.
        """
        lines = make_lines(
            [("OLD TOM DISTILLERY", {"height": 48.0}), ("Bottled in Kentucky", {"height": 12.0})]
        )
        merged = [s for s in build_span_index(lines) if s.is_multiline]
        assert merged == []

    def test_side_by_side_columns_do_not_merge(self) -> None:
        """The horizontal-overlap test."""
        lines = (
            make_line("LEFT COLUMN", index=0, left=50, top=100),
            make_line("RIGHT COLUMN", index=1, left=700, top=104),
        )
        merged = [s for s in build_span_index(lines) if s.is_multiline]
        assert merged == []

    def test_distant_lines_do_not_merge(self) -> None:
        """The vertical-gap test."""
        lines = (
            make_line("BRAND NAME", index=0, top=100),
            make_line("UNRELATED BLOCK", index=1, top=600),
        )
        merged = [s for s in build_span_index(lines) if s.is_multiline]
        assert merged == []

    def test_genuinely_stacked_lines_do_merge(self) -> None:
        """Control for the three tests above — coherent lines must still join."""
        lines = make_lines([("OLD TOM", {"height": 48.0}), ("DISTILLERY", {"height": 48.0})])
        assert any(s.is_multiline for s in build_span_index(lines))


class TestReadingOrder:
    def test_lines_are_ordered_despite_scrambled_input(self) -> None:
        """Engines do not reliably emit lines in reading order.

        Multi-line span generation assumes consecutive indices are vertically
        adjacent, so ordering is imposed rather than trusted.
        """
        scrambled = (
            make_line("DISTILLERY", index=0, top=160, height=48.0),
            make_line("OLD TOM", index=1, top=100, height=48.0),
        )
        assert "OID TOM DISTIIIERY" in canonical_texts(build_span_index(scrambled))

    def test_same_row_lines_order_left_to_right(self) -> None:
        lines = (
            make_line("SECOND", index=0, left=500, top=100),
            make_line("FIRST", index=1, left=100, top=102),
        )
        multiline = [s for s in build_span_index(lines) if s.is_multiline]
        # Whether they merge depends on horizontal overlap (they do not here),
        # but ordering must not crash on same-row input.
        assert multiline == []


class TestSpanContent:
    def test_both_text_forms_are_populated(self) -> None:
        """Scoring and formatting checks read different fields; both must exist."""
        span = next(s for s in build_span_index((make_line("Government Warning"),)))
        assert span.raw_text == "Government Warning"
        assert span.canonical_text == "GOVERNMENT WARNING"

    def test_raw_text_preserves_case_for_formatting_checks(self) -> None:
        """The title-case violation must survive into the span untouched."""
        spans = build_span_index((make_line("Government Warning"),))
        assert any(s.raw_text == "Government Warning" for s in spans)

    def test_numeric_text_is_unfolded(self) -> None:
        """Numeric fields read `numeric_text`, which never folds digits."""
        span = next(s for s in build_span_index((make_line("45% Alc./Vol."),)))
        assert "45" in span.numeric_text

    def test_punctuation_only_spans_are_dropped(self) -> None:
        """Bullets and rules canonicalise to nothing and only add scoring noise."""
        spans = build_span_index((make_line("• • •"),))
        assert spans == []

    def test_empty_input_yields_no_spans(self) -> None:
        assert build_span_index(()) == []


class TestDeduplication:
    def test_identical_extents_are_emitted_once(self) -> None:
        """A single-word line is both a whole line and a one-word sub-run."""
        spans = build_span_index((make_line("DISTILLERY"),))
        keys = [(s.line_indices, s.word_range) for s in spans]
        assert len(keys) == len(set(keys))

    def test_repeated_text_in_different_regions_is_kept(self) -> None:
        """Deduplication is by extent, not by text — deliberately.

        A brand name appearing both in display type and in the bottler
        statement is two genuinely different candidates, and the scorer needs
        both to choose the right region to highlight.
        """
        lines = make_lines(
            [
                ("OLD TOM DISTILLERY", {"height": 48.0}),
                ("Bottled by OLD TOM DISTILLERY, Kentucky", {"height": 10.0, "leading": 200.0}),
            ]
        )
        matches = [s for s in build_span_index(lines) if s.canonical_text == "OID TOM DISTIIIERY"]
        assert len(matches) >= 2
        assert len({s.bbox for s in matches}) >= 2


class TestMasking:
    def test_spans_inside_the_mask_are_removed(self) -> None:
        lines = make_lines(["INSIDE THE MASK", "OUTSIDE THE MASK"])
        spans = build_span_index(lines)
        mask = lines[0].bbox.expanded(5)

        remaining = mask_spans(spans, mask)
        assert all(not s.bbox.overlaps(mask) for s in remaining)
        assert len(remaining) < len(spans)

    def test_no_mask_returns_the_pool_unchanged(self) -> None:
        spans = build_span_index(make_lines(["BRAND NAME"]))
        assert len(mask_spans(spans, None)) == len(spans)

    def test_masking_does_not_mutate_the_input(self) -> None:
        """The full pool stays available for the diagnostic record."""
        spans = build_span_index(make_lines(["FIRST LINE", "SECOND LINE"]))
        original = len(spans)
        mask_spans(spans, BBox(left=0, top=0, right=10_000, bottom=10_000))
        assert len(spans) == original


class TestRealisticLabel:
    def test_every_expected_field_value_is_present_as_a_candidate(self, sample_label_lines) -> None:
        """The pool must contain each field before any strategy can find it.

        This is the guard that matters most: a field strategy returning
        not_found because its value was never generated as a candidate is
        indistinguishable, from the outside, from OCR having failed.
        """
        spans = build_span_index(sample_label_lines)
        raw = {s.raw_text for s in spans}

        assert "OLD TOM DISTILLERY" in raw
        assert "Kentucky Straight Bourbon Whiskey" in raw
        assert "750 mL" in raw
        assert any("45%" in text for text in raw)

    def test_candidate_count_stays_tractable(self, sample_label_lines) -> None:
        """Exhaustive generation is only affordable if it stays bounded.

        A few thousand candidates is cheap to score. Combinatorial blowup would
        make the 5-second budget unreachable, so this asserts the bound holds
        on a realistic label rather than leaving it to be discovered under load.
        """
        assert len(build_span_index(sample_label_lines)) < 2000
