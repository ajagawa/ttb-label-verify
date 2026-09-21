"""Tests for the geometry primitives.

`overlaps` gets the most attention here because it is asymmetric on purpose and
that asymmetry is load-bearing for field masking. A symmetric IoU test would
silently fail to exclude small spans sitting inside a large resolved region,
which is exactly the cross-field contention masking exists to prevent.
"""

from __future__ import annotations

import pytest

from extraction.geometry import BBox


def box(left: float, top: float, right: float, bottom: float) -> BBox:
    return BBox(left=left, top=top, right=right, bottom=bottom)


class TestConstruction:
    def test_derived_geometry(self) -> None:
        b = box(10, 20, 110, 70)
        assert b.width == 100
        assert b.height == 50
        assert b.area == 5000
        assert b.center_x == 60
        assert b.center_y == 45

    @pytest.mark.parametrize(
        ("left", "top", "right", "bottom"),
        [(100, 0, 10, 50), (0, 100, 50, 10)],
    )
    def test_inverted_box_is_rejected(self, left, top, right, bottom) -> None:
        """Fail at construction rather than producing a negative area.

        A negative-area box propagates silently into overlap fractions and
        produces nonsense masking decisions far from the actual bug.
        """
        with pytest.raises(ValueError, match="inverted"):
            box(left, top, right, bottom)

    def test_degenerate_box_is_allowed(self) -> None:
        """Zero-width boxes occur legitimately for empty recognitions."""
        assert box(10, 10, 10, 10).area == 0


class TestUnion:
    def test_union_spans_all_inputs(self) -> None:
        u = BBox.union([box(10, 10, 50, 30), box(60, 40, 100, 80)])
        assert (u.left, u.top, u.right, u.bottom) == (10, 10, 100, 80)

    def test_union_of_one_is_itself(self) -> None:
        b = box(5, 5, 15, 15)
        assert BBox.union([b]) == b

    def test_union_of_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            BBox.union([])


class TestOverlapsIsAsymmetric:
    """The masking test: "is this candidate inside the mask", not "do these touch"."""

    def test_small_box_fully_inside_large_one_overlaps(self) -> None:
        small, large = box(40, 40, 60, 60), box(0, 0, 200, 200)
        assert small.overlaps(large) is True

    def test_large_box_is_not_inside_the_small_one(self) -> None:
        """The reverse must be False — this is the asymmetry.

        A symmetric measure (IoU) would report these two boxes as barely
        related in both directions, and the small span would survive masking.
        """
        small, large = box(40, 40, 60, 60), box(0, 0, 200, 200)
        assert large.overlaps(small) is False

    def test_half_covered_box_meets_the_default_threshold(self) -> None:
        candidate = box(0, 0, 100, 100)
        mask = box(50, 0, 200, 100)  # covers exactly the right half
        assert candidate.overlaps(mask, min_fraction=0.5) is True
        assert candidate.overlaps(mask, min_fraction=0.51) is False

    def test_disjoint_boxes_do_not_overlap(self) -> None:
        assert box(0, 0, 10, 10).overlaps(box(100, 100, 110, 110)) is False

    def test_zero_area_box_never_overlaps(self) -> None:
        """Guards a division by zero in the fraction calculation."""
        assert box(10, 10, 10, 10).overlaps(box(0, 0, 100, 100)) is False


class TestSpanCoherenceHelpers:
    """Used by the multi-line span test to tell stacked lines from columns."""

    def test_stacked_lines_overlap_horizontally(self) -> None:
        """ "OLD TOM" above "DISTILLERY" — a wrapped brand name."""
        upper, lower = box(100, 50, 300, 90), box(90, 95, 320, 135)
        assert upper.horizontal_overlap_ratio(lower) > 0.9

    def test_side_by_side_text_does_not(self) -> None:
        """Two columns: adjacent vertically but not stacked."""
        left_col, right_col = box(0, 50, 100, 90), box(200, 50, 300, 90)
        assert left_col.horizontal_overlap_ratio(right_col) == 0.0

    def test_vertical_gap_between_separated_boxes(self) -> None:
        assert box(0, 0, 100, 40).vertical_gap(box(0, 60, 100, 100)) == 20

    def test_vertical_gap_is_zero_when_boxes_overlap(self) -> None:
        assert box(0, 0, 100, 50).vertical_gap(box(0, 30, 100, 80)) == 0

    def test_vertical_gap_is_order_independent(self) -> None:
        a, b = box(0, 0, 100, 40), box(0, 60, 100, 100)
        assert a.vertical_gap(b) == b.vertical_gap(a)


class TestSerialisation:
    def test_wire_format_is_css_friendly(self) -> None:
        """left/top/width/height, because that is what the overlay consumes."""
        assert box(10, 20, 110, 70).to_dict() == {
            "left": 10,
            "top": 20,
            "width": 100,
            "height": 50,
        }


class TestExpanded:
    def test_expansion_grows_every_side(self) -> None:
        e = box(50, 50, 100, 100).expanded(10)
        assert (e.left, e.top, e.right, e.bottom) == (40, 40, 110, 110)
