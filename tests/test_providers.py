"""Tests for the extraction providers.

The stub's job is to be trustworthy scaffolding, so the tests concentrate on
the ways it could quietly stop being trustworthy: inventing content it was not
given, or masking a setup error as an empty result.

The RapidOCR adapter is tested only where it can be: its pure functions. The
engine itself cannot run here — the weight CDN is not reachable from this
environment — and that gap is recorded in the module docstring rather than
papered over with a mock that would assert only that the mock was called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extraction.geometry import BBox
from extraction.providers import available_providers, get_provider
from extraction.providers.base import ExtractionError
from extraction.providers.rapidocr_provider import _quad_to_bbox, _split_line_into_words
from extraction.providers.stub import StubProvider
from tests.conftest import make_lines


class TestStubRefusesToInvent:
    def test_construction_without_a_source_raises(self) -> None:
        """A stub that returns a plausible default hides wiring mistakes.

        If it invented a label, an end-to-end test could pass with the image
        pipeline entirely disconnected.
        """
        with pytest.raises(ValueError, match="will not invent a label"):
            StubProvider()

    def test_missing_sidecar_is_an_error_not_an_empty_result(self, tmp_path: Path) -> None:
        """A missing sidecar is a setup error, not a blank label.

        Returning an empty extraction would surface downstream as every field
        resolving to REVIEW — indistinguishable from a genuinely unreadable
        image, and pointing at the wrong half of the system.
        """
        with pytest.raises(ExtractionError, match="no ground-truth sidecar"):
            StubProvider.for_image(tmp_path / "absent.png")


class TestStubReplaysAccurately:
    def test_injected_lines_are_returned_verbatim(self) -> None:
        lines = make_lines(["OLD TOM DISTILLERY", "750 mL"])
        result = StubProvider(lines=lines).extract(None)

        assert [line.text for line in result.lines] == ["OLD TOM DISTILLERY", "750 mL"]
        assert result.provider_name == "stub"

    def test_sidecar_round_trips_geometry_and_confidence(self, tmp_path: Path) -> None:
        image = tmp_path / "label.png"
        image.write_bytes(b"")
        sidecar = tmp_path / "label.png.ocr.json"
        sidecar.write_text(
            json.dumps(
                {
                    "width": 800,
                    "height": 1200,
                    "lines": [
                        {
                            "words": [
                                {
                                    "text": "OLD",
                                    "left": 100,
                                    "top": 50,
                                    "width": 80,
                                    "height": 40,
                                    "confidence": 0.99,
                                },
                                {
                                    "text": "TOM",
                                    "left": 190,
                                    "top": 50,
                                    "width": 80,
                                    "height": 40,
                                    "confidence": 0.97,
                                },
                            ]
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = StubProvider.for_image(image).extract(None)

        assert result.image_width == 800
        assert result.image_height == 1200
        assert result.lines[0].text == "OLD TOM"
        assert result.lines[0].words[0].bbox == BBox(left=100, top=50, right=180, bottom=90)
        assert result.lines[0].confidence == pytest.approx(0.98)

    def test_malformed_sidecar_reports_the_path(self, tmp_path: Path) -> None:
        image = tmp_path / "label.png"
        image.write_bytes(b"")
        (tmp_path / "label.png.ocr.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(ExtractionError, match="could not read sidecar"):
            StubProvider.for_image(image).extract(None)

    def test_is_always_available(self) -> None:
        assert StubProvider(lines=make_lines(["X"])).is_available() is True


class TestProviderRegistry:
    def test_stub_is_not_selectable_globally(self) -> None:
        """It replays known text, so a global stub would silently fake results.

        The error is explicit rather than the stub being omitted from the
        registry, so that someone setting LABEL_VERIFY_PROVIDER=stub learns why
        instead of seeing "unknown provider".
        """
        with pytest.raises(ExtractionError, match="cannot be selected globally"):
            get_provider("stub")

    def test_unknown_provider_names_the_alternatives(self) -> None:
        with pytest.raises(ExtractionError, match="unknown extraction provider"):
            get_provider("azure-document-intelligence")

    def test_registry_lists_both_implementations(self) -> None:
        """Two implementations is what makes the seam a fact rather than a claim."""
        assert set(available_providers()) == {"rapidocr", "stub"}


class TestQuadConversion:
    def test_axis_aligned_quad_converts_exactly(self) -> None:
        quad = [[10, 20], [110, 20], [110, 60], [10, 60]]
        assert _quad_to_bbox(quad) == BBox(left=10, top=20, right=110, bottom=60)

    def test_rotated_quad_yields_its_bounding_rectangle(self) -> None:
        """Looser than the ink, and deliberately so.

        Every consumer wants an axis-aligned box. The cost is a slightly
        generous highlight, not a wrong verdict.
        """
        quad = [[50, 10], [110, 30], [100, 70], [40, 50]]
        assert _quad_to_bbox(quad) == BBox(left=40, top=10, right=110, bottom=70)

    @pytest.mark.parametrize("bad", [None, [], "not-a-quad", [[1]]])
    def test_malformed_quads_are_dropped_not_raised(self, bad) -> None:
        """One bad detection must not fail the whole extraction."""
        assert _quad_to_bbox(bad) is None


class TestWordInterpolation:
    """PP-OCR returns line-level boxes; sub-line spans need word-level geometry."""

    def test_single_word_takes_the_whole_line_box(self) -> None:
        line = BBox(left=0, top=0, right=100, bottom=20)
        words = _split_line_into_words("DISTILLERY", line, 0.95)
        assert len(words) == 1
        assert words[0].bbox == line

    def test_words_tile_left_to_right_without_overlapping(self) -> None:
        line = BBox(left=0, top=0, right=300, bottom=20)
        words = _split_line_into_words("750 mL 45%", line, 0.9)

        assert [w.text for w in words] == ["750", "mL", "45%"]
        for earlier, later in zip(words, words[1:], strict=False):
            assert earlier.bbox.right <= later.bbox.left

    def test_interpolated_words_stay_within_the_line_box(self) -> None:
        """Rounding must not push a box outside the region it came from."""
        line = BBox(left=10, top=0, right=210, bottom=20)
        words = _split_line_into_words("A BB CCC DDDD", line, 0.9)

        assert words[0].bbox.left >= line.left
        assert words[-1].bbox.right <= line.right

    def test_longer_words_get_wider_boxes(self) -> None:
        """Apportioning by character count is the approximation being asserted."""
        line = BBox(left=0, top=0, right=400, bottom=20)
        short, long = _split_line_into_words("AB CCCCCCCC", line, 0.9)
        assert long.bbox.width > short.bbox.width

    def test_empty_text_yields_no_words(self) -> None:
        assert _split_line_into_words("   ", BBox(left=0, top=0, right=10, bottom=10), 0.9) == ()
