"""Tests for the batch worklist ordering.

The ordering is the product of the batch feature: its value is that an agent
works down a list in which the labels needing attention come first. These tests
pin the tier order, the within-tier order, and stability.
"""

from __future__ import annotations

import pytest

from rules.results import BoxModel, FieldResult, ImageQuality, Strategy, Verdict, VerificationResult
from rules.triage import Tier, sort_key, summarise


def field(field_id="brand_name", verdict=Verdict.MATCH, score=0.99, **extra) -> FieldResult:
    located = verdict is not Verdict.REVIEW or extra.get("strategy") is not Strategy.NOT_FOUND
    defaults = dict(
        field_id=field_id,
        label=field_id.replace("_", " ").title(),
        verdict=verdict,
        strategy=Strategy.ANCHORED,
        expected="x",
        found="x",
        bbox=BoxModel(left=0, top=0, width=10, height=10) if located else None,
        score=score,
        ocr_confidence=0.95,
        reason=f"{field_id} {verdict.value.lower()}",
        citation="27 CFR 5.63",
    )
    defaults.update(extra)
    return FieldResult(**defaults)


def result(*fields: FieldResult) -> VerificationResult:
    return VerificationResult(
        ruleset_version="ttb-v1",
        provider_name="test",
        quality=ImageQuality(usable=True, score=0.95),
        fields=fields,
        image_width=1000,
        image_height=1400,
        elapsed_ms=1.0,
        extraction_ms=1.0,
    )


CLEAR = result(field("brand_name"), field("class_type"))
REVIEWED = result(field("brand_name", Verdict.REVIEW, 0.9), field("class_type"))
ASSERTED = result(field("brand_name", Verdict.MISMATCH, 0.4), field("class_type"))
MISSING_WARNING = result(
    field("brand_name"),
    field(
        "government_warning",
        Verdict.REVIEW,
        0.0,
        strategy=Strategy.NOT_FOUND,
        found=None,
        bbox=None,
        elevated=True,
    ),
)
UNCHECKED = result(
    field("brand_name"),
    field(
        "class_type",
        Verdict.REVIEW,
        0.0,
        strategy=Strategy.NOT_FOUND,
        found=None,
        bbox=None,
        automated=False,
    ),
)


class TestTiers:
    @pytest.mark.parametrize(
        ("outcome", "tier"),
        [
            (ASSERTED, Tier.ASSERTED),
            (MISSING_WARNING, Tier.MISSING_MANDATORY),
            (REVIEWED, Tier.REVIEW),
            (UNCHECKED, Tier.UNCHECKED),
            (CLEAR, Tier.CLEAR),
        ],
    )
    def test_each_outcome_lands_in_its_tier(self, outcome, tier) -> None:
        assert summarise(outcome).tier is tier

    def test_unreadable_image_is_its_own_tier(self) -> None:
        summary = summarise(None, error="could not decode the uploaded image")
        assert summary.tier is Tier.UNREADABLE
        assert summary.headline == "could not decode the uploaded image"

    def test_no_outcome_at_all_is_refused(self) -> None:
        """Ranking a label with no outcome as clear would be the worst guess."""
        with pytest.raises(ValueError, match="result or an error"):
            summarise(None)

    def test_unchecked_is_not_counted_as_uncertainty(self) -> None:
        """ "Not automated" and "the tool was unsure" are different messages."""
        summary = summarise(UNCHECKED)
        assert summary.unchecked_count == 1
        assert summary.review_count == 0

    def test_only_clear_is_left_out_of_needs_attention(self) -> None:
        assert not summarise(CLEAR).needs_attention
        for outcome in (ASSERTED, MISSING_WARNING, REVIEWED, UNCHECKED):
            assert summarise(outcome).needs_attention


class TestHeadline:
    def test_headline_comes_from_the_most_severe_field(self) -> None:
        both = result(
            field("brand_name", Verdict.REVIEW, 0.9, reason="brand is close"),
            field("class_type", Verdict.MISMATCH, 0.3, reason="Vodka, not Gin"),
        )
        assert summarise(both).headline == "Class Type: Vodka, not Gin"

    def test_clear_label_has_no_headline(self) -> None:
        assert summarise(CLEAR).headline is None


class TestOrdering:
    def ordered(
        self, entries: list[tuple[str, VerificationResult | None, str | None]]
    ) -> list[str]:
        return [
            name
            for name, _, _ in sorted(entries, key=lambda e: sort_key(summarise(e[1], e[2]), e[0]))
        ]

    def test_tiers_order_worst_first(self) -> None:
        entries = [
            ("clear.png", CLEAR, None),
            ("review.png", REVIEWED, None),
            ("unchecked.png", UNCHECKED, None),
            ("broken.png", None, "unreadable"),
            ("missing.png", MISSING_WARNING, None),
            ("asserted.png", ASSERTED, None),
        ]
        assert self.ordered(entries) == [
            "asserted.png",
            "missing.png",
            "broken.png",
            "review.png",
            "unchecked.png",
            "clear.png",
        ]

    def test_more_problems_rank_above_fewer_in_the_same_tier(self) -> None:
        two = result(
            field("brand_name", Verdict.MISMATCH, 0.5),
            field("class_type", Verdict.MISMATCH, 0.5),
        )
        one = result(field("brand_name", Verdict.MISMATCH, 0.2), field("class_type"))
        assert self.ordered([("one.png", one, None), ("two.png", two, None)]) == [
            "two.png",
            "one.png",
        ]

    def test_weaker_score_breaks_a_tie_in_problem_count(self) -> None:
        weak = result(field("brand_name", Verdict.REVIEW, 0.62), field("class_type"))
        strong = result(field("brand_name", Verdict.REVIEW, 0.91), field("class_type"))
        assert self.ordered([("strong.png", strong, None), ("weak.png", weak, None)]) == [
            "weak.png",
            "strong.png",
        ]

    def test_order_is_stable_for_identical_outcomes(self) -> None:
        """Without a filename tie-break, identical rows could swap on every
        fetch, and "work down the list" stops meaning anything."""
        entries = [(f"label_{i:03}.png", CLEAR, None) for i in (7, 2, 9, 1)]
        first = self.ordered(entries)
        assert first == self.ordered(list(reversed(entries)))
        assert first == ["label_001.png", "label_002.png", "label_007.png", "label_009.png"]
