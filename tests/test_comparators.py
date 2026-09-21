"""Tests for the shared comparison helpers and the anchored (brand name) strategy.

The residual-difference tests exist because of a false negative the corpus
could not see: similarity ratios forgive a wrong letter more as the name gets
longer, so "OLD TON DISTILLERY" matched "OLD TOM DISTILLERY" while "OLD TON"
was correctly flagged against "OLD TOM". The fix counts surviving character
differences instead, and these tests pin both halves of it — real differences
are counted, and the variation normalisation exists to absorb is not.
"""

from __future__ import annotations

import pytest

from extraction.providers.stub import StubProvider
from rules.comparators import explain_residual, residual_difference, text_similarity
from rules.engine import verify
from rules.results import ApplicationRecord, Strategy, Verdict
from rules.schema import load_ruleset
from tests.conftest import make_lines


def record(brand: str) -> ApplicationRecord:
    return ApplicationRecord(
        brand_name=brand,
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol.",
        net_contents="750 mL",
    )


def brand_result(label_brand: str, application_brand: str, ruleset=None):
    lines = make_lines([(label_brand, {"height": 48.0})])
    extraction = StubProvider(lines=lines).extract(None)
    return verify(extraction, record(application_brand), ruleset=ruleset).field("brand_name")


class TestResidualDifferenceCounts:
    """What survives normalisation is counted; what normalisation removes is not."""

    @pytest.mark.parametrize(
        ("found", "expected", "count"),
        [
            ("OLD TON DISTILLERY", "OLD TOM DISTILLERY", 1),  # substitution
            ("OLD TOM DISTILERY", "OLD TOM DISTILLERY", 1),  # deletion
            ("OLD TOMM DISTILLERY", "OLD TOM DISTILLERY", 1),  # insertion
        ],
    )
    def test_real_differences_are_counted(self, found: str, expected: str, count: int) -> None:
        assert residual_difference(found, expected).count == count

    def test_unrelated_names_differ_by_many_characters(self) -> None:
        """The exact count for unrelated names is incidental; that it is large is not."""
        assert residual_difference("RIVERBEND", "OLD TOM").count >= 5

    @pytest.mark.parametrize(
        ("found", "expected", "reason"),
        [
            ("STONE'S THROW", "Stone's Throw", "case — Dave Morrison's example"),
            ("OLDTOMDISTILLERY", "OLD TOM DISTILLERY", "PP-OCR runs letterspaced type together"),
            ("OLD T0M DISTILLERY", "OLD TOM DISTILLERY", "zero read for O"),
            ("OID TOM DISTILLERY", "OLD TOM DISTILLERY", "I read for L"),
            ("CRÈME", "CREME", "accent"),
            ("STONES THROW", "Stone's Throw", "apostrophe"),
        ],
    )
    def test_meaningless_variation_is_not_counted(
        self, found: str, expected: str, reason: str
    ) -> None:
        """If these counted, the fix would trade a false negative for a flood of
        false flags — every OCR confusion the fold table exists for would reach
        an agent."""
        assert residual_difference(found, expected).count == 0, reason

    def test_count_does_not_shrink_with_length(self) -> None:
        """The property the fix exists for, stated directly.

        The same one-letter defect gets the same count whatever the length,
        where a similarity ratio gets steadily more forgiving.
        """
        short = residual_difference("OLD TON", "OLD TOM")
        long = residual_difference(
            "OLD TON DISTILLERY COMPANY LIMITED", "OLD TOM DISTILLERY COMPANY LIMITED"
        )
        assert short.count == long.count == 1

        # The ratio, for contrast, drifts upward past the match threshold.
        assert (
            text_similarity(
                "OID TOM DISTIIIERY COMPANY IIMITED", "OID TON DISTIIIERY COMPANY IIMITED"
            )
            > 0.92
        )
        assert text_similarity("OID TOM", "OID TON") < 0.92


class TestResidualDescriptions:
    """What the agent is told must match what the verdict counted."""

    def test_substitution_names_both_letters(self) -> None:
        described = residual_difference("OLD TON DISTILLERY", "OLD TOM DISTILLERY").described
        assert described == ("“M” in the application reads “N” on the label",)

    def test_deletion_names_the_missing_letter(self) -> None:
        described = residual_difference("OLD TOM DISTILERY", "OLD TOM DISTILLERY").described
        assert described == ("“L” is missing from the label",)

    def test_insertion_names_the_extra_letter(self) -> None:
        described = residual_difference("OLD TOMM DISTILLERY", "OLD TOM DISTILLERY").described
        assert described == ("the label has an extra “M”",)

    def test_folded_pairs_are_never_described(self) -> None:
        """A 0/O pair is not counted, so it must not be explained either.

        Otherwise a reason could cite a difference the verdict ignored, and an
        agent would be told the wrong thing about why a field was raised.
        """
        diff = residual_difference("OLD T0N DISTILLERY", "OLD TOM DISTILLERY")
        assert diff.count == 1
        assert all("0" not in item for item in diff.described)

    def test_descriptions_are_capped(self) -> None:
        assert len(residual_difference("ABCDEFGH", "ZYXWVUTS").described) == 3

    def test_explanation_says_it_may_be_a_misread(self) -> None:
        """The tool cannot tell a misread from a misprint, and must not imply it can."""
        text = explain_residual(
            "OLD TON DISTILLERY",
            "OLD TOM DISTILLERY",
            residual_difference("OLD TON DISTILLERY", "OLD TOM DISTILLERY"),
        )
        assert "1 character" in text
        assert "misread" in text
        assert "confirm" in text


class TestBrandNameVerdicts:
    """End to end through the engine."""

    def test_long_name_one_letter_off_is_reviewed_not_matched(self) -> None:
        """The false negative this change fixes."""
        result = brand_result("OLD TON DISTILLERY", "OLD TOM DISTILLERY")
        assert result.verdict is Verdict.REVIEW
        assert "“M” in the application reads “N”" in result.reason

    def test_label_misspelling_is_reviewed(self) -> None:
        """A genuinely misspelled label must reach a person."""
        assert brand_result("OLD TOM DISTILERY", "OLD TOM DISTILLERY").verdict is Verdict.REVIEW

    def test_residual_difference_is_never_asserted(self) -> None:
        """REVIEW, not MISMATCH: a misread and a misprint look identical here."""
        result = brand_result("OLD TON DISTILLERY", "OLD TOM DISTILLERY")
        assert result.verdict is not Verdict.MISMATCH
        assert result.strategy is Strategy.ANCHORED
        assert result.bbox is not None  # still located, still highlighted

    @pytest.mark.parametrize(
        ("label", "application"),
        [
            ("OLD TOM DISTILLERY", "OLD TOM DISTILLERY"),
            ("STONE'S THROW", "Stone's Throw"),
            ("OLDTOMDISTILLERY", "OLD TOM DISTILLERY"),
            ("OLD T0M DISTILLERY", "OLD TOM DISTILLERY"),
        ],
    )
    def test_equivalent_readings_still_match(self, label: str, application: str) -> None:
        assert brand_result(label, application).verdict is Verdict.MATCH

    def test_budget_is_read_from_the_rule_set(self) -> None:
        """Move the versioned value and the verdict must follow.

        Guards against the threshold being reintroduced as a code constant,
        which would change verdicts without changing the rule-set version
        stamped on them.
        """
        base = load_ruleset()
        lenient = base.model_copy(
            update={
                "tuning": base.tuning.model_copy(
                    update={
                        "names": base.tuning.names.model_copy(
                            update={"max_residual_character_edits": 1}
                        )
                    }
                )
            }
        )
        assert brand_result("OLD TON DISTILLERY", "OLD TOM DISTILLERY").verdict is Verdict.REVIEW
        assert (
            brand_result("OLD TON DISTILLERY", "OLD TOM DISTILLERY", ruleset=lenient).verdict
            is Verdict.MATCH
        )
