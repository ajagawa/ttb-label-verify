"""Tests for the verification result contract.

The invariant tests are the important ones. "Not-found is never a mismatch" is
a safety property, not a style preference, and enforcing it in the model means
a future field strategy that violates it fails in its own test suite rather
than shipping a confident false accusation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rules.results import (
    ApplicationRecord,
    BoxModel,
    CheckOutcome,
    CheckResult,
    FieldResult,
    ImageQuality,
    Strategy,
    Verdict,
    VerificationResult,
)


def make_field(**overrides) -> FieldResult:
    defaults = dict(
        field_id="brand_name",
        label="Brand Name",
        verdict=Verdict.MATCH,
        strategy=Strategy.ANCHORED,
        expected="OLD TOM DISTILLERY",
        found="OLD TOM DISTILLERY",
        bbox=BoxModel(left=10, top=10, width=200, height=40),
        score=0.99,
        ocr_confidence=0.95,
        reason="Matches the application record.",
        citation="27 CFR 5.63",
    )
    return FieldResult(**{**defaults, **overrides})


class TestNotFoundIsNeverAMismatch:
    """The central safety invariant, enforced structurally."""

    def test_not_found_mismatch_is_rejected(self) -> None:
        """The failure this guard exists to prevent.

        A stylized brand name that OCR cannot read is an extraction failure.
        Reporting it as MISMATCH tells the agent the label is wrong, which is
        an assertion the tool has no basis for and which actively misleads
        someone who is trusting it.
        """
        with pytest.raises(ValidationError, match="cannot be a MISMATCH"):
            make_field(
                strategy=Strategy.NOT_FOUND,
                verdict=Verdict.MISMATCH,
                found=None,
                bbox=None,
            )

    def test_not_found_review_is_the_correct_construction(self) -> None:
        result = make_field(
            strategy=Strategy.NOT_FOUND,
            verdict=Verdict.REVIEW,
            found=None,
            bbox=None,
            score=0.0,
            reason="Could not locate the brand name on this image.",
        )
        assert result.verdict is Verdict.REVIEW

    def test_missing_mandatory_warning_is_elevated_review_not_mismatch(self) -> None:
        """The case that pulls hardest against the invariant, and still obeys it.

        A mandatory warning absent from a good-quality image is
        rejection-grade. It is still REVIEW — the agent looks, the tool does
        not assert — but `elevated` keeps it from being presented at the same
        weight as an ordinary low-confidence read.
        """
        result = make_field(
            field_id="government_warning",
            label="Government Warning Statement",
            strategy=Strategy.NOT_FOUND,
            verdict=Verdict.REVIEW,
            found=None,
            bbox=None,
            score=0.0,
            elevated=True,
            reason="No warning statement located; image quality is good. Verify manually.",
        )
        assert result.elevated is True
        assert result.verdict is Verdict.REVIEW

    def test_mismatch_requires_a_region(self) -> None:
        """An assertion the agent cannot visually verify is not actionable.

        Every MISMATCH is a claim about what is printed. The agent must be able
        to click it and see the pixels it was read from, or the claim cannot be
        checked.
        """
        with pytest.raises(ValidationError, match="must carry the region"):
            make_field(verdict=Verdict.MISMATCH, bbox=None, found="NEW TOM DISTILLERY")

    def test_located_mismatch_is_allowed(self) -> None:
        result = make_field(
            verdict=Verdict.MISMATCH,
            found="NEW TOM DISTILLERY",
            score=0.55,
            reason="Label reads NEW TOM DISTILLERY; application states OLD TOM DISTILLERY.",
        )
        assert result.verdict is Verdict.MISMATCH


class TestFieldResult:
    def test_found_text_is_raw_not_canonical(self) -> None:
        """The agent sees what is printed, including its casing.

        Showing canonical text would hide exactly the difference a
        capitalisation finding is about.
        """
        result = make_field(found="Old Tom Distillery")
        assert result.found == "Old Tom Distillery"

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_score_is_bounded(self, score: float) -> None:
        with pytest.raises(ValidationError):
            make_field(score=score)

    def test_results_are_immutable(self) -> None:
        """A verdict must not be mutable after construction.

        Reproducibility is the point of the deterministic half of the system;
        a result object that can be edited downstream undermines it.
        """
        result = make_field()
        with pytest.raises(ValidationError):
            result.verdict = Verdict.MISMATCH


class TestCheckResult:
    def test_advisory_outcome_carries_its_measurement(self) -> None:
        """Advisory checks report numbers, not conclusions."""
        check = CheckResult(
            id="header_bold",
            outcome=CheckOutcome.ADVISORY,
            citation="27 CFR 16.22(a)(2)",
            summary="Header stroke width is 1.6x the statement body.",
            measurement={"header_stroke_ratio": 1.6},
            confidence=0.72,
        )
        assert check.measurement["header_stroke_ratio"] == pytest.approx(1.6)

    def test_not_evaluable_is_distinct_from_pass(self) -> None:
        """ "We could not check this" must never read as "this passed"."""
        check = CheckResult(
            id="type_size",
            outcome=CheckOutcome.NOT_EVALUABLE,
            citation="27 CFR 16.22(b)",
            summary="Requires physical label dimensions; none supplied.",
        )
        assert check.outcome is not CheckOutcome.PASS


class TestApplicationRecord:
    def test_label_dimensions_are_optional(self) -> None:
        """Their absence is why the type-size check reports NOT_EVALUABLE."""
        record = ApplicationRecord(
            brand_name="OLD TOM DISTILLERY",
            class_type="Kentucky Straight Bourbon Whiskey",
            alcohol_content="45% Alc./Vol.",
            net_contents="750 mL",
        )
        assert record.label_width_mm is None

    def test_non_positive_dimensions_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApplicationRecord(
                brand_name="B",
                class_type="C",
                alcohol_content="45%",
                net_contents="750 mL",
                label_width_mm=0,
            )


class TestVerificationResult:
    def _result(self, *fields) -> VerificationResult:
        return VerificationResult(
            ruleset_version="ttb-v1",
            provider_name="stub",
            quality=ImageQuality(usable=True, score=0.9),
            fields=fields,
            image_width=1000,
            image_height=1400,
            elapsed_ms=1200.0,
            extraction_ms=900.0,
        )

    def test_flagged_count_counts_everything_but_match(self) -> None:
        result = self._result(
            make_field(field_id="a", verdict=Verdict.MATCH),
            make_field(field_id="b", verdict=Verdict.REVIEW, score=0.8),
            make_field(field_id="c", verdict=Verdict.MISMATCH, score=0.3),
        )
        assert result.flagged_count == 2

    def test_lowest_score_drives_batch_triage(self) -> None:
        """Batch returns a worklist ordered worst-first.

        Batch is throughput-bound, not latency-bound, so its value is putting
        agent attention where it is needed rather than in arrival order.
        """
        result = self._result(
            make_field(field_id="a", score=0.99),
            make_field(field_id="b", score=0.42, verdict=Verdict.REVIEW),
        )
        assert result.lowest_score == pytest.approx(0.42)

    def test_ruleset_version_is_recorded(self) -> None:
        """A verdict must be reproducible against the rules in force when made."""
        assert self._result(make_field()).ruleset_version == "ttb-v1"

    def test_diagnostics_are_absent_by_default(self) -> None:
        """Verbose and gated — no place in a normal response."""
        assert self._result(make_field()).diagnostics is None

    def test_unknown_field_lookup_raises(self) -> None:
        with pytest.raises(KeyError):
            self._result(make_field()).field("nonexistent")
