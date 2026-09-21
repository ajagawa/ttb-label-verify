"""Tests for the numeric field strategies: alcohol content and net contents.

These fields carry the tool's sharpest safety property, so the tests are
organised around what must *never* happen rather than around code paths:

*   A numeric difference must never resolve to MATCH. 46% against 45% is the
    one-character discrepancy an OCR engine can produce and exactly the
    violation the tool exists to catch.
*   Failing to find a value must never produce MISMATCH.
*   A label that contradicts itself (45% ABV printed beside 80 proof) must
    never come out clean, even when the ABV agrees with the application.
*   A number without a unit must never be read as a volume.

Strategies are exercised directly through a `StrategyContext` built from
synthetic lines, and once end to end through `rules.engine.verify` on the
shared compliant label.
"""

from __future__ import annotations

import pytest

from extraction.providers.stub import StubProvider
from extraction.spans import build_span_index
from rules import numeric
from rules.comparators import STRATEGIES, StrategyContext
from rules.engine import load_strategies, verify
from rules.numeric import (
    alcohol_content_strategy,
    net_contents_strategy,
    parse_abv,
    parse_volume_ml,
)
from rules.results import (
    ApplicationRecord,
    CheckOutcome,
    FieldResult,
    ImageQuality,
    Strategy,
    Verdict,
)
from rules.schema import load_ruleset
from tests.conftest import STATUTORY_WARNING, make_lines

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def context_for(
    field_id: str,
    specs: list,
    expected: str,
    *,
    ruleset=None,
    quality: ImageQuality | None = None,
) -> StrategyContext:
    """Build the context a strategy would receive from the engine.

    Uses the real rule set rather than a hand-built one, so these tests fail if
    a tolerance or the standards-of-fill table changes underneath them — that
    is a behaviour change worth noticing. `ruleset` takes a modified copy, for
    asserting that a threshold is genuinely read from the rule set.
    """
    ruleset = ruleset or load_ruleset()
    return StrategyContext(
        field_rule=ruleset.field(field_id),
        ruleset=ruleset,
        beverage=ruleset.beverage_class("distilled_spirits"),
        spans=build_span_index(make_lines(specs)),
        expected=expected,
        quality=quality,
    )


def retuned(**numeric_tuning) -> object:
    """The real rule set with `tuning.numeric` values replaced.

    Pydantic models here are frozen, so this is a deep copy rather than a
    mutation — the loader's cache is shared with every other test.
    """
    ruleset = load_ruleset()
    numeric_section = ruleset.tuning.numeric.model_copy(update=numeric_tuning)
    tuning = ruleset.tuning.model_copy(update={"numeric": numeric_section})
    return ruleset.model_copy(update={"tuning": tuning})


def abv_with(specs: list, expected: str, **kwargs) -> FieldResult:
    return alcohol_content_strategy(context_for("alcohol_content", specs, expected, **kwargs))


def abv(specs: list, expected: str = "45% Alc./Vol.") -> FieldResult:
    return alcohol_content_strategy(context_for("alcohol_content", specs, expected))


def volume(specs: list, expected: str = "750 mL") -> FieldResult:
    return net_contents_strategy(context_for("net_contents", specs, expected))


def check(result: FieldResult, check_id: str):
    matches = [c for c in result.checks if c.id == check_id]
    assert matches, f"no {check_id!r} check in {[c.id for c in result.checks]}"
    return matches[0]


def low(text: str, confidence: float) -> tuple[str, dict]:
    return (text, {"confidence": confidence})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_engine_loads_both_strategies(self) -> None:
        """Importing via the engine replaces the placeholders, with no import cycle."""
        load_strategies()
        assert STRATEGIES["alcohol_content"] is numeric.alcohol_content_strategy
        assert STRATEGIES["net_contents"] is numeric.net_contents_strategy


# ---------------------------------------------------------------------------
# Alcohol content
# ---------------------------------------------------------------------------


class TestAbvMatch:
    def test_exact_match(self) -> None:
        result = abv(["45% Alc./Vol."])
        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.PATTERN
        assert result.found == "45% Alc./Vol."

    def test_result_carries_evidence_and_citation(self) -> None:
        result = abv(["45% Alc./Vol."])
        assert result.bbox is not None and result.bbox.width > 0
        assert result.citation == "27 CFR 5.65"
        assert result.citation_url and "5.65" in result.citation_url
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.parametrize(
        "printed",
        [
            "45% Alc./Vol.",
            "45% Alc/Vol",
            "45%Alc./Vol.",
            "45 % ALC / VOL",
            "45% Alc. / Vol.",
            "45% Alcohol by Volume",
            "45% ABV",
            "Alc./Vol. 45%",
            "Alc/Vol 45%",
            "Alcohol 45% by Volume",
        ],
    )
    def test_spacing_and_wording_variants(self, printed: str) -> None:
        assert abv([printed]).verdict is Verdict.MATCH

    def test_decimal_abv(self) -> None:
        result = abv(["40.5% Alc./Vol."], expected="40.5%")
        assert result.verdict is Verdict.MATCH
        assert "40.5%" in result.reason

    def test_application_stated_as_proof(self) -> None:
        """The application's own value is parsed too; 90 proof is 45%."""
        assert abv(["45% Alc./Vol."], expected="90 Proof").verdict is Verdict.MATCH

    def test_statement_found_among_other_text(self) -> None:
        """Sub-line spans recover the statement from a line carrying other fields."""
        result = abv(["750 mL 45% Alc./Vol. (90 Proof)"])
        assert result.verdict is Verdict.MATCH
        assert result.found == "45% Alc./Vol."


class TestAbvNeverFuzzyForgiven:
    def test_46_against_45_is_not_a_match(self) -> None:
        """The case the whole numeric design exists for."""
        result = abv(["46% Alc./Vol."])
        assert result.verdict is not Verdict.MATCH

    def test_clear_high_confidence_difference_is_mismatch(self) -> None:
        result = abv(["46% Alc./Vol."])
        assert result.verdict is Verdict.MISMATCH
        assert result.bbox is not None

    def test_difference_states_both_readings_and_the_ocr_caveat(self) -> None:
        reason = abv(["46% Alc./Vol."]).reason
        assert "46%" in reason and "45%" in reason
        assert "OCR" in reason
        assert "visually" in reason

    def test_mid_confidence_difference_is_review_not_mismatch(self) -> None:
        """A digit read at middling confidence is where 6-for-5 misreads live."""
        bar = load_ruleset().tuning.numeric.mismatch_min_ocr_confidence
        result = abv([low("46% Alc./Vol.", bar - 0.1)])
        assert result.verdict is Verdict.REVIEW
        assert "46%" in result.reason and "45%" in result.reason

    def test_difference_within_tolerance_is_still_reported(self) -> None:
        """0.2 points is inside 27 CFR 5.65 tolerance, but it is a printed difference."""
        result = abv(["45.2% Alc./Vol."])
        assert result.verdict is Verdict.REVIEW
        assert "45.2%" in result.reason and "45%" in result.reason

    def test_mismatch_sorts_below_a_match(self) -> None:
        """Batch triage is worst-first on score; a clear difference must lead."""
        assert abv(["46% Alc./Vol."]).score < abv(["45% Alc./Vol."]).score

    def test_digits_are_not_folded(self) -> None:
        """Trap 2: a letter-for-digit misread must not become a match.

        Canonical text folds 5->S, so "4S%" and "45%" canonicalise alike. The
        numeric path must refuse to read "4S" as a number at all.
        """
        result = abv(["4S% Alc./Vol."])
        assert result.verdict is not Verdict.MATCH


class TestAbvProof:
    def test_consistent_proof_passes_and_corroborates(self) -> None:
        result = abv(["45% Alc./Vol. (90 Proof)"])
        assert result.verdict is Verdict.MATCH
        assert check(result, "proof_consistency").outcome is CheckOutcome.PASS
        assert "Proof" in result.reason

    def test_inconsistent_proof_is_flagged_even_when_abv_matches(self) -> None:
        """The label is wrong on its face, whatever the application says."""
        result = abv(["45% Alc./Vol. (80 Proof)"])
        assert result.verdict is Verdict.REVIEW
        proof = check(result, "proof_consistency")
        assert proof.outcome is CheckOutcome.FAIL
        assert "80" in proof.summary and "90" in proof.summary

    def test_inconsistent_proof_on_a_separate_line(self) -> None:
        result = abv(["45% Alc./Vol.", "80 PROOF"])
        assert result.verdict is Verdict.REVIEW
        assert check(result, "proof_consistency").outcome is CheckOutcome.FAIL

    def test_proof_within_tolerance_is_consistent(self) -> None:
        """40.3% is 80.6 proof; a label rounding to 80.5 is within 0.6."""
        result = abv(["40.3% Alc./Vol. (80.5 Proof)"], expected="40.3%")
        assert check(result, "proof_consistency").outcome is CheckOutcome.PASS

    def test_abv_only_as_proof_agreeing_is_review(self) -> None:
        """Value agrees, but a percentage-by-volume statement is required."""
        result = abv(["90 Proof"])
        assert result.verdict is Verdict.REVIEW
        assert result.found == "90 Proof"
        assert "90 proof" in result.reason and "45%" in result.reason
        assert check(result, "abv_statement_form").outcome is CheckOutcome.ADVISORY

    def test_abv_only_as_proof_differing_is_mismatch(self) -> None:
        result = abv(["80 Proof"])
        assert result.verdict is Verdict.MISMATCH
        assert "40%" in result.reason

    def test_two_abv_statements_that_disagree(self) -> None:
        result = abv(["45% Alc./Vol.", "46% Alc./Vol."])
        assert result.verdict is Verdict.REVIEW
        assert check(result, "abv_statement_consistency").outcome is CheckOutcome.FAIL


class TestAbvGarbledUnitWording:
    """A single-character OCR error in the *wording* must not demote a statement.

    Regression for a false flag seen with PP-OCR on
    fixtures/labels/brand_twice_bottler.png: a compliant "45% Alc./Vol.
    (90 Proof)" was read as "45% Alc./Nol.", which matched none of the strict
    patterns, so the proof became the best hit and the tool announced that the
    label had no alcohol-by-volume statement.
    """

    MISREAD = "45% Alc./Nol. (90 Proof)"

    def test_real_ocr_misread_of_vol_still_matches(self) -> None:
        result = abv([self.MISREAD])
        assert result.verdict is Verdict.MATCH
        assert result.found == "45% Alc./Nol."

    def test_the_misread_wording_is_disclosed_to_the_agent(self) -> None:
        """MATCH on the number, but the garbled wording is still surfaced."""
        reason = abv([self.MISREAD]).reason
        assert "Alc./Nol." in reason
        assert "misreading" in reason

    def test_proof_is_still_cross_validated(self) -> None:
        assert check(abv([self.MISREAD]), "proof_consistency").outcome is CheckOutcome.PASS

    @pytest.mark.parametrize(
        "printed",
        [
            "45% Alc./Nol.",
            "45% Aic./Vol.",
            "45% ALC.NOL.",
            "45% Alcohol by Volurne",
            "Alc./Nol. 45%",
        ],
    )
    def test_other_single_character_wording_errors(self, printed: str) -> None:
        assert abv([printed]).verdict is Verdict.MATCH, printed

    def test_digits_stay_strict_under_garbled_wording(self) -> None:
        """The value is never fuzzy-forgiven, however noisy the wording is."""
        result = abv(["46% Alc./Nol."])
        assert result.verdict is not Verdict.MATCH
        assert "46%" in result.reason and "45%" in result.reason

    def test_garbled_wording_difference_is_review_not_mismatch(self) -> None:
        """A garbled character beside the digits is evidence the read was noisy.

        The difference is reported, but as a flag rather than an assertion —
        unlike the same difference read cleanly, which is a MISMATCH.
        """
        assert abv(["46% Alc./Nol."]).verdict is Verdict.REVIEW
        assert abv(["46% Alc./Vol."]).verdict is Verdict.MISMATCH

    def test_clean_statement_outranks_a_garbled_one(self) -> None:
        result = abv(["45% Alc./Nol.", "45% Alc./Vol."])
        assert result.found == "45% Alc./Vol."
        assert result.verdict is Verdict.MATCH

    @pytest.mark.parametrize(
        "printed",
        [
            "51% all natural corn",
            "45% agave",
            "40% of the barrel",
            "12% ABC",
        ],
    )
    def test_unrelated_wording_is_not_accepted_as_alcohol_content(self, printed: str) -> None:
        """Whole-phrase comparison, not token-by-token, is what makes this safe."""
        result = abv([printed], expected="45%")
        assert result.verdict is not Verdict.MATCH
        assert result.verdict is not Verdict.MISMATCH

    def test_wording_similarity_helper(self) -> None:
        threshold = load_ruleset().tuning.numeric.wording_min_similarity
        assert numeric._fuzzy_wording("ALC./NOL.", numeric._ABV_WORDINGS, threshold)
        assert numeric._fuzzy_wording("PROOL", numeric._PROOF_WORDINGS, threshold)
        # Two characters wrong, and short accidental tokens, are rejected.
        assert not numeric._fuzzy_wording("ANN./NOL.", numeric._ABV_WORDINGS, threshold)
        assert not numeric._fuzzy_wording("OIL", numeric._ABV_WORDINGS, threshold)
        assert not numeric._fuzzy_wording("ALL", numeric._ABV_WORDINGS, threshold)


class TestUnsegmentedOcrTokens:
    """OCR that returns no spaces must not read as "could not locate".

    Regression for the degraded fixtures, where PP-OCR reads a compliant
    "45% Alc./Vol. (90 Proof)" as the single token "45%Alc.Vol.(90Proof)".
    The span index cannot split inside a token, so full-matching alone found
    nothing and the field was reported as not located on a label whose
    alcohol content is present and correct.
    """

    @pytest.mark.parametrize("printed", ["45%Alc.Vol.(90Proof)", "45%Alc.Nol.(90Proof)"])
    def test_run_together_statement_is_read(self, printed: str) -> None:
        result = abv([printed])
        assert result.verdict is Verdict.MATCH
        assert result.bbox is not None

    def test_both_statements_inside_one_token_are_cross_validated(self) -> None:
        """The bracketed proof is a separate statement even without a space."""
        assert (
            check(abv(["45%Alc.Vol.(90Proof)"]), "proof_consistency").outcome is CheckOutcome.PASS
        )
        assert (
            check(abv(["45%Alc.Vol.(80Proof)"]), "proof_consistency").outcome is CheckOutcome.FAIL
        )

    def test_digits_stay_strict_inside_a_token(self) -> None:
        assert abv(["46%Alc.Vol.(92Proof)"]).verdict is Verdict.MISMATCH

    def test_run_together_volume_is_read(self) -> None:
        assert volume(["750mL"]).verdict is Verdict.MATCH

    def test_a_token_with_no_statement_in_it_is_still_not_found(self) -> None:
        """Looking inside a token does not turn every number into a field."""
        assert volume(["Lot750-2", "Batch1750"]).strategy is Strategy.NOT_FOUND
        assert abv(["Lot45-2"]).strategy is Strategy.NOT_FOUND

    def test_a_longer_number_is_not_mined_for_a_shorter_one(self) -> None:
        """Full-matching, not searching: "1750 mL" does not contain "750 mL"."""
        result = volume(["1750 mL"])
        assert result.found == "1750 mL"
        assert result.verdict is Verdict.MISMATCH

    def test_a_second_volume_elsewhere_on_the_label_is_reported_not_hidden(self) -> None:
        """Two located statements that disagree are a finding, not a silent pick."""
        result = volume(["Bottled in 1995 from 700 mL casks", "750 mL"])
        assert result.found == "750 mL"
        assert result.verdict is Verdict.REVIEW
        assert check(result, "volume_statement_consistency").outcome is CheckOutcome.FAIL


class TestAbvBarePercentGuard:
    def test_bare_percent_ignored_when_a_labelled_statement_exists(self) -> None:
        """ "51% corn" must not contradict or outrank "45% Alc./Vol."."""
        result = abv(["Mash: 51% corn", "45% Alc./Vol."])
        assert result.verdict is Verdict.MATCH
        assert result.found == "45% Alc./Vol."

    def test_bare_percent_alone_is_review_even_when_equal(self) -> None:
        result = abv(["45%"])
        assert result.verdict is Verdict.REVIEW
        assert check(result, "abv_statement_form").outcome is CheckOutcome.ADVISORY

    def test_bare_percent_alone_never_mismatches(self) -> None:
        """It may not be the alcohol statement at all."""
        result = abv(["Mash: 51% corn"])
        assert result.verdict is Verdict.REVIEW


class TestAbvNotFoundAndConfidence:
    def test_no_abv_on_label_is_review_not_mismatch(self) -> None:
        result = abv(["OLD TOM DISTILLERY", "750 mL"])
        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.NOT_FOUND
        assert result.bbox is None

    def test_empty_pool_is_not_found(self) -> None:
        assert abv([]).strategy is Strategy.NOT_FOUND

    def test_low_ocr_confidence_is_review_even_on_exact_match(self) -> None:
        result = abv([low("45% Alc./Vol.", 0.3)])
        assert result.verdict is Verdict.REVIEW
        assert "confidence" in result.reason

    def test_low_ocr_confidence_difference_is_review(self) -> None:
        assert abv([low("46% Alc./Vol.", 0.3)]).verdict is Verdict.REVIEW

    def test_unreadable_application_value_is_review(self) -> None:
        result = abv(["45% Alc./Vol."], expected="forty-five")
        assert result.verdict is Verdict.REVIEW
        assert "forty-five" in result.reason

    def test_warning_numerals_are_not_read(self) -> None:
        """ "(1)" and "(2)" in the warning are not an alcohol statement."""
        assert abv([STATUTORY_WARNING[:60]]).strategy is Strategy.NOT_FOUND

    def test_warning_lines_are_excluded_even_unmasked(self) -> None:
        """Belt and braces for when the warning was not resolved and masked.

        A percentage on a line carrying one of the warning's anchor phrases is
        not read, even though the bare pattern would otherwise take it.
        """
        result = abv(["the risk of birth defects 45%"])
        assert result.strategy is Strategy.NOT_FOUND


class TestParseAbv:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("45% Alc./Vol.", 45.0),
            ("45%", 45.0),
            ("45", 45.0),
            ("90 Proof", 45.0),
            ("Alc/Vol 40.5%", 40.5),
            ("40,5% vol", 40.5),
        ],
    )
    def test_application_formats(self, text: str, value: float) -> None:
        parsed = parse_abv(text)
        assert parsed is not None and parsed[0] == pytest.approx(value)

    def test_unreadable(self) -> None:
        assert parse_abv("forty-five") is None


# ---------------------------------------------------------------------------
# Net contents
# ---------------------------------------------------------------------------


class TestVolumeUnits:
    @pytest.mark.parametrize("printed", ["750 mL", "750mL", "75 cL", "0.75 L", "0,75 L"])
    def test_metric_forms_equal_750(self, printed: str) -> None:
        result = volume([printed])
        assert result.verdict is Verdict.MATCH, result.reason
        assert result.found == printed

    @pytest.mark.parametrize("printed", ["25.4 fl oz", "25.4 FL. OZ.", "25.4 fl.oz"])
    def test_fluid_ounces_equal_750_within_conversion_tolerance(self, printed: str) -> None:
        result = volume([printed])
        assert result.verdict is Verdict.MATCH, result.reason
        assert "751.2 mL" in result.reason

    def test_application_in_litres(self) -> None:
        assert volume(["750 mL"], expected="0.75 L").verdict is Verdict.MATCH

    def test_metric_preferred_over_fluid_ounces(self) -> None:
        result = volume(["25.4 FL OZ", "750 mL"])
        assert result.verdict is Verdict.MATCH
        assert result.found == "750 mL"

    def test_parse_volume(self) -> None:
        for text in ("750 mL", "75 cL", "0.75 L", "750 ml."):
            parsed = parse_volume_ml(text)
            assert parsed is not None and parsed[0] == pytest.approx(750.0)
        assert parse_volume_ml("25.4 fl oz")[0] == pytest.approx(751.17, abs=0.01)
        assert parse_volume_ml("1,000 mL")[0] == pytest.approx(1000.0)
        assert parse_volume_ml("1,75 L")[0] == pytest.approx(1750.0)


class TestVolumeNeverFuzzyForgiven:
    def test_different_volume_is_mismatch_with_both_readings(self) -> None:
        result = volume(["700 mL"])
        assert result.verdict is Verdict.MISMATCH
        assert "700 mL" in result.reason and "750 mL" in result.reason
        assert "OCR" in result.reason

    def test_one_millilitre_metric_difference_is_not_absorbed(self) -> None:
        """`tolerance_ml` covers fl-oz conversion rounding, not printed differences."""
        assert volume(["749 mL"]).verdict is not Verdict.MATCH

    def test_mid_confidence_difference_is_review(self) -> None:
        bar = load_ruleset().tuning.numeric.mismatch_min_ocr_confidence
        result = volume([low("700 mL", bar - 0.1)])
        assert result.verdict is Verdict.REVIEW

    def test_low_confidence_is_review(self) -> None:
        assert volume([low("750 mL", 0.3)]).verdict is Verdict.REVIEW


class TestStandardsOfFill:
    def test_standard_fill_passes(self) -> None:
        result = volume(["750 mL"])
        assert check(result, "standard_of_fill").outcome is CheckOutcome.PASS

    def test_nonstandard_fill_matching_application_is_review(self) -> None:
        """Agrees with the application, but the size itself is reportable."""
        result = volume(["740 mL"], expected="740 mL")
        assert result.verdict is Verdict.REVIEW
        fill = check(result, "standard_of_fill")
        assert fill.outcome is CheckOutcome.FAIL
        assert fill.citation == "27 CFR 5.203"
        assert "standard of fill" in result.reason

    def test_nonstandard_fill_differing_from_application_is_mismatch(self) -> None:
        result = volume(["740 mL"])
        assert result.verdict is Verdict.MISMATCH
        assert check(result, "standard_of_fill").outcome is CheckOutcome.FAIL


class TestVolumeGuards:
    def test_unitless_number_is_not_a_volume(self) -> None:
        """A bare 750 may be a batch or bottle number. Never read as 750 mL."""
        result = volume(["Batch 750", "Bottle No. 750"])
        assert result.strategy is Strategy.NOT_FOUND
        assert result.verdict is Verdict.REVIEW

    def test_unitless_application_value_is_review(self) -> None:
        result = volume(["750 mL"], expected="750")
        assert result.verdict is Verdict.REVIEW
        assert "unit" in result.reason

    def test_abv_is_not_read_as_a_volume(self) -> None:
        assert volume(["45% Alc./Vol. (90 Proof)"]).strategy is Strategy.NOT_FOUND

    def test_warning_statement_is_not_read(self) -> None:
        result = volume([STATUTORY_WARNING[:60], "Surgeon General 2 L"])
        assert result.strategy is Strategy.NOT_FOUND

    def test_disagreeing_volume_statements_are_flagged(self) -> None:
        result = volume(["750 mL", "33.8 FL OZ"])
        assert result.verdict is Verdict.REVIEW
        assert check(result, "volume_statement_consistency").outcome is CheckOutcome.FAIL

    def test_agreeing_dual_statements_are_consistent(self) -> None:
        result = volume(["750 mL", "25.4 FL OZ"])
        assert result.verdict is Verdict.MATCH
        assert not [c for c in result.checks if c.id == "volume_statement_consistency"]


# ---------------------------------------------------------------------------
# Assertion thresholds come from the rule set
# ---------------------------------------------------------------------------


class TestThresholdsComeFromTheRuleSet:
    """A verdict must be reproducible from the recorded rule-set version.

    Each test re-runs an unchanged label under a modified rule set and asserts
    the verdict follows it. A future refactor that reintroduces a hardcoded
    threshold — or a default that shadows the rule set — fails here rather
    than silently making results unreproducible.
    """

    def test_raising_the_confidence_bar_withdraws_a_mismatch(self) -> None:
        clean = ["46% Alc./Vol."]  # read at 0.95 confidence by the factory
        assert abv(clean).verdict is Verdict.MISMATCH
        strict = retuned(mismatch_min_ocr_confidence=0.99)
        assert abv_with(clean, "45%", ruleset=strict).verdict is Verdict.REVIEW

    def test_lowering_the_confidence_bar_admits_a_mismatch(self) -> None:
        murky = [low("46% Alc./Vol.", 0.5)]
        assert abv(murky).verdict is Verdict.REVIEW
        lenient = retuned(mismatch_min_ocr_confidence=0.45)
        assert abv_with(murky, "45%", ruleset=lenient).verdict is Verdict.MISMATCH

    def test_the_confidence_bar_governs_net_contents_too(self) -> None:
        strict = retuned(mismatch_min_ocr_confidence=0.99)
        context = context_for("net_contents", ["700 mL"], "750 mL", ruleset=strict)
        assert net_contents_strategy(context).verdict is Verdict.REVIEW

    def test_raising_the_wording_bar_rejects_a_garbled_statement(self) -> None:
        """At 0.95, "Alc./Nol." no longer clears the bar and proof-only remains."""
        garbled = ["45% Alc./Nol. (90 Proof)"]
        assert abv(garbled).verdict is Verdict.MATCH
        strict = retuned(wording_min_similarity=0.95)
        assert abv_with(garbled, "45% Alc./Vol.", ruleset=strict).verdict is Verdict.REVIEW

    def test_lowering_the_wording_bar_admits_a_worse_misread(self) -> None:
        """ "ALL NOL" scores 0.71 against "ALC VOL" — rejected at 0.80, taken at 0.70."""
        worse = ["45% All./Nol."]
        assert abv(worse).verdict is not Verdict.MATCH
        lenient = retuned(wording_min_similarity=0.70)
        assert abv_with(worse, "45%", ruleset=lenient).verdict is Verdict.MATCH

    def test_no_module_constant_shadows_the_rule_set(self) -> None:
        """The deleted constants must not come back as module-level defaults."""
        assert not hasattr(numeric, "MISMATCH_MIN_OCR_CONFIDENCE")
        assert not hasattr(numeric, "_WORDING_MIN_SIMILARITY")


class TestImageQualityWithdrawsAssertions:
    """Quality can only ever take an assertion away, never add one."""

    GOOD = ImageQuality(usable=True, score=0.9)
    POOR = ImageQuality(usable=True, score=0.3)
    UNUSABLE = ImageQuality(usable=False, score=0.2)

    def test_mismatch_stands_on_a_good_image(self) -> None:
        assert abv_with(["46% Alc./Vol."], "45%", quality=self.GOOD).verdict is Verdict.MISMATCH

    @pytest.mark.parametrize("quality", [POOR, UNUSABLE])
    def test_mismatch_is_withdrawn_on_a_degraded_image(self, quality: ImageQuality) -> None:
        result = abv_with(["46% Alc./Vol."], "45%", quality=quality)
        assert result.verdict is Verdict.REVIEW
        assert "46%" in result.reason and "45%" in result.reason

    def test_net_contents_mismatch_is_withdrawn_too(self) -> None:
        context = context_for("net_contents", ["700 mL"], "750 mL", quality=self.POOR)
        assert net_contents_strategy(context).verdict is Verdict.REVIEW

    def test_good_quality_does_not_promote_a_low_confidence_read(self) -> None:
        """A clean photograph of an unreadable digit is still unreadable."""
        result = abv_with([low("46% Alc./Vol.", 0.5)], "45%", quality=self.GOOD)
        assert result.verdict is Verdict.REVIEW

    def test_absent_quality_behaves_as_before(self) -> None:
        assert abv_with(["46% Alc./Vol."], "45%", quality=None).verdict is Verdict.MISMATCH


# ---------------------------------------------------------------------------
# Determinism and end to end
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_result(self) -> None:
        specs = ["45% Alc./Vol.", "46% Alc./Vol.", "(90 Proof)"]
        assert abv(specs) == abv(specs)

    def test_line_order_does_not_change_the_verdict(self) -> None:
        a = abv(["45% Alc./Vol.", "80 Proof"])
        b = abv(["80 Proof", "45% Alc./Vol."])
        assert (a.verdict, a.found) == (b.verdict, b.found)


class TestEndToEnd:
    RECORD = ApplicationRecord(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol.",
        net_contents="750 mL",
    )

    def test_both_fields_now_resolve_on_the_compliant_label(self, sample_label_lines) -> None:
        result = verify(StubProvider(lines=sample_label_lines).extract(None), self.RECORD)

        alcohol = result.field("alcohol_content")
        contents = result.field("net_contents")

        for field in (alcohol, contents):
            assert "does not yet check" not in field.reason
            assert field.strategy is Strategy.PATTERN
            assert field.verdict is Verdict.MATCH

        assert alcohol.found == "45% Alc./Vol."
        assert contents.found == "750 mL"
        assert check(alcohol, "proof_consistency").outcome is CheckOutcome.PASS

    def test_mismatched_application_end_to_end(self, sample_label_lines) -> None:
        record = self.RECORD.model_copy(update={"alcohol_content": "46%", "net_contents": "1 L"})
        result = verify(StubProvider(lines=sample_label_lines).extract(None), record)

        assert result.field("alcohol_content").verdict is Verdict.MISMATCH
        assert result.field("net_contents").verdict is Verdict.MISMATCH
