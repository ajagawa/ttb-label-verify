"""Tests for the class/type strategy (rules/class_type.py).

Organised by the three resolution stages, then by the safety properties that
cut across them. The concentration is where the field is hard:

*   **Abbreviations and spelling variants** — "Ky." and "Whisky" must resolve,
    and the abbreviation table must work in the same canonical space as the
    scorer, including for values that OCR folding rewrites ("SINGLE" is
    "SINGIE" once L folds to I).
*   **REVIEW versus MISMATCH** — a vocabulary finding asserts a violation only
    when every piece of corroboration holds. Each blocker is tested on its own,
    because a regression in any one of them would turn a doubtful read into a
    confident accusation.
*   **Not-found is never a mismatch** — the invariant that matters most.

Most tests call the strategy directly with a hand-built context, because that
is what lets a test place the brand anchor precisely. A handful go through
`rules.engine.verify` to show the strategy behaves the same inside the real
resolution order, with the brand masked by the engine rather than by the test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from extraction.geometry import BBox
from extraction.normalize import canonicalize
from extraction.providers.base import Line
from extraction.providers.stub import StubProvider
from extraction.spans import build_span_index, mask_spans
from rules import class_type as ct
from rules.comparators import (
    STRATEGIES,
    StrategyContext,
    explain_residual,
    residual_difference,
)
from rules.engine import load_strategies, verify
from rules.results import ApplicationRecord, FieldResult, ImageQuality, Strategy, Verdict
from rules.schema import BeverageClass, RuleSet, load_ruleset
from tests.conftest import make_lines

KSBW = "Kentucky Straight Bourbon Whiskey"

RECORD = ApplicationRecord(
    brand_name="OLD TOM DISTILLERY",
    class_type=KSBW,
    alcohol_content="45% Alc./Vol.",
    net_contents="750 mL",
)

BRAND = ("OLD TOM DISTILLERY", {"height": 48.0})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _beverage(**overrides) -> BeverageClass:
    return load_ruleset().beverage_class("distilled_spirits").model_copy(update=overrides)


def tuned(**overrides) -> RuleSet:
    """The rule set with `tuning.class_type` values overridden.

    Used to prove the strategy reads its assertion thresholds from the rule
    set rather than from constants: changing a value here must change the
    verdict. `load_ruleset` is cached and `model_copy` does not mutate it, so
    the override is local to one test.
    """
    ruleset = load_ruleset()
    class_type = ruleset.tuning.class_type.model_copy(update=overrides)
    return ruleset.model_copy(
        update={"tuning": ruleset.tuning.model_copy(update={"class_type": class_type})}
    )


def tuned_names(**overrides) -> RuleSet:
    """The rule set with `tuning.names` overridden — the budget shared with brand."""
    ruleset = load_ruleset()
    names = ruleset.tuning.names.model_copy(update=overrides)
    return ruleset.model_copy(update={"tuning": ruleset.tuning.model_copy(update={"names": names})})


def run(
    lines: tuple[Line, ...],
    expected: str,
    *,
    brand_line: int | None = 0,
    mask_brand: bool = True,
    beverage: BeverageClass | None = None,
    ruleset: RuleSet | None = None,
    quality: ImageQuality | None = None,
) -> FieldResult:
    """Run the strategy on laid-out lines, the way the engine would.

    Args:
        lines: The label.
        expected: The application's class/type.
        brand_line: Index of the line standing in for the resolved brand name;
            its bbox becomes `anchor_bbox`. None simulates a brand that was not
            located.
        mask_brand: Remove the brand region from the pool first, as the engine
            does. False feeds the brand text to the strategy, to test its own
            guard.
        beverage: Override the beverage class (for custom abbreviations).
        ruleset: Override the rule set, for the tuning tests.
        quality: The engine's image-quality summary. None means the engine did
            not supply one.
    """
    ruleset = ruleset or load_ruleset()
    anchor = lines[brand_line].bbox if brand_line is not None else None
    spans = build_span_index(lines)
    if mask_brand and anchor is not None:
        spans = mask_spans(spans, anchor)
    context = StrategyContext(
        field_rule=ruleset.field("class_type"),
        ruleset=ruleset,
        beverage=beverage or ruleset.beverage_class("distilled_spirits"),
        spans=spans,
        expected=expected,
        anchor_bbox=anchor,
        quality=quality,
    )
    return ct.class_type_strategy(context)


def label(*below_brand: str | tuple[str, dict]) -> tuple[Line, ...]:
    """A brand name in display type with the given lines beneath it."""
    return make_lines([BRAND, *below_brand])


def extract(lines: tuple[Line, ...]):
    return StubProvider(lines=lines).extract(None)


def _as_bbox(result: FieldResult) -> BBox:
    assert result.bbox is not None
    b = result.bbox
    return BBox(b.left, b.top, b.left + b.width, b.top + b.height)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_strategy_is_registered_with_the_engine(self) -> None:
        """Importing the module replaces the declared placeholder."""
        load_strategies()
        assert STRATEGIES["class_type"] is ct.class_type_strategy

    @pytest.mark.parametrize(
        "imports",
        ["import rules.class_type", "import rules.engine; rules.engine.load_strategies()"],
    )
    def test_no_import_cycle_in_either_order(self, imports: str) -> None:
        """rules.class_type imports rules.engine, which imports it lazily.

        Run in a fresh interpreter: inside this process both are already
        imported and a cycle could not show itself.
        """
        root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"{imports}; import rules.class_type as m; "
                "from rules.comparators import STRATEGIES; "
                "assert STRATEGIES['class_type'] is m.class_type_strategy",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# Stage 1 — anchored
# ---------------------------------------------------------------------------


class TestAnchored:
    def test_exact_match(self) -> None:
        result = run(label(KSBW), KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED
        assert result.found == KSBW
        assert result.reason == "Label matches the application record exactly."

    def test_case_variant_matches(self) -> None:
        result = run(label(KSBW.upper()), KSBW)

        assert result.verdict is Verdict.MATCH
        assert "capitalisation" in result.reason

    def test_wrapped_over_two_lines(self) -> None:
        """The multi-line span family recovers a designation set on two lines."""
        result = run(label("Kentucky Straight", "Bourbon Whiskey"), KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED
        assert result.found == KSBW
        # The highlighted region covers both lines, not just one of them.
        assert result.bbox is not None and result.bbox.height > 40

    def test_low_confidence_exact_read_is_review(self) -> None:
        """Agreeing text the engine is unsure it read is not evidence."""
        result = run(label((KSBW, {"confidence": 0.3})), KSBW)

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.ANCHORED
        assert result.found == KSBW
        # Says the words agree, rather than implying a textual difference.
        assert "agrees with the application" in result.reason
        assert "not an exact match" not in result.reason

    def test_prefers_the_display_instance_to_a_longer_line(self) -> None:
        """Length penalty: the designation itself, not a sentence mentioning it."""
        lines = label(KSBW, "A fine Kentucky Straight Bourbon Whiskey aged in oak")
        result = run(lines, KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.found == KSBW


# ---------------------------------------------------------------------------
# Abbreviations and spelling variants
# ---------------------------------------------------------------------------


class TestAbbreviations:
    def test_ky_on_label_matches_kentucky_in_application(self) -> None:
        result = run(label("Ky. Straight Bourbon Whiskey"), KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED
        assert "“KY” read as “KENTUCKY”" in result.reason

    def test_abbreviation_in_application_matches_full_form_on_label(self) -> None:
        """Expansion is applied to both sides, not just the label."""
        result = run(label(KSBW), "Ky. Straight Bourbon Whiskey")
        assert result.verdict is Verdict.MATCH

    def test_abbreviation_is_what_makes_it_match(self) -> None:
        """Without the table, "Ky." alone costs enough to fall out of MATCH.

        Guards the test above against passing for the wrong reason (fuzzy
        tolerance absorbing the abbreviation on its own).
        """
        result = run(
            label("Ky. Straight Bourbon Whiskey"), KSBW, beverage=_beverage(abbreviations={})
        )
        assert result.verdict is not Verdict.MATCH

    def test_whisky_matches_whiskey(self) -> None:
        result = run(label("Kentucky Straight Bourbon Whisky"), KSBW)

        assert result.verdict is Verdict.MATCH
        assert "“WHISKY” read as “WHISKEY”" in result.reason

    def test_ocr_confused_abbreviation_still_expands(self) -> None:
        """ "WH1SKY" folds to "WHISKY" and then expands — the order matters."""
        table = ct._table_for({"WHISKY": "WHISKEY"})
        assert ct._expand(canonicalize("Wh1sky"), table).text == "WHISKEY"

    def test_values_containing_folded_letters_compare_in_canonical_space(self) -> None:
        """The trap: "SINGLE" is "SINGIE" and "MALT" is "MAIT" once L folds to I.

        An abbreviation table left in plain English would expand "Sgl" to
        "SINGLE", which never equals the expected value's canonical "SINGIE".
        Canonicalising the table itself is what makes this match.
        """
        beverage = _beverage(abbreviations={"Sgl": "Single", "Mlt": "Malt", "Whisky": "Whiskey"})
        result = run(label("Sgl Mlt Scotch Whisky"), "Single Malt Scotch Whisky", beverage=beverage)

        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED

    def test_table_keys_and_values_are_canonical(self) -> None:
        table = ct._table_for({"Mlt.": "Malt"})
        assert ("MIT",) in table
        assert table[("MIT",)].value_tokens == ("MAIT",)
        # Display forms are kept for the reason text.
        assert table[("MIT",)].display_value == "Malt"

    def test_expansion_is_token_level_not_substring(self) -> None:
        table = ct._table_for({"KY": "KENTUCKY"})
        assert ct._expand("KYOTO KY", table).text == "KYOTO KENTUCKY"

    def test_expansion_is_single_pass(self) -> None:
        """A chained table must not depend on iteration order or loop."""
        table = ct._table_for({"A": "B", "B": "A"})
        assert ct._expand("A B", table).text == "B A"

    def test_longest_multi_token_key_wins(self) -> None:
        table = ct._table_for({"STR": "STRAIGHT", "STR BRBN": "STRAIGHT BOURBON"})
        expansion = ct._expand("STR BRBN WHISKEY", table)
        assert expansion.text == "STRAIGHT BOURBON WHISKEY"
        assert len(expansion.applied) == 1

    @pytest.mark.parametrize("text", [KSBW, "Single Malt Scotch Whisky", "Old No. 7", "45.5 Añejo"])
    def test_canonicalize_is_idempotent(self, text: str) -> None:
        """`_score_expanded` relies on re-canonicalising canonical text being a no-op."""
        once = canonicalize(text)
        assert canonicalize(once) == once


class TestResidualCharacterDifference:
    """One wrong letter in a long designation must not resolve to MATCH.

    Similarity is a ratio, so a single differing character costs less the
    longer the value: "Kentucky Straight Burbon Whiskey" scores 0.97 against
    "Bourbon" and clears the 0.90 threshold. After normalisation and
    abbreviation expansion, any character difference that survives is either a
    misread or a genuinely different word; the tool cannot tell which, so it
    reports rather than decides. This is fixture `class_long_one_letter_off`.
    """

    def test_missing_letter_in_a_long_designation_is_review(self) -> None:
        result = run(label("Kentucky Straight Burbon Whiskey"), KSBW)

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.ANCHORED
        assert result.found == "Kentucky Straight Burbon Whiskey"
        assert result.bbox is not None
        assert "differ by 1 character" in result.reason
        assert "“O” is missing from the label" in result.reason

    def test_substituted_letter_in_a_long_designation_is_review(self) -> None:
        result = run(label("Kentucky Straight Bourbon Whiskee"), KSBW)

        assert result.verdict is Verdict.REVIEW
        assert "“Y” in the application reads “E” on the label" in result.reason

    def test_letterspaced_single_token_misread_is_review(self) -> None:
        """PP-OCR's one-token reading of display type gets the same treatment."""
        result = run(label("KentuckyStraightBurbonWhiskey"), KSBW)
        assert result.verdict is Verdict.REVIEW

    def test_residual_is_never_a_mismatch(self) -> None:
        """A misread and a misprint look identical from here — report, not assert."""
        result = run(label("Kentucky Straight Burbon Whiskey"), KSBW)
        assert result.verdict is not Verdict.MISMATCH

    def test_wording_is_shared_with_the_brand_name(self) -> None:
        """One kind of finding, one wording, whichever field raised it."""
        found = "Kentucky Straight Burbon Whiskey"
        result = run(label(found), KSBW)
        expected_reason = explain_residual(found, KSBW, residual_difference(found, KSBW))
        assert result.reason == expected_reason

    # --- what must still MATCH ---------------------------------------------

    def test_exact_still_matches(self) -> None:
        assert run(label(KSBW), KSBW).verdict is Verdict.MATCH

    def test_case_only_still_matches(self) -> None:
        assert run(label(KSBW.upper()), KSBW).verdict is Verdict.MATCH

    def test_ky_abbreviation_counts_zero(self) -> None:
        """Measured after expansion — before it, "Ky." is six differing characters."""
        assert residual_difference("Ky. Straight Bourbon Whiskey", KSBW).count > 0

        result = run(label("Ky. Straight Bourbon Whiskey"), KSBW)
        assert result.verdict is Verdict.MATCH

    def test_whisky_spelling_counts_zero(self) -> None:
        result = run(label("Kentucky Straight Bourbon Whisky"), KSBW)
        assert result.verdict is Verdict.MATCH

    def test_abbreviated_application_counts_zero(self) -> None:
        """Expansion applies to the expected side too."""
        result = run(label(KSBW), "Ky. Straight Bourbon Whisky")
        assert result.verdict is Verdict.MATCH

    def test_ocr_confusion_class_counts_zero(self) -> None:
        """ "Wh1skey" folds to WHISKEY; the fold table already removed it."""
        result = run(label("Kentucky Straight Bourbon Wh1skey"), KSBW)
        assert result.verdict is Verdict.MATCH

    def test_folded_letters_in_expanded_values_count_zero(self) -> None:
        """ "Sgl" → "Single": the L must survive expansion as an L, not an I."""
        beverage = _beverage(abbreviations={"Sgl": "Single", "Mlt": "Malt", "Whisky": "Whiskey"})
        result = run(label("Sgl Mlt Scotch Whisky"), "Single Malt Scotch Whisky", beverage=beverage)
        assert result.verdict is Verdict.MATCH

    def test_readable_expansion_keeps_real_letters(self) -> None:
        """What the agent is shown must be what is printed, not its fold class."""
        table = ct._table_for({"Sgl": "Single", "WHISKY": "WHISKEY"})
        assert ct._readable_expansion("Sgl Wh1sky", table) == "SINGLE WHISKEY"
        # Folding the readable form lands exactly on the scored form.
        assert canonicalize("SINGLE WHISKEY") == ct._expand(canonicalize("Sgl Wh1sky"), table).text

    def test_an_exact_copy_is_preferred_to_a_misread_one(self) -> None:
        """A cleaner-confidence misread elsewhere must not flag a compliant label.

        The misread fine-print copy scores higher (0.97 at full confidence)
        than the display line read at 0.80 confidence (0.94). Both clear the
        threshold; the one with no residual is the one to report.
        """
        lines = label(
            (KSBW, {"confidence": 0.80}),
            ("Kentucky Straight Burbon Whiskey", {"confidence": 1.0, "top": 1200.0}),
        )
        result = run(lines, KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.found == KSBW

    # --- the budget is rule-set data ---------------------------------------

    def test_budget_is_read_from_the_rule_set(self) -> None:
        """Allow one edit and "Burbon" matches again — proof the value is read."""
        lines = label("Kentucky Straight Burbon Whiskey")
        assert run(lines, KSBW).verdict is Verdict.REVIEW

        result = run(lines, KSBW, ruleset=tuned_names(max_residual_character_edits=1))
        assert result.verdict is Verdict.MATCH

    def test_budget_is_a_count_not_a_ratio(self) -> None:
        """Two wrong letters exceed a budget of one, however long the value."""
        lines = label("Kentucky Straight Burbon Whiskee")
        result = run(lines, KSBW, ruleset=tuned_names(max_residual_character_edits=1))

        assert result.verdict is Verdict.REVIEW
        assert "differ by 2 characters" in result.reason

    # --- the other stages are unchanged -------------------------------------

    def test_vocabulary_refinement_is_not_reworded(self) -> None:
        """A known, different designation keeps its own finding and wording."""
        result = run(label(KSBW), "Straight Bourbon Whiskey")

        assert result.strategy is Strategy.VOCABULARY
        assert "more or less specific" in result.reason
        assert "differ by" not in result.reason

    def test_vocabulary_mismatch_is_unchanged(self) -> None:
        result = run(label("Gin"), "Vodka")

        assert result.verdict is Verdict.MISMATCH
        assert result.strategy is Strategy.VOCABULARY
        assert "differ by" not in result.reason

    def test_positional_stage_is_unaffected(self) -> None:
        """Already always REVIEW, with its own explanation."""
        result = run(label(GARBLED), KSBW)

        assert result.strategy is Strategy.POSITIONAL
        assert "beside the brand name" in result.reason
        assert "differ by" not in result.reason


# ---------------------------------------------------------------------------
# Stage 2 — positional prior
# ---------------------------------------------------------------------------


#: A badly garbled, low-confidence read of KSBW. Its composite score falls
#: below the class/type mismatch floor (0.55), so stage 1 finds nothing; its
#: text-only similarity (~0.71) clears the positional floor.
GARBLED = ("Kentuc Straig Bour", {"confidence": 0.2})


class TestPositional:
    def test_rescues_a_weak_read_below_the_brand(self) -> None:
        lines = label(GARBLED)
        result = run(lines, KSBW)

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.POSITIONAL
        assert result.found == "Kentuc Straig Bour"
        assert result.bbox is not None and result.bbox.top == lines[1].bbox.top
        assert "beside the brand name" in result.reason
        assert "low confidence" in result.reason

    def test_same_read_without_a_brand_anchor_is_not_found(self) -> None:
        """Stage 2 is skipped when the brand was not located."""
        result = run(label(GARBLED), KSBW, brand_line=None)

        assert result.strategy is Strategy.NOT_FOUND
        assert result.verdict is Verdict.REVIEW

    def test_same_read_far_from_the_brand_is_not_found(self) -> None:
        lines = label(("Kentuc Straig Bour", {"confidence": 0.2, "top": 1200.0}))
        result = run(lines, KSBW)
        assert result.strategy is Strategy.NOT_FOUND

    def test_same_read_beside_brand_but_in_another_column_is_not_found(self) -> None:
        lines = label(("Kentuc Straig Bour", {"confidence": 0.2, "left": 900.0}))
        result = run(lines, KSBW)
        assert result.strategy is Strategy.NOT_FOUND

    def test_ordinary_decorative_text_near_the_brand_is_not_reported(self) -> None:
        """ "Small Batch" beside the brand is not a guess at the class/type."""
        result = run(label("Small Batch", "Single Barrel"), KSBW)
        assert result.strategy is Strategy.NOT_FOUND

    def test_prefers_the_read_beside_the_brand_to_a_stronger_one_elsewhere(self) -> None:
        """A review-band fine-print mention far away loses to the brand's neighbour."""
        lines = label(GARBLED, ("Kntcky Strght Brbn Whsky", {"top": 1200.0, "height": 12.0}))
        result = run(lines, KSBW)

        assert result.strategy is Strategy.POSITIONAL
        assert result.found == "Kentuc Straig Bour"

    def test_review_band_read_beside_the_brand_is_reported_as_anchored(self) -> None:
        """When stage 1's own weak read *is* beside the brand, stage 1 gets credit."""
        result = run(label("Kntcky Strght Brbn Whsky"), KSBW)

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.ANCHORED
        assert "Close but not an exact match" in result.reason


# ---------------------------------------------------------------------------
# Stage 3 — vocabulary
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_reports_a_more_specific_designation_than_the_application(self) -> None:
        """The design doc's example, and why this stage exists.

        "Straight Bourbon Whiskey" is a sub-span of the label's line and scores
        1.0 against the application — but the label says more than that, and
        silently calling it a MATCH would hide a real disagreement.
        """
        result = run(label(KSBW), "Straight Bourbon Whiskey")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == KSBW
        assert "Straight Bourbon Whiskey" in result.reason
        assert "more or less specific" in result.reason

    def test_reports_a_less_specific_designation_than_the_application(self) -> None:
        result = run(label("Bourbon Whiskey"), "Straight Bourbon Whiskey")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == "Bourbon Whiskey"

    def test_different_designation_in_the_same_family_is_review(self) -> None:
        """Rye against bourbon: reported plainly, but not asserted."""
        result = run(label("Straight Bourbon Whiskey"), "Straight Rye Whiskey")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == "Straight Bourbon Whiskey"
        assert "Straight Rye Whiskey" in result.reason

    def test_reports_the_most_specific_designation_on_the_line(self) -> None:
        """Sub-spans of the KSBW line are exact hits too; report the whole line."""
        result = run(label(KSBW), "Straight Rye Whiskey")
        assert result.found == KSBW

    def test_class_and_type_of_a_class_are_not_asserted_as_conflicting(self) -> None:
        """Cognac *is* a brandy; the flat vocabulary cannot say so. REVIEW."""
        result = run(label("Cognac"), "Brandy")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY

    def test_synonymous_classes_are_not_asserted_as_conflicting(self) -> None:
        """Liqueur and cordial are one class in 27 CFR 5.150."""
        result = run(label("Cordial"), "Liqueur")
        assert result.verdict is Verdict.REVIEW

    def test_score_reflects_agreement_with_the_application(self) -> None:
        """Not the 1.0 vocabulary similarity — batch triage sorts on this."""
        result = run(label("Gin"), "Vodka")
        assert result.score < 0.2


class TestClearlyDifferentDesignation:
    """Gin against Vodka — the one case the tool asserts.

    The first test is the MISMATCH. Every other test removes exactly one piece
    of corroboration and shows the verdict falls back to REVIEW with the same
    finding, so a regression in any single guard is caught on its own.
    """

    def test_vodka_versus_gin_is_a_mismatch(self) -> None:
        result = run(label("Gin"), "Vodka")

        assert result.verdict is Verdict.MISMATCH
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == "Gin"
        assert result.bbox is not None
        assert "different class/type designation" in result.reason

    def test_rum_versus_tequila_is_a_mismatch(self) -> None:
        assert run(label("Gold Rum"), "Blanco Tequila").verdict is Verdict.MISMATCH

    def test_without_a_brand_anchor_it_is_review(self) -> None:
        result = run(label("Gin"), "Vodka", brand_line=None)

        assert result.verdict is Verdict.REVIEW
        assert result.found == "Gin"
        assert "brand name was not located" in result.reason

    def test_away_from_the_brand_it_is_review(self) -> None:
        """A stray "gin" in fine print is not the class/type statement."""
        result = run(label(("Gin", {"top": 1200.0})), "Vodka")

        assert result.verdict is Verdict.REVIEW
        assert "not beside the brand name" in result.reason

    def test_low_confidence_read_is_review(self) -> None:
        result = run(label(("Gin", {"confidence": 0.6})), "Vodka")

        assert result.verdict is Verdict.REVIEW
        assert "limited confidence" in result.reason

    def test_competing_read_of_the_expected_value_is_review(self) -> None:
        """Both on the label: the tool cannot tell which is the class/type."""
        lines = label("Gin", ("Vodka", {"top": 1200.0, "confidence": 0.3}))
        result = run(lines, "Vodka")

        assert result.verdict is not Verdict.MISMATCH

    def test_unclassifiable_designation_is_review(self) -> None:
        """Unknown family → the tool does not decide."""
        result = run(label("Gin"), "Aquavit")

        assert result.verdict is Verdict.REVIEW
        assert "cannot classify" in result.reason


class TestThresholdsComeFromTheRuleSet:
    """The three assertion thresholds are rule-set data, not module constants.

    Each test changes one value in `tuning.class_type` and shows the verdict
    follows it. A future refactor that reintroduced a hardcoded threshold —
    or a default that shadowed the rule set — would leave the verdict
    unchanged and fail here.

    This matters beyond tidiness: every result records `ruleset_version`, and
    that is only a reproducible record of how a verdict was reached if the
    numbers that decided it live in the versioned file.
    """

    def test_no_shadowing_constants_remain_in_the_module(self) -> None:
        for name in (
            "POSITIONAL_MIN_SIMILARITY",
            "VOCABULARY_MISMATCH_MIN_SIMILARITY",
            "MISMATCH_MIN_CONFIDENCE",
            "POSITIONAL_REACH",
        ):
            assert not hasattr(ct, name), f"{name} shadows ruleset.tuning.class_type"

    def test_positional_reach_is_read_from_the_rule_set(self) -> None:
        """Narrow the reach and the line below the brand stops being adjacent."""
        assert run(label(GARBLED), KSBW).strategy is Strategy.POSITIONAL

        result = run(label(GARBLED), KSBW, ruleset=tuned(positional_reach_brand_heights=0.1))
        assert result.strategy is Strategy.NOT_FOUND

    def test_widening_the_reach_brings_distant_text_into_position(self) -> None:
        """The other direction, on the condition that makes the reach policy.

        "Gin" at the foot of the panel is held back from MISMATCH by
        `not_in_position`. Widen the reach until it counts as beside the brand
        and the same reading is asserted — which is why this value belongs in
        the rule set rather than in the module.
        """
        lines = label(("Gin", {"top": 1200.0}))

        held_back = run(lines, "Vodka")
        assert held_back.verdict is Verdict.REVIEW
        assert "not beside the brand name" in held_back.reason

        asserted = run(lines, "Vodka", ruleset=tuned(positional_reach_brand_heights=30.0))
        assert asserted.verdict is Verdict.MISMATCH

    def test_positional_floor_is_read_from_the_rule_set(self) -> None:
        """Raise it above the garbled read and the positional rescue stops."""
        assert run(label(GARBLED), KSBW).strategy is Strategy.POSITIONAL

        result = run(label(GARBLED), KSBW, ruleset=tuned(positional_min_similarity=0.99))
        assert result.strategy is Strategy.NOT_FOUND

    def test_lowering_the_positional_floor_admits_weaker_text(self) -> None:
        """The other direction, so the test cannot pass by the read vanishing."""
        assert run(label("Small Batch"), KSBW).strategy is Strategy.NOT_FOUND

        result = run(label("Small Batch"), KSBW, ruleset=tuned(positional_min_similarity=0.30))
        assert result.strategy is Strategy.POSITIONAL
        assert result.found == "Small Batch"

    def test_near_exact_read_threshold_is_read_from_the_rule_set(self) -> None:
        """ "Tequilla" reads 0.93 against "Tequila" — near-exact, but not at 0.95."""
        lines = label("Tequilla")

        held_back = run(lines, "Gin")
        assert held_back.verdict is Verdict.REVIEW
        assert "close to, but not exactly" in held_back.reason

        asserted = run(lines, "Gin", ruleset=tuned(vocabulary_mismatch_min_similarity=0.90))
        assert asserted.verdict is Verdict.MISMATCH

    def test_mismatch_confidence_floor_is_read_from_the_rule_set(self) -> None:
        lines = label(("Gin", {"confidence": 0.7}))

        held_back = run(lines, "Vodka")
        assert held_back.verdict is Verdict.REVIEW
        assert "limited confidence" in held_back.reason

        asserted = run(lines, "Vodka", ruleset=tuned(mismatch_min_confidence=0.50))
        assert asserted.verdict is Verdict.MISMATCH

    def test_ocr_confidence_floor_never_falls_below_the_rule_set_default(self) -> None:
        """Tuning may raise the bar for asserting; it may not lower it past
        the confidence at which the tool trusts a read at all."""
        result = run(
            label(("Gin", {"confidence": 0.4})), "Vodka", ruleset=tuned(mismatch_min_confidence=0.0)
        )
        assert result.verdict is Verdict.REVIEW


class TestFamiliesComeFromTheRuleSet:
    """Which designations conflict is rule-set data, not module data.

    The families decide the one question on which this strategy asserts that a
    label is wrong, so they belong in the versioned file a verdict can be
    reproduced against.
    """

    def test_no_hardcoded_family_table_remains_in_the_module(self) -> None:
        assert not hasattr(ct, "_DESIGNATION_FAMILIES")

    def test_empty_mapping_downgrades_a_mismatch_to_review(self) -> None:
        """A beverage class with no families declared never asserts.

        Under-asserting is the safe failure for an incomplete list: the agent
        still sees "label reads Gin", without the tool deciding.
        """
        result = run(label("Gin"), "Vodka", beverage=_beverage(class_type_families={}))

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == "Gin"
        assert "cannot classify" in result.reason

    def test_putting_two_designations_in_one_family_withdraws_the_assertion(self) -> None:
        """Change the data, change the verdict — the proof it is read."""
        together = _beverage(class_type_families={"neutral": ("Vodka", "Gin")})
        result = run(label("Gin"), "Vodka", beverage=together)

        assert result.verdict is Verdict.REVIEW
        assert "same general class" in result.reason

    def test_splitting_a_family_permits_an_assertion(self) -> None:
        """The other direction: Cognac is asserted against Brandy if the rule set says so.

        The shipped rule set puts both in `brandy`, because Cognac *is* a
        brandy under 27 CFR 5.143 and the tool should not assert. Separate
        them and the same reading becomes a MISMATCH — nothing else changed.
        """
        assert run(label("Cognac"), "Brandy").verdict is Verdict.REVIEW

        split = _beverage(class_type_families={"cognac": ("Cognac",), "brandy": ("Brandy",)})
        assert run(label("Cognac"), "Brandy", beverage=split).verdict is Verdict.MISMATCH

    def test_a_textually_close_designation_is_never_asserted(self) -> None:
        """Families permit an assertion; they do not force one.

        "Straight Bourbon Whiskey" still scores 0.81 against "Straight Rye
        Whiskey", inside the field's REVIEW band. Asserting a MISMATCH on a
        score the bands call reviewable would contradict the rule set, so the
        `close_to_expected` blocker holds even when the families differ.
        """
        split = _beverage(class_type_families={"bourbon": ("Bourbon",), "rye": ("Rye",)})
        result = run(label("Straight Bourbon Whiskey"), "Straight Rye Whiskey", beverage=split)

        assert result.verdict is Verdict.REVIEW
        assert "may be a misread" in result.reason

    def test_family_terms_are_abbreviation_expanded(self) -> None:
        """A family listing only "Whisky" still matches an expanded WHISKEY.

        The rule set's abbreviations rewrite WHISKY to WHISKEY before anything
        is compared. If family terms were only canonicalised, a list written
        with the other legal spelling would silently match nothing and the
        designation would fall back to "unknown family".
        """
        beverage = _beverage(class_type_families={"whisky": ("Whisky",), "gin": ("Gin",)})
        result = run(label("Gin"), "Straight Bourbon Whiskey", beverage=beverage)

        assert result.verdict is Verdict.MISMATCH


class TestImageQualityOnlyWithdrawsAssertions:
    """Quality may take an assertion away, never add one."""

    def test_degraded_image_downgrades_a_mismatch_to_review(self) -> None:
        poor = ImageQuality(usable=True, score=0.4)
        result = run(label("Gin"), "Vodka", quality=poor)

        assert result.verdict is Verdict.REVIEW
        assert result.found == "Gin"
        assert "image quality is poor" in result.reason

    def test_good_image_leaves_the_verdict_alone(self) -> None:
        good = ImageQuality(usable=True, score=0.95)
        assert run(label("Gin"), "Vodka", quality=good).verdict is Verdict.MISMATCH

    def test_good_image_does_not_create_an_assertion(self) -> None:
        """A blocker that still applies is not overridden by a clean photo."""
        good = ImageQuality(usable=True, score=0.95)
        result = run(label("Cognac"), "Brandy", quality=good)
        assert result.verdict is Verdict.REVIEW


class TestLetterspacedDisplayType:
    """PP-OCR returns letterspaced display type as one token.

    `rules.comparators.text_similarity` handles that for scoring; the
    vocabulary stage's prefilter has to do the same or the designation is
    discarded before it is ever scored.
    """

    def test_single_token_designation_matches(self) -> None:
        result = run(label("KentuckyStraightBourbonWhiskey"), KSBW)
        assert result.verdict is Verdict.MATCH

    def test_single_token_designation_reaches_the_vocabulary_stage(self) -> None:
        result = run(label("KentuckyStraightBourbonWhiskey"), "Vodka")

        assert result.strategy is Strategy.VOCABULARY
        assert result.verdict is Verdict.MISMATCH


# ---------------------------------------------------------------------------
# Not found, and guards against the wrong text
# ---------------------------------------------------------------------------


class TestNotFound:
    def test_nothing_present_is_review_not_mismatch(self) -> None:
        result = run(label("750 mL"), KSBW)

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.NOT_FOUND
        assert result.bbox is None

    def test_empty_pool_is_not_found(self) -> None:
        result = run(make_lines([BRAND]), KSBW)
        assert result.strategy is Strategy.NOT_FOUND

    def test_blank_application_value_says_so(self) -> None:
        """Not "could not locate" — the tool had nothing to look for."""
        result = run(label(KSBW), "  ")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.NOT_FOUND
        assert "no class/type designation" in result.reason


class TestWrongTextGuards:
    def test_brand_text_is_not_read_as_the_class_type(self) -> None:
        """Brand "Straight Bourbon" resembles the designation; it is not it.

        The pool is deliberately *not* masked here, so this exercises the
        strategy's own guard rather than the engine's masking.
        """
        lines = make_lines([("Straight Bourbon", {"height": 48.0}), "750 mL"])
        result = run(lines, "Straight Bourbon Whiskey", mask_brand=False)

        assert result.verdict is not Verdict.MATCH
        assert result.bbox is None or not _as_bbox(result).overlaps(lines[0].bbox, 0.25)

    def test_span_straddling_brand_and_next_line_is_excluded(self) -> None:
        """Same-size brand and class/type lines can merge into one span.

        The engine's 0.5 mask lets a straddling span through when less than
        half of it is brand. The strategy's stricter guard removes it, so the
        highlighted region is the class/type line alone.
        """
        lines = make_lines(["Old Rye Whiskey Co", KSBW])
        result = run(lines, KSBW)

        assert result.verdict is Verdict.MATCH
        assert result.found == KSBW
        assert not _as_bbox(result).overlaps(lines[0].bbox, 0.1)

    def test_bottler_statement_is_not_read_as_the_class_type(self) -> None:
        """ "Distilled ... Kentucky" shares words with the designation."""
        lines = label("Distilled and bottled by Old Tom Distillery Bardstown Kentucky")
        result = run(lines, KSBW)

        assert result.strategy is Strategy.NOT_FOUND

    def test_distilled_in_bottler_line_is_not_a_vocabulary_hit(self) -> None:
        """ "Distilled" alone must not be reported as "Distilled Gin"."""
        lines = label("Distilled and bottled by Old Tom Distillery")
        result = run(lines, "Vodka")
        assert result.strategy is Strategy.NOT_FOUND


# ---------------------------------------------------------------------------
# End to end, through the engine
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_sample_label_resolves_to_match(self, sample_label_lines) -> None:
        result = verify(extract(sample_label_lines), RECORD).field("class_type")

        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED
        assert result.found == KSBW
        assert "does not yet check" not in result.reason

    def test_sample_label_against_a_less_specific_application(self, sample_label_lines) -> None:
        record = RECORD.model_copy(update={"class_type": "Straight Bourbon Whiskey"})
        result = verify(extract(sample_label_lines), record).field("class_type")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.VOCABULARY
        assert result.found == KSBW

    def test_sample_label_against_a_different_class(self, sample_label_lines) -> None:
        record = RECORD.model_copy(update={"class_type": "Vodka"})
        result = verify(extract(sample_label_lines), record).field("class_type")

        # Whiskey against vodka, beside the brand, read clearly.
        assert result.verdict is Verdict.MISMATCH
        assert result.found == KSBW

    def test_brand_resolved_by_the_engine_is_not_the_class_type(self) -> None:
        """Brand "OLD RYE WHISKEY" must not satisfy class/type "Rye Whiskey"."""
        lines = make_lines([("OLD RYE WHISKEY", {"height": 48.0}), "750 mL"])
        record = RECORD.model_copy(
            update={"brand_name": "OLD RYE WHISKEY", "class_type": "Rye Whiskey"}
        )
        result = verify(extract(lines), record)

        assert result.field("brand_name").verdict is Verdict.MATCH
        assert result.field("class_type").strategy is Strategy.NOT_FOUND

    def test_deterministic(self, sample_label_lines) -> None:
        record = RECORD.model_copy(update={"class_type": "Straight Rye Whiskey"})
        runs = [verify(extract(sample_label_lines), record).field("class_type") for _ in range(3)]
        assert len({(r.verdict, r.strategy, r.found, r.score, r.reason) for r in runs}) == 1
