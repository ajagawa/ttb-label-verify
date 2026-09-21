"""Tests for text normalisation.

The two traps documented in extraction/normalize.py are the reason this module
has dedicated tests rather than being covered incidentally. Both failure modes
are silent: the system keeps working and produces wrong compliance findings.
"""

from __future__ import annotations

import pytest

from extraction.normalize import (
    canonicalize,
    canonicalize_numeric,
    collapse_for_anchor,
    is_all_caps,
    word_tokens,
)


class TestCanonicalizeConvergence:
    """Variants that should compare equal after canonicalisation."""

    @pytest.mark.parametrize(
        ("left", "right", "reason"),
        [
            ("STONE'S THROW", "Stone's Throw", "case only — Dave Morrison's example"),
            ("Stone's Throw", "Stones Throw", "apostrophe present vs absent"),
            ("STONE’S THROW", "STONE'S THROW", "smart vs straight apostrophe"),
            ("Old  Tom   Distillery", "Old Tom Distillery", "whitespace runs"),
            ("CRÈME DE CACAO", "CREME DE CACAO", "combining accents"),
            ("Alc./Vol.", "Alc Vol", "punctuation as separator"),
            ("CAFÉ BRAND", "CAFE BRAND", "accented capital"),
        ],
    )
    def test_variants_converge(self, left: str, right: str, reason: str) -> None:
        assert canonicalize(left) == canonicalize(right), reason

    def test_genuinely_different_names_stay_different(self) -> None:
        """Canonicalisation must not collapse distinct brands into each other.

        Fuzzy scoring tolerates near-misses; that is the comparator's job, with
        a threshold that can be tuned. Normalisation collapsing them outright
        would remove the decision from the tunable layer entirely.
        """
        assert canonicalize("OLD TOM DISTILLERY") != canonicalize("NEW TOM DISTILLERY")


class TestTrapTwoNumericFolding:
    """Character folding must not rewrite real numbers."""

    @pytest.mark.parametrize(
        ("text", "must_contain"),
        [
            ("750 mL", "750"),
            ("45.5% ABV", "45.5"),
            ("1792 Ridgemont", "1792"),
            ("Old No. 7", "7"),
            ("100 Proof", "100"),
        ],
    )
    def test_numeric_tokens_survive_folding(self, text: str, must_contain: str) -> None:
        """A purely numeric token passes through the confusion table untouched.

        Without the guard, "750" folds to "7SO" and "45.5" to "4S.S". Matching
        still works because both sides fold alike, but every diagnostic becomes
        unreadable and any downstream numeric parse gets nonsense.
        """
        assert must_contain in canonicalize(text)

    def test_numeric_canonicalisation_does_no_folding_at_all(self) -> None:
        result = canonicalize_numeric("45% Alc./Vol. (90 Proof)")
        assert "45" in result
        assert "90" in result
        assert "S" not in result.replace("PROOF", "")  # no 5 -> S anywhere

    def test_abv_discrepancy_is_not_normalised_away(self) -> None:
        """The core safety property for numeric fields.

        A 46 on the label against a 45 in the application must remain visibly
        different after normalisation. If folding closed this gap, the tool
        would fail precisely on the violation it exists to catch.
        """
        assert canonicalize_numeric("46%") != canonicalize_numeric("45%")


class TestTrapOneCapitalisation:
    """Formatting checks read raw text; canonical text destroys the signal."""

    def test_all_caps_detected_on_raw_text(self) -> None:
        assert is_all_caps("GOVERNMENT WARNING") is True

    def test_title_case_violation_detected_on_raw_text(self) -> None:
        """Jenny Park's rejected label: title case where caps are required."""
        assert is_all_caps("Government Warning") is False

    def test_canonical_text_would_hide_the_violation(self) -> None:
        """Demonstrates the trap rather than guarding against it.

        This asserts that the *wrong* input produces the *wrong* answer, which
        is the point: it documents why the check must take raw text, and it
        fails loudly if someone ever makes canonicalize() stop uppercasing
        without revisiting the checks that depend on the current behaviour.
        """
        assert is_all_caps(canonicalize("Government Warning")) is True

    @pytest.mark.parametrize("text", ["123", "", "45%", "   "])
    def test_uncased_text_is_not_all_caps(self, text: str) -> None:
        """Absence of letters is not satisfaction of a capitalisation rule.

        Returning True here would let an empty or numeric extraction silently
        pass § 16.22(a)(2).
        """
        assert is_all_caps(text) is False


class TestAnchorCollapsing:
    """The anchor key must not depend on the input's casing."""

    def test_caps_and_title_case_produce_the_same_anchor(self) -> None:
        """Regression test for a bug found during the build.

        An earlier normaliser folded "rn" -> "m" on lowercase text before
        uppercasing. That made "GOVERNMENT WARNING" and "Government Warning"
        canonicalise differently, so the warning statement would anchor on
        compliant labels and fail to anchor on exactly the title-case violation
        the tool is meant to catch — a silent false negative on the highest
        priority check in the system.
        """
        assert collapse_for_anchor("GOVERNMENT WARNING") == collapse_for_anchor(
            "Government Warning"
        )

    def test_anchor_tolerates_segmentation_differences(self) -> None:
        """OCR may return the phrase as one token, two, or split across lines."""
        variants = [
            "GOVERNMENT WARNING",
            "GOVERNMENTWARNING",
            "GOVERNMENT  WARNING",
            "GOVERNMENT\nWARNING",
        ]
        anchors = {collapse_for_anchor(v) for v in variants}
        assert len(anchors) == 1


class TestWordTokens:
    def test_tokenises_canonical_form(self) -> None:
        assert word_tokens("Government Warning:") == ["GOVERNMENT", "WARNING"]

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_input_yields_no_tokens(self, text: str) -> None:
        assert word_tokens(text) == []
