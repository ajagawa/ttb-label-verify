"""Tests for rule set loading and validation.

The validation tests matter more than they look. A rule set that loads with a
misspelled key and silently falls back to a default produces verdicts that
cannot be defended — the recorded threshold would not be the threshold that
actually ran. These tests assert that such a file fails at startup instead.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rules.schema import (
    RuleSet,
    UnsupportedBeverageClass,
    load_ruleset,
)


@pytest.fixture(scope="module")
def ruleset() -> RuleSet:
    return load_ruleset()


class TestShippedRuleSet:
    def test_loads_and_validates(self, ruleset: RuleSet) -> None:
        assert ruleset.version == "ttb-v1"

    def test_all_five_fields_present(self, ruleset: RuleSet) -> None:
        assert {f.id for f in ruleset.fields} == {
            "brand_name",
            "class_type",
            "alcohol_content",
            "net_contents",
            "government_warning",
        }

    def test_warning_resolves_first(self, ruleset: RuleSet) -> None:
        """Masking depends on it.

        The warning statement contains "alcoholic beverages", which scores
        against class/type values containing "alcohol", and numerals that a
        loose net-contents pattern will latch onto. Resolving it first and
        masking its region removes most cross-field contention.
        """
        assert ruleset.resolution_order()[0].id == "government_warning"

    def test_display_order_matches_the_printed_checklist(self, ruleset: RuleSet) -> None:
        """Brand, ABV, contents, class/type, warning — the order agents use."""
        assert [f.id for f in ruleset.display_order()] == [
            "brand_name",
            "alcohol_content",
            "net_contents",
            "class_type",
            "government_warning",
        ]

    def test_numeric_fields_are_marked(self, ruleset: RuleSet) -> None:
        """The flag that routes a field away from fuzzy comparison."""
        assert ruleset.field("alcohol_content").numeric is True
        assert ruleset.field("net_contents").numeric is True
        assert ruleset.field("brand_name").numeric is False

    def test_statutory_text_is_a_single_paragraph(self, ruleset: RuleSet) -> None:
        """YAML folding must not leave newlines in the comparison target."""
        text = ruleset.government_warning.statutory_text
        assert "\n" not in text
        assert text.startswith("GOVERNMENT WARNING:")
        assert text.endswith("may cause health problems.")

    def test_binding_and_advisory_checks_are_distinguished(self, ruleset: RuleSet) -> None:
        """Pixel-derived measurements must never be binding.

        Bold, contrast, separation and type size are all estimates. A tool that
        asserts a violation from an estimate misleads the agent trusting it.
        """
        by_id = {c.id: c for c in ruleset.government_warning.checks}
        assert by_id["header_capitalisation"].advisory is False
        assert by_id["text_accuracy"].advisory is False
        for estimate in ("header_bold", "contrast", "separation", "type_size"):
            assert by_id[estimate].advisory is True, estimate

    def test_type_size_check_declares_its_prerequisite(self, ruleset: RuleSet) -> None:
        """Millimetres cannot be derived from pixels without a scale reference."""
        type_size = next(c for c in ruleset.government_warning.checks if c.id == "type_size")
        assert "label_width_mm" in type_size.requires
        assert len(type_size.minimum_mm) == 3

    def test_every_field_carries_a_citation(self, ruleset: RuleSet) -> None:
        """A finding without a rule reference is not auditable."""
        for rule in ruleset.fields:
            assert rule.citation.startswith("27 CFR"), rule.id


class TestBeverageClassHandling:
    def test_distilled_spirits_is_implemented(self, ruleset: RuleSet) -> None:
        spirits = ruleset.beverage_class("distilled_spirits")
        assert spirits.alcohol_content.tolerance_abv == pytest.approx(0.3)
        assert 750 in spirits.net_contents.standards_of_fill_ml

    def test_unimplemented_class_raises_rather_than_falling_back(self, ruleset: RuleSet) -> None:
        """Silently applying spirits rules to a wine label is the failure to avoid.

        It would produce confident wrong answers, which is worse than an honest
        refusal — and worse than the REVIEW verdict the three-state design uses
        everywhere else for uncertainty.
        """
        with pytest.raises(UnsupportedBeverageClass, match="does not implement"):
            ruleset.beverage_class("wine")

    def test_the_error_names_what_is_supported(self, ruleset: RuleSet) -> None:
        with pytest.raises(UnsupportedBeverageClass, match="distilled_spirits"):
            ruleset.beverage_class("malt_beverage")


class TestThresholds:
    def test_field_thresholds_override_defaults(self, ruleset: RuleSet) -> None:
        """Statutory text is fixed, so near-misses deserve a closer look."""
        assert (
            ruleset.thresholds_for("government_warning").match > ruleset.defaults.thresholds.match
        )

    def test_fields_without_overrides_get_the_defaults(self, ruleset: RuleSet) -> None:
        assert ruleset.thresholds_for("brand_name") == ruleset.defaults.thresholds

    def test_review_band_is_wide(self, ruleset: RuleSet) -> None:
        """Deliberate over-flagging: the band between verdicts is not narrow.

        A false flag costs about twenty seconds. A missed violation reaches the
        market. The band reflects that asymmetry rather than balancing error
        rates.
        """
        thresholds = ruleset.defaults.thresholds
        assert thresholds.match - thresholds.mismatch > 0.25


class TestValidationRejectsBadRuleSets:
    """A malformed rule set must fail at startup, never degrade to defaults."""

    def _write(self, tmp_path: Path, mutate) -> Path:
        source = yaml.safe_load((Path("rules") / "ttb-v1.yaml").read_text(encoding="utf-8"))
        mutate(source)
        target = tmp_path / "broken.yaml"
        target.write_text(yaml.safe_dump(source), encoding="utf-8")
        return target

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """A typo in a threshold key would otherwise silently use the default."""
        raw = yaml.safe_load(
            textwrap.dedent("""
                match: 0.92
                mismatch: 0.60
                treshold: 0.5
            """)
        )
        from rules.schema import Thresholds

        with pytest.raises(ValidationError):
            Thresholds.model_validate(raw)

    def test_inverted_thresholds_are_rejected(self) -> None:
        """An empty REVIEW band forces every field to a verdict."""
        from rules.schema import Thresholds

        with pytest.raises(ValidationError, match="REVIEW band is empty"):
            Thresholds.model_validate({"match": 0.5, "mismatch": 0.9})

    def test_duplicate_resolution_order_is_rejected(self, tmp_path: Path) -> None:
        """Ambiguous ordering makes masking — and therefore verdicts — unstable."""

        def duplicate_order(source: dict) -> None:
            source["fields"][1]["order"] = source["fields"][0]["order"]

        path = self._write(tmp_path, duplicate_order)
        with pytest.raises(ValidationError, match="duplicate field order"):
            RuleSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def test_missing_rule_set_names_the_alternatives(self) -> None:
        with pytest.raises(FileNotFoundError, match="ttb-v1"):
            load_ruleset("ttb-v99")


class TestCaching:
    def test_repeated_loads_return_the_same_object(self) -> None:
        """Re-parsing per request is waste against a 5-second budget."""
        assert load_ruleset() is load_ruleset()
