"""Typed loader for the versioned rule set.

The rule set is data (rules/ttb-v1.yaml). This module gives it a schema, so
that a malformed or incomplete rule file fails loudly at startup rather than
producing a silently wrong verdict on the first upload.

Why pydantic rather than reading the YAML into dicts: a rule set that ships a
typo in a threshold key would otherwise fall back to a default and keep
running. In a compliance tool the difference between "threshold 0.92" and
"threshold silently defaulted because someone wrote `treshold`" is the
difference between a defensible finding and an indefensible one.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

RULES_DIR = Path(__file__).parent
DEFAULT_RULESET = "ttb-v1"


class _Strict(BaseModel):
    """Base for every rule-set model: unknown keys are an error, not a shrug.

    This is what turns a misspelled key into a startup failure. It is also why
    adding a field to the YAML requires adding it here — a deliberate friction,
    since every key in the rule set influences a compliance verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(_Strict):
    """Score bands for a field's verdict.

    Between `mismatch` and `match` is the REVIEW band. The gap is wide on
    purpose: the cost of a false flag (about twenty seconds of an agent's
    attention) is not comparable to the cost of a missed violation.
    """

    match: float = Field(ge=0.0, le=1.0)
    mismatch: float = Field(ge=0.0, le=1.0)

    @field_validator("mismatch")
    @classmethod
    def mismatch_below_match(cls, value: float, info) -> float:
        match_value = info.data.get("match")
        if match_value is not None and value >= match_value:
            raise ValueError(
                f"mismatch threshold ({value}) must be below match threshold ({match_value}); "
                "otherwise the REVIEW band is empty and every field is forced to a verdict"
            )
        return value


class Defaults(_Strict):
    thresholds: Thresholds
    min_ocr_confidence: float = Field(ge=0.0, le=1.0)
    good_image_quality: float = Field(ge=0.0, le=1.0)


class NameTuning(_Strict):
    max_residual_character_edits: int = Field(ge=0)


class NumericTuning(_Strict):
    mismatch_min_ocr_confidence: float = Field(ge=0.0, le=1.0)
    wording_min_similarity: float = Field(ge=0.0, le=1.0)


class ClassTypeTuning(_Strict):
    positional_min_similarity: float = Field(ge=0.0, le=1.0)
    vocabulary_mismatch_min_similarity: float = Field(ge=0.0, le=1.0)
    mismatch_min_confidence: float = Field(ge=0.0, le=1.0)
    positional_reach_brand_heights: float = Field(gt=0.0)


class WarningTuning(_Strict):
    clean_read_min_precision: float = Field(ge=0.0, le=1.0)
    max_sporadic_lowercase: int = Field(ge=0)
    min_case_style_run: int = Field(ge=1)
    noise_min_token_length: int = Field(ge=1)
    ragged_right_tolerance: float = Field(ge=0.0, le=1.0)
    line_gap_slack_ratio: float = Field(ge=0.0)
    max_word_gap_ratio: float = Field(ge=0.0)
    indent_tolerance_ratio: float = Field(ge=0.0)


class Tuning(_Strict):
    """Thresholds that decide whether the tool asserts a violation.

    Typed per strategy rather than a free-form mapping so that a misspelled key
    fails at startup. A tuning value that silently falls back to a default is
    exactly the failure the strict schema exists to prevent: the recorded rule
    set version would no longer describe the thresholds that actually ran.
    """

    names: NameTuning
    numeric: NumericTuning
    class_type: ClassTypeTuning
    warning: WarningTuning


FieldStrategy = Literal["warning", "alcohol_content", "net_contents", "anchored", "class_type"]


class FieldRule(_Strict):
    """One checked field.

    `order` drives resolution sequence (most precise first, so each resolved
    region can be masked out). `display_order` drives the result screen, which
    follows the printed checklist order the agents already use.
    """

    id: str
    label: str
    order: int
    display_order: int
    required: bool
    strategy: FieldStrategy
    citation: str
    citation_url: str | None = None
    thresholds: Thresholds | None = None
    numeric: bool = False
    length_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class TypeSizeRule(_Strict):
    max_volume_ml: float | None
    min_height_mm: float
    max_chars_per_inch: int


class WarningCheck(_Strict):
    """One § 16.21 / § 16.22 check.

    `advisory` is the load-bearing field. An advisory check is a measurement
    reported to the agent; it never produces a MISMATCH on its own. Bold
    detection, contrast and separation are all advisory because they are
    estimates from pixels, and a tool that asserts a violation on an estimate
    misleads the person trusting it.
    """

    id: str
    citation: str
    advisory: bool
    description: str
    min_contrast_ratio: float | None = None
    requires: tuple[str, ...] = ()
    minimum_mm: tuple[TypeSizeRule, ...] = ()


class WarningAnchor(_Strict):
    header_min_score: float = Field(ge=0.0, le=1.0)
    tail_min_score: float = Field(ge=0.0, le=1.0)
    tail_phrases: tuple[str, ...]


class GovernmentWarningRule(_Strict):
    statutory_text: str
    header_text: str
    anchor: WarningAnchor
    checks: tuple[WarningCheck, ...]

    @field_validator("statutory_text", "header_text")
    @classmethod
    def collapse_yaml_folding(cls, value: str) -> str:
        """YAML folded scalars introduce newlines; the statute is one paragraph."""
        return " ".join(value.split())


class AlcoholContentRule(_Strict):
    tolerance_abv: float = Field(ge=0.0)
    proof_is_double_abv: bool
    proof_tolerance: float = Field(ge=0.0)


class NetContentsRule(_Strict):
    standards_of_fill_ml: tuple[float, ...]
    tolerance_ml: float = Field(ge=0.0)


class BeverageClass(_Strict):
    label: str
    alcohol_content: AlcoholContentRule
    net_contents: NetContentsRule
    class_type_vocabulary: tuple[str, ...]
    abbreviations: dict[str, str] = Field(default_factory=dict)

    #: Coarse families of designations, used only to permit a class/type
    #: MISMATCH: a reading is asserted as wrong only when it and the
    #: application's value fall in families that do not overlap.
    #:
    #: Optional, and an empty mapping means no class/type MISMATCH is ever
    #: asserted. That is the safe default for a beverage class someone adds
    #: later: an incomplete list under-asserts rather than over-asserts.
    class_type_families: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("class_type_families")
    @classmethod
    def families_are_disjoint(cls, value: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        """Reject empty terms and terms claimed by two families.

        A designation that lands in two families makes the disjointness test
        depend on which one matched first, which would make a MISMATCH verdict
        depend on dictionary ordering — non-reproducible, and in the one place
        where the tool asserts that a label is wrong.
        """
        seen: dict[str, str] = {}
        for family, terms in value.items():
            for term in terms:
                key = " ".join(term.split()).upper()
                if not key:
                    raise ValueError(f"family {family!r} contains an empty term")
                if key in seen and seen[key] != family:
                    raise ValueError(
                        f"term {term!r} appears in both {seen[key]!r} and {family!r}; "
                        "families must be disjoint or a MISMATCH depends on match order"
                    )
                seen[key] = family
        return value


class RuleSet(_Strict):
    """A complete, versioned rule set.

    Every verification result carries `version`. That is what makes a verdict
    reproducible after the rules change — and they will change: TTB Notices 237
    and 238 propose mandatory Alcohol Facts and allergen disclosures with a
    five-year compliance window after any final rule.
    """

    version: str
    effective_date: str
    description: str
    defaults: Defaults
    tuning: Tuning
    fields: tuple[FieldRule, ...]
    government_warning: GovernmentWarningRule
    beverage_classes: dict[str, BeverageClass]

    @field_validator("fields")
    @classmethod
    def orders_are_unique(cls, value: tuple[FieldRule, ...]) -> tuple[FieldRule, ...]:
        """Ambiguous resolution order would make results non-reproducible.

        Two fields sharing an `order` resolve in whatever sequence the sort
        happens to produce, which means the masking step could remove a region
        before or after a competing field runs — and the same inputs could then
        yield different verdicts across runs. That would quietly break the
        auditability property the whole architecture exists to provide.
        """
        for attribute in ("order", "display_order", "id"):
            seen = [getattr(f, attribute) for f in value]
            duplicates = {v for v in seen if seen.count(v) > 1}
            if duplicates:
                raise ValueError(f"duplicate field {attribute}: {sorted(duplicates)}")
        return value

    # -- lookups ------------------------------------------------------------

    def resolution_order(self) -> list[FieldRule]:
        """Fields most-precise-first, for assignment and masking."""
        return sorted(self.fields, key=lambda f: f.order)

    def display_order(self) -> list[FieldRule]:
        """Fields in printed-checklist order, for the result screen."""
        return sorted(self.fields, key=lambda f: f.display_order)

    def field(self, field_id: str) -> FieldRule:
        for rule in self.fields:
            if rule.id == field_id:
                return rule
        raise KeyError(f"no field rule with id {field_id!r} in rule set {self.version}")

    def thresholds_for(self, field_id: str) -> Thresholds:
        """Field-specific thresholds, falling back to the defaults."""
        return self.field(field_id).thresholds or self.defaults.thresholds

    def beverage_class(self, class_id: str) -> BeverageClass:
        try:
            return self.beverage_classes[class_id]
        except KeyError:
            supported = ", ".join(sorted(self.beverage_classes))
            raise UnsupportedBeverageClass(
                f"rule set {self.version} does not implement {class_id!r}. "
                f"Implemented: {supported}. Wine and malt beverage rule sets are "
                f"additive but are deliberately absent rather than stubbed — see "
                f"TRADEOFFS.md, 'Distilled spirits only'."
            ) from None


class UnsupportedBeverageClass(KeyError):
    """A beverage class this rule set does not implement.

    Raised rather than falling back to distilled spirits. Silently applying
    spirits rules to a wine label would produce confident wrong answers, which
    is the failure mode the three-state verdict design exists to avoid.
    """


@functools.lru_cache(maxsize=4)
def load_ruleset(version: str = DEFAULT_RULESET) -> RuleSet:
    """Load and validate a rule set by version.

    Cached: the rule set is immutable and parsing it per request would be pure
    waste against a 5-second budget.

    Args:
        version: Rule-set version, matching a `<version>.yaml` in rules/.

    Returns:
        A validated `RuleSet`.

    Raises:
        FileNotFoundError: when no rule file matches `version`.
        pydantic.ValidationError: when the file is malformed. Deliberately not
            caught — a rule set that does not validate must stop startup, not
            degrade to defaults.
    """
    path = RULES_DIR / f"{version}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in RULES_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no rule set {version!r} in {RULES_DIR}. Available: {available}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return RuleSet.model_validate(raw)
