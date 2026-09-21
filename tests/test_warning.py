"""Tests for the government health warning strategy (rules/warning.py).

The warning is the highest-priority check in the product, so the tests are
organised around the ways it can lie to an agent:

*   **Passing a non-compliant label.** The title-case header is the canonical
    case — Jenny Park's rejected label from the discovery notes — and the one
    trap 1 in extraction/normalize.py exists to prevent. If the
    capitalisation check ever reads canonical text, `TestHeaderCapitalisation`
    fails.
*   **Accusing a compliant label.** OCR noise must not become a MISMATCH;
    missing text the engine simply failed to read must not become a MISMATCH;
    advisory measurements must never become a MISMATCH.
*   **Saying "passed" about something it never checked.** Bold, contrast and
    type size need pixels or dimensions the strategy does not have, and must
    say so.

Labels are built with the shared factories in tests/conftest.py and run
through the real span index, so every test exercises reconstruction and
expansion as well as the checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from extraction.geometry import BBox
from extraction.providers.base import Line
from extraction.providers.stub import StubProvider
from extraction.spans import build_span_index, mask_spans
from rules.comparators import StrategyContext
from rules.engine import STRATEGIES, load_strategies, verify
from rules.results import (
    ApplicationRecord,
    CheckOutcome,
    FieldResult,
    ImageQuality,
    Strategy,
    Verdict,
)
from rules.schema import load_ruleset
from rules.warning import _classify, _degraded_image, warning_strategy
from tests.conftest import STATUTORY_WARNING, make_extraction, make_lines

RULESET = load_ruleset()
FIELD = "government_warning"
ADVISORY_IDS = {c.id for c in RULESET.government_warning.checks if c.advisory}
WARNING_TUNING = RULESET.tuning.warning
DECLARED_IDS = [c.id for c in RULESET.government_warning.checks]

SPIRITS_RECORD = ApplicationRecord(
    brand_name="OLD TOM DISTILLERY",
    class_type="Kentucky Straight Bourbon Whiskey",
    alcohol_content="45% Alc./Vol.",
    net_contents="750 mL",
)

#: The compliant statement as laid out in conftest's `sample_label_lines`:
#: five lines of 12px fine print. Tests mutate copies of this.
WARNING_LINES = [
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women",
    "should not drink alcoholic beverages during pregnancy because of",
    "the risk of birth defects. (2) Consumption of alcoholic beverages",
    "impairs your ability to drive a car or operate machinery, and may",
    "cause health problems.",
]

#: The label above the warning, as in `sample_label_lines`. The 40px leading
#: after the last line sets the warning well apart, as a real label would.
LABEL_TOP = [
    ("OLD TOM", {"height": 48.0}),
    ("DISTILLERY", {"height": 48.0}),
    ("Kentucky Straight Bourbon Whiskey", {"height": 22.0}),
    ("750 mL 45% Alc./Vol. (90 Proof)", {"height": 18.0, "leading": 40.0}),
]

FINE = 12.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fine(texts: list[str], **overrides) -> list[tuple[str, dict]]:
    """Fine-print line specs for `make_lines`."""
    return [(text, {"height": FINE, **overrides}) for text in texts]


def label(warning: list[str], *, top=LABEL_TOP, after: list[str] | None = None, **kw):
    """A label: the display block, the warning lines, then optional trailing text."""
    return make_lines([*top, *fine(warning), *fine(after or [])], **kw)


def run(lines: tuple[Line, ...]) -> FieldResult:
    """Run the strategy directly, as the engine would for the first field."""
    context = StrategyContext(
        field_rule=RULESET.field(FIELD),
        ruleset=RULESET,
        beverage=RULESET.beverage_class("distilled_spirits"),
        spans=build_span_index(lines),
        expected="",
    )
    return warning_strategy(context)


def check(result: FieldResult, check_id: str):
    matches = [c for c in result.checks if c.id == check_id]
    assert len(matches) == 1, f"expected exactly one {check_id!r} check"
    return matches[0]


def replaced(lines: list[str], index: int, old: str, new: str) -> list[str]:
    copy = list(lines)
    assert old in copy[index], f"{old!r} not in line {index}"
    copy[index] = copy[index].replace(old, new)
    return copy


def _context_with(quality: ImageQuality | None) -> StrategyContext:
    """A minimal context carrying just a quality assessment, for `_degraded_image`."""
    return StrategyContext(
        field_rule=RULESET.field(FIELD),
        ruleset=RULESET,
        beverage=RULESET.beverage_class("distilled_spirits"),
        spans=[],
        expected="",
        quality=quality,
    )


def to_bbox(result: FieldResult) -> BBox:
    box = result.bbox
    return BBox(box.left, box.top, box.left + box.width, box.top + box.height)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_strategy_is_registered_under_its_rule_name(self) -> None:
        load_strategies()
        assert STRATEGIES["warning"] is warning_strategy
        assert RULESET.field(FIELD).strategy == "warning"

    @pytest.mark.parametrize(
        "statement",
        [
            "import rules.warning",
            "import rules.engine; rules.engine.load_strategies(); import rules.warning",
        ],
    )
    def test_no_import_cycle_in_either_order(self, statement: str) -> None:
        """Imported directly or via `load_strategies`, in a fresh interpreter.

        A fresh process because this suite has already imported everything,
        which would hide a cycle that only bites on a cold start.
        """
        completed = subprocess.run(
            [sys.executable, "-c", statement], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# The compliant label
# ---------------------------------------------------------------------------


class TestCompliantWarning:
    @pytest.fixture
    def result(self, sample_label_lines) -> FieldResult:
        return run(sample_label_lines)

    def test_warning_split_over_five_lines_matches(self, result: FieldResult) -> None:
        """Longer than the span index's four-line cap; expansion must recover it."""
        assert result.verdict is Verdict.MATCH
        assert result.strategy is Strategy.ANCHORED
        assert result.score == 1.0
        assert result.found == STATUTORY_WARNING
        assert result.differences == ()

    def test_expected_is_the_statute_not_the_empty_application_value(
        self, result: FieldResult
    ) -> None:
        assert result.expected == RULESET.government_warning.statutory_text

    def test_every_declared_check_is_reported_once_in_rule_set_order(
        self, result: FieldResult
    ) -> None:
        assert [c.id for c in result.checks] == DECLARED_IDS

    def test_every_check_carries_its_citation(self, result: FieldResult) -> None:
        declared = {c.id: c.citation for c in RULESET.government_warning.checks}
        for item in result.checks:
            assert item.citation == declared[item.id]

    def test_binding_checks_pass(self, result: FieldResult) -> None:
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS
        assert check(result, "text_accuracy").outcome is CheckOutcome.PASS

    def test_bbox_covers_the_whole_statement(self, result: FieldResult, sample_label_lines) -> None:
        """The engine masks this region, so it must cover all five lines."""
        box = to_bbox(result)
        for line in sample_label_lines[4:]:
            assert box.left <= line.bbox.left and box.right >= line.bbox.right
            assert box.top <= line.bbox.top and box.bottom >= line.bbox.bottom
        # ...and nothing above it.
        assert box.top > sample_label_lines[3].bbox.bottom

    def test_reason_does_not_overclaim(self, result: FieldResult) -> None:
        """A MATCH on words and capitals is not a MATCH on bold or type size."""
        assert "not checked automatically" in result.reason


# ---------------------------------------------------------------------------
# Header capitalisation — trap 1
# ---------------------------------------------------------------------------


class TestHeaderCapitalisation:
    def test_title_case_header_fails_and_is_not_a_match(self) -> None:
        """THE regression test. Jenny Park's rejected label.

        "Government Warning" in title case on a clearly read label. The anchor
        is case-insensitive and must still locate it (a formatting violation,
        not a location failure); the capitalisation check must read RAW text
        and fail it. If this ever passes, the check is reading canonical text.
        """
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning")
        result = run(label(lines))

        header = check(result, "header_capitalisation")
        assert header.outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH
        assert result.verdict is not Verdict.MATCH
        assert "title case" in result.reason
        assert "Government Warning" in result.reason
        assert "16.22(a)(2)" in result.reason
        assert "capital letters" in header.summary

    def test_title_case_is_still_located_and_the_wording_still_passes(self) -> None:
        """Located, with the right region: the agent needs the highlight."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning")
        result = run(label(lines))

        assert result.strategy is Strategy.ANCHORED
        assert result.bbox is not None
        assert result.found.startswith("Government Warning:")
        assert check(result, "text_accuracy").outcome is CheckOutcome.PASS
        assert result.differences == ()

    def test_title_case_at_the_end_to_end_level(self) -> None:
        """The same violation through `rules.engine.verify` with a stub provider."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning")
        extraction = StubProvider(lines=label(lines)).extract(None)
        result = verify(extraction, SPIRITS_RECORD).field(FIELD)

        assert result.verdict is Verdict.MISMATCH
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL

    def test_lowercase_header_fails(self) -> None:
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "government warning")
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH
        assert "not in capital letters" in result.reason

    def test_word_in_title_case_among_capitals_fails(self) -> None:
        """One word set in title case is still a printing style, not noise."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "GOVERNMENT Warning")
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH

    def test_scattered_lowercase_is_reported_for_review(self) -> None:
        """Neither clean capitals nor a printing style: eight lowercase letters
        in no pattern. Too noisy to pass, too irregular to assert."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "GoVeRnMeNt WaRnInG")
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.REVIEW
        assert "confirm visually" in result.reason

    @pytest.mark.parametrize(
        "header",
        [
            "GOVERNMENT WARNiNG",  # real read, fixtures/labels/degraded_low_contrast.png
            "GoVERNMENT WARNING",  # real read, fixtures/labels/degraded_noise_blur.jpg
            "GOVERNMENT WARNlNG",  # lowercase L for capital I
            "GOVERNMENT WARNING",  # (control)
        ],
    )
    def test_sporadic_case_noise_passes(self, header: str) -> None:
        """Isolated lowercase letters inside capitalised words are OCR noise.

        Both mis-cased headers here come from the real engine on degraded
        photographs of labels printed in correct capitals. Failing the check
        and then declining to stand behind the failure would still cost the
        agent the twenty seconds; the honest answer is that on this evidence
        the header is in capitals.
        """
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", header)
        result = run(label(lines))

        item = check(result, "header_capitalisation")
        assert item.outcome is CheckOutcome.PASS
        assert result.verdict is Verdict.MATCH
        assert header in item.summary

    def test_sporadic_case_noise_is_recorded_not_silently_dropped(self) -> None:
        """A pass, but the letters read in lowercase stay auditable."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "GOVERNMENT WARNiNG")
        item = check(run(label(lines)), "header_capitalisation")
        assert "isolated lowercase letter(s) (i)" in item.summary
        assert "systematically" in item.detail

    def test_low_confidence_title_case_is_review(self) -> None:
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning")
        specs = [*LABEL_TOP, *[(t, {"height": FINE, "confidence": 0.3}) for t in lines]]
        result = run(make_lines(specs))
        assert result.verdict is Verdict.REVIEW
        assert "low confidence" in result.reason

    def test_colon_missing_is_reported_but_only_for_review(self) -> None:
        """16.21 prescribes "GOVERNMENT WARNING:" — the colon is statutory text.

        But a colon is the least reliable thing OCR reports, so its absence in
        the read is a REVIEW, never a MISMATCH on its own.
        """
        lines = replaced(WARNING_LINES, 0, "WARNING:", "WARNING")
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS
        assert check(result, "text_accuracy").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.REVIEW
        assert "colon" in result.reason


# ---------------------------------------------------------------------------
# Text accuracy
# ---------------------------------------------------------------------------


class TestRuleSetTuning:
    """The assertion thresholds come from the rule set, not from this module.

    Every result is stamped with a rule-set version so a verdict can be
    reproduced against the thresholds in force when it was made. A module
    constant shadowing one of these — or a fallback default standing in when
    the rule set is silent — breaks that quietly: the recorded version would
    no longer describe the thresholds that ran.

    So each test below moves one rule-set value and requires the verdict to
    move with it. A hardcoded threshold reintroduced by a later refactor
    cannot pass them.
    """

    def with_warning_tuning(self, **overrides):
        tuning = RULESET.tuning
        return RULESET.model_copy(
            update={
                "tuning": tuning.model_copy(
                    update={"warning": tuning.warning.model_copy(update=overrides)}
                )
            }
        )

    def with_defaults(self, **overrides):
        return RULESET.model_copy(
            update={"defaults": RULESET.defaults.model_copy(update=overrides)}
        )

    def run_under(self, lines, ruleset, quality: ImageQuality | None = None) -> FieldResult:
        return warning_strategy(
            StrategyContext(
                field_rule=ruleset.field(FIELD),
                ruleset=ruleset,
                beverage=ruleset.beverage_class("distilled_spirits"),
                spans=build_span_index(lines),
                expected="",
                quality=quality,
            )
        )

    def test_no_module_constant_shadows_a_rule_set_threshold(self) -> None:
        """The names are gone, so nothing can drift out of step with the YAML."""
        import rules.warning as module

        for name in (
            "CLEAN_READ_MIN_PRECISION",
            "MAX_SPORADIC_LOWERCASE",
            "MIN_CASE_STYLE_RUN",
            "NOISE_MIN_TOKEN_LENGTH",
            "RAGGED_RIGHT_TOLERANCE",
            "LINE_GAP_SLACK_RATIO",
            "MAX_WORD_GAP_RATIO",
            "INDENT_TOLERANCE_RATIO",
            "QUALITY_TYPOGRAPHY_THRESHOLD",
            "GOOD_QUALITY_THRESHOLD",
        ):
            assert not hasattr(module, name), f"{name} shadows a rule-set value"

    def test_max_sporadic_lowercase_is_read_from_the_rule_set(self) -> None:
        """At 0, no mis-cased letter is tolerated as noise any more."""
        lines = label(replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "GOVERNMENT WARNiNG"))

        assert check(self.run_under(lines, RULESET), "header_capitalisation").outcome is (
            CheckOutcome.PASS
        )
        strict = self.run_under(lines, self.with_warning_tuning(max_sporadic_lowercase=0))
        assert check(strict, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert strict.verdict is Verdict.REVIEW

    def test_min_case_style_run_is_read_from_the_rule_set(self) -> None:
        """Raised past the length of "Government", title case stops reading as
        a deliberate style — which is exactly why this value is policy and
        belongs in the versioned rule set rather than in code."""
        lines = self.title_case_lines()

        assert self.run_under(lines, RULESET).verdict is Verdict.MISMATCH
        relaxed = self.run_under(lines, self.with_warning_tuning(min_case_style_run=99))
        assert relaxed.verdict is Verdict.REVIEW

    def test_clean_read_min_precision_is_read_from_the_rule_set(self) -> None:
        """At 1.0 every token must be statutory before a wording change may be
        asserted, and the changed word itself is not."""
        lines = label(replaced(WARNING_LINES, 3, "and may", "and can"))

        assert self.run_under(lines, RULESET).verdict is Verdict.MISMATCH
        strict = self.run_under(lines, self.with_warning_tuning(clean_read_min_precision=1.0))
        assert strict.verdict is Verdict.REVIEW
        assert strict.differences  # still reported, just not asserted

    def test_noise_min_token_length_is_read_from_the_rule_set_both_ways(self) -> None:
        """The line between OCR noise and a wording change, moved each way.

        It governs which words are *eligible* for the one-edit tolerance, so
        each direction has to be tested with a word of the right length:

        *   "risk" as "rick" is one edit on a four-letter word. At the rule
            set's 5 it is a wording change and asserted; lower the value to 4
            and the same reading becomes tolerated noise.
        *   "Consumption" as "Consumptiom" is one edit on an eleven-letter
            word, tolerated at 5; raise the value past its length and the same
            reading becomes an asserted wording change.

        Note what it does *not* do: no setting makes `may` → `can` noise, since
        that is three edits on a three-letter word and fails the distance test
        whatever the length gate says.
        """
        short_misread = label(replaced(WARNING_LINES, 2, "the risk of", "the rick of"))
        long_misread = label(replaced(WARNING_LINES, 2, "Consumption", "Consumptiom"))

        asserted = self.run_under(short_misread, RULESET)
        assert asserted.verdict is Verdict.MISMATCH
        assert [(d.expected, d.found) for d in asserted.differences] == [("risk", "rick")]
        assert self.run_under(long_misread, RULESET).verdict is Verdict.MATCH

        forgiving = self.run_under(
            short_misread, self.with_warning_tuning(noise_min_token_length=4)
        )
        assert forgiving.verdict is Verdict.MATCH
        assert forgiving.differences == ()

        strict = self.run_under(long_misread, self.with_warning_tuning(noise_min_token_length=12))
        assert strict.verdict is Verdict.MISMATCH
        assert [d.found for d in strict.differences] == ["Consumptiom"]

    def test_line_gap_slack_ratio_is_read_from_the_rule_set(self) -> None:
        """Vertical room for a line the engine failed to read.

        A missing line with a line-sized gap where it was is REVIEW. Widen the
        slack far enough and the same gap stops counting as room, so the
        missing words become assertable.
        """
        specs = [
            *LABEL_TOP,
            (WARNING_LINES[0], {"height": FINE}),
            (WARNING_LINES[1], {"height": FINE, "leading": 10.0 + FINE + 10.0}),
            *fine(WARNING_LINES[3:]),
        ]
        lines = make_lines(specs)

        assert self.run_under(lines, RULESET).verdict is Verdict.REVIEW
        assert self.run_under(
            lines, self.with_warning_tuning(line_gap_slack_ratio=8.0)
        ).verdict is (Verdict.MISMATCH)

    def test_ragged_right_tolerance_is_read_from_the_rule_set(self) -> None:
        """Horizontal room at the end of a line that stops short.

        Here the missing clause spans a line break, which is the branch this
        value governs: the line before the gap has to reach the block's right
        edge for the absence to be asserted. Tightened to zero, ordinary
        ragged-right setting no longer qualifies, so the words could have gone
        unread and the finding is reported rather than asserted.
        """
        lines = label(
            [
                "GOVERNMENT WARNING: (1) According to the Surgeon General, women",
                "because of the risk of birth defects. (2) Consumption of alcoholic",
                "beverages impairs your ability to drive a car or operate machinery,",
                "and may cause health problems.",
            ]
        )

        assert self.run_under(lines, RULESET).verdict is Verdict.MISMATCH
        relaxed = self.run_under(lines, self.with_warning_tuning(ragged_right_tolerance=0.0))
        assert relaxed.verdict is Verdict.REVIEW
        assert any(d.kind == "missing" for d in relaxed.differences)

    def test_max_word_gap_ratio_is_read_from_the_rule_set(self) -> None:
        """Horizontal room between two words on the same line.

        The missing sentence sits between two words an ordinary space apart.
        Tighten the ratio to zero and that space becomes room for unread text,
        so the finding drops to REVIEW.
        """
        lines = label(
            [
                "GOVERNMENT WARNING: (2) Consumption of alcoholic beverages impairs",
                "your ability to drive a car or operate machinery, and may cause",
                "health problems.",
            ]
        )

        assert self.run_under(lines, RULESET).verdict is Verdict.MISMATCH
        assert self.run_under(lines, self.with_warning_tuning(max_word_gap_ratio=0.0)).verdict is (
            Verdict.REVIEW
        )

    def test_indent_tolerance_ratio_is_read_from_the_rule_set(self) -> None:
        """Room before the first word of the block's first line.

        The header-missing case: the body starts flush at the block's left
        edge with other text at ordinary leading above, so there is nowhere
        for the header to be. Tighten the indent tolerance below zero width
        and even a flush line reads as indented, which reopens the doubt.
        """
        body = [
            "(1) According to the Surgeon General, women should not drink",
            "alcoholic beverages during pregnancy because of the risk of birth",
            "defects. (2) Consumption of alcoholic beverages impairs your ability",
            "to drive a car or operate machinery, and may cause health problems.",
        ]
        lines = make_lines(
            [
                *LABEL_TOP,
                ("Distilled and bottled by Old Tom Distillery", {"height": FINE}),
                *fine(body),
            ]
        )

        assert self.run_under(lines, RULESET).verdict is Verdict.MISMATCH
        relaxed = self.run_under(lines, self.with_warning_tuning(indent_tolerance_ratio=-1.0))
        assert relaxed.verdict is Verdict.REVIEW
        assert relaxed.differences[0].expected == "GOVERNMENT WARNING"

    def test_good_image_quality_is_read_from_the_rule_set(self) -> None:
        """The degraded-image guard reads the rule set's threshold, not the
        engine's module-level alias and not a constant of its own."""
        lines = self.title_case_lines()
        clean = ImageQuality(usable=True, score=0.95, issues=())

        assert self.run_under(lines, RULESET, clean).verdict is Verdict.MISMATCH
        strict = self.run_under(lines, self.with_defaults(good_image_quality=0.99), clean)
        assert strict.verdict is Verdict.REVIEW
        assert "image quality 95%" in strict.reason

    def title_case_lines(self):
        return label(replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning"))


class TestDegradedImageGuard:
    """The second line of defence behind the case-pattern rule, now live.

    `StrategyContext.quality` carries the engine's assessment, and this module
    may use it only to withdraw an assertion. The tests below cover the four
    states that matter: flagged, clean, unassessed, and absent.

    Note what the guard is *for*. It does not fire on any fixture in the
    corpus today, because no fixture combines a degraded photograph with a
    systematically miscapitalised header — the degraded fixtures read as
    sporadic case noise, which the pattern rule passes on its own. Its value
    is the case the corpus does not contain, and
    `test_matches_the_real_gates_verdict_on_a_degraded_fixture` pins it to the
    real assessment of a real degraded image rather than to invented numbers.
    """

    #: The real assessments the pipeline produces, so these tests move if the
    #: gate's calibration moves. Measured on the fixture corpus.
    CLEAN = ImageQuality(usable=True, score=0.950, issues=())
    LOW_CONTRAST = ImageQuality(usable=True, score=0.531, issues=("low_contrast",))
    UNASSESSED = ImageQuality(
        usable=True, score=0.0, issues=(), assessed=False, guidance=("not assessed",)
    )

    def run_with_quality(self, lines, quality: ImageQuality | None) -> FieldResult:
        context = StrategyContext(
            field_rule=RULESET.field(FIELD),
            ruleset=RULESET,
            beverage=RULESET.beverage_class("distilled_spirits"),
            spans=build_span_index(lines),
            expected="",
            quality=quality,
        )
        return warning_strategy(context)

    def title_case(self) -> tuple:
        return label(replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning"))

    def test_title_case_on_a_flagged_image_is_reported_not_asserted(self) -> None:
        """A systematic case violation, but on a photograph the gate flagged.

        Still FAIL on the check — the reading stands and the agent sees it —
        but not a MISMATCH: the tool does not assert a printing violation off
        a photograph it has itself called unreliable.
        """
        result = self.run_with_quality(self.title_case(), self.LOW_CONTRAST)

        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.REVIEW
        assert "low_contrast" in result.reason

    def test_title_case_on_a_clean_image_is_still_asserted(self) -> None:
        """The guard must not fire on a good photograph. This is Jenny Park's
        label: `warning_title_case` is a clean image and scores 0.95."""
        result = self.run_with_quality(self.title_case(), self.CLEAN)

        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH
        assert "image quality" not in result.reason

    def test_unassessed_quality_does_not_enable_the_guard(self) -> None:
        """An unassessed image is not a degraded one.

        Withdrawing assertions whenever quality is unknown would mean every
        verification that skipped preprocessing quietly stopped reporting
        printing violations — the check this module exists for.
        """
        result = self.run_with_quality(self.title_case(), self.UNASSESSED)
        assert result.verdict is Verdict.MISMATCH

    def test_unassessed_quality_does_not_count_as_a_good_image(self) -> None:
        """Nor does it read as a favourable measurement: the guard reports no
        evidence of degradation, which is not the same as evidence of quality,
        and it never cites a score it does not have."""
        assert _degraded_image(_context_with(self.UNASSESSED)) is None
        assert _degraded_image(_context_with(None)) is None
        assert _degraded_image(_context_with(self.CLEAN)) is None
        assert "0%" not in (self.run_with_quality(self.title_case(), self.UNASSESSED).reason or "")

    def test_absent_quality_behaves_as_before_the_gate_was_wired(self) -> None:
        result = self.run_with_quality(self.title_case(), None)
        assert result.verdict is Verdict.MISMATCH

    def test_unusable_image_is_reported_not_asserted(self) -> None:
        unusable = ImageQuality(usable=False, score=0.2, issues=("too_small",))
        result = self.run_with_quality(self.title_case(), unusable)
        assert result.verdict is Verdict.REVIEW
        assert "not usable" in result.reason

    def test_a_flagged_issue_degrades_even_at_a_good_score(self) -> None:
        """The issue list is the sturdier half of the signal, so a flagged
        issue counts whatever the score says."""
        flagged = ImageQuality(usable=True, score=0.9, issues=("glare",))
        result = self.run_with_quality(self.title_case(), flagged)
        assert result.verdict is Verdict.REVIEW
        assert "glare" in result.reason

    def test_wording_changes_are_unaffected_by_image_quality(self) -> None:
        """Only typographic findings are withdrawn. A word read confidently as
        a different word is still a word read as a different word."""
        lines = replaced(WARNING_LINES, 3, "and may", "and can")
        unusable = ImageQuality(usable=False, score=0.3, issues=("blurred",))
        assert self.run_with_quality(label(lines), unusable).verdict is Verdict.MISMATCH

    @pytest.mark.parametrize(
        ("fixture", "fires"),
        [("degraded_low_contrast.png", True), ("old_tom_compliant.png", False)],
    )
    def test_matches_the_real_gates_verdict_on_a_degraded_fixture(
        self, fixture: str, fires: bool
    ) -> None:
        """Pin the guard to the gate's real calibration, not to my constants.

        Runs the actual preprocessing assessment over a real fixture image —
        no OCR engine involved — so that if the gate's scoring is retuned, the
        test that says "this is what a degraded image looks like" moves with
        it rather than silently going stale.
        """
        image = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / fixture
        if not image.exists():
            pytest.skip(f"fixture {fixture} not present")
        from extraction.pipeline import preprocess

        assessment = preprocess(image.read_bytes()).quality
        quality = ImageQuality(
            usable=assessment.usable,
            score=assessment.score,
            issues=assessment.issues,
            measurements=assessment.metrics,
        )
        assert (_degraded_image(_context_with(quality)) is not None) is fires


class TestWordingChanges:
    def test_may_cause_as_can_cause_is_detected(self) -> None:
        """The brief's example, reported the way the brief phrases it."""
        lines = replaced(WARNING_LINES, 3, "and may", "and can")
        result = run(label(lines))

        assert result.verdict is Verdict.MISMATCH
        assert check(result, "text_accuracy").outcome is CheckOutcome.FAIL
        assert len(result.differences) == 1
        diff = result.differences[0]
        assert (diff.kind, diff.expected, diff.found) == ("substituted", "may", "can")
        assert "“may cause health problems” appears as “can cause health problems”" in (
            result.reason
        )

    def test_difference_position_indexes_the_statute(self) -> None:
        lines = replaced(WARNING_LINES, 3, "and may", "and can")
        diff = run(label(lines)).differences[0]
        # "may" is the 40th canonical token of the statute (0-based 39).
        assert diff.position == STATUTORY_WARNING.split().index("may")

    def test_short_word_change_is_never_noise(self) -> None:
        """ "not" dropped from "should not drink" changes the meaning entirely."""
        lines = replaced(WARNING_LINES, 1, "should not drink", "should drink")
        result = run(label(lines))
        assert result.verdict is Verdict.MISMATCH
        assert any(d.kind == "missing" and d.expected == "not" for d in result.differences)

    def test_added_word_is_detected(self) -> None:
        lines = replaced(WARNING_LINES, 4, "cause health", "cause serious health")
        result = run(label(lines))
        assert result.verdict is Verdict.MISMATCH
        assert [(d.kind, d.found) for d in result.differences] == [("extra", "serious")]

    @pytest.mark.parametrize(
        ("line", "old", "new"),
        [
            (0, "women", "woman"),  # vowel swap: a different word
            (2, "defects.", "defect."),  # grammatical number
            (1, "alcoholic", "alcohol"),  # truncation to another word
        ],
    )
    def test_near_misses_that_are_real_words_are_reported(
        self, line: int, old: str, new: str
    ) -> None:
        """One edit away, but a recognisable variant: wording, not noise."""
        result = run(label(replaced(WARNING_LINES, line, old, new)))
        assert result.verdict is Verdict.MISMATCH
        assert [d.kind for d in result.differences] == ["substituted"]
        assert result.differences[0].found.rstrip(".") == new.rstrip(".")

    def test_two_edits_on_a_long_word_is_reported_for_review(self) -> None:
        """ "machinery" as "machines": could be either. Reported; not asserted."""
        result = run(label(replaced(WARNING_LINES, 3, "machinery,", "machines,")))
        assert result.verdict is Verdict.REVIEW
        assert [d.kind for d in result.differences] == ["substituted"]
        assert "misread" in result.reason


class TestMergedWords:
    """Dropped spaces, from the real engine on the fixture corpus.

    These were three of four false flags in the first `--provider rapidocr`
    run. The diff always tolerated merges; the *location scorer* did not, so a
    merged token scored as matching nothing, expansion stopped before it, and
    the words inside it were then reported missing — a MISMATCH on a compliant
    label caused by a dropped space. The lines below carry the engine's actual
    output so the regression runs without the OCR engine installed.
    """

    def test_merged_final_words_are_tolerated(self) -> None:
        """fixtures/labels/shared_line_abv_net.png: "healthproblems."."""
        lines = replaced(WARNING_LINES, 4, "cause health problems.", "cause healthproblems.")
        result = run(label(lines))

        assert result.verdict is Verdict.MATCH
        assert result.differences == ()
        assert "healthproblems." in result.found  # the merged word is inside the block
        assert check(result, "text_accuracy").outcome is CheckOutcome.PASS

    def test_merged_words_run_on_across_the_statement(self) -> None:
        """fixtures/labels/degraded_noise_blur.jpg, as the engine read it.

        Six statutory words merged into one token, a merged pair, a merged
        header/enumerator, and a full stop where a comma belongs.
        """
        lines = [
            "GoVERNMENT WARNING:(1) According to the Surgeon General,women should",
            "not drink alcoholic beverages during pregnancy because of the risk of",
            "birth defects. (2) Consumption ofalcoholicbeveragesimpairsyourability",
            "to drive a caroroperate machinery.and may cause healthproblems.",
        ]
        result = run(label(lines))

        assert result.verdict is Verdict.MATCH
        assert result.differences == ()
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS

    def test_a_merge_never_swallows_a_wording_change(self) -> None:
        """Merge tolerance is about spaces, not about the words themselves."""
        lines = replaced(WARNING_LINES, 4, "cause health problems.", "cause healthissues.")
        result = run(label(lines))
        assert result.verdict is not Verdict.MATCH
        assert result.differences

    def test_merged_header_still_fails_title_case(self) -> None:
        """The key regression, run through the merge path as well."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING:", "GovernmentWarning:")
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH


class TestOcrNoiseTolerance:
    @pytest.mark.parametrize(
        ("line", "old", "new"),
        [
            (2, "Consumption", "Consumptiom"),  # single-character misread
            (1, "beverages", "bevcrages"),  # e/c confusion
            (0, "Surgeon", "Surqeon"),  # g/q confusion
            (3, "machinery,", "machinerv,"),  # y/v confusion
        ],
    )
    def test_single_character_misread_of_a_long_word_still_matches(
        self, line: int, old: str, new: str
    ) -> None:
        result = run(label(replaced(WARNING_LINES, line, old, new)))
        assert result.verdict is Verdict.MATCH
        assert result.differences == ()

    def test_tolerated_noise_is_listed_not_hidden(self) -> None:
        result = run(label(replaced(WARNING_LINES, 2, "Consumption", "Consumptiom")))
        accuracy = check(result, "text_accuracy")
        assert accuracy.measurement["tolerated_ocr_variations"] == 1
        assert "Consumptiom" in accuracy.detail

    def test_folded_confusions_are_invisible(self) -> None:
        """O/0 and I/1/L are folded by canonicalisation before the diff runs."""
        result = run(label(replaced(WARNING_LINES, 0, "(1) According", "(l) Accord1ng")))
        assert result.verdict is Verdict.MATCH

    def test_header_run_together_is_segmentation_noise(self) -> None:
        """ "GOVERNMENTWARNING:" — the engine dropped a space, not the label."""
        result = run(label(replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "GOVERNMENTWARNING")))
        assert result.verdict is Verdict.MATCH
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS

    @pytest.mark.parametrize(
        ("expected", "found", "grade"),
        [
            ("MAY", "CAN", "wording"),
            ("CAUSE", "CAUSE", "noise"),
            ("CONSUMPTION", "CONSUMPTIOM", "noise"),
            ("WOMEN", "WOMAN", "wording"),
            ("DEFECTS", "DEFECT", "wording"),
            ("ALCOHOIIC", "ALCOHOI", "wording"),
            ("MACHINERY", "MACHINES", "ambiguous"),
            ("PREGNANCY", "PRESENCE", "wording"),
        ],
    )
    def test_classification_table(self, expected: str, found: str, grade: str) -> None:
        """The noise-versus-wording line, pinned. Tokens are canonical here."""
        assert _classify(expected, found, WARNING_TUNING) == grade


class TestMissingText:
    def test_missing_sentence_mid_statement_is_a_mismatch(self) -> None:
        """Sentence (1) absent, with (2) following the header on the same line.

        The words either side of the gap sit an ordinary word space apart, so
        there is nowhere for the sentence to be — positive evidence of absence.
        """
        lines = [
            "GOVERNMENT WARNING: (2) Consumption of alcoholic beverages impairs",
            "your ability to drive a car or operate machinery, and may cause",
            "health problems.",
        ]
        result = run(label(lines))

        assert result.verdict is Verdict.MISMATCH
        missing = [d for d in result.differences if d.kind == "missing"]
        assert len(missing) == 1
        assert "Surgeon General" in missing[0].expected
        assert "birth defects" in missing[0].expected
        assert "are missing" in result.reason

    def test_missing_line_where_ocr_could_have_dropped_it_is_review(self) -> None:
        """A whole line gone, and a line-sized gap where it was.

        The engine failing to detect a line looks exactly like this, so the
        difference is reported but not asserted.
        """
        specs = [
            *LABEL_TOP,
            (WARNING_LINES[0], {"height": FINE}),
            (WARNING_LINES[1], {"height": FINE, "leading": 10.0 + FINE + 10.0}),
            *fine(WARNING_LINES[3:]),
        ]
        result = run(make_lines(specs))

        assert result.verdict is Verdict.REVIEW
        assert any(d.kind == "missing" for d in result.differences)
        assert "did not detect" in result.reason

    def test_missing_final_clause_at_the_end_of_the_label_is_review(self) -> None:
        """Nothing after the statement: the image may simply be cropped."""
        lines = [*WARNING_LINES[:3], "impairs your ability to drive a car or operate machinery."]
        result = run(label(lines))
        assert result.verdict is Verdict.REVIEW
        assert any(
            d.kind == "missing" and "health problems" in d.expected for d in result.differences
        )


class TestHeaderMissing:
    BODY = [
        "(1) According to the Surgeon General, women should not drink",
        "alcoholic beverages during pregnancy because of the risk of birth",
        "defects. (2) Consumption of alcoholic beverages impairs your ability",
        "to drive a car or operate machinery, and may cause health problems.",
    ]

    def test_header_missing_body_intact_with_no_room_for_it_is_a_mismatch(self) -> None:
        """Anchored on the tail phrases; the header is mandatory and absent.

        The line directly above is other text at ordinary leading and the
        body's first line is not indented, so the header cannot be sitting
        unread above or beside it.
        """
        specs = [
            *LABEL_TOP,
            ("Distilled and bottled by Old Tom Distillery", {"height": FINE}),
            *fine(self.BODY),
        ]
        result = run(make_lines(specs))

        assert result.verdict is Verdict.MISMATCH
        assert result.found.startswith("(1) According")
        first = result.differences[0]
        assert (first.kind, first.position) == ("missing", 0)
        assert first.expected == "GOVERNMENT WARNING"
        assert "header was not found" in result.reason
        assert check(result, "header_capitalisation").outcome is CheckOutcome.NOT_EVALUABLE
        assert check(result, "text_accuracy").outcome is CheckOutcome.FAIL

    def test_header_missing_with_a_gap_above_is_review(self) -> None:
        """A line-sized gap above: the header may be there, unread (often it is
        the bold, stylised part the engine struggles with)."""
        result = run(label(self.BODY))  # 40px leading below the ABV line
        assert result.verdict is Verdict.REVIEW
        assert result.differences[0].expected == "GOVERNMENT WARNING"

    def test_header_missing_with_nothing_above_is_review(self) -> None:
        result = run(make_lines(fine(self.BODY)))
        assert result.verdict is Verdict.REVIEW
        assert "header was not found" in result.reason

    def test_wrong_header_wording_is_reported(self) -> None:
        """ "HEALTH WARNING:" does not anchor as the header; the tail does.

        Expansion keeps "WARNING:" and leaves "HEALTH" out (an unmatched word
        costs more similarity than it adds), so the finding is "GOVERNMENT"
        missing from the header — bounded on the left by "HEALTH" an ordinary
        word space away, so it is asserted.
        """
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "HEALTH WARNING")
        result = run(label(lines))
        assert result.verdict is Verdict.MISMATCH
        first = result.differences[0]
        assert (first.kind, first.position, first.expected) == ("missing", 0, "GOVERNMENT")
        assert "header does not read" in result.reason


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class TestLocation:
    def test_header_split_across_two_lines_still_anchors(self) -> None:
        lines = [
            "GOVERNMENT",
            "WARNING: (1) According to the Surgeon General, women",
            *WARNING_LINES[1:],
        ]
        result = run(label(lines))
        assert result.verdict is Verdict.MATCH
        assert result.found.startswith("GOVERNMENT WARNING: (1)")
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS

    def test_title_case_header_split_across_lines_still_fails(self) -> None:
        lines = [
            "Government",
            "Warning: (1) According to the Surgeon General, women",
            *WARNING_LINES[1:],
        ]
        result = run(label(lines))
        assert check(result, "header_capitalisation").outcome is CheckOutcome.FAIL
        assert result.verdict is Verdict.MISMATCH

    def test_warning_starting_mid_line_excludes_the_preceding_text(self) -> None:
        lines = [
            "Bottled by Old Tom Distillery. GOVERNMENT WARNING: (1) According to",
            "the Surgeon General, women should not drink alcoholic beverages during",
            "pregnancy because of the risk of birth defects. (2) Consumption of",
            "alcoholic beverages impairs your ability to drive a car or operate",
            "machinery, and may cause health problems.",
        ]
        result = run(label(lines))
        assert result.verdict is Verdict.MATCH
        assert result.found == STATUTORY_WARNING
        assert "Bottled" not in result.found
        # The header is judged on its own words, not the line it shares.
        assert check(result, "header_capitalisation").outcome is CheckOutcome.PASS

    def test_expansion_stops_before_following_text(self) -> None:
        result = run(label(WARNING_LINES, after=["Distilled and bottled by Old Tom Distillery"]))
        assert result.verdict is Verdict.MATCH
        assert result.found == STATUTORY_WARNING

    def test_statement_ending_mid_line_is_cut_at_the_end(self) -> None:
        lines = [*WARNING_LINES[:4], "cause health problems. Product of USA"]
        result = run(label(lines))
        assert result.verdict is Verdict.MATCH
        assert result.found.endswith("health problems.")

    def test_no_warning_at_all_is_review_not_found(self) -> None:
        result = run(make_lines(LABEL_TOP))
        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.NOT_FOUND
        assert result.bbox is None
        assert "Could not locate" in result.reason

    def test_not_found_lists_every_check_as_not_evaluated(self) -> None:
        """So that nothing in the result can be read as a pass."""
        result = run(make_lines(LABEL_TOP))
        assert [c.id for c in result.checks] == DECLARED_IDS
        assert all(c.outcome is CheckOutcome.NOT_EVALUABLE for c in result.checks)

    def test_empty_pool_is_not_found(self) -> None:
        assert run(()).strategy is Strategy.NOT_FOUND

    def test_low_confidence_wording_change_is_review(self) -> None:
        lines = replaced(WARNING_LINES, 3, "and may", "and can")
        specs = [*LABEL_TOP, *[(t, {"height": FINE, "confidence": 0.3}) for t in lines]]
        result = run(make_lines(specs))
        assert result.verdict is Verdict.REVIEW
        assert result.differences  # still reported


# ---------------------------------------------------------------------------
# Advisory and not-evaluable checks
# ---------------------------------------------------------------------------


class TestAdvisoryChecks:
    @pytest.mark.parametrize(
        ("check_id", "phrase"),
        [
            ("header_bold", "pixel measurement"),
            ("contrast", "pixel measurement"),
            ("type_size", "label_width_mm"),
        ],
    )
    def test_unmeasurable_checks_are_not_evaluable_and_say_why(
        self, sample_label_lines, check_id: str, phrase: str
    ) -> None:
        item = check(run(sample_label_lines), check_id)
        assert item.outcome is CheckOutcome.NOT_EVALUABLE
        assert phrase in item.summary
        assert "Verify" in item.summary

    def test_separation_is_advisory_with_measurements(self, sample_label_lines) -> None:
        item = check(run(sample_label_lines), "separation")
        assert item.outcome is CheckOutcome.ADVISORY
        assert item.confidence is not None and item.confidence < 0.5
        # 40px of leading above 12px fine print.
        assert item.measurement["gap_above_line_heights"] == pytest.approx(40 / FINE, abs=0.01)
        assert item.measurement["shares_line_with_other_text"] == 0.0

    def test_crowded_warning_is_flagged_but_still_matches(self) -> None:
        """Advisory findings never change the verdict.

        Text crammed against the statement above, below and on its own first
        line: separation reports it, the verdict stays MATCH.
        """
        lines = [
            "Bottled by Old Tom Distillery. GOVERNMENT WARNING: (1) According to",
            "the Surgeon General, women should not drink alcoholic beverages during",
            "pregnancy because of the risk of birth defects. (2) Consumption of",
            "alcoholic beverages impairs your ability to drive a car or operate",
            "machinery, and may cause health problems.",
        ]
        specs = [
            ("750 mL 45% Alc./Vol. (90 Proof)", {"height": FINE, "leading": 1.0}),
            *fine(lines, leading=1.0),
            ("Product of USA", {"height": FINE}),
        ]
        result = run(make_lines(specs))

        separation = check(result, "separation")
        assert separation.outcome is CheckOutcome.ADVISORY
        assert separation.measurement["shares_line_with_other_text"] == 1.0
        assert "shares a line" in separation.summary
        assert result.verdict is Verdict.MATCH

    @pytest.mark.parametrize(
        "lines",
        [
            WARNING_LINES,
            replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning"),
            replaced(WARNING_LINES, 3, "and may", "and can"),
            replaced(WARNING_LINES, 3, "machinery,", "machines,"),
        ],
    )
    def test_advisory_checks_never_fail(self, lines: list[str]) -> None:
        for item in run(label(lines)).checks:
            if item.id in ADVISORY_IDS:
                assert item.outcome in {CheckOutcome.ADVISORY, CheckOutcome.NOT_EVALUABLE}

    @pytest.mark.parametrize("gap", [0.5, 5.0, 80.0])
    def test_separation_does_not_move_the_verdict(self, gap: float) -> None:
        specs = [("750 mL 45% Alc./Vol. (90 Proof)", {"height": FINE, "leading": gap})]
        result = run(make_lines([*specs, *fine(WARNING_LINES)]))
        assert result.verdict is Verdict.MATCH


# ---------------------------------------------------------------------------
# End to end through the engine
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_compliant_label_matches_through_the_engine(self, sample_label_lines) -> None:
        extraction = StubProvider(lines=sample_label_lines).extract(None)
        result = verify(extraction, SPIRITS_RECORD).field(FIELD)
        assert result.verdict is Verdict.MATCH
        assert "does not yet check" not in result.reason

    def test_missing_warning_on_a_good_image_is_elevated_review(self) -> None:
        """The one case that pulls against not-found-is-never-a-mismatch.

        Still REVIEW — the tool does not assert — but at raised prominence.
        """
        result = verify(make_extraction(make_lines(LABEL_TOP), quality_score=0.9), SPIRITS_RECORD)
        warning = result.field(FIELD)
        assert warning.verdict is Verdict.REVIEW
        assert warning.strategy is Strategy.NOT_FOUND
        assert warning.elevated is True
        assert "image quality is good" in warning.reason

    def test_missing_warning_on_a_poor_image_is_not_elevated(self) -> None:
        result = verify(make_extraction(make_lines(LABEL_TOP), quality_score=0.4), SPIRITS_RECORD)
        warning = result.field(FIELD)
        assert warning.verdict is Verdict.REVIEW
        assert warning.elevated is False

    def test_warning_words_and_numerals_are_masked_from_later_fields(
        self, sample_label_lines
    ) -> None:
        """ "alcoholic beverages", "(1)", "(2)" must not reach class/type or the
        numeric fields once the warning has resolved and been masked."""
        spans = build_span_index(sample_label_lines)
        context = StrategyContext(
            field_rule=RULESET.field(FIELD),
            ruleset=RULESET,
            beverage=RULESET.beverage_class("distilled_spirits"),
            spans=spans,
            expected="",
        )
        remaining = mask_spans(spans, to_bbox(warning_strategy(context)))

        texts = [s.canonical_text for s in remaining]
        for decoy in ("ALCOHOIIC", "SURGEON", "MACHINERY", "HEAITH"):
            assert not any(decoy in t for t in texts), decoy
        assert not any(s.numeric_text in {"(1)", "(2)", "1", "2"} for s in remaining)
        # The real fields survive.
        assert any(s.canonical_text == "750 MI" for s in remaining)
        assert any("KENTUCKY" in t for t in texts)

    def test_other_fields_do_not_resolve_inside_the_warning(self, sample_label_lines) -> None:
        extraction = StubProvider(lines=sample_label_lines).extract(None)
        result = verify(extraction, SPIRITS_RECORD, diagnostics=True)
        warning_box = to_bbox(result.field(FIELD))

        for field in result.fields:
            if field.field_id != FIELD and field.bbox is not None:
                assert not to_bbox(field).overlaps(warning_box), field.field_id

        trace = {entry["field"]: entry for entry in result.diagnostics["resolution"]}
        assert trace[FIELD]["candidates_remaining"] < trace[FIELD]["candidates_available"]

    def test_mismatch_always_carries_a_region(self) -> None:
        """Enforced by FieldResult's validator; asserted here on a real MISMATCH."""
        lines = replaced(WARNING_LINES, 0, "GOVERNMENT WARNING", "Government Warning")
        extraction = StubProvider(lines=label(lines)).extract(None)
        result = verify(extraction, SPIRITS_RECORD).field(FIELD)
        assert result.verdict is Verdict.MISMATCH
        assert result.bbox is not None and result.bbox.width > 0

    def test_deterministic(self, sample_label_lines) -> None:
        first, second = run(sample_label_lines), run(sample_label_lines)
        assert first == second
