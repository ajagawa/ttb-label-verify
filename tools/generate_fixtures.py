"""Generate the synthetic label corpus and its ground truth.

Run with `make fixtures` (``python -m tools.generate_fixtures``). Writes, under
``fixtures/``:

*   ``labels/<id>.png`` (or ``.jpg`` for photographic-degradation variants) —
    a label composed with Pillow from vendored SIL OFL fonts.
*   ``labels/<id>.png.ocr.json`` — the **sidecar**: every word that was drawn,
    with its pixel box and a synthetic confidence. The stub provider
    (`extraction/providers/stub.py`) replays it verbatim, which is what lets
    `tools/evaluate.py` measure the rules engine against perfect OCR.
*   ``expected.json`` — per fixture, the application record it is checked
    against and the verdict(s) a *correct* tool should return per field.
*   ``batch_manifest.csv`` — the corpus as a batch-upload manifest, one row
    per fixture, columns per `api.batch_models.MANIFEST_COLUMNS`.
*   ``batch_manifest_with_issues.csv`` — a six-row sibling with exactly three
    deliberate pairing defects, for pairing tests and the frontend's e2e.

Why generate rather than collect
--------------------------------
Three reasons, the same ones THIRD-PARTY-NOTICES.md section 5 gives: ground
truth is exact (we know what was drawn because we drew it), non-compliant
variants can be produced deliberately (a real title-case warning label is hard
to find on purpose), and there is no third-party artwork to license. The cost
is realism, which TRADEOFFS.md states: numbers measured here are optimistic.

Why the spec and the expectation live in one table
--------------------------------------------------
Each `Fixture` below carries both what to draw and what the tool should say
about it. Keeping them together means a fixture cannot be edited without its
expected verdict being in view, and `expected.json` is generated rather than
hand-maintained, so it cannot drift from the images.

Why expected verdicts are *sets*
--------------------------------
The docs' policy is over-flag, and it deliberately leaves REVIEW-versus-
MISMATCH to the strategy for a located value that differs (a one-digit ABV
difference is "REVIEW — single-digit OCR confusion is possible" in
docs/field-assignment.md, while a clear located difference may be MISMATCH).
Pinning one of the two would make the harness grade a policy choice rather
than correctness. So each field carries an ``allowed`` list. The distinction
that *is* non-negotiable — a violation must never come back MATCH, and a field
that is absent must never come back MISMATCH — is fully encoded.

Determinism
-----------
Every random choice (per-word confidence jitter, noise) is drawn from a
generator seeded from the fixture id via SHA-256, never from Python's
``hash()``, which is salted per process. PNGs are written without metadata.
Regenerating on the same Pillow/FreeType build produces identical bytes; a
different FreeType version may rasterise glyphs slightly differently, which is
why the tests compare two generations in one environment rather than against
the committed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
FONTS_DIR = FIXTURES_DIR / "fonts"

#: Sub-directory of the output root that holds images and sidecars. Kept
#: separate from `expected.json` and the fonts so a glob over `labels/` sees
#: only the corpus.
LABELS_SUBDIR = "labels"

#: Must match `extraction.providers.stub.SIDECAR_SUFFIX`. Duplicated rather
#: than imported so the generator has no dependency on the code it is
#: producing test data for; a test asserts the two agree.
SIDECAR_SUFFIX = ".ocr.json"

#: Bumped when the expected.json layout changes, so the evaluator can refuse a
#: file it does not understand rather than misreading it.
EXPECTED_SCHEMA_VERSION = 1

#: The rule set these expectations are written against. A future ttb-v2 with
#: different warning text would need its own corpus, not a silent reuse.
RULESET_VERSION = "ttb-v1"

#: Canvas size. Portrait, like a front label, and below the pipeline's working
#: size (`extraction.pipeline.TARGET_LONGEST_EDGE`, now 2600px), so a fixture
#: is upscaled rather than shrunk. Shrinking is the direction that destroys
#: ink, and the warning statement is the smallest type on the label — a corpus
#: that arrived pre-shrunk would be harder to read than any real submission.
#: Sidecar boxes are in *saved-image* pixels and the stub replays them as
#: given; a live OCR run reports boxes in the rescaled space instead. Each
#: mode is internally consistent, and nothing compares boxes across them.
CANVAS_W = 1000
CANVAS_H = 1400

#: Horizontal space available to text, inside the decorative border.
CONTENT_LEFT = 70
CONTENT_RIGHT = CANVAS_W - 70
CONTENT_W = CONTENT_RIGHT - CONTENT_LEFT

#: 27 CFR 16.21, verbatim. Duplicated from rules/ttb-v1.yaml on purpose: the
#: corpus is ground truth *for* the rule set and must not change just because
#: someone edits the YAML. `tests/test_fixtures.py` asserts they are equal, so
#: an amendment is a deliberate two-place change.
STATUTORY_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)
WARNING_HEADER = "GOVERNMENT WARNING:"
WARNING_BODY = STATUTORY_WARNING[len(WARNING_HEADER) :].strip()

# Verdict shorthands. `NOT_MATCH` is the set for "a violation that the correct
# tool must flag, at REVIEW or MISMATCH as its evidence warrants".
MATCH = ("MATCH",)
REVIEW = ("REVIEW",)
NOT_MATCH = ("REVIEW", "MISMATCH")
MATCH_OR_REVIEW = ("MATCH", "REVIEW")

FIELD_IDS = ("brand_name", "class_type", "alcohol_content", "net_contents", "government_warning")

#: Warning sub-checks the evaluator grades (the two binding ones — see
#: rules/ttb-v1.yaml). Advisory checks are measurements and are not graded.
GRADED_CHECKS = ("header_capitalisation", "text_accuracy")

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
#
# All vendored from the npm `@fontsource/*` packages (v5.3.0), which repackage
# Google Fonts under SIL OFL 1.1. Chosen specifically because none of the
# three declares a Reserved Font Name: fontsource ships *subset* files, and a
# subset is arguably a "Modified Version" under the OFL, which may not carry a
# reserved name. Picking RFN-free families sidesteps the question. WOFF (zlib)
# rather than WOFF2 because FreeType reads WOFF natively; WOFF2 needs Brotli.
# Provenance and hashes: fixtures/README.md.

FONT_FILES = {
    "body": FONTS_DIR / "open-sans" / "open-sans-latin-400-normal.woff",
    "body_bold": FONTS_DIR / "open-sans" / "open-sans-latin-700-normal.woff",
    "serif_display": FONTS_DIR / "libre-baskerville" / "libre-baskerville-latin-700-normal.woff",
    "condensed_display": FONTS_DIR / "oswald" / "oswald-latin-700-normal.woff",
}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache) a vendored font by role and pixel size.

    Raises a readable error if the fonts are missing rather than letting
    Pillow fall back to its bitmap default, which would silently produce a
    corpus with completely different geometry.
    """
    key = (role, size)
    if key not in _font_cache:
        path = FONT_FILES[role]
        if not path.exists():
            raise SystemExit(
                f"font file missing: {path}. The fixture fonts are vendored in "
                "fixtures/fonts/ — see fixtures/README.md for where they came from."
            )
        _font_cache[key] = ImageFont.truetype(str(path), size)
    return _font_cache[key]


# ---------------------------------------------------------------------------
# Fixture specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Background, text and border colours for one label design."""

    background: tuple[int, int, int] = (247, 241, 227)  # cream stock
    ink: tuple[int, int, int] = (28, 24, 20)
    accent: tuple[int, int, int] = (120, 30, 30)  # brand and border


CREAM = Palette()
NAVY = Palette(background=(22, 34, 58), ink=(240, 236, 224), accent=(214, 180, 106))
WHITE_GREEN = Palette(background=(252, 252, 250), ink=(20, 40, 30), accent=(30, 90, 60))
KRAFT = Palette(background=(214, 190, 150), ink=(40, 28, 16), accent=(70, 40, 20))
#: Deliberately poor: text only slightly darker than the stock. Still above
#: the pipeline's usability floor in intent — it exists to exercise the OCR
#: path, not to be rejected outright.
LOW_CONTRAST = Palette(background=(206, 200, 188), ink=(150, 144, 134), accent=(160, 140, 130))


@dataclass(frozen=True)
class WarningBlock:
    """What the warning block says, as printed.

    ``header`` is drawn in the bold weight and ``body`` in the regular weight,
    because 27 CFR 16.22(a)(2) requires exactly that and the advisory bold
    check needs a real contrast to measure. ``header=None`` omits the header
    (body intact) — the "header missing" fixture.
    """

    header: str | None = WARNING_HEADER
    body: str = WARNING_BODY


@dataclass(frozen=True)
class Fixture:
    """One label: what to draw, what it is checked against, what to expect.

    Attributes:
        id: File stem and expected.json key.
        brand_lines: Brand display, one entry per printed line.
        class_type, abv, net: As printed. ``None`` omits the element.
        abv_net_same_line: Print ABV and net contents on one line separated by
            a bullet — the "two fields share a line" layout.
        extra_lines: Extra centred lines under the ABV (e.g. a separate proof
            statement), in body type.
        bottler: Fine-print bottler statement, or ``None``.
        warning: Warning block, or ``None`` for "warning entirely absent".
        record: `ApplicationRecord` values.
        expect: Allowed verdicts per field id.
        checks: Allowed outcomes per graded warning check id. Omitted ids are
            not graded for this fixture.
        elevated: Expected ``FieldResult.elevated`` per field id, where it is
            asserted (the absent-warning case).
        why: One-line rationale, shown by the evaluator on disagreement.
        field_why: Per-field rationale where the verdict is a judgment call.
        tags: Scenario tags for filtering and reporting.
        degrade: ``None``, ``"low_contrast"``, ``"rotate"`` or ``"noise_blur"``.
    """

    id: str
    brand_lines: tuple[str, ...]
    class_type: str | None
    abv: str | None
    net: str | None
    record: dict
    why: str
    tags: tuple[str, ...]
    expect: dict[str, tuple[str, ...]] = field(default_factory=dict)
    checks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    elevated: dict[str, bool] = field(default_factory=dict)
    field_why: dict[str, str] = field(default_factory=dict)
    abv_net_same_line: bool = False
    extra_lines: tuple[str, ...] = ()
    bottler: str | None = None
    warning: WarningBlock | None = field(default_factory=WarningBlock)
    display_font: str = "serif_display"
    palette: Palette = CREAM
    degrade: str | None = None


#: The brief's example, used as the baseline and as the base for single-change
#: variants. A variant that differs from this in exactly one respect isolates
#: exactly one behaviour, which is what makes a failure diagnosable.
OLD_TOM_RECORD = {
    "brand_name": "OLD TOM DISTILLERY",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol.",
    "net_contents": "750 mL",
}

#: Standard graded outcome for a correct, correctly capitalised warning.
WARNING_OK = {"header_capitalisation": ("pass",), "text_accuracy": ("pass",)}


def _all_match(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Every field MATCH except the named overrides.

    Most fixtures change one thing; spelling out five verdicts each time would
    bury that one thing.
    """
    expect = dict.fromkeys(FIELD_IDS, MATCH)
    expect.update(overrides)
    return expect


OLD_TOM = Fixture(
    id="old_tom_compliant",
    brand_lines=("OLD TOM DISTILLERY",),
    class_type="Kentucky Straight Bourbon Whiskey",
    abv="45% Alc./Vol. (90 Proof)",
    net="750 mL",
    record=OLD_TOM_RECORD,
    expect=_all_match(),
    checks=WARNING_OK,
    why="The brief's example label, fully compliant and matching its application.",
    tags=("baseline", "compliant", "abv_with_proof"),
)


def _variant(fixture_id: str, **changes) -> Fixture:
    """A single-change variant of the Old Tom baseline."""
    return replace(OLD_TOM, id=fixture_id, **changes)


FIXTURES: tuple[Fixture, ...] = (
    OLD_TOM,
    # -- layout: fields that do not respect line boundaries -----------------
    Fixture(
        id="brand_two_lines",
        brand_lines=("LANTERN HOLLOW", "DISTILLING CO."),
        class_type="Straight Rye Whiskey",
        abv="50% Alc./Vol. (100 Proof)",
        net="750 mL",
        record={
            "brand_name": "LANTERN HOLLOW DISTILLING CO.",
            "class_type": "Straight Rye Whiskey",
            "alcohol_content": "50% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(),
        checks=WARNING_OK,
        why="Brand wrapped over two display lines; must be recovered as one multi-line span.",
        tags=("compliant", "layout", "brand_wrapped"),
        palette=NAVY,
    ),
    Fixture(
        id="shared_line_abv_net",
        brand_lines=("GREY HERON",),
        class_type="London Dry Gin",
        abv="47% Alc./Vol.",
        net="750 mL",
        abv_net_same_line=True,
        record={
            "brand_name": "GREY HERON",
            "class_type": "London Dry Gin",
            "alcohol_content": "47% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(),
        checks=WARNING_OK,
        why="Net contents and ABV share one printed line; each must be recovered as a sub-line span.",
        tags=("compliant", "layout", "shared_line"),
        display_font="condensed_display",
        palette=WHITE_GREEN,
    ),
    Fixture(
        id="brand_twice_bottler",
        brand_lines=("BRASS KESTREL",),
        class_type="Straight Rye Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net="750 mL",
        bottler="Bottled by BRASS KESTREL, Lawrenceburg, Indiana",
        record={
            "brand_name": "BRASS KESTREL",
            "class_type": "Straight Rye Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(),
        checks=WARNING_OK,
        why="Brand appears in display type and again in the bottler line; either is a MATCH, display preferred.",
        tags=("compliant", "layout", "brand_twice", "bottler_line"),
    ),
    # -- government warning ------------------------------------------------
    _variant(
        "warning_title_case",
        warning=WarningBlock(header="Government Warning:"),
        expect=_all_match(government_warning=NOT_MATCH),
        checks={"header_capitalisation": ("fail",), "text_accuracy": ("pass",)},
        field_why={
            "government_warning": "Header in title case violates 16.22(a)(2); binding check.",
            "text_accuracy": "Wording is verbatim; capitalisation is graded by its own check.",
        },
        why="Jenny Park's rejected label: 'Government Warning' in title case. Caught only by reading raw text.",
        tags=("non_compliant", "warning", "title_case_header", "binding"),
    ),
    _variant(
        "warning_title_case_low_contrast",
        warning=WarningBlock(header="Government Warning:"),
        palette=LOW_CONTRAST,
        degrade="low_contrast",
        # REVIEW *only*, and that is the whole point of this fixture. The
        # violation is systematic, so the capitalisation check must still
        # FAIL — but the image is one the quality gate has flagged, and on a
        # flagged image a typographic assertion is withdrawn to REVIEW,
        # because "GOVERNMENT WARNiNG" is far likelier to be a misread than a
        # label genuinely printed that way. With the guard: REVIEW. Without
        # it: MISMATCH. No other fixture separates the two, so without this
        # one the guard could be deleted and the corpus would not notice.
        expect=_all_match(government_warning=REVIEW),
        checks={"header_capitalisation": ("fail",), "text_accuracy": ("pass",)},
        field_why={
            "government_warning": (
                "Header is miscapitalised and the check must fail, but the quality gate flagged "
                "this image (low contrast), so the verdict is withdrawn to REVIEW rather than "
                "asserted as MISMATCH. MISMATCH here means the degraded-image guard is gone."
            ),
            "header_capitalisation": (
                "The violation is systematic, not a sporadic mis-cased letter: the check fails "
                "on the evidence even though the field verdict is withheld."
            ),
        },
        why="Title-case header on a flagged low-contrast image: the check fails, the assertion is withdrawn to REVIEW.",
        tags=(
            "non_compliant",
            "warning",
            "title_case_header",
            "degraded",
            "low_contrast",
            "quality_guard",
            "binding",
        ),
    ),
    _variant(
        "warning_can_cause",
        warning=WarningBlock(body=WARNING_BODY.replace("may cause", "can cause")),
        expect=_all_match(government_warning=NOT_MATCH),
        checks={"header_capitalisation": ("pass",), "text_accuracy": ("fail",)},
        why="One word changed ('can cause' for 'may cause'). Scores ~0.99 fuzzily; must be caught by a word diff.",
        tags=("non_compliant", "warning", "wording_change", "binding"),
    ),
    _variant(
        "warning_header_missing",
        warning=WarningBlock(header=None),
        expect=_all_match(government_warning=NOT_MATCH),
        checks={
            # With no header there is nothing to judge the case of. Either
            # answer is defensible; PASS is not (it would claim the header is
            # correctly capitalised).
            "header_capitalisation": ("fail", "not_evaluable"),
            "text_accuracy": ("fail",),
        },
        why="Warning body intact but 'GOVERNMENT WARNING:' omitted; anchored via tail phrases, statement incomplete.",
        tags=("non_compliant", "warning", "header_missing", "binding"),
    ),
    _variant(
        "warning_absent",
        warning=None,
        expect=_all_match(government_warning=REVIEW),
        elevated={"government_warning": True},
        # Nothing to check, so neither check may claim PASS: "not evaluated"
        # must never read as "passed". FAIL is also honest (no compliant
        # header or text exists on the label).
        checks={
            "header_capitalisation": ("not_evaluable", "fail"),
            "text_accuracy": ("not_evaluable", "fail"),
        },
        field_why={
            "government_warning": "Not found is never MISMATCH; mandatory + good image => REVIEW, elevated.",
        },
        why="No warning statement at all on a clean image: REVIEW at elevated prominence, never MISMATCH.",
        tags=("non_compliant", "warning", "warning_absent", "not_found"),
    ),
    # -- alcohol content ---------------------------------------------------
    _variant(
        "abv_mismatch",
        abv="46% Alc./Vol. (92 Proof)",
        expect=_all_match(alcohol_content=NOT_MATCH),
        why="Label 46% vs application 45%. Numeric fields are never fuzzy-forgiven.",
        tags=("non_compliant", "abv", "abv_mismatch"),
    ),
    Fixture(
        id="abv_proof_separate_line",
        brand_lines=("HALVARD & PIKE",),
        class_type="Vodka",
        abv="40% Alc./Vol.",
        extra_lines=("80 Proof",),
        net="1 L",
        record={
            "brand_name": "HALVARD & PIKE",
            "class_type": "Vodka",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "1 L",
        },
        expect=_all_match(),
        checks=WARNING_OK,
        why="ABV and proof both present, consistent (80 = 2 x 40), on separate lines.",
        tags=("compliant", "abv", "abv_with_proof"),
        display_font="condensed_display",
        palette=NAVY,
    ),
    _variant(
        "abv_proof_inconsistent",
        abv="45% Alc./Vol. (92 Proof)",
        expect=_all_match(alcohol_content=NOT_MATCH),
        why="ABV matches the application but proof (92) is not 2 x 45: an internal inconsistency on the label.",
        tags=("non_compliant", "abv", "abv_proof_inconsistent"),
    ),
    _variant(
        "abv_proof_only",
        abv="90 Proof",
        expect=_all_match(alcohol_content=NOT_MATCH),
        field_why={
            "alcohol_content": (
                "90 proof equals 45% ABV, but 27 CFR 5.65 requires alcohol content as percent "
                "by volume; proof may only appear in addition. Flag rather than MATCH."
            ),
        },
        why="Alcohol content stated only as proof. Numerically equivalent, but not the required statement.",
        tags=("non_compliant", "abv", "proof_only", "judgment"),
    ),
    # -- net contents ------------------------------------------------------
    _variant(
        "net_litres",
        net="0.75 L",
        expect=_all_match(),
        why="0.75 L on the label, 750 mL on the application: same volume after normalising to mL.",
        tags=("compliant", "net", "unit_litres"),
    ),
    _variant(
        "net_centilitres",
        net="75 cL",
        expect=_all_match(net_contents=MATCH_OR_REVIEW),
        field_why={
            "net_contents": (
                "75 cL = 750 mL (docs/field-assignment.md normalises cL). Whether cL is an "
                "acceptable unit on a US label is a question for the agent, so REVIEW is also correct."
            ),
        },
        why="Net contents in centilitres; numerically equal to the application.",
        tags=("net", "unit_centilitres", "judgment"),
    ),
    _variant(
        "net_dual_units",
        net="750 mL (25.4 FL. OZ.)",
        expect=_all_match(),
        why="Metric with a US customary equivalent; 25.4 fl oz is within the 2 mL reading tolerance.",
        tags=("compliant", "net", "unit_fl_oz"),
    ),
    _variant(
        "net_nonstandard_fill",
        net="730 mL",
        record={**OLD_TOM_RECORD, "net_contents": "730 mL"},
        expect=_all_match(net_contents=NOT_MATCH),
        why="Label and application agree on 730 mL, which is not a 27 CFR 5.203 standard of fill.",
        tags=("non_compliant", "net", "standard_of_fill"),
    ),
    _variant(
        "net_mismatch",
        net="700 mL",
        expect=_all_match(net_contents=NOT_MATCH),
        why="Label 700 mL vs application 750 mL (700 is itself a standard fill, so only the comparison catches it).",
        tags=("non_compliant", "net", "net_mismatch"),
    ),
    _variant(
        "net_missing",
        net=None,
        expect=_all_match(net_contents=REVIEW),
        field_why={"net_contents": "Not located => REVIEW. Never MISMATCH."},
        why="No net contents statement printed. Absence resolves to REVIEW, not MISMATCH.",
        tags=("non_compliant", "net", "not_found"),
    ),
    # -- class/type --------------------------------------------------------
    Fixture(
        id="class_vodka_vs_gin",
        brand_lines=("GREY HERON",),
        class_type="Vodka",
        abv="40% Alc./Vol. (80 Proof)",
        net="750 mL",
        record={
            "brand_name": "GREY HERON",
            "class_type": "Gin",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(class_type=NOT_MATCH),
        checks=WARNING_OK,
        why="Label says Vodka, application says Gin. Vocabulary fallback should report what it found.",
        tags=("non_compliant", "class_type", "class_mismatch"),
        display_font="condensed_display",
        palette=WHITE_GREEN,
    ),
    _variant(
        "class_abbreviation_ky",
        class_type="Ky. Straight Bourbon Whiskey",
        expect=_all_match(),
        field_why={
            "class_type": "rules/ttb-v1.yaml declares KY -> KENTUCKY; the equivalence is explicit.",
        },
        why="'Ky.' abbreviation for Kentucky; the rule set's abbreviation table makes this a MATCH.",
        tags=("compliant", "class_type", "abbreviation"),
    ),
    # -- brand name --------------------------------------------------------
    Fixture(
        id="brand_case_only",
        brand_lines=("STONE'S THROW",),
        class_type="Blanco Tequila",
        abv="40% Alc./Vol. (80 Proof)",
        net="750 mL",
        record={
            "brand_name": "Stone's Throw",
            "class_type": "Blanco Tequila",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(brand_name=MATCH_OR_REVIEW),
        checks=WARNING_OK,
        field_why={
            "brand_name": (
                "Case-only difference. rules/results.py calls it 'technically a mismatch and "
                "obviously the same thing'; MATCH or REVIEW are both correct, MISMATCH is not."
            ),
        },
        why="STONE'S THROW on the label, Stone's Throw in the application.",
        tags=("brand", "case_only", "judgment"),
        palette=KRAFT,
    ),
    Fixture(
        id="brand_one_letter_off",
        brand_lines=("MARROW CREEK",),
        class_type="Bourbon Whiskey",
        abv="43% Alc./Vol. (86 Proof)",
        net="750 mL",
        record={
            "brand_name": "HARROW CREEK",
            "class_type": "Bourbon Whiskey",
            "alcohol_content": "43% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(brand_name=NOT_MATCH),
        checks=WARNING_OK,
        field_why={
            "brand_name": (
                "A different brand one letter away. Fuzzy tolerance exists to forgive OCR, "
                "and here it must not forgive a real difference on perfect text."
            ),
        },
        why="Label MARROW CREEK vs application HARROW CREEK. Probes whether fuzzy brand tolerance hides a real mismatch.",
        tags=("non_compliant", "brand", "near_miss"),
        palette=KRAFT,
    ),
    # -- one-character differences in LONG values ----------------------------
    #
    # `brand_one_letter_off` above passes, but only because its name is short:
    # similarity is a ratio, so one substituted letter costs less the longer
    # the string. On an 18-character brand the same defect scored 0.944 against
    # a 0.92 match threshold and was reported as MATCH — a false negative that
    # the corpus could not see, because every one-letter fixture in it was
    # short. These three exist to make that failure visible, and they were
    # added *before* the fix so the harness could be shown to catch it.
    Fixture(
        id="brand_long_one_letter_off",
        brand_lines=("OLD TON DISTILLERY",),
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net="750 mL",
        record={
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(brand_name=NOT_MATCH),
        checks=WARNING_OK,
        field_why={
            "brand_name": (
                "One letter different on an 18-character name. The same defect as "
                "brand_one_letter_off, but long enough that a ratio-based score "
                "clears the match threshold."
            ),
        },
        why="Label OLD TON DISTILLERY vs application OLD TOM DISTILLERY. The long-name false negative.",
        tags=("non_compliant", "brand", "near_miss", "long_value"),
    ),
    Fixture(
        id="brand_misspelled_on_label",
        brand_lines=("OLD TOM DISTILERY",),
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net="750 mL",
        record={
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(brand_name=NOT_MATCH),
        checks=WARNING_OK,
        field_why={
            "brand_name": (
                "The label is genuinely misspelled — a dropped letter. From the text "
                "alone this is indistinguishable from an OCR misread, which is exactly "
                "why it must reach a person rather than be forgiven."
            ),
        },
        why="Label printed OLD TOM DISTILERY against application OLD TOM DISTILLERY.",
        tags=("non_compliant", "brand", "near_miss", "long_value"),
    ),
    Fixture(
        id="class_long_one_letter_off",
        brand_lines=("OLD TOM DISTILLERY",),
        class_type="Kentucky Straight Burbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net="750 mL",
        record={
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol.",
            "net_contents": "750 mL",
        },
        expect=_all_match(class_type=NOT_MATCH),
        checks=WARNING_OK,
        field_why={
            "class_type": (
                "One letter missing from a 33-character designation. Scored 0.970 "
                "before the fix — the ratio problem at its most extreme."
            ),
        },
        why="Label Kentucky Straight Burbon Whiskey vs application Kentucky Straight Bourbon Whiskey.",
        tags=("non_compliant", "class_type", "near_miss", "long_value"),
    ),
    # -- degraded images (sidecar is still exact ground truth) --------------
    _variant(
        "degraded_low_contrast",
        palette=LOW_CONTRAST,
        degrade="low_contrast",
        expect=_all_match(),
        why="Compliant baseline printed with poor ink/stock contrast. Exercises the real OCR path.",
        tags=("compliant", "degraded", "low_contrast"),
    ),
    _variant(
        "degraded_rotated",
        degrade="rotate",
        expect=_all_match(),
        why="Compliant baseline rotated 2.5 degrees, as from a hand-held photo. Boxes are rotated with it.",
        tags=("compliant", "degraded", "rotation"),
    ),
    _variant(
        "degraded_noise_blur",
        degrade="noise_blur",
        expect=_all_match(),
        why="Compliant baseline with blur, sensor noise and JPEG compression.",
        tags=("compliant", "degraded", "noise", "blur"),
    ),
)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


@dataclass
class DrawnWord:
    """One word as drawn, in pixel coordinates of the saved image."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float = 0.98


@dataclass
class Canvas:
    """A label image plus the record of every word drawn on it.

    Drawing and recording happen in the same call (`draw_line`) so the sidecar
    cannot describe text that was not drawn, or omit text that was.
    """

    image: Image.Image
    draw: ImageDraw.ImageDraw
    palette: Palette
    lines: list[list[DrawnWord]] = field(default_factory=list)

    @classmethod
    def blank(cls, palette: Palette) -> Canvas:
        image = Image.new("RGB", (CANVAS_W, CANVAS_H), palette.background)
        draw = ImageDraw.Draw(image)
        # Decorative double border. Drawn as rules, not text, so it appears in
        # the image (and challenges real OCR and the separation check) without
        # appearing in the sidecar.
        draw.rectangle((28, 28, CANVAS_W - 29, CANVAS_H - 29), outline=palette.accent, width=4)
        draw.rectangle((40, 40, CANVAS_W - 41, CANVAS_H - 41), outline=palette.accent, width=1)
        return cls(image=image, draw=draw, palette=palette)

    def draw_line(
        self,
        runs: list[tuple[str, ImageFont.FreeTypeFont]],
        *,
        baseline: float,
        align: str = "center",
        color: tuple[int, int, int] | None = None,
    ) -> float:
        """Draw one visual line of words and record their boxes.

        Args:
            runs: ``(word, font)`` pairs, in order. A line may mix fonts — the
                warning's first line starts bold and continues regular — so
                fonts are per word, not per line.
            baseline: Y of the shared baseline. Words are anchored on the
                baseline (``anchor="ls"``) so bold and regular runs of the same
                size line up exactly as typeset text does.
            align: ``"center"`` or ``"left"`` (left edge at CONTENT_LEFT).
            color: Ink colour; defaults to the palette's.

        Returns:
            The width of the drawn line, for callers laying out columns.

        Word boxes take their **horizontal** extent from ``textbbox`` of the
        word itself (tight ink bounds), and their **vertical** extent from the
        font's ascender-to-descender box for that size. The vertical choice is
        deliberate: a tight box around "a" is half the height of one around
        "(1)", and span coherence in `extraction/spans.py` compares glyph
        heights between lines. Real OCR engines report line-height boxes, so
        this is also the more faithful model of what a provider returns.
        """
        color = color or self.palette.ink
        widths = [f.getlength(w) for w, f in runs]
        spaces = [f.getlength(" ") for _, f in runs[:-1]]
        total = sum(widths) + sum(spaces)
        x = CONTENT_LEFT if align == "left" else (CANVAS_W - total) / 2

        recorded: list[DrawnWord] = []
        for index, (word, word_font) in enumerate(runs):
            self.draw.text((x, baseline), word, font=word_font, fill=color, anchor="ls")
            ink = self.draw.textbbox((x, baseline), word, font=word_font, anchor="ls")
            ascender, descender = _vertical_extent(word_font)
            recorded.append(
                DrawnWord(
                    text=word,
                    left=ink[0],
                    top=baseline - ascender,
                    right=ink[2],
                    bottom=baseline + descender,
                )
            )
            x += widths[index] + (spaces[index] if index < len(spaces) else 0)

        self.lines.append(recorded)
        return total


def _vertical_extent(word_font: ImageFont.FreeTypeFont) -> tuple[float, float]:
    """Cap-height-plus-descender extent of a font, relative to the baseline.

    Measured from a reference string with a tall capital and a deep descender
    rather than from `getmetrics()`, whose ascent includes line-gap padding
    that no OCR box would.
    """
    box = word_font.getbbox("Hgjy(", anchor="ls")
    return float(-box[1]), float(box[3])


def _fit_size(text: str, role: str, start: int, max_width: float, floor: int = 40) -> int:
    """Largest size <= `start` at which `text` fits `max_width`."""
    size = start
    while size > floor and font(role, size).getlength(text) > max_width:
        size -= 2
    return size


def _wrap(
    tokens: list[tuple[str, ImageFont.FreeTypeFont]], max_width: float
) -> list[list[tuple[str, ImageFont.FreeTypeFont]]]:
    """Greedy word wrap over mixed-font tokens."""
    lines: list[list[tuple[str, ImageFont.FreeTypeFont]]] = [[]]
    width = 0.0
    for word, word_font in tokens:
        word_w = word_font.getlength(word)
        space = word_font.getlength(" ") if lines[-1] else 0.0
        if lines[-1] and width + space + word_w > max_width:
            lines.append([])
            width, space = 0.0, 0.0
        lines[-1].append((word, word_font))
        width += space + word_w
    return lines


def compose(fixture: Fixture) -> Canvas:
    """Lay out a label top to bottom and return the drawn canvas.

    Layout is the conventional front-label order: brand in display type, then
    class/type directly beneath (the spatial prior class/type resolution uses),
    then alcohol content and net contents, then the bottler statement in fine
    print, then the warning block at the foot, set left-aligned in small type
    and separated from everything above by whitespace.
    """
    canvas = Canvas.blank(fixture.palette)
    y = 150.0

    # Brand. Every line of a wrapped brand is set at the same size — the size
    # at which the longest line fits — as a designer would.
    longest = max(fixture.brand_lines, key=lambda s: font(fixture.display_font, 96).getlength(s))
    brand_size = _fit_size(longest, fixture.display_font, 96, CONTENT_W - 40)
    brand_font = font(fixture.display_font, brand_size)
    for text in fixture.brand_lines:
        y += brand_size
        canvas.draw_line(
            [(w, brand_font) for w in text.split()], baseline=y, color=fixture.palette.accent
        )
        y += brand_size * 0.25

    if fixture.class_type:
        y += 70
        class_font = font("body_bold", _fit_size(fixture.class_type, "body_bold", 40, CONTENT_W))
        canvas.draw_line([(w, class_font) for w in fixture.class_type.split()], baseline=y)

    y += 60
    info_font = font("body", 34)
    info_lines: list[str] = []
    if fixture.abv_net_same_line and fixture.abv and fixture.net:
        info_lines.append(f"{fixture.net} • {fixture.abv}")
    else:
        info_lines += [s for s in (fixture.abv, *fixture.extra_lines, fixture.net) if s]
    for text in info_lines:
        y += 60
        canvas.draw_line([(w, info_font) for w in text.split()], baseline=y)

    if fixture.bottler:
        y += 70
        canvas.draw_line([(w, font("body", 22)) for w in fixture.bottler.split()], baseline=y)

    if fixture.warning is not None:
        _draw_warning(canvas, fixture.warning)

    return canvas


#: Warning type size in pixels. Small, as on a real label, but large enough
#: that a real OCR engine at 1600px has a fair chance — the corpus is meant to
#: exercise the rules first and OCR second.
WARNING_SIZE = 21
WARNING_LEADING = 1.45


def _draw_warning(canvas: Canvas, warning: WarningBlock) -> None:
    """Set the warning block at the foot of the label.

    The header is bold and the body regular (27 CFR 16.22(a)(2)). Wrapping is
    computed first so the block can be anchored to the bottom margin, leaving
    whitespace above it — the "separate and apart" requirement of 16.21.
    """
    bold = font("body_bold", WARNING_SIZE)
    regular = font("body", WARNING_SIZE)
    tokens: list[tuple[str, ImageFont.FreeTypeFont]] = []
    if warning.header:
        tokens += [(w, bold) for w in warning.header.split()]
    tokens += [(w, regular) for w in warning.body.split()]

    wrapped = _wrap(tokens, CONTENT_W)
    step = WARNING_SIZE * WARNING_LEADING
    baseline = CANVAS_H - 80 - step * (len(wrapped) - 1)
    for runs in wrapped:
        canvas.draw_line(runs, baseline=baseline, align="left")
        baseline += step


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

#: Rotation for the "hand-held photo" variant. Small on purpose: large enough
#: that line boxes are no longer tight, small enough that the label is still a
#: plausible upload rather than a resubmit request.
ROTATION_DEGREES = 2.5


def _rng(fixture_id: str, purpose: str) -> np.random.Generator:
    """A generator seeded from the fixture id and purpose.

    SHA-256 rather than `hash()`: Python salts string hashes per process, so
    `hash()` would make every regeneration different.
    """
    digest = hashlib.sha256(f"{fixture_id}:{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _rotate_point(x: float, y: float, degrees: float) -> tuple[float, float]:
    """Where a point lands after `Image.rotate(degrees)` about the centre.

    Pillow rotates counter-clockwise as seen on screen. With y pointing down,
    that is ``x' = cx + dx cos + dy sin``, ``y' = cy - dx sin + dy cos``.
    """
    theta = math.radians(degrees)
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    dx, dy = x - cx, y - cy
    return (
        cx + dx * math.cos(theta) + dy * math.sin(theta),
        cy - dx * math.sin(theta) + dy * math.cos(theta),
    )


def degrade(canvas: Canvas, fixture: Fixture) -> None:
    """Apply the fixture's image degradation in place, keeping boxes truthful.

    Only rotation moves text, so only rotation rewrites boxes: each word's box
    becomes the axis-aligned bound of its rotated corners — larger than the
    word, which is also what a real OCR engine reports for tilted text.
    Confidence drops on every degraded variant, since a real engine would be
    less sure; the text itself stays exact ground truth.
    """
    kind = fixture.degrade
    if kind is None:
        return

    if kind == "rotate":
        canvas.image = canvas.image.rotate(
            ROTATION_DEGREES,
            resample=Image.Resampling.BICUBIC,
            fillcolor=fixture.palette.background,
        )
        for line in canvas.lines:
            for word in line:
                corners = [
                    _rotate_point(x, y, ROTATION_DEGREES)
                    for x in (word.left, word.right)
                    for y in (word.top, word.bottom)
                ]
                word.left = min(c[0] for c in corners)
                word.right = max(c[0] for c in corners)
                word.top = min(c[1] for c in corners)
                word.bottom = max(c[1] for c in corners)
    elif kind == "noise_blur":
        blurred = canvas.image.filter(ImageFilter.GaussianBlur(radius=1.3))
        pixels = np.asarray(blurred, dtype=np.float32)
        noise = _rng(fixture.id, "noise").normal(0.0, 14.0, size=pixels.shape)
        canvas.image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    elif kind == "low_contrast":
        pass  # the palette already did the work; nothing further to apply
    else:  # pragma: no cover - guarded by tests on the fixture table
        raise ValueError(f"unknown degradation {kind!r}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def image_filename(fixture: Fixture) -> str:
    """JPEG for the noisy variant, PNG otherwise.

    Noise is incompressible in PNG (several MB per image); JPEG is also what a
    phone photograph actually arrives as, so it is the more faithful format
    for the one variant that imitates a photograph's sensor noise.
    """
    return f"{fixture.id}.jpg" if fixture.degrade == "noise_blur" else f"{fixture.id}.png"


def _assign_confidences(canvas: Canvas, fixture: Fixture) -> None:
    """Deterministic per-word confidence: 0.97-0.99 clean, 0.84-0.93 degraded."""
    low, high = (0.84, 0.93) if fixture.degrade else (0.97, 0.99)
    rng = _rng(fixture.id, "confidence")
    for line in canvas.lines:
        for word in line:
            word.confidence = round(float(rng.uniform(low, high)), 3)


def measure_quality(image_path: Path) -> dict:
    """The pipeline's own quality assessment of the image just written.

    **Measured, not invented.** The number comes from
    `extraction.pipeline.preprocess` run on the saved file — the same call,
    on the same bytes, that a real submission goes through — so the stub
    replays the assessment a real run would produce instead of the stub's
    neutral placeholder.

    That matters for one behaviour the corpus otherwise could not exercise:
    the warning strategy withdraws a typographic assertion (a case
    difference) to REVIEW when the quality gate has flagged the image. With
    no ``quality`` block, every stub-mode verification sees an unflagged image
    and the guard can never fire, so `warning_title_case_low_contrast` would
    come back MISMATCH under the stub and REVIEW under real OCR — the same
    fixture grading differently by mode, which would make the harness's two
    numbers incomparable.

    It does couple the generator to the module under test, which is the thing
    the sidecar convention otherwise avoids. The trade is deliberate and
    narrow: the *text and geometry* stay independent ground truth, and only
    the quality block — which is a measurement of pixels, not a claim about
    what the label says — is taken from the pipeline. Fabricating a plausible
    score instead would be worse: the guard would then fire against a number
    nobody measured.

    Raises:
        ExtractionError: if the pipeline refuses the image. Loud on purpose —
            a fixture the pipeline will not accept is a broken fixture, not a
            fixture with an interesting quality score.
    """
    from extraction.pipeline import preprocess

    quality = preprocess(image_path.read_bytes()).quality
    return {
        "usable": quality.usable,
        "score": round(float(quality.score), 4),
        "issues": list(quality.issues),
        "guidance": list(quality.guidance),
        "metrics": {k: round(float(v), 2) for k, v in quality.metrics.items()},
    }


def sidecar_payload(canvas: Canvas, quality: dict) -> dict:
    """The stub provider's sidecar format (see extraction/providers/stub.py).

    Coordinates are rounded to 0.1px: sub-pixel precision beyond that is
    noise, and rounding keeps the committed JSON diff-able. ``width`` and
    ``height`` are the *saved image's* pixels; the real pipeline rescales
    (it now upscales a small label to 2600px), so boxes from a live OCR run
    live in that prepared space instead — the two modes are each internally
    consistent, and nothing compares boxes across them.
    """
    return {
        "width": canvas.image.width,
        "height": canvas.image.height,
        "quality": quality,
        "lines": [
            {
                "words": [
                    {
                        "text": w.text,
                        "left": round(w.left, 1),
                        "top": round(w.top, 1),
                        "width": round(w.right - w.left, 1),
                        "height": round(w.bottom - w.top, 1),
                        "confidence": w.confidence,
                    }
                    for w in line
                ]
            }
            for line in canvas.lines
        ],
    }


def expected_entry(fixture: Fixture) -> dict:
    """One fixture's entry in expected.json.

    ``compliant`` is derived, not declared: a fixture is compliant when every
    field's only acceptable verdict is MATCH, and non-compliant when at least
    one field's acceptable set excludes MATCH. Fixtures that are neither
    (a judgment call such as the case-only brand) count toward accuracy but
    toward neither the false-flag nor the false-negative totals.
    """
    fields = {}
    for field_id in FIELD_IDS:
        allowed = fixture.expect[field_id]
        entry: dict = {"allowed": list(allowed)}
        if field_id in fixture.field_why:
            entry["why"] = fixture.field_why[field_id]
        if field_id in fixture.elevated:
            entry["elevated"] = fixture.elevated[field_id]
        fields[field_id] = entry

    checks = {}
    for check_id, allowed in fixture.checks.items():
        entry = {"allowed": list(allowed)}
        if check_id in fixture.field_why:
            entry["why"] = fixture.field_why[check_id]
        checks[check_id] = entry

    all_match = all(fixture.expect[f] == MATCH for f in FIELD_IDS)
    any_violation = any("MATCH" not in fixture.expect[f] for f in FIELD_IDS)
    return {
        "id": fixture.id,
        "image": f"{LABELS_SUBDIR}/{image_filename(fixture)}",
        "record": dict(fixture.record),
        "compliance": "compliant"
        if all_match
        else ("non_compliant" if any_violation else "judgment"),
        "fields": fields,
        "checks": checks,
        "why": fixture.why,
        "tags": list(fixture.tags),
        "degradation": fixture.degrade,
    }


def _save_image(image: Image.Image, path: Path) -> None:
    """Save without metadata so the bytes depend only on the pixels."""
    if path.suffix == ".jpg":
        image.save(path, format="JPEG", quality=88, optimize=False, progressive=False)
    else:
        image.save(path, format="PNG", optimize=False, compress_level=9)


def generate(out_dir: Path, only: set[str] | None = None) -> list[str]:
    """Generate the corpus (or a subset) under `out_dir`.

    Args:
        out_dir: Output root; images go in ``out_dir/labels``.
        only: Fixture ids to generate. ``None`` generates all and also clears
            stale files from ``labels/`` so a renamed fixture does not leave
            an orphan image the evaluator would then trip over.

    Returns:
        The ids generated, in table order.
    """
    selected = [f for f in FIXTURES if only is None or f.id in only]
    if only is not None and len(selected) != len(only):
        unknown = sorted(only - {f.id for f in FIXTURES})
        raise SystemExit(f"unknown fixture id(s): {', '.join(unknown)}")

    labels = out_dir / LABELS_SUBDIR
    if only is None and labels.exists():
        shutil.rmtree(labels)
    labels.mkdir(parents=True, exist_ok=True)

    for fixture in selected:
        canvas = compose(fixture)
        degrade(canvas, fixture)
        _assign_confidences(canvas, fixture)
        image_path = labels / image_filename(fixture)
        _save_image(canvas.image, image_path)
        sidecar = image_path.with_name(image_path.name + SIDECAR_SUFFIX)
        payload = sidecar_payload(canvas, measure_quality(image_path))
        sidecar.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    expected = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "ruleset_version": RULESET_VERSION,
        "note": (
            "Generated by tools/generate_fixtures.py; do not edit by hand. 'allowed' lists every "
            "verdict a correct tool may return. Policy: over-flag; not-found is never MISMATCH."
        ),
        "fixtures": [expected_entry(f) for f in selected],
    }
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    write_manifests(out_dir, expected["fixtures"])
    return [f.id for f in selected]


# ---------------------------------------------------------------------------
# Batch manifests
# ---------------------------------------------------------------------------
#
# Generated from the same entries as expected.json, in the same call, so the
# manifest cannot drift from the corpus: a renamed fixture renames its row, and
# a changed application record changes its row. Hand-maintaining a CSV beside a
# generated corpus is exactly the drift `expected.json` was generated to avoid.

MANIFEST_FILENAME = "batch_manifest.csv"
BROKEN_MANIFEST_FILENAME = "batch_manifest_with_issues.csv"

#: The image the broken manifest names that does not exist. Chosen to read as
#: an obvious typo-class mistake to anyone looking at the e2e test's output.
MISSING_IMAGE_NAME = "missing_label.png"


def manifest_columns() -> tuple[str, ...]:
    """Column order from the API contract, the single source of truth.

    Imported lazily: the contract module pulls in pydantic and the result
    models, which the drawing code does not otherwise need.
    """
    from api.batch_models import MANIFEST_COLUMNS

    return MANIFEST_COLUMNS


def manifest_row(entry: dict) -> dict[str, str]:
    """One expected.json entry as a manifest row.

    ``filename`` is the image's basename, because a batch upload pairs by
    basename (`api.manifest.basename`), not by path. Optional columns the
    record does not carry are written empty, which is how an agent's
    spreadsheet would leave them.
    """
    record = entry["record"]
    row = {column: "" for column in manifest_columns()}
    row["filename"] = entry["image"].rsplit("/", 1)[-1]
    for column in row:
        if column != "filename" and record.get(column) is not None:
            row[column] = str(record[column])
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """RFC 4180 CSV with LF endings, deterministic byte for byte."""
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_columns()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def broken_manifest_rows(entries: list[dict]) -> list[dict[str, str]] | None:
    """A small manifest with exactly three deliberate defects, or None.

    Built from the first four fixtures so that pairing it with *the images it
    names* gives a known report:

    *   rows 2-3: two valid rows -> 2 matched pairs
    *   row 4: names `missing_label.png`, which does not exist -> ROW_WITHOUT_IMAGE
    *   row 5: a real image with an empty ``brand_name`` -> INVALID_ROW (and,
        per `api.manifest.pair_files`, *no* extra IMAGE_WITHOUT_ROW for that
        image — the row-side issue already explains it)
    *   rows 6-7: the same row twice -> DUPLICATE_ROW, neither used

    Exactly three issues, each a distinct kind. Returns None for a subset of
    fewer than four fixtures, where the pattern cannot be built.
    """
    if len(entries) < 4:
        return None
    first, second, duplicated, invalid = (manifest_row(e) for e in entries[:4])
    missing = {**first, "filename": MISSING_IMAGE_NAME}
    invalid = {**invalid, "brand_name": ""}
    return [first, second, missing, invalid, duplicated, dict(duplicated)]


def write_manifests(out_dir: Path, entries: list[dict]) -> None:
    """Write the clean corpus manifest and its deliberately broken sibling."""
    _write_csv(out_dir / MANIFEST_FILENAME, [manifest_row(e) for e in entries])
    broken = broken_manifest_rows(entries)
    if broken is not None:
        _write_csv(out_dir / BROKEN_MANIFEST_FILENAME, broken)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=FIXTURES_DIR, help="output root (default: fixtures/)"
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated fixture ids to generate (default: all)"
    )
    args = parser.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    ids = generate(args.out, only)
    print(f"generated {len(ids)} fixture(s) in {args.out / LABELS_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
