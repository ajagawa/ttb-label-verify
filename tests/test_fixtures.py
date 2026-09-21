"""Tests for the synthetic fixture corpus and the evaluation harness.

These guard the *test data*, which is easy to forget needs guarding. A corpus
whose sidecar drifts from its image, or whose expected verdicts are malformed,
does not fail loudly: it quietly produces a performance table that measures
the wrong thing. Each test below pins one property the evaluation numbers
depend on.

Generation of the full corpus takes a few seconds, so most tests generate a
small representative subset into `tmp_path` instead; the committed corpus is
checked only for structure, never for byte equality, because glyph
rasterisation may differ between FreeType builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from extraction.pipeline import TARGET_LONGEST_EDGE
from extraction.providers.stub import SIDECAR_SUFFIX, StubProvider
from rules.results import ApplicationRecord, Verdict
from rules.schema import load_ruleset
from tools import evaluate
from tools import generate_fixtures as gen

REPO_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

#: A subset that covers every code path in the generator: plain layout, the
#: wrapped brand, the shared line, a title-case header, a missing warning, and
#: all three degradations (including the JPEG output and the box rotation).
SUBSET = {
    "old_tom_compliant",
    "brand_two_lines",
    "shared_line_abv_net",
    "warning_title_case",
    "warning_title_case_low_contrast",
    "warning_absent",
    "degraded_low_contrast",
    "degraded_rotated",
    "degraded_noise_blur",
}

VALID_VERDICTS = {v.value for v in Verdict}
VALID_CHECK_OUTCOMES = {"pass", "fail", "advisory", "not_evaluable"}


@pytest.fixture(scope="module")
def subset_dir(tmp_path_factory) -> Path:
    """The subset, generated once for the module."""
    out = tmp_path_factory.mktemp("corpus")
    gen.generate(out, SUBSET)
    return out


def _sidecar(image: Path) -> Path:
    return image.with_name(image.name + SIDECAR_SUFFIX)


def _raw_text(sidecar: Path) -> str:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return " ".join(w["text"] for line in payload["lines"] for w in line["words"])


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_regeneration_is_byte_identical(self, subset_dir: Path, tmp_path: Path) -> None:
        """Same inputs, same bytes — images, sidecars and expected.json.

        Determinism is what lets a changed evaluation number be attributed to
        a code change rather than to the corpus having silently moved.
        """
        gen.generate(tmp_path, SUBSET)
        first = sorted(p.relative_to(subset_dir) for p in subset_dir.rglob("*") if p.is_file())
        second = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
        assert first == second
        for rel in first:
            assert (subset_dir / rel).read_bytes() == (tmp_path / rel).read_bytes(), rel

    def test_sidecar_suffix_matches_stub(self) -> None:
        """The generator duplicates the suffix to avoid importing the code under test."""
        assert gen.SIDECAR_SUFFIX == SIDECAR_SUFFIX

    def test_statutory_text_matches_ruleset(self) -> None:
        """The corpus's warning must be the rule set's warning, word for word.

        Duplicated deliberately (see the generator's comment); this makes an
        amendment to either copy a failing test rather than a silently wrong
        corpus.
        """
        assert load_ruleset().government_warning.statutory_text == gen.STATUTORY_WARNING

    def test_fixture_ids_are_unique(self) -> None:
        ids = [f.id for f in gen.FIXTURES]
        assert len(ids) == len(set(ids))

    def test_corpus_covers_the_required_scenarios(self) -> None:
        """The scenarios docs/field-assignment.md says to write before the code."""
        tags = {t for f in gen.FIXTURES for t in f.tags}
        required = {
            "baseline",
            "brand_wrapped",
            "shared_line",
            "brand_twice",
            "title_case_header",
            "wording_change",
            "header_missing",
            "warning_absent",
            "abv_mismatch",
            "abv_with_proof",
            "abv_proof_inconsistent",
            "proof_only",
            "unit_litres",
            "unit_centilitres",
            "standard_of_fill",
            "net_mismatch",
            "class_mismatch",
            "abbreviation",
            "case_only",
            "low_contrast",
            "rotation",
            "noise",
            "quality_guard",
        }
        assert required <= tags, sorted(required - tags)
        assert len(gen.FIXTURES) >= 20

    def test_unknown_fixture_id_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            gen.generate(tmp_path, {"no_such_fixture"})


# ---------------------------------------------------------------------------
# Generated images and sidecars
# ---------------------------------------------------------------------------


class TestSidecars:
    def test_every_image_has_a_sidecar_and_an_expectation(self, subset_dir: Path) -> None:
        expected = json.loads((subset_dir / "expected.json").read_text())
        listed = {e["image"] for e in expected["fixtures"]}
        images = {
            str(p.relative_to(subset_dir))
            for p in (subset_dir / gen.LABELS_SUBDIR).iterdir()
            if not p.name.endswith(SIDECAR_SUFFIX)
        }
        assert images == listed
        for rel in images:
            assert _sidecar(subset_dir / rel).exists(), rel

    def test_images_fit_the_preprocessing_target(self, subset_dir: Path) -> None:
        """No fixture is larger than the pipeline's working size.

        The pipeline now resizes toward `TARGET_LONGEST_EDGE` in both
        directions (it upscales a small label, because the detector was
        dropping whole lines of warning text at native size). A fixture above
        that target would be *downscaled*, which is the one direction that
        loses ink and makes the small warning type harder to read than any
        real submission would be.
        """
        for image in (subset_dir / gen.LABELS_SUBDIR).glob("*.[pj][np]g"):
            with Image.open(image) as im:
                assert max(im.size) <= TARGET_LONGEST_EDGE, image.name

    def test_sidecar_quality_is_the_pipelines_own_measurement(self, subset_dir: Path) -> None:
        """The recorded quality must be what the pipeline measures, not a guess.

        The stub replays this block, and the warning strategy's degraded-image
        guard fires off it. A drifted or invented number would make stub-mode
        verdicts diverge from real-pipeline verdicts on exactly the fixtures
        that exist to test that guard.
        """
        for image in (subset_dir / gen.LABELS_SUBDIR).glob("*.[pj][np]g"):
            recorded = json.loads(_sidecar(image).read_text())["quality"]
            assert recorded == gen.measure_quality(image), image.name
            assert recorded["usable"], image.name

    def test_low_contrast_fixtures_are_flagged_and_others_are_not(self, subset_dir: Path) -> None:
        """The guard only fires on a flagged image, so the flag is part of the fixture.

        `warning_title_case_low_contrast` proves nothing if the gate stops
        flagging it — the verdict would become an ordinary MISMATCH and the
        test would still pass. Pin the flag here, where the failure is legible.
        """
        good = load_ruleset().defaults.good_image_quality

        def quality(name: str) -> dict:
            return json.loads(_sidecar(subset_dir / gen.LABELS_SUBDIR / name).read_text())[
                "quality"
            ]

        for name in ("degraded_low_contrast.png", "warning_title_case_low_contrast.png"):
            recorded = quality(name)
            assert "low_contrast" in recorded["issues"], name
            assert recorded["score"] < good, name

        # The clean fixtures must stay unflagged and above the "good image"
        # threshold: the absent-warning fixture's elevated expectation depends
        # on it, and so does every fixture that expects an asserted MISMATCH.
        for name in ("old_tom_compliant.png", "warning_absent.png", "warning_title_case.png"):
            recorded = quality(name)
            assert recorded["issues"] == [], name
            assert recorded["score"] >= good, name

    def test_sidecar_dimensions_match_the_image(self, subset_dir: Path) -> None:
        for image in (subset_dir / gen.LABELS_SUBDIR).glob("*.[pj][np]g"):
            payload = json.loads(_sidecar(image).read_text())
            with Image.open(image) as im:
                assert (payload["width"], payload["height"]) == im.size

    def test_boxes_lie_inside_the_image(self, subset_dir: Path) -> None:
        for image in (subset_dir / gen.LABELS_SUBDIR).glob("*.[pj][np]g"):
            payload = json.loads(_sidecar(image).read_text())
            for line in payload["lines"]:
                for w in line["words"]:
                    assert w["left"] >= 0 and w["top"] >= 0, (image.name, w)
                    assert w["left"] + w["width"] <= payload["width"], (image.name, w)
                    assert w["top"] + w["height"] <= payload["height"], (image.name, w)
                    assert w["width"] > 0 and w["height"] > 0

    def test_sidecar_loads_through_the_stub_provider(self, subset_dir: Path) -> None:
        for image in (subset_dir / gen.LABELS_SUBDIR).glob("*.[pj][np]g"):
            result = StubProvider.for_image(image).extract(None)
            assert result.lines, image.name
            with Image.open(image) as im:
                assert (result.image_width, result.image_height) == im.size

    def test_confidence_is_lower_on_degraded_variants(self, subset_dir: Path) -> None:
        def confidences(name: str) -> list[float]:
            payload = json.loads(_sidecar(subset_dir / "labels" / name).read_text())
            return [w["confidence"] for line in payload["lines"] for w in line["words"]]

        clean = confidences("old_tom_compliant.png")
        noisy = confidences("degraded_noise_blur.jpg")
        assert min(clean) >= 0.97 and max(clean) <= 0.99
        assert max(noisy) < min(clean)

    def test_title_case_fixture_really_is_title_case(self, subset_dir: Path) -> None:
        """The raw text must carry the violation, or the fixture proves nothing.

        This is the end-to-end half of trap 1 in extraction/normalize.py: a
        capitalisation check that reads canonical text passes this label.
        """
        text = _raw_text(_sidecar(subset_dir / "labels" / "warning_title_case.png"))
        assert "Government Warning:" in text
        assert "GOVERNMENT WARNING" not in text

    def test_compliant_fixture_has_uppercase_header(self, subset_dir: Path) -> None:
        text = _raw_text(_sidecar(subset_dir / "labels" / "old_tom_compliant.png"))
        assert gen.STATUTORY_WARNING in text

    def test_absent_warning_fixture_has_no_warning(self, subset_dir: Path) -> None:
        text = _raw_text(_sidecar(subset_dir / "labels" / "warning_absent.png")).upper()
        assert "WARNING" not in text and "SURGEON" not in text

    def test_brand_wrap_is_two_sidecar_lines(self, subset_dir: Path) -> None:
        payload = json.loads(_sidecar(subset_dir / "labels" / "brand_two_lines.png").read_text())
        texts = [" ".join(w["text"] for w in line["words"]) for line in payload["lines"]]
        assert texts[:2] == ["LANTERN HOLLOW", "DISTILLING CO."]


# ---------------------------------------------------------------------------
# expected.json schema
# ---------------------------------------------------------------------------


def _validate_expected(data: dict) -> None:
    """Schema check shared by the generated subset and the committed corpus."""
    assert data["schema_version"] in evaluate.SUPPORTED_SCHEMA_VERSIONS
    assert data["ruleset_version"] == "ttb-v1"
    seen = set()
    for entry in data["fixtures"]:
        fid = entry["id"]
        assert fid not in seen, fid
        seen.add(fid)
        ApplicationRecord(**entry["record"])  # raises on a malformed record
        assert entry["compliance"] in {"compliant", "non_compliant", "judgment"}, fid
        assert entry["why"] and entry["tags"], fid
        assert set(entry["fields"]) == set(gen.FIELD_IDS), fid
        for field_id, spec in entry["fields"].items():
            allowed = spec["allowed"]
            assert allowed and set(allowed) <= VALID_VERDICTS, (fid, field_id)
        for check_id, spec in entry["checks"].items():
            assert check_id in gen.GRADED_CHECKS, (fid, check_id)
            assert spec["allowed"] and set(spec["allowed"]) <= VALID_CHECK_OUTCOMES

        # Compliance is derived from the field sets; check the derivation.
        sets = [entry["fields"][f]["allowed"] for f in gen.FIELD_IDS]
        if entry["compliance"] == "compliant":
            assert all(s == ["MATCH"] for s in sets), fid
        elif entry["compliance"] == "non_compliant":
            assert any("MATCH" not in s for s in sets), fid


class TestExpectedJson:
    def test_generated_expectations_are_valid(self, subset_dir: Path) -> None:
        _validate_expected(json.loads((subset_dir / "expected.json").read_text()))

    def test_committed_expectations_are_valid_and_current(self) -> None:
        """The committed expected.json is valid and matches the generator's table.

        expected.json is pure data (no rasterisation), so unlike the images it
        can be compared exactly: a mismatch means someone edited the fixture
        table and forgot `make fixtures`, or edited the JSON by hand.
        """
        path = REPO_FIXTURES / "expected.json"
        if not path.exists():
            pytest.skip("corpus not generated; run `make fixtures`")
        committed = json.loads(path.read_text())
        _validate_expected(committed)
        assert committed["fixtures"] == [gen.expected_entry(f) for f in gen.FIXTURES]
        for entry in committed["fixtures"]:
            image = REPO_FIXTURES / entry["image"]
            assert image.exists() and _sidecar(image).exists(), entry["id"]

    def test_not_found_fixtures_never_allow_mismatch(self) -> None:
        """Absent fields must expect REVIEW only — the central safety invariant."""
        by_id = {f.id: f for f in gen.FIXTURES}
        assert by_id["warning_absent"].expect["government_warning"] == ("REVIEW",)
        assert by_id["warning_absent"].elevated == {"government_warning": True}
        assert by_id["net_missing"].expect["net_contents"] == ("REVIEW",)

    def test_violations_never_allow_match(self) -> None:
        """Every fixture tagged non_compliant must have a field that forbids MATCH."""
        for fixture in gen.FIXTURES:
            if "non_compliant" in fixture.tags:
                assert any("MATCH" not in s for s in fixture.expect.values()), fixture.id

    def test_defective_warnings_never_allow_a_passing_check(self) -> None:
        """A missing or wrong warning must not let a binding check claim PASS.

        Guards against a variant silently inheriting the baseline's all-pass
        check expectations, which is how a "not evaluated" would get graded
        as correct-if-passed.
        """
        by_id = {f.id: f for f in gen.FIXTURES}
        assert "pass" not in by_id["warning_absent"].checks["header_capitalisation"]
        assert "pass" not in by_id["warning_absent"].checks["text_accuracy"]
        assert "pass" not in by_id["warning_title_case"].checks["header_capitalisation"]
        assert "pass" not in by_id["warning_can_cause"].checks["text_accuracy"]
        assert "pass" not in by_id["warning_header_missing"].checks["header_capitalisation"]
        assert (
            "pass" not in by_id["warning_title_case_low_contrast"].checks["header_capitalisation"]
        )

    def test_degraded_guard_fixture_expects_review_only(self) -> None:
        """The whole value of this fixture is that MISMATCH must fail it.

        A title-case header on a *flagged* image: the capitalisation check
        still fails on the evidence, but the typographic assertion is
        withdrawn to REVIEW. Widening this to REVIEW-or-MISMATCH would make
        the fixture pass whether the degraded-image guard exists or not.
        """
        fixture = next(f for f in gen.FIXTURES if f.id == "warning_title_case_low_contrast")
        assert fixture.expect["government_warning"] == ("REVIEW",)
        assert fixture.checks["header_capitalisation"] == ("fail",)
        assert fixture.degrade == "low_contrast"

    def test_degraded_fixtures_are_held_to_exact_expectations(self) -> None:
        """No degraded fixture may hide behind a widened verdict set.

        Real OCR reads this corpus cleanly, so a degraded fixture whose
        expectations were loosened "because OCR might struggle" would be a
        test that can no longer fail.
        """
        for fixture in gen.FIXTURES:
            if fixture.degrade is None:
                continue
            for field_id, allowed in fixture.expect.items():
                assert len(allowed) == 1, (fixture.id, field_id, allowed)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


class _FakeResult:
    """Just enough of a FieldResult for the graders."""

    def __init__(self, verdict: str, reason: str = "ok", elevated: bool = False, checks=()):
        self.verdict = Verdict(verdict)
        self.reason = reason
        self.elevated = elevated
        self.checks = checks


class TestGrading:
    ENTRY = {
        "id": "x",
        "compliance": "non_compliant",
        "fields": {
            "alcohol_content": {"allowed": ["REVIEW", "MISMATCH"]},
            "government_warning": {"allowed": ["REVIEW"], "elevated": True},
            "brand_name": {"allowed": ["MATCH"]},
        },
        "checks": {},
    }

    def test_match_on_a_violation_is_a_false_negative(self) -> None:
        out = evaluate.grade_field(self.ENTRY, "alcohol_content", _FakeResult("MATCH"))
        assert out.status == "incorrect" and out.false_negative

    def test_either_flag_is_correct_on_a_violation(self) -> None:
        for verdict in ("REVIEW", "MISMATCH"):
            out = evaluate.grade_field(self.ENTRY, "alcohol_content", _FakeResult(verdict))
            assert out.status == "correct" and not out.false_negative

    def test_unimplemented_strategy_is_unchecked_not_graded(self) -> None:
        reason = "This prototype does not yet check alcohol content automatically."
        out = evaluate.grade_field(self.ENTRY, "alcohol_content", _FakeResult("REVIEW", reason))
        assert out.status == "unchecked" and not out.false_negative

    def test_missing_elevation_is_incorrect(self) -> None:
        plain = evaluate.grade_field(self.ENTRY, "government_warning", _FakeResult("REVIEW"))
        raised = evaluate.grade_field(
            self.ENTRY, "government_warning", _FakeResult("REVIEW", elevated=True)
        )
        assert plain.status == "incorrect" and raised.status == "correct"

    def test_false_flags_only_count_on_compliant_fixtures(self) -> None:
        entry = {**self.ENTRY, "compliance": "compliant"}
        flagged = evaluate.grade_field(entry, "brand_name", _FakeResult("REVIEW"))
        not_counted = evaluate.grade_field(self.ENTRY, "brand_name", _FakeResult("REVIEW"))
        assert flagged.false_flag and not not_counted.false_flag

    def test_percentile_nearest_rank(self) -> None:
        values = [float(v) for v in range(1, 21)]
        assert evaluate.percentile(values, 50) == 10.0
        assert evaluate.percentile(values, 95) == 19.0


class TestEvaluateEndToEnd:
    def test_stub_run_produces_the_table(self, subset_dir: Path, capsys) -> None:
        code = evaluate.main(["--fixtures", str(subset_dir), "--runs", "1"])
        out = capsys.readouterr().out
        assert code == 0
        assert "not OCR accuracy" in out
        for label, _, _ in evaluate.TABLE_ROWS:
            assert f"| {label} — correct classification |" in out
        assert "False negatives on non-compliant fixtures" in out
        assert "False flags on compliant fixtures" in out

    def test_json_output_is_parseable(self, subset_dir: Path, capsys) -> None:
        evaluate.main(["--fixtures", str(subset_dir), "--runs", "1", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["provider"] == "stub"
        assert len(payload["fixtures"]) == len(SUBSET)

    def test_unavailable_provider_exits_cleanly(self, subset_dir: Path, capsys) -> None:
        """No traceback when the OCR engine is absent — a message and exit 2."""
        try:
            evaluate.make_runner("rapidocr")
        except evaluate.ProviderUnavailable:
            pass
        else:
            pytest.skip("rapidocr is installed here; the unavailable path cannot be exercised")
        code = evaluate.main(["--fixtures", str(subset_dir), "--provider", "rapidocr"])
        assert code == 2
        assert "not available" in capsys.readouterr().err
