"""Tests for the verification engine and the API surface.

Two things get concentrated attention:

*   **Determinism.** The whole auditability argument rests on identical inputs
    producing identical verdicts. It is asserted directly rather than assumed.
*   **Honesty about gaps.** An unimplemented field must be reported as
    unchecked, never as checked-and-fine, and never confused with a field the
    tool looked for and failed to find.
"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

from extraction.providers.stub import StubProvider
from rules.engine import verify
from rules.results import ApplicationRecord, Strategy, Verdict
from rules.schema import UnsupportedBeverageClass
from tests.conftest import make_lines

warnings.filterwarnings("ignore", category=DeprecationWarning)


SPIRITS_RECORD = ApplicationRecord(
    brand_name="OLD TOM DISTILLERY",
    class_type="Kentucky Straight Bourbon Whiskey",
    alcohol_content="45% Alc./Vol.",
    net_contents="750 mL",
)


def extract(lines) -> object:
    return StubProvider(lines=lines).extract(None)


@pytest.fixture
def compliant_extraction(sample_label_lines):
    return extract(sample_label_lines)


class TestDeterminism:
    def test_identical_inputs_produce_identical_verdicts(self, compliant_extraction) -> None:
        """The property the entire auditability argument rests on.

        "The model concluded it matched" is not a defensible finding. "Rule set
        ttb-v1, brand_name, score 0.985, threshold 0.92" is — but only if
        re-running produces the same numbers.
        """
        first = verify(compliant_extraction, SPIRITS_RECORD)
        second = verify(compliant_extraction, SPIRITS_RECORD)

        assert [(f.field_id, f.verdict, f.score) for f in first.fields] == [
            (f.field_id, f.verdict, f.score) for f in second.fields
        ]

    def test_ruleset_version_is_stamped_on_every_result(self, compliant_extraction) -> None:
        """A verdict must be reproducible against the rules in force when made."""
        assert verify(compliant_extraction, SPIRITS_RECORD).ruleset_version == "ttb-v1"


class TestBrandNameResolution:
    """The one fully implemented strategy, and the reference for the rest."""

    def test_wrapped_brand_name_matches(self, compliant_extraction) -> None:
        """ "OLD TOM" / "DISTILLERY" across two lines resolves to one match."""
        result = verify(compliant_extraction, SPIRITS_RECORD).field("brand_name")
        assert result.verdict is Verdict.MATCH
        assert result.found == "OLD TOM DISTILLERY"

    def test_match_carries_the_region_it_was_read_from(self, compliant_extraction) -> None:
        """The overlay is the trust-builder; a verdict without evidence is not."""
        result = verify(compliant_extraction, SPIRITS_RECORD).field("brand_name")
        assert result.bbox is not None
        assert result.bbox.width > 0

    def test_case_only_difference_still_matches(self) -> None:
        """Dave Morrison's example: STONE'S THROW against Stone's Throw.

        Technically a mismatch, obviously the same product. A binary system
        flags it; this one should not.
        """
        record = SPIRITS_RECORD.model_copy(update={"brand_name": "Stone's Throw"})
        result = verify(extract(make_lines(["STONE'S THROW"])), record).field("brand_name")

        assert result.verdict is Verdict.MATCH
        assert "capitalisation" in result.reason

    def test_genuinely_different_brand_is_flagged(self) -> None:
        record = SPIRITS_RECORD.model_copy(update={"brand_name": "OLD TOM DISTILLERY"})
        result = verify(extract(make_lines(["RIVERBEND SPIRITS CO"])), record).field("brand_name")
        assert result.verdict is not Verdict.MATCH

    def test_absent_brand_is_review_not_mismatch(self) -> None:
        """Failing to read display type is not evidence the brand is wrong.

        This is the invariant that matters most: asserting a violation because
        extraction failed misleads an agent who is trusting the tool.
        """
        result = verify(extract(make_lines(["750 mL"])), SPIRITS_RECORD).field("brand_name")

        assert result.verdict is Verdict.REVIEW
        assert result.strategy is Strategy.NOT_FOUND
        assert result.bbox is None

    def test_reason_is_written_for_an_agent_not_a_log(self, compliant_extraction) -> None:
        """Every flag costs attention, so it has to say what to look at."""
        result = verify(compliant_extraction, SPIRITS_RECORD).field("brand_name")
        assert result.reason
        assert not result.reason.startswith("score")


@pytest.fixture
def unwritten(monkeypatch):
    """Replace a field's strategy with an unwritten placeholder.

    Every field is implemented now, but the honesty guarantee is about the
    mechanism, not about which fields happen to be missing today. The next
    field added — an allergen disclosure under a future rule set, say — will
    start life unimplemented, and these tests are what keep it from shipping as
    a silent pass. Simulating the gap keeps the guarantee under test.
    """
    from rules.comparators import STRATEGIES, StrategyNotImplemented
    from rules.engine import load_strategies

    load_strategies()  # register the real strategies first, then override one

    def placeholder(_context):
        raise StrategyNotImplemented("simulated")

    def apply(strategy_name: str) -> None:
        monkeypatch.setitem(STRATEGIES, strategy_name, placeholder)

    return apply


class TestUnimplementedFieldsAreHonest:
    def test_unwritten_strategy_is_reported_as_unchecked(
        self, compliant_extraction, unwritten
    ) -> None:
        """Never silently MATCH.

        A field that passes because nobody implemented its comparator is the
        most dangerous gap possible: the output looks complete and is not.
        """
        unwritten("alcohol_content")
        result = verify(compliant_extraction, SPIRITS_RECORD).field("alcohol_content")

        assert result.verdict is Verdict.REVIEW
        assert result.automated is False
        assert "does not yet check" in result.reason

    def test_unchecked_is_distinct_from_not_found(self, compliant_extraction, unwritten) -> None:
        """ "The tool never looked" must not read as "the tool looked and found nothing".

        Both are NOT_FOUND with score zero, so the distinction lives in
        `automated` and in what the agent is told. A warning statement reported
        as "not located" when the check never ran is a false statement about
        what was verified — and the warning is the field the engine elevates,
        so it is where the two paths are most likely to be confused.
        """
        unwritten("warning")
        warning = verify(compliant_extraction, SPIRITS_RECORD).field("government_warning")

        assert warning.automated is False
        assert warning.elevated is False
        assert "does not yet check" in warning.reason
        assert "Could not locate" not in warning.reason

    def test_implemented_fields_are_marked_automated(self, compliant_extraction) -> None:
        for result in verify(compliant_extraction, SPIRITS_RECORD).fields:
            assert result.automated is True, result.field_id

    def test_genuinely_missing_warning_is_automated_and_elevated(self) -> None:
        """The other side of the distinction: looked, did not find, said so."""
        lines = make_lines(
            [("OLD TOM DISTILLERY", {"height": 48.0}), "Kentucky Straight Bourbon Whiskey"]
        )
        warning = verify(extract(lines), SPIRITS_RECORD).field("government_warning")

        assert warning.automated is True
        assert warning.strategy is Strategy.NOT_FOUND
        assert warning.elevated is True


class TestFieldOrdering:
    def test_results_follow_the_printed_checklist_order(self, compliant_extraction) -> None:
        """Brand, ABV, contents, class/type, warning — how agents already work."""
        result = verify(compliant_extraction, SPIRITS_RECORD)
        assert [f.field_id for f in result.fields] == [
            "brand_name",
            "alcohol_content",
            "net_contents",
            "class_type",
            "government_warning",
        ]


class TestMasking:
    def test_resolved_regions_are_removed_from_the_pool(self, compliant_extraction) -> None:
        """Each resolved field shrinks the candidate pool for the next."""
        result = verify(compliant_extraction, SPIRITS_RECORD, diagnostics=True)
        trace = {entry["field"]: entry for entry in result.diagnostics["resolution"]}

        brand = trace["brand_name"]
        assert brand["candidates_remaining"] < brand["candidates_available"]


class TestBeverageClassGating:
    def test_unimplemented_class_raises_rather_than_guessing(self, compliant_extraction) -> None:
        """Applying spirits rules to a wine label would be confidently wrong."""
        record = SPIRITS_RECORD.model_copy(update={"beverage_class": "wine"})
        with pytest.raises(UnsupportedBeverageClass):
            verify(compliant_extraction, record)


class TestTiming:
    def test_elapsed_covers_the_whole_request_when_given_a_start(
        self, compliant_extraction
    ) -> None:
        """The displayed number must be the time the agent actually waited.

        This tool's predecessor failed on latency. A figure that quietly
        excluded upload and decode would be worse than showing none.
        """
        import time

        started = time.perf_counter() - 2.0  # pretend 2s elapsed before verify()
        result = verify(compliant_extraction, SPIRITS_RECORD, started_at=started)
        assert result.elapsed_ms >= 2000


class TestApi:
    @pytest.fixture
    def client(self) -> TestClient:
        from api.main import app

        return TestClient(app)

    def test_health_reports_ruleset_and_provider_state(self, client: TestClient) -> None:
        payload = client.get("/api/health").json()
        assert payload["ruleset_version"] == "ttb-v1"
        assert payload["status"] in {"ok", "degraded"}

    def test_health_is_degraded_not_dead_without_an_ocr_engine(self, client: TestClient) -> None:
        """An operator must be able to tell a dead process from a modelless one."""
        payload = client.get("/api/health").json()
        if payload["provider"] is None:
            assert payload["status"] == "degraded"
            assert payload["provider_error"]

    def test_ruleset_endpoint_declares_which_checks_are_automated(
        self, client: TestClient, unwritten
    ) -> None:
        """The UI says what it does not check rather than presenting a partial
        tool as a complete one."""
        by_id = {f["id"]: f for f in client.get("/api/ruleset").json()["fields"]}
        assert all(f["implemented"] for f in by_id.values())

        unwritten("alcohol_content")
        by_id = {f["id"]: f for f in client.get("/api/ruleset").json()["fields"]}
        assert by_id["alcohol_content"]["implemented"] is False
        assert by_id["brand_name"]["implemented"] is True

    def test_every_field_exposes_its_citation(self, client: TestClient) -> None:
        for descriptor in client.get("/api/ruleset").json()["fields"]:
            assert descriptor["citation"].startswith("27 CFR")

    def test_non_image_upload_is_rejected_before_processing(self, client: TestClient) -> None:
        response = client.post(
            "/api/verify",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
            data={
                "brand_name": "OLD TOM DISTILLERY",
                "class_type": "Kentucky Straight Bourbon Whiskey",
                "alcohol_content": "45% Alc./Vol.",
                "net_contents": "750 mL",
            },
        )
        assert response.status_code == 415
        assert "PDF" in response.json()["detail"]

    def test_empty_upload_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/verify",
            files={"image": ("empty.png", b"", "image/png")},
            data={
                "brand_name": "B",
                "class_type": "C",
                "alcohol_content": "45%",
                "net_contents": "750 mL",
            },
        )
        assert response.status_code == 400
