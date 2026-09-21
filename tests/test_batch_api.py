"""Batch API: submission, worklist order, parity with single-label, limits.

No OCR engine runs here. A `FixtureProvider` recognises each fixture image by
the hash of its preprocessed pixels and replays that fixture's ground-truth
sidecar through `StubProvider` — so the full pipeline (decode, rescale, quality
gate, verification, triage) runs for real, and only the recognition step is
replayed. That keeps the suite fast and makes every verdict here a statement
about the batch machinery rather than about OCR.

The test most worth reading is `test_batch_result_equals_single_label_result`:
batch exists to do exactly what the single-label screen does, many times, and a
file must get the same verdicts whichever way it was checked.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from api import batch as batch_module
from api.batch import get_runner
from api.batch_models import BatchState, ItemState, PairingIssueKind
from api.jobs import BatchRunner, BatchSettings, neutralise_cell
from extraction.pipeline import preprocess
from extraction.providers.base import ExtractionProvider, ExtractionResult
from extraction.providers.stub import StubProvider
from rules.triage import sort_key, summarise

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "fixtures" / "labels"
EXPECTED = json.loads((ROOT / "fixtures" / "expected.json").read_text())
RECORDS = {Path(f["image"]).name: f["record"] for f in EXPECTED["fixtures"]}

#: A spread of outcomes: clear, asserted mismatches, a missing warning.
CORPUS = (
    "old_tom_compliant.png",
    "abv_mismatch.png",
    "warning_absent.png",
    "brand_case_only.png",
    "net_mismatch.png",
)

HEADER = "filename,brand_name,class_type,alcohol_content,net_contents"


def _key(image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


class FixtureProvider(ExtractionProvider):
    """Replays the sidecar for whichever fixture image it is handed.

    `gate`, when given, is waited on inside `extract` — the way to hold a batch
    mid-run so a test can look at it. `calls` counts extractions, which is how
    a cancel is shown to have stopped scheduling.
    """

    name = "stub"

    def __init__(self, names: tuple[str, ...], *, delay: float = 0.0) -> None:
        self._by_key = {
            _key(preprocess((LABELS / n).read_bytes()).image): StubProvider.for_image(LABELS / n)
            for n in names
        }
        self.delay = delay
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def extract(self, image) -> ExtractionResult:
        with self._lock:
            self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "test gate never opened"
        if self.delay:
            time.sleep(self.delay)  # blocking on purpose: stands in for CPU-bound OCR
        return self._by_key[_key(image)].extract(image)


@pytest.fixture(scope="module")
def provider_template() -> FixtureProvider:
    # Preprocessing the corpus once per module; each test gets a fresh counter.
    return FixtureProvider(CORPUS)


@pytest.fixture
def provider(provider_template: FixtureProvider) -> FixtureProvider:
    fresh = object.__new__(FixtureProvider)
    fresh.__dict__.update(provider_template.__dict__)
    fresh.gate = None
    fresh.entered = threading.Event()
    fresh.calls = 0
    fresh._lock = threading.Lock()
    return fresh


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def app_state(monkeypatch, provider):
    from api import main

    monkeypatch.setattr(main.state, "provider", provider)
    monkeypatch.setattr(main.state, "access_token", None)
    return main


@pytest.fixture
def runner(app_state, provider, clock) -> Iterator[BatchRunner]:
    runner = BatchRunner(
        ruleset=app_state.state.ruleset,
        provider_factory=lambda: provider,
        settings=BatchSettings(workers=1, nice=0, sweep_interval_s=0.05),
        clock=clock,
    )
    yield runner
    runner.close()


@pytest.fixture
def client(app_state, runner) -> Iterator[TestClient]:
    app_state.app.dependency_overrides[get_runner] = lambda: runner
    with TestClient(app_state.app) as test_client:
        yield test_client
    app_state.app.dependency_overrides.pop(get_runner, None)


def manifest_for(names, *, brand_overrides: dict[str, str] | None = None) -> bytes:
    lines = [HEADER]
    for name in names:
        record = RECORDS.get(name, RECORDS["old_tom_compliant.png"])
        brand = (brand_overrides or {}).get(name, record["brand_name"])
        cells = [
            name,
            brand,
            record["class_type"],
            record["alcohol_content"],
            record["net_contents"],
        ]
        lines.append(",".join(f'"{c}"' if "," in c else c for c in cells))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8-sig")


def image_part(name: str, data: bytes | None = None, content_type: str = "image/png"):
    return (
        "images",
        (name, data if data is not None else (LABELS / name).read_bytes(), content_type),
    )


def submit(client: TestClient, manifest: bytes, parts, **kwargs):
    return client.post(
        "/api/batch", files=[("manifest", ("manifest.csv", manifest, "text/csv")), *parts], **kwargs
    )


def wait_done(client: TestClient, batch_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/batch/{batch_id}").json()
        if body["state"] in (BatchState.DONE, BatchState.CANCELLED):
            return body
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} did not finish: {body}")


# ---------------------------------------------------------------------------
# The event-loop fix in api/main.py
# ---------------------------------------------------------------------------


class TestEventLoopStaysResponsive:
    def test_health_answers_while_single_label_extraction_is_running(
        self, app_state, provider
    ) -> None:
        """Before the fix, OCR ran on the event loop and froze every request.

        The provider here blocks for 1.5 s with `time.sleep`, as CPU-bound OCR
        would. Health must answer in a fraction of that while it is running.
        """
        provider.delay = 1.5
        record = RECORDS["old_tom_compliant.png"]
        outcome: dict = {}

        with TestClient(app_state.app) as client:

            def check_label() -> None:
                outcome["response"] = client.post(
                    "/api/verify",
                    files={
                        "image": (
                            "l.png",
                            (LABELS / "old_tom_compliant.png").read_bytes(),
                            "image/png",
                        )
                    },
                    data=record,
                )

            worker = threading.Thread(target=check_label)
            worker.start()
            assert provider.entered.wait(10), "extraction never started"

            started = time.perf_counter()
            health = client.get("/api/health")
            health_seconds = time.perf_counter() - started
            still_extracting = worker.is_alive()
            worker.join(10)

        assert health.status_code == 200
        assert still_extracting, "the label finished before health was measured"
        assert health_seconds < 0.5, f"health took {health_seconds:.2f}s during extraction"
        assert outcome["response"].status_code == 200


# ---------------------------------------------------------------------------
# Template and check
# ---------------------------------------------------------------------------


class TestTemplateAndCheck:
    def test_template_csv(self, client: TestClient) -> None:
        response = client.get("/api/batch/template.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        lines = response.text.splitlines()
        assert lines[0] == (
            "filename,brand_name,class_type,alcohol_content,net_contents,"
            "beverage_class,label_width_mm"
        )
        assert (
            "OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL" in lines[1]
        )

    def test_check_pairs_filenames_without_images(self, client: TestClient) -> None:
        manifest = manifest_for(["old_tom_compliant.png", "abv_mismatch.png", "gone.png"])
        response = client.post(
            "/api/batch/check",
            files=[("manifest", ("m.csv", manifest, "text/csv"))],
            data={"filenames": ["folder/old_tom_compliant.png", "abv_mismatch.png", "extra.jpg"]},
        )
        assert response.status_code == 200
        report = response.json()
        assert report["matched"] == 2
        assert report["blocking"] is False
        assert {i["kind"] for i in report["issues"]} == {
            PairingIssueKind.ROW_WITHOUT_IMAGE,
            PairingIssueKind.IMAGE_WITHOUT_ROW,
        }

    def test_check_rejects_too_many_filenames_before_upload(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.setattr(batch_module, "MAX_BATCH_SIZE", 3)
        response = client.post(
            "/api/batch/check",
            files=[("manifest", ("m.csv", manifest_for(["a.png"]), "text/csv"))],
            data={"filenames": ["a.png", "b.png", "c.png", "d.png"]},
        )
        assert response.status_code == 413
        assert "at most 3" in response.json()["detail"]

    def test_check_requires_a_manifest(self, client: TestClient) -> None:
        response = client.post("/api/batch/check", data={"filenames": ["a.png"]})
        assert response.status_code == 400
        assert "manifest" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Running a batch
# ---------------------------------------------------------------------------


class TestBatchRun:
    def test_batch_runs_and_orders_worst_first(self, client: TestClient) -> None:
        response = submit(client, manifest_for(CORPUS), [image_part(n) for n in CORPUS])
        assert response.status_code == 202, response.text
        accepted = response.json()
        assert accepted["total"] == len(CORPUS)
        assert accepted["pairing"]["matched"] == len(CORPUS)

        body = wait_done(client, accepted["batch_id"])
        assert body["state"] == BatchState.DONE
        assert body["completed"] == len(CORPUS)
        assert body["failed"] == 0
        assert body["estimated_remaining_ms"] == 0.0
        assert all(item["state"] == ItemState.DONE for item in body["items"])

        # The order is exactly rules.triage's, recomputed from the full results.
        from rules.results import VerificationResult

        # Polls carry summaries only; the full result is fetched per item.
        assert all(i["result"] is None for i in body["items"])
        full = [
            client.get(f"/api/batch/{accepted['batch_id']}/items/{i['row_number']}").json()
            for i in body["items"]
        ]
        keys = [
            sort_key(summarise(VerificationResult.model_validate(i["result"])), i["filename"])
            for i in full
        ]
        assert keys == sorted(keys)
        tiers = [i["tier"] for i in body["items"]]
        assert tiers == sorted(tiers)
        assert body["items"][0]["tier_label"] == "Differs from the application"
        assert body["items"][-1]["needs_attention"] is False
        assert body["needs_attention"] == sum(1 for i in body["items"] if i["needs_attention"])
        assert 0 < body["needs_attention"] < len(CORPUS)

    def test_batch_result_equals_single_label_result(self, client: TestClient) -> None:
        """Same bytes, same record, same verdicts and scores — whichever screen."""
        names = ("abv_mismatch.png", "warning_absent.png")
        batch_id = submit(client, manifest_for(names), [image_part(n) for n in names]).json()[
            "batch_id"
        ]
        body = wait_done(client, batch_id)
        by_name = {
            i["filename"]: client.get(f"/api/batch/{batch_id}/items/{i['row_number']}").json()
            for i in body["items"]
        }

        for name in names:
            single = client.post(
                "/api/verify",
                files={"image": (name, (LABELS / name).read_bytes(), "image/png")},
                data=RECORDS[name],
            )
            assert single.status_code == 200
            single_result = single.json()
            batch_result = by_name[name]["result"]

            def essentials(result: dict) -> list:
                return [
                    (f["field_id"], f["verdict"], f["score"], f["found"], f["reason"])
                    for f in result["fields"]
                ]

            assert essentials(batch_result) == essentials(single_result)
            assert batch_result["quality"] == single_result["quality"]
            assert batch_result["ruleset_version"] == single_result["ruleset_version"]

    def test_one_bad_image_does_not_fail_the_batch(self, client: TestClient) -> None:
        names = ["old_tom_compliant.png", "broken.png"]
        parts = [image_part("old_tom_compliant.png"), image_part("broken.png", b"not a png")]
        body = wait_done(client, submit(client, manifest_for(names), parts).json()["batch_id"])

        assert body["state"] == BatchState.DONE
        assert (body["completed"], body["failed"]) == (2, 1)
        broken = next(i for i in body["items"] if i["filename"] == "broken.png")
        assert broken["state"] == ItemState.ERROR
        assert broken["error"].startswith("Could not read this image")
        assert broken["tier_label"] == "Could not read this image"
        assert broken["result"] is None
        # Unreadable ranks above a clear label.
        assert body["items"][0]["filename"] == "broken.png"

    def test_unusable_files_are_reported_and_skipped(self, client: TestClient) -> None:
        names = ["old_tom_compliant.png", "notes.png", "empty.png"]
        parts = [
            image_part("old_tom_compliant.png"),
            image_part("notes.png", b"hello", content_type="text/plain"),
            image_part("empty.png", b""),
        ]
        response = submit(client, manifest_for(names), parts)
        assert response.status_code == 202
        pairing = response.json()["pairing"]
        assert pairing["matched"] == 1
        messages = {i["filename"]: i["message"] for i in pairing["issues"]}
        assert "not a supported image type" in messages["notes.png"]
        assert messages["empty.png"] == "empty.png is empty."
        assert response.json()["total"] == 1

    def test_nothing_to_run_is_rejected_with_the_pairing(self, client: TestClient) -> None:
        response = submit(
            client, manifest_for(["old_tom_compliant.png"]), [image_part("abv_mismatch.png")]
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["pairing"]["blocking"] is True
        assert detail["pairing"]["matched"] == 0

    def test_running_item_precedes_pending_items_in_upload_order(
        self, client: TestClient, provider: FixtureProvider
    ) -> None:
        provider.gate = threading.Event()
        order = ["net_mismatch.png", "old_tom_compliant.png", "abv_mismatch.png"]
        batch_id = submit(client, manifest_for(order), [image_part(n) for n in order]).json()[
            "batch_id"
        ]
        assert provider.entered.wait(10)
        mid = client.get(f"/api/batch/{batch_id}").json()
        assert mid["state"] == BatchState.RUNNING
        assert mid["estimated_remaining_ms"] is None  # nothing has finished yet
        assert [i["filename"] for i in mid["items"]] == order
        assert [i["state"] for i in mid["items"]] == [
            ItemState.RUNNING,
            ItemState.PENDING,
            ItemState.PENDING,
        ]
        assert all(i["tier"] is None and i["needs_attention"] is None for i in mid["items"])

        provider.gate.set()
        done = wait_done(client, batch_id)
        assert done["items"][-1]["filename"] == "old_tom_compliant.png"
        assert done["items"][-1]["tier_label"] == "All clear"

    def test_second_batch_queues_behind_the_first(
        self, client: TestClient, provider: FixtureProvider
    ) -> None:
        provider.gate = threading.Event()
        first = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        assert provider.entered.wait(10)
        second = submit(
            client, manifest_for(["net_mismatch.png"]), [image_part("net_mismatch.png")]
        ).json()
        assert second["state"] == BatchState.QUEUED
        provider.gate.set()
        wait_done(client, first)
        assert wait_done(client, second["batch_id"])["state"] == BatchState.DONE

    def test_estimate_appears_after_the_first_label(
        self, client: TestClient, provider: FixtureProvider, runner: BatchRunner
    ) -> None:
        gate = threading.Event()
        provider.gate = gate
        names = ["abv_mismatch.png", "net_mismatch.png"]
        batch_id = submit(client, manifest_for(names), [image_part(n) for n in names]).json()[
            "batch_id"
        ]
        assert provider.entered.wait(10)
        # Let exactly the first label through, then hold the second.
        provider.entered.clear()
        second_gate = threading.Event()
        provider.gate = second_gate
        gate.set()
        assert provider.entered.wait(10)
        deadline = time.monotonic() + 5
        while client.get(f"/api/batch/{batch_id}").json()["completed"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        mid = client.get(f"/api/batch/{batch_id}").json()
        assert mid["estimated_remaining_ms"] is not None
        assert mid["estimated_remaining_ms"] >= 0
        second_gate.set()
        wait_done(client, batch_id)

    def test_images_are_released_after_processing(
        self, client: TestClient, runner: BatchRunner
    ) -> None:
        body = wait_done(
            client,
            submit(
                client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
            ).json()["batch_id"],
        )
        assert body["state"] == BatchState.DONE
        assert runner.held_bytes == 0


# ---------------------------------------------------------------------------
# Cancel, expiry, 404, access
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_cancel_stops_scheduling_and_keeps_finished_work(
        self, client: TestClient, provider: FixtureProvider, runner: BatchRunner, clock: Clock
    ) -> None:
        """DELETE stops the run, keeps the record, and is visible through GET."""
        first_gate = threading.Event()
        provider.gate = first_gate
        order = ["abv_mismatch.png", "net_mismatch.png", "old_tom_compliant.png"]
        batch_id = submit(client, manifest_for(order), [image_part(n) for n in order]).json()[
            "batch_id"
        ]
        # Let the first label finish, then hold the second inside "OCR".
        assert provider.entered.wait(10)
        provider.entered.clear()
        second_gate = threading.Event()
        provider.gate = second_gate
        first_gate.set()
        assert provider.entered.wait(10)

        cancelled = client.delete(f"/api/batch/{batch_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == BatchState.CANCELLED
        # At once: the label inside OCR is RUNNING, the one never started SKIPPED.
        states = {i["filename"]: i["state"] for i in cancelled.json()["items"]}
        assert states == {
            "abv_mismatch.png": ItemState.DONE,
            "net_mismatch.png": ItemState.RUNNING,
            "old_tom_compliant.png": ItemState.SKIPPED,
        }
        # Only the in-flight label's image is still held; the pending one is released.
        assert runner.held_bytes == (LABELS / "net_mismatch.png").stat().st_size

        second_gate.set()
        deadline = time.monotonic() + 5
        while client.get(f"/api/batch/{batch_id}").json()["completed"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        time.sleep(0.1)
        assert provider.calls == 2  # the third label was never started
        assert runner.held_bytes == 0

        body = client.get(f"/api/batch/{batch_id}").json()
        assert body["state"] == BatchState.CANCELLED
        assert body["estimated_remaining_ms"] == 0.0
        by_name = {i["filename"]: i for i in body["items"]}
        # The label that was already running finished and its result was kept.
        assert by_name["net_mismatch.png"]["state"] == ItemState.DONE
        assert by_name["abv_mismatch.png"]["state"] == ItemState.DONE
        never = by_name["old_tom_compliant.png"]
        assert never["state"] == ItemState.SKIPPED
        assert (never["tier"], never["tier_label"], never["headline"]) == (None, None, None)
        assert never["needs_attention"] is True
        assert body["items"][-1]["filename"] == "old_tom_compliant.png"
        # Both checked labels are mismatches, and the skipped one counts too.
        assert body["needs_attention"] == 3
        assert body["completed"] == 2

        # The full result of a finished item is still there.
        row = by_name["net_mismatch.png"]["row_number"]
        assert client.get(f"/api/batch/{batch_id}/items/{row}").json()["result"] is not None

        # The CSV records the unchecked label rather than dropping it.
        import csv
        import io

        text = client.get(f"/api/batch/{batch_id}/results.csv").content.decode("utf-8-sig")
        rows = {r["filename"]: r for r in csv.DictReader(io.StringIO(text))}
        assert len(rows) == 3
        skipped = rows["old_tom_compliant.png"]
        assert skipped["status"] == "not checked (batch stopped)"
        assert skipped["needs_attention"] == "yes"
        assert skipped["tier"] == ""
        verdict_columns = [k for k in skipped if k.endswith(" verdict")]
        assert verdict_columns and all(skipped[k] == "" for k in verdict_columns)
        assert list(rows)[-1] == "old_tom_compliant.png"
        assert rows["net_mismatch.png"]["status"] == "done"

        # It then expires on the normal TTL, counted from the cancel.
        finished = datetime.fromisoformat(body["finished_at"])
        assert datetime.fromisoformat(body["expires_at"]) == finished + runner.settings.ttl
        clock.now += runner.settings.ttl + timedelta(seconds=1)
        assert client.get(f"/api/batch/{batch_id}").status_code == 404

    def test_cancel_while_queued_runs_nothing(
        self, client: TestClient, provider: FixtureProvider
    ) -> None:
        provider.gate = threading.Event()
        first = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        assert provider.entered.wait(10)
        second = submit(
            client, manifest_for(["net_mismatch.png"]), [image_part("net_mismatch.png")]
        ).json()["batch_id"]
        assert client.delete(f"/api/batch/{second}").json()["state"] == BatchState.CANCELLED
        provider.gate.set()
        wait_done(client, first)
        time.sleep(0.1)
        assert provider.calls == 1
        body = client.get(f"/api/batch/{second}").json()
        assert body["state"] == BatchState.CANCELLED
        assert body["items"][0]["state"] == ItemState.SKIPPED
        assert body["needs_attention"] == 1

    def test_cancelling_a_finished_batch_changes_nothing(self, client: TestClient) -> None:
        batch_id = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        wait_done(client, batch_id)
        assert client.delete(f"/api/batch/{batch_id}").json()["state"] == BatchState.DONE
        assert client.get(f"/api/batch/{batch_id}").json()["state"] == BatchState.DONE

    def test_batch_expires_after_the_configured_time(
        self, client: TestClient, clock: Clock, runner: BatchRunner
    ) -> None:
        batch_id = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        body = wait_done(client, batch_id)
        finished = datetime.fromisoformat(body["finished_at"])
        assert datetime.fromisoformat(body["expires_at"]) == finished + runner.settings.ttl

        clock.now += runner.settings.ttl - timedelta(seconds=1)
        assert client.get(f"/api/batch/{batch_id}").status_code == 200
        clock.now += timedelta(seconds=2)
        response = client.get(f"/api/batch/{batch_id}")
        assert response.status_code == 404
        assert "2 hours" in response.json()["detail"]

    def test_expired_batches_are_swept_without_any_request(
        self, client: TestClient, clock: Clock, runner: BatchRunner
    ) -> None:
        batch_id = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        wait_done(client, batch_id)
        clock.now += runner.settings.ttl + timedelta(seconds=1)
        time.sleep(0.3)  # the dispatcher sweeps every 50 ms in this fixture
        assert batch_id not in runner._batches

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/api/batch/nope"),
            ("get", "/api/batch/nope/results.csv"),
            ("delete", "/api/batch/nope"),
            ("get", "/api/batch/nope/items/2"),
        ],
    )
    def test_unknown_batch_is_404_with_a_plain_message(
        self, client: TestClient, method: str, path: str
    ) -> None:
        response = getattr(client, method)(path)
        assert response.status_code == 404
        assert "could not be found" in response.json()["detail"]

    def test_every_batch_endpoint_is_behind_the_access_gate(
        self, client: TestClient, app_state, monkeypatch
    ) -> None:
        monkeypatch.setattr(app_state.state, "access_token", "s3cret")
        assert client.get("/api/batch/template.csv").status_code == 401
        assert client.get("/api/batch/x").status_code == 401
        assert client.delete("/api/batch/x").status_code == 401
        assert client.get("/api/batch/x/results.csv").status_code == 401
        assert client.get("/api/batch/x/items/2").status_code == 401
        assert submit(client, b"", []).status_code == 401
        assert client.post("/api/batch/check", data={"filenames": ["a"]}).status_code == 401
        assert client.get("/api/batch/template.csv?token=s3cret").status_code == 200


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


class TestLimits:
    def test_too_many_images_rejected_before_processing(
        self, client: TestClient, monkeypatch, provider: FixtureProvider
    ) -> None:
        monkeypatch.setattr(batch_module, "MAX_BATCH_SIZE", 2)
        names = ["old_tom_compliant.png", "abv_mismatch.png", "net_mismatch.png"]
        response = submit(client, manifest_for(names), [image_part(n) for n in names])
        assert response.status_code == 413
        assert "at most 2 images" in response.json()["detail"]
        assert provider.calls == 0

    def test_total_bytes_cap(self, app_state, provider, clock, monkeypatch) -> None:
        small = BatchRunner(
            ruleset=app_state.state.ruleset,
            provider_factory=lambda: provider,
            settings=BatchSettings(max_bytes=1000, nice=0),
            clock=clock,
        )
        monkeypatch.setattr(batch_module, "_FRAMING_ALLOWANCE", 0)
        app_state.app.dependency_overrides[get_runner] = lambda: small
        try:
            with TestClient(app_state.app) as client:
                response = submit(
                    client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
                )
        finally:
            app_state.app.dependency_overrides.pop(get_runner, None)
            small.close()
        assert response.status_code == 413
        assert "limit for one batch" in response.json()["detail"]
        assert provider.calls == 0

    def test_runner_refuses_when_other_batches_hold_the_budget(
        self, app_state, provider, clock
    ) -> None:
        from api.jobs import BatchCapacityError
        from api.manifest import pair_files, parse_manifest

        runner = BatchRunner(
            ruleset=app_state.state.ruleset,
            provider_factory=lambda: provider,
            settings=BatchSettings(max_bytes=150, nice=0),
            clock=clock,
        )
        provider.gate = threading.Event()
        try:
            pairing = pair_files(parse_manifest(manifest_for(["a.png"])), ["a.png"])
            runner.submit(pairing.pairs, [b"x" * 100], pairing.report)
            with pytest.raises(BatchCapacityError) as excinfo:
                runner.submit(pairing.pairs, [b"y" * 100], pairing.report)
            assert excinfo.value.retryable
            with pytest.raises(BatchCapacityError) as excinfo:
                runner.submit(pairing.pairs, [b"z" * 200], pairing.report)
            assert not excinfo.value.retryable
        finally:
            provider.gate.set()
            runner.close()

    def test_oversized_single_image_is_an_issue_not_a_failure(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.setattr(batch_module._config, "max_upload_bytes", 1000)
        names = ["old_tom_compliant.png", "tiny.png"]
        parts = [image_part("old_tom_compliant.png"), image_part("tiny.png", b"x" * 10)]
        response = submit(client, manifest_for(names), parts)
        # The fixture itself is over the tiny limit; the 10-byte file is not.
        assert response.status_code == 202
        issue = next(
            i
            for i in response.json()["pairing"]["issues"]
            if i["filename"] == "old_tom_compliant.png"
        )
        assert "larger than the" in issue["message"]

    def test_no_extraction_engine_is_503_before_reading_images(
        self, client: TestClient, app_state, monkeypatch
    ) -> None:
        monkeypatch.setattr(app_state.state, "provider", None)
        monkeypatch.setattr(app_state.state, "provider_error", "no model")
        response = submit(client, manifest_for(["a.png"]), [])
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "extraction_unavailable"
        assert "no model" in detail["message"]
        assert "retry-after" not in response.headers  # retrying will not help


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------


class TestResultsCsv:
    def test_neutralise_cell(self) -> None:
        for dangerous in ("=1+1", "+1", "-1", "@SUM(A1)", "\tx", "\rx"):
            assert neutralise_cell(dangerous) == "'" + dangerous
        assert neutralise_cell("OLD TOM") == "OLD TOM"
        assert neutralise_cell(None) == ""
        assert neutralise_cell(14) == "14"

    def test_results_csv_neutralises_formula_injection(self, client: TestClient) -> None:
        import csv
        import io

        evil_file = "=cmd.png"
        names = ["old_tom_compliant.png", evil_file]
        manifest = manifest_for(
            names,
            brand_overrides={
                "old_tom_compliant.png": '=HYPERLINK("http://x.test","click")',
                evil_file: "@SUM(1)",
            },
        )
        parts = [
            image_part("old_tom_compliant.png"),
            image_part(evil_file, (LABELS / "abv_mismatch.png").read_bytes()),
        ]
        batch_id = submit(client, manifest, parts).json()["batch_id"]
        wait_done(client, batch_id)

        response = client.get(f"/api/batch/{batch_id}/results.csv")
        assert response.status_code == 200
        assert response.content.startswith(b"\xef\xbb\xbf")  # Excel reads it as UTF-8
        rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
        assert len(rows) == 2
        by_file = {r["filename"]: r for r in rows}
        assert "'=cmd.png" in by_file
        assert by_file["'=cmd.png"]["brand_name"] == "'@SUM(1)"
        assert by_file["old_tom_compliant.png"]["brand_name"].startswith("'=HYPERLINK")
        for r in rows:
            for value in r.values():
                assert not value.startswith(("=", "+", "-", "@", "\t", "\r"))

    def test_results_csv_columns_and_order(self, client: TestClient) -> None:
        import csv
        import io

        names = ["old_tom_compliant.png", "abv_mismatch.png", "broken.png"]
        parts = [image_part(n) for n in names[:2]] + [image_part("broken.png", b"nope")]
        batch_id = submit(client, manifest_for(names), parts).json()["batch_id"]
        status_body = wait_done(client, batch_id)

        text = client.get(f"/api/batch/{batch_id}/results.csv").content.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        assert [r["filename"] for r in rows] == [i["filename"] for i in status_body["items"]]
        header = list(rows[0])
        assert header[:6] == [
            "filename",
            "manifest_row",
            "brand_name",
            "status",
            "tier",
            "needs_attention",
        ]
        assert "Brand Name verdict" in header
        assert header[-2:] == ["headline", "error"]
        by_file = {r["filename"]: r for r in rows}
        assert by_file["abv_mismatch.png"]["Alcohol Content verdict"] == "MISMATCH"
        assert by_file["old_tom_compliant.png"]["needs_attention"] == "no"
        assert by_file["broken.png"]["status"] == "error"
        assert by_file["broken.png"]["error"].startswith("Could not read this image")


# ---------------------------------------------------------------------------
# Revised contract: summaries in polls, full result per item
# ---------------------------------------------------------------------------


class TestItemsEndpoint:
    def test_item_returns_the_full_result(self, client: TestClient) -> None:
        batch_id = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        body = wait_done(client, batch_id)
        (summary,) = body["items"]
        assert summary["result"] is None

        response = client.get(f"/api/batch/{batch_id}/items/{summary['row_number']}")
        assert response.status_code == 200
        item = response.json()
        assert item["filename"] == "abv_mismatch.png"
        assert item["tier"] == summary["tier"]
        assert {f["field_id"] for f in item["result"]["fields"]} >= {"brand_name", "net_contents"}

    def test_pending_item_returns_its_state_without_a_result(
        self, client: TestClient, provider: FixtureProvider
    ) -> None:
        provider.gate = threading.Event()
        names = ["abv_mismatch.png", "net_mismatch.png"]
        batch_id = submit(client, manifest_for(names), [image_part(n) for n in names]).json()[
            "batch_id"
        ]
        assert provider.entered.wait(10)
        item = client.get(f"/api/batch/{batch_id}/items/3").json()
        assert (item["filename"], item["state"], item["result"]) == (
            "net_mismatch.png",
            ItemState.PENDING,
            None,
        )
        provider.gate.set()
        wait_done(client, batch_id)

    def test_unknown_row_is_404_with_a_plain_message(self, client: TestClient) -> None:
        batch_id = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        ).json()["batch_id"]
        wait_done(client, batch_id)
        response = client.get(f"/api/batch/{batch_id}/items/99")
        assert response.status_code == 404
        assert response.json()["detail"].startswith("Row 99 of the manifest is not part")

    def test_poll_payload_stays_small_at_300_labels(
        self, client: TestClient, runner: BatchRunner, provider: FixtureProvider, monkeypatch
    ) -> None:
        """The measured problem: ~2 MB per poll at 300 labels, every 2 s.

        OCR is irrelevant to payload size, so every item reuses one real
        verification result (a mismatch, so each carries a headline) and the
        300 labels finish instantly.
        """
        from api import jobs
        from api.manifest import pair_files, parse_manifest
        from rules.results import ApplicationRecord

        real = jobs.run_pipeline(
            (LABELS / "abv_mismatch.png").read_bytes(),
            ApplicationRecord(**RECORDS["abv_mismatch.png"]),
            provider=provider,
            ruleset=runner.ruleset,
            diagnostics=False,
            started_at=time.perf_counter(),
        )
        monkeypatch.setattr(jobs, "run_pipeline", lambda *a, **k: real)

        names = [f"import-0412_label_{i:03d}.jpg" for i in range(300)]
        lines = [HEADER] + [
            f"{n},OLD TOM DISTILLERY,Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL"
            for n in names
        ]
        pairing = pair_files(parse_manifest("\n".join(lines).encode()), names)
        batch_id = runner.submit(pairing.pairs, [b"x"] * 300, pairing.report)
        body = wait_done(client, batch_id)
        assert body["completed"] == 300

        poll = client.get(f"/api/batch/{batch_id}")
        poll_bytes = len(poll.content)
        full_bytes = len(runner.status(batch_id, include_results=True).model_dump_json())
        # Measured: ~173 kB summaries-only vs ~2.2 MB with results (0.6 vs 7.3 kB/label).
        assert poll_bytes < 250 * 1024, poll_bytes
        assert full_bytes > 10 * poll_bytes, (poll_bytes, full_bytes)
        assert all(i["result"] is None and i["headline"] for i in poll.json()["items"])


# ---------------------------------------------------------------------------
# Machine-readable 503s
# ---------------------------------------------------------------------------


class TestServiceUnavailableCodes:
    def test_capacity_503_has_a_code_and_retry_after(
        self, app_state, provider, clock, monkeypatch
    ) -> None:
        size = (LABELS / "abv_mismatch.png").stat().st_size
        small = BatchRunner(
            ruleset=app_state.state.ruleset,
            provider_factory=lambda: provider,
            settings=BatchSettings(max_bytes=int(size * 1.5), nice=0),
            clock=clock,
        )
        monkeypatch.setattr(batch_module, "_FRAMING_ALLOWANCE", 0)
        provider.gate = threading.Event()
        app_state.app.dependency_overrides[get_runner] = lambda: small
        try:
            with TestClient(app_state.app) as client:
                first = submit(
                    client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
                )
                assert first.status_code == 202
                assert provider.entered.wait(10)
                second = submit(
                    client, manifest_for(["net_mismatch.png"]), [image_part("net_mismatch.png")]
                )
        finally:
            provider.gate.set()
            app_state.app.dependency_overrides.pop(get_runner, None)
            small.close()
        assert second.status_code == 503
        detail = second.json()["detail"]
        assert detail["code"] == "batch_capacity"
        assert "Try again" in detail["message"]
        assert 30 <= int(second.headers["retry-after"]) <= 600

    def test_runner_capacity_error_maps_to_the_same_code(
        self, client: TestClient, runner: BatchRunner, monkeypatch
    ) -> None:
        from api.jobs import BatchCapacityError

        def full(*_a, **_k):
            raise BatchCapacityError("busy right now", retryable=True)

        monkeypatch.setattr(runner, "submit", full)
        response = submit(
            client, manifest_for(["abv_mismatch.png"]), [image_part("abv_mismatch.png")]
        )
        assert response.status_code == 503
        assert response.json()["detail"] == {"code": "batch_capacity", "message": "busy right now"}
        assert response.headers["retry-after"] == "60"  # nothing running to estimate from


# ---------------------------------------------------------------------------
# Access token: header or query
# ---------------------------------------------------------------------------


class TestAccessToken:
    @pytest.fixture
    def gated(self, client: TestClient, app_state, monkeypatch) -> TestClient:
        monkeypatch.setattr(app_state.state, "access_token", "s3cret")
        return client

    def test_header_is_accepted(self, gated: TestClient) -> None:
        response = gated.get("/api/batch/template.csv", headers={"X-Access-Token": "s3cret"})
        assert response.status_code == 200

    def test_query_parameter_still_works(self, gated: TestClient) -> None:
        assert gated.get("/api/batch/template.csv?token=s3cret").status_code == 200

    def test_wrong_header_is_refused(self, gated: TestClient) -> None:
        response = gated.get("/api/batch/template.csv", headers={"X-Access-Token": "nope"})
        assert response.status_code == 401

    def test_header_takes_precedence_over_query(self, gated: TestClient) -> None:
        wrong_header = gated.get(
            "/api/batch/template.csv?token=s3cret", headers={"X-Access-Token": "stale"}
        )
        assert wrong_header.status_code == 401
        right_header = gated.get(
            "/api/batch/template.csv?token=stale", headers={"X-Access-Token": "s3cret"}
        )
        assert right_header.status_code == 200

    def test_single_label_accepts_the_header(self, gated: TestClient) -> None:
        response = gated.post(
            "/api/verify",
            headers={"X-Access-Token": "s3cret"},
            files={"image": ("l.png", (LABELS / "abv_mismatch.png").read_bytes(), "image/png")},
            data=RECORDS["abv_mismatch.png"],
        )
        assert response.status_code == 200
        assert gated.post("/api/verify", data={}).status_code == 401


# ---------------------------------------------------------------------------
# /api/verify never writes an upload to disk
# ---------------------------------------------------------------------------


@pytest.fixture
def disk_writes(monkeypatch) -> list[str]:
    """Record every temporary-file creation in the process.

    `SpooledTemporaryFile.rollover` — how Starlette spills a large upload to
    disk — creates its file through `tempfile.TemporaryFile`, so patching the
    tempfile constructors catches it along with any direct use.
    """
    import tempfile

    calls: list[str] = []
    for name in ("TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        original = getattr(tempfile, name)

        def recorder(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(tempfile, name, recorder)
    return calls


class TestSingleLabelUploadStaysInMemory:
    #: A 5 MB upload: five times Starlette's 1 MB spool threshold. PNG decoders
    #: stop at the IEND chunk, so the padding does not affect the label.
    LARGE = (LABELS / "abv_mismatch.png").read_bytes() + b"\0" * (5 * 1024 * 1024)

    def test_the_detector_catches_starlette_spooling(self, disk_writes: list[str]) -> None:
        """Control: the old `UploadFile` path does write this upload to disk."""
        control = FastAPI()

        @control.post("/upload")
        async def upload(image: UploadFile) -> dict:
            return {"size": len(await image.read())}

        response = TestClient(control).post(
            "/upload", files={"image": ("l.png", self.LARGE, "image/png")}
        )
        assert response.json()["size"] == len(self.LARGE)
        assert disk_writes, "the detector failed to see Starlette's rollover to disk"

    def test_large_single_label_upload_creates_no_temporary_file(
        self, client: TestClient, disk_writes: list[str]
    ) -> None:
        response = client.post(
            "/api/verify",
            files={"image": ("l.png", self.LARGE, "image/png")},
            data=RECORDS["abv_mismatch.png"],
        )
        assert response.status_code == 200, response.text
        assert disk_writes == []

    def test_oversized_upload_is_413(self, client: TestClient, app_state, monkeypatch) -> None:
        monkeypatch.setattr(app_state, "MAX_UPLOAD_BYTES", 1024 * 1024)
        response = client.post(
            "/api/verify",
            files={"image": ("l.png", self.LARGE, "image/png")},
            data=RECORDS["abv_mismatch.png"],
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "Image exceeds the 1 MB limit."

    def test_missing_field_is_422_naming_the_field(self, client: TestClient) -> None:
        record = dict(RECORDS["abv_mismatch.png"])
        del record["net_contents"]
        response = client.post(
            "/api/verify",
            files={"image": ("l.png", (LABELS / "abv_mismatch.png").read_bytes(), "image/png")},
            data=record,
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "net_contents"]

    def test_missing_image_is_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/verify", files={"x": ("x", b"", "text/plain")}, data=RECORDS["abv_mismatch.png"]
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "image"]

    def test_label_width_is_parsed_and_blank_is_absent(self, client: TestClient) -> None:
        image = (LABELS / "abv_mismatch.png").read_bytes()
        for width, expected in (("80", 200), ("", 200), ("wide", 422)):
            response = client.post(
                "/api/verify",
                files={"image": ("l.png", image, "image/png")},
                data={**RECORDS["abv_mismatch.png"], "label_width_mm": width},
            )
            assert response.status_code == expected, (width, response.text)


# ---------------------------------------------------------------------------
# RapidOcrProvider engine options
# ---------------------------------------------------------------------------


class TestProviderEngineOptions:
    def test_options_are_passed_through_and_default_is_unchanged(self, monkeypatch) -> None:
        import rapidocr_onnxruntime

        from api.jobs import build_batch_rapidocr
        from extraction.providers.rapidocr_provider import RapidOcrProvider

        seen: list[dict] = []

        class FakeEngine:
            def __init__(self, **kwargs) -> None:
                seen.append(kwargs)

        monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", FakeEngine)
        RapidOcrProvider()
        build_batch_rapidocr(1)
        assert seen == [{}, {"intra_op_num_threads": 1, "inter_op_num_threads": 1}]


# ---------------------------------------------------------------------------
# Item states: PENDING -> RUNNING -> DONE/ERROR, PENDING -> SKIPPED
# ---------------------------------------------------------------------------


class TestItemStates:
    def _state(self, client: TestClient, batch_id: str, row: int) -> str:
        return client.get(f"/api/batch/{batch_id}/items/{row}").json()["state"]

    def test_pending_running_done(self, client: TestClient, provider: FixtureProvider) -> None:
        gate = threading.Event()
        provider.gate = gate
        names = ["abv_mismatch.png", "net_mismatch.png"]
        batch_id = submit(client, manifest_for(names), [image_part(n) for n in names]).json()[
            "batch_id"
        ]
        assert provider.entered.wait(10)
        assert self._state(client, batch_id, 2) == ItemState.RUNNING
        assert self._state(client, batch_id, 3) == ItemState.PENDING
        gate.set()
        wait_done(client, batch_id)
        assert self._state(client, batch_id, 2) == ItemState.DONE
        assert self._state(client, batch_id, 3) == ItemState.DONE

    def test_running_to_error(self, client: TestClient) -> None:
        batch_id = submit(
            client, manifest_for(["broken.png"]), [image_part("broken.png", b"not an image")]
        ).json()["batch_id"]
        wait_done(client, batch_id)
        item = client.get(f"/api/batch/{batch_id}/items/2").json()
        assert item["state"] == ItemState.ERROR
        assert item["error"].startswith("Could not read this image")

    def test_an_item_handed_to_the_pool_before_a_cancel_never_starts(
        self, app_state, provider, clock
    ) -> None:
        """The start check and the PENDING -> SKIPPED move share one lock."""
        from api.jobs import _Batch, _Item
        from api.manifest import pair_files, parse_manifest

        runner = BatchRunner(
            ruleset=app_state.state.ruleset, provider_factory=lambda: provider, clock=clock
        )
        pairing = pair_files(parse_manifest(manifest_for(["a.png"])), ["a.png"])
        item = _Item(pair=pairing.pairs[0], order=0, image=b"x", size=1)
        batch = _Batch(batch_id="b", items=[item], pairing=pairing.report, created_at=clock())
        runner._batches["b"] = batch
        runner._held_bytes = 1
        runner.cancel("b")
        runner._run_item(batch, item)  # as if the pool picked it up just after
        assert item.state is ItemState.SKIPPED
        assert provider.calls == 0
        assert runner.held_bytes == 0


class TestWorklistOrderAllStates:
    def test_done_and_error_worst_first_then_running_pending_skipped(
        self, app_state, provider, clock
    ) -> None:
        """All five states in one snapshot, ordered exactly as the contract says.

        A live batch never shows PENDING and SKIPPED together (a stop turns
        every PENDING item into SKIPPED at once), so the snapshot is built
        directly — the ordering is a property of the snapshot, not of timing.
        """
        from api.jobs import _Batch, _Item, run_pipeline
        from api.manifest import pair_files, parse_manifest
        from rules.results import ApplicationRecord

        runner = BatchRunner(
            ruleset=app_state.state.ruleset, provider_factory=lambda: provider, clock=clock
        )
        names = [
            "p1.png",  # pending
            "s1.png",  # skipped
            "clear.png",  # done, all clear
            "r1.png",  # running
            "p2.png",  # pending
            "err.png",  # error
            "mismatch.png",  # done, asserted mismatch
            "s2.png",  # skipped
        ]
        pairing = pair_files(parse_manifest(manifest_for(names)), names)
        items = {
            p.filename: _Item(pair=p, order=i, image=None, size=0)
            for i, p in enumerate(pairing.pairs)
        }

        def result_for(fixture: str):
            return run_pipeline(
                (LABELS / fixture).read_bytes(),
                ApplicationRecord(**RECORDS[fixture]),
                provider=provider,
                ruleset=runner.ruleset,
                diagnostics=False,
                started_at=time.perf_counter(),
            )

        for name, fixture in (
            ("clear.png", "old_tom_compliant.png"),
            ("mismatch.png", "abv_mismatch.png"),
        ):
            item = items[name]
            item.result = result_for(fixture)
            item.summary = summarise(item.result)
            item.state = ItemState.DONE
        items["err.png"].error = "Could not read this image: broken"
        items["err.png"].summary = summarise(None, items["err.png"].error)
        items["err.png"].state = ItemState.ERROR
        items["r1.png"].state = ItemState.RUNNING
        for name in ("s1.png", "s2.png"):
            items[name].state = ItemState.SKIPPED

        batch = _Batch(
            batch_id="b",
            items=[items[p.filename] for p in pairing.pairs],
            pairing=pairing.report,
            created_at=clock(),
            state=BatchState.RUNNING,
            finished_count=3,
            failed_count=1,
        )
        status = runner._snapshot_locked(batch, False)
        order = [(i.filename, i.state) for i in status.items]

        finished = [i for i in status.items if i.state in (ItemState.DONE, ItemState.ERROR)]
        keys = [sort_key(items[i.filename].summary, i.filename) for i in finished]
        assert keys == sorted(keys)
        assert [name for name, _ in order[:3]] == [i.filename for i in finished]
        assert order[3:] == [
            ("r1.png", ItemState.RUNNING),
            ("p1.png", ItemState.PENDING),
            ("p2.png", ItemState.PENDING),
            ("s1.png", ItemState.SKIPPED),
            ("s2.png", ItemState.SKIPPED),
        ]
        # needs_attention: the mismatch and the unreadable image, plus both skipped.
        assert status.needs_attention == (sum(1 for i in finished if i.needs_attention) + 2)
        assert status.completed == 3
