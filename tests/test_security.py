"""Security regressions: each test repeats an attack from the security review.

The review (see docs/SECURITY.md) found ways for one request to exhaust the
server's memory or CPU, error replies that leaked internals, and missing
response headers. Every fix here is pinned by the attack it closes, so a later
change that reopens one fails the suite.
"""

from __future__ import annotations

import io
import struct
import time
import zlib
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.jobs import BatchCapacityError, BatchRunner, BatchSettings, neutralise_cell
from api.manifest import MAX_CELL_CHARS, MAX_MANIFEST_RECORDS, ManifestRejected, parse_manifest
from extraction import pipeline
from extraction.providers.base import ExtractionError

# Shared fixtures from the batch API suite (pytest resolves them by name).
from tests.test_batch_api import (  # noqa: F401
    LABELS,
    RECORDS,
    app_state,
    client,
    clock,
    image_part,
    manifest_for,
    provider,
    provider_template,
    runner,
    submit,
    wait_done,
)


def _png_header_only(width: int, height: int) -> bytes:
    """A tiny PNG that *declares* huge dimensions: the decompression-bomb shape."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 64)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _record() -> dict[str, str]:
    return RECORDS["old_tom_compliant.png"]


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------


class TestImageDecoding:
    def test_decompression_bomb_is_refused_before_decoding(self) -> None:
        """13300x13300 in a few hundred bytes: once decoded to about 1.4 GB."""
        bomb = _png_header_only(13300, 13300)
        assert len(bomb) < 1000
        started = time.perf_counter()
        with pytest.raises(ExtractionError, match="megapixels"):
            pipeline.preprocess(bomb)
        assert time.perf_counter() - started < 1.0, "refused by header, not after decoding"

    def test_pillow_warning_band_is_an_error_too(self) -> None:
        """Pillow only *warns* between its two thresholds; that is not enough."""
        width = 12_000
        height = int(1.2 * Image.MAX_IMAGE_PIXELS) // width  # inside the warn-only band
        with pytest.raises(ExtractionError, match="megapixels"):
            pipeline.preprocess(_png_header_only(width, height))

    def test_between_the_limit_and_pillows_backstop_is_refused_before_decoding(self) -> None:
        with pytest.raises(ExtractionError, match="megapixels"):
            pipeline.preprocess(_png_header_only(8000, 7000))  # 56 MP

    def test_format_is_judged_by_content_not_declared_type(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, "GIF")
        with pytest.raises(ExtractionError) as info:
            pipeline.preprocess(buffer.getvalue())
        assert str(info.value) == pipeline.UNDECODABLE_MESSAGE

    def test_undecodable_message_carries_no_internals(self) -> None:
        with pytest.raises(ExtractionError) as info:
            pipeline.preprocess(b"not an image at all")
        assert "BytesIO" not in str(info.value) and "0x" not in str(info.value)

    def test_large_jpeg_is_reduced_by_the_decoder(self) -> None:
        """A 48 MP phone JPEG is allowed, decoded at reduced scale, not refused."""
        buffer = io.BytesIO()
        Image.new("RGB", (8000, 6000), "white").save(buffer, "JPEG", quality=50)
        image = pipeline._decode(buffer.getvalue())
        assert image.size[0] * image.size[1] <= pipeline.MAX_SOURCE_PIXELS
        assert max(image.size) >= pipeline.TARGET_LONGEST_EDGE

    def test_exif_orientation_still_applied(self) -> None:
        source = Image.new("RGB", (900, 450), "white")
        exif = source.getexif()
        exif[0x0112] = 6  # rotate 90
        buffer = io.BytesIO()
        source.save(buffer, "PNG", exif=exif.tobytes())
        assert pipeline._decode(buffer.getvalue()).size == (450, 900)

    def test_bomb_through_the_api_is_a_plain_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/verify",
            files={"image": ("big.png", _png_header_only(13300, 13300), "image/png")},
            data=_record(),
        )
        assert response.status_code == 422
        assert "megapixels" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Multipart parsing
# ---------------------------------------------------------------------------


def _raw_multipart(client: TestClient, path: str, body: bytes, boundary: str = "b0undary"):
    return client.post(
        path,
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )


class TestMultipartLimits:
    def test_oversized_part_header_is_refused(self, client: TestClient) -> None:
        """A 20 MB header once took ten seconds of CPU on the event loop."""
        body = (
            b'--b0undary\r\nContent-Disposition: form-data; name="x"\r\nX-Pad: '
            + b"a" * 100_000
            + b"\r\n\r\nv\r\n--b0undary--\r\n"
        )
        response = _raw_multipart(client, "/api/verify", body)
        assert response.status_code == 400

    def test_too_many_parts_is_refused(self, client: TestClient) -> None:
        part = b'--b0undary\r\nContent-Disposition: form-data; name="x"\r\n\r\n\r\n'
        body = part * 2000 + b"--b0undary--\r\n"
        response = _raw_multipart(client, "/api/verify", body)
        assert response.status_code == 400

    def test_declared_length_over_the_limit_is_refused_before_reading(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/verify",
            content=b"x",
            headers={
                "content-type": "multipart/form-data; boundary=b0undary",
                "content-length": str(500 * 1024 * 1024),
            },
        )
        assert response.status_code == 413

    def test_unexpected_file_fields_are_not_held(self, client: TestClient) -> None:
        """Files under a field the endpoint does not read are discarded."""
        response = client.post(
            "/api/verify",
            files=[("junk", ("j.bin", b"z" * 1024, "application/octet-stream"))],
            data=_record(),
        )
        assert response.status_code == 422  # "image missing", as before


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifestLimits:
    def test_too_many_rows_is_refused(self) -> None:
        text = "filename,brand_name,class_type,alcohol_content,net_contents\n" + "a\n" * (
            MAX_MANIFEST_RECORDS + 5
        )
        with pytest.raises(ManifestRejected) as info:
            parse_manifest(text.encode())
        assert info.value.status_code == 413

    def test_long_cell_is_refused(self) -> None:
        text = "filename,brand_name\n" + "x.png," + "B" * (MAX_CELL_CHARS + 1) + "\n"
        with pytest.raises(ManifestRejected):
            parse_manifest(text.encode())

    def test_unterminated_quote_past_the_csv_field_limit_is_400_not_500(
        self, client: TestClient
    ) -> None:
        """Once an unhandled `_csv.Error` and a 500."""
        manifest = b'filename,brand_name\n"' + b"a" * 200_000 + b"\n"
        response = client.post(
            "/api/batch/check",
            files=[("manifest", ("m.csv", manifest, "text/csv"))],
            data={"filenames": ["a.png"]},
        )
        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_csv_export_neutralises_whitespace_before_a_formula(self) -> None:
        assert neutralise_cell("  =HYPERLINK()") == "'  =HYPERLINK()"
        assert neutralise_cell("\n@SUM(1)") == "'\n@SUM(1)"
        assert neutralise_cell(" plain") == " plain"


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


class TestResourceLimits:
    def test_single_label_checks_beyond_the_limit_get_503(
        self, client: TestClient, app_state, monkeypatch
    ) -> None:
        class Full:
            def locked(self) -> bool:
                return True

        monkeypatch.setattr(app_state, "_VERIFY_SLOTS", Full())
        response = client.post(
            "/api/verify",
            files={
                "image": ("l.png", (LABELS / "old_tom_compliant.png").read_bytes(), "image/png")
            },
            data=_record(),
        )
        assert response.status_code == 503
        assert response.headers["retry-after"]

    def test_second_concurrent_batch_upload_gets_503(self, client: TestClient, monkeypatch) -> None:
        from api import batch as batch_module

        class Busy:
            def locked(self) -> bool:
                return True

        monkeypatch.setattr(batch_module, "_UPLOAD_SLOT", Busy())
        response = submit(
            client, manifest_for(["old_tom_compliant.png"]), [image_part("old_tom_compliant.png")]
        )
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "batch_capacity"

    def test_batch_cap_drops_the_oldest_finished_batch(self, app_state, provider, clock) -> None:
        from api.manifest import pair_files

        runner = BatchRunner(
            ruleset=app_state.state.ruleset,
            provider_factory=lambda: provider,
            settings=BatchSettings(workers=1, nice=0, sweep_interval_s=0.05, max_batches=2),
            clock=clock,
        )
        try:
            data = (LABELS / "old_tom_compliant.png").read_bytes()
            parsed = parse_manifest(manifest_for(["old_tom_compliant.png"]))
            pairing = pair_files(parsed, ["old_tom_compliant.png"])

            def run_one() -> str:
                batch_id = runner.submit(pairing.pairs, [data], pairing.report)
                deadline = time.monotonic() + 10
                while runner.status(batch_id).state not in ("done", "cancelled"):
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                clock.now += timedelta(seconds=1)
                return batch_id

            first, second = run_one(), run_one()
            third = run_one()
            assert third
            with pytest.raises(KeyError):
                runner.status(first)
            assert runner.status(second).state == "done"
        finally:
            runner.close()

    def test_batch_cap_refuses_when_every_held_batch_is_still_running(
        self, app_state, provider, clock
    ) -> None:
        import threading

        from api.manifest import pair_files

        provider.gate = threading.Event()
        runner = BatchRunner(
            ruleset=app_state.state.ruleset,
            provider_factory=lambda: provider,
            settings=BatchSettings(workers=1, nice=0, sweep_interval_s=0.05, max_batches=1),
            clock=clock,
        )
        try:
            data = (LABELS / "old_tom_compliant.png").read_bytes()
            parsed = parse_manifest(manifest_for(["old_tom_compliant.png"]))
            pairing = pair_files(parsed, ["old_tom_compliant.png"])
            runner.submit(pairing.pairs, [data], pairing.report)
            with pytest.raises(BatchCapacityError):
                runner.submit(pairing.pairs, [data], pairing.report)
        finally:
            provider.gate.set()
            runner.close()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TestResponses:
    def test_security_headers_on_page_and_api(self, client: TestClient) -> None:
        for path in ("/api/health", "/api/ruleset"):
            headers = client.get(path).headers
            assert "frame-ancestors 'none'" in headers["content-security-policy"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["referrer-policy"] == "no-referrer"
            assert headers["cache-control"] == "no-store"

    def test_api_docs_are_off(self, client: TestClient) -> None:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404

    def test_health_does_not_publish_engine_errors(
        self, client: TestClient, app_state, monkeypatch
    ) -> None:
        monkeypatch.setattr(app_state.state, "provider_error", "/opt/secret/path: bad onnx")
        body = client.get("/api/health").json()
        assert "/opt/secret" not in str(body)

    def test_wrong_token_is_401_and_right_token_passes(
        self, client: TestClient, app_state, monkeypatch
    ) -> None:
        monkeypatch.setattr(app_state.state, "access_token", "s3cret")
        assert client.get("/api/batch/template.csv").status_code == 401
        assert (
            client.get("/api/batch/template.csv", headers={"X-Access-Token": "s3creT"}).status_code
            == 401
        )
        assert (
            client.get("/api/batch/template.csv", headers={"X-Access-Token": "s3cret"}).status_code
            == 200
        )
