"""Batch endpoints. The contract is `api.batch_models`; this module implements it.

Wiring: `api.main` calls `configure(state, ...)` once and includes `router`
with the shared access-token dependency, so every route here sits behind the
same gate as single-label verification without this module importing
`api.main` (which would be circular). Tests swap the runner through FastAPI's
`dependency_overrides[get_runner]` rather than monkeypatching globals.

Why the upload is parsed by hand
--------------------------------

Starlette's form parser spools every uploaded file larger than 1 MB to a
temporary file **on disk**. For a batch — hundreds of phone photographs — that
would put label images on the instance's disk, breaking the "nothing is
persisted" boundary in TRADEOFFS.md. It also reads the entire body before the
handler runs, so a 3 GB submission would be accepted in full before being told
it was too large. `read_multipart` streams the body through python-multipart
into memory instead, and stops reading the moment a count or size limit is
crossed.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from python_multipart.multipart import MultipartParser, parse_options_header

from api.batch_models import MAX_BATCH_SIZE, BatchItem, BatchStatus, PairingReport
from api.jobs import (
    BatchCapacityError,
    BatchNotFound,
    BatchRunner,
    BatchSettings,
    ItemNotFound,
    default_provider_factory,
    results_csv,
)
from api.manifest import (
    FileEntry,
    Pairing,
    is_os_clutter,
    pair_files,
    parse_manifest,
    template_csv,
)

#: A manifest for 500 labels is tens of kilobytes; this is generous headroom
#: while still refusing a mistaken multi-megabyte upload in the manifest slot.
MAX_MANIFEST_BYTES = 2 * 1024 * 1024

#: Allowance for multipart framing (boundaries, part headers) on top of the
#: image bytes when judging Content-Length before reading anything.
_FRAMING_ALLOWANCE = 4 * 1024 * 1024

#: Non-file fields (filenames in the check request) are short strings.
_MAX_FIELD_BYTES = 4096

router = APIRouter(prefix="/api/batch", tags=["batch"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    runner: BatchRunner | None = None
    service_state: Any = None
    accepted_content_types: Collection[str] = frozenset()
    max_upload_bytes: int = 20 * 1024 * 1024


_config = _Config()


def configure(
    service_state: Any,
    *,
    accepted_content_types: Collection[str],
    max_upload_bytes: int,
    runner: BatchRunner | None = None,
) -> BatchRunner:
    """Bind the router to the running service. Called once by `api.main`.

    The per-image rules (accepted content types, per-file size) are passed in
    rather than redefined so that batch and single-label validate uploads
    identically — a file accepted on one screen and refused on the other would
    be indefensible to an agent.
    """
    _config.service_state = service_state
    _config.accepted_content_types = frozenset(accepted_content_types)
    _config.max_upload_bytes = max_upload_bytes
    if runner is None:
        settings = BatchSettings.from_env()
        runner = BatchRunner(
            ruleset=service_state.ruleset,
            provider_factory=default_provider_factory(service_state.provider, settings.ocr_threads),
            diagnostics=service_state.diagnostics_enabled,
            settings=settings,
        )
    _config.runner = runner
    return runner


def get_runner() -> BatchRunner:
    """Dependency: the process's batch runner. Overridden in tests."""
    if _config.runner is None:  # pragma: no cover - wiring error
        raise HTTPException(
            status_code=503,
            detail={"code": "batch_unavailable", "message": "Batch processing is not configured."},
        )
    return _config.runner


def _extraction_available() -> bool:
    state = _config.service_state
    return state is None or getattr(state, "provider", None) is not None


def _not_found(batch_id: str, runner: BatchRunner) -> HTTPException:
    hours = runner.settings.ttl.total_seconds() / 3600
    kept = f"{hours:g} hour{'s' if hours != 1 else ''}"
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            "This batch could not be found. Batches are kept for "
            f"{kept} after they finish and are discarded when cancelled, so it may "
            "have expired. Download the results CSV to keep a record."
        ),
    )


# ---------------------------------------------------------------------------
# Streaming multipart reader
# ---------------------------------------------------------------------------


@dataclass
class Upload:
    field_name: str
    filename: str | None
    content_type: str | None
    data: bytearray | None = field(default_factory=bytearray)
    size: int = 0
    too_large: bool = False


@dataclass
class Form:
    fields: dict[str, list[str]] = field(default_factory=dict)
    files: list[Upload] = field(default_factory=list)


class Rejected(Exception):
    """A request refused before processing, with the response to send.

    A 503 always carries a machine-readable `code` (see the contract docstring
    in `api.batch_models`), because a client has to tell "the OCR engine is
    missing" (retrying will not help) from "the service is busy" (retry later)
    without parsing English. Other statuses keep a plain-string `detail`.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        code: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.retry_after = retry_after

    def response(self) -> JSONResponse:
        body: Any = self.detail
        if self.code is not None:
            body = {"code": self.code, "message": self.detail}
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after is not None else None
        return JSONResponse(status_code=self.status_code, content={"detail": body}, headers=headers)


#: 503 codes. Documented in the `api.batch_models` docstring.
CODE_EXTRACTION_UNAVAILABLE = "extraction_unavailable"
CODE_BATCH_CAPACITY = "batch_capacity"


async def read_multipart(
    request: Request,
    *,
    max_files: int,
    max_total_bytes: int,
    per_file_bytes: int,
    total_limit_message: str,
    total_limit_rejection: Rejected | None = None,
    counted_field: str = "images",
) -> Form:
    """Parse a multipart body into memory, enforcing limits while streaming.

    Used by both batch submission and `/api/verify`: Starlette's own parser
    spools any file over 1 MB to a temporary file on disk, and this one never
    touches disk. `total_limit_rejection`, when given, replaces the default
    413 for crossing `max_total_bytes` (the batch endpoint uses it to answer
    503 when the limit is the shared budget rather than the batch's size).

    Limits are checked as bytes arrive, not afterwards: exceeding one stops
    the read with a clear message instead of first accepting gigabytes. A
    single file over `per_file_bytes` is not an error for the whole request —
    its bytes are discarded and it is marked `too_large`, so it can be reported
    as one unusable file among many.
    """
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise Rejected(400, "Send the manifest and images as a multipart form upload.")

    form = Form()
    current: dict[str, Any] = {}
    header_name = bytearray()
    header_value = bytearray()
    totals = {"bytes": 0, "images": 0}

    def on_part_begin() -> None:
        current.clear()
        current["headers"] = {}

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_name.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        current["headers"][bytes(header_name).lower()] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def on_headers_finished() -> None:
        headers = current["headers"]
        _, disposition = parse_options_header(headers.get(b"content-disposition", b""))
        name = disposition.get(b"name", b"").decode("utf-8", "replace")
        filename = disposition.get(b"filename")
        if filename is None:
            current["field"] = (name, bytearray())
            return
        decoded = filename.decode("utf-8", "replace")
        part_type = headers.get(b"content-type", b"").decode("latin-1").split(";")[0].strip()
        upload = Upload(field_name=name, filename=decoded, content_type=part_type or None)
        if name == counted_field and not is_os_clutter(decoded):
            totals["images"] += 1
            if totals["images"] > max_files:
                raise Rejected(
                    413,
                    f"A batch can hold at most {max_files} images. Split the submission "
                    "into smaller batches.",
                )
        form.files.append(upload)
        current["file"] = upload

    def on_part_data(data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        totals["bytes"] += len(chunk)
        if totals["bytes"] > max_total_bytes:
            raise total_limit_rejection or Rejected(413, total_limit_message)
        upload: Upload | None = current.get("file")
        if upload is None:
            buffer = current["field"][1]
            if len(buffer) + len(chunk) > _MAX_FIELD_BYTES:
                raise Rejected(400, "A form field in this request is unexpectedly long.")
            buffer.extend(chunk)
            return
        upload.size += len(chunk)
        limit = MAX_MANIFEST_BYTES if upload.field_name == "manifest" else per_file_bytes
        if upload.size > limit:
            upload.too_large = True
            upload.data = None  # stop holding it; keep counting for the message
        elif upload.data is not None:
            upload.data.extend(chunk)

    def on_part_end() -> None:
        if "field" in current:
            name, buffer = current["field"]
            form.fields.setdefault(name, []).append(buffer.decode("utf-8", "replace"))

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )
    async for chunk in request.stream():
        parser.write(chunk)
    parser.finalize()
    return form


def _manifest_bytes(form: Form) -> bytes:
    manifests = [f for f in form.files if f.field_name == "manifest"]
    if not manifests:
        raise Rejected(400, "Include the manifest CSV file in the upload.")
    manifest = manifests[0]
    if manifest.too_large or manifest.data is None:
        raise Rejected(
            413,
            f"The manifest is larger than {MAX_MANIFEST_BYTES // (1024 * 1024)} MB. "
            "It should be a CSV of filenames and application details.",
        )
    return bytes(manifest.data)


def _rejection_response(exc: Rejected) -> JSONResponse:
    return exc.response()


def _busy(message: str, runner: BatchRunner) -> Rejected:
    return Rejected(
        503, message, code=CODE_BATCH_CAPACITY, retry_after=runner.retry_after_seconds()
    )


def _beverage_classes() -> Collection[str] | None:
    state = _config.service_state
    ruleset = getattr(state, "ruleset", None)
    return set(ruleset.beverage_classes) if ruleset is not None else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/template.csv")
async def manifest_template() -> Response:
    """The manifest template: required and optional columns, one example row."""
    return Response(
        content=template_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="label-manifest-template.csv"'},
    )


@router.post("/check", response_model=PairingReport)
async def check_pairing(request: Request) -> Any:
    """Pair a manifest with a list of filenames, before any image is uploaded.

    Form fields: `manifest` (the CSV file) and `filenames` (repeated, one per
    image the agent selected). Folder paths in the names are ignored.
    """
    try:
        form = await read_multipart(
            request,
            max_files=0,
            max_total_bytes=MAX_MANIFEST_BYTES + 5 * MAX_BATCH_SIZE * _MAX_FIELD_BYTES,
            per_file_bytes=MAX_MANIFEST_BYTES,
            total_limit_message="This request is too large to be a manifest and a filename list.",
        )
        manifest = _manifest_bytes(form)
    except Rejected as exc:
        return _rejection_response(exc)

    names = [n for n in form.fields.get("filenames", []) if n.strip()]
    images = [n for n in names if not is_os_clutter(n)]
    if len(images) > MAX_BATCH_SIZE:
        return _rejection_response(
            Rejected(
                413,
                f"{len(images)} images were selected; a batch can hold at most "
                f"{MAX_BATCH_SIZE}. Split the submission into smaller batches.",
            )
        )

    parsed = parse_manifest(manifest, beverage_classes=_beverage_classes())
    return pair_files(parsed, names).report


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=BatchStatus)
async def submit_batch(request: Request, runner: BatchRunner = Depends(get_runner)) -> Any:
    """Accept a manifest and images; queue every matched pair.

    Form fields: `manifest` (the CSV file) and `images` (repeated files).
    Unusable files — wrong type, empty, over the per-image limit — are reported
    in the pairing and skipped, not treated as a failure of the whole batch.
    """
    # Before reading the body, unlike single-label: here the body may be a
    # gigabyte, and uploading it only to hear that OCR is down wastes the
    # agent's time and the network's.
    if not _extraction_available():
        return Rejected(
            503,
            f"Extraction is unavailable: {_config.service_state.provider_error}",
            code=CODE_EXTRACTION_UNAVAILABLE,
        ).response()

    cap = runner.settings.max_bytes
    budget = runner.remaining_budget()
    too_big = (
        f"This upload is over the {cap / 1024**3:.1f} GB limit for one batch. "
        "Split the submission into smaller batches."
    )
    busy = (
        "Other batches are still being processed and the service is holding as many "
        "images as it can. Try again when they have finished."
    )
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        if int(declared) > cap + _FRAMING_ALLOWANCE:
            return _rejection_response(Rejected(413, too_big))
        if int(declared) > budget + _FRAMING_ALLOWANCE:
            return _busy(busy, runner).response()

    try:
        form = await read_multipart(
            request,
            max_files=MAX_BATCH_SIZE,
            max_total_bytes=min(cap, budget) + _FRAMING_ALLOWANCE,
            per_file_bytes=_config.max_upload_bytes,
            total_limit_message=too_big,
            total_limit_rejection=None if budget >= cap else _busy(busy, runner),
        )
        manifest = _manifest_bytes(form)
    except Rejected as exc:
        return _rejection_response(exc)

    uploads = [f for f in form.files if f.field_name == "images"]
    entries: list[FileEntry] = []
    payloads: list[bytes] = []
    limit_mb = _config.max_upload_bytes // (1024 * 1024)
    for upload in uploads:
        problem: str | None = None
        if upload.content_type not in _config.accepted_content_types:
            problem = (
                "is not a supported image type. Use JPEG, PNG, WebP, TIFF or BMP; "
                "PDF artwork is not handled by this tool"
            )
        elif upload.too_large:
            problem = f"is larger than the {limit_mb} MB limit for one image"
        elif upload.size == 0:
            problem = "is empty"
        entries.append(FileEntry(name=upload.filename or "", problem=problem))
        payloads.append(bytes(upload.data) if problem is None and upload.data else b"")
        upload.data = None  # the bytes object above is now the only copy

    parsed = parse_manifest(manifest, beverage_classes=_beverage_classes())
    pairing: Pairing = pair_files(parsed, entries)
    if pairing.report.blocking:
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "message": "Nothing in this batch could be checked. See the issues listed.",
                    "pairing": pairing.report.model_dump(mode="json"),
                }
            },
        )

    try:
        batch_id = runner.submit(pairing.pairs, payloads, pairing.report)
    except BatchCapacityError as exc:
        if exc.retryable:
            return _busy(str(exc), runner).response()
        return Rejected(413, str(exc)).response()
    finally:
        del payloads

    return runner.status(batch_id)


@router.get("/{batch_id}", response_model=BatchStatus)
async def batch_status(batch_id: str, runner: BatchRunner = Depends(get_runner)) -> Any:
    """The batch and its worklist, completed items worst-first, summaries only.

    `result` is None on every item here; fetch one item's full result from
    `/items/{row_number}` when the agent opens it.
    """
    try:
        return runner.status(batch_id)
    except BatchNotFound:
        raise _not_found(batch_id, runner) from None


@router.get("/{batch_id}/items/{row_number}", response_model=BatchItem)
async def batch_item(
    batch_id: str, row_number: int, runner: BatchRunner = Depends(get_runner)
) -> Any:
    """One label with its full verification result, keyed by manifest row.

    A pending item is returned in its current state with `result` None.
    """
    try:
        return runner.item(batch_id, row_number)
    except BatchNotFound:
        raise _not_found(batch_id, runner) from None
    except ItemNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Row {row_number} of the manifest is not part of this batch. Only rows "
                "that were paired with an image are checked."
            ),
        ) from None


@router.get("/{batch_id}/results.csv")
async def batch_results_csv(batch_id: str, runner: BatchRunner = Depends(get_runner)) -> Response:
    """The worklist as a spreadsheet — how an agent keeps a record.

    Written with a UTF-8 byte order mark: without it Excel on Windows reads the
    file as Windows-1252 and mangles any accented brand name.
    """
    try:
        snapshot = runner.status(batch_id, include_results=True)
    except BatchNotFound:
        raise _not_found(batch_id, runner) from None
    body = "﻿" + results_csv(snapshot, runner.ruleset)
    return Response(
        content=body.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_id[:8]}-results.csv"'},
    )


@router.delete("/{batch_id}", response_model=BatchStatus)
async def cancel_batch(batch_id: str, runner: BatchRunner = Depends(get_runner)) -> Any:
    """Stop the batch; keep what it has done.

    No further item is started. Finished items, the CSV and the status stay
    available (state `cancelled`) until the batch expires on the normal TTL.
    Every PENDING item becomes SKIPPED at once; a RUNNING item finishes (DONE
    or ERROR) and its result is kept. Returns the batch as it now stands
    (summaries only), so the RUNNING item may still show as RUNNING.
    """
    try:
        return runner.cancel(batch_id)
    except BatchNotFound:
        raise _not_found(batch_id, runner) from None
