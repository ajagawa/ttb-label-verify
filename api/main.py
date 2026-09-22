"""FastAPI application.

One process serves both the API and the built frontend. No database, no queue
broker, no external service: uploaded images are held in memory, passed through
the pipeline, and discarded when the response is written. Nothing touches disk.

That is a functional constraint rather than a simplification — the discovery
notes specified storing nothing sensitive, and making ephemeral processing
structural means there is no retention policy to write and no residue to audit.
The cost is real and recorded in TRADEOFFS.md: no result history, and no
capture of agent overrides, which is the only route to measuring whether the
tool's judgements are correct over time.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from api.batch import Form, Rejected, read_multipart
from api.batch import configure as configure_batch
from api.batch import router as batch_router
from api.jobs import run_pipeline
from extraction.providers import get_provider
from extraction.providers.base import ExtractionError, ExtractionProvider
from rules.results import ApplicationRecord, VerificationResult
from rules.schema import UnsupportedBeverageClass, load_ruleset

logger = logging.getLogger("label_verify")

#: Reject oversized uploads before reading them into memory. A label photograph
#: from any phone is comfortably inside this; anything larger is a mistake or an
#: attempt to exhaust memory on a small instance.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ACCEPTED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
)


# Before anything starts a thread: confine the process to the cores its CPU
# quota allows, so batch work's lower priority actually applies. Affinity is
# inherited only by threads created afterwards. See extraction/cpu.py.
from extraction.cpu import pin_to_quota as _pin_to_quota  # noqa: E402
from extraction.cpu import visible_cpus as _visible_cpus  # noqa: E402

HOST_VISIBLE_CPUS = _visible_cpus()
PINNED_CPUS = _pin_to_quota()


class ServiceState:
    """Process-wide singletons.

    The extraction provider loads model weights, and the rule set is parsed and
    validated. Both are constructed once at startup: paying either cost per
    request would put the five-second target out of reach, and a rule set that
    fails validation must stop the process rather than surface as a runtime
    error on somebody's upload.
    """

    def __init__(self) -> None:
        self.ruleset = load_ruleset()
        self.provider: ExtractionProvider | None = None
        self.provider_error: str | None = None
        self.diagnostics_enabled = os.environ.get("LABEL_VERIFY_DIAGNOSTICS", "0") == "1"
        self.access_token = os.environ.get("LABEL_VERIFY_ACCESS_TOKEN") or None

        try:
            self.provider = get_provider()
        except ExtractionError as exc:
            # Recorded rather than raised. The API still starts, /api/health
            # reports the degradation, and the failure is legible instead of
            # being a container that will not boot with no explanation.
            self.provider_error = str(exc)
            logger.warning("extraction provider unavailable: %s", exc)

        from extraction.cpu import cgroup_cpu_quota, ocr_threads

        self.cpu_quota = cgroup_cpu_quota()
        self.visible_cpus = HOST_VISIBLE_CPUS
        self.pinned_cpus = PINNED_CPUS
        self.ocr_threads = ocr_threads()
        self.warmup_ms: list[float] | None = None
        logger.info(
            "cpu: quota=%s visible=%d pinned=%s ocr_threads=%d",
            self.cpu_quota,
            self.visible_cpus,
            self.pinned_cpus,
            self.ocr_threads,
        )
        if self.provider is not None and os.environ.get("LABEL_VERIFY_WARMUP") == "1":
            self.warmup_ms = self._warm_up()

    def _warm_up(self, runs: int = 2) -> list[float] | None:
        """Run the full pipeline on a bundled sample before serving traffic.

        The first inference in a fresh process is much slower than the rest
        (ONNX Runtime allocates its memory arenas and picks kernels on first
        use): measured 5.4 s against 2.8 s under a one-CPU quota. Without this,
        the first person to use a freshly deployed instance pays that cost.
        Runs at startup, before the server accepts connections, so the platform
        does not route traffic here until it is done.

        Returns the time of each run in ms. The last one is what a check costs
        on this host once warm, and /api/health reports it, so the deployed
        machine's real speed is visible without uploading anything.
        Enabled by `LABEL_VERIFY_WARMUP=1` (set in the Dockerfile) so the test
        suite does not pay for it on every import.
        """
        sample = (
            Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "old_tom_compliant.png"
        )
        record = ApplicationRecord(
            brand_name="OLD TOM DISTILLERY",
            class_type="Kentucky Straight Bourbon Whiskey",
            alcohol_content="45% Alc./Vol.",
            net_contents="750 mL",
        )
        try:
            payload = sample.read_bytes()
            timings = []
            for _ in range(runs):
                started = time.perf_counter()
                run_pipeline(
                    payload,
                    record,
                    provider=self.provider,
                    ruleset=self.ruleset,
                    diagnostics=False,
                    started_at=started,
                )
                timings.append(round((time.perf_counter() - started) * 1000, 1))
        except Exception as exc:  # noqa: BLE001 - a failed warm-up must not stop the service
            logger.warning("warm-up skipped: %s", exc)
            return None
        logger.info("warm-up: %s ms", timings)
        return timings


state = ServiceState()

#: The interactive API docs (/docs, /redoc, /openapi.json) are off unless
#: asked for. They need no token, map every endpoint for anyone scanning the
#: host, and the docs pages load their scripts from a public CDN — outbound
#: traffic this service otherwise never causes (security review).
_API_DOCS = os.environ.get("LABEL_VERIFY_API_DOCS") == "1"

app = FastAPI(
    title="TTB Label Verification",
    version="0.1.0",
    description=(
        "Verifies label artwork against an application record and against the "
        "health warning requirements of 27 CFR Part 16. Recommends; does not adjudicate."
    ),
    docs_url="/docs" if _API_DOCS else None,
    redoc_url="/redoc" if _API_DOCS else None,
    openapi_url="/openapi.json" if _API_DOCS else None,
)

#: Response headers on everything. The page loads only its own script and
#: stylesheet; label images are shown from blob: URLs and the favicon is a
#: data: URL. frame-ancestors stops the page being framed (clickjacking of
#: "Stop batch"); no-referrer keeps a `?token=` page URL out of Referer
#: headers; results are never cached, since they describe an applicant's
#: label.
SECURITY_HEADERS: dict[bytes, bytes] = {
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
        b"connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        b"frame-ancestors 'none'"
    ),
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"strict-transport-security": b"max-age=31536000",
    b"permissions-policy": b"geolocation=(), microphone=(), camera=()",
}


class SecurityHeadersMiddleware:
    """Plain ASGI middleware, so request bodies still stream untouched."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope.get("path", "").startswith("/api/")

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name not in present:
                        headers.append((name, value))
                if is_api and b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityHeadersMiddleware)

# The frontend is served from this same origin in the container. CORS is opened
# only for local development, where Vite runs on its own port.
if os.environ.get("LABEL_VERIFY_DEV_CORS") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


async def require_access(
    token: str | None = None,
    x_access_token: str | None = Header(default=None),
) -> None:
    """Shared-secret gate for the public demo instance.

    Not authentication. A deployed prototype is a public URL running an image
    pipeline, and leaving it ungated invites abuse of the instance. Production
    would use agency SSO or PIV/CAC — see TRADEOFFS.md, "What production would
    require".

    Unset means open, which is correct for local use and wrong for a deployed
    URL.
    """
    if state.access_token is None:
        return
    # The `X-Access-Token` header is preferred: a query string lands in server
    # and proxy access logs, a header does not. `?token=` still works so that
    # existing demo links keep working. When both are sent the header decides,
    # so a stale link cannot override a client that sends the header.
    supplied = x_access_token if x_access_token is not None else token
    # Constant-time: an ordinary comparison returns as soon as a character
    # differs, which leaks how much of a guess was right.
    if supplied is None or not hmac.compare_digest(
        supplied.encode("utf-8"), state.access_token.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This demo instance requires an access token.",
        )


# ---------------------------------------------------------------------------
# Health and metadata
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    ruleset_version: str
    provider: str | None
    provider_error: str | None
    diagnostics_enabled: bool
    #: CPUs allowed by the container's quota (None: no quota), cores visible,
    #: and the ONNX thread count chosen from them. See extraction/cpu.py.
    cpu_quota: float | None = None
    visible_cpus: int | None = None
    ocr_threads: int | None = None
    #: Cores the process was confined to at startup (None: not pinned).
    pinned_cpus: list[int] | None = None
    #: Startup warm-up timings in ms; the last is a warm check on this host.
    warmup_ms: list[float] | None = None


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness plus the two facts that explain most failures.

    Reports "degraded" rather than failing when the OCR engine is missing: the
    rules engine still works, and an operator needs to be able to tell the
    difference between a dead process and one that booted without its model.
    """
    return HealthResponse(
        status="ok" if state.provider is not None else "degraded",
        ruleset_version=state.ruleset.version,
        provider=state.provider.name if state.provider else None,
        # The detail is logged at startup, not published: /api/health needs no
        # token, and an exception message can describe the server's internals.
        provider_error="OCR engine did not start; see the server log."
        if state.provider_error
        else None,
        diagnostics_enabled=state.diagnostics_enabled,
        cpu_quota=state.cpu_quota,
        visible_cpus=state.visible_cpus,
        ocr_threads=state.ocr_threads,
        pinned_cpus=state.pinned_cpus,
        warmup_ms=state.warmup_ms,
    )


class FieldDescriptor(BaseModel):
    id: str
    label: str
    citation: str
    citation_url: str | None
    required: bool
    implemented: bool


@app.get("/api/ruleset")
async def ruleset_metadata() -> dict:
    """What this deployment checks, and under which rules.

    The frontend renders its form and result rows from this rather than from a
    hardcoded list, so adding a field to the rule set changes the UI without a
    frontend change. `implemented` is exposed deliberately: the interface says
    which checks are automated and which are not, rather than presenting a
    partial tool as a complete one.
    """
    from rules.comparators import STRATEGIES, StrategyNotImplemented
    from rules.engine import load_strategies

    load_strategies()

    def is_implemented(strategy_name: str) -> bool:
        implementation = STRATEGIES.get(strategy_name)
        if implementation is None:
            return False
        try:
            # Declared-but-unwritten strategies raise on their first line, so
            # probing with an empty context is enough to tell them apart.
            implementation(None)  # type: ignore[arg-type]
        except StrategyNotImplemented:
            return False
        except Exception:  # noqa: BLE001 - any other failure means it is written
            return True
        return True

    return {
        "version": state.ruleset.version,
        "effective_date": state.ruleset.effective_date,
        "description": state.ruleset.description,
        "beverage_classes": {
            key: value.label for key, value in state.ruleset.beverage_classes.items()
        },
        "fields": [
            FieldDescriptor(
                id=rule.id,
                label=rule.label,
                citation=rule.citation,
                citation_url=rule.citation_url,
                required=rule.required,
                implemented=is_implemented(rule.strategy),
            ).model_dump()
            for rule in state.ruleset.display_order()
        ],
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


#: Documents the multipart body for OpenAPI. The handler reads the body itself
#: (see `verify_label`), so FastAPI cannot infer this from parameters.
_VERIFY_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": [
                    "image",
                    "brand_name",
                    "class_type",
                    "alcohol_content",
                    "net_contents",
                ],
                "properties": {
                    "image": {
                        "type": "string",
                        "format": "binary",
                        "description": "Label artwork, raster image",
                    },
                    "brand_name": {"type": "string"},
                    "class_type": {"type": "string"},
                    "alcohol_content": {"type": "string"},
                    "net_contents": {"type": "string"},
                    "beverage_class": {"type": "string", "default": "distilled_spirits"},
                    "label_width_mm": {"type": "number"},
                },
            }
        }
    },
}

_RECORD_FIELDS = (
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
    "beverage_class",
    "label_width_mm",
)


@app.post(
    "/api/verify",
    response_model=VerificationResult,
    dependencies=[Depends(require_access)],
    openapi_extra={"requestBody": _VERIFY_REQUEST_BODY},
)
async def verify_label(request: Request) -> Any:
    """Verify one label; refuse promptly when too many are already in flight.

    Each in-flight check holds its upload (up to MAX_UPLOAD_BYTES) in memory,
    and checks queue for the one OCR engine anyway, so beyond a handful there
    is nothing to gain from accepting more — only memory to lose. A client
    told "busy, retry in a few seconds" is better served than one whose
    request is held, or than a server that runs out of memory.
    """
    if _VERIFY_SLOTS.locked():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "The service is busy checking other labels. Try again in a few seconds."
            },
            headers={"Retry-After": "5"},
        )
    async with _VERIFY_SLOTS:
        return await _verify_label(request)


#: Single-label checks accepted at once (uploading, waiting or running).
MAX_CONCURRENT_VERIFY = 8
_VERIFY_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_VERIFY)


async def _verify_label(request: Request) -> VerificationResult:
    """Verify one label against its application record.

    The clock starts before the image is read, not after: the elapsed time
    shown to the agent has to be the time they actually waited. This tool's
    predecessor failed on latency, so a number that quietly excludes upload and
    decode would be worse than showing no number at all.

    The body is read with `api.batch.read_multipart` rather than FastAPI's
    `File()`/`Form()` parameters. Those go through Starlette's form parser,
    which writes any uploaded file over 1 MB to a temporary file on disk —
    and "nothing touches disk" is a claim this service makes. The in-memory
    reader also stops reading as soon as the upload passes the size limit,
    instead of accepting the whole body first.
    """
    started = time.perf_counter()

    # Request validation precedes the service-state check, deliberately.
    #
    # A text file uploaded to an image endpoint is a client error whether or
    # not the OCR engine happens to be loaded, and answering it with 503 tells
    # the caller to retry later when retrying will never help. Validating first
    # also means a malformed request costs nothing beyond parsing.
    try:
        form = await read_multipart(
            request,
            max_files=1,
            counted_field="image",
            max_total_bytes=MAX_UPLOAD_BYTES + _FORM_OVERHEAD_BYTES,
            per_file_bytes=MAX_UPLOAD_BYTES,
            total_limit_message=_too_large_message(),
        )
    except Rejected as exc:
        return exc.response()

    payload = _read_upload(form)

    # Empty strings count as absent, as they did with FastAPI's Form(): a
    # browser submits an untouched optional input as "".
    values = {
        name: form.fields[name][0]
        for name in _RECORD_FIELDS
        if form.fields.get(name) and form.fields[name][0] != ""
    }
    try:
        record = ApplicationRecord(**values)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=[
                {**error, "loc": ("body", *error["loc"])}
                for error in exc.errors(include_url=False, include_context=False)
            ],
        ) from exc

    if state.provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Extraction is unavailable: the OCR engine did not start. See the server log.",
        )

    # Preprocessing, OCR and verification are blocking, CPU-bound work of about
    # two seconds. Run directly in this `async def` they would freeze the event
    # loop for that whole time: every other request — health checks, other
    # agents' uploads, batch status polls — would wait behind this one. They run
    # on the request threadpool instead, and the loop stays free.
    #
    # The pipeline itself is `api.jobs.run_pipeline`, the same function every
    # batch item runs, so the two paths cannot drift apart.
    try:
        return await run_in_threadpool(
            run_pipeline,
            payload,
            record,
            provider=state.provider,
            ruleset=state.ruleset,
            diagnostics=state.diagnostics_enabled,
            started_at=started,
            extract_lock=_INTERACTIVE_EXTRACT_LOCK,
        )
    except ExtractionError as exc:
        # A failed extraction is a pipeline error, never a compliance finding.
        # Returning a result full of REVIEW verdicts would let an infrastructure
        # problem masquerade as a judgement about the label.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read this image: {exc}",
        ) from exc
    except UnsupportedBeverageClass as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


#: Serialises calls into the shared interactive OCR engine. RapidOCR is not
#: safe to call from two threads at once (its text detector stores per-image
#: state on the instance between two statements), and while this work ran on
#: the event loop that serialisation was implicit. Moving it to the threadpool
#: must not quietly remove it. Only `extract` is held: decoding, preprocessing
#: and verification of concurrent requests still overlap. Batch work never
#: touches this engine or this lock — see api/jobs.py.
_INTERACTIVE_EXTRACT_LOCK = threading.Lock()


#: Room for the record fields and multipart framing on top of the image.
_FORM_OVERHEAD_BYTES = 256 * 1024


def _too_large_message() -> str:
    return f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."


def _read_upload(form: Form) -> bytes:
    """The uploaded image, with the type, emptiness and size checks."""
    images = [f for f in form.files if f.field_name == "image"]
    if not images:
        raise HTTPException(
            status_code=422,
            detail=[{"type": "missing", "loc": ["body", "image"], "msg": "Field required"}],
        )
    image = images[0]
    if image.content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image type {image.content_type!r}. "
                f"Accepted: {', '.join(sorted(ACCEPTED_CONTENT_TYPES))}. "
                "Vector artwork (PDF) is not handled by this prototype."
            ),
        )
    if image.too_large or image.data is None:
        raise HTTPException(status_code=413, detail=_too_large_message())
    if not image.data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return bytes(image.data)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

# Registered before the static mount so the frontend can never shadow it. The
# access gate is attached here, once, so every batch route shares it.

configure_batch(
    state,
    accepted_content_types=ACCEPTED_CONTENT_TYPES,
    max_upload_bytes=MAX_UPLOAD_BYTES,
)
app.include_router(batch_router, dependencies=[Depends(require_access)])


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

_WEB_DIST = Path(os.environ.get("LABEL_VERIFY_WEB_DIST", "web/dist"))

if _WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_WEB_DIST / "index.html")
