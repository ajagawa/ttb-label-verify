"""Batch API contract.

The shapes the batch endpoints accept and return. Fixed before the backend,
frontend and evaluation were built in parallel, so that all three agree.

Endpoints (all behind the same access gate as single-label verification):

    GET    /api/batch/template.csv          manifest template with the brief's example row
    POST   /api/batch/check                 manifest + filenames -> PairingReport  (no images)
    POST   /api/batch                       manifest + images    -> BatchStatus    (202)
    GET    /api/batch/{batch_id}            -> BatchStatus, items worst-first, WITHOUT results
    GET    /api/batch/{batch_id}/items/{row_number}
                                            -> BatchItem WITH its full result
    GET    /api/batch/{batch_id}/results.csv
    DELETE /api/batch/{batch_id}            cancel: stop scheduling, KEEP finished results

Form fields: `manifest` (file), `images` (repeated files), `filenames` (repeated
strings, check endpoint only).

Access token: `X-Access-Token` header (preferred — a query string lands in
server and proxy logs) or the `token` query parameter; the header wins when
both are sent.

503 responses carry a machine-readable code, so a client never has to tell the
cases apart by message text:

    {"detail": {"code": "<code>", "message": "<plain language for the agent>"}}

    extraction_unavailable   the OCR engine is not loaded. Retrying will not
                             help; an operator must fix the deployment.
    batch_capacity           other batches hold the in-memory image budget.
                             Sent with a `Retry-After` header (seconds, 30-600,
                             from the running batch's estimate); retry then.
    batch_unavailable        batch processing is not configured (wiring fault).

Other errors keep FastAPI's `{"detail": "<message>"}`: 400 malformed request,
404 unknown or expired batch or row, 413 over a count or size limit (not
retryable: split the batch), 422 nothing could be paired
(`{"detail": {"message": ..., "pairing": PairingReport}}`).

Two decisions revised after the first build, both from measurement or use:

*   **Status polls carry summaries, not results.** The first version returned
    every finished item's full `VerificationResult` on every poll — about 2 MB
    per poll at 300 labels, fetched every two seconds for ten minutes or more
    over a government network. A poll now carries only what the worklist
    renders (tier, counts, headline); `result` is None there, and one item's
    full result is fetched when an agent opens it.

*   **Stopping keeps the work already done.** The first version discarded a
    batch on DELETE, so an agent who stopped a run lost the record of every
    label already checked. DELETE now cancels: remaining items are never
    started, finished items and the CSV stay available, `state` becomes
    CANCELLED and is visible to GET, and the batch is discarded at its normal
    expiry like any other.

The check endpoint exists so pairing problems surface *before* anything is
uploaded. Three hundred phone photographs can be a gigabyte over a government
network; discovering a filename typo after that upload, or ten minutes into
processing, is the failure it prevents. It takes filenames, not images.

Nothing here is persisted. A batch lives in memory until it expires, then is
discarded with its images — the same "store nothing" boundary as single-label
verification. The CSV export is how an agent keeps a record.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rules.results import VerificationResult

#: Manifest columns, in template order. The first five are required.
MANIFEST_REQUIRED_COLUMNS = (
    "filename",
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
)
MANIFEST_OPTIONAL_COLUMNS = ("beverage_class", "label_width_mm")
MANIFEST_COLUMNS = MANIFEST_REQUIRED_COLUMNS + MANIFEST_OPTIONAL_COLUMNS

#: Largest batch accepted. The discovery notes describe importers submitting
#: 200-300 labels at once; the ceiling leaves headroom without letting one
#: request occupy the service indefinitely.
MAX_BATCH_SIZE = 500


class ManifestRow(BaseModel):
    """One label's application record, as read from the manifest."""

    model_config = ConfigDict(frozen=True)

    row_number: int = Field(ge=2, description="Spreadsheet row, counting the header as row 1")
    filename: str
    brand_name: str
    class_type: str
    alcohol_content: str
    net_contents: str
    beverage_class: str = "distilled_spirits"
    label_width_mm: float | None = Field(default=None, gt=0)


class PairingIssueKind(enum.StrEnum):
    IMAGE_WITHOUT_ROW = "image_without_row"
    ROW_WITHOUT_IMAGE = "row_without_image"
    DUPLICATE_ROW = "duplicate_row"
    DUPLICATE_IMAGE = "duplicate_image"
    INVALID_ROW = "invalid_row"
    MISSING_COLUMN = "missing_column"
    UNSUPPORTED_FILE = "unsupported_file"


class PairingIssue(BaseModel):
    """One problem pairing images with manifest rows, in plain language."""

    model_config = ConfigDict(frozen=True)

    kind: PairingIssueKind
    filename: str | None = None
    row_number: int | None = None
    #: Written for the agent: "Row 14 names old_tom.jpg, but no image with
    #: that name was selected."
    message: str


class PairingReport(BaseModel):
    """Whether a manifest and a set of files line up, before anything runs.

    `blocking` is True only when nothing could be checked at all (no pairs, or
    a missing required column). Otherwise issues are reported and the matched
    pairs can proceed: one mistyped filename should not hold up 299 labels.
    """

    model_config = ConfigDict(frozen=True)

    matched: int
    total_images: int
    total_rows: int
    issues: tuple[PairingIssue, ...] = ()
    blocking: bool

    @property
    def ok(self) -> bool:
        return not self.blocking and not self.issues


class BatchState(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"


class ItemState(enum.StrEnum):
    """Where one label is in its life.

    RUNNING and SKIPPED were added after the first build, when both the backend
    and frontend agents independently hit the same gap: without them, a label
    being read and a label that would never be read were both PENDING, told
    apart only by headline text and extra polling. A record that cannot say
    "this was never checked" is the batch-level version of the failure the
    single-label result guards against with `FieldResult.automated` — the tool
    must never let "not looked at" read like "waiting" or like "fine".
    """

    PENDING = "pending"  # not started yet; will run
    RUNNING = "running"  # being read now
    DONE = "done"  # verified; `result` available from the item endpoint
    ERROR = "error"  # could not be processed; `error` says why
    SKIPPED = "skipped"  # never checked, because the batch was stopped first


class BatchItem(BaseModel):
    """One label in a batch.

    The triage fields are denormalised from `rules.triage.summarise` so the
    worklist can render without unpacking every full result. `result` carries
    the full verification for the detail view.
    """

    model_config = ConfigDict(frozen=True)

    filename: str
    row_number: int
    brand_name: str
    state: ItemState

    #: `rules.triage.Tier` as an int (0 most urgent) and its plain label.
    #: None while pending.
    tier: int | None = None
    tier_label: str | None = None
    needs_attention: bool | None = None
    mismatch_count: int = 0
    elevated_count: int = 0
    review_count: int = 0
    unchecked_count: int = 0
    headline: str | None = None

    #: The full verification. Present only from the single-item endpoint; always
    #: None in a status poll. See the module docstring.
    result: VerificationResult | None = None
    error: str | None = None


class BatchStatus(BaseModel):
    """A batch and its worklist.

    `items` are ordered:

    1.  DONE and ERROR items, worst-first by `rules.triage.sort_key`;
    2.  then RUNNING items, in upload order;
    3.  then PENDING items, in upload order;
    4.  then SKIPPED items, in upload order.

    So the list an agent sees mid-run is already the right way up, and settles
    rather than reshuffles as it fills. SKIPPED items come last, outside the
    triage order, because they have no outcome to rank; they carry
    `needs_attention=True` and `tier` None.
    """

    model_config = ConfigDict(frozen=True)

    batch_id: str
    state: BatchState
    total: int
    #: Items that finished, DONE or ERROR. Excludes SKIPPED.
    completed: int
    #: The ERROR subset of `completed`.
    failed: int
    #: Finished items whose triage says they need a person, PLUS every SKIPPED
    #: item: a label nobody checked needs attention, and an agent must see that
    #: 40 labels were never looked at rather than a reassuringly small number.
    needs_attention: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: float = 0.0
    #: Straight-line estimate from the per-label rate so far. None until at
    #: least one label has finished — an estimate from nothing is a guess.
    estimated_remaining_ms: float | None = None
    #: When the batch and its images will be discarded.
    expires_at: datetime
    pairing: PairingReport
    ruleset_version: str
    items: tuple[BatchItem, ...] = ()
