"""In-memory batch jobs: a queue, a bounded worker pool, and a store that forgets.

Batch exists for throughput and triage, and it has one constraint that
outranks both: **it must never slow down an agent doing an interactive
single-label check.** The predecessor tool failed at 30-40 seconds per label
and agents abandoned it; a batch of 300 labels that made everybody else's
check take ten seconds would repeat that failure for every agent on the
instance. Everything below is arranged around that.

How batch work is kept away from interactive work
-------------------------------------------------

*   **Not on the event loop.** A dispatcher thread owns the queue and a
    dedicated `ThreadPoolExecutor` runs the items. The request threadpool that
    serves `/api/verify` is never used for batch work, so a 300-label batch
    cannot occupy the threads interactive requests need.

*   **Bounded.** `LABEL_VERIFY_BATCH_WORKERS` threads (default 1) and one batch
    at a time; later batches wait in the QUEUED state. On a 2-vCPU instance a
    second batch worker buys little throughput and takes CPU straight from the
    interactive path.

*   **Its own OCR engine.** The shared RapidOCR engine is not safe to call from
    two threads at once: `TextDetector.__call__` stores the per-image
    preprocessing operator on the instance (`self.preprocess_op = ...`) and
    reads it back a line later, so a concurrent call with a differently sized
    image can swap the detector's input resolution under the other. Each batch
    worker therefore builds a private engine instead of sharing the interactive
    one.

*   **Cheaper per thread, and lower priority.** The batch engine is limited to
    `LABEL_VERIFY_BATCH_OCR_THREADS` ONNX intra-op threads (default 1) so it
    cannot fan out across every core, and each batch worker thread lowers its
    own scheduling priority (`LABEL_VERIFY_BATCH_NICE`, default 10). With one
    intra-op thread ONNX Runtime runs inference on the calling thread, so the
    niceness applies to the inference itself: when an interactive request
    arrives, the kernel gives it the CPU and the batch yields.

Measured on 2 vCPU with real OCR, median single-label latency while an 8-13
label batch ran (idle medians were 1.8-2.25 s across runs, n=5-8 each):

    batch nice  ORT threads   during batch   batch s/label
    0           2             4.17 s         3.8     <- no isolation
    0           1             3.28 s         4.0
    10          2             2.07-2.30 s    3.2-3.3
    10          1 (default)   2.29-2.33 s    4.1-4.5

Lowering priority is what protects the interactive path; the thread cap is the
fallback that still bounds the damage where niceness has no effect. Two ORT
threads bought ~20% batch throughput at no measured interactive cost and is a
reasonable setting where `nice` is known to work.

What is held, and for how long
------------------------------

Each image is kept as the exact bytes received — not re-encoded — so that a
batch verdict for a file is identical to a single-label check of the same file.
The bytes are dropped the moment that item finishes. Results stay in memory
until `LABEL_VERIFY_BATCH_TTL_SECONDS` (default two hours) after the batch
finishes or is cancelled, then the batch is discarded. Status polls carry
summaries only; a full result is served one item at a time. Nothing is ever written to disk. The
total of unprocessed image bytes held across all batches is capped by
`LABEL_VERIFY_BATCH_MAX_BYTES` (default 2 GiB), which bounds memory whatever
the queue looks like.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from api.batch_models import BatchItem, BatchState, BatchStatus, ItemState, PairingReport
from api.manifest import Pair
from extraction.providers.base import ExtractionError, ExtractionProvider
from rules.engine import verify
from rules.results import ApplicationRecord, Verdict, VerificationResult
from rules.schema import RuleSet, UnsupportedBeverageClass
from rules.triage import TriageSummary, sort_key, summarise

logger = logging.getLogger("label_verify.batch")

GIB = 1024**3


# ---------------------------------------------------------------------------
# The pipeline — one definition shared with single-label verification
# ---------------------------------------------------------------------------


def run_pipeline(
    payload: bytes,
    record: ApplicationRecord,
    *,
    provider: ExtractionProvider,
    ruleset: RuleSet,
    diagnostics: bool,
    started_at: float,
    extract_lock: threading.Lock | None = None,
) -> VerificationResult:
    """Preprocess, extract, attach quality, verify. Blocking; call off the event loop.

    Both `/api/verify` and every batch item run through this one function. That
    is the guarantee behind "a batch result for a file equals a single-label
    check of the same file": there is no second copy of the pipeline to drift.

    Args:
        payload: Image bytes exactly as uploaded.
        record: The application record to check against.
        provider: The OCR engine to use.
        ruleset: Rule set to verify under.
        diagnostics: Attach the verbose diagnostics record.
        started_at: `time.perf_counter()` from when the caller's clock started.
        extract_lock: Held around `provider.extract` when the provider is
            shared between threads and is not safe to call concurrently.

    Raises:
        ExtractionError: undecodable image, quality below the floor, or an
            engine failure. A pipeline error, never a compliance finding.
        UnsupportedBeverageClass: the record names a class the rule set lacks.
    """
    from extraction.pipeline import preprocess

    prepared = preprocess(payload)
    if extract_lock is None:
        extraction = provider.extract(prepared.image)
    else:
        with extract_lock:
            extraction = provider.extract(prepared.image)
    extraction = prepared.apply_quality(extraction)
    return verify(
        extraction, record, ruleset=ruleset, started_at=started_at, diagnostics=diagnostics
    )


# ---------------------------------------------------------------------------
# Batch OCR engine
# ---------------------------------------------------------------------------


def build_batch_rapidocr(intra_op_threads: int) -> ExtractionProvider:
    """A private RapidOCR provider with a capped ONNX thread count.

    Same model, same thresholds and the same line/word conversion as the
    single-label engine; only the number of ONNX threads differs. A different
    thread count could in principle reorder floating-point reductions;
    measured on nine fixtures, the one-thread engine returned byte-identical
    text, boxes and confidences to the default engine.
    """
    from extraction.providers.rapidocr_provider import RapidOcrProvider

    provider = RapidOcrProvider(
        engine_options={"intra_op_num_threads": intra_op_threads, "inter_op_num_threads": 1}
    )
    if not provider.is_available():
        raise ExtractionError(f"batch OCR engine could not start: {provider.unavailable_reason()}")
    return provider


def default_provider_factory(
    interactive: ExtractionProvider | None, intra_op_threads: int
) -> Callable[[], ExtractionProvider]:
    """Factory for per-worker batch providers, matched to the interactive engine."""

    def factory() -> ExtractionProvider:
        if interactive is None:
            raise ExtractionError("no extraction engine is available")
        if interactive.name == "rapidocr":
            return build_batch_rapidocr(intra_op_threads)
        # Any other engine: a fresh instance of the same kind, never the shared one.
        from extraction.providers import get_provider

        return get_provider(interactive.name)

    return factory


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("ignoring %s=%r: not an integer", name, raw)
        return default


@dataclass(frozen=True, slots=True)
class BatchSettings:
    """Operator-tunable limits, read once from the environment."""

    workers: int = 1
    ocr_threads: int = 1
    nice: int = 10
    max_bytes: int = 2 * GIB
    ttl: timedelta = timedelta(hours=2)
    #: How often an idle dispatcher wakes to discard expired batches. Without
    #: it, an expired batch would linger until the next request touched the
    #: store — "discarded after two hours" must not depend on traffic.
    sweep_interval_s: float = 60.0

    @classmethod
    def from_env(cls) -> BatchSettings:
        return cls(
            workers=_env_int("LABEL_VERIFY_BATCH_WORKERS", 1, minimum=1),
            ocr_threads=_env_int("LABEL_VERIFY_BATCH_OCR_THREADS", 1, minimum=1),
            nice=_env_int("LABEL_VERIFY_BATCH_NICE", 10, minimum=0),
            max_bytes=_env_int("LABEL_VERIFY_BATCH_MAX_BYTES", 2 * GIB, minimum=1),
            ttl=timedelta(seconds=_env_int("LABEL_VERIFY_BATCH_TTL_SECONDS", 7200, minimum=1)),
        )


# ---------------------------------------------------------------------------
# Batch state
# ---------------------------------------------------------------------------


class BatchNotFound(KeyError):
    """Unknown, expired or discarded batch id."""


class ItemNotFound(KeyError):
    """No item with that manifest row in this batch."""


class BatchCapacityError(RuntimeError):
    """Accepting this batch would exceed the memory budget for held images.

    `retryable` distinguishes "this batch alone is too big" (split it) from
    "other batches are holding the budget right now" (try again later).
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(slots=True)
class _Item:
    pair: Pair
    order: int
    image: bytes | None
    size: int
    state: ItemState = ItemState.PENDING
    result: VerificationResult | None = None
    error: str | None = None
    summary: TriageSummary | None = None


@dataclass(slots=True)
class _Batch:
    batch_id: str
    items: list[_Item]
    pairing: PairingReport
    created_at: datetime
    state: BatchState = BatchState.QUEUED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancelled: bool = False
    finished_count: int = 0
    failed_count: int = 0


def _lower_priority(nice: int) -> None:
    """Worker-thread initialiser: make this thread yield to interactive work.

    On Linux, `setpriority(PRIO_PROCESS, tid)` applies to a single thread, and
    threads it spawns inherit the value. Unprivileged processes may always
    lower their own priority. Failing to do so is logged, not fatal: batch
    still works, it just competes on equal terms.
    """
    if nice <= 0:
        return
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except (AttributeError, OSError) as exc:  # pragma: no cover - platform dependent
        logger.warning("could not lower batch worker priority: %s", exc)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BatchRunner:
    """Queue, worker pool and in-memory store for batches.

    Thread model: the event loop only ever takes `self._lock` briefly to submit,
    snapshot or cancel. A single dispatcher thread takes batches off the queue
    in arrival order and feeds their items to the worker pool, never more than
    `workers` at a time — which is what lets a cancel take effect on the next
    item rather than after everything already handed to the pool.
    """

    def __init__(
        self,
        *,
        ruleset: RuleSet,
        provider_factory: Callable[[], ExtractionProvider],
        diagnostics: bool = False,
        settings: BatchSettings | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.ruleset = ruleset
        self.settings = settings or BatchSettings()
        self._provider_factory = provider_factory
        self._diagnostics = diagnostics
        self._clock = clock or (lambda: datetime.now(UTC))

        self._lock = threading.Lock()
        self._batches: dict[str, _Batch] = {}
        self._queue: queue.Queue[_Batch] = queue.Queue()
        self._held_bytes = 0

        self._started = False
        self._executor: ThreadPoolExecutor | None = None
        self._local = threading.local()
        self._closed = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def _ensure_started(self) -> None:
        """Start threads on first use, so importing the app spawns nothing."""
        if self._started:
            return
        self._started = True
        nice = self.settings.nice
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.workers,
            thread_name_prefix="batch-worker",
            initializer=_lower_priority,
            initargs=(nice,),
        )
        threading.Thread(
            target=self._dispatch_forever, name="batch-dispatcher", daemon=True
        ).start()

    def close(self) -> None:
        """Stop the dispatcher and pool. Used by tests; the service runs until exit."""
        self._closed.set()
        with self._lock:
            for batch in self._batches.values():
                batch.cancelled = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    @property
    def held_bytes(self) -> int:
        with self._lock:
            return self._held_bytes

    def remaining_budget(self) -> int:
        with self._lock:
            return max(0, self.settings.max_bytes - self._held_bytes)

    # -- public operations (called from the event loop) ----------------------

    def submit(self, pairs: Sequence[Pair], images: Sequence[bytes], pairing: PairingReport) -> str:
        """Queue a batch of paired images. Returns its id.

        Args:
            pairs: Runnable pairs in upload order.
            images: Bytes for every uploaded file, indexed by `Pair.image_index`.
                Only paired images are retained.
            pairing: The report shown to the agent alongside the results.

        Raises:
            BatchCapacityError: when the images would exceed the held-bytes cap.
        """
        now = self._clock()
        items = [
            _Item(pair=p, order=i, image=images[p.image_index], size=len(images[p.image_index]))
            for i, p in enumerate(pairs)
        ]
        size = sum(item.size for item in items)
        cap = self.settings.max_bytes

        with self._lock:
            self._purge_expired_locked(now)
            if size > cap:
                raise BatchCapacityError(
                    f"These images total {_human_bytes(size)}, over the {_human_bytes(cap)} "
                    "limit for one batch. Split the submission into smaller batches.",
                    retryable=False,
                )
            if self._held_bytes + size > cap:
                raise BatchCapacityError(
                    "Other batches are still being processed and the service is holding as "
                    "many images as it can. Try again when they have finished.",
                    retryable=True,
                )
            batch_id = uuid.uuid4().hex
            batch = _Batch(batch_id=batch_id, items=items, pairing=pairing, created_at=now)
            self._batches[batch_id] = batch
            self._held_bytes += size

        self._ensure_started()
        self._queue.put(batch)
        return batch_id

    def status(self, batch_id: str, *, include_results: bool = False) -> BatchStatus:
        """The batch and its worklist.

        Summaries only by default. A poll arrives every couple of seconds for
        the length of a run; carrying every finished item's full result made it
        about 2 MB per poll at 300 labels. `include_results` is for the CSV
        export, which needs every verdict once.
        """
        with self._lock:
            return self._snapshot_locked(self._get_locked(batch_id), include_results)

    def item(self, batch_id: str, row_number: int) -> BatchItem:
        """One item with its full result — fetched when an agent opens it."""
        with self._lock:
            batch = self._get_locked(batch_id)
            for item in batch.items:
                if item.pair.row.row_number == row_number:
                    return _to_item(item, include_result=True)
            raise ItemNotFound(row_number)

    def cancel(self, batch_id: str) -> BatchStatus:
        """Stop a batch: no further item is started; everything done is kept.

        The state becomes CANCELLED at once and the batch then expires on the
        normal TTL, counted from the cancel, so the finished results and the
        CSV stay available exactly as long as a completed batch's would.

        An item already inside the OCR engine when the cancel arrives is
        allowed to finish **and its result is kept**. ONNX inference cannot be
        interrupted, so the work is done either way, and throwing a finished
        verdict away would make the record less complete for no saving: a RUNNING
        item goes on to DONE or ERROR. Every PENDING item becomes SKIPPED at
        once, in the same locked step, and its image is released — so no poll
        can ever show a stopped batch with work that looks as if it will run.

        Cancelling a batch that already finished changes nothing.
        """
        with self._lock:
            batch = self._get_locked(batch_id)
            if batch.state in (BatchState.QUEUED, BatchState.RUNNING):
                batch.cancelled = True
                batch.state = BatchState.CANCELLED
                batch.finished_at = self._clock()
                for item in batch.items:
                    if item.state is ItemState.PENDING:
                        item.state = ItemState.SKIPPED
                        self._release_locked(item)
            return self._snapshot_locked(batch, False)

    def retry_after_seconds(self) -> int:
        """A polite retry interval for a client told the service is full.

        Held bytes are released as the running batch works through its items,
        so its estimated remaining time is an upper bound on a sensible wait.
        Clamped to 30 s-10 min so a client neither hammers the service nor
        waits on a stale estimate.
        """
        with self._lock:
            now = self._clock()
            estimates = [
                self._estimate_remaining_ms_locked(b, now)
                for b in self._batches.values()
                if b.state is BatchState.RUNNING
            ]
        known = [e for e in estimates if e is not None]
        if not known:
            return 60
        return int(min(600, max(30, max(known) / 1000.0)))

    def _get_locked(self, batch_id: str) -> _Batch:
        self._purge_expired_locked(self._clock())
        batch = self._batches.get(batch_id)
        if batch is None:
            raise BatchNotFound(batch_id)
        return batch

    def purge_expired(self) -> None:
        with self._lock:
            self._purge_expired_locked(self._clock())

    # -- dispatcher and workers (own threads) --------------------------------

    def _dispatch_forever(self) -> None:
        while not self._closed.is_set():
            try:
                batch = self._queue.get(timeout=self.settings.sweep_interval_s)
            except queue.Empty:
                self.purge_expired()
                continue
            try:
                self._run_batch(batch)
            except Exception:  # noqa: BLE001 - the dispatcher must survive anything
                logger.exception("batch %s failed in the dispatcher", batch.batch_id)
            self.purge_expired()

    def _run_batch(self, batch: _Batch) -> None:
        with self._lock:
            if batch.cancelled:
                return
            batch.state = BatchState.RUNNING
            batch.started_at = self._clock()

        assert self._executor is not None
        slots = threading.Semaphore(self.settings.workers)
        futures: list[Future] = []
        for item in batch.items:
            slots.acquire()
            if batch.cancelled or self._closed.is_set():
                slots.release()
                break
            future = self._executor.submit(self._run_item, batch, item)
            future.add_done_callback(lambda _f: slots.release())
            futures.append(future)
        wait(futures)

        with self._lock:
            if not batch.cancelled:
                batch.state = BatchState.DONE
                batch.finished_at = self._clock()

    def _provider(self) -> ExtractionProvider:
        """This worker thread's private engine, built on first use."""
        provider = getattr(self._local, "provider", None)
        if provider is None:
            provider = self._provider_factory()
            self._local.provider = provider
        return provider

    def _run_item(self, batch: _Batch, item: _Item) -> None:
        with self._lock:
            payload = item.image
            # A cancel moves PENDING to SKIPPED under this same lock, so an
            # item handed to the pool just before a cancel never starts.
            if batch.cancelled or payload is None or item.state is not ItemState.PENDING:
                return
            item.state = ItemState.RUNNING

        result: VerificationResult | None = None
        error: str | None = None
        row = item.pair.row
        try:
            # Built (once per worker) before the clock starts, so the first
            # label's elapsed time is not inflated by loading model weights.
            provider = self._provider()
            started = time.perf_counter()
            record = ApplicationRecord(
                brand_name=row.brand_name,
                class_type=row.class_type,
                alcohol_content=row.alcohol_content,
                net_contents=row.net_contents,
                beverage_class=row.beverage_class,
                label_width_mm=row.label_width_mm,
            )
            result = run_pipeline(
                payload,
                record,
                provider=provider,
                ruleset=self.ruleset,
                diagnostics=self._diagnostics,
                started_at=started,
            )
        except ExtractionError as exc:
            # Same wording as the single-label endpoint's 422, so an agent sees
            # one message for one problem wherever they met it.
            error = f"Could not read this image: {exc}"
        except UnsupportedBeverageClass as exc:
            error = str(exc).strip("'\"")
        except Exception:  # noqa: BLE001 - one bad label must not stop the batch
            logger.exception("unexpected failure on batch item %s", item.pair.filename)
            error = "Something went wrong while checking this label. Try it on its own."
        finally:
            del payload

        summary = summarise(result, error)
        with self._lock:
            self._release_locked(item)
            # Kept even if the batch was cancelled while this item ran: the
            # work is done, and the record should say what it found.
            item.result, item.error, item.summary = result, error, summary
            item.state = ItemState.DONE if result is not None else ItemState.ERROR
            batch.finished_count += 1
            if result is None:
                batch.failed_count += 1

    # -- helpers (caller holds the lock) --------------------------------------

    def _release_locked(self, item: _Item) -> None:
        if item.image is not None:
            item.image = None
            self._held_bytes -= item.size

    def _expires_at_locked(self, batch: _Batch, now: datetime) -> datetime:
        if batch.finished_at is not None:
            return batch.finished_at + self.settings.ttl
        # Not finished yet: a projection, moving with the estimate. The
        # contract promises a time, and "two hours after it finishes" is only
        # a time once the finish can be estimated.
        remaining = self._estimate_remaining_ms_locked(batch, now) or 0.0
        return now + timedelta(milliseconds=remaining) + self.settings.ttl

    def _purge_expired_locked(self, now: datetime) -> None:
        expired = [
            batch_id
            for batch_id, batch in self._batches.items()
            if batch.finished_at is not None and now >= batch.finished_at + self.settings.ttl
        ]
        for batch_id in expired:
            batch = self._batches.pop(batch_id)
            for item in batch.items:
                self._release_locked(item)

    def _estimate_remaining_ms_locked(self, batch: _Batch, now: datetime) -> float | None:
        if batch.state in (BatchState.DONE, BatchState.CANCELLED):
            return 0.0
        if batch.started_at is None or batch.finished_count == 0:
            return None
        elapsed = (now - batch.started_at).total_seconds() * 1000.0
        per_label = elapsed / batch.finished_count
        return per_label * (len(batch.items) - batch.finished_count)

    def _snapshot_locked(self, batch: _Batch, include_results: bool) -> BatchStatus:
        now = self._clock()
        # Worklist order (the contract on `BatchStatus.items`): checked items
        # worst-first by the one triage ordering, then what is being read now,
        # then what will be read, then what never will be. Skipped labels go
        # last rather than into the triage order because they have no verdict
        # to rank: placing them among checked labels would claim a judgement
        # the tool never made. They are still counted in `needs_attention`.
        finished = sorted(
            (i for i in batch.items if i.summary is not None),
            key=lambda i: sort_key(i.summary, i.pair.filename),
        )
        by_state = {
            state: [i for i in batch.items if i.state is state]  # already in upload order
            for state in (ItemState.RUNNING, ItemState.PENDING, ItemState.SKIPPED)
        }
        ordered = (
            finished
            + by_state[ItemState.RUNNING]
            + by_state[ItemState.PENDING]
            + by_state[ItemState.SKIPPED]
        )

        end = batch.finished_at or now
        elapsed = (end - batch.started_at).total_seconds() * 1000.0 if batch.started_at else 0.0

        return BatchStatus(
            batch_id=batch.batch_id,
            state=batch.state,
            total=len(batch.items),
            completed=batch.finished_count,
            failed=batch.failed_count,
            needs_attention=sum(1 for i in finished if i.summary.needs_attention)
            + len(by_state[ItemState.SKIPPED]),
            created_at=batch.created_at,
            started_at=batch.started_at,
            finished_at=batch.finished_at,
            elapsed_ms=max(0.0, elapsed),
            estimated_remaining_ms=self._estimate_remaining_ms_locked(batch, now),
            expires_at=self._expires_at_locked(batch, now),
            pairing=batch.pairing,
            ruleset_version=self.ruleset.version,
            items=tuple(_to_item(i, include_result=include_results) for i in ordered),
        )


def _to_item(item: _Item, *, include_result: bool) -> BatchItem:
    """The contract view of one item.

    PENDING and RUNNING items carry no triage fields: nothing is known yet.

    A SKIPPED item has `needs_attention=True` and `tier`/`tier_label` None.
    It needs attention because nobody has checked it — the batch-level
    failure `FieldResult.automated` guards against for a single field. It has
    no tier because `rules.triage.Tier` ranks *outcomes*, and there is none;
    inventing one (say, UNREADABLE) would tell the agent something about the
    image that the tool never observed. Its place in the list, last, is set by
    `BatchRunner._snapshot_locked`, and its state says why.
    """
    row = item.pair.row
    base: dict[str, Any] = {
        "filename": item.pair.filename,
        "row_number": row.row_number,
        "brand_name": row.brand_name,
        "state": item.state,
    }
    summary = item.summary
    if item.state is ItemState.SKIPPED:
        return BatchItem(**base, needs_attention=True)
    if summary is None:
        return BatchItem(**base)
    return BatchItem(
        **base,
        tier=int(summary.tier),
        tier_label=summary.tier.label,
        needs_attention=summary.needs_attention,
        mismatch_count=summary.mismatch_count,
        elevated_count=summary.elevated_count,
        review_count=summary.review_count,
        unchecked_count=summary.unchecked_count,
        headline=summary.headline,
        result=item.result if include_result else None,
        error=item.error,
    )


def _human_bytes(size: int) -> str:
    if size >= GIB:
        return f"{size / GIB:.1f} GB"
    return f"{size / 1024**2:.0f} MB"


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

#: Leading characters that make a spreadsheet treat a cell as a formula (or,
#: for tab and carriage return, that some importers strip to reveal one).
#: Filenames and brand names come from the submitter, and agents open this
#: file in Excel; `=HYPERLINK(...)` in a brand name must arrive as text.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def neutralise_cell(value: object) -> str:
    """Render one CSV cell so that a spreadsheet can never evaluate it.

    Prefixing a single quote is the OWASP-recommended defence: Excel and
    LibreOffice display the rest of the cell as literal text.
    """
    text = "" if value is None else str(value)
    if text.startswith(_FORMULA_TRIGGERS):
        return "'" + text
    return text


#: Status words in the export where the enum value would be unclear to a reader.
_CSV_STATUS = {ItemState.SKIPPED: "not checked (batch stopped)"}


def results_csv(status: BatchStatus, ruleset: RuleSet) -> str:
    """One row per label, in worklist order, safe to open in a spreadsheet.

    `status` must carry full results (`BatchRunner.status(include_results=True)`).
    SKIPPED items — never reached because the batch was stopped — are written
    with status "not checked (batch stopped)", needs_attention "yes" and blank
    verdicts, never omitted: the export is the record of what was and was not
    done.

    Columns: filename, manifest row, brand name, status, tier label,
    needs_attention, one verdict column per rule-set field (in display order),
    headline, error. A field the tool does not check automatically reads
    "NOT CHECKED" rather than its placeholder REVIEW verdict: the export is the
    agent's record, and it must not claim a check that never ran.
    """
    fields = ruleset.display_order()
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(
        ["filename", "manifest_row", "brand_name", "status", "tier", "needs_attention"]
        + [f"{rule.label} verdict" for rule in fields]
        + ["headline", "error"]
    )
    for item in status.items:
        verdicts: list[str] = []
        by_id = {f.field_id: f for f in item.result.fields} if item.result else {}
        for rule in fields:
            result = by_id.get(rule.id)
            if result is None:
                verdicts.append("")
            elif not result.automated:
                verdicts.append("NOT CHECKED")
            else:
                verdicts.append(Verdict(result.verdict).value)
        needs = "" if item.needs_attention is None else ("yes" if item.needs_attention else "no")
        row = [
            item.filename,
            item.row_number,
            item.brand_name,
            _CSV_STATUS.get(item.state, item.state.value),
            item.tier_label or "",
            needs,
            *verdicts,
            item.headline or "",
            item.error or "",
        ]
        writer.writerow([neutralise_cell(cell) for cell in row])
    return buffer.getvalue()
