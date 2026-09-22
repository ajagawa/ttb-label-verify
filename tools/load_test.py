"""Load test: single-label latency while a batch is running.

Run against a live server (real API, real OCR):

    python3 -m uvicorn api.main:app --port 8091 &
    python3 -m tools.load_test --url http://127.0.0.1:8091 --server-pid $!

What it measures, and why that and not something else
------------------------------------------------------
Batch is throughput-bound: nobody watches a progress bar for 300 labels. The
single-label check is latency-bound: an agent is at the counter waiting, and
the previous vendor pilot failed on exactly that. Batch shares the server with
single-label checks, so the risk batch introduces is not that *batch* is slow —
it is that batch makes *everyone else* slow. The number that decides whether
batch is safe to ship is therefore single-label p95 **during** a running batch,
against the five-second requirement, with the idle figure beside it for scale.

Phases
------
1.  **Idle baseline.** K sequential ``POST /api/verify`` requests.
2.  **Under load.** Submit N labels via ``POST /api/batch``, then, while the
    batch reports RUNNING, fire the same K single-label requests at a steady
    interval. A background thread polls ``GET /api/batch/{id}`` throughout, so
    each single-label request is tagged with the batch state it overlapped.
    Only requests sent while the batch was RUNNING count as "during batch"; if
    too few overlapped (a small batch that finishes quickly), the result is
    reported INCONCLUSIVE rather than as a pass it did not earn.

The batch is built in a temporary directory by repeating the fixture images
under unique filenames, with a manifest generated from `fixtures/expected.json`
through the same row builder as `fixtures/batch_manifest.csv`. Nothing is
written to `fixtures/`.

Standard library only (urllib, threading, csv): a load test that needs its
own dependency set is one more thing that will not be installed on the box it
has to run on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tools.evaluate import percentile

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "fixtures"

#: The single-label requirement from the discovery notes: a result in about
#: five seconds, or agents stop using the tool.
LATENCY_REQUIREMENT_S = 5.0

#: Fewer overlapping requests than this and a p95 is not meaningful. Below it
#: the verdict is INCONCLUSIVE, never PASS.
MIN_OVERLAPPING_REQUESTS = 3

CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class LoadTestError(RuntimeError):
    """A condition that stops the run, with a message meant for a person."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelFile:
    """One image to upload and the application record it is checked against."""

    path: Path
    upload_name: str
    record: dict[str, str]


def load_corpus(fixtures_dir: Path) -> list[LabelFile]:
    """Every fixture image with its record, under its own name."""
    expected = json.loads((fixtures_dir / "expected.json").read_text(encoding="utf-8"))
    return [
        LabelFile(
            path=fixtures_dir / entry["image"],
            upload_name=Path(entry["image"]).name,
            record=dict(entry["record"]),
        )
        for entry in expected["fixtures"]
    ]


def build_batch(corpus: list[LabelFile], size: int, dest: Path) -> tuple[Path, list[LabelFile]]:
    """Materialise a batch of `size` labels in `dest`, with its manifest.

    Images are the corpus repeated in order, each copy under a unique name
    (``b0001_old_tom_compliant.png``) so pairing is one-to-one — duplicate
    names would be reported as ambiguous and left out, and the batch would
    silently be smaller than asked for.

    Rows use `tools.generate_fixtures.manifest_row`, the builder behind
    `fixtures/batch_manifest.csv`, so the columns track the API contract.

    Returns:
        The manifest path and the files, in manifest order.
    """
    from tools.generate_fixtures import manifest_columns, manifest_row

    if size < 1:
        raise ValueError("a batch needs at least one label")
    if not corpus:
        raise ValueError("the fixture corpus is empty; run `make fixtures`")

    dest.mkdir(parents=True, exist_ok=True)
    files: list[LabelFile] = []
    rows: list[dict[str, str]] = []
    for index in range(size):
        source = corpus[index % len(corpus)]
        name = f"b{index + 1:04d}_{source.upload_name}"
        target = dest / name
        target.write_bytes(source.path.read_bytes())
        files.append(LabelFile(path=target, upload_name=name, record=source.record))
        row = manifest_row({"image": name, "record": source.record})
        rows.append(row)

    manifest = dest / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_columns()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest, files


def encode_multipart(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, str, bytes]],
    boundary: str | None = None,
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body.

    Args:
        fields: ``(name, value)`` text fields.
        files: ``(field name, filename, content type, bytes)``.
        boundary: Fixed boundary for tests; random otherwise.

    Returns:
        The body and the Content-Type header value.
    """
    boundary = boundary or f"loadtest-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode("utf-8")
            + b"\r\n"
        )
    for name, filename, content_type, data in files:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def latency_stats(seconds: list[float]) -> dict:
    """n, p50, p95 and max of a latency sample, in seconds (nearest rank)."""
    if not seconds:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {
        "n": len(seconds),
        "p50": percentile(seconds, 50),
        "p95": percentile(seconds, 95),
        "max": max(seconds),
    }


def read_peak_rss_kb(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Peak resident set size (VmHWM) of a process, in kB, or None.

    VmHWM is the kernel's high-water mark, so one read at the end captures the
    peak over the whole run with no sampling thread and no dependency.
    """
    try:
        text = (proc_root / str(pid) / "status").read_text()
    except OSError:
        return None

    # A wrong PID does not fail — it returns a plausible small number. The first
    # 300-label run reported a 5 MB peak because the PID passed in belonged to
    # the shell that launched the server, not the server. Refuse anything that
    # is not a Python process running uvicorn rather than report its memory.
    # Judge the process by its executable and arguments, not by whether its
    # command line *mentions* python and uvicorn: the shell that launched the
    # server has a command line containing both words as text, and a substring
    # check waved exactly that process through. argv[0] must be a Python
    # interpreter and a later argument must be uvicorn.
    try:
        argv = [
            a.decode(errors="replace")
            for a in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
            if a
        ]
    except OSError:
        argv = []
    if argv:
        is_python = Path(argv[0]).name.startswith("python")
        runs_uvicorn = any(Path(a).name.startswith("uvicorn") for a in argv[1:])
        if not (is_python and runs_uvicorn):
            raise LoadTestError(
                f"--server-pid {pid} is not the API server (it is {' '.join(argv)[:80]!r}). "
                "Pass the PID of the python process running uvicorn."
            )
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def assess(during: dict) -> tuple[str, str]:
    """PASS / FAIL / INCONCLUSIVE against the five-second requirement."""
    if during["n"] < MIN_OVERLAPPING_REQUESTS:
        return (
            "INCONCLUSIVE",
            f"only {during['n']} single-label request(s) overlapped the running batch; "
            f"need at least {MIN_OVERLAPPING_REQUESTS}. Use a larger --batch-size.",
        )
    if during["p95"] < LATENCY_REQUIREMENT_S:
        return (
            "PASS",
            f"single-label p95 during batch {during['p95']:.2f} s < {LATENCY_REQUIREMENT_S:.0f} s",
        )
    return (
        "FAIL",
        f"single-label p95 during batch {during['p95']:.2f} s >= {LATENCY_REQUIREMENT_S:.0f} s",
    )


def _fmt(value: float | None, unit: str = " s") -> str:
    return "—" if value is None else f"{value:.2f}{unit}"


def render_markdown(report: dict) -> str:
    """The load-test report, in the shape of the TRADEOFFS.md table."""
    idle, during = report["idle"], report["during_batch"]
    batch = report["batch"]
    rss = report.get("server_peak_rss_mb")
    out = [
        f"### Batch load test — {report['url']}",
        "",
        f"**{report['verdict']}** — {report['verdict_reason']}.",
        "",
        "| Metric | Idle | During batch |",
        "|---|---|---|",
        f"| Single-label requests | {idle['n']} | {during['n']} |",
        f"| Single-label p50 | {_fmt(idle['p50'])} | {_fmt(during['p50'])} |",
        f"| Single-label p95 | {_fmt(idle['p95'])} | {_fmt(during['p95'])} |",
        f"| Single-label max | {_fmt(idle['max'])} | {_fmt(during['max'])} |",
        "",
        "| Batch | Value |",
        "|---|---|",
        f"| Labels | {batch['size']} ({batch['completed']} completed, {batch['failed']} failed) |",
        f"| Final state | {batch['state']} |",
        f"| Upload (submit request) | {_fmt(batch['submit_s'])} |",
        f"| Wall time (started → finished) | {_fmt(batch['wall_s'])} |",
        f"| Throughput | {_fmt(batch['labels_per_minute'], ' labels/min')} |",
        f"| Per-label verification p50 (server-side) | {_fmt(batch['per_label_p50_s'])} |",
        f"| Status polls | {batch['polls']} (mean {_fmt(batch['poll_mean_s'])}, "
        f"largest response {batch['poll_max_kb']:.0f} kB) |",
        f"| Server peak RSS (VmHWM) | {'not measured (no --server-pid)' if rss is None else f'{rss:.0f} MB'} |",
    ]
    if during.get("after_batch"):
        out += [
            "",
            f"{during['after_batch']} single-label request(s) were sent after the batch finished "
            "and are excluded from the during-batch column.",
        ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass
class Client:
    """Thin urllib wrapper that adds the access token and times each call."""

    base: str
    token: str | None = None
    timeout: float = 120.0

    def url(self, path: str) -> str:
        # The token travels as the X-Access-Token header, never in the URL:
        # query strings end up in server and proxy access logs.
        target = self.base.rstrip("/") + path
        # urlopen honours file:// and other schemes; this tool only ever talks
        # to an HTTP server, and saying so keeps a mistyped --url from reading
        # a local file.
        if not target.startswith(("http://", "https://")):
            raise SystemExit(f"--url must be an http:// or https:// address, not {self.base!r}")
        return target

    def request(
        self, method: str, path: str, body: bytes | None = None, content_type: str | None = None
    ) -> tuple[int, bytes, float]:
        headers = {"Content-Type": content_type} if content_type else {}
        if self.token:
            headers["X-Access-Token"] = self.token
        req = urllib.request.Request(self.url(path), data=body, method=method, headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - scheme checked in url()
                payload = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            payload, status = exc.read(), exc.code
        return status, payload, time.perf_counter() - started


def _detail(payload: bytes) -> str:
    try:
        return str(json.loads(payload).get("detail"))[:300]
    except (ValueError, AttributeError):
        return payload[:300].decode("utf-8", "replace")


def check_server(client: Client) -> None:
    """Fail early and legibly if the server is not there."""
    try:
        status, payload, _ = client.request("GET", "/api/health")
    except (urllib.error.URLError, OSError) as exc:
        raise LoadTestError(
            f"server not reachable at {client.base}: {exc}. Start it with "
            "`python3 -m uvicorn api.main:app --port 8091` and pass --url."
        ) from exc
    if status != 200:
        raise LoadTestError(f"GET /api/health returned {status}: {_detail(payload)}")


def verify_once(client: Client, label: LabelFile) -> float:
    """One single-label verification; returns wall-clock seconds."""
    body, ctype = encode_multipart(
        list(label.record.items()),
        [("image", label.upload_name, _content_type(label.path), label.path.read_bytes())],
    )
    status, payload, elapsed = client.request("POST", "/api/verify", body, ctype)
    if status == 401:
        raise LoadTestError("the server requires an access token; pass --token")
    if status != 200:
        raise LoadTestError(f"POST /api/verify returned {status}: {_detail(payload)}")
    return elapsed


def _content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


@dataclass
class Poller:
    """Polls batch status in the background and remembers the latest state."""

    client: Client
    batch_id: str
    interval: float
    state: str = "queued"
    last: dict = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    max_bytes: int = 0
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                status, payload, elapsed = self.client.request("GET", f"/api/batch/{self.batch_id}")
            except (urllib.error.URLError, OSError) as exc:
                self.error = str(exc)
                return
            if status != 200:
                self.error = f"GET /api/batch/{self.batch_id} returned {status}"
                return
            self.latencies.append(elapsed)
            self.max_bytes = max(self.max_bytes, len(payload))
            self.last = json.loads(payload)
            self.state = self.last.get("state", self.state)
            if self.state in ("done", "cancelled"):
                return
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def submit_batch(client: Client, manifest: Path, files: list[LabelFile]) -> tuple[dict, float]:
    body, ctype = encode_multipart(
        [],
        [("manifest", manifest.name, "text/csv", manifest.read_bytes())]
        + [("images", f.upload_name, _content_type(f.path), f.path.read_bytes()) for f in files],
    )
    status, payload, elapsed = client.request("POST", "/api/batch", body, ctype)
    if status in (404, 405):
        raise LoadTestError(
            f"POST /api/batch returned {status}: the batch endpoints do not exist on this "
            "server yet (api/batch.py not deployed or not mounted)."
        )
    if status == 401:
        raise LoadTestError("the server requires an access token; pass --token")
    if status != 202:
        raise LoadTestError(f"POST /api/batch returned {status}: {_detail(payload)}")
    return json.loads(payload), elapsed


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    client = Client(args.url, args.token)
    check_server(client)
    corpus = load_corpus(args.fixtures)
    rss_before = read_peak_rss_kb(args.server_pid) if args.server_pid else None

    # Phase 1: idle baseline.
    idle = [verify_once(client, corpus[i % len(corpus)]) for i in range(args.single_requests)]

    with tempfile.TemporaryDirectory(prefix="label-verify-load-") as tmp:
        manifest, files = build_batch(corpus, args.batch_size, Path(tmp))
        submitted, submit_s = submit_batch(client, manifest, files)

    poller = Poller(client, submitted["batch_id"], args.poll_interval)
    poller.state = submitted.get("state", "queued")
    thread = threading.Thread(target=poller.run, daemon=True)
    thread.start()

    # Wait (briefly) for RUNNING, so the first request really overlaps work.
    deadline = time.monotonic() + 60
    while poller.state == "queued" and time.monotonic() < deadline and poller.error is None:
        time.sleep(0.1)

    # Phase 2: the same single-label requests, at a steady interval.
    during: list[float] = []
    after = 0
    for i in range(args.single_requests):
        slot = time.monotonic() + args.interval
        overlapping = poller.state == "running"
        elapsed = verify_once(client, corpus[i % len(corpus)])
        if overlapping:
            during.append(elapsed)
        else:
            after += 1
        time.sleep(max(0.0, slot - time.monotonic()))

    thread.join(timeout=args.batch_timeout)
    poller.stop()
    if poller.error:
        raise LoadTestError(f"polling the batch failed: {poller.error}")
    final = poller.last or submitted
    if final.get("state") not in ("done", "cancelled"):
        raise LoadTestError(
            f"batch {submitted['batch_id']} did not finish within {args.batch_timeout:.0f} s "
            f"(state {final.get('state')}, {final.get('completed')}/{final.get('total')} done)"
        )

    started, finished = _parse_time(final.get("started_at")), _parse_time(final.get("finished_at"))
    wall_s = (finished - started).total_seconds() if started and finished else None
    # Status polls carry summaries only (see api/batch_models.py), so each
    # finished item's timing is fetched from the item endpoint, once, after the
    # batch has finished — never during the measured window.
    per_label: list[float] = []
    for item in final.get("items", []):
        if item.get("state") != "done":
            continue
        status, payload, _ = client.request(
            "GET", f"/api/batch/{submitted['batch_id']}/items/{item['row_number']}"
        )
        if status != 200:
            continue
        result = json.loads(payload).get("result") or {}
        if result.get("elapsed_ms") is not None:
            per_label.append(result["elapsed_ms"] / 1000.0)
    rss_after = read_peak_rss_kb(args.server_pid) if args.server_pid else None

    during_stats = {**latency_stats(during), "after_batch": after}
    verdict, reason = assess(during_stats)
    return {
        "url": args.url,
        "verdict": verdict,
        "verdict_reason": reason,
        "requirement_s": LATENCY_REQUIREMENT_S,
        "idle": latency_stats(idle),
        "during_batch": during_stats,
        "batch": {
            "id": submitted["batch_id"],
            "size": args.batch_size,
            "state": final.get("state"),
            "completed": final.get("completed"),
            "failed": final.get("failed"),
            "submit_s": submit_s,
            "wall_s": wall_s,
            "labels_per_minute": (60.0 * final.get("completed", 0) / wall_s) if wall_s else None,
            "per_label_p50_s": percentile(per_label, 50) if per_label else None,
            "polls": len(poller.latencies),
            "poll_mean_s": (sum(poller.latencies) / len(poller.latencies))
            if poller.latencies
            else None,
            "poll_max_kb": poller.max_bytes / 1024.0,
        },
        "server_peak_rss_mb": rss_after / 1024.0 if rss_after else None,
        "server_peak_rss_before_mb": rss_before / 1024.0 if rss_before else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8091", help="server base URL")
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--single-requests", type=int, default=20)
    parser.add_argument(
        "--interval", type=float, default=3.0, help="seconds between single-label requests"
    )
    parser.add_argument("--token", default=os.environ.get("LABEL_VERIFY_ACCESS_TOKEN"))
    parser.add_argument("--server-pid", type=int, default=None, help="read VmHWM for peak RSS")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--batch-timeout", type=float, default=3600.0)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args(argv)

    try:
        report = run(args)
    except LoadTestError as exc:
        print(f"load_test: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2) if args.json else render_markdown(report))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
