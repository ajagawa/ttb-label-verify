# Security review

A review of this prototype, run after it was deployed: dependency scanning, static
analysis, and a manual read of the attack surface by a reviewer who had not written
the code. This records what was checked, what was found, what was fixed, and what is
knowingly accepted for a demo. Every fix has a test in `tests/test_security.py` named
after the attack it closes.

Re-run everything with `make audit` (dependency CVEs, Python static analysis, npm
audit), plus `make test`.

## What this prototype handles

Label artwork and the application values an agent types in. No personal data, no
credentials beyond one shared demo access code, nothing written to disk, no database,
no outbound network calls. The consequence of a breach here is a denial of service or
someone else's label image — not a data loss. The fixes below are still worth making:
the point of the prototype is to show how the real thing would be built.

## Found and fixed

### Dependencies: 25 known vulnerabilities

`pip-audit` against the lock file reported 25 advisories across five packages. The
lock had been written to pin the exact tested versions, which is right for
reproducibility and wrong when a pinned version is later found vulnerable — nothing
was re-checking. Upgraded to the first fixed release of each, re-ran the full suite
and the real-OCR evaluation, and confirmed no accuracy change:

| Package | Was | Now | Why it mattered here |
|---|---|---|---|
| Pillow | 12.2.0 | 12.3.0 | Twelve advisories in image parsing, including heap corruption reachable from decoding an uploaded image. |
| python-multipart | 0.0.26 | 0.0.31 | Denial of service in multipart parsing — unbounded part headers, and a `Content-Length` that could turn a bounded read into read-until-EOF. Exactly the upload path this service exposes. |
| Starlette | 1.0.0 | 1.3.1 | Host-header injection into reconstructed URLs; form limits silently ignored for urlencoded bodies. |
| AnyIO | 4.13.0 | 4.14.2 | TLS hostname handling; a process-pool pipe that is never drained. |
| idna | 3.13 | 3.15 | Quadratic parsing of crafted internationalised names. |

The npm dependencies reported no vulnerabilities, production or development.

### A 0.7 MB upload could use 1.4 GB of memory

A PNG's header can declare any dimensions. Pillow *warns* rather than refuses between
89 and 179 megapixels, and nothing turned that warning into an error, so a 713 KB file
declaring 13300×13300 decoded to about 1.4 GB — twice, concurrently, was more than the
2 GB instance had. Now (`extraction/pipeline.py`):

- the size is read from the header and refused before any pixel is decoded, above
  **40 megapixels** (any ordinary phone photograph is 12–24);
- Pillow's own bomb *warning* is raised as an error;
- large JPEGs are decoded at reduced scale by the decoder itself (`draft`), so a
  48-megapixel phone shot is still accepted — everything is resized to 2600 px anyway;
- at most **two images are decoded at once** across single checks and batch workers,
  so the worst case does not grow with the number of requests.

### The upload parser counted only the data

Part *headers* were unbounded and uncounted: a single 20 MB header was accepted and
took about ten seconds of CPU — on the event loop, so every other request waited,
health checks included, which is how a container gets restarted mid-batch. The number
of parts and fields was unbounded too. Now (`api/batch.py`): every byte received counts
towards the limit, part headers are capped (4 KB, 8 headers), the number of parts is
capped, `Content-Length` is checked before reading, file parts under unexpected field
names are discarded as they stream, parsing runs on a worker thread, and a malformed
body is a plain 400 rather than a server error.

### Unbounded in-memory batches

`MAX_BATCH_SIZE` limited images but nothing limited manifest *rows*, and a batch was
kept for two hours with its full pairing report. A 2 MB manifest of 100,000 junk rows
produced about 85 MB of retained issues and a 19 MB reply to every status poll; about
twenty submissions filled the instance. Now: manifests are capped at
`MAX_MANIFEST_RECORDS` rows and 1000 characters per cell (`api/manifest.py`), and the
runner holds at most `LABEL_VERIFY_BATCH_MAX_BATCHES` batches (default 20), dropping
the oldest **finished** batch to make room and refusing a new one only when every held
batch is still queued or running (`api/jobs.py`).

### Concurrent uploads could race past the memory budget

The batch memory budget was read before a body was streamed but only charged when the
batch was queued, so several uploads could each pass the check and together buffer
several times the budget. Now one batch upload is received at a time, and at most eight
single-label checks are in flight; beyond that the server answers 503 with `Retry-After`
straight away rather than holding the bytes.

### Any image format Pillow knows was decoded

The type check read only the client's declared `Content-Type`; the decoder then sniffed
the real format and would happily decode GIF, TGA, PCX, SGI, DDS and about thirty
others — a large parser surface, historically full of CVEs, that no label photograph
needs. Decoding is now restricted to **JPEG, PNG, WebP, TIFF and BMP**, matched against
the file's actual content.

### A malformed manifest returned a server error

A quoted cell over the CSV module's field limit raised an uncaught error and returned
500. It is now a plain 400 explaining that the file could not be read as CSV.

### Error replies carried internal detail

An undecodable image returned Pillow's own message, which includes a heap address; OCR
failures returned the engine's internals; the unauthenticated health endpoint published
the startup error. All now return fixed, user-facing text, with the detail logged
server-side.

### Missing response headers

No CSP, no `nosniff`, no framing or referrer policy, and API responses were cacheable.
The page could be framed (clickjacking "Stop batch"), and a page URL carrying `?token=`
would travel in `Referer` headers. All are set now (`api/main.py`), with `no-store` on
API responses.

### Access-code handling

- Compared with `hmac.compare_digest`, not `!=`: the timing of a plain comparison leaks
  how much of a guess was right.
- The one-click link's `?token=` is removed from the address bar as soon as it is read,
  so the code does not persist in history, bookmarks or copied links.
- The interactive API docs (`/docs`, `/redoc`, `/openapi.json`) are off unless
  `LABEL_VERIFY_API_DOCS=1`: they needed no code, mapped every endpoint, and loaded
  scripts from a public CDN — outbound traffic this service otherwise never causes.

### Spreadsheet formula injection, tightened

Export cells were already prefixed when they began with `= + - @ tab CR`. Now a trigger
after leading whitespace or a line break is caught too. Not reachable from today's
fields, which are stripped, but cheap.

## Checked and found sound

- **Path traversal** in static files: `/assets/../`, `%2e%2e` and `..%2f` all 404. The
  only `FileResponse` is a fixed path; there is no catch-all route.
- **`Content-Disposition` injection**: download filenames are constants or a hex batch
  id. Upload filenames never reach the filesystem or a header.
- **Batch ids** are 122 random bits, so one batch cannot be guessed from another.
- **CORS** is off unless `LABEL_VERIFY_DEV_CORS=1`, and then only for the Vite dev
  server.
- **Frontend XSS**: no `dangerouslySetInnerHTML`, no `innerHTML`, no `eval`; the access
  code travels only in a header, including uploads and CSV downloads; no redirects.
- **Unsafe deserialisation or execution**: `yaml.safe_load`; no pickle, `eval`, `exec`
  or `subprocess`; dynamic imports are limited to fixed internal module names.
- **Regular-expression denial of service**: the numeric patterns are quadratic on long
  whitespace runs in isolation, but input is whitespace-collapsed first; measured flat
  at 32,000 characters.
- **Container**: non-root (uid 10001), no secrets in the image, diagnostics off by
  default, compose file sets read-only root, `no-new-privileges` and a tmpfs.
- **Static analysis**: `bandit` reported seven findings, all false positives — internal
  `assert`s, a verdict value named `"pass"`, a local function named `extra` mistaken for
  Django's ORM, and a URL fetch in a development load-test tool (which now refuses any
  scheme but http/https, so a mistyped `--url` cannot read a local file). The first two
  rules are skipped in `pyproject.toml` with reasons, so `make audit` is clean and a
  *new* finding fails it.

## Accepted for a demo, not for production

- **One shared access code, published in the README** so reviewers can open the app. It
  gates the API but identifies nobody: anyone with it can read or cancel any batch whose
  id they have. Production replaces it with the agency's own sign-in and per-user
  ownership checks.
- **No rate limiting.** A determined caller can still occupy the single CPU with
  legitimate-looking requests. The limits above bound memory, not fairness; a real
  deployment puts a rate limiter in the gateway in front.
- **Base images are not pinned by digest** (`python:3.11-slim`, `node:22-slim`), so a
  rebuild can pick up a different base. Pinning needs a registry, which the build
  environment here cannot reach.
- **The first request from a one-click link carries `?token=`** and lands in the
  platform's HTTP logs. Subsequent requests use the header.
- **No supply-chain verification of wheels** beyond version pinning: hashes
  (`--require-hashes`) would be the next step.

## Re-running the checks

```bash
make audit   # pip-audit against the lock, bandit, npm audit
make test    # 693 Python tests, including tests/test_security.py
```

`make audit` needs network access to the advisory databases. It is not part of `make
test`, which runs offline by design.
