# TTB Label Verification Prototype

Verifies that the information printed on an alcohol beverage label matches the
corresponding application record, and that the mandatory health warning statement
meets the formatting requirements of 27 CFR Part 16.

Built as a standalone proof of concept. It does not integrate with COLA and is not
intended to.

> **Status.** All five field checks, the warning statement checks, batch upload, the
> fixture corpus, the evaluation harness and the frontend are built and tested. Measured
> through the real OCR engine on 29 synthetic labels: **100% correct classification on
> every field, 0 false negatives, 0 false flags.** Single-label checks take 2.2 s (p50)
> and stay under five seconds while a 300-label batch runs (p95 3.24 s). Figures come
> from synthetic labels and should be read as optimistic — see
> [TRADEOFFS.md](./TRADEOFFS.md#measured-performance). Not built: deskew and glare
> detection, and pixel-based bold/contrast measurement.
> [`docs/BUILD-STATUS.md`](./docs/BUILD-STATUS.md) states exactly what exists.

**Live demo:** https://ttb-label-verify-255v.onrender.com
**One-click link (code included):** https://ttb-label-verify-255v.onrender.com/?token=chxB2aAeXzyJK8%2FH2enlrUgkSSiuP1hsOnwDhgPdEdI%3D
**Access code:** `chxB2aAeXzyJK8/H2enlrUgkSSiuP1hsOnwDhgPdEdI=` — only needed if you open the plain URL; the app asks once
and the browser remembers it. It is a throwaway secret for this demo, rotated after review.
**Repository:** https://github.com/ajagawa/ttb-label-verify

> The demo host has one CPU and serves one reviewer at a time comfortably. The first
> request after a redeploy can take a few extra seconds while the OCR engine warms up.
**Sample data to try:** `fixtures/labels/` — includes both compliant and deliberately
non-compliant labels, with `fixtures/expected.json` recording what each one should produce.

---

## Contents

- [What it does](#what-it-does)
- [Requirements traceability](#requirements-traceability)
- [Approach](#approach)
- [Running it](#running-it)
- [Running air-gapped](#running-air-gapped)
- [Architecture](#architecture)
- [Rule sets](#rule-sets)
- [Testing and evaluation](#testing-and-evaluation)
- [Assumptions](#assumptions)
- [Build status — what is and is not implemented](./docs/BUILD-STATUS.md)
- [Trade-offs and limitations](./TRADEOFFS.md)
- [Bill of materials](./THIRD-PARTY-NOTICES.md)
- [Deployment](#deployment)

---

## What it does

Upload a label image and the application record. The tool returns, per field:

- what the application says
- what it found on the label
- a verdict — `MATCH`, `REVIEW`, or `MISMATCH`
- the region of the label the value was read from, highlighted on click
- for flagged fields, a plain-language reason and the governing CFR citation

Fields checked: brand name, class/type designation, alcohol content, net contents, and
the government health warning statement.

The warning statement additionally gets a word-level comparison against the text required
by § 16.21, a capitalization check on `GOVERNMENT WARNING`, and advisory measurements for
bold weight, background contrast, and separation from surrounding content.

**The tool recommends. The agent decides.** There is no auto-approve or auto-reject path.

---

## Requirements traceability

Most of the requirements for this project were stated in the discovery interviews rather
than in the technical requirements section. This table records what was extracted, who
raised it, and where it is addressed.

| Requirement | Raised by | How it is addressed | Where |
|---|---|---|---|
| Results in ~5 seconds; the prior vendor pilot failed at 30–40s | Sarah Chen | Measured p50/p95; elapsed time displayed on every result | [Performance](./TRADEOFFS.md#measured-performance) |
| Batch upload for 200–300 label submissions | Sarah Chen (via Janet, Seattle) | Manifest-driven batch with a pairing check *before* upload, a worst-first worklist, and a CSV export; runs at lower priority on its own OCR engine so it cannot slow single-label checks | `api/batch.py`, `api/jobs.py`, `rules/triage.py` |
| Usable across a wide range of tech comfort; half the team is over 50 | Sarah Chen | Single screen, no nested navigation, large targets, WCAG 2.1 AA, system font stack | `web/` |
| Outbound traffic to many domains is blocked | Marcus Williams | Zero external network calls at runtime; all inference in-process | [Air-gapped](#running-air-gapped) |
| Standalone PoC; no COLA integration | Marcus Williams | No COLA coupling anywhere in the codebase | — |
| Store nothing sensitive | Marcus Williams | Images held in memory, never written to disk; no database | [Trade-off 7](./TRADEOFFS.md#7-nothing-is-persisted) |
| Azure environment; FedRAMP already in place | Marcus Williams | Deployment path documented for TTB's existing tenant | [Deployment](#deployment) |
| Matching needs judgment (`STONE'S THROW` vs `Stone's Throw`) | Dave Morrison | Three-state verdicts; normalisation before fuzzy comparison so case and punctuation variance resolves to `MATCH` | `rules/comparators.py`, `extraction/normalize.py` |
| Don't make the agent's job harder | Dave Morrison | No auto-reject; every finding shows its evidence on the label; per-field override | `web/` |
| Warning text must be exact, `GOVERNMENT WARNING` in caps and bold | Jenny Park | Word-level diff against § 16.21; capitalisation reads raw text, never canonical, and distinguishes a systematic printing violation from sporadic OCR case noise | `rules/warning.py` — bold reported as *couldn't check* until pixel access is wired |
| Handle labels shot at an angle, with glare, or poorly lit | Jenny Park | EXIF orientation and rescaling implemented; quality gate reports *what* is wrong rather than rejecting generically, and a degraded image withdraws typographic assertions rather than flagging them as violations. Deskew and blur detection *not yet built* | `extraction/pipeline.py` |
| Current process is a printed per-field checklist | Jenny Park | Result fields presented in the same order as the checklist | `web/` |

---

## Approach

**Extraction and verification are separate stages, and only the first involves a model.**

1. **Extraction.** OCR produces text, word-level bounding boxes, and per-region
   confidence. A quality assessment runs first and can short-circuit with specific
   feedback if the image is unusable.
2. **Verification.** A deterministic rules engine compares extracted values against the
   application record. No model calls, no I/O, no network.

This split is the central design decision. Three things follow from it:

- **Auditability.** Identical inputs produce identical verdicts, each citing the rule it
  was evaluated against. "The model concluded it matched" is not a defensible finding in
  a compliance context.
- **Testability.** The rules engine is exercised exhaustively without inference, which is
  why `make test` runs in seconds.
- **Versioning.** TTB labeling requirements are under active rulemaking. Rules live in a
  versioned data file, and every result records the rule-set version it was evaluated
  under.

Extraction sits behind a provider interface so the engine can be swapped — including for
a cloud model, if the network boundary permits one. Reasoning on that choice is in
[Trade-off 1](./TRADEOFFS.md#1-local-inference-instead-of-a-cloud-vision-model).

---

## Running it

Requires Docker.

```bash
git clone https://github.com/ajagawa/ttb-label-verify.git
cd ttb-label-verify
docker compose up
```

Open `http://localhost:8080`. Nothing is downloaded at runtime: the OCR model weights
ship inside the pinned `rapidocr-onnxruntime` wheel and are baked into the image.

### Development

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn api.main:app --reload

# Frontend
cd web && npm install && npm run dev
```

---

## Running air-gapped

The deployed instance and the local container are the same image, and it makes no
outbound calls once the weights are present.

```bash
docker compose pull          # while connected
docker compose up
# disconnect the network entirely, then use the tool normally
```

`[Optional: link a short screen recording of exactly this.]`

This is the property that matters most for the deployment environment described during
discovery, and it is the reason extraction runs locally rather than against a hosted API.

---

## Architecture

```
label-verify/
├── api/
│   ├── main.py                  single-label route, upload validation, access gate
│   ├── batch.py                 batch endpoints
│   ├── batch_models.py          batch contract: shapes, states, error codes
│   ├── manifest.py              manifest parsing and image pairing (pure)
│   └── jobs.py                  in-memory job store and isolated batch runner
├── extraction/               probabilistic — OCR and image handling
│   ├── geometry.py              BBox and the spatial predicates
│   ├── normalize.py             dual-form text normalisation
│   ├── spans.py                 candidate span index + masking
│   ├── pipeline.py              preprocessing and the quality gate
│   └── providers/
│       ├── base.py              ExtractionProvider interface + shared types
│       ├── rapidocr_provider.py default — PP-OCR via ONNX Runtime
│       └── stub.py              deterministic replay, for tests and fixtures
├── rules/                    deterministic — no I/O, no model calls
│   ├── ttb-v1.yaml              versioned rule set
│   ├── schema.py                typed loader and validation
│   ├── results.py               verdict types + safety invariants
│   ├── comparators.py           shared scoring, residual differences, brand name
│   ├── numeric.py               alcohol content, net contents
│   ├── class_type.py            class/type designation
│   ├── warning.py               § 16.21 / § 16.22 warning checks
│   ├── engine.py                resolution order, masking, assembly
│   └── triage.py                worst-first ordering for batch worklists
├── tests/                    one module per source module
├── web/                      React + Vite frontend
├── fixtures/labels/          generated test corpus
├── tools/
└── docs/
    └── field-assignment.md      the heuristics this is built on
```

**Flow:** upload → decode → quality gate → extract → span index → resolve fields
(masking each resolved region) → verdicts.

The boundary that matters is `extraction/` → `rules/`: everything upstream is
probabilistic, everything downstream is deterministic, and they communicate through a
single typed structure, `ExtractionResult`, defined in `extraction/providers/base.py`.
A provider returns text and geometry and nothing else — no field assignments, no
judgements — so it cannot move a compliance decision into the non-reproducible half of
the system.

Two safety invariants are enforced in code rather than by convention, in
`rules/results.py`:

- A field that was not located can never carry a `MISMATCH`. Asserting a violation
  because extraction failed is the worst error this system can make.
- A `MISMATCH` must carry the region it was read from. An assertion the agent cannot
  visually verify is not actionable.

---

## Rule sets

`rules/ttb-v1.yaml` holds the field definitions, comparison strategies, thresholds, and
CFR citations. It is organized by beverage class. Adding wine or malt beverage support
requires no code changes.

**Every threshold that decides whether the tool asserts a violation lives here**, under
`defaults` and `tuning` — not in the strategy modules. Each result is stamped with the
rule set version, so a threshold sitting in code would change verdicts without changing
the version recorded against them, and an old decision could no longer be reproduced.
Each value has a test proving it is read from the rule set rather than shadowed by a
constant. Physical constants and layout geometry stay in code, since neither can turn a
`REVIEW` into a `MISMATCH`.

Every verification result carries the rule-set version. This is deliberate: pending TTB
rulemaking on "Alcohol Facts" and allergen disclosure (Notices 237 and 238) would add
mandatory fields, and a decision needs to be reproducible against the rules in force when
it was made.

Threshold selection and the sensitivity analysis behind it are in
[`docs/threshold-analysis.md`](./docs/threshold-analysis.md).

---

## Testing and evaluation

```bash
make test        # rules engine + assignment — no inference, no network, no weights
make evaluate    # rules engine vs ground-truth fixtures; fails on any false negative
make evaluate-ocr  # the same corpus through the real OCR pipeline
```

`make test` running with no OCR engine installed is a deliberate property, not a
convenience. It means a reviewer can clone and get a green suite immediately, and it
means a failure against a replayed fixture is unambiguously a bug in assignment or
comparison rather than a misread.

The fixture corpus is generated rather than sourced, so ground truth is exact and
non-compliant variants can be produced deliberately — including title-case warning text,
incorrect ABV, missing net contents, and degraded image conditions.

It is synthetic, and the accuracy figures should be read as optimistic. See
[The fixture corpus is synthetic](./TRADEOFFS.md#the-fixture-corpus-is-synthetic).

---

## Assumptions

Where the brief was silent, these are the gaps filled in and the reasoning:

1. **The application record is supplied with the image.** No COLA lookup, per the scope
   boundary set during discovery. In the demo it is entered in a form; the API accepts it
   as JSON.
2. **One label panel per verification.** Real submissions often include separate front,
   back, and neck images. See
   [Multi-panel labels](./TRADEOFFS.md#multi-panel-labels-are-not-handled) — this is the
   largest gap relative to production use.
3. **Distilled spirits.** The sample label provided is spirits, and one class implemented
   properly is more useful than three implemented shallowly.
4. **Raster images.** Photographs and scans. Vector artwork PDFs, which appear in real
   submissions, are not handled.
5. **English-language labels.** Imports with foreign-language text are out of scope.
6. **Container size is supplied when the type-size check is wanted.** Millimeters cannot
   be derived from pixels without a scale reference; the check reports *not evaluable*
   rather than guessing.
7. **Current regulations only.** Evaluated against 27 CFR Part 16 as in force, not
   against pending proposed rules.
8. **Authentication happens upstream in production.** The prototype is protected by a
   shared secret only.

---

## Deployment

The demo instance runs on **Render**, Standard plan (1 CPU, 2 GB), as a single Docker
container built from this repository's `Dockerfile` and configured by `render.yaml`.
Render terminates TLS and routes to the port it names in `$PORT`. No database, no
external service dependencies, nothing written to disk. Standard is the smallest plan
that fits: the OCR engine peaks near 1.4 GB during a batch. `docker-compose.yml` runs
the same image anywhere else, with any TLS-terminating proxy in front.

### Settings

| Variable | Default | Purpose |
|---|---|---|
| `LABEL_VERIFY_ACCESS_TOKEN` | unset (open) | Shared secret for the demo instance. Clients send it as the `X-Access-Token` header; a `?token=` query parameter also works for demo links, but lands in proxy logs. |
| `LABEL_VERIFY_BATCH_WORKERS` | 1 | Batch worker threads. Raise on larger hosts. |
| `LABEL_VERIFY_BATCH_OCR_THREADS` | 1 | ONNX threads per batch OCR engine. |
| `LABEL_VERIFY_BATCH_NICE` | 10 | Scheduling priority of batch work. This is what keeps single-label checks fast while a batch runs — see TRADEOFFS.md, trade-off 6. |
| `LABEL_VERIFY_BATCH_MAX_BYTES` | 2 GiB | Unprocessed image bytes held across all batches. Bounds memory. |
| `LABEL_VERIFY_BATCH_TTL_SECONDS` | 7200 | How long finished batch results are kept in memory. Nothing is ever written to disk. |
| `LABEL_VERIFY_DIAGNOSTICS` | 0 | Attach the verbose per-verification diagnostic record. |
| `LABEL_VERIFY_OCR_THREADS` | CPUs allowed by the container's quota | ONNX threads for single-label checks. Detected from the cgroup CPU quota, not the host's core count — see `extraction/cpu.py`. |
| `LABEL_VERIFY_WARMUP` | 1 in the image | Run two sample checks at startup, so the first real check is not the slow one. `/api/health` reports their timings (`warmup_ms`) — the deployed host's real speed. |

For a production path inside TTB's environment: the same container runs in Azure
Container Apps within the existing tenant, which keeps it inside the authorization
boundary already established and avoids the outbound-traffic problem that affected the
prior vendor pilot. If a stronger extraction engine is wanted, Azure OpenAI and Document
Intelligence deployed in-tenant would likely outperform local OCR on stylized typography
while staying inside the same boundary.

What a production deployment would additionally require is listed in
[What production would require](./TRADEOFFS.md#what-production-would-require).

---

## License

MIT — see [`LICENSE`](./LICENSE).
Third-party components and their terms: [`THIRD-PARTY-NOTICES.md`](./THIRD-PARTY-NOTICES.md).
