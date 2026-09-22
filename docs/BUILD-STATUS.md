# Build status

What exists, what does not, and what is known to be wrong. Maintained because the
README describes the system as designed, and a reviewer needs the difference stated
plainly rather than inferred from which tests are missing.

---

## Complete and tested

| Component | Module | Tests |
|---|---|---|
| Geometry, spatial predicates | `extraction/geometry.py` | 18 |
| Dual-form text normalisation | `extraction/normalize.py` | 28 |
| Candidate span index + masking | `extraction/spans.py` | 21 |
| Preprocessing, rescaling, quality gate | `extraction/pipeline.py` | in fixture tests |
| Provider interface, stub, RapidOCR | `extraction/providers/` | 20 |
| CPU budget detection and pinning (container quota) | `extraction/cpu.py` | 19 |
| Security limits: decode bombs, upload framing, manifest size, headers | `tests/test_security.py` | 24 |
| Rule set schema + validation | `rules/schema.py`, `rules/ttb-v1.yaml` | 20 |
| Result types + safety invariants | `rules/results.py` | 18 |
| Brand name (anchored search) + residual-difference rule | `rules/comparators.py` | 25 |
| Alcohol content, net contents | `rules/numeric.py` | 51 |
| Class/type (anchored → positional → vocabulary) | `rules/class_type.py` | 93 |
| Government warning: location, capitalisation, word diff, separation | `rules/warning.py` | 109 |
| Engine: resolution order, masking, elevation | `rules/engine.py` | 24 |
| API, single-label | `api/main.py` | in engine and batch tests |
| Batch: manifest parsing and pairing | `api/manifest.py` | 40 |
| Batch: job runner, endpoints, CSV export | `api/jobs.py`, `api/batch.py` | 33 |
| Batch: worst-first ordering | `rules/triage.py` | 15 |
| Batch evaluation and load test | `tools/evaluate.py`, `tools/load_test.py` | 25 |
| Fixture generator (29 labels) + evaluator | `tools/`, `fixtures/` | 33 |
| Frontend: single label and batch, overlay, diff, a11y | `web/` | 95 (vitest) |

**693 Python tests + 95 frontend tests.** `make test` runs with no OCR engine,
no network and no model weights.

### Measured, end to end, through the real OCR engine

29 fixtures, 3 runs each, 2 vCPU:

- **100% correct classification on every field.**
- **0 false negatives of 16 violation fields.** No violation passed as compliant
  in any run, including the earliest and worst ones.
- **0 false flags of 55 fields** on compliant labels.
- **p50 1.95 s, p95 2.51 s** — roughly half the five-second budget spare.

The harness was mutation-tested to confirm it can fail: reintroducing either of
the two documented traps turns a fixture into a false negative and exits
non-zero. See `TRADEOFFS.md`, "Can the harness fail?".

### Batch, measured through the real API

300 labels on 2 vCPU: 17 min 46 s, 16.9 labels/minute, 0 failed. Single-label
p95 was 2.84 s idle and **3.24 s during the batch** (slowest 3.66 s), so the
five-second requirement holds while a batch runs. Largest status poll 123 kB;
server peak memory 1.39 GiB. Verified end to end through the real UI, including
stopping a batch part-way. Details: `TRADEOFFS.md`, trade-off 6.

**These numbers are from synthetic labels and should be read as optimistic.**
They include no real print, foil, embossing, curved surfaces or script lettering.

---

## Not built

- **Deskew, glare and blur detection** in `extraction/pipeline.py`. Thresholds
  need calibrating against real photographs; a detector tuned by intuition gives
  confident wrong guidance. Note that blur is currently invisible to the quality
  gate — `degraded_noise_blur` scores 0.773 with no issue flagged.
- **Pixel measurements for the warning** — bold and contrast. `StrategyContext`
  carries the quality summary but not the image. The recommended design is a
  separate typography step after text verification that receives the image and
  the resolved warning box.
- **Type-size check.** Needs `label_width_mm` and the parsed container volume
  passed to the warning step; currently always reports *couldn't check*.

---

## Unverified

- **Penetration testing and an accredited security assessment.** The review in
  `docs/SECURITY.md` was a code review plus automated scanning, not an attempt
  to break a running deployment, and nothing here has been through a
  government ATO process.
- **Real submitted labels.** Everything measured rests on synthetic fixtures.
  This remains the single most likely source of a materially worse number, and
  the first thing to do before a pilot.
- **Performance on the live host.** Measured instead on a simulation of Render
  Standard: the locked environment, the Dockerfile's start command, pinned to
  one CPU. 60-label batch plus interactive checks: single-label p95 3.92 s idle and
  3.43 s during the batch, 14.8 labels/minute, peak memory 1411 MB. **That
  simulation was optimistic.** It limited CPUs with `taskset`, which also hides
  the host's other cores; a real container sees every host core but may use one
  CPU's worth of time. The first check on the deployed instance took 16 s.
  Re-simulated with a real cgroup quota: the first check in a fresh process
  5.5 s, warm checks 4.3-5.2 s with the OCR engine's default threading, and
  3.3-3.7 s with one thread. Two fixes followed: the thread count now comes from
  the CPU quota (`extraction/cpu.py`), and the service runs two sample checks at
  startup so no user pays the first-inference cost. `/api/health` reports the
  host's CPU budget and those warm-up timings.
  **Confirmed on the live host:** Render reports a 1-CPU quota on a machine with
  16 visible cores (so the engine's default had been starting a thread per host
  core). With one thread, startup warm-up took 5.9 s for the first check and
  **2.8 s for a warm check**; a check through the live UI took 2.7 s. A
  29-label batch through the live UI finished in 1 min 37 s (3.3 s per label,
  about 18 labels/minute, which projects to roughly 17 minutes for 300).
  **Single-label checks during a batch failed on the live host: 6.0 s and
  6.5 s**, against 2.7 s idle — an even split of the one-CPU budget. Cause: the
  batch's lower priority (nice) only arbitrates between threads on the same
  core, and under a quota the process ran on any of 16 cores, so batch and
  interactive work ran side by side and shared the quota equally. Reproduced
  under a local one-CPU quota (p50 9.3 s during a batch against 4.4 s idle).
  Fix: the process is now confined to as many cores as its quota allows
  (`pin_to_quota` in `extraction/cpu.py`); re-measured locally, single-label
  p50 4.82 s during a batch against 4.85 s idle — the batch no longer slows
  interactive checks at all. (That machine is slower than the live host, whose
  idle checks take 2.7 s.) **Confirmed live after the fix:** single-label checks
  about 3 s on repeated tries while a 29-label batch ran; the batch took
  1 min 41 s.
- **Batch on a larger host.** Throughput with more than one worker, and the
  memory ceiling with real multi-megabyte phone photographs, are unmeasured.
- **Screen reader pass and measured contrast audit** of the frontend. Colours
  were chosen for WCAG AA but not measured with a tool.
- **Regulatory readings** behind three verdict choices: proof-only ABV under
  27 CFR 5.65, non-standard fill under 5.203, and whether centilitres are
  acceptable on a US label. Check against the current eCFR text.

---

## Known problems

- **Blur is not detected**, so the degraded-image guard cannot fire on a blurred
  photograph. The guard works (a fixture exercises it on low contrast), but its
  coverage is only as good as the quality gate's issue list.
- **The warning's single bounding box** over-masks when the statement starts
  mid-line: preceding words on that line are removed from later fields' pools.
- **Layout thresholds are reasoned, not measured.** The spacing tolerances in
  `rules/warning.py` were set by argument. They now live in the rule set, so
  retuning them produces a new rule-set version rather than silently changing
  what old verdicts meant.
- **The dev-only frontend mock** shows bold and contrast as measurements; the
  real backend reports them as *couldn't check*.

### Resolved since the last revision

- **Security review done and acted on.** Dependency scanning found 25 known
  vulnerabilities in five pinned packages (Pillow, python-multipart, Starlette,
  AnyIO, idna), all upgraded and re-validated against the real-OCR evaluation.
  An independent review found several ways a single request could exhaust
  memory or CPU — a 0.7 MB image declaring 13300x13300 pixels, unbounded
  multipart part headers, unbounded manifest rows — plus missing response
  headers and error replies carrying internal detail. All fixed, each with a
  test named after the attack. Full account: `docs/SECURITY.md`.

- **Deployed.** The Docker image builds and runs on Render Standard. The build's
  OCR check passed, and the live `/api/health` reports the RapidOCR provider
  loaded with no error. Before the first build, `constraints.txt` locked all 28
  transitive packages to the tested environment, and the full opencv build
  (which `rapidocr` pulls in by default) was shown to be absent.

- **Batch upload is built.** Manifest-driven, with pairing checked before any
  upload, a worst-first worklist from one shared ordering, isolated low-priority
  OCR so interactive checks stay under five seconds, a CSV export, and a Stop
  that keeps finished work and marks the rest "not checked".
- **The server no longer blocks while reading an image.** The single-label
  endpoint was `async` but ran OCR directly on the event loop, so every check
  froze the whole server for about two seconds — health checks included. Found
  while building batch, where it would have stalled progress polling.
- **Uploads no longer touch disk.** The multipart parser wrote any upload over
  1 MB to a temporary file, contradicting the "nothing is persisted" design. Both
  endpoints now read uploads into memory with a size limit.
- **The access token can travel as a header** (`X-Access-Token`), keeping it out
  of proxy logs.

- **Long-name false negative.** A one-character difference in a long brand
  name or class/type designation scored above the match threshold and resolved
  to `MATCH`, because similarity ratios forgive a wrong letter more as strings
  get longer. `MATCH` on name-like fields now also requires that no character
  difference survive normalisation (`tuning.names.max_residual_character_edits`),
  an absolute count that does not shrink with length. Three fixtures reproduce
  the bug and failed the harness before the fix; loosening the budget brings the
  false negatives back. Residual differences are reported as `REVIEW` with the
  characters named, never asserted.

- Verdict-affecting thresholds are no longer scattered through the strategy
  modules. Everything that decides whether the tool *asserts* a violation now
  lives in `rules/ttb-v1.yaml` under `defaults` and `tuning`, with a strict
  typed schema, and each value has a test proving it is read from the rule set
  rather than shadowed by a constant. Physical constants and search bounds stay
  in code, since neither can turn a `REVIEW` into a `MISMATCH`.
- The quality metric no longer measures ink coverage (see `TRADEOFFS.md`).
- Providers no longer fabricate a quality assessment. `ExtractionResult.quality`
  is optional and `ImageQuality.assessed` distinguishes "not measured" from
  "measured and good", so a missing measurement cannot read as a favourable one.

---

## Divergences from the design docs

- Multi-character OCR folding (`rn`→`m`) in `docs/field-assignment.md` is not
  implemented, deliberately; `extraction/normalize.py` documents why both
  attempts broke anchoring on the title-case header.
- The design doc's "downscale only" preprocessing was wrong and is documented as
  such in `extraction/pipeline.py`: detection needs the *opposite* on small
  images.
- Unchecked fields are marked with an explicit `FieldResult.automated = False`
  rather than a marker string.
