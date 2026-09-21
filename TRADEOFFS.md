# Trade-offs and Limitations

This prototype makes several deliberate choices that constrain what it can do. This
section documents those choices, the alternatives that were considered, and the things
the tool cannot do regardless of implementation effort.

Every number here comes from the fixture evaluation run — see
[Measured performance](#measured-performance). Nothing is estimated.

---

## What this tool does, and what it does not

**It verifies consistency and formatting.** Given a label image and an application
record, it checks whether the values on the label match the values in the application,
and whether the mandatory health warning meets the formatting requirements of 27 CFR
§ 16.22.

**It does not verify truthfulness.** Whether the stated alcohol content is actually
correct, whether the bottler's name and address correspond to a valid TTB permit,
whether the class/type designation matches the production record, whether the net
contents are accurate — none of these are determinable from a label image, and none are
attempted here. Those checks require fill testing, permit database access, and formula
records.

**It never rejects an application.** Every result is a recommendation with supporting
evidence. The agent decides. There is no configuration that enables auto-rejection, and
adding one would be a mistake: the tool's value is triage, not adjudication.

---

## Deliberate trade-offs

### 1. Local inference instead of a cloud vision model

**Chosen:** all extraction runs in-process with no outbound network calls.

**Why:** the discovery sessions noted that TTB's network blocks outbound traffic to many
domains, and that roughly half the prior scanning vendor's features failed for exactly
this reason. A prototype that cannot run inside the network boundary cannot inform a
procurement decision about a system that has to.

**Cost:** a cloud vision-language model would extract stylized typography more reliably
and would assign fields semantically rather than heuristically. Accuracy on decorative
brand lettering is measurably lower here — see [Known limitations](#known-limitations).

**Mitigation:** the task is verification, not open-ended extraction. The expected value
is already known from the application record, so fuzzy matching against a known target
tolerates OCR error that would be unacceptable in a transcription context. Where
extraction confidence is genuinely low, the tool returns `REVIEW` rather than guessing.

**Alternative retained:** extraction sits behind a provider interface. A cloud provider
can be added without touching the verification engine. If TTB's boundary can accommodate
Azure OpenAI and Document Intelligence deployed inside its existing tenant, that path
inherits the FedRAMP authorization already in place and would likely outperform this one.

### 2. AI extracts; deterministic rules verify

**Chosen:** the model produces text and coordinates. Every compliance judgment is made
by deterministic code against a versioned rule set.

**Why:** a federal compliance decision needs to be reproducible and explainable. "The
model concluded it matched" is not an auditable finding. Identical inputs produce
identical verdicts, and every verdict cites the rule it was evaluated against.

**Cost:** the verification layer has no semantic understanding. It cannot recognize that
"Ky. Straight Bourbon" and "Kentucky Straight Bourbon Whiskey" describe the same class
unless that equivalence is encoded in the rule set. Class/type is the weakest field as a
direct result.

**Not a trade-off:** this also makes the rules engine testable without inference, which
is why the test suite runs in seconds against the full fixture corpus with zero model
calls.

### 3. Three-state results instead of pass/fail

**Chosen:** every field resolves to `MATCH`, `REVIEW`, or `MISMATCH`.

**Why:** one of the agents interviewed gave a specific example — `STONE'S THROW` on the
label against `Stone's Throw` in the application — and observed that a system without
judgment would flag it as a mismatch when it plainly is not one. Binary output forces
every ambiguity into one of two wrong answers.

**Cost:** the tool does not fully automate anything. An agent still reviews every
`REVIEW`. The time saving comes from collapsing the routine cases, not eliminating
review.

### 4. Tuned to over-flag

**Chosen:** thresholds favor false positives over false negatives.

**Why:** the costs are asymmetric. A missed violation is a compliance failure that
reaches the market. A false flag costs an agent roughly twenty seconds to dismiss. These
are not comparable, and the thresholds reflect that.

**Cost:** on the synthetic corpus the conservative tuning currently costs nothing
measurable — 0 of 55 fields on compliant labels resolve to `REVIEW` when they
could have matched. That figure should not be trusted as a prediction. These
labels are cleanly rendered; real print, foil and script lettering will push
recognition confidence down, and the thresholds are deliberately set so that
uncertainty becomes a flag. Agents will dismiss flags that turn out to be fine,
and the interface should say so, or a deliberately conservative tool will be
read as a noisy one.

**A failure mode of the fuzzy comparator, found and fixed.** Similarity is a
ratio, so a single wrong letter costs less the longer the name. Measured before
the fix:

| Application | Label | Score | Outcome at 0.92, before |
|---|---|---|---|
| `MARROW CREEK` | `HARROW CREEK` | 0.833 | flagged |
| `OLD TOM` | `OLD TON` | 0.857 | flagged |
| `OLD TOM DISTILLERY` | `OLD TON DISTILLERY` | 0.944 | **matched** |
| `Kentucky Straight Bourbon Whiskey` | `...Burbon...` | 0.970 | **matched** |

The same defect was caught or missed depending only on length — a false
negative, and the corpus could not see it because every one-letter fixture in it
was short. No single threshold fixes this; tightening it would flag the case
and punctuation variants trade-off #3 exists to absorb.

The fix adds a second condition for `MATCH` on name-like fields: after
normalisation removes everything that carries no meaning (case, punctuation,
accents, spacing, and the OCR confusion classes such as 0/O and 1/I/L), the
**count** of surviving character differences must not exceed
`tuning.names.max_residual_character_edits` (0). A count does not shrink with
length the way a ratio does. Any residual difference becomes `REVIEW` with the
characters named — "“M” in the application reads “N” on the label" — and is
never asserted as `MISMATCH`, because a misread and a misprint look identical
from the text.

Three fixtures were added **before** the fix and failed the harness on the
unfixed code; loosening the budget back to 1 reproduces all three false
negatives. After the fix, false negatives are 0 of 16 in both modes and false
flags stay at 0 of 55.

**The honest cost.** This is stricter than the previous rule, and the stated
thesis — that mediocre OCR is enough — now reads more precisely: mediocre OCR is
enough to *locate* a value and rule out large differences, but not to certify a
residual one-character difference away. On real photographs, some genuine
misreads outside the fold table (PP-OCR once read `Tequila` as `Teguila` at a
lower resolution) will reach an agent as `REVIEW`. That frequency is not yet
measured — OCR is deterministic for a given image, so repeated runs over the
synthetic corpus cannot sample it — and is part of what validation against real
labels needs to establish. The policy direction is deliberate: a false flag
costs a glance, and a label with a misspelled brand name is exactly what an
agent should see.

Every threshold that decides whether the tool **asserts** a violation lives in the
versioned rule set rather than in code, under `defaults` and `tuning`. That is not
tidiness: each result is stamped with the rule set version, so a constant sitting
in a module would change verdicts without changing the version recorded against
them, and an old decision could no longer be reproduced. Physical constants and
layout geometry stay in code, since neither can turn a `REVIEW` into a
`MISMATCH`.

### 5. Distilled spirits only

**Chosen:** one beverage class implemented thoroughly rather than three superficially.

**Why:** the sample label provided is distilled spirits. Class/type vocabularies,
alcohol content tolerances, and standards of fill differ materially across beer, wine,
and spirits, and a shallow implementation of all three would be less useful and less
demonstrative than a complete implementation of one.

**Cost:** the tool will produce incorrect results on wine and malt beverage labels. It
does not detect that it has been given one and does not warn.

**Extensibility:** wine and malt beverage rule sets are additive — `rules/ttb-v1.yaml`
is structured by beverage class, and adding one requires no code changes. This is the
same mechanism that handles the pending rulemaking described in the notices file.

### 6. Batch processing is asynchronous, not sub-five-seconds

**Chosen:** the five-second target applies to interactive single-label review. Batch
submissions queue and return a triage-ordered worklist.

**Why:** these are different problems. Interactive review is latency-bound — an agent is
waiting, and the prior vendor pilot failed at 30–40 seconds per label because agents
could work faster by eye. A 300-label import submission is throughput-bound, and nobody
is watching the progress bar. Optimizing batch for per-label latency would waste effort
on a constraint that does not apply.

**What this produces:** a worst-first worklist. Labels the tool found differing
from the application come first, then a missing mandatory statement, then images
that could not be read, then labels needing a look, then fields not checked
automatically, then all-clear. The order is defined once, in `rules/triage.py`,
and the API, the frontend and the evaluator all use that one definition, so the
order the harness measures is the order an agent sees.

**Measured** — 300 labels through the real API and real OCR on a 2-vCPU host,
with 36 single-label checks fired at 30-second intervals across the whole run:

| | Idle | During the 300-label batch |
|---|---|---|
| Single-label p50 | 2.20 s | 2.57 s |
| Single-label p95 | 2.84 s | **3.24 s** |
| Single-label slowest | 3.00 s | 3.66 s |

| Batch | |
|---|---|
| 300 labels | 17 min 46 s, 0 failed |
| Throughput | 16.9 labels / minute |
| Per label, server-side (p50) | 3.44 s |
| Largest status poll | 123 kB |
| Server peak memory | 1.39 GiB |

The five-second single-label requirement holds throughout a full batch, with the
slowest interactive check at 3.66 s.

**Why batch is slower per label than a single check, deliberately.** A batch
item takes about 3.4 s against 2.2 s for an interactive one, because batch runs
on its own OCR engine at lower scheduling priority (`LABEL_VERIFY_BATCH_NICE`,
default 10) with one ONNX thread. That is the cost of protecting interactive
checks, and it was measured, not assumed: with batch at normal priority and two
threads, single-label median rose to 4.17 s with a worst case above five seconds
— the vendor-pilot failure, reproduced. An earlier estimate in this document put
300 labels at "roughly 10 minutes"; the measured 17.8 minutes is slower because
of exactly this trade, on a small host. It scales with `LABEL_VERIFY_BATCH_WORKERS`
on a larger one.

**Why a separate engine at all.** RapidOCR is not safe to call from two threads:
its text detector stores a per-image setting on the shared object and reads it
back a line later. Batch workers therefore build their own engine (about 43 MB
each) rather than sharing the interactive one.

**Pairing is checked before anything uploads.** The manifest and the list of
filenames go to the server first, without any images, so a mistyped filename is
found before a large upload rather than after it or ten minutes into processing.

**Verified end to end through the real UI** — 29 fixtures, real OCR: the
"Needs attention" list held exactly the 16 non-compliant labels and "All clear"
the 13 others; stopping a batch part-way kept the label being read at that
moment, and reported every remaining label as "Not checked — the batch was
stopped" rather than as waiting or as clear.

### 7. Nothing is persisted

**Chosen:** images are held in memory, never written to disk, and discarded after the
response. No database.

**Why:** the discovery notes specified that nothing sensitive should be stored for this
exercise. Making ephemeral processing structural rather than a policy setting means
there is no retention question to answer.

Batch follows the same rule: images are held in memory only until their label is
checked, and results are discarded two hours after a batch finishes. The CSV export
is how an agent keeps a record. Single-label uploads are read straight into memory
too — the multipart parser's default would have written any upload over 1 MB to a
temporary file, which contradicted this section until it was found and fixed.

**Cost:** this is a real functional limitation, not just a simplification. There is no
result history, no cross-session audit trail, and — most significantly — no capture of
agent overrides. Override data is the only way to measure whether the tool's judgments
are actually correct over time, and it is the input any future accuracy improvement
would need. A production system requires it, along with a records retention schedule.

---

## Known limitations

### Extraction

| Condition | Behavior | Severity |
|---|---|---|
| Script, decorative, or heavily stylized brand lettering | Recognition degrades; often resolves `REVIEW` | High — common on spirits labels |
| Text curved around a cylindrical container | Detection degrades or fragments the line | Medium |
| Metallic, foil, or embossed labels with low ink contrast | May fail to detect text regions | Medium |
| Severe perspective distortion (beyond ~30°) | Deskew fails; quality gate rejects with specific feedback | Low — handled explicitly |
| Specular glare over a text region | Region-level failure; quality gate reports the affected area | Low — handled explicitly |
| Non-English text on imported labels | Not supported | Medium — affects imports |
| Handwritten text | Not supported | Low — not expected on labels |

Per-field accuracy against the fixture corpus is in [Measured performance](#measured-performance).

### Bold detection is approximate

27 CFR § 16.22(a)(2) requires that `GOVERNMENT WARNING` appear in capital letters and
bold type, and that the remainder of the statement not appear in bold. Capitalization is
determined reliably from recognized text. Boldness is estimated by comparing stroke
width against the surrounding statement body, normalized for glyph height.

This is a measurement, not a certainty. It is reported with a confidence value and
labeled *advisory*; it never produces a `MISMATCH` on its own. A label where the warning
is set in a naturally heavy typeface throughout, or where the body text is condensed,
can produce an ambiguous reading. The agent sees the measurement and the highlighted
region and decides.

### Multi-panel labels are not handled

Real COLA submissions frequently include separate front, back, and neck label images.
The mandatory warning commonly appears on the back label while the brand name and
class/type appear on the front. This prototype accepts one image per verification and
will report the warning as absent if it is not on the submitted panel.

This is the most significant functional gap relative to real-world use. It is a scope
decision rather than a technical obstacle — the rules engine already evaluates fields
independently, so accepting a set of images and unioning the extracted fields across
them is a contained change. It was cut for time, not because it is hard.

### The fixture corpus is synthetic

Accuracy figures below are measured against programmatically generated labels. This
gives exact ground truth and permits deliberate non-compliant variants, but it does not
reproduce the real distribution of label designs, print quality, or photographic
conditions. **The reported numbers should be read as optimistic.** Validation against a
sample of real submitted labels would be the first step before any pilot.

---

## Not verifiable from a label image

These are properties of the problem, not gaps in the implementation. No amount of
engineering effort resolves them without additional inputs.

### Physical type size

§ 16.22 sets minimum type sizes for the warning statement — 1 mm for containers of
237 mL or less, 2 mm for containers above 237 mL up to 3 L, and 3 mm above 3 L — along
with maximum characters-per-inch limits. A photograph contains no scale reference, so
millimeters cannot be derived from pixels.

**Partial mitigation implemented:** when the application record includes the physical
label dimensions, the tool computes a pixels-per-millimeter ratio from the image width
and measures glyph height against it. The result is reported as *measured, advisory*
with the derivation shown, never as a pass or fail. When dimensions are absent, the
check is reported as *not evaluable* rather than silently skipped.

### "Readily legible under ordinary conditions"

§ 16.22(a)(1) requires the warning to be readily legible and on a contrasting
background. Contrast ratio is measurable and is reported. Legibility "under ordinary
conditions" is a judgment standard, not a numeric threshold, and the tool does not
adjudicate it. It surfaces the measurement and leaves the determination to the agent.

### "Separate and apart from all other information"

§ 16.21 requires the warning to appear separate from other label content. This is
approximated with a spatial heuristic — whitespace margins around the detected warning
block relative to neighboring text. It is reported as advisory. Cases involving
decorative rules, borders, or background imagery are unreliable.

### Facts requiring external systems

- Bottler name and address validity → TTB permit database
- Truthfulness of class/type designation → formula and production records
- Actual alcohol content and net contents → laboratory and fill testing
- Whether a prior COLA exists for the same product → COLA system

COLA integration was explicitly placed out of scope during discovery, on the basis that
it carries separate authorization requirements and that this prototype is intended as a
standalone proof of concept.

---

## Measured performance

Generated by `make evaluate` (rules only) and `make evaluate-ocr` (full pipeline)
against `fixtures/` — 29 synthetic labels, 3 runs each. Re-run before reading.

| Metric | Full pipeline (real OCR) | Rules only (stub) |
|---|---|---|
| Single-label verification, p50 | 1.95 s | 0.009 s |
| Single-label verification, p95 | 2.51 s | 0.012 s |
| Brand name — correct classification | 100% (29/29) | 100% |
| Alcohol content — correct classification | 100% (29/29) | 100% |
| Net contents — correct classification | 100% (29/29) | 100% |
| Class/type — correct classification | 100% (29/29) | 100% |
| Warning statement — correct classification | 100% (29/29) | 100% |
| Warning capitalisation — correct classification | 100% (29/29) | 100% |
| **False negatives on non-compliant fixtures** | **0 of 16** | **0 of 16** |
| False flags on compliant fixtures | 0 of 55 fields | 0 of 55 |

Hardware: 2 vCPU x86_64, Python 3.11. p95 of 2.51 s sits inside the five-second
target with roughly half the budget spare, on hardware weaker than any
workstation this would run on.

### These numbers were earned, not designed

The first real-OCR run scored **brand name 24%, warning statement 12%, and 23
false flags across 55 fields on compliant labels** — every compliant fixture in
the corpus was flagged. A tool that flags every clean label is the vendor pilot
again. Three defects produced nearly all of it, and none was visible from the
stub-mode results:

1. **The preprocessing stage never upscaled.** The original rule was "resize
   down to 1600px, never up", on the reasoning that interpolation adds no
   information. True of recognition, false of detection: PP-OCR's detector
   silently dropped whole lines of small, tightly leaded warning text. Line
   recall across the corpus was 88%. The tool then told the agent that the
   missing words were absent from the label — asserting a violation because it
   could not see the text. Resizing to 2600px in both directions took line
   recall to 100%.
2. **Word-space loss.** PP-OCR reads letterspaced display type as one token:
   `OLDTOMDISTILLERY`. Token-based comparison scored that below the match
   threshold, so the largest, clearest text on the label came back as "could
   not locate". Name-like fields are now compared with spaces removed.
3. **A quality metric that measured the wrong thing.** Image quality was scored
   from the standard deviation of luminance, which on a label is dominated by
   how *much* ink is present rather than how well ink separates from paper — so
   a label with less text scored as a worse photograph. Removing the warning
   block from a test label lowered the very score used to decide whether a
   *missing* warning was serious. It now uses Otsu ink-to-paper separation,
   which is coverage-independent.

**False negatives were 0 throughout, including in the worst run.** The
conservative design held where it mattered: every defect above produced false
*flags*, never a violation passed as compliant.

### Can the harness fail?

A 100% score from a corpus written alongside the code is exactly what an
over-permissive grader would also produce, so this was tested directly by
reintroducing defects and confirming the harness catches them:

- Reading the capitalisation check on uppercased text (the trap that would make
  every title-case header pass) turns the title-case fixture into a false
  negative and fails the run.
- Treating a one-point ABV difference as exact turns the ABV-mismatch fixture
  into a false negative and fails the run.

Both were also caught by unit tests. `MATCH` is forbidden on all 16 violation
fields; two fields accept `MATCH` or `REVIEW` (the centilitre fixture and the
case-only brand name) and count toward neither total. `make evaluate` exits
non-zero on any false negative.

### What these numbers are not

The corpus is synthetic, so **read them as optimistic.** They do not include
real print, real photography, foil, embossing, curved bottle surfaces, or
script and decorative brand lettering — which is where local OCR is expected to
be weakest and which no fixture currently exercises. Validation against a
sample of genuine submitted labels remains the first thing to do before any
pilot, and is the most likely source of a number materially worse than these.

---

## What production would require

Beyond the functional gaps above:

- **Authentication and authorization** — agency SSO or PIV/CAC. The prototype is
  protected by a shared secret only.
- **Audit logging and records retention** — per the applicable NARA schedule, including
  agent override capture.
- **Authorization boundary** — deployment within a FedRAMP-authorized environment and an
  ATO covering this system. Running inside TTB's existing Azure tenant is the shortest
  path, as noted during discovery.
- **Section 508 conformance audit** — this prototype was built against WCAG 2.1 AA (no
  color-only status indication, keyboard navigable, legible at 200% zoom, large targets),
  but a good-faith implementation is not a formal audit and should not be represented as
  one.
- **Accuracy validation on real submissions**, with a measured baseline against current
  agent performance rather than against synthetic fixtures.
- **Model provenance documentation** sufficient for federal ML use — see the model weight
  note in `THIRD-PARTY-NOTICES.md`.

---

## Where the next effort would go

In priority order, given more time:

1. **Multi-panel submissions.** The largest gap between this prototype and real use, and
   a contained change.
2. **Validation against real label images.** Everything reported here rests on synthetic
   fixtures. This is the cheapest way to find out what is actually broken.
3. **Override capture.** Not because the feature is valuable on its own, but because it
   is the only route to knowing whether the tool is right.
4. **Class/type vocabulary.** The weakest field, and the one most likely to generate
   flags an agent finds unhelpful.
5. **Wine and malt beverage rule sets.** Additive, and the mechanism is already in place.

Deliberately *not* on this list: increasing automation, raising the match threshold to
reduce flags, or adding an auto-approve path. Each would improve a metric while making
the tool worse at its actual job.
