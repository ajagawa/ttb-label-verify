# Field Assignment Heuristics

How OCR output — an unordered bag of text regions — becomes five located, scored fields.

This is the part of the pipeline with the most hidden complexity, and the part most
likely to consume build time if approached as "parse the label." It is not a parsing
problem. It is a search problem, and framing it correctly makes it considerably easier.

---

## The framing that makes this tractable

**You are not extracting the label. You are looking for known values on it.**

The application record already contains the expected brand name, class/type, alcohol
content, and net contents. The required warning text is fixed by regulation. So the
question is never "what does this label say?" — it is "does `OLD TOM DISTILLERY` appear
on this image, and where?"

This matters because it changes the accuracy requirement. If OCR reads
`OID TOM DISTILERY`, open-ended extraction has failed. Targeted search has not: that
string scores about 0.95 against the known target, so the value is located
confidently. Mediocre OCR is sufficient to *find* a known value.

It is not sufficient to *certify* a residual difference away — a correction made
after running the real engine. `OID` for `OLD` is an OCR confusion that
normalisation folds, but the missing `L` is not, and it is indistinguishable
from a label that is genuinely misspelled. Ratio-based scores also forgive such
a difference more as the name gets longer, which let `OLD TON DISTILLERY` match
`OLD TOM DISTILLERY`. `MATCH` on name-like fields therefore also requires that no
character difference survive normalisation; any that does resolves to `REVIEW`
with the characters named. See `residual_difference` in `rules/comparators.py`.

Two consequences shape everything below:

- Fields with a known expected value use **anchored search** — score every candidate
  region against the target, take the best.
- Fields with a canonical syntactic form use **pattern search** — find by shape, then
  compare to expectation.

These have different failure modes. Anchored search fails by finding nothing. Pattern
search fails by finding the *wrong instance*. They need different safeguards.

---

## Stage 1 — Building the candidate index

OCR returns lines. Fields do not respect line boundaries in either direction:

- A brand name splits across lines: `OLD TOM` / `DISTILLERY`
- A single line holds two fields: `750 mL  •  45% Alc./Vol.`

So matching against raw OCR lines directly will fail on real labels. Build a candidate
span index first.

### Data structures

```python
@dataclass
class Word:
    text: str
    bbox: BBox
    confidence: float


@dataclass
class Line:
    index: int
    words: list[Word]
    bbox: BBox
    confidence: float  # mean of constituent words


@dataclass
class Span:
    """A contiguous run of words or lines, treated as one match candidate."""

    line_indices: list[int]
    word_range: tuple[int, int] | None  # set for sub-line spans
    raw_text: str  # exactly as recognized — never modified
    canonical_text: str  # normalized, for scoring only
    bbox: BBox  # union of constituents
    mean_confidence: float
    glyph_height: float  # median height, for size ranking
```

### Generating spans

1. **Order the lines.** Sort by `bbox.top`, grouping into visual rows where vertical
   centers fall within 0.5 × median line height, then left-to-right within each row.

2. **Multi-line spans.** For every line, generate spans of 1 to 4 consecutive lines that
   pass a spatial coherence test:

   - vertical gap between them < 1.5 × mean line height
   - horizontal overlap ratio > 0.3 (stacked, not side by side)
   - glyph height ratio between 0.6 and 1.67 (same apparent type size)

   The height test is what stops a span from merging a display-size brand name with the
   6pt legal text beneath it.

3. **Sub-line spans.** Within each line, generate every contiguous run of 1 to 6 words.
   This is what recovers `750 mL` from a line that also contains the ABV.

For a typical label of 30–60 lines this produces a few thousand candidates. That is
cheap to score and worth the exhaustiveness.

---

## Stage 2 — Normalization

Every span carries two text forms. **Score against `canonical_text`; display and check
formatting against `raw_text`.**

Canonicalization applies, in order:

1. Unicode normalization (NFKD), smart quotes and dashes to ASCII, ligatures expanded
2. Uppercase
3. Punctuation stripped, except decimal points inside digit runs
4. Whitespace collapsed
5. Common OCR confusion classes folded: `O`↔`0`, `I`↔`1`↔`L`, `S`↔`5`, `B`↔`8`,
   `rn`↔`M`, `CL`↔`D`

### The trap worth naming

**Never run formatting checks against canonical text.** Step 2 uppercases everything,
which destroys exactly the signal the `GOVERNMENT WARNING` capitalization check exists to
detect. The title-case violation described in discovery — `Government Warning` instead of
`GOVERNMENT WARNING` — becomes invisible if the check reads the wrong field.

This is the single easiest bug to introduce here and the hardest to notice, because
everything still appears to work. Add a fixture for it specifically.

### The second trap

**Step 5 applies to names, never to numbers.** Folding `S`→`5` is helpful when matching
a brand name and dangerous when reading alcohol content — it can silently convert a
genuine discrepancy into an apparent match. Numeric fields use a separate, stricter
canonicalization that does no character folding at all. See
[Numeric fields are never fuzzy-forgiven](#numeric-fields-are-never-fuzzy-forgiven).

---

## Stage 3 — Resolution order

Fields are resolved in descending order of precision, and each resolved field's region is
**masked out of the candidate pool** before the next field runs.

```
1. Government warning     (longest, most distinctive, fixed text)
2. Alcohol content        (strong syntactic pattern)
3. Net contents           (strong syntactic pattern)
4. Brand name             (anchored, stylized, size prior available)
5. Class/type             (anchored, weak, depends on brand position)
```

Masking prevents a real failure mode: the warning statement contains the phrase
*alcoholic beverages*, which partially matches class/type values containing *alcohol*,
and it contains numerals that a loose net-contents pattern can latch onto. Resolving and
removing it first eliminates most cross-field contention in one step.

Ordering also creates a useful dependency — class/type resolution uses the brand name's
resolved position as a spatial prior, so brand must run first.

If assignment conflicts persist after masking, the principled fix is optimal assignment
(Hungarian) over a field × span cost matrix. Greedy-by-precision is simpler and has been
sufficient on the fixture corpus; escalate only if diagnostics show contention.

---

## Stage 4 — Per-field strategies

### Government warning

The most reliable anchor on any label. Resolve it first and let everything else benefit.

```python
def locate_warning(spans: list[Span], lines: list[Line]) -> FieldLocation:
    # 1. Anchor on the header bigram, fuzzily
    anchor = best_match("GOVERNMENTWARNING", spans, min_score=0.75)

    if anchor is None:
        # 2. Fallback: distinctive tail phrases
        #    "SURGEON GENERAL" / "BIRTH DEFECTS" / "OPERATE MACHINERY"
        #    A hit here means the warning is present but the header is
        #    mangled or missing — itself a finding.
        anchor = best_match_any(TAIL_PHRASES, spans, min_score=0.80)

    if anchor is None:
        return FieldLocation(strategy="not_found", ...)

    # 3. Expand forward through reading order, maximizing similarity to
    #    the full § 16.21 text. Stop when adding the next line lowers it.
    return expand_until_peak(anchor, lines, target=STATUTORY_WARNING_TEXT)
```

The expanded span's bbox is what the typographic checks operate on — stroke width,
background contrast, separation from neighbors. Getting this region right matters more
than getting the text perfect.

Note that Jenny Park's title-case example still *matches* at this stage, and correctly so.
It is not a location failure; it is a formatting violation, caught downstream by the
capitalization check reading `raw_text`.

### Alcohol content

Pattern search over concatenated text with positions preserved, so the source span is
recoverable for the bbox.

```python
ABV_PATTERNS = [
    r"(\d{1,2}(?:\.\d{1,2})?)\s*%\s*(?:ALC|ALCOHOL)",
    r"ALC(?:OHOL)?\.?\s*(?:/|BY)?\s*VOL\.?\s*:?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%",
    r"(\d{1,3}(?:\.\d)?)\s*PROOF",
    r"(\d{1,2}(?:\.\d{1,2})?)\s*%",  # last resort, low precedence
]
```

Multiple hits are common — `45% Alc./Vol. (90 Proof)` matches three patterns. Collect all,
then:

- **Cross-validate.** If both ABV and proof are present, check `proof == 2 × abv`.
  Consistent → raise confidence. Inconsistent → that is a finding in its own right
  (internal inconsistency on the label) and should surface even if the ABV matches the
  application.
- **Disambiguate by proximity to expectation** only for selecting *which* hit to report,
  never for deciding whether it matches.

### Net contents

```python
VOLUME_PATTERN = r"(\d+(?:\.\d+)?)\s*(ML|L|CL|LITERS?|LITRES?|FL\.?\s*OZ)"
```

Normalize everything to milliliters before comparison: `750 mL`, `75 cL`, `0.75 L`, and
`25.4 fl oz` are the same value. Compare numerically against the application record, then
separately against the standards of fill table.

### Numeric fields are never fuzzy-forgiven

If the label reads `46%` and the application says `45%`, that is reported — never
silently resolved as a probable OCR error, even though a single-character confusion could
explain it.

The reasoning is the asymmetry that governs the whole tool. A one-point ABV discrepancy is
exactly the kind of real violation this exists to catch. Auto-correcting it toward the
expected value would mean the tool is most likely to fail precisely where it matters most.

What the tool *does* do is annotate: `REVIEW — label reads 46%, application states 45%.
Single-digit OCR confusion is possible; verify visually.` The agent gets both readings and
the highlighted region. The tool does not decide.

This applies to alcohol content and net contents. It does not apply to brand name or
class/type, where fuzzy tolerance is the correct behavior.

### Brand name

Anchored search, with a positional prior as backup.

```python
def locate_brand(spans, expected: str, mask: Region) -> FieldLocation:
    candidates = [s for s in spans if not s.bbox.overlaps(mask)]

    scored = [
        (span, composite_score(span, expected))
        for span in candidates
    ]
    best = max(scored, key=lambda x: x[1])

    if best[1] >= BRAND_MATCH_THRESHOLD:
        return FieldLocation(strategy="anchored", ...)

    # Fallback: the brand is typically the largest text on the panel.
    # A weak fuzzy match on display-size type in the upper third beats
    # a strong match on 6pt type at the bottom.
    return locate_by_size_and_position(candidates, expected)
```

Composite scoring:

```
score = fuzz_ratio(span.canonical, expected.canonical)
      × (0.70 + 0.30 × span.mean_confidence)
      × length_penalty(span, expected)
```

`length_penalty` discounts spans substantially longer than the target — a five-word span
scoring 0.90 against a two-word brand is usually matching a fine-print mention
(`Bottled by OLD TOM DISTILLERY`) rather than the brand display itself. Both are the brand
name, but the display instance is the one the agent wants highlighted.

Stylized and script lettering is where this field fails. When nothing clears the
threshold, return `not_found` and let it resolve to `REVIEW`. **Do not guess.**

### Class/type

The weakest field, and the one to set expectations about. Three strategies in order:

1. Anchored search against the expected value.
2. **Positional prior** — class/type sits adjacent to the brand name, usually directly
   below, in smaller type. Restrict candidates to spans within ~2 × brand glyph height
   of the resolved brand bbox and re-score.
3. **Controlled vocabulary** — match against known designations from `rules/ttb-v1.yaml`.
   This lets the tool report something useful even when the expected value doesn't match:
   *found `Kentucky Straight Bourbon Whiskey`, application states `Straight Bourbon
   Whiskey`* is a far more actionable output than *not found*.

Strategy 3 is what makes this field worth including at all. Without it, a class/type
mismatch reads identically to an OCR failure.

---

## Stage 5 — Scoring into verdicts

```python
@dataclass
class FieldLocation:
    field: str
    raw_text: str | None
    bbox: BBox | None
    score: float
    strategy: Literal["anchored", "pattern", "positional", "vocabulary", "not_found"]
    ocr_confidence: float
    diagnostics: dict
```

Bands, tuned per field in `rules/ttb-v1.yaml`:

| Verdict | Condition |
|---|---|
| `MATCH` | score ≥ field threshold (default 0.92) and OCR confidence adequate |
| `REVIEW` | score in the ambiguous band, **or** not found, **or** OCR confidence low, **or** numeric value differs |
| `MISMATCH` | located with high confidence **and** the value clearly differs |

### Not-found is never a mismatch

This is a safety-critical distinction. `MISMATCH` asserts that the label is wrong.
Absence of a match asserts only that the tool didn't find something — which on a
stylized label frequently means OCR failed, not that the label is non-compliant.

Asserting a violation because extraction failed is the worst error this system can make.
It is worse than missing a violation, because it actively misleads an agent who is
trusting the tool.

The one case that pulls the other way: a mandatory warning statement genuinely absent from
a good-quality image is a rejection-grade finding. It still reports as `REVIEW`, but at
elevated prominence with explicit text — *no warning statement located; image quality is
good; verify manually* — rather than as `MISMATCH`. The agent looks. The tool does not
assert.

---

## Diagnostics

Emit a per-verification diagnostic record: every candidate span, every score, which
strategy won each field, and what was masked when.

This is not optional polish. Tuning thresholds without it means guessing, and it is the
raw material for `docs/threshold-analysis.md`. It costs an afternoon to add and saves
considerably more than that during tuning.

Gate it behind a flag; it is verbose and has no place in a production response.

---

## Suggested build order

The dependency chain is real — following it avoids rework:

1. Span index and normalization, with unit tests on fragmentation cases
2. Warning location and expansion — highest value, most reliable, unblocks masking
3. Pattern fields (ABV, net contents) — self-contained, quick wins
4. Brand name anchored search
5. Class/type, which depends on brand position
6. Diagnostics, then threshold tuning against the fixture corpus

Fixtures worth writing before the code they test:

- brand name split across two lines
- two fields sharing one line
- title-case `Government Warning`
- ABV present as both percentage and proof, consistent
- ABV present as both, *inconsistent*
- net contents in litres rather than millilitres
- brand name appearing twice (display and fine print)
- warning statement with the header missing but body intact
