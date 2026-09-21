"""Government health warning strategy — 27 CFR 16.21 and 16.22.

The highest-priority check in the product, and the one field whose expected
value comes from regulation rather than from the application record: the
target is `ruleset.government_warning.statutory_text`, and `context.expected`
is deliberately empty.

What this module does, in order:

1.  **Reconstruct words and lines** from the candidate span index. The index
    caps multi-line spans at four lines and the statement routinely runs to
    five or more, so the warning cannot be matched as one span. Every line
    appears in the index as a whole-line span, and every word as a one-word
    sub-line span, so the original reading-order word sequence is recoverable
    without widening `StrategyContext` to carry the OCR lines.
2.  **Anchor** on the header, fuzzily and case-insensitively (the title-case
    violation must still *anchor*; it is caught later). Fall back to the tail
    phrases when the header does not anchor — a tail hit alone means the
    statement is present but its header is missing or mangled, which is
    itself a finding.
3.  **Expand** through reading order line by line, keeping whichever cut
    maximises similarity to the statutory text, and stopping at the first line
    that does not improve it.
4.  **Check** — capitalisation from RAW text, wording by a token diff, and the
    four pixel/dimension checks reported honestly as not evaluable or
    advisory.
5.  **Decide** — MISMATCH only on a binding finding the tool can stand behind;
    REVIEW for everything it cannot. The asymmetry is the same one that runs
    through the whole system: a false flag costs an agent twenty seconds, a
    false accusation costs their trust in every other result.

Trap 1 from extraction/normalize.py lives here. `canonicalize()` uppercases,
so the capitalisation check reads `Span.raw_text` and nothing else. The
regression test for the title-case label (Jenny Park's rejected example in the
discovery notes) is `tests/test_warning.py::TestHeaderCapitalisation`.
"""

from __future__ import annotations

import difflib
import functools
import re
import statistics
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Indel, Levenshtein

from extraction.geometry import BBox
from extraction.normalize import collapse_for_anchor, is_all_caps, word_tokens
from extraction.spans import Span
from rules.comparators import StrategyContext, not_found_result, verdict_for
from rules.engine import register_strategy
from rules.results import (
    BoxModel,
    CheckOutcome,
    CheckResult,
    FieldResult,
    Strategy,
    TextDifference,
    Verdict,
)
from rules.schema import GovernmentWarningRule, WarningCheck, WarningTuning

# --- Tuning constants --------------------------------------------------------
#
# As in extraction/spans.py, every spatial threshold is a multiple of the
# warning block's own glyph height, never an absolute pixel count.

#: Longest run of words tried as a header candidate. "GOVERNMENT WARNING" is
#: two words, but OCR may split a word ("GOVERN-" / "MENT") or glue the colon
#: or the "(1)" onto it. Four covers every segmentation seen in practice
#: without letting the window drift into the statement body.
HEADER_MAX_UNITS = 4

# The thresholds that decide whether this module *asserts* a violation live in
# the rule set (`tuning.warning`, plus `defaults.good_image_quality`), not
# here. Every verification result is stamped with the rule-set version, so a
# verdict recorded today can be reproduced against the thresholds that were in
# force when it was made — which a module constant would quietly break. They
# are read from `context.ruleset` at call time and deliberately have no
# fallback default: a value that silently defaults is indistinguishable from
# the rule set having been ignored.
#
# What remains below is mechanism rather than policy: geometry that describes
# how set type behaves, and one confidence attached to an advisory measurement
# that can never change a verdict.

#: Confidence attached to the separation estimate. Low on purpose: it is read
#: from OCR boxes alone, which cannot see decorative rules, borders, or a
#: change of background panel — any of which can set a statement apart.
SEPARATION_CONFIDENCE = 0.4

#: One run of letters in a single case pattern: "Government", "WARNING",
#: "warning". ASCII by construction — the header is prescribed English.
_CASE_RUN = re.compile(r"[A-Z][a-z]+|[A-Z]+|[a-z]+")

#: Pulls the check ids out of the rule set so a typo here fails loudly rather
#: than silently skipping a check. See `_build_checks`.
_HEADER_CHECK = "header_capitalisation"
_TEXT_CHECK = "text_accuracy"

#: Where the header's own text ends in a raw token run: at the colon, or at
#: the "(1)" that follows it when an engine glues the two together.
_HEADER_TERMINATOR = re.compile(r"[:;(]")

_VOWELS = frozenset("AEIOU")


# ---------------------------------------------------------------------------
# Reconstructed OCR structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Unit:
    """One word as recognised, in reading order.

    Usually a one-word sub-line span. Falls back to a whole line when the
    index holds no per-word spans for it (single-word lines, where the index
    de-duplicates the word span into the line span).
    """

    line: int  # reading-order position of the source line
    raw: str  # exactly as recognised — the only form formatting checks read
    tokens: tuple[str, ...]  # canonical, for the diff and for scoring
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class _Line:
    position: int
    bbox: BBox
    glyph_height: float
    first_unit: int  # index into the unit sequence
    end_unit: int  # exclusive


@dataclass(frozen=True, slots=True)
class _Layout:
    units: tuple[_Unit, ...]
    lines: tuple[_Line, ...]

    def line_of(self, unit_index: int) -> _Line:
        position = self.units[unit_index].line
        for line in self.lines:
            if line.position == position:
                return line
        raise KeyError(position)  # pragma: no cover - construction guarantees presence


def _reconstruct(spans: list[Span]) -> _Layout:
    """Rebuild the reading-order word sequence from the span index.

    `build_span_index` emits, for every line, one whole-line span
    (`word_range is None`, one line index) and one sub-line span per word
    (`word_range == (i, i + 1)`), all keyed by the line's reading-order
    position. Those two families are enough to recover the original words with
    their boxes and confidences, which is what expansion and the word diff
    need. Multi-line spans are ignored: they are unions of lines we already
    have.

    Words whose text canonicalises to nothing ("•", "—") are absent from the
    index and therefore from the reconstruction. They carry no wording, so the
    diff does not miss them; the found text shown to the agent omits them.
    """
    whole: dict[int, Span] = {}
    words: dict[int, dict[int, Span]] = {}

    for span in spans:
        if len(span.line_indices) != 1:
            continue
        position = span.line_indices[0]
        if span.word_range is None:
            whole[position] = span
        elif span.word_range[1] - span.word_range[0] == 1:
            words.setdefault(position, {})[span.word_range[0]] = span

    units: list[_Unit] = []
    lines: list[_Line] = []

    for position in sorted(set(whole) | set(words)):
        members = [words[position][i] for i in sorted(words.get(position, {}))]
        if not members:
            members = [whole[position]]

        first = len(units)
        units.extend(
            _Unit(
                line=position,
                raw=span.raw_text,
                tokens=tuple(span.canonical_text.split()),
                bbox=span.bbox,
                confidence=span.mean_confidence,
            )
            for span in members
        )

        line_span = whole.get(position)
        lines.append(
            _Line(
                position=position,
                bbox=line_span.bbox if line_span else BBox.union([s.bbox for s in members]),
                glyph_height=(
                    line_span.glyph_height
                    if line_span
                    else statistics.median(s.bbox.height for s in members)
                ),
                first_unit=first,
                end_unit=len(units),
            )
        )

    return _Layout(units=tuple(units), lines=tuple(lines))


# ---------------------------------------------------------------------------
# The statutory text, tokenised once
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Statute:
    raw_words: tuple[str, ...]
    tokens: tuple[str, ...]
    #: For each canonical token, the index of the raw word it came from, so a
    #: difference can be shown to the agent as printed ("defects.") rather
    #: than as canonicalised ("DEFECTS").
    token_word: tuple[int, ...]
    vocabulary: frozenset[str]


@functools.lru_cache(maxsize=4)
def _statute(statutory_text: str) -> _Statute:
    raw_words = tuple(statutory_text.split())
    tokens: list[str] = []
    token_word: list[int] = []
    for index, word in enumerate(raw_words):
        for token in word_tokens(word):
            tokens.append(token)
            token_word.append(index)
    return _Statute(raw_words, tuple(tokens), tuple(token_word), frozenset(tokens))


# ---------------------------------------------------------------------------
# OCR noise versus a wording change
# ---------------------------------------------------------------------------
#
# This is the line the whole text-accuracy check rests on, so it is drawn
# explicitly and conservatively. Three classes:
#
#   noise     Tolerated. Not reported as a difference (it is listed in the
#             check's detail so it is never *hidden*). A single-character
#             edit on a statutory word of five or more letters, where the edit
#             does not produce a recognisable variant of the word:
#               - not a prefix/suffix truncation ("defects"->"defect",
#                 "alcoholic"->"alcohol", "impairs"->"impair"): grammatical
#                 number and inflection are wording, and OCR rarely drops a
#                 clean trailing letter;
#               - not a vowel-for-vowel substitution ("women"->"woman",
#                 "drink"->"drank"): vowel swaps make different words, and
#                 engines seldom confuse two vowels (the common OCR vowel
#                 confusions, O/0 and I/1/l, are already folded away by
#                 canonicalisation).
#             Also noise: the engine splitting or merging words
#             ("GOVERNMENTWARNING", "CONSUMP TION") when the letters agree to
#             within one edit.
#
#   ambiguous Reported as a difference, but not by itself grounds for a
#             MISMATCH: two edits on a long word ("machinery"->"machines",
#             "impairs"->"impaired"). Could be a real change, could be a
#             badly read word; the agent looks.
#
#   wording   Reported and assertable. Any change to a word under five
#             letters ("may"->"can"), anything further than two edits, and
#             the variants excluded from noise above.
#
# Canonicalisation has already removed case, punctuation and the
# single-character confusion classes before any of this runs, so "(l)" and
# "(1)", "Surgeon" and "SURGEON" never reach it.


#: Digit-to-letter folding for the statute's enumerators "(1)" and "(2)".
#: Mirrors the single-character table in extraction/normalize.py, which skips
#: purely numeric tokens on purpose (trap 2: never fold a quantity). These are
#: not quantities — they are fixed labels in fixed text, and "(l)" for "(1)" is
#: among the most common misreads there is — so the warning diff folds them.
_ENUMERATOR_FOLDING = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})


def _classify(expected: str, found: str, tuning: WarningTuning) -> str:
    """Classify a substituted token pair as "noise", "ambiguous" or "wording".

    `tuning.noise_min_token_length` is the line itself: statutory words shorter
    than it must match exactly, longer ones absorb a single-character edit.
    """
    if expected.translate(_ENUMERATOR_FOLDING) == found.translate(_ENUMERATOR_FOLDING):
        return "noise"
    if len(expected) < tuning.noise_min_token_length:
        return "wording"
    if expected.startswith(found) or found.startswith(expected):
        return "wording"

    distance = Levenshtein.distance(expected, found)
    if distance == 1:
        if len(expected) == len(found):
            changed = [(a, b) for a, b in zip(expected, found, strict=True) if a != b]
            if all(a in _VOWELS and b in _VOWELS for a, b in changed):
                return "wording"
        return "noise"
    if distance == 2:
        return "ambiguous"
    return "wording"


def _is_segmentation_noise(expected: list[str], found: list[str], tuning: WarningTuning) -> bool:
    """Whether a replaced run differs only in where the engine put spaces.

    Only meaningful when the word counts differ — a one-for-one replacement
    is a substitution and goes through `_classify` token by token, where the
    inflection and vowel rules apply. Here the joined letters are held to the
    same rules, so a merge cannot smuggle a wording change past them.
    """
    if len(expected) == len(found):
        return False
    return _classify("".join(expected), "".join(found), tuning) == "noise"


# ---------------------------------------------------------------------------
# Location: anchor, then expand
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Located:
    start: int  # unit index, inclusive
    end: int  # unit index, exclusive
    anchor: str  # "header" | "tail"
    anchor_score: float
    #: Unit range of the header, when the header anchored. None on a tail
    #: anchor: then the header was either absent or unreadable.
    header: tuple[int, int] | None


def _windows(layout: _Layout, max_units: int):
    """Every run of 1..max_units consecutive units that crosses at most
    adjacent lines — so a header split across a line break is a candidate, but
    two words from opposite ends of the label are not."""
    units = layout.units
    for start in range(len(units)):
        for end in range(start + 1, min(start + max_units, len(units)) + 1):
            if end - start > 1 and units[end - 1].line - units[end - 2].line > 1:
                break
            yield start, end


def _anchor_header(layout: _Layout, rule: GovernmentWarningRule) -> tuple[int, int, float] | None:
    """Best fuzzy match for the header, on whitespace-collapsed canonical text.

    Collapsing makes "GOVERNMENT WARNING", "GOVERNMENTWARNING" and a header
    broken across two lines one key. Case-insensitive by construction — the
    title-case violation must anchor here, because it is a formatting failure
    for the capitalisation check, not a location failure.

    Ties go to the shorter window, then the earlier one, so a window that
    happens to include the following "(1)" does not displace the header
    itself.
    """
    target = collapse_for_anchor(rule.header_text)
    best: tuple[float, int, int] | None = None  # (score, -size, -start)
    best_range = None

    for start, end in _windows(layout, HEADER_MAX_UNITS):
        key = collapse_for_anchor(" ".join(u.raw for u in layout.units[start:end]))
        if not key:
            continue
        score = fuzz.ratio(key, target) / 100.0
        rank = (score, -(end - start), -start)
        if best is None or rank > best:
            best, best_range = rank, (start, end)

    if best is None or best[0] < rule.anchor.header_min_score:
        return None
    return best_range[0], best_range[1], best[0]


def _anchor_tail(layout: _Layout, rule: GovernmentWarningRule) -> tuple[int, int, float] | None:
    """Best fuzzy match for any distinctive tail phrase ("SURGEON GENERAL")."""
    best: tuple[float, int, int] | None = None
    for phrase in rule.anchor.tail_phrases:
        target = collapse_for_anchor(phrase)
        width = len(phrase.split())
        for start, end in _windows(layout, width + 1):
            key = collapse_for_anchor(" ".join(u.raw for u in layout.units[start:end]))
            if not key:
                continue
            score = fuzz.ratio(key, target) / 100.0
            if best is None or score > best[0]:
                best = (score, start, end)

    if best is None or best[0] < rule.anchor.tail_min_score:
        return None
    return best[1], best[2], best[0]


class _Scorer:
    """Token-level similarity of a unit range to the statutory text.

    Token level, not character level, and the difference is decisive for
    expansion. A character-level ratio against a long target rewards almost
    any appended English text — "Bottled by Old Tom Distillery" shares plenty
    of letters, in order, with the remaining statute — so expansion would run
    on into the next block. At token level an unrelated line matches almost
    nothing and lowers the score, which is the stopping signal.

    Tokens within one tolerated edit of a statutory word are snapped to it for
    scoring only, so OCR noise does not cost location accuracy, and a token
    that merges several statutory words is split back into them. The diff
    never sees either transformation.

    The merge handling is load-bearing, not a refinement. Real OCR drops
    spaces constantly — the corpus produced "healthproblems.",
    "ofalcoholicbeveragesimpairsyourability", "caroroperate". Scored naively
    a merged token matches no statutory token at all, so including it *lowers*
    similarity and expansion stops short of it. The statement then appears to
    end early and its last words are reported missing: a MISMATCH on a
    compliant label, produced by a dropped space. The diff already tolerated
    merges (`_is_segmentation_noise`); it never got the chance, because the
    merged word had been left outside the located block.
    """

    #: Longest run of statutory words recognised as one merged token. Six
    #: covers the worst merge in the corpus ("of alcoholic beverages impairs
    #: your ability"); beyond that the diff still tolerates it, only the
    #: location score misses it.
    MAX_MERGED_WORDS = 6

    def __init__(self, layout: _Layout, statute: _Statute, tuning: WarningTuning) -> None:
        self._layout = layout
        self._statute = statute
        self._tuning = tuning
        self._cache: dict[str, tuple[str, ...]] = {}
        self._merged: dict[str, tuple[str, ...]] = {}
        tokens = statute.tokens
        for size in range(2, self.MAX_MERGED_WORDS + 1):
            for index in range(len(tokens) - size + 1):
                run = tokens[index : index + size]
                self._merged.setdefault("".join(run), run)

    def resolve(self, token: str) -> tuple[str, ...]:
        """The statutory token(s) this read token stands for, for scoring only."""
        if token in self._statute.vocabulary:
            return (token,)
        if token not in self._cache:
            self._cache[token] = self._resolve(token)
        return self._cache[token]

    def _resolve(self, token: str) -> tuple[str, ...]:
        run = self._merged.get(token)
        if run is not None:
            return run
        snapped = next(
            (w for w in self._statute.vocabulary if _classify(w, token, self._tuning) == "noise"),
            None,
        )
        if snapped is not None:
            return (snapped,)
        # A merged run read with a character error: compare against the
        # concatenations, on the same one-edit budget a single word gets.
        for joined, run in self._merged.items():
            if abs(len(joined) - len(token)) <= 1 and Levenshtein.distance(joined, token) <= 1:
                return run
        return (token,)

    def __call__(self, start: int, end: int) -> float:
        found = [
            resolved
            for unit in self._layout.units[start:end]
            for token in unit.tokens
            for resolved in self.resolve(token)
        ]
        return Indel.normalized_similarity(found, list(self._statute.tokens))


def _expand_forward(layout: _Layout, score: _Scorer, start: int, end: int) -> int:
    """Extend `end` through reading order while similarity improves.

    Considers every cut point in the current line and then in each following
    line, and stops at the first line that offers no improvement. Evaluating
    every cut rather than whole lines is what handles a statement that ends
    mid-line with other text after it; considering the *next* line even when
    the best cut in this one was partial is what stops a single garbled token
    at a line end from truncating the statement.
    """
    best_end, best = end, score(start, end)
    line_index = next(
        i for i, ln in enumerate(layout.lines) if ln.position == layout.units[end - 1].line
    )

    # Remainder of the anchor's own line.
    for cut in range(end + 1, layout.lines[line_index].end_unit + 1):
        value = score(start, cut)
        if value > best:
            best_end, best = cut, value

    for line in layout.lines[line_index + 1 :]:
        candidates = [
            (score(start, cut), cut) for cut in range(line.first_unit + 1, line.end_unit + 1)
        ]
        value, cut = max(candidates, key=lambda c: (c[0], -c[1]))
        if value <= best:
            break
        best_end, best = cut, value

    return best_end


def _expand_backward(layout: _Layout, score: _Scorer, start: int, end: int) -> int:
    """Mirror of `_expand_forward`, for a tail anchor that sits mid-statement."""
    best_start, best = start, score(start, end)
    line_index = next(
        i for i, ln in enumerate(layout.lines) if ln.position == layout.units[start].line
    )

    for cut in range(layout.lines[line_index].first_unit, start):
        value = score(cut, end)
        if value > best:
            best_start, best = cut, value

    for line in reversed(layout.lines[:line_index]):
        candidates = [(score(cut, end), cut) for cut in range(line.first_unit, line.end_unit)]
        value, cut = max(candidates, key=lambda c: (c[0], c[1]))
        if value <= best:
            break
        best_start, best = cut, value

    return best_start


def _locate(layout: _Layout, rule: GovernmentWarningRule, score: _Scorer) -> _Located | None:
    """Anchor on the header, fall back to the tail phrases, then expand.

    A header anchor fixes the start: the statute begins with the header, so
    anything before it on the page is not the statement. A tail anchor sits
    mid-statement, so it expands both ways — backward to recover the body
    that precedes it (and a mangled header, if one is there to recover).
    """
    if not layout.units:
        return None

    header = _anchor_header(layout, rule)
    if header is not None:
        start, end, anchor_score = header
        end = _expand_forward(layout, score, start, end)
        return _Located(start, end, "header", anchor_score, (header[0], header[1]))

    tail = _anchor_tail(layout, rule)
    if tail is None:
        return None
    start, end, anchor_score = tail
    end = _expand_forward(layout, score, start, end)
    start = _expand_backward(layout, score, start, end)
    end = _expand_forward(layout, score, start, end)
    return _Located(start, end, "tail", anchor_score, None)


# ---------------------------------------------------------------------------
# Evidence that a gap is real rather than unread
# ---------------------------------------------------------------------------
#
# A statutory word the tool did not read is either missing from the label or
# missed by the engine. Only geometry can tell those apart, and only in one
# direction: if the words either side of the gap sit where consecutive words
# of set type would sit — same line with an ordinary word gap, or the end of
# one full line and the start of the next with ordinary leading — there is
# nowhere for the missing words to be. That is positive evidence of absence,
# and it is what allows a "missing" difference to be asserted. Without it the
# difference is still reported, as REVIEW.


@dataclass(frozen=True, slots=True)
class _BlockGeometry:
    bbox: BBox
    glyph_height: float
    max_line_gap: float


def _block_geometry(layout: _Layout, located: _Located, tuning: WarningTuning) -> _BlockGeometry:
    units = layout.units[located.start : located.end]
    positions = sorted({u.line for u in units})
    lines = [ln for ln in layout.lines if ln.position in positions]
    glyph = statistics.median(ln.glyph_height for ln in lines) or 1.0
    gaps = [a.bbox.vertical_gap(b.bbox) for a, b in zip(lines, lines[1:], strict=False)]
    typical = statistics.median(gaps) if gaps else glyph * 0.5
    return _BlockGeometry(
        bbox=BBox.union([u.bbox for u in units]),
        glyph_height=glyph,
        max_line_gap=typical + glyph * tuning.line_gap_slack_ratio,
    )


def _no_room_between(
    layout: _Layout,
    geo: _BlockGeometry,
    before: int,
    after: int,
    tuning: WarningTuning,
    *,
    check_line_end: bool,
    check_line_start: bool,
) -> bool:
    """Whether two consecutive units leave no room for undetected words.

    Across a line break there are three places missed words could hide: at
    the end of the upper line, in a whole undetected line between, or at the
    start of the lower line. The caller says which ends belong to the warning
    block — only those are held to the block's margins. (The line of other
    text above a block has its own width; whether it reaches the warning's
    right margin says nothing about the warning.)
    """
    first, second = layout.units[before], layout.units[after]
    if first.line == second.line:
        gap = second.bbox.left - first.bbox.right
        return gap <= geo.glyph_height * tuning.max_word_gap_ratio

    upper, lower = layout.line_of(before), layout.line_of(after)
    if lower.position - upper.position != 1:
        return False
    if upper.bbox.vertical_gap(lower.bbox) > geo.max_line_gap:
        return False
    if check_line_end and first.bbox.right < (
        geo.bbox.right - geo.bbox.width * tuning.ragged_right_tolerance
    ):
        return False
    return not (
        check_line_start
        and second.bbox.left > geo.bbox.left + geo.glyph_height * tuning.indent_tolerance_ratio
    )


def _gap_is_closed(
    layout: _Layout,
    located: _Located,
    geo: _BlockGeometry,
    at: int,
    tuning: WarningTuning,
) -> bool:
    """Whether a missing run just before unit `at` is bounded on both sides.

    `at == located.start` is the start of the block (missing header or
    opening words); `at == located.end` is its end (missing closing words).
    At a block edge the neighbour is the adjacent non-warning text: a line of
    other text sitting at ordinary leading directly above (or below) is
    positive evidence the statement was not simply cut off by the engine. No
    neighbour at all — the statement is the first or last text read — is not
    evidence of anything, since the image may simply be cropped.
    """
    if at <= 0 or at >= len(layout.units):
        return False
    return _no_room_between(
        layout,
        geo,
        at - 1,
        at,
        tuning,
        check_line_end=at > located.start,  # upper unit is inside the block
        check_line_start=at < located.end,  # lower unit is inside the block
    )


# ---------------------------------------------------------------------------
# The word diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Finding:
    """One thing wrong, phrased for the agent.

    `assertable` is the whole point: True only when the tool has positive
    evidence the label, not the read, is at fault.
    """

    check_id: str
    message: str
    assertable: bool


@dataclass(frozen=True, slots=True)
class _Diff:
    differences: tuple[TextDifference, ...]
    findings: tuple[_Finding, ...]
    tolerated: tuple[tuple[str, str], ...]
    #: Share of the tokens read that are statutory words or tolerated
    #: variants of them. The "was this read cleanly" gate for assertion.
    precision: float
    header_missing: bool


def _quote(words: list[str]) -> str:
    return " ".join(words).strip(" ,.;:")


def _diff(
    layout: _Layout,
    located: _Located,
    statute: _Statute,
    rule_min_conf: float,
    tuning: WarningTuning,
) -> _Diff:
    """Word-level diff of the located block against the statute.

    `TextDifference.position` is an index into the statute's canonical token
    sequence — the first affected statutory word for "missing" and
    "substituted", the insertion point for "extra" — so differences sort into
    reading order and the frontend can map each back to the statutory text.
    `expected` and `found` are shown as printed, not canonicalised.
    """
    found_tokens: list[str] = []
    found_unit: list[int] = []
    for index in range(located.start, located.end):
        for token in layout.units[index].tokens:
            found_tokens.append(token)
            found_unit.append(index)

    geo = _block_geometry(layout, located, tuning)
    clean_tokens = 0
    differences: list[TextDifference] = []
    findings: list[_Finding] = []
    tolerated: list[tuple[str, str]] = []
    header_missing = False

    def exp_raw(i1: int, i2: int) -> list[str]:
        words = dict.fromkeys(statute.token_word[i1:i2])
        return [statute.raw_words[w] for w in words]

    def found_raw(j1: int, j2: int) -> list[str]:
        units = dict.fromkeys(found_unit[j1:j2])
        return [layout.units[u].raw for u in units]

    def context(i1: int, i2: int, j1: int, j2: int) -> tuple[str, str]:
        """The difference with up to three words of following context (one
        preceding word when it sits near the end), so the agent reads a phrase
        they can find on the label rather than a lone token."""
        trail = min(3, len(statute.tokens) - i2, len(found_tokens) - j2)
        lead = min(0 if trail >= 2 else 1, i1, j1)
        return (
            _quote(exp_raw(i1 - lead, i2 + trail)),
            _quote(found_raw(j1 - lead, j2 + trail)),
        )

    def unit_confident(j1: int, j2: int) -> bool:
        return all(layout.units[found_unit[j]].confidence >= rule_min_conf for j in range(j1, j2))

    def in_header(i1: int, i2: int) -> bool:
        return i1 < 2  # statute tokens 0 and 1 are GOVERNMENT WARNING

    def missing(i1: int, i2: int, j: int) -> None:
        nonlocal header_missing
        at = found_unit[j] if j < len(found_unit) else located.end
        if j > 0 and j < len(found_unit) and found_unit[j] == found_unit[j - 1]:
            closed = True  # the gap is inside one recognised word
        else:
            closed = _gap_is_closed(layout, located, geo, at, tuning)
        words = _quote(exp_raw(i1, i2))
        differences.append(TextDifference(position=i1, expected=words, found=None, kind="missing"))
        if in_header(i1, i2) and located.anchor == "tail":
            header_missing = True
            if i2 >= 2:
                message = (
                    "The statement body is present but the “GOVERNMENT WARNING” header was "
                    "not found; 27 CFR 16.21 requires the statement to begin with it"
                )
            else:
                message = (
                    f"the header does not read “GOVERNMENT WARNING”: “{words}” is missing; "
                    "27 CFR 16.21 requires the statement to begin with it"
                )
        else:
            where = _quote(exp_raw(max(0, i1 - 2), i1))
            message = f"the words “{words}” are missing" + (
                f" (expected after “{where}”)" if where else ""
            )
        if not closed:
            message += (
                " — the text around the gap leaves room for words the OCR did not "
                "detect, so confirm visually"
            )
        findings.append(_Finding(_TEXT_CHECK, message, assertable=closed))

    def extra(j1: int, j2: int, i: int) -> None:
        words = _quote(found_raw(j1, j2))
        differences.append(TextDifference(position=i, expected=None, found=words, kind="extra"))
        expected_ctx, found_ctx = context(i, i, j1, j2)
        findings.append(
            _Finding(
                _TEXT_CHECK,
                f"“{expected_ctx}” appears as “{found_ctx}” (added words: “{words}”)",
                assertable=unit_confident(j1, j2),
            )
        )

    def substituted(i1: int, i2: int, j1: int, j2: int, grade: str) -> None:
        differences.append(
            TextDifference(
                position=i1,
                expected=_quote(exp_raw(i1, i2)),
                found=_quote(found_raw(j1, j2)),
                kind="substituted",
            )
        )
        expected_ctx, found_ctx = context(i1, i2, j1, j2)
        message = f"“{expected_ctx}” appears as “{found_ctx}”"
        assertable = grade == "wording" and unit_confident(j1, j2)
        if located.anchor == "tail" and in_header(i1, i2):
            # The header did not anchor, so whatever sits in its place was read
            # too poorly to clear a lenient fuzzy threshold. Asserting a
            # wording change on text that bad is asserting on the engine.
            assertable = False
            message = f"the header reads “{found_ctx}” instead of “GOVERNMENT WARNING”"
        elif grade == "ambiguous":
            message += " — this may be a misread, confirm visually"
        findings.append(_Finding(_TEXT_CHECK, message, assertable=assertable))

    matcher = difflib.SequenceMatcher(a=list(statute.tokens), b=found_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            clean_tokens += j2 - j1
        elif tag == "delete":
            missing(i1, i2, j1)
        elif tag == "insert":
            extra(j1, j2, i1)
        else:  # replace
            expected_run, found_run = list(statute.tokens[i1:i2]), found_tokens[j1:j2]
            if _is_segmentation_noise(expected_run, found_run, tuning):
                clean_tokens += j2 - j1
                tolerated.append((_quote(exp_raw(i1, i2)), _quote(found_raw(j1, j2))))
                continue
            paired = min(i2 - i1, j2 - j1)
            for k in range(paired):
                grade = _classify(expected_run[k], found_run[k], tuning)
                if grade == "noise":
                    clean_tokens += 1
                    tolerated.append(
                        (_quote(exp_raw(i1 + k, i1 + k + 1)), _quote(found_raw(j1 + k, j1 + k + 1)))
                    )
                else:
                    substituted(i1 + k, i1 + k + 1, j1 + k, j1 + k + 1, grade)
            if i2 - i1 > paired:
                missing(i1 + paired, i2, j1 + paired)
            if j2 - j1 > paired:
                extra(j1 + paired, j2, i2)

    precision = clean_tokens / len(found_tokens) if found_tokens else 0.0
    return _Diff(
        differences=tuple(sorted(differences, key=lambda d: d.position)),
        findings=tuple(findings),
        tolerated=tuple(tolerated),
        precision=precision,
        header_missing=header_missing,
    )


# ---------------------------------------------------------------------------
# Header capitalisation and the colon
# ---------------------------------------------------------------------------
#
# The colon decision. 27 CFR 16.21 prescribes the statement "as follows" and
# reproduces it with a colon: "GOVERNMENT WARNING: (1) ...". The colon is
# therefore part of the verbatim text, and its absence is a text-accuracy
# deviation under 16.21 — not a capitalisation matter under 16.22(a)(2),
# which concerns only capital letters and bold type. But a colon is two dots
# a fraction of the x-height, the least reliable thing an OCR engine reports,
# and a missing one in the read is weak evidence of a missing one in print.
# So: a missing colon fails `text_accuracy` (it is a real requirement and the
# agent should look) and resolves to REVIEW, never MISMATCH on its own.


@dataclass(frozen=True, slots=True)
class _Header:
    raw: str | None  # header letters as printed; None when not located
    confidence: float
    has_colon: bool | None  # None when the header was not located


def _read_header(layout: _Layout, located: _Located) -> _Header:
    """The header's RAW text — never canonical. See trap 1."""
    if located.header is None:
        return _Header(raw=None, confidence=0.0, has_colon=None)

    start, end = located.header
    units = layout.units[start:end]
    joined = " ".join(u.raw for u in units)

    # Cut at the colon or at a glued "(1)", so that only the header's own
    # letters are judged: "WARNING:(1)According" must not fail for the "ccording".
    terminator = _HEADER_TERMINATOR.search(joined)
    header_text = joined[: terminator.start()] if terminator else joined

    after = layout.units[end].raw if end < len(layout.units) else ""
    has_colon = ":" in joined[len(header_text) :][:1] or after.startswith(":")
    return _Header(
        raw=header_text.strip(),
        confidence=statistics.fmean(u.confidence for u in units),
        has_colon=has_colon,
    )


def _degraded_image(context: StrategyContext) -> str | None:
    """How the quality gate described this image, phrased for the agent.

    The second line of defence behind `_miscapitalised_words`, used only to
    *withdraw* a typographic assertion — a case difference, a missing colon —
    never to add one. On a photograph the gate has flagged, one mis-cased
    letter is far more likely to be a misread than a label genuinely printed
    that way. Because it can only ever downgrade MISMATCH to REVIEW, no
    reading of the quality score can make this layer less conservative.

    Returns None when there is no *evidence of degradation*, which is not the
    same as evidence of a good image, and the difference matters twice over:

    *   `assessed=False` means nothing was measured (an extraction that never
        went through preprocessing). It is not a good image and this does not
        treat it as one — but it is not a degraded one either, so the guard
        stays out of the way and the case-pattern rule decides alone.
    *   Suppressing assertions whenever quality is unknown would be the worse
        failure: every stub-based verification would quietly stop reporting
        real printing violations, which is the check this module exists for.

    The issue list is the sturdier half of the signal. Any flagged issue
    counts as degrading whatever the score, because the score's calibration
    leaves a thin margin at the threshold.

    The threshold is `ruleset.defaults.good_image_quality` — the same "good
    enough image" the engine elevates a missing warning against, read from the
    rule set the verdict will be stamped with rather than from a module
    constant or a default rule set loaded at import.
    """
    quality = context.quality
    if quality is None or not quality.assessed:
        return None
    if not quality.usable:
        return "the image was assessed as not usable"
    if quality.issues:
        return f"image quality {quality.score:.0%}: {', '.join(quality.issues)}"
    if quality.score < context.ruleset.defaults.good_image_quality:
        return f"image quality {quality.score:.0%}"
    return None


@dataclass(frozen=True, slots=True)
class _Capitalisation:
    """Verdict of the capitalisation check, with the finding it may carry."""

    outcome: CheckOutcome
    summary: str
    detail: str | None = None
    finding: _Finding | None = None


def _miscapitalised_words(raw: str, tuning: WarningTuning) -> list[str]:
    """Header words whose *pattern* of capitals is a printing style, not noise.

    The distinction that matters, learned from real OCR on the fixture corpus:

    *   A label printed in the wrong case is wrong **systematically**. Every
        word goes the same way — "Government Warning", "government warning",
        "GOVERNMENT Warning". Type is set once; a printer does not lowercase
        one letter in the middle of a word.
    *   OCR mis-casing is **sporadic**: one or two letters inside otherwise
        capitalised words — "GOVERNMENT WARNiNG" on a low-contrast photograph,
        "GoVERNMENT WARNING" on a blurred one. Both of those labels are
        printed in correct capitals.

    So a word counts as miscapitalised only when its whole case pattern is
    title case or lowercase. This replaced an earlier rule that listed
    case-ambiguous glyph shapes (c, o, s, l...), which was the right instinct
    and the wrong mechanism: the list can never be complete — the corpus
    produced a lowercase "i", which was not on it — and the pattern carries
    the signal the list was trying to approximate.

    Read in *case runs* rather than whitespace-separated words, because the
    engine drops spaces: "GovernmentWarning:" is the same violation as
    "Government Warning" and has to be caught the same way.

    `tuning.min_case_style_run` is the shortest run read as a deliberate style
    rather than noise: two-letter runs are what alternating misreads produce
    ("GoVeRnMeNt"), while "Government" and "warning" are far longer.
    """
    violations: list[str] = []
    for word in raw.split():
        runs = [run for run in _CASE_RUN.findall(word) if len(run) >= 2]
        if not runs:
            continue  # a lone letter has no pattern to read
        styled = [run for run in runs if run.islower() or (run[0].isupper() and run[1:].islower())]
        if len(styled) == len(runs) and any(
            len(run) >= tuning.min_case_style_run for run in styled
        ):
            violations.append(word)
    return violations


def _capitalisation(
    header: _Header, min_conf: float, degraded: str | None, tuning: WarningTuning
) -> _Capitalisation:
    """Whether "GOVERNMENT WARNING" is in capital letters, from raw text.

    Reads `header.raw`, which came from `Span.raw_text`. Passing canonical text
    here would make every label pass — trap 1 in extraction/normalize.py.

    Sporadic lowercase letters PASS rather than failing non-assertably. The
    check is answering "is this label printed in capitals", and on the
    evidence a header read as "GOVERNMENT WARNiNG" is: reporting a failure the
    tool then declines to stand behind spends an agent's attention on the
    engine's noise. The letters read in lowercase are recorded in the check's
    detail, so the reading is still auditable.
    """
    if header.raw is None:
        return _Capitalisation(
            outcome=CheckOutcome.NOT_EVALUABLE,
            summary=(
                "The “GOVERNMENT WARNING” header was not located, so its capitalisation "
                "could not be checked. Its absence is reported under text accuracy."
            ),
        )

    if is_all_caps(header.raw):
        return _Capitalisation(
            outcome=CheckOutcome.PASS,
            summary=f"Header reads “{header.raw}”, in capital letters.",
            detail="Checked against the text exactly as recognised, not normalised.",
        )

    lowercase = [ch for ch in header.raw if ch.isalpha() and ch.islower()]
    systematic = bool(_miscapitalised_words(header.raw, tuning))

    # `tuning.max_sporadic_lowercase`: how many isolated lowercase letters in
    # an otherwise capitalised header are read as engine noise. One or two is
    # what degraded photographs produce ("GOVERNMENT WARNiNG").
    if not systematic and len(lowercase) > tuning.max_sporadic_lowercase:
        # Too many mis-cased letters to be the engine's usual noise, too
        # irregular to be a printing style ("GoVeRnMeNt WaRnInG"). Neither
        # answer is supportable, which is what REVIEW is for.
        return _Capitalisation(
            outcome=CheckOutcome.FAIL,
            summary=(
                f"Header reads “{header.raw}”, with {len(lowercase)} letters in lowercase in "
                "no consistent pattern — more than a typical misread, but not the uniform "
                "title case or lowercase a miscapitalised label shows. Confirm visually."
            ),
            detail="Checked against the text exactly as recognised, not normalised.",
            finding=_Finding(
                _HEADER_CHECK,
                f"“{header.raw}” was read with scattered lowercase letters; 27 CFR "
                "16.22(a)(2) requires “GOVERNMENT WARNING” in capital letters — the "
                "pattern fits neither clean capitals nor a miscapitalised label, so "
                "confirm visually",
                assertable=False,
            ),
        )

    if not systematic:
        return _Capitalisation(
            outcome=CheckOutcome.PASS,
            summary=(
                f"Header reads “{header.raw}” — capitals throughout apart from isolated "
                f"lowercase letter(s) ({', '.join(sorted(set(lowercase)))}), which is a "
                "characteristic OCR misread rather than a printing style."
            ),
            detail=(
                "Checked against the text exactly as recognised, not normalised. A label "
                "printed in the wrong case is wrong systematically — every word title case "
                "or lowercase — so a single mis-cased letter inside capitalised words is "
                "read as engine noise. Confirm visually if the image is degraded."
            ),
        )

    description = "appears in title case" if header.raw.istitle() else "is not in capital letters"
    message = (
        f"“{header.raw}” {description}; 27 CFR 16.22(a)(2) requires "
        "“GOVERNMENT WARNING” in capital letters"
    )
    if header.confidence < min_conf:
        message += f" — read with low confidence ({header.confidence:.0%}), confirm visually"
        assertable = False
    elif degraded is not None:
        message += (
            f" — but the image quality gate flagged this image ({degraded}), so confirm "
            "visually before rejecting"
        )
        assertable = False
    else:
        assertable = True

    return _Capitalisation(
        outcome=CheckOutcome.FAIL,
        summary=message[0].upper() + message[1:] + ".",
        detail="Checked against the text exactly as recognised, not normalised.",
        finding=_Finding(_HEADER_CHECK, message, assertable=assertable),
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _separation(
    check: WarningCheck, layout: _Layout, located: _Located, tuning: WarningTuning
) -> CheckResult:
    """Whitespace around the block, from OCR boxes, relative to line height.

    Advisory by rule and by construction. "Separate and apart from all other
    information" is satisfied by a border, a rule, a panel change or a colour
    block as readily as by whitespace, and none of those is visible in OCR
    output. What *is* measurable: how far the nearest other text sits above,
    below and beside the block, and whether other text shares a line with it.
    """
    geo = _block_geometry(layout, located, tuning)
    block_lines = {u.line for u in layout.units[located.start : located.end]}
    box, height = geo.bbox, geo.glyph_height

    above = [
        box.top - ln.bbox.bottom
        for ln in layout.lines
        if ln.position not in block_lines
        and ln.bbox.bottom <= box.top
        and ln.bbox.horizontal_overlap_ratio(box) > 0
    ]
    below = [
        ln.bbox.top - box.bottom
        for ln in layout.lines
        if ln.position not in block_lines
        and ln.bbox.top >= box.bottom
        and ln.bbox.horizontal_overlap_ratio(box) > 0
    ]
    beside = [
        max(box.left - ln.bbox.right, ln.bbox.left - box.right)
        for ln in layout.lines
        if ln.position not in block_lines and ln.bbox.vertical_gap(box) == 0
    ]
    shared = [
        u.raw
        for i, u in enumerate(layout.units)
        if u.line in block_lines and not located.start <= i < located.end
    ]

    measurement: dict[str, float] = {"line_height_px": round(height, 2)}
    parts: list[str] = []
    for name, gaps in (("above", above), ("below", below), ("beside", beside)):
        if gaps:
            gap = max(0.0, min(gaps))
            measurement[f"gap_{name}_px"] = round(gap, 2)
            measurement[f"gap_{name}_line_heights"] = round(gap / height, 2)
            parts.append(f"{gap / height:.1f} line heights {name}")
        else:
            parts.append(f"no text {name}")
    measurement["shares_line_with_other_text"] = 1.0 if shared else 0.0

    summary = "Nearest other text: " + ", ".join(parts) + "."
    if shared:
        summary += (
            f" Other text shares a line with the statement (“{_quote(shared[:6])}”), "
            "so it may not be set apart."
        )
    return CheckResult(
        id=check.id,
        outcome=CheckOutcome.ADVISORY,
        citation=check.citation,
        summary=summary,
        detail=(
            "Estimated from OCR text boxes only. Borders, rules, background panels "
            "and colour changes also separate a statement and are invisible to this "
            "measure, so it is reported for the agent to judge, never adjudicated."
        ),
        measurement=measurement,
        confidence=SEPARATION_CONFIDENCE * (0.75 if shared else 1.0),
    )


def _pixel_check(check: WarningCheck, what: str) -> CheckResult:
    """Bold and contrast need the image, which strategies deliberately never see.

    `StrategyContext` is narrow by design ("widening this is how a
    deterministic layer stops being deterministic"). Reporting these as
    NOT_EVALUABLE with the reason is the honest output until a pixel-measuring
    stage exists; silently skipping them would let "not checked" read as
    "passed".
    """
    return CheckResult(
        id=check.id,
        outcome=CheckOutcome.NOT_EVALUABLE,
        citation=check.citation,
        summary=(
            f"Not checked automatically: {what} requires pixel measurement, which is not "
            "available to the text-based check. Verify visually."
        ),
        detail=" ".join(check.description.split()),
    )


def _type_size(check: WarningCheck) -> CheckResult:
    requires = ", ".join(check.requires) or "label dimensions"
    return CheckResult(
        id=check.id,
        outcome=CheckOutcome.NOT_EVALUABLE,
        citation=check.citation,
        summary=(
            "Not checked automatically: minimum type size depends on the container "
            f"volume and needs the physical label size ({requires}) to convert pixels "
            "to millimetres; neither reaches this check. Verify type size manually."
        ),
        detail=" ".join(check.description.split()),
    )


def _not_located_checks(rule: GovernmentWarningRule) -> tuple[CheckResult, ...]:
    """Every declared check, reported as not evaluated when nothing was found.

    So the result still lists what the tool would have checked, and nothing in
    it can be read as a pass.
    """
    return tuple(
        CheckResult(
            id=check.id,
            outcome=CheckOutcome.NOT_EVALUABLE,
            citation=check.citation,
            summary="Not evaluated: the warning statement was not located on this image.",
        )
        for check in rule.checks
    )


def _build_checks(
    rule: GovernmentWarningRule,
    layout: _Layout,
    located: _Located,
    header: _Header,
    capitalisation: _Capitalisation,
    diff: _Diff,
    findings: list[_Finding],
    similarity: float,
    ocr_confidence: float,
    tuning: WarningTuning,
) -> tuple[CheckResult, ...]:
    """One CheckResult per check the rule set declares, in its order.

    Driven by the YAML, not by a list in code, so that a check added to the
    rule set appears in the result as NOT_EVALUABLE ("no evaluator") rather
    than silently vanishing.
    """
    by_check: dict[str, list[_Finding]] = {}
    for finding in findings:
        by_check.setdefault(finding.check_id, []).append(finding)

    results: list[CheckResult] = []
    for check in rule.checks:
        mine = by_check.get(check.id, [])
        if check.id == _HEADER_CHECK:
            results.append(
                CheckResult(
                    id=check.id,
                    outcome=capitalisation.outcome,
                    citation=check.citation,
                    summary=capitalisation.summary,
                    detail=capitalisation.detail,
                    measurement=(
                        {}
                        if header.raw is None
                        else {"header_ocr_confidence": round(header.confidence, 4)}
                    ),
                    confidence=None if header.raw is None else round(header.confidence, 4),
                )
            )
        elif check.id == _TEXT_CHECK:
            tolerated = "; ".join(f"“{e}” read as “{f}”" for e, f in diff.tolerated)
            if mine:
                summary = (
                    f"{len(mine)} difference{'s' if len(mine) != 1 else ''} from the "
                    f"statutory text: " + "; ".join(f.message for f in mine[:3])
                )
                if len(mine) > 3:
                    summary += f"; and {len(mine) - 3} more"
                summary += "."
            else:
                summary = "Statement matches the statutory text word for word."
            results.append(
                CheckResult(
                    id=check.id,
                    outcome=CheckOutcome.FAIL if mine else CheckOutcome.PASS,
                    citation=check.citation,
                    summary=summary,
                    detail=(
                        f"Tolerated as OCR noise (single-character misreads of long words): "
                        f"{tolerated}."
                        if tolerated
                        else None
                    ),
                    measurement={
                        "similarity": round(similarity, 4),
                        "differences": float(len(diff.differences)),
                        "tolerated_ocr_variations": float(len(diff.tolerated)),
                        "clean_read_fraction": round(diff.precision, 4),
                    },
                    confidence=round(ocr_confidence, 4),
                )
            )
        elif check.id == "header_bold":
            results.append(_pixel_check(check, "bold detection (stroke width)"))
        elif check.id == "contrast":
            ratio = f" (minimum {check.min_contrast_ratio}:1)" if check.min_contrast_ratio else ""
            results.append(_pixel_check(check, f"text-to-background contrast{ratio}"))
        elif check.id == "separation":
            results.append(_separation(check, layout, located, tuning))
        elif check.id == "type_size":
            results.append(_type_size(check))
        else:
            results.append(
                CheckResult(
                    id=check.id,
                    outcome=CheckOutcome.NOT_EVALUABLE,
                    citation=check.citation,
                    summary=f"This prototype has no evaluator for the {check.id!r} check. "
                    "Verify manually.",
                )
            )
    return tuple(results)


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


def warning_strategy(context: StrategyContext) -> FieldResult:
    """Locate and check the government health warning statement.

    Verdict rules, in order:

    *   **Not located** — `not_found_result`: REVIEW, never MISMATCH. The engine
        elevates it when image quality is good. Every declared check is listed
        as not evaluated.
    *   **Low OCR confidence** over the block — REVIEW, whatever the findings.
        A finding against text the engine is unsure it read is not evidence.
    *   **An assertable binding finding** — MISMATCH. Assertable means the
        tool has positive evidence the label is at fault: a title-case header
        with unambiguous lowercase letters read confidently; a different word
        read confidently; words missing where the surrounding geometry leaves
        no room for them to have been missed. The block must also have read
        cleanly overall (`tuning.warning.clean_read_min_precision`) — if most of what was
        read is not statutory text, "differences" are as likely the engine's.
        This includes the **header missing, body intact** case: the header is
        mandatory, and when the body reads cleanly and the line above sits at
        ordinary leading there is nowhere for it to be. Where the geometry
        leaves room (a wide gap above, an indented first line, no text above
        at all) the same finding is REVIEW.
    *   **Only non-assertable findings** — REVIEW, with each one named.
    *   **No findings** — `verdict_for` on the similarity: MATCH at or above
        the field's threshold, otherwise REVIEW.

    Advisory and NOT_EVALUABLE checks never enter this logic.
    """
    rule = context.ruleset.government_warning
    statute = _statute(rule.statutory_text)
    min_conf = context.ruleset.defaults.min_ocr_confidence
    tuning = context.ruleset.tuning.warning

    layout = _reconstruct(context.spans)
    score = _Scorer(layout, statute, tuning)
    located = _locate(layout, rule, score)

    if located is None:
        return not_found_result(context).model_copy(
            update={"expected": rule.statutory_text, "checks": _not_located_checks(rule)}
        )

    block = layout.units[located.start : located.end]
    similarity = score(located.start, located.end)
    ocr_confidence = statistics.fmean(u.confidence for u in block)
    bbox = BBox.union([u.bbox for u in block])

    header = _read_header(layout, located)
    diff = _diff(layout, located, statute, min_conf, tuning)

    findings: list[_Finding] = []
    capitalisation = _capitalisation(header, min_conf, _degraded_image(context), tuning)
    if capitalisation.finding is not None:
        findings.append(capitalisation.finding)
    if header.has_colon is False:
        findings.append(
            _Finding(
                _TEXT_CHECK,
                "no colon was read after “GOVERNMENT WARNING” (27 CFR 16.21 prescribes "
                "“GOVERNMENT WARNING:”) — small punctuation is the least reliable part of "
                "an OCR read, so confirm visually",
                assertable=False,
            )
        )
    findings.extend(diff.findings)

    clean_read = diff.precision >= tuning.clean_read_min_precision
    if ocr_confidence < min_conf:
        verdict = Verdict.REVIEW
    elif findings:
        assertable = any(
            f.assertable and (f.check_id == _HEADER_CHECK or clean_read) for f in findings
        )
        verdict = Verdict.MISMATCH if assertable else Verdict.REVIEW
    else:
        verdict = verdict_for(similarity, ocr_confidence, context.field_rule.id, context.ruleset)
        if verdict is Verdict.MISMATCH:
            # Nothing specific is wrong, so there is nothing to assert.
            verdict = Verdict.REVIEW

    return FieldResult(
        field_id=context.field_rule.id,
        label=context.field_rule.label,
        verdict=verdict,
        strategy=Strategy.ANCHORED,
        expected=rule.statutory_text,
        found=" ".join(u.raw for u in block),
        bbox=BoxModel(**bbox.to_dict()),
        score=round(max(0.0, min(1.0, similarity)), 4),
        ocr_confidence=round(ocr_confidence, 4),
        reason=_explain(verdict, findings, ocr_confidence, clean_read, min_conf),
        citation=context.field_rule.citation,
        citation_url=context.field_rule.citation_url,
        checks=_build_checks(
            rule,
            layout,
            located,
            header,
            capitalisation,
            diff,
            findings,
            similarity,
            ocr_confidence,
            tuning,
        ),
        differences=diff.differences,
    )


def _explain(
    verdict: Verdict,
    findings: list[_Finding],
    ocr_confidence: float,
    clean_read: bool,
    min_conf: float,
) -> str:
    """Plain-language reason, naming exactly what is wrong.

    Always ends by saying what was *not* checked, because a MATCH on wording
    and capitals is not a MATCH on bold type or type size, and an agent must
    never read it as one.
    """
    unchecked = " Bold type, contrast and type size are not checked automatically."

    def sentence(text: str) -> str:
        return text[0].upper() + text[1:] + "."

    listed = " ".join(sentence(f.message) for f in findings[:3])
    if len(findings) > 3:
        listed += f" {len(findings) - 3} further difference(s) are listed below."

    if ocr_confidence < min_conf:
        lead = (
            f"Located the warning statement but read it with low confidence "
            f"({ocr_confidence:.0%}); confirm it visually."
        )
        return f"{lead} {listed}".strip() + unchecked

    if verdict is Verdict.MATCH:
        return (
            "The statement matches 27 CFR 16.21 word for word and “GOVERNMENT WARNING” "
            "is in capital letters." + unchecked
        )

    if not findings:
        return (
            "Located the warning statement, but parts of it read imperfectly; confirm "
            "the wording visually." + unchecked
        )

    if verdict is Verdict.REVIEW and not clean_read:
        listed += (
            " Much of the text in this region did not read as the statutory wording, "
            "so these may be OCR errors rather than label errors."
        )
    return listed + unchecked


register_strategy("warning", warning_strategy)
