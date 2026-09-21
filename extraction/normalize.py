"""Text normalisation for scoring.

Every span carries two text forms and they are not interchangeable:

*   ``raw_text``       — exactly as recognised. Displayed to the agent, and the
                         **only** form formatting checks may read.
*   ``canonical_text`` — aggressively normalised. Used for fuzzy scoring and
                         nothing else.

Two traps live here, both from docs/field-assignment.md. Both are the kind that
leave the system apparently working while silently producing wrong compliance
findings, so each has a dedicated guard and a dedicated test.

**Trap 1 — formatting checks must never read canonical text.**
`canonicalize()` uppercases. The regulation at 27 CFR 16.22(a)(2) requires
"GOVERNMENT WARNING" in capital letters; the violation an agent described
catching in the discovery notes was a label using "Government Warning" in title
case. Run that check against canonical text and every label passes, including
the non-compliant ones. The guard is naming: `canonicalize()` is documented as
scoring-only, the capitalisation check in rules/warning.py takes `raw_text`
explicitly, and a test asserts the title-case case is still detected end to end.

**Trap 2 — character folding must never touch numbers.**
`canonicalize()` folds OCR confusion classes (S↔5, O↔0, I↔1). That is helpful
when matching a brand name and dangerous when reading alcohol content: it can
turn a genuine 46%/45% discrepancy into an apparent match, which is precisely
the violation this tool exists to catch. Numeric fields therefore use
`canonicalize_numeric()`, which does no folding at all.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures and typographic substitutes that NFKD does not decompose on its own.
_CHARACTER_SUBSTITUTIONS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",  # single quotes
    "“": '"',
    "”": '"',
    "„": '"',  # double quotes
    "–": "-",
    "—": "-",
    "‒": "-",
    "―": "-",  # dashes
    "…": "...",  # ellipsis
    " ": " ",
    " ": " ",
    " ": " ",  # hard spaces
    "ﬀ": "FF",
    "ﬁ": "FI",
    "ﬂ": "FL",  # ligatures
    "ﬃ": "FFI",
    "ﬄ": "FFL",
    "æ": "AE",
    "Æ": "AE",
    "œ": "OE",
    "Œ": "OE",
    "ß": "SS",
    "•": " ",
    "·": " ",
    "●": " ",  # bullets
}

# OCR confusion classes, folded to a single representative so that a
# misrecognised character does not cost a match. Applied to name-like fields
# only. The direction is arbitrary but must be consistent: both the candidate
# and the expected value pass through the same folding, so folding "0" to "O"
# and folding "O" to "0" are equivalent as long as only one is done.
_OCR_CONFUSION_FOLDING = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "L": "I",
        "5": "S",
        "8": "B",
        "2": "Z",
        "6": "G",
    }
)

# Multi-character confusions ("rn" rendering as "m", "cl" as "d") are
# deliberately NOT folded. Two attempts and both were wrong:
#
#   Folding after uppercasing: "RN" is an ordinary uppercase bigram in English
#   ("GOVERNMENT", "WARNING"), so the rule fired on nearly every label and
#   mangled the anchor phrase it was meant to help match.
#
#   Folding before uppercasing, on lowercase only: this makes the canonical
#   form depend on the input's casing. "GOVERNMENT WARNING" and "Government
#   Warning" then canonicalise differently — the all-caps form has no lowercase
#   "rn" to fold and the title-case form does. Anchoring the warning statement
#   would work on compliant labels and fail on exactly the title-case violation
#   the discovery notes describe catching. A normaliser whose output depends on
#   input case is not a normaliser.
#
# The single-character table below runs after uppercasing and is therefore
# case-independent, which is the property that matters. These bigram
# confusions are rare enough on printed labels that the fuzzy comparator
# absorbs them as ordinary edit distance.

# A token made entirely of digits, separators and a decimal point — "750",
# "45.5", "1,000". Character folding is skipped for these; see
# `_fold_confusions`.
_PURE_NUMERIC_TOKEN = re.compile(r"^[\d.,]+$")


def _fold_confusions(text: str) -> str:
    """Fold OCR confusion classes, leaving purely numeric tokens alone.

    The single-character table maps digits onto letters (5->S, 0->O, 1->I),
    which is the right direction for name-like text where letters dominate and
    a stray digit is almost certainly a misrecognition.

    Applied indiscriminately, though, it also rewrites real numbers: "750" would
    become "7SO" and "45.5" would become "4S.S". Both sides of a comparison fold
    identically so matching still works, but the diagnostics become unreadable
    and any downstream code that reaches for a number in canonical text gets
    nonsense. Numeric fields have their own normaliser
    (`canonicalize_numeric`) and never come through here, but brand names and
    class/type designations legitimately contain numbers — "1792 Ridgemont",
    "Old No. 7" — so the guard belongs here too.
    """
    return " ".join(
        token if _PURE_NUMERIC_TOKEN.match(token) else token.translate(_OCR_CONFUSION_FOLDING)
        for token in text.split(" ")
    )


_WHITESPACE = re.compile(r"\s+")

# Apostrophes are DELETED rather than replaced with a space, so that "Stone's"
# and "Stones" converge on "STONES". Replacing them with a space yields
# "STONE S" against "STONES", which the fuzzy comparator would still score
# highly but which needlessly spends tolerance budget on a difference that
# carries no information. Applied after _CHARACTER_SUBSTITUTIONS has folded the
# smart-quote forms onto the straight one.
_APOSTROPHE = re.compile(r"['ʼ’]")

# Every other punctuation mark becomes a space, so "Alc./Vol." separates into
# two tokens. The lookarounds preserve a decimal point sitting between two
# digits, keeping "45.5" intact while stripping the periods from "Alc./Vol.".
_PUNCTUATION = re.compile(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)|(?<=\d)[^\w\s.](?=\d)")


def _apply_substitutions(text: str) -> str:
    for source, replacement in _CHARACTER_SUBSTITUTIONS.items():
        if source in text:
            text = text.replace(source, replacement)
    return text


def fold_confusable(character: str) -> str:
    """The OCR-confusion class representative for one character, uppercased.

    Exposed so that code describing a difference to a person can skip pairs
    that canonicalisation already treats as the same character. Without it, a
    reason could say "0 on the application reads O on the label" about a pair
    the verdict never counted — explaining a difference the tool did not act on.

    Args:
        character: A single character.

    Returns:
        Its folded, uppercased form.

    Examples:
        >>> fold_confusable("0") == fold_confusable("O")
        True
        >>> fold_confusable("M") == fold_confusable("N")
        False
    """
    return character.upper().translate(_OCR_CONFUSION_FOLDING)


def canonicalize(text: str, *, fold_ocr_confusions: bool = True) -> str:
    """Normalise text for fuzzy scoring. **Never** for formatting checks.

    The pipeline, in order:

    1. Unicode NFKD, then explicit substitution of quotes, dashes and ligatures
       that NFKD leaves alone.
    2. Combining marks dropped, so "CREME" and "CRÈME" compare equal.
    3. Uppercase.  <-- this is what makes the result unfit for capitalisation
       checks. See trap 1 in the module docstring.
    4. Apostrophes deleted; other punctuation replaced with a space, except a
       decimal point between digits.
    5. Whitespace collapsed and trimmed.
    6. Single-character confusion classes folded, skipping purely numeric
       tokens, if requested.

    Args:
        text: Raw recognised text, or an expected value from the application
            record. Both sides of a comparison must be canonicalised the same
            way or the scores are meaningless.
        fold_ocr_confusions: Set False to skip step 6 while keeping the rest.
            Useful when comparing two values that are both trusted (two fields
            of the application record, say) and folding adds only noise.

    Returns:
        A normalised string suitable for passing to a fuzzy comparator.

    Examples:
        >>> canonicalize("Stone's Throw")
        'STONES THROW'
        >>> canonicalize("STONE'S THROW")
        'STONES THROW'

        The two forms Dave Morrison described in the discovery notes converge,
        which is the whole point of this function existing.

        >>> canonicalize("750 mL")
        '750 MI'

        The volume survives folding intact; only the unit letters are folded.
    """
    text = unicodedata.normalize("NFKD", text)
    text = _apply_substitutions(text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.upper()
    text = _APOSTROPHE.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()

    if fold_ocr_confusions:
        text = _fold_confusions(text)

    return text


def canonicalize_numeric(text: str) -> str:
    """Normalise text that contains a number, with **no** character folding.

    Digits are left exactly as recognised. A 4 that should have been a 1 stays
    a 4 and surfaces as a discrepancy for the agent to resolve, rather than
    being quietly corrected toward whatever the application record says.

    This is the deliberate asymmetry described in docs/field-assignment.md:
    fuzzy tolerance is correct for names and wrong for numbers, because a
    one-character difference in a name is almost always noise and a
    one-character difference in an ABV is almost always the finding.

    Args:
        text: Raw recognised text expected to contain a numeric value.

    Returns:
        Uppercased, whitespace-collapsed text with digits untouched.

    Examples:
        >>> canonicalize_numeric("45% Alc./Vol. (90 Proof)")
        '45% ALC./VOL. (90 PROOF)'
    """
    text = unicodedata.normalize("NFKD", text)
    text = _apply_substitutions(text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    return _WHITESPACE.sub(" ", text).strip()


def collapse_for_anchor(text: str) -> str:
    """Canonicalise and remove all whitespace.

    For anchoring on a phrase whose internal spacing cannot be trusted —
    "GOVERNMENT WARNING" may arrive as one token, two tokens, or split across
    a line break depending on how the engine segmented it. Removing spaces
    makes the anchor robust to all three.

    Args:
        text: Text to reduce to an anchor key.

    Returns:
        Canonical text with every space removed.

    Examples:
        >>> collapse_for_anchor("GOVERNMENT WARNING")
        'GOVERNMENTWARNING'
        >>> collapse_for_anchor("Government  Warning")
        'GOVERNMENTWARNING'

        Note that the anchor is case-insensitive by construction, so the
        title-case violation still *anchors* here — correctly. It is a
        formatting violation, not a location failure, and it is caught
        downstream by `is_all_caps` reading raw text.
    """
    return canonicalize(text).replace(" ", "")


def is_all_caps(text: str) -> bool:
    """Whether every cased character in `text` is uppercase.

    **Takes raw text.** Passing canonical text here always returns True, which
    is trap 1 in its purest form. Callers in rules/warning.py pass `raw_text`
    and a test asserts the title-case violation is caught end to end.

    Text containing no cased characters returns False rather than True: "is
    every letter uppercase" is not a meaningful claim about a string with no
    letters, and returning True would let an empty or numeric extraction pass
    a capitalisation requirement.

    Args:
        text: Raw recognised text, exactly as the engine reported it.

    Returns:
        True when at least one cased character is present and all are upper.

    Examples:
        >>> is_all_caps("GOVERNMENT WARNING")
        True
        >>> is_all_caps("Government Warning")
        False
        >>> is_all_caps("123")
        False
    """
    cased = [ch for ch in text if ch.isalpha()]
    if not cased:
        return False
    return all(ch.isupper() for ch in cased)


def word_tokens(text: str) -> list[str]:
    """Canonical text split into words, for the word-level warning diff.

    Args:
        text: Text to tokenise.

    Returns:
        Canonical tokens, empty list for empty input.
    """
    canonical = canonicalize(text)
    return canonical.split() if canonical else []
