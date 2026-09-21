"""The extraction/verification contract.

This module defines the single typed structure that crosses the boundary
between the probabilistic half of the system and the deterministic half, plus
the interface every extraction backend implements.

Everything upstream of this boundary (OCR, preprocessing, quality assessment)
may be approximate, model-driven and non-reproducible. Everything downstream
(field assignment, comparison, verdicts) is deterministic: given the same
`ExtractionResult`, the rest of the pipeline produces byte-identical output
every time. That property is what makes a verdict auditable, and it is the
reason this boundary is a hard one rather than a convention.

Consequences worth stating, because they constrain implementations:

*   A provider returns text and geometry. It does **not** return field
    assignments, judgements, or anything resembling a verdict. A provider that
    decides "this is the brand name" has moved a compliance judgement into the
    non-reproducible half of the system.

*   A provider must not mutate its input image. Preprocessing happens before
    the provider is called, and the coordinates a provider returns must refer
    to the image it was given.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from extraction.geometry import BBox


@dataclass(frozen=True, slots=True)
class Word:
    """A single recognised token with its location and the engine's confidence.

    `confidence` is in [0, 1]. Engines that report confidence per line rather
    than per word assign the line's confidence to each of its words; the
    difference matters little downstream, where confidence is used to weight
    scores rather than as an absolute measure.
    """

    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class Line:
    """A run of words the engine considered one line of text.

    Line segmentation is the engine's, not ours. We do not re-segment, because
    the span index (see extraction/spans.py) generates both sub-line and
    multi-line candidates, which recovers from segmentation that is too coarse
    or too fine without needing to second-guess the engine.
    """

    index: int
    words: tuple[Word, ...]
    bbox: BBox

    @property
    def text(self) -> str:
        """The line as recognised, words joined with single spaces.

        This is `raw_text`: original casing and punctuation, suitable for
        display and for formatting checks. It is never the thing scores are
        computed against — see extraction/normalize.py for why that
        distinction is load-bearing.
        """
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        """Mean word confidence. Zero for an empty line."""
        if not self.words:
            return 0.0
        return statistics.fmean(w.confidence for w in self.words)

    @property
    def glyph_height(self) -> float:
        """Median word-box height — a proxy for apparent type size.

        Median rather than mean because a line containing one tall glyph (a
        parenthesis, an accented capital) should not be reported as larger than
        it looks. Used for the span coherence test and for the brand-name size
        prior.
        """
        if not self.words:
            return 0.0
        return statistics.median(w.bbox.height for w in self.words)


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """What the quality gate concluded about the submitted image.

    The design intent, from the discovery notes: when an image is unusable, say
    *what is wrong with it* rather than rejecting generically. An agent who is
    told "glare obscures the lower-right portion of the label; reshoot without
    direct flash" fixes it once. An agent who is told "image rejected" starts a
    round trip with the applicant.

    `usable` False short-circuits the pipeline before extraction. `usable` True
    with a non-empty `issues` list is the common case: the image is workable
    but something about it should temper confidence in the result.
    """

    usable: bool
    score: float  # [0, 1], higher is better
    issues: tuple[str, ...] = ()  # machine-readable issue codes
    guidance: tuple[str, ...] = ()  # plain-language, shown to the agent
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything the deterministic half of the system is allowed to know.

    Note what is *absent*: no field names, no expected values, no verdicts. A
    provider has no access to the application record and therefore cannot bias
    its output toward what the record says. That is not an accident of the
    design — it is the reason a mismatch found by this system means something.
    """

    lines: tuple[Line, ...]
    image_width: int
    image_height: int
    provider_name: str
    extraction_ms: float

    #: The preprocessing quality assessment, attached by
    #: `extraction.pipeline.PreparedImage.apply_quality` after extraction.
    #:
    #: Optional, and None until then, because a provider is not positioned to
    #: judge image quality and must not appear to. The first version had
    #: providers return a neutral placeholder — `usable=True, score=1.0` — and
    #: that was a quiet trap: the placeholder is indistinguishable from a
    #: genuine assessment of a perfect photograph. Any consumer gating on it
    #: would read "good image" on precisely the degraded images the gate exists
    #: for. None cannot be misread that way, and the type makes the omission
    #: visible at every use site.
    quality: QualityAssessment | None = None

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(w for line in self.lines for w in line.words)

    @property
    def mean_confidence(self) -> float:
        words = self.words
        if not words:
            return 0.0
        return statistics.fmean(w.confidence for w in words)

    @property
    def median_line_height(self) -> float:
        """Reference scale for the whole image.

        Several heuristics are expressed as multiples of this rather than in
        absolute pixels, so that they hold for a 900px phone photo and a 4000px
        scan alike. Absolute pixel thresholds are a bug waiting for a
        differently sized upload.
        """
        heights = [line.glyph_height for line in self.lines if line.glyph_height > 0]
        if not heights:
            return 0.0
        return statistics.median(heights)


class ExtractionProvider(ABC):
    """Interface for an OCR backend.

    Implementations in this repository:

    *   `extraction.providers.stub.StubProvider` — deterministic, no model.
        Used by the test suite and as a fallback when no engine is installed.
    *   `extraction.providers.rapidocr_provider.RapidOcrProvider` — the default.

    A cloud provider would implement the same interface. That it *could* is the
    point of the seam: if TTB's network boundary can accommodate an in-tenant
    vision model, extraction improves without the verification engine changing
    at all. See TRADEOFFS.md, trade-off 1.
    """

    #: Short stable identifier recorded on every result, so a stored verdict
    #: can be traced to the engine that produced its inputs.
    name: str = "abstract"

    @abstractmethod
    def extract(self, image: object) -> ExtractionResult:
        """Recognise text in a preprocessed image.

        Args:
            image: A preprocessed image as a numpy array (H, W, 3) in RGB.
                Typed loosely to keep this module free of a numpy import at
                definition time; implementations narrow it.

        Returns:
            An `ExtractionResult` whose coordinates refer to `image` as given.

        Raises:
            ExtractionError: when the engine fails. Callers surface this as a
                pipeline error rather than as a compliance finding — a failed
                extraction is never evidence that a label is non-compliant.
                See docs/field-assignment.md, "Not-found is never a mismatch".
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can actually run.

        Checked at startup so that a missing OCR engine produces a clear
        message at boot rather than a stack trace on the first upload.
        """


class ExtractionError(RuntimeError):
    """The extraction engine failed.

    Distinct from "the engine ran and found nothing", which is an ordinary
    result with empty lines and is handled by the assignment layer.
    """
