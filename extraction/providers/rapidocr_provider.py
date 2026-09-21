"""RapidOCR extraction provider — the default.

RapidOCR ships the PP-OCR models as ONNX and runs them on onnxruntime, which
avoids pulling the full PaddlePaddle framework: same weights, same Apache-2.0
terms, roughly an order of magnitude smaller container image. Both are
Apache-2.0 including the weights themselves — see THIRD-PARTY-NOTICES.md,
Note 2, for the provenance trace and its one honest caveat.

This module is **not verified against real images in this environment.** The
package registry is reachable here but the model weight CDN is not, so the
adapter logic below is written against RapidOCR's documented output shape and
must be validated on a machine that can download weights. The stub provider and
the fixture sidecars exist partly so that everything downstream of this file
could be built and tested regardless.

Two behaviours are worth knowing before reading the code:

*   RapidOCR returns a list of ``(box, text, confidence)`` triples, where
    ``box`` is four corner points. It does **not** return word-level boxes —
    the unit is a detected text line. Word geometry is therefore interpolated;
    see `_split_line_into_words`.

*   It returns detections in its own order, which is not reliably reading
    order. Ordering is imposed downstream in `extraction.spans`, so nothing
    here needs to sort.
"""

from __future__ import annotations

import time
from typing import Any

from extraction.geometry import BBox
from extraction.providers.base import (
    ExtractionError,
    ExtractionProvider,
    ExtractionResult,
    Line,
    Word,
)

#: Detections below this confidence are dropped entirely rather than passed on
#: as low-confidence candidates. They are overwhelmingly noise — texture picked
#: up from foil, bottle edges, background — and each one adds spans to the
#: candidate pool that can only produce spurious matches.
MIN_DETECTION_CONFIDENCE = 0.30


class RapidOcrProvider(ExtractionProvider):
    """PP-OCR detection and recognition via ONNX Runtime, entirely in-process.

    The engine is constructed once and reused: model load is the expensive part
    and paying it per request would put the 5-second target out of reach.
    """

    name = "rapidocr"

    def __init__(
        self,
        *,
        min_confidence: float = MIN_DETECTION_CONFIDENCE,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """Load the engine.

        Args:
            min_confidence: Detections below this are dropped.
            engine_options: Keyword arguments passed through to RapidOCR's
                constructor, e.g. ``{"intra_op_num_threads": 1}``. None — the
                single-label default — constructs RapidOCR exactly as before,
                with its own defaults. The batch runner uses this to give its
                private engine a capped ONNX thread count (see api/jobs.py);
                options change how inference is scheduled, never the model or
                the thresholds, so output for a given image is unchanged.
        """
        self._min_confidence = min_confidence
        self._engine: Any = None
        self._import_error: Exception | None = None

        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR(**(engine_options or {}))
        except Exception as exc:  # noqa: BLE001 - reported through is_available()
            # Swallowed deliberately: a missing OCR engine must be reportable
            # as "not available" at startup, not as an import crash that stops
            # the rules engine and its tests from loading at all.
            self._import_error = exc

    def is_available(self) -> bool:
        return self._engine is not None

    def unavailable_reason(self) -> str | None:
        return None if self._engine is not None else str(self._import_error)

    def extract(self, image: Any) -> ExtractionResult:
        """Recognise text in a preprocessed RGB image.

        Args:
            image: numpy array, shape (H, W, 3), RGB.

        Returns:
            An `ExtractionResult` whose coordinates refer to `image` as given.

        Raises:
            ExtractionError: when the engine is unavailable or fails. Callers
                surface this as a pipeline error, never as a compliance
                finding — see docs/field-assignment.md, "Not-found is never a
                mismatch".
        """
        if self._engine is None:
            raise ExtractionError(
                f"RapidOCR is not available: {self._import_error}. Install with `make install-ocr`."
            )

        started = time.perf_counter()

        try:
            detections, _elapsed = self._engine(image)
        except Exception as exc:  # noqa: BLE001 - normalised to ExtractionError
            raise ExtractionError(f"RapidOCR failed: {exc}") from exc

        lines = self._to_lines(detections or [])
        height, width = (image.shape[0], image.shape[1]) if hasattr(image, "shape") else (0, 0)

        return ExtractionResult(
            lines=lines,
            image_width=int(width),
            image_height=int(height),
            # No quality assessment: the gate runs before extraction and the
            # pipeline attaches its verdict afterwards. Returning a placeholder
            # here would be indistinguishable from judging the image perfect.
            provider_name=self.name,
            extraction_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- adapters -----------------------------------------------------------

    def _to_lines(self, detections: list) -> tuple[Line, ...]:
        lines: list[Line] = []

        for index, detection in enumerate(detections):
            try:
                quad, text, confidence = detection[0], detection[1], float(detection[2])
            except (IndexError, TypeError, ValueError):
                continue

            if not text or not text.strip():
                continue
            if confidence < self._min_confidence:
                continue

            bbox = _quad_to_bbox(quad)
            if bbox is None or bbox.area <= 0:
                continue

            words = _split_line_into_words(text.strip(), bbox, confidence)
            if not words:
                continue

            lines.append(Line(index=index, words=words, bbox=BBox.union([w.bbox for w in words])))

        return tuple(lines)


def _quad_to_bbox(quad: Any) -> BBox | None:
    """Reduce four corner points to an axis-aligned bounding rectangle.

    Rotated text produces a genuinely rotated quad, and the rectangle around it
    is looser than the text. That is acceptable: every consumer (overlay
    rendering, masking, proximity tests) wants an axis-aligned box, and the
    looseness costs a slightly generous highlight rather than a wrong verdict.
    Deskewing before extraction keeps the rotation small in the normal case.
    """
    try:
        xs = [float(point[0]) for point in quad]
        ys = [float(point[1]) for point in quad]
    except (TypeError, ValueError, IndexError):
        return None

    if not xs or not ys:
        return None

    return BBox(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


def _split_line_into_words(text: str, line_bbox: BBox, confidence: float) -> tuple[Word, ...]:
    """Interpolate per-word boxes across a line box.

    PP-OCR recognises a whole detected line at once and returns one box for it,
    but the span index needs word-level geometry to generate sub-line
    candidates — which is what recovers "750 mL" from a line that also carries
    the ABV.

    Widths are apportioned by character count. This assumes roughly monospaced
    advance, which proportional type violates: a word of narrow letters gets a
    box slightly wider than the ink, a word of wide letters slightly narrower.

    That inaccuracy is tolerable because of what the boxes are used for. The
    overlay highlight is approximate by nature, and the spatial tests
    (horizontal overlap, vertical gap, glyph-height ratio) operate on ratios
    that interpolation error does not meaningfully shift. It would *not* be
    tolerable if a box drove a compliance judgement directly — which is why the
    type-size check requires physical label dimensions and reports
    NOT_EVALUABLE without them, rather than measuring an interpolated box.

    Args:
        text: The recognised line.
        line_bbox: The engine's box for the whole line.
        confidence: The engine's confidence for the line, applied to each word.

    Returns:
        Word-level boxes tiling `line_bbox` left to right.
    """
    tokens = text.split()
    if not tokens:
        return ()

    if len(tokens) == 1:
        return (Word(text=tokens[0], bbox=line_bbox, confidence=confidence),)

    # One space of advance between tokens, same width as a character.
    total_units = sum(len(token) for token in tokens) + (len(tokens) - 1)
    unit_width = line_bbox.width / total_units if total_units else 0.0

    words: list[Word] = []
    cursor = line_bbox.left

    for position, token in enumerate(tokens):
        width = len(token) * unit_width
        words.append(
            Word(
                text=token,
                bbox=BBox(
                    left=cursor,
                    top=line_bbox.top,
                    right=min(cursor + width, line_bbox.right),
                    bottom=line_bbox.bottom,
                ),
                confidence=confidence,
            )
        )
        cursor += width + unit_width
        if position == len(tokens) - 1:
            break

    return tuple(words)
