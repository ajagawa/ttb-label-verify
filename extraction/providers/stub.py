"""Deterministic extraction provider — no model, no weights, no network.

This exists for three reasons, and only the first is obvious:

1.  **The pipeline runs without an OCR engine installed.** A clean checkout can
    run `make test` and start the API immediately. For a reviewer whose first
    action is to clone and run something, that matters more than any feature.

2.  **It separates rules bugs from OCR bugs.** When a fixture carries a sidecar
    describing exactly what text was drawn and where, this provider replays it
    perfectly. A verification failure against a replayed fixture is
    unambiguously a bug in assignment or comparison, never a misread. Debugging
    the rules engine through a real OCR engine means never being sure which
    half is wrong.

3.  **It proves the provider seam is real.** An interface with one
    implementation is an assertion; an interface with two is a fact. The claim
    in TRADEOFFS.md that extraction could be swapped for an in-tenant cloud
    model is only credible if swapping has actually been done once.

The sidecar convention: `tools/generate_fixtures.py` writes `<label>.ocr.json`
next to each generated image, recording the text, geometry and a synthetic
confidence for every word it drew. That file is ground truth, not a recording
of a previous OCR run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from extraction.geometry import BBox
from extraction.providers.base import (
    ExtractionError,
    ExtractionProvider,
    ExtractionResult,
    Line,
    QualityAssessment,
    Word,
)

#: Suffix for the ground-truth sidecar written beside a generated fixture.
SIDECAR_SUFFIX = ".ocr.json"


class StubProvider(ExtractionProvider):
    """Replays known text rather than recognising it.

    Two modes:

    *   **Sidecar** — constructed with a path, reads `<image>.ocr.json`. Used by
        the fixture-driven tests and by `make evaluate --provider stub`.
    *   **Injected** — constructed with explicit lines. Used by unit tests that
        want a specific layout without touching the filesystem.

    A stub given neither raises rather than inventing a label. Silently
    returning a plausible default would make a wiring mistake look like a
    passing test.
    """

    name = "stub"

    def __init__(
        self,
        *,
        lines: tuple[Line, ...] | None = None,
        sidecar_path: Path | None = None,
        quality: QualityAssessment | None = None,
    ) -> None:
        if lines is None and sidecar_path is None:
            raise ValueError(
                "StubProvider needs either explicit lines or a sidecar path. "
                "It will not invent a label: a stub that returns a plausible default "
                "makes a wiring mistake look like a passing test."
            )
        self._lines = lines
        self._sidecar_path = sidecar_path
        self._quality = quality or QualityAssessment(
            usable=True,
            score=1.0,
            guidance=("Synthetic ground truth; no image quality assessment performed.",),
        )

    # -- construction helpers ----------------------------------------------

    @classmethod
    def for_image(cls, image_path: str | Path) -> StubProvider:
        """Build a provider that replays the sidecar beside `image_path`.

        Args:
            image_path: Path to a fixture image. The sidecar is expected at the
                same path with `.ocr.json` appended to the stem.

        Returns:
            A provider that will replay the recorded geometry.

        Raises:
            ExtractionError: when no sidecar exists. Distinguished from an
                empty result on purpose — a missing sidecar is a setup error,
                not a label with no text on it.
        """
        image_path = Path(image_path)
        sidecar = image_path.with_suffix(image_path.suffix + SIDECAR_SUFFIX)
        if not sidecar.exists():
            raise ExtractionError(
                f"no ground-truth sidecar at {sidecar}. Generate the fixture corpus with "
                "`make fixtures`, or use a real extraction provider for this image."
            )
        return cls(sidecar_path=sidecar)

    # -- provider interface -------------------------------------------------

    def is_available(self) -> bool:
        """Always true. That is the point of the stub."""
        return True

    def extract(self, image: Any) -> ExtractionResult:
        """Return the known text for this image.

        `image` is accepted for interface compatibility and used only for its
        dimensions when they are not recorded in the sidecar. The stub does not
        look at pixels.
        """
        started = time.perf_counter()

        if self._lines is not None:
            lines = self._lines
            width, height = self._dimensions_from(image, lines)
            quality = self._quality
        else:
            lines, width, height, quality = self._load_sidecar()

        return ExtractionResult(
            lines=lines,
            image_width=width,
            image_height=height,
            quality=quality,
            provider_name=self.name,
            extraction_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- internals ----------------------------------------------------------

    def _load_sidecar(self) -> tuple[tuple[Line, ...], int, int, QualityAssessment]:
        assert self._sidecar_path is not None

        try:
            payload = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExtractionError(f"could not read sidecar {self._sidecar_path}: {exc}") from exc

        lines = tuple(
            _line_from_payload(index, entry) for index, entry in enumerate(payload.get("lines", []))
        )

        quality_payload = payload.get("quality")
        quality = (
            QualityAssessment(
                usable=quality_payload.get("usable", True),
                score=quality_payload.get("score", 1.0),
                issues=tuple(quality_payload.get("issues", ())),
                guidance=tuple(quality_payload.get("guidance", ())),
                metrics=quality_payload.get("metrics", {}),
            )
            if quality_payload
            else self._quality
        )

        return lines, int(payload.get("width", 1000)), int(payload.get("height", 1400)), quality

    @staticmethod
    def _dimensions_from(image: Any, lines: tuple[Line, ...]) -> tuple[int, int]:
        """Image dimensions, preferring the array's shape and falling back to content.

        The fallback matters for unit tests, which pass `None` as the image and
        care only about the text layout. Deriving a bounding size from the
        lines keeps any downstream normalisation by image size sensible rather
        than dividing by a made-up constant.
        """
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])

        if not lines:
            return 1000, 1400

        extent = BBox.union([line.bbox for line in lines])
        return int(extent.right * 1.1) or 1000, int(extent.bottom * 1.1) or 1400


def _line_from_payload(index: int, entry: dict) -> Line:
    """Rebuild a Line from its sidecar representation."""
    words = tuple(
        Word(
            text=word["text"],
            bbox=BBox(
                left=float(word["left"]),
                top=float(word["top"]),
                right=float(word["left"]) + float(word["width"]),
                bottom=float(word["top"]) + float(word["height"]),
            ),
            confidence=float(word.get("confidence", 0.98)),
        )
        for word in entry["words"]
    )

    if not words:
        raise ExtractionError(f"sidecar line {index} has no words")

    return Line(index=index, words=words, bbox=BBox.union([w.bbox for w in words]))
