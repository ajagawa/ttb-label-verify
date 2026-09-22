"""Image preprocessing and the quality gate.

Runs before extraction. Two jobs:

1.  **Correct what can be corrected** — EXIF orientation, deskew, contrast
    normalisation, downscale. A phone photograph taken at an angle under office
    lighting is the normal case, not the exception.

2.  **Say what is wrong when it cannot be corrected.** This is the part that
    matters operationally. The discovery notes described agents rejecting
    unreadable images and asking the applicant for a better one — a round trip
    measured in days. An agent told "glare obscures the lower-right portion of
    the label; reshoot without direct flash" resolves it once.

The downscale is not cosmetic. On a government network the upload leg dominates
wall-clock time for a 12-megapixel phone photograph, and OCR gains nothing from
resolution beyond what the text needs. It is the single largest contributor to
staying inside the five-second budget.

**Status.** Decode, EXIF orientation, rescaling and the quality gate are
implemented. Deskew, glare detection and blur estimation are not: their
thresholds need calibrating against real photographs, and a detector tuned by
intuition produces confident wrong guidance, which is worse than none.
"""

from __future__ import annotations

import io
import threading
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from extraction.providers.base import ExtractionError, ExtractionResult, QualityAssessment

#: Longest edge the detector sees. Images are resized to this in **both**
#: directions — small ones up as well as large ones down.
#:
#: This number was measured, not chosen. The first version was 1600 with an
#: explicit "never upscale" rule, on the reasoning that interpolating a small
#: image adds no information. That reasoning is right about information and
#: wrong about detection: PP-OCR's text detector works on absolute glyph size,
#: and the warning statement is the smallest, most tightly leaded text on any
#: label. At native size the detector silently dropped whole lines of it.
#:
#: Measured line recall across the fixture corpus, with mean latency:
#:
#:     native   88%   1.1 s
#:     1600     93%   1.3 s
#:     2000     94%   2.0 s
#:     2600    100%   1.9 s
#:
#: A dropped line is not a soft failure. It reaches the agent as "the words
#: 'Consumption of alcoholic beverages impairs your ability to drive a car or
#: operate' are missing" — the tool asserting a violation of 27 CFR 16.21
#: because it could not see the text. That is the worst failure this system
#: has, and it was invisible until the real engine ran against real pixels.
TARGET_LONGEST_EDGE = 2600

#: Ceiling on upscaling. Beyond roughly this factor the interpolation stops
#: helping the detector and only costs inference time, and an image that small
#: has already failed `MIN_USABLE_EDGE`.
MAX_UPSCALE_FACTOR = 3.0

#: Below this, an image is too small for the warning statement to be readable
#: at any processing setting, and saying so immediately beats a result full of
#: low-confidence REVIEW verdicts.
MIN_USABLE_EDGE = 400

#: Upload size at which resolution stops limiting what can be read. Kept
#: separate from TARGET_LONGEST_EDGE: that one is a detector input size and may
#: be tuned for the engine, whereas this is a statement about the photograph
#: the agent was given.
RESOLUTION_REFERENCE_EDGE = 1600

#: Ink-to-paper separation, in luminance levels, at which legibility stops
#: improving. Clean labels in the fixture corpus measure 138-192 regardless of
#: how much text they carry.
INK_SEPARATION_REFERENCE = 170.0

#: Quality score below which the pipeline stops rather than extracting.
USABILITY_FLOOR = 0.35

# ---------------------------------------------------------------------------
# Decoding limits — security, not image quality
# ---------------------------------------------------------------------------
#
# Found in a security review. The upload limit is on *compressed* bytes, and
# an image's memory cost is its pixel count: a 0.7 MB PNG declaring
# 13300x13300 decoded to about 1.4 GB, because Pillow only warns (does not
# refuse) between 89 and 179 megapixels, and the decode, the EXIF-rotated copy
# and the RGB copy all existed at once. Two such uploads at the same time ran a
# 2 GB instance out of memory.

#: Formats accepted, matched against the file's actual content — not the
#: Content-Type the client declared. Pillow otherwise sniffs and decodes about
#: forty formats (GIF, TGA, PCX, SGI, DDS, ...), a large and historically
#: CVE-prone parser surface nobody needs to photograph a label.
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "TIFF", "BMP")

#: Refuse images larger than this *before* decoding them. 40 megapixels is
#: above any ordinary phone photograph (12-24 MP); larger JPEGs, such as a
#: 48 MP phone mode, are first reduced by the JPEG decoder itself (`draft`),
#: which costs nothing in legibility since everything is resized to
#: TARGET_LONGEST_EDGE anyway.
MAX_SOURCE_PIXELS = 40_000_000

#: Backstop inside Pillow, applied when the header is read: over this, and
#: the file is refused before any pixel is decoded. Set above
#: MAX_SOURCE_PIXELS so that a large JPEG gets as far as `draft` (which can
#: shrink it); the exact limit is checked just after.
Image.MAX_IMAGE_PIXELS = int(2.5 * MAX_SOURCE_PIXELS)

#: At most this many images are decoded at once, across single-label checks
#: and batch workers, so the worst case is bounded at roughly this many times
#: the largest allowed decode instead of growing with concurrent requests.
MAX_CONCURRENT_DECODES = 2
_DECODE_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_DECODES)

#: What the agent is told when an upload cannot be decoded. Fixed text: the
#: underlying exception can carry internals (object addresses, parser state).
UNDECODABLE_MESSAGE = (
    "could not read the uploaded file as an image. Upload a JPEG, PNG, WebP, TIFF or BMP "
    "photograph of the label."
)


def _too_large_message(width: int | None = None, height: int | None = None) -> str:
    size = f"{width}x{height}px, " if width and height else ""
    return (
        f"image is {size}larger than this service accepts "
        f"({MAX_SOURCE_PIXELS // 1_000_000} megapixels). Resize it and upload again."
    )


def _decode(payload: bytes) -> Image.Image:
    """Decode an upload to RGB with EXIF orientation applied, within the limits."""
    with warnings.catch_warnings():
        # Pillow's decompression-bomb *warning* becomes an error: nothing
        # between the warning and error thresholds is a legitimate label.
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            source = Image.open(io.BytesIO(payload), formats=ALLOWED_FORMATS)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ExtractionError(_too_large_message()) from exc
        except Exception as exc:  # noqa: BLE001 - normalised to ExtractionError
            raise ExtractionError(UNDECODABLE_MESSAGE) from exc

        if source.format == "JPEG":
            # Let libjpeg decode at 1/2, 1/4 or 1/8 scale when the result is
            # still at least the size the detector will be given.
            source.draft("RGB", (TARGET_LONGEST_EDGE, TARGET_LONGEST_EDGE))
        width, height = source.size
        if width * height > MAX_SOURCE_PIXELS:
            raise ExtractionError(_too_large_message(width, height))
        try:
            # Convert before rotating, so no copy is ever made at four
            # channels; convert() carries the EXIF block across. Orientation
            # matters: a phone photograph is frequently stored rotated with
            # the correction in metadata, and every geometric assumption
            # downstream breaks on an unrotated image.
            image = ImageOps.exif_transpose(source.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - normalised to ExtractionError
            raise ExtractionError(UNDECODABLE_MESSAGE) from exc
    return image


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A decoded, corrected image with its quality assessment.

    The assessment is carried separately from the extraction result because it
    is made before extraction and must not be attributed to the provider — a
    provider that appeared to judge image quality would be making a call it is
    not positioned to make.
    """

    image: np.ndarray
    quality: QualityAssessment
    original_width: int
    original_height: int
    scale_factor: float

    def apply_quality(self, extraction: ExtractionResult) -> ExtractionResult:
        """Attach this assessment to an extraction result.

        Providers return a neutral placeholder; this replaces it with the real
        assessment once extraction has run.
        """
        return ExtractionResult(
            lines=extraction.lines,
            image_width=extraction.image_width,
            image_height=extraction.image_height,
            quality=self.quality,
            provider_name=extraction.provider_name,
            extraction_ms=extraction.extraction_ms,
        )


def preprocess(payload: bytes) -> PreparedImage:
    """Decode and prepare an uploaded image for extraction.

    Args:
        payload: Raw uploaded bytes.

    Returns:
        A `PreparedImage` whose coordinates every downstream box refers to.

    Raises:
        ExtractionError: when the bytes cannot be decoded as an image, or when
            quality falls below the usability floor. Both are pipeline errors
            rather than compliance findings.
    """
    with _DECODE_SLOTS:
        return _prepare(_decode(payload))


def _prepare(source: Image.Image) -> PreparedImage:
    original_width, original_height = source.size

    if max(original_width, original_height) < MIN_USABLE_EDGE:
        raise ExtractionError(
            f"image is {original_width}x{original_height}px, too small to read a label "
            f"reliably (minimum {MIN_USABLE_EDGE}px on the longest edge). "
            "Resubmit a higher-resolution photograph."
        )

    prepared, scale = _rescale(source)
    array = np.asarray(prepared, dtype=np.uint8)

    quality = assess_quality(array, source_longest_edge=max(original_width, original_height))
    if not quality.usable:
        raise ExtractionError("image quality is too low to read: " + "; ".join(quality.guidance))

    return PreparedImage(
        image=array,
        quality=quality,
        original_width=original_width,
        original_height=original_height,
        scale_factor=scale,
    )


def _rescale(image: Image.Image) -> tuple[Image.Image, float]:
    """Resize so the longest edge is `TARGET_LONGEST_EDGE`, up or down.

    Downscaling a large photograph is the latency win: on a government network
    a 12-megapixel upload dominates wall-clock time and the detector gains
    nothing from pixels finer than the strokes.

    Upscaling a small one is the accuracy win, which is the counter-intuitive
    half. Interpolation adds no information, so it cannot help *recognition* —
    but detection is a different problem, and PP-OCR's detector misses small,
    tightly leaded lines that it finds reliably once they are physically
    larger. See the measurements on `TARGET_LONGEST_EDGE`.

    Returns the resized image and the factor applied, so that a caller can map
    coordinates back to the upload if it ever needs to. Nothing does today —
    boxes are reported in preprocessed-image space and the frontend scales by
    the reported dimensions — but the factor is cheap to carry and expensive to
    reconstruct.
    """
    longest = max(image.size)
    if longest == 0:
        return image, 1.0

    scale = TARGET_LONGEST_EDGE / longest
    scale = min(scale, MAX_UPSCALE_FACTOR)

    if abs(scale - 1.0) < 0.01:
        return image, 1.0

    target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    # LANCZOS both ways: it is the better downsampling filter, and on upscaling
    # it keeps stroke edges crisper than bilinear, which matters because the
    # detector is looking for stroke boundaries.
    return image.resize(target, Image.Resampling.LANCZOS), scale


def _ink_paper_separation(grayscale: np.ndarray) -> float:
    """Distance between the mean ink level and the mean paper level.

    Otsu's method chooses the split that maximises between-class variance,
    which for a label is the boundary between print and background. The
    distance between the two class means is then a coverage-independent
    measure of legibility: it answers "can the text be told from the paper",
    not "how much text is there".

    Args:
        grayscale: Single-channel luminance, values 0-255.

    Returns:
        Separation in luminance levels, 0 when the image has only one tone.
    """
    flat = grayscale.reshape(-1)
    histogram, _ = np.histogram(flat, bins=256, range=(0, 256))
    total = flat.size
    if total == 0:
        return 0.0

    intensities = np.arange(256, dtype=np.float64)
    sum_all = float(np.dot(intensities, histogram))

    weight_background = 0.0
    sum_background = 0.0
    best_variance = -1.0
    best_threshold = 0

    for level in range(256):
        weight_background += histogram[level]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += level * histogram[level]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > best_variance:
            best_variance, best_threshold = variance, level

    dark = flat[flat <= best_threshold]
    light = flat[flat > best_threshold]
    if dark.size == 0 or light.size == 0:
        return 0.0

    return float(light.mean() - dark.mean())


def assess_quality(
    image: np.ndarray, *, source_longest_edge: int | None = None
) -> QualityAssessment:
    """Score an image and say what is wrong with it.

    Current checks are resolution and global contrast, both cheap and both
    reliable. Glare detection, blur estimation and skew measurement are
    declared in the issue vocabulary below but not implemented — the thresholds
    need the fixture corpus to calibrate against, and a detector tuned by
    intuition would produce confident wrong guidance, which is worse than none.

    Args:
        image: RGB array, shape (H, W, 3).

    Returns:
        A `QualityAssessment`. `usable` False stops the pipeline before
        extraction.
    """
    height, width = image.shape[:2]
    grayscale = image.mean(axis=2)

    # Separation between ink and paper, not spread of luminance.
    #
    # The first version used the standard deviation of luminance, which is
    # wrong in a way that took a real measurement to see: on a label, ~95% of
    # pixels are background, so the statistic is dominated by how MUCH ink is
    # present rather than how well it is separated from the paper. A label with
    # less text scored as a worse photograph. That had a direct consequence —
    # removing the warning block from a test label lowered the very score the
    # engine uses to decide whether a MISSING warning is serious enough to
    # elevate, so the fixture with no warning looked like a bad photo instead.
    #
    # Otsu's threshold splits ink from paper and the distance between the two
    # class means answers the question actually being asked: is the text
    # legible against its background? Measured across the corpus, clean labels
    # score 138-192 whatever their ink coverage, a low-contrast photograph 52,
    # and a noisy/blurred one 119.
    contrast = _ink_paper_separation(grayscale)
    contrast_score = min(1.0, contrast / INK_SEPARATION_REFERENCE)

    # Scored against the ORIGINAL upload. Scoring the rescaled copy would let
    # upscaling inflate the very number that is supposed to warn an agent the
    # photograph was too small to read fine print.
    effective_edge = source_longest_edge if source_longest_edge is not None else max(width, height)
    resolution_score = min(1.0, effective_edge / RESOLUTION_REFERENCE_EDGE)

    score = round(0.6 * contrast_score + 0.4 * resolution_score, 3)

    issues: list[str] = []
    guidance: list[str] = []

    if contrast_score < 0.5:
        issues.append("low_contrast")
        guidance.append(
            "The image looks flat or washed out. Reshoot in even lighting without "
            "direct flash, keeping the whole label in focus."
        )

    if resolution_score < 0.5:
        issues.append("low_resolution")
        guidance.append(
            f"The image is {width}x{height}px. Fine print such as the warning statement "
            "may not be readable; a closer or higher-resolution photograph would help."
        )

    return QualityAssessment(
        usable=score >= USABILITY_FLOOR,
        score=score,
        issues=tuple(issues),
        guidance=tuple(guidance),
        metrics={
            "ink_paper_separation": round(contrast, 2),
            "width": float(width),
            "height": float(height),
        },
    )
