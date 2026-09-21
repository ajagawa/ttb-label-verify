"""Geometry primitives shared across extraction, assignment and verification.

These are deliberately plain and dependency-free. They sit at the boundary
between the probabilistic half of the system (extraction) and the deterministic
half (verification), and both halves plus the frontend serialise them, so
nothing here should acquire a dependency on an image library or a model runtime.

Coordinates are in **pixels of the preprocessed image**, with the origin at the
top-left. The preprocessed image is what the frontend displays, so boxes can be
overlaid directly without a coordinate transform. If preprocessing rotates or
deskews an image, the boxes describe the corrected image, not the original
upload — this is why the API returns the preprocessed image alongside the
result rather than echoing the upload back.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned bounding box in preprocessed-image pixel space.

    Axis-aligned rather than a quadrilateral: OCR engines return four corner
    points for rotated text, but every consumer here (overlay rendering,
    masking, spatial-proximity tests) wants an axis-aligned rectangle. The
    engine's quad is reduced to its bounding rectangle at the provider boundary.
    The cost is a slightly loose box on rotated text, which is acceptable for
    a highlight overlay and irrelevant for the proximity tests.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(
                f"BBox is inverted: ({self.left}, {self.top}) -> ({self.right}, {self.bottom})"
            )

    # -- derived geometry ---------------------------------------------------

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    # -- combination --------------------------------------------------------

    @classmethod
    def union(cls, boxes: list[BBox]) -> BBox:
        """Smallest box containing all of `boxes`.

        Used to give a multi-word or multi-line span a single region for the
        overlay, and to give the warning statement a region for the
        typographic checks.
        """
        if not boxes:
            raise ValueError("union of no boxes is undefined")
        return cls(
            left=min(b.left for b in boxes),
            top=min(b.top for b in boxes),
            right=max(b.right for b in boxes),
            bottom=max(b.bottom for b in boxes),
        )

    # -- relationships ------------------------------------------------------

    def intersection_area(self, other: BBox) -> float:
        overlap_w = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        overlap_h = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return overlap_w * overlap_h

    def overlaps(self, other: BBox, min_fraction: float = 0.5) -> bool:
        """True when `min_fraction` or more of *this* box lies inside `other`.

        Asymmetric on purpose. This is the masking test from
        docs/field-assignment.md stage 3: once a field is resolved, spans that
        sit mostly inside its region are removed from the candidate pool. The
        question is always "is this candidate inside the mask", never "do these
        two regions touch", so a symmetric IoU would be the wrong measure — a
        small span fully inside a large mask has a low IoU but should certainly
        be excluded.
        """
        if self.area <= 0:
            return False
        return (self.intersection_area(other) / self.area) >= min_fraction

    def horizontal_overlap_ratio(self, other: BBox) -> float:
        """Fraction of the narrower box's width that overlaps horizontally.

        Used by the multi-line span coherence test: stacked lines (a brand name
        wrapping onto a second line) overlap heavily in x, whereas two columns
        of text side by side do not.
        """
        overlap = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        narrower = min(self.width, other.width)
        return overlap / narrower if narrower > 0 else 0.0

    def vertical_gap(self, other: BBox) -> float:
        """Vertical whitespace between two boxes. Zero when they overlap in y."""
        if self.bottom <= other.top:
            return other.top - self.bottom
        if other.bottom <= self.top:
            return self.top - other.bottom
        return 0.0

    def expanded(self, margin: float) -> BBox:
        """This box grown by `margin` on every side. Used for proximity tests."""
        return BBox(
            left=self.left - margin,
            top=self.top - margin,
            right=self.right + margin,
            bottom=self.bottom + margin,
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Wire format for the frontend overlay.

        Emits left/top/width/height rather than the corner pair because that is
        what CSS absolute positioning wants, and doing the conversion once here
        is better than doing it in every component that draws a box.
        """
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
