"""Extraction provider registry.

Selection is by name, resolved at startup from `LABEL_VERIFY_PROVIDER`. The
registry exists so that adding a provider — including a cloud one, if TTB's
network boundary can accommodate an in-tenant model — touches one dict and
nothing in the verification engine.

Providers are constructed lazily. Importing this module must not load model
weights: `make test` runs with no OCR engine installed and must not pay for one.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from extraction.providers.base import (
    ExtractionError,
    ExtractionProvider,
    ExtractionResult,
    Line,
    QualityAssessment,
    Word,
)

__all__ = [
    "ExtractionError",
    "ExtractionProvider",
    "ExtractionResult",
    "Line",
    "QualityAssessment",
    "Word",
    "get_provider",
    "available_providers",
    "DEFAULT_PROVIDER",
]

DEFAULT_PROVIDER = "rapidocr"


def _build_rapidocr() -> ExtractionProvider:
    from extraction.providers.rapidocr_provider import RapidOcrProvider

    return RapidOcrProvider()


def _build_stub() -> ExtractionProvider:
    from extraction.providers.stub import StubProvider

    raise ExtractionError(
        "the stub provider replays known text and cannot be selected globally; "
        "construct it per image with StubProvider.for_image(path) or with explicit lines. "
        f"({StubProvider.__name__} is available for direct use.)"
    )


#: Name -> factory. Factories are called once, at startup.
_FACTORIES: dict[str, Callable[[], ExtractionProvider]] = {
    "rapidocr": _build_rapidocr,
    "stub": _build_stub,
}


def available_providers() -> list[str]:
    return sorted(_FACTORIES)


def get_provider(name: str | None = None) -> ExtractionProvider:
    """Construct the named extraction provider.

    Args:
        name: Provider name. Defaults to `LABEL_VERIFY_PROVIDER`, then to
            `DEFAULT_PROVIDER`.

    Returns:
        A constructed provider.

    Raises:
        ExtractionError: when the name is unknown, or when the provider cannot
            run (usually a missing OCR engine). Raised at startup rather than
            on the first upload, so a deployment problem surfaces as a boot
            failure rather than as a user-visible error.
    """
    resolved = name or os.environ.get("LABEL_VERIFY_PROVIDER") or DEFAULT_PROVIDER

    try:
        factory = _FACTORIES[resolved]
    except KeyError:
        raise ExtractionError(
            f"unknown extraction provider {resolved!r}. Available: {available_providers()}"
        ) from None

    provider = factory()

    if not provider.is_available():
        raise ExtractionError(
            f"extraction provider {resolved!r} is not available. "
            "Install the OCR engine with `make install-ocr`, or run against the "
            "fixture corpus with the stub provider."
        )

    return provider
