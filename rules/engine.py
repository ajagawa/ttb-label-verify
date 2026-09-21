"""Verification engine — orchestration only.

Resolves fields in the order the rule set declares, masking each resolved
region out of the candidate pool before the next field runs, and assembles the
results. All comparison logic lives in rules/comparators.py and rules/warning.py;
this module decides *what runs when*, not *what is true*.

The resolution order matters and is not arbitrary. The warning statement goes
first because it is the longest and most distinctive text on the label and
because it contains decoys for two other fields: the phrase "alcoholic
beverages" scores against class/type values containing "alcohol", and its
numerals attract a loose net-contents pattern. Resolving it first and removing
its region eliminates most cross-field contention in one step.

A field whose strategy is unimplemented is reported as an explicit gap rather
than being omitted or silently passed. An incomplete tool that looks complete
is worse than one that says what it did not check.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from collections.abc import Callable

from extraction.geometry import BBox
from extraction.providers.base import ExtractionResult
from extraction.spans import Span, build_span_index, mask_spans
from rules.comparators import (
    STRATEGIES,
    StrategyContext,
    StrategyNotImplemented,
)
from rules.results import (
    ApplicationRecord,
    FieldResult,
    ImageQuality,
    Strategy,
    Verdict,
    VerificationResult,
)
from rules.schema import FieldRule, RuleSet, load_ruleset

#: Fields whose absence from a good-quality image is rejection-grade. They
#: still resolve to REVIEW — the tool never asserts — but at raised prominence.
#: See rules/results.py, `FieldResult.elevated`.
ELEVATED_WHEN_MISSING = frozenset({"government_warning"})


def _expected_value(record: ApplicationRecord, field_id: str) -> str:
    """Pull a field's expected value out of the application record."""
    return {
        "brand_name": record.brand_name,
        "class_type": record.class_type,
        "alcohol_content": record.alcohol_content,
        "net_contents": record.net_contents,
        "government_warning": "",  # fixed by regulation, not by the application
    }.get(field_id, "")


def _unimplemented_result(rule: FieldRule, expected: str, detail: str) -> FieldResult:
    """Report an unwritten strategy as a visible gap.

    Silently omitting the field would leave a result that looks like a complete
    check and is not. REVIEW with an honest reason is the only defensible
    output: the agent is told the tool did not check this, and checks it
    themselves.
    """
    return FieldResult(
        field_id=rule.id,
        label=rule.label,
        verdict=Verdict.REVIEW,
        strategy=Strategy.NOT_FOUND,
        expected=expected or None,
        found=None,
        bbox=None,
        score=0.0,
        ocr_confidence=0.0,
        reason=(
            f"This prototype does not yet check {rule.label.lower()} automatically. "
            "Verify manually."
        ),
        citation=rule.citation,
        citation_url=rule.citation_url,
        elevated=False,
        automated=False,
    )


def register_strategy(name: str, implementation: Callable[[StrategyContext], FieldResult]) -> None:
    """Register a field strategy, replacing any declared placeholder.

    Strategy modules call this at import time. Registration by name, rather
    than each module editing a shared table, is what lets the field strategies
    be developed in parallel without contending for one file.
    """
    STRATEGIES[name] = implementation


#: Modules that register field strategies when imported. Each owns one or more
#: strategies and calls `register_strategy` at import time.
STRATEGY_MODULES = ("rules.numeric", "rules.class_type", "rules.warning")

_strategies_loaded = False


def load_strategies() -> None:
    """Import every strategy module that exists, once.

    A module that is **absent** is skipped: its strategies keep their declared
    placeholders and are reported to the agent as unchecked. A module that is
    present but **fails to import** raises. The distinction matters — swallowing
    an import error would turn a bug in, say, the warning checks into a field
    silently reported as "not yet implemented", which is a false statement
    about the state of the tool.
    """
    global _strategies_loaded
    if _strategies_loaded:
        return

    for module_name in STRATEGY_MODULES:
        if importlib.util.find_spec(module_name) is None:
            continue
        importlib.import_module(module_name)

    _strategies_loaded = True


def verify(
    extraction: ExtractionResult,
    record: ApplicationRecord,
    *,
    ruleset: RuleSet | None = None,
    started_at: float | None = None,
    diagnostics: bool = False,
) -> VerificationResult:
    """Verify one label against its application record.

    Deterministic: the same extraction and the same record always produce the
    same result, which is what makes a verdict auditable.

    Args:
        extraction: Output of an extraction provider.
        record: The application-side values to check against.
        ruleset: Rule set to evaluate under. Defaults to the current version.
        started_at: `time.perf_counter()` reading from before extraction, so
            that the reported elapsed time covers the whole request rather than
            only this call. The displayed number has to be the one the agent
            actually waited through — this is a tool whose predecessor failed
            on latency, and a number that excludes half the pipeline would be
            worse than showing none.
        diagnostics: Attach the verbose per-verification record.

    Returns:
        A `VerificationResult` carrying one `FieldResult` per declared field,
        in display order.
    """
    load_strategies()
    ruleset = ruleset or load_ruleset()
    request_start = started_at if started_at is not None else time.perf_counter()

    beverage = ruleset.beverage_class(record.beverage_class)
    all_spans = build_span_index(extraction.lines)

    # An extraction that never went through preprocessing carries no
    # assessment. That is reported as unassessed rather than defaulted to a
    # good score: `assessed` is what stops a missing measurement from reading
    # as a favourable one.
    assessment = extraction.quality
    quality = (
        ImageQuality(
            usable=assessment.usable,
            score=assessment.score,
            issues=assessment.issues,
            guidance=assessment.guidance,
            measurements=assessment.metrics,
            assessed=True,
        )
        if assessment is not None
        else ImageQuality(
            usable=True,
            score=0.0,
            guidance=("Image quality was not assessed for this result.",),
            assessed=False,
        )
    )

    available: list[Span] = list(all_spans)
    resolved: dict[str, FieldResult] = {}
    trace: list[dict] = []

    for rule in ruleset.resolution_order():
        expected = _expected_value(record, rule.id)
        context = StrategyContext(
            field_rule=rule,
            ruleset=ruleset,
            beverage=beverage,
            spans=available,
            expected=expected,
            anchor_bbox=_anchor_for(rule.id, resolved),
            quality=quality,
        )

        strategy = STRATEGIES.get(rule.strategy)
        if strategy is None:
            result = _unimplemented_result(rule, expected, f"no strategy named {rule.strategy!r}")
        else:
            try:
                result = strategy(context)
            except StrategyNotImplemented as exc:
                result = _unimplemented_result(rule, expected, str(exc))

        result = _elevate_if_mandatory_and_missing(result, rule, quality, ruleset)
        resolved[rule.id] = result

        # Mask the resolved region out of the pool before the next field runs.
        if result.bbox is not None:
            available = mask_spans(available, _to_bbox(result))

        if diagnostics:
            trace.append(
                {
                    "field": rule.id,
                    "strategy": result.strategy.value,
                    "score": result.score,
                    "verdict": result.verdict.value,
                    "candidates_available": len(context.spans),
                    "candidates_remaining": len(available),
                }
            )

    ordered = tuple(resolved[rule.id] for rule in ruleset.display_order())
    elapsed_ms = (time.perf_counter() - request_start) * 1000.0

    return VerificationResult(
        ruleset_version=ruleset.version,
        provider_name=extraction.provider_name,
        quality=quality,
        fields=ordered,
        image_width=extraction.image_width,
        image_height=extraction.image_height,
        elapsed_ms=round(elapsed_ms, 2),
        extraction_ms=round(extraction.extraction_ms, 2),
        diagnostics=(
            {
                "ruleset_version": ruleset.version,
                "total_candidates": len(all_spans),
                "resolution": trace,
            }
            if diagnostics
            else None
        ),
    )


def _anchor_for(field_id: str, resolved: dict[str, FieldResult]) -> BBox | None:
    """Spatial prior for strategies that need one.

    Class/type resolution uses the brand name's position: it sits adjacent to
    the brand, usually directly below and in smaller type. This is the reason
    brand name resolves before class/type in the rule set's ordering.
    """
    if field_id != "class_type":
        return None
    brand = resolved.get("brand_name")
    return _to_bbox(brand) if brand is not None else None


def _to_bbox(result: FieldResult | None) -> BBox | None:
    if result is None or result.bbox is None:
        return None
    box = result.bbox
    return BBox(
        left=box.left,
        top=box.top,
        right=box.left + box.width,
        bottom=box.top + box.height,
    )


def _elevate_if_mandatory_and_missing(
    result: FieldResult, rule: FieldRule, quality: ImageQuality, ruleset: RuleSet
) -> FieldResult:
    """Raise prominence for a mandatory field absent from a good image.

    The verdict stays REVIEW. This is the one case that pulls against
    "not-found is never a mismatch", and it still obeys it: a missing warning
    statement is rejection-grade, but the tool reports that it could not find
    it and lets the agent look, rather than asserting the label is
    non-compliant on the strength of an extraction failure.

    The quality gate is what makes this defensible. On a poor image, a missing
    warning is far more likely to be a failed read than a missing statement,
    so the finding is not elevated.
    """
    if not result.automated:
        # The tool never looked. Saying "no warning statement located" here
        # would be a false statement about what was checked — the most
        # corrosive kind of error in a tool whose value rests on an agent
        # trusting what it reports.
        return result
    if result.strategy is not Strategy.NOT_FOUND:
        return result
    if rule.id not in ELEVATED_WHEN_MISSING or not rule.required:
        return result
    if not quality.assessed or quality.score < ruleset.defaults.good_image_quality:
        # Unassessed counts as not-good-enough. Elevating a missing mandatory
        # field is a statement that the image was clear enough for its absence
        # to mean something, and that statement cannot be made without a
        # measurement behind it.
        return result

    return result.model_copy(
        update={
            "elevated": True,
            "reason": (
                f"No {rule.label.lower()} located, and the image quality is good "
                f"({quality.score:.0%}). This statement is mandatory on every container. "
                "Verify manually before proceeding."
            ),
        }
    )
