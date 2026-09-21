"""Evaluate the verifier against the fixture corpus and print the performance table.

Run with `make evaluate` (``python -m tools.evaluate``). Produces the table
TRADEOFFS.md "Measured performance" expects, from `fixtures/expected.json`.

Two modes, and they measure different things
--------------------------------------------
``--provider stub`` (default)
    Each fixture's `.ocr.json` sidecar is replayed as perfect OCR. The numbers
    therefore measure **the rules engine against ground truth, not OCR
    accuracy**: every miss is a bug or a limitation in assignment and
    comparison, never a misread. The output says so on every table, because a
    stub-mode accuracy figure quoted without that qualifier would overstate
    the tool.

``--provider rapidocr``
    The real pipeline — `extraction.pipeline.preprocess` then the OCR engine
    then the rules — on the fixture images. This is the end-to-end figure,
    still optimistic because the corpus is synthetic (TRADEOFFS.md, "The
    fixture corpus is synthetic"). If the engine or its weights are not
    installed the harness says so and exits; it does not fall back to the stub
    and report stub numbers under the rapidocr heading.

How results are graded
----------------------
Each fixture/field in expected.json lists every verdict a correct tool may
return (``allowed``). A result is:

*   **unchecked** — the field's strategy is not implemented yet (its reason
    says "does not yet check"). Excluded from every numerator *and*
    denominator and reported separately. Counting it as correct would credit
    the tool for a check it never made; counting it as wrong would blame it
    for one it never claimed to make. Several strategies are being written in
    parallel, so this state is expected and must be visible.
*   **correct** — the verdict is in ``allowed`` (and, where the fixture says
    so, the field's ``elevated`` flag matches).
*   **false negative** — ``allowed`` excludes MATCH and the tool said MATCH.
    A violation that would reach the market. The number that matters most.
*   **false flag** — ``allowed`` is exactly MATCH on a compliant fixture and
    the tool said REVIEW or MISMATCH. Costs an agent about twenty seconds.

Latency is wall-clock per label over ``--runs`` repetitions, measured from
before extraction to the end of `verify`, which is the interval an agent
waits through.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "fixtures"

#: expected.json layouts this harness understands.
SUPPORTED_SCHEMA_VERSIONS = {1}

#: Marker text in the reason of a field whose strategy is not implemented
#: (rules/engine.py `_unimplemented_result`). Matched case-insensitively.
UNCHECKED_MARKER = "does not yet check"

#: Rows of the TRADEOFFS.md table, in its order: (row label, kind, id).
#: ``kind`` is "field" for a field verdict or "check" for a warning sub-check.
TABLE_ROWS = (
    ("Brand name", "field", "brand_name"),
    ("Alcohol content", "field", "alcohol_content"),
    ("Net contents", "field", "net_contents"),
    ("Class/type", "field", "class_type"),
    ("Warning statement (field verdict)", "field", "government_warning"),
    ("Warning text", "check", "text_accuracy"),
    ("Warning capitalization", "check", "header_capitalisation"),
)

FIELD_IDS = tuple(i for _, kind, i in TABLE_ROWS if kind == "field")
CHECK_IDS = tuple(i for _, kind, i in TABLE_ROWS if kind == "check")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """How one fixture/field (or fixture/check) was graded."""

    fixture: str
    item: str  # field id or check id
    kind: str  # "field" | "check"
    allowed: list[str]
    got: str | None  # verdict or check outcome; None when unchecked or errored
    status: str  # "correct" | "incorrect" | "unchecked" | "error"
    false_negative: bool = False
    false_flag: bool = False
    note: str = ""


@dataclass
class FixtureRun:
    """Everything measured for one fixture."""

    id: str
    compliance: str
    latencies_ms: list[float] = field(default_factory=list)
    error: str | None = None
    outcomes: list[Outcome] = field(default_factory=list)
    #: The image's basename — the name the batch API would sort it under.
    filename: str = ""
    #: `rules.triage.Tier` name and the full `rules.triage.sort_key`, stored
    #: rather than recomputed so that the ranking reported is, by
    #: construction, the ranking the API would have shown.
    tier: str | None = None
    sort_key: list = field(default_factory=list)
    #: 1-based position in the worst-first worklist.
    rank: int | None = None


def is_unchecked(result: Any) -> bool:
    """True when the engine reports that it never looked at this field."""
    return UNCHECKED_MARKER in (result.reason or "").lower()


def grade_field(entry: dict, field_id: str, result: Any) -> Outcome:
    """Grade one field result against its expectation."""
    spec = entry["fields"][field_id]
    allowed = spec["allowed"]
    base = {"fixture": entry["id"], "item": field_id, "kind": "field", "allowed": allowed}

    if is_unchecked(result):
        return Outcome(**base, got=None, status="unchecked", note="strategy not implemented")

    verdict = str(result.verdict.value if hasattr(result.verdict, "value") else result.verdict)
    correct = verdict in allowed
    note = ""
    if correct and "elevated" in spec and bool(result.elevated) != spec["elevated"]:
        # The verdict is right but the prominence is wrong. For the absent
        # warning, prominence *is* the finding — a plain REVIEW buries it
        # among ordinary low-confidence reads — so this counts as incorrect.
        correct = False
        note = f"elevated={result.elevated}, expected {spec['elevated']}"

    return Outcome(
        **base,
        got=verdict,
        status="correct" if correct else "incorrect",
        false_negative="MATCH" not in allowed and verdict == "MATCH",
        false_flag=(
            entry["compliance"] == "compliant" and allowed == ["MATCH"] and verdict != "MATCH"
        ),
        note=note or ("" if correct else _short(result.reason)),
    )


def grade_check(entry: dict, check_id: str, warning_result: Any) -> Outcome | None:
    """Grade one warning sub-check, or None if this fixture does not grade it.

    A check is *unchecked* when the warning strategy is unimplemented or when
    it ran but did not report this check id. The second case is reported
    rather than treated as a pass, because "the check did not run" must never
    read as "the check passed".
    """
    spec = entry.get("checks", {}).get(check_id)
    if spec is None:
        return None
    allowed = spec["allowed"]
    base = {"fixture": entry["id"], "item": check_id, "kind": "check", "allowed": allowed}

    if warning_result is None or is_unchecked(warning_result):
        return Outcome(
            **base, got=None, status="unchecked", note="warning strategy not implemented"
        )

    found = next((c for c in warning_result.checks if c.id == check_id), None)
    if found is None:
        return Outcome(**base, got=None, status="unchecked", note="check not reported by strategy")

    outcome = str(found.outcome.value if hasattr(found.outcome, "value") else found.outcome)
    correct = outcome in allowed
    return Outcome(
        **base,
        got=outcome,
        status="correct" if correct else "incorrect",
        false_negative="pass" not in allowed and outcome == "pass",
        note="" if correct else _short(found.summary),
    )


def _short(text: str | None, limit: int = 110) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class ProviderUnavailable(RuntimeError):
    """The requested extraction provider cannot run here."""


def make_runner(provider_name: str):
    """Return ``run(image_path, record) -> VerificationResult`` for a mode.

    Imports are local so `--help` and the unavailable-provider message work
    even if the project's own imports are broken mid-build.
    """
    from rules.engine import verify

    if provider_name == "stub":
        from extraction.providers.stub import StubProvider

        def run_stub(image_path: Path, record):
            started = time.perf_counter()
            extraction = StubProvider.for_image(image_path).extract(None)
            return verify(extraction, record, started_at=started)

        return run_stub

    if provider_name == "rapidocr":
        from extraction.pipeline import preprocess
        from extraction.providers import ExtractionError, get_provider

        try:
            provider = get_provider("rapidocr")
        except (ExtractionError, ImportError) as exc:
            raise ProviderUnavailable(
                f"the rapidocr provider is not available here. {exc} "
                "(Stub mode, the default, measures the rules engine alone.)"
            ) from exc

        def run_ocr(image_path: Path, record):
            started = time.perf_counter()
            prepared = preprocess(image_path.read_bytes())
            extraction = prepared.apply_quality(provider.extract(prepared.image))
            return verify(extraction, record, started_at=started)

        return run_ocr

    raise ProviderUnavailable(f"unknown provider {provider_name!r}; choose stub or rapidocr")


def load_expected(fixtures_dir: Path) -> dict:
    path = fixtures_dir / "expected.json"
    if not path.exists():
        raise SystemExit(f"no {path}. Generate the corpus first: `make fixtures`.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise SystemExit(
            f"{path} has schema_version {data.get('schema_version')!r}; this harness "
            f"understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return data


def evaluate(fixtures_dir: Path, provider_name: str, runs: int) -> list[FixtureRun]:
    """Run every fixture through the chosen pipeline and grade it.

    The first run's result is graded (verification is deterministic); all
    runs contribute to latency. Per-fixture failures — an image the quality
    gate rejects, say — are recorded as errors on that fixture rather than
    aborting the whole evaluation.
    """
    from extraction.providers.base import ExtractionError
    from rules import triage
    from rules.results import ApplicationRecord

    expected = load_expected(fixtures_dir)
    runner = make_runner(provider_name)
    results: list[FixtureRun] = []

    for entry in expected["fixtures"]:
        image_path = fixtures_dir / entry["image"]
        fixture = FixtureRun(
            id=entry["id"], compliance=entry["compliance"], filename=image_path.name
        )
        record = ApplicationRecord(**entry["record"])
        verification = None
        try:
            for _ in range(max(1, runs)):
                started = time.perf_counter()
                result = runner(image_path, record)
                fixture.latencies_ms.append((time.perf_counter() - started) * 1000.0)
                verification = verification or result
        except ExtractionError as exc:
            fixture.error = str(exc)

        if verification is None:
            fixture.outcomes = [
                Outcome(entry["id"], fid, "field", entry["fields"][fid]["allowed"], None, "error")
                for fid in FIELD_IDS
            ]
        else:
            by_id = {f.field_id: f for f in verification.fields}
            for field_id in FIELD_IDS:
                fixture.outcomes.append(grade_field(entry, field_id, by_id[field_id]))
            for check_id in CHECK_IDS:
                graded = grade_check(entry, check_id, by_id.get("government_warning"))
                if graded is not None:
                    fixture.outcomes.append(graded)

        # Place the fixture in the worklist exactly as the batch API does:
        # `rules.triage` is the single definition of worst-first, and an
        # evaluator with its own ordering could report a ranking no agent sees.
        summary = triage.summarise(verification, None if verification else fixture.error)
        fixture.tier = summary.tier.name
        fixture.sort_key = list(triage.sort_key(summary, fixture.filename))
        results.append(fixture)

    return results


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
#
# Batch is throughput-bound: its value is the order of the worklist, not the
# speed of any one label. So the question the harness asks of the order is the
# one an agent working down the list cares about — does every label with a
# violation come before every label without one? A single inversion means an
# agent who stops partway down the list has seen a clean label and skipped a
# non-compliant one.


def rank_fixtures(runs: list[FixtureRun]) -> list[FixtureRun]:
    """Sort by the stored triage key and record each fixture's position."""
    ordered = sorted(runs, key=lambda r: tuple(r.sort_key))
    for position, run in enumerate(ordered, start=1):
        run.rank = position
    return ordered


def separation(ordered_compliance: list[str]) -> dict:
    """How cleanly non-compliant labels sort above compliant ones.

    Args:
        ordered_compliance: Each fixture's ``compliance`` in worklist order.
            ``judgment`` fixtures are dropped from both sides: their correct
            outcome is itself a judgment call, so their position proves
            nothing either way.

    Returns:
        ``inversions`` — (compliant, non-compliant) pairs where the compliant
        label comes first; ``pairs`` — the number of such pairs possible;
        ``lowest_non_compliant_position`` — 1-based position of the last
        non-compliant label among graded ones; ``ideal_position`` — where it
        would sit with perfect separation (the non-compliant count).
    """
    graded = [c for c in ordered_compliance if c in ("compliant", "non_compliant")]
    inversions = 0
    compliant_seen = 0
    for label in graded:
        if label == "compliant":
            compliant_seen += 1
        else:
            inversions += compliant_seen
    non = graded.count("non_compliant")
    comp = graded.count("compliant")
    positions = [i for i, c in enumerate(graded, start=1) if c == "non_compliant"]
    return {
        "inversions": inversions,
        "pairs": non * comp,
        "non_compliant": non,
        "compliant": comp,
        "lowest_non_compliant_position": max(positions) if positions else None,
        "ideal_position": non,
    }


def ranking_summary(runs: list[FixtureRun]) -> dict:
    """Separation, tier breakdown, and the offending fixtures if any."""
    from rules.triage import Tier

    ordered = rank_fixtures(runs)
    sep = separation([r.compliance for r in ordered])
    lowest = next(
        (r for r in reversed(ordered) if r.compliance == "non_compliant"),
        None,
    )
    tiers = {}
    for tier in Tier:
        members = [r for r in ordered if r.tier == tier.name]
        tiers[tier.name] = {
            "label": tier.label,
            "total": len(members),
            **{
                c: sum(r.compliance == c for r in members)
                for c in ("non_compliant", "compliant", "judgment")
            },
        }
    # The compliant fixtures that outrank at least one non-compliant one:
    # what a reader needs to see to understand an inversion.
    above = (
        [r.id for r in ordered if r.compliance == "compliant" and r.rank < lowest.rank]
        if lowest is not None
        else []
    )
    return {
        **sep,
        "worklist_size": len(ordered),
        "lowest_non_compliant": lowest.id if lowest else None,
        "lowest_non_compliant_worklist_position": lowest.rank if lowest else None,
        "compliant_ranked_above_a_violation": above,
        "tiers": tiers,
        "order": [
            {"rank": r.rank, "id": r.id, "tier": r.tier, "compliance": r.compliance}
            for r in ordered
        ],
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Simple and exact on small samples."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def summarise(runs: list[FixtureRun], provider_name: str, repetitions: int) -> dict:
    """Aggregate graded outcomes into the numbers the table reports."""
    outcomes = [o for r in runs for o in r.outcomes]
    latencies = [ms for r in runs for ms in r.latencies_ms]

    rows = []
    for label, kind, item in TABLE_ROWS:
        mine = [o for o in outcomes if o.kind == kind and o.item == item]
        checked = [o for o in mine if o.status in ("correct", "incorrect")]
        rows.append(
            {
                "label": label,
                "kind": kind,
                "id": item,
                "correct": sum(o.status == "correct" for o in checked),
                "checked": len(checked),
                "unchecked": sum(o.status == "unchecked" for o in mine),
                "errors": sum(o.status == "error" for o in mine),
            }
        )

    field_outcomes = [o for o in outcomes if o.kind == "field"]
    violations = [o for o in field_outcomes if "MATCH" not in o.allowed]
    compliant_fields = [
        o
        for r in runs
        if r.compliance == "compliant"
        for o in r.outcomes
        if o.kind == "field" and o.allowed == ["MATCH"]
    ]
    checked_violations = [o for o in violations if o.status in ("correct", "incorrect")]
    checked_compliant = [o for o in compliant_fields if o.status in ("correct", "incorrect")]

    # Fixture-level view: a non-compliant fixture is "missed" when at least one
    # of its violations came back MATCH and none of its other violations were
    # flagged either — i.e. the agent would have seen an all-clear.
    noncompliant = [r for r in runs if r.compliance == "non_compliant"]
    fixture_missed = 0
    fixture_graded = 0
    for r in noncompliant:
        vio = [o for o in r.outcomes if o.kind == "field" and "MATCH" not in o.allowed]
        graded = [o for o in vio if o.status in ("correct", "incorrect")]
        if graded:
            fixture_graded += 1
            if all(o.false_negative for o in graded):
                fixture_missed += 1

    total_ms = sum(latencies)
    return {
        "provider": provider_name,
        "mode_note": (
            "rules engine vs ground truth, not OCR accuracy"
            if provider_name == "stub"
            else "full pipeline: preprocess + OCR + rules, on synthetic labels"
        ),
        "fixtures": len(runs),
        "fixtures_errored": sum(r.error is not None for r in runs),
        "runs_per_fixture": repetitions,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "samples": len(latencies),
        },
        "sequential_labels_per_minute": (60000.0 * len(latencies) / total_ms) if total_ms else None,
        "rows": rows,
        "ranking": ranking_summary(runs),
        "false_negatives": {
            "count": sum(o.false_negative for o in checked_violations),
            "of": len(checked_violations),
            "unchecked": sum(o.status == "unchecked" for o in violations),
            "fixtures_missed": fixture_missed,
            "fixtures_graded": fixture_graded,
            "check_level": sum(o.false_negative for o in outcomes if o.kind == "check"),
        },
        "false_flags": {
            "count": sum(o.false_flag for o in checked_compliant),
            "of": len(checked_compliant),
            "unchecked": sum(o.status == "unchecked" for o in compliant_fields),
            "fixtures_flagged": sum(
                any(o.false_flag for o in r.outcomes) for r in runs if r.compliance == "compliant"
            ),
            "fixtures_compliant": sum(r.compliance == "compliant" for r in runs),
        },
        "hardware": (
            f"{platform.machine()}, {os.cpu_count()} CPU(s), {platform.system()} "
            f"{platform.release()}, Python {platform.python_version()}"
        ),
    }


def _pct_cell(row: dict) -> str:
    if row["checked"] == 0:
        detail = "strategy not implemented" if row["unchecked"] else "no graded fixtures"
        return f"unchecked ({detail})"
    cell = f"{100.0 * row['correct'] / row['checked']:.0f}% ({row['correct']}/{row['checked']})"
    extras = []
    if row["unchecked"]:
        extras.append(f"{row['unchecked']} unchecked")
    if row["errors"]:
        extras.append(f"{row['errors']} errored")
    return cell + (f", {', '.join(extras)}" if extras else "")


def render_markdown(summary: dict, runs: list[FixtureRun]) -> str:
    """The TRADEOFFS.md table, plus the disagreements that explain it."""
    lat = summary["latency_ms"]
    fn = summary["false_negatives"]
    ff = summary["false_flags"]
    stub = summary["provider"] == "stub"
    unit = "rules only" if stub else "end to end"

    out = [
        f"### Measured performance — provider `{summary['provider']}`",
        "",
        f"**{summary['mode_note'][0].upper()}{summary['mode_note'][1:]}.** "
        + (
            "Each fixture's sidecar is replayed as perfect OCR, so these figures measure field "
            "assignment and comparison only. They say nothing about recognition accuracy."
            if stub
            else "Synthetic corpus: read as optimistic."
        ),
        "",
        f"{summary['fixtures']} fixtures, {summary['runs_per_fixture']} run(s) each"
        + (f", {summary['fixtures_errored']} errored" if summary["fixtures_errored"] else "")
        + ". Accuracy cells show correct/graded; *unchecked* fields (strategy not yet "
        "implemented) are excluded from both.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Single-label verification, p50 ({unit}) | {lat['p50'] / 1000.0:.4f} s |",
        f"| Single-label verification, p95 ({unit}) | {lat['p95'] / 1000.0:.4f} s |",
        "| Batch throughput (labels/minute) | not measured here — see `tools/load_test.py`, which measures it against a running server"
        + (
            f"; sequential single-process equivalent {summary['sequential_labels_per_minute']:.0f}"
            if summary["sequential_labels_per_minute"]
            else ""
        )
        + " |",
    ]
    for row in summary["rows"]:
        out.append(f"| {row['label']} — correct classification | {_pct_cell(row)} |")
    out += [
        f"| False negatives on non-compliant fixtures | {fn['count']} of {fn['of']} violation "
        f"fields"
        + (f" ({fn['unchecked']} more unchecked)" if fn["unchecked"] else "")
        + f"; {fn['fixtures_missed']} of {fn['fixtures_graded']} fixtures passed as all-clear"
        + (
            f"; {fn['check_level']} binding check(s) passed that should fail"
            if fn["check_level"]
            else ""
        )
        + " |",
        f"| False flags on compliant fixtures | {ff['count']} of {ff['of']} fields"
        + (f" ({ff['unchecked']} more unchecked)" if ff["unchecked"] else "")
        + f"; {ff['fixtures_flagged']} of {ff['fixtures_compliant']} fixtures |",
        *_ranking_rows(summary["ranking"]),
        "",
        f"Hardware: {summary['hardware']}.",
    ]
    if summary["ranking"]["inversions"]:
        out += [
            "",
            "#### Ranking inversions",
            "",
            "Compliant fixtures ranked above the lowest non-compliant one "
            f"(`{summary['ranking']['lowest_non_compliant']}`): "
            + ", ".join(f"`{i}`" for i in summary["ranking"]["compliant_ranked_above_a_violation"])
            + ".",
        ]

    incorrect = [o for r in runs for o in r.outcomes if o.status == "incorrect"]
    if incorrect:
        out += [
            "",
            "#### Disagreements",
            "",
            "| Fixture | Item | Expected | Got | Note |",
            "|---|---|---|---|---|",
        ]
        for o in incorrect:
            tag = (
                " **FALSE NEGATIVE**"
                if o.false_negative
                else (" (false flag)" if o.false_flag else "")
            )
            out.append(
                f"| {o.fixture} | {o.item} | {' / '.join(o.allowed)} | {o.got}{tag} | "
                f"{o.note.replace('|', '/')} |"
            )

    errored = [r for r in runs if r.error]
    if errored:
        out += ["", "#### Errors", ""]
        out += [f"- `{r.id}`: {_short(r.error, 200)}" for r in errored]

    unchecked = sorted({o.item for r in runs for o in r.outcomes if o.status == "unchecked"})
    if unchecked:
        out += ["", f"Unchecked (not graded): {', '.join(unchecked)}."]
    return "\n".join(out)


def _ranking_rows(ranking: dict) -> list[str]:
    """The two ranking rows of the table: separation, then why."""
    if ranking["lowest_non_compliant_position"] is None:
        sep = "no non-compliant fixtures to rank"
    else:
        sep = (
            f"{ranking['inversions']} inversions of {ranking['pairs']} compliant/non-compliant "
            f"pairs; lowest non-compliant (`{ranking['lowest_non_compliant']}`) is "
            f"{ranking['lowest_non_compliant_position']} of "
            f"{ranking['non_compliant'] + ranking['compliant']} graded "
            f"(ideal {ranking['ideal_position']}), "
            f"{ranking['lowest_non_compliant_worklist_position']} of "
            f"{ranking['worklist_size']} in the full worklist"
        )
    parts = []
    for name, t in ranking["tiers"].items():
        if not t["total"]:
            continue
        mix = ", ".join(
            f"{t[c]} {c.replace('_', '-')}"
            for c in ("non_compliant", "compliant", "judgment")
            if t[c]
        )
        parts.append(f"{name} {t['total']} ({mix})")
    return [
        f"| Worst-first ranking — separation (rules.triage) | {sep} |",
        f"| Worst-first ranking — tiers | {'; '.join(parts) or 'none'} |",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("stub", "rapidocr"), default="stub")
    parser.add_argument(
        "--fixtures", type=Path, default=DEFAULT_FIXTURES, help="corpus root (default: fixtures/)"
    )
    parser.add_argument("--runs", type=int, default=5, help="repetitions per fixture for latency")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    parser.add_argument(
        "--fail-on-false-negative",
        action="store_true",
        help="exit 1 if any violation came back MATCH (for CI)",
    )
    parser.add_argument(
        "--fail-on-ranking",
        action="store_true",
        help="exit 1 if any compliant fixture ranks above a non-compliant one (for CI)",
    )
    args = parser.parse_args(argv)

    try:
        runs = evaluate(args.fixtures, args.provider, args.runs)
    except ProviderUnavailable as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2

    summary = summarise(runs, args.provider, args.runs)
    if args.json:
        print(json.dumps({"summary": summary, "fixtures": [asdict(r) for r in runs]}, indent=2))
    else:
        print(render_markdown(summary, runs))

    if args.fail_on_false_negative and (
        summary["false_negatives"]["count"] or summary["false_negatives"]["check_level"]
    ):
        return 1
    if args.fail_on_ranking and summary["ranking"]["inversions"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
