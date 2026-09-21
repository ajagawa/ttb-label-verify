"""Tests for the batch side of evaluation: manifests, ranking, load-test helpers.

Everything here is pure or file-local — no server, no OCR — so it runs in the
default suite. The live load test itself is exercised by hand against a
running server; what can silently go wrong *without* a server (a manifest that
drifted from the corpus, an inversion count that miscounts, a percentile that
is off by one rank) is pinned here.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from api.batch_models import MANIFEST_COLUMNS, PairingIssueKind
from api.manifest import pair_files, parse_manifest
from tools import evaluate, load_test
from tools import generate_fixtures as gen

REPO_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

#: Enough fixtures for the broken manifest's four-row pattern, cheap to draw.
SUBSET = {"old_tom_compliant", "brand_two_lines", "shared_line_abv_net", "brand_twice_bottler"}


@pytest.fixture(scope="module")
def subset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("batch-corpus")
    gen.generate(out, SUBSET)
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


class TestCorpusManifest:
    def test_columns_follow_the_api_contract(self, subset_dir: Path) -> None:
        header = (subset_dir / gen.MANIFEST_FILENAME).read_text().splitlines()[0]
        assert tuple(header.split(",")) == MANIFEST_COLUMNS

    def test_round_trips_with_expected_json(self, subset_dir: Path) -> None:
        """One row per fixture, basename + record, in expected.json order."""
        expected = json.loads((subset_dir / "expected.json").read_text())["fixtures"]
        rows = _rows(subset_dir / gen.MANIFEST_FILENAME)
        assert [r["filename"] for r in rows] == [Path(e["image"]).name for e in expected]
        for row, entry in zip(rows, expected, strict=True):
            for key, value in entry["record"].items():
                assert row[key] == str(value), (entry["id"], key)

    def test_committed_manifest_is_current(self) -> None:
        """The committed CSV must be exactly what the generator would write now."""
        path = REPO_FIXTURES / gen.MANIFEST_FILENAME
        if not path.exists():
            pytest.skip("corpus not generated; run `make fixtures`")
        committed = json.loads((REPO_FIXTURES / "expected.json").read_text())["fixtures"]
        assert _rows(path) == [gen.manifest_row(e) for e in committed]

    def test_pairs_cleanly_with_the_corpus_images(self) -> None:
        """Parsed by the backend's own parser, every row pairs with its image."""
        manifest = REPO_FIXTURES / gen.MANIFEST_FILENAME
        if not manifest.exists():
            pytest.skip("corpus not generated; run `make fixtures`")
        images = [r["filename"] for r in _rows(manifest)]
        report = pair_files(parse_manifest(manifest.read_bytes()), images).report
        assert report.ok, report.issues
        assert report.matched == len(images)


class TestBrokenManifest:
    def test_has_exactly_the_three_intended_defects(self, subset_dir: Path) -> None:
        """Paired with the images it names, the backend reports three issues, one of each kind.

        The e2e and pairing tests depend on this being exactly three: a fourth,
        accidental defect would make "the UI showed every issue" unfalsifiable.
        """
        manifest = subset_dir / gen.BROKEN_MANIFEST_FILENAME
        rows = _rows(manifest)
        images = sorted({r["filename"] for r in rows} - {gen.MISSING_IMAGE_NAME})
        assert all((subset_dir / gen.LABELS_SUBDIR / name).exists() for name in images)

        report = pair_files(parse_manifest(manifest.read_bytes()), images).report
        kinds = sorted(issue.kind for issue in report.issues)
        assert kinds == sorted(
            [
                PairingIssueKind.ROW_WITHOUT_IMAGE,
                PairingIssueKind.INVALID_ROW,
                PairingIssueKind.DUPLICATE_ROW,
            ]
        )
        assert report.matched == 2
        assert not report.blocking

    def test_defects_are_where_the_readme_says(self, subset_dir: Path) -> None:
        rows = _rows(subset_dir / gen.BROKEN_MANIFEST_FILENAME)
        assert len(rows) == 6
        assert rows[2]["filename"] == gen.MISSING_IMAGE_NAME
        assert rows[3]["brand_name"] == ""
        assert rows[4] == rows[5]

    def test_not_written_for_a_tiny_subset(self, tmp_path: Path) -> None:
        gen.generate(tmp_path, {"old_tom_compliant"})
        assert (tmp_path / gen.MANIFEST_FILENAME).exists()
        assert not (tmp_path / gen.BROKEN_MANIFEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# Ranking metric
# ---------------------------------------------------------------------------


class TestSeparation:
    def test_perfect_order_has_no_inversions(self) -> None:
        result = evaluate.separation(["non_compliant", "non_compliant", "compliant", "compliant"])
        assert result["inversions"] == 0
        assert result["pairs"] == 4
        assert result["lowest_non_compliant_position"] == result["ideal_position"] == 2

    def test_hand_built_order_with_known_inversions(self) -> None:
        """C N C N N: first C outranks 3 Ns, second C outranks 2 -> 5 inversions."""
        order = ["compliant", "non_compliant", "compliant", "non_compliant", "non_compliant"]
        result = evaluate.separation(order)
        assert result["inversions"] == 5
        assert result["pairs"] == 6
        assert result["lowest_non_compliant_position"] == 5
        assert result["ideal_position"] == 3

    def test_fully_reversed_order_inverts_every_pair(self) -> None:
        result = evaluate.separation(["compliant"] * 3 + ["non_compliant"] * 2)
        assert result["inversions"] == result["pairs"] == 6

    def test_judgment_fixtures_are_excluded_from_both_sides(self) -> None:
        with_judgment = ["judgment", "non_compliant", "judgment", "compliant", "judgment"]
        result = evaluate.separation(with_judgment)
        assert result["inversions"] == 0
        assert result["lowest_non_compliant_position"] == 1

    def test_ranking_uses_the_triage_sort_key(self) -> None:
        """rank_fixtures orders by the stored `rules.triage.sort_key`, nothing else."""
        from rules.triage import Tier

        runs = [
            evaluate.FixtureRun(
                id="clean", compliance="compliant", tier="CLEAR", sort_key=[int(Tier.CLEAR), 0]
            ),
            evaluate.FixtureRun(
                id="bad", compliance="non_compliant", tier="ASSERTED", sort_key=[0, -1]
            ),
        ]
        summary = evaluate.ranking_summary(runs)
        assert [o["id"] for o in summary["order"]] == ["bad", "clean"]
        assert summary["inversions"] == 0
        assert summary["tiers"]["ASSERTED"]["non_compliant"] == 1
        assert summary["tiers"]["CLEAR"]["compliant"] == 1

    def test_fail_on_ranking_flag_exits_nonzero_on_inversion(
        self, subset_dir: Path, monkeypatch, capsys
    ) -> None:
        """--fail-on-ranking turns an inversion into exit 1; without it, exit 0."""
        real = evaluate.ranking_summary

        def inverted(runs):
            summary = real(runs)
            summary["inversions"] = 1
            return summary

        monkeypatch.setattr(evaluate, "ranking_summary", inverted)
        args = ["--fixtures", str(subset_dir), "--runs", "1"]
        assert evaluate.main(args) == 0
        assert evaluate.main([*args, "--fail-on-ranking"]) == 1
        capsys.readouterr()

    def test_ranking_rows_appear_in_markdown_and_json(self, subset_dir: Path, capsys) -> None:
        evaluate.main(["--fixtures", str(subset_dir), "--runs", "1"])
        out = capsys.readouterr().out
        assert "| Worst-first ranking — separation (rules.triage) |" in out
        assert "| Worst-first ranking — tiers |" in out
        evaluate.main(["--fixtures", str(subset_dir), "--runs", "1", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "inversions" in payload["summary"]["ranking"]
        assert all(f["tier"] for f in payload["fixtures"])


# ---------------------------------------------------------------------------
# Load-test pure helpers
# ---------------------------------------------------------------------------


class TestLoadTestHelpers:
    def test_build_batch_repeats_the_corpus_under_unique_names(
        self, subset_dir: Path, tmp_path: Path
    ) -> None:
        corpus = load_test.load_corpus(subset_dir)
        manifest, files = load_test.build_batch(corpus, 10, tmp_path / "batch")
        names = [f.upload_name for f in files]
        assert len(names) == len(set(names)) == 10
        assert all(f.path.parent == tmp_path / "batch" for f in files)
        # Contents cycle through the corpus in order.
        assert files[len(corpus)].path.read_bytes() == corpus[0].path.read_bytes()

        rows = _rows(manifest)
        assert tuple(rows[0]) == MANIFEST_COLUMNS
        assert [r["filename"] for r in rows] == names
        report = pair_files(parse_manifest(manifest.read_bytes()), names).report
        assert report.ok and report.matched == 10

    def test_build_batch_rejects_an_empty_request(self, subset_dir: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_test.build_batch(load_test.load_corpus(subset_dir), 0, tmp_path)

    def test_latency_stats_nearest_rank(self) -> None:
        stats = load_test.latency_stats([float(v) for v in range(1, 21)])
        assert (stats["n"], stats["p50"], stats["p95"], stats["max"]) == (20, 10.0, 19.0, 20.0)
        assert load_test.latency_stats([])["p95"] is None

    def test_assess_against_the_five_second_requirement(self) -> None:
        ok = load_test.latency_stats([1.0, 2.0, 3.0, 4.0])
        slow = load_test.latency_stats([1.0, 2.0, 6.0, 7.0])
        few = load_test.latency_stats([1.0])
        assert load_test.assess(ok)[0] == "PASS"
        assert load_test.assess(slow)[0] == "FAIL"
        assert load_test.assess(few)[0] == "INCONCLUSIVE"

    def test_peak_rss_reads_vmhwm(self, tmp_path: Path) -> None:
        (tmp_path / "123").mkdir()
        (tmp_path / "123" / "status").write_text("Name:\tuvicorn\nVmHWM:\t  204800 kB\n")
        assert load_test.read_peak_rss_kb(123, proc_root=tmp_path) == 204800
        assert load_test.read_peak_rss_kb(999, proc_root=tmp_path) is None

    def test_multipart_body_is_parseable(self) -> None:
        """Round-trip through the stdlib's MIME parser, as a server would see it."""
        from email.parser import BytesParser
        from email.policy import HTTP

        body, ctype = load_test.encode_multipart(
            [("brand_name", "STONE'S THROW")],
            [("image", "a.png", "image/png", b"\x89PNG-bytes")],
            boundary="xyz",
        )
        message = BytesParser(policy=HTTP).parse(
            io.BytesIO(f"Content-Type: {ctype}\r\n\r\n".encode() + body)
        )
        parts = list(message.iter_parts())
        assert parts[0].get_param("name", header="content-disposition") == "brand_name"
        assert parts[0].get_content() == "STONE'S THROW"
        assert parts[1].get_filename() == "a.png"
        assert parts[1].get_content() == b"\x89PNG-bytes"

    def test_report_renders_verdict_and_both_columns(self) -> None:
        report = {
            "url": "http://x",
            "verdict": "PASS",
            "verdict_reason": "fine",
            "idle": load_test.latency_stats([1.0, 1.2]),
            "during_batch": {**load_test.latency_stats([2.0, 2.5, 3.0]), "after_batch": 1},
            "batch": {
                "size": 20,
                "completed": 20,
                "failed": 0,
                "state": "done",
                "submit_s": 0.5,
                "wall_s": 40.0,
                "labels_per_minute": 30.0,
                "per_label_p50_s": 1.9,
                "polls": 12,
                "poll_mean_s": 0.05,
                "poll_max_kb": 300.0,
            },
            "server_peak_rss_mb": None,
        }
        text = load_test.render_markdown(report)
        assert "**PASS**" in text
        assert "| Single-label p95 | 1.20 s | 3.00 s |" in text
        assert "30.00 labels/min" in text
        assert "not measured (no --server-pid)" in text
        assert "1 single-label request(s) were sent after the batch finished" in text

    def test_unreachable_server_is_a_clear_message(self, capsys) -> None:
        code = load_test.main(["--url", "http://127.0.0.1:9", "--single-requests", "1"])
        assert code == 2
        assert "not reachable" in capsys.readouterr().err


class TestServerPidGuard:
    """A wrong PID must fail loudly, not report a plausible small number.

    The first 300-label run reported a 5 MB peak because the PID belonged to
    the shell that launched the server. A substring check on the command line
    then failed to catch it, because that shell's command line contains
    "python" and "uvicorn" as text. These tests build fake /proc entries.
    """

    def _proc(self, tmp_path, pid: int, argv: list[str], hwm_kb: int = 1_000_000):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\tx\nVmHWM:\t{hwm_kb} kB\n")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        return tmp_path

    def test_the_real_server_is_accepted(self, tmp_path) -> None:
        from tools.load_test import read_peak_rss_kb

        root = self._proc(tmp_path, 7, ["python3", "-m", "uvicorn", "api.main:app"])
        assert read_peak_rss_kb(7, proc_root=root) == 1_000_000

    def test_the_launching_shell_is_rejected_despite_mentioning_both(self, tmp_path) -> None:
        import pytest

        from tools.load_test import LoadTestError, read_peak_rss_kb

        root = self._proc(
            tmp_path, 8, ["/bin/bash", "-c", "python3 -m uvicorn api.main:app --port 8095"], 4816
        )
        with pytest.raises(LoadTestError, match="not the API server"):
            read_peak_rss_kb(8, proc_root=root)

    def test_a_python_process_that_is_not_uvicorn_is_rejected(self, tmp_path) -> None:
        import pytest

        from tools.load_test import LoadTestError, read_peak_rss_kb

        root = self._proc(tmp_path, 9, ["python3", "-m", "pytest"])
        with pytest.raises(LoadTestError):
            read_peak_rss_kb(9, proc_root=root)
