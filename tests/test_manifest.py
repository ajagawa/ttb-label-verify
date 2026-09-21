"""Manifest parsing and image pairing.

These are the checks that decide whether an agent's spreadsheet and folder of
photographs line up. Every case here is something a real submission produces —
Excel's byte order mark, a header typed with a capital letter, a filename typed
in lower case against a camera's upper case, a row pasted twice — and the tests
pin down two things for each: that nothing raises, and that the message tells a
non-technical agent exactly what to fix.
"""

from __future__ import annotations

from api.batch_models import MANIFEST_COLUMNS, PairingIssueKind
from api.manifest import (
    FileEntry,
    basename,
    normalise_header,
    pair_files,
    parse_manifest,
    template_csv,
)

HEADER = "filename,brand_name,class_type,alcohol_content,net_contents"


def csv_bytes(*lines: str, newline: str = "\n", bom: bool = False) -> bytes:
    text = newline.join(lines) + newline
    return (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")


def row(filename: str, brand: str = "OLD TOM DISTILLERY") -> str:
    return f"{filename},{brand},Kentucky Straight Bourbon Whiskey,45% Alc./Vol.,750 mL"


def kinds(report) -> list[PairingIssueKind]:
    return [issue.kind for issue in report.issues]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_plain_manifest_parses_with_spreadsheet_row_numbers(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg")))
        assert [r.filename for r in parsed.rows] == ["a.jpg", "b.jpg"]
        # The header is row 1, so the first label is row 2 — as Excel shows it.
        assert [r.row_number for r in parsed.rows] == [2, 3]
        assert parsed.total_rows == 2
        assert parsed.issues == ()

    def test_utf8_bom_from_excel_does_not_hide_the_first_column(self) -> None:
        """Left in, the BOM glues itself to "filename" and the column looks missing."""
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), bom=True))
        assert parsed.missing_columns == ()
        assert parsed.rows[0].filename == "a.jpg"

    def test_windows_line_endings(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg"), newline="\r\n"))
        assert [r.filename for r in parsed.rows] == ["a.jpg", "b.jpg"]
        assert parsed.rows[1].net_contents == "750 mL"  # no stray \r on the last cell

    def test_quoted_cell_with_a_comma_and_a_line_break(self) -> None:
        data = csv_bytes(
            HEADER,
            'a.jpg,"OLD TOM, DISTILLERY",Bourbon,45%,750 mL',
            'b.jpg,"TWO\r\nLINES",Bourbon,45%,750 mL',
            newline="\r\n",
        )
        parsed = parse_manifest(data)
        assert parsed.rows[0].brand_name == "OLD TOM, DISTILLERY"
        assert parsed.rows[1].brand_name == "TWO\r\nLINES"
        assert [r.row_number for r in parsed.rows] == [2, 3]

    def test_headers_are_case_and_whitespace_insensitive(self) -> None:
        data = csv_bytes(
            " Filename , Brand Name,CLASS_TYPE,alcohol-content ,Net Contents",
            row("a.jpg"),
        )
        parsed = parse_manifest(data)
        assert parsed.missing_columns == ()
        assert parsed.rows[0].brand_name == "OLD TOM DISTILLERY"

    def test_normalise_header(self) -> None:
        assert normalise_header("  Brand   Name ") == "brand_name"
        assert normalise_header("LABEL-WIDTH-MM") == "label_width_mm"

    def test_columns_may_come_in_any_order_and_extras_are_ignored(self) -> None:
        data = csv_bytes(
            "notes,net_contents,alcohol_content,class_type,brand_name,filename",
            "check twice,750 mL,45%,Bourbon,OLD TOM,a.jpg",
        )
        (only,) = parse_manifest(data).rows
        assert (only.filename, only.brand_name, only.net_contents) == ("a.jpg", "OLD TOM", "750 mL")

    def test_cells_are_trimmed(self) -> None:
        data = csv_bytes(HEADER, "  a.jpg ,  OLD TOM  , Bourbon , 45% , 750 mL ")
        (only,) = parse_manifest(data).rows
        assert only.filename == "a.jpg"
        assert only.brand_name == "OLD TOM"

    def test_blank_trailing_rows_are_ignored(self) -> None:
        data = csv_bytes(HEADER, row("a.jpg"), ",,,,", "", "  ,  ,,,")
        parsed = parse_manifest(data)
        assert parsed.total_rows == 1
        assert parsed.issues == ()

    def test_blank_rows_in_the_middle_still_count_toward_row_numbers(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), "", row("b.jpg")))
        assert [r.row_number for r in parsed.rows] == [2, 4]

    def test_missing_required_column_is_reported_by_name(self) -> None:
        parsed = parse_manifest(csv_bytes("filename,brand_name,class_type,net_contents", "a,b,c,d"))
        assert parsed.missing_columns == ("alcohol_content",)
        (issue,) = parsed.issues
        assert issue.kind is PairingIssueKind.MISSING_COLUMN
        assert '"alcohol_content"' in issue.message

    def test_empty_manifest_reports_missing_columns_rather_than_raising(self) -> None:
        parsed = parse_manifest(b"")
        assert set(parsed.missing_columns) == {
            "filename",
            "brand_name",
            "class_type",
            "alcohol_content",
            "net_contents",
        }

    def test_header_only_manifest_says_there_are_no_rows(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER))
        assert parsed.total_rows == 0
        assert any("no label rows" in i.message for i in parsed.issues)

    def test_empty_brand_is_an_invalid_row_not_an_exception(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), "b.jpg,,Bourbon,45%,750 mL"))
        assert [r.filename for r in parsed.rows] == ["a.jpg"]
        (issue,) = parsed.issues
        assert issue.kind is PairingIssueKind.INVALID_ROW
        assert issue.row_number == 3
        assert issue.message == "Row 3 was skipped: the brand name is blank."
        # Invalid rows still count: the agent's sheet has two rows.
        assert parsed.total_rows == 2

    def test_several_blank_cells_are_listed_together(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, "b.jpg,,,45%,750 mL"))
        assert "the brand name and class/type are blank" in parsed.issues[0].message

    def test_short_row_is_invalid_not_an_index_error(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, "b.jpg,OLD TOM"))
        assert parsed.issues[0].kind is PairingIssueKind.INVALID_ROW

    def test_non_numeric_label_width_is_invalid(self) -> None:
        data = csv_bytes(HEADER + ",label_width_mm", row("a.jpg") + ",about eighty")
        parsed = parse_manifest(data)
        assert parsed.rows == ()
        assert "not a number of millimetres" in parsed.issues[0].message

    def test_label_width_accepts_a_unit_and_rejects_non_positive(self) -> None:
        data = csv_bytes(
            HEADER + ",label_width_mm",
            row("a.jpg") + ",80 mm",
            row("b.jpg") + ",0",
            row("c.jpg") + ",",
            row("d.jpg") + ",nan",
        )
        parsed = parse_manifest(data)
        widths = {r.filename: r.label_width_mm for r in parsed.rows}
        assert widths == {"a.jpg": 80.0, "c.jpg": None}
        assert {i.row_number for i in parsed.issues} == {3, 5}

    def test_beverage_class_defaults_and_is_checked_against_the_ruleset(self) -> None:
        data = csv_bytes(
            HEADER + ",beverage_class",
            row("a.jpg") + ",",
            row("b.jpg") + ",wine",
        )
        parsed = parse_manifest(data, beverage_classes={"distilled_spirits"})
        assert [(r.filename, r.beverage_class) for r in parsed.rows] == [
            ("a.jpg", "distilled_spirits")
        ]
        assert '"wine" is not supported' in parsed.issues[0].message

    def test_windows_1252_manifest_decodes_instead_of_failing(self) -> None:
        data = (HEADER + "\n" + row("a.jpg", brand="AÑEJO REAL") + "\n").encode("cp1252")
        assert parse_manifest(data).rows[0].brand_name == "AÑEJO REAL"

    def test_utf16_manifest_decodes(self) -> None:
        data = (HEADER + "\n" + row("a.jpg") + "\n").encode("utf-16")
        assert parse_manifest(data).rows[0].filename == "a.jpg"

    def test_filename_cell_with_a_path_keeps_only_the_basename(self) -> None:
        data = csv_bytes(HEADER, row(r"C:\labels\a.jpg"), row("import/b.jpg"))
        assert [r.filename for r in parse_manifest(data).rows] == ["a.jpg", "b.jpg"]

    def test_template_parses_cleanly_with_the_brief_example(self) -> None:
        text = template_csv()
        assert text.splitlines()[0] == ",".join(MANIFEST_COLUMNS)
        (example,) = parse_manifest(text.encode()).rows
        assert example.brand_name == "OLD TOM DISTILLERY"
        assert example.class_type == "Kentucky Straight Bourbon Whiskey"
        assert example.alcohol_content == "45% Alc./Vol."
        assert example.net_contents == "750 mL"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_basename_strips_either_separator(self) -> None:
        assert basename("import-0412/labels/a.jpg") == "a.jpg"
        assert basename(r"C:\x\a.jpg") == "a.jpg"

    def test_exact_matches_pair_in_upload_order(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg")))
        pairing = pair_files(parsed, ["b.jpg", "a.jpg"])
        assert [(p.filename, p.image_index, p.row.row_number) for p in pairing.pairs] == [
            ("b.jpg", 0, 3),
            ("a.jpg", 1, 2),
        ]
        report = pairing.report
        assert (report.matched, report.total_images, report.total_rows) == (2, 2, 2)
        assert report.ok

    def test_folder_upload_paths_are_stripped(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg")))
        pairing = pair_files(parsed, ["batch-17/labels/a.jpg"])
        assert pairing.pairs[0].filename == "a.jpg"
        assert pairing.report.ok

    def test_case_insensitive_fallback_when_unambiguous(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("img_0412.jpg")))
        pairing = pair_files(parsed, ["IMG_0412.JPG"])
        assert pairing.report.matched == 1
        assert pairing.pairs[0].filename == "IMG_0412.JPG"

    def test_exact_match_wins_over_case_insensitive(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg")))
        pairing = pair_files(parsed, ["A.JPG", "a.jpg"])
        assert pairing.pairs[0].image_index == 1
        # The other one really has no row.
        assert kinds(pairing.report) == [PairingIssueKind.IMAGE_WITHOUT_ROW]

    def test_case_only_difference_between_two_images_is_not_guessed(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg")))
        pairing = pair_files(parsed, ["A.jpg", "A.JPG"])
        assert pairing.report.matched == 0
        assert pairing.report.blocking
        messages = " ".join(i.message for i in pairing.report.issues)
        assert "apart from capital letters" in messages

    def test_case_only_difference_between_two_rows_is_not_guessed(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("A.jpg"), row("ok.jpg")))
        pairing = pair_files(parsed, ["A.JPG", "ok.jpg"])
        assert [p.filename for p in pairing.pairs] == ["ok.jpg"]
        assert PairingIssueKind.ROW_WITHOUT_IMAGE in kinds(pairing.report)

    def test_row_without_image_message_is_plain_language(self) -> None:
        parsed = parse_manifest(
            csv_bytes(HEADER, *[row(f"x{i}.jpg") for i in range(12)], row("old_tom.jpg"))
        )
        pairing = pair_files(parsed, [f"x{i}.jpg" for i in range(12)])
        (issue,) = pairing.report.issues
        assert issue.kind is PairingIssueKind.ROW_WITHOUT_IMAGE
        assert issue.row_number == 14
        assert issue.message == (
            "Row 14 names old_tom.jpg, but no image with that name was selected."
        )
        assert not pairing.report.blocking  # 12 labels can still run

    def test_image_without_row(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg")))
        pairing = pair_files(parsed, ["a.jpg", "stray.png"])
        (issue,) = pairing.report.issues
        assert issue.kind is PairingIssueKind.IMAGE_WITHOUT_ROW
        assert issue.message == "stray.png was selected, but no row in the manifest names it."

    def test_duplicate_rows_are_left_out_and_reported_once(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg"), row("a.jpg")))
        pairing = pair_files(parsed, ["a.jpg", "b.jpg"])
        assert [p.filename for p in pairing.pairs] == ["b.jpg"]
        # Reported as a duplicate, not also as an image without a row.
        (issue,) = pairing.report.issues
        assert issue.kind is PairingIssueKind.DUPLICATE_ROW
        assert issue.message.startswith("Rows 2 and 4 all name a.jpg.")

    def test_duplicate_images_from_different_folders_are_left_out(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg")))
        pairing = pair_files(parsed, ["one/a.jpg", "two/a.jpg", "b.jpg"])
        assert [p.filename for p in pairing.pairs] == ["b.jpg"]
        (issue,) = pairing.report.issues
        assert issue.kind is PairingIssueKind.DUPLICATE_IMAGE
        assert "2 images named a.jpg" in issue.message

    def test_unsupported_file_types_are_reported(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("art.pdf")))
        pairing = pair_files(parsed, ["a.jpg", "art.pdf", "notes.txt"])
        unsupported = [
            i for i in pairing.report.issues if i.kind is PairingIssueKind.UNSUPPORTED_FILE
        ]
        assert [i.filename for i in unsupported] == ["art.pdf", "notes.txt"]
        assert "PDF" in unsupported[0].message
        # The row naming art.pdf is explained by the file issue, not repeated.
        assert PairingIssueKind.ROW_WITHOUT_IMAGE not in kinds(pairing.report)

    def test_file_entry_problems_are_reported_with_their_reason(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("big.jpg")))
        pairing = pair_files(
            parsed, ["a.jpg", FileEntry("big.jpg", problem="is larger than the 20 MB limit")]
        )
        assert pairing.report.matched == 1
        assert pairing.report.issues[0].message == "big.jpg is larger than the 20 MB limit."

    def test_operating_system_clutter_is_ignored(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg")))
        pairing = pair_files(parsed, ["x/.DS_Store", "a.jpg", "Thumbs.db", "._a.jpg"])
        assert pairing.report.ok
        assert pairing.report.total_images == 1

    def test_invalid_row_explains_its_image(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), "b.jpg,,Bourbon,45%,750 mL"))
        pairing = pair_files(parsed, ["a.jpg", "b.jpg"])
        assert kinds(pairing.report) == [PairingIssueKind.INVALID_ROW]

    def test_blocking_only_when_nothing_can_run(self) -> None:
        parsed = parse_manifest(csv_bytes(HEADER, row("a.jpg"), row("b.jpg")))
        assert not pair_files(parsed, ["a.jpg", "zzz.jpg"]).report.blocking
        assert pair_files(parsed, ["zzz.jpg"]).report.blocking
        assert pair_files(parsed, []).report.blocking

    def test_missing_column_blocks_even_if_names_would_match(self) -> None:
        parsed = parse_manifest(csv_bytes("filename,brand_name", "a.jpg,OLD TOM"))
        report = pair_files(parsed, ["a.jpg"]).report
        assert report.blocking
        assert report.matched == 0
        assert PairingIssueKind.MISSING_COLUMN in kinds(report)
