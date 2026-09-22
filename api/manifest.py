"""Reading a batch manifest and pairing its rows with image files.

A batch arrives as two things an agent assembled by hand: a spreadsheet of
application records and a folder of photographs. Everything that can go wrong
between them is a *clerical* problem — a typo in a filename, a row pasted twice,
a stray PDF in the folder — and none of it says anything about whether a label
complies. This module's job is to find those problems before any OCR runs and
to describe each one so that a non-technical agent can fix it in the
spreadsheet without asking anybody.

Three principles, each from the discovery notes:

*   **One bad row never holds up the rest.** Problems become `PairingIssue`s,
    never exceptions. A batch is blocked only when nothing at all could run: a
    required column is missing, or not a single row pairs with an image. One
    mistyped filename should not stop 299 other labels.

*   **Never guess which record a label belongs to.** When a filename is named
    by two rows, or two uploaded images share a name, the tool cannot know
    which pairing the agent meant. Checking a label against the wrong
    application record would produce a confident, wrong MISMATCH — the most
    expensive kind of error in this system — so ambiguous pairs are left out
    and reported, not resolved by picking one.

*   **Accept what spreadsheets actually produce.** Excel writes a UTF-8 byte
    order mark and CRLF line endings; people type "Brand Name" as a header;
    cells pick up stray spaces; the last few rows are often blank. None of that
    is an error, so none of it is reported as one.

Pure functions only: no I/O, no FastAPI, no global state. The API module calls
these and the tests call them directly.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from api.batch_models import (
    MANIFEST_COLUMNS,
    MANIFEST_REQUIRED_COLUMNS,
    MAX_BATCH_SIZE,
    ManifestRow,
    PairingIssue,
    PairingIssueKind,
    PairingReport,
)

#: Records (lines, blank or not, header included) a manifest may contain. A
#: manifest names at most MAX_BATCH_SIZE images; this leaves room for blank
#: lines and rows the agent will be told to fix. Found in a security review:
#: with no row limit, a 2 MB manifest of 100,000 one-character rows produced
#: 100,000 issues, about 85 MB held per batch for two hours and a 19 MB reply
#: to every status poll.
MAX_MANIFEST_RECORDS = 4 * MAX_BATCH_SIZE

#: Characters in one cell. The longest real value — a brand name or class
#: designation — is well under 200.
MAX_CELL_CHARS = 1000


class ManifestRejected(ValueError):
    """The manifest is unreadable or over a size limit, as a whole.

    Distinct from the per-row problems reported as issues: nothing in it can
    be paired, and the fix is to the file itself. `status_code` is the HTTP
    status the endpoints answer with.
    """

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _read_records(text: str) -> list[list[str]]:
    """Parse CSV text into records, within the limits above."""
    records: list[list[str]] = []
    try:
        # newline="" hands line-ending handling to the csv module, which is
        # the only way a quoted cell containing a line break survives a CRLF
        # file.
        for record in csv.reader(io.StringIO(text, newline="")):
            records.append(record)
            if len(records) > MAX_MANIFEST_RECORDS:
                raise ManifestRejected(
                    f"The manifest has more than {MAX_MANIFEST_RECORDS} rows. A batch can hold "
                    f"at most {MAX_BATCH_SIZE} labels; split it into smaller manifests.",
                    status_code=413,
                )
            if any(len(cell) > MAX_CELL_CHARS for cell in record):
                raise ManifestRejected(
                    f"Row {len(records)} of the manifest has a cell over {MAX_CELL_CHARS} "
                    "characters. Check that the file is the manifest, saved as CSV."
                )
    except csv.Error as exc:
        # Raised for, among others, a quoted cell with no closing quote that
        # runs past the csv module's field limit.
        raise ManifestRejected(
            "The manifest could not be read as a CSV file. Save it from the spreadsheet as "
            "CSV and try again."
        ) from exc
    return records


#: Extensions accepted by name. Mirrors `api.main.ACCEPTED_CONTENT_TYPES`: the
#: check endpoint only sees filenames, so the type has to be judged from the
#: extension there, and the two lists must agree or the check would approve a
#: file that the upload then rejects.
ACCEPTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"})

#: Files an operating system drops into folders on its own. A folder upload of
#: 300 photographs from a Mac carries `.DS_Store`; one from Windows carries
#: `Thumbs.db`. Reporting those as "not an image" would bury the real issues
#: under noise the agent never created and cannot see in their file browser.
_OS_CLUTTER = frozenset({".ds_store", "thumbs.db", "desktop.ini"})

#: The example row from the brief, used by the downloadable template. Kept here
#: beside the parser so the template is guaranteed to parse cleanly.
TEMPLATE_EXAMPLE_ROW = {
    "filename": "old_tom.jpg",
    "brand_name": "OLD TOM DISTILLERY",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol.",
    "net_contents": "750 mL",
    "beverage_class": "distilled_spirits",
    "label_width_mm": "",
}

#: Friendly names for columns, used in messages. An agent reading "the
#: brand_name column" can map it to the spreadsheet; the friendly form reads
#: better in a sentence.
_COLUMN_WORDS = {
    "filename": "filename",
    "brand_name": "brand name",
    "class_type": "class/type",
    "alcohol_content": "alcohol content",
    "net_contents": "net contents",
    "beverage_class": "beverage class",
    "label_width_mm": "label width",
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedManifest:
    """What could be read from a manifest.

    `rows` holds only the rows that validated. `total_rows` counts every
    non-blank data row, valid or not, because that is the number the agent
    sees in their spreadsheet — reporting "12 rows" for a 14-row sheet with
    two invalid rows would make them hunt for rows that were never lost.
    """

    rows: tuple[ManifestRow, ...]
    total_rows: int
    issues: tuple[PairingIssue, ...]
    missing_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pair:
    """A manifest row matched to one uploaded file.

    `image_index` is the file's position in the list given to `pair_files`,
    which is how the caller finds the bytes — filenames alone are not a key,
    because the list may legitimately contain the same name twice.
    """

    row: ManifestRow
    image_index: int
    filename: str


@dataclass(frozen=True, slots=True)
class Pairing:
    """Pairs ready to run, plus the report shown to the agent."""

    pairs: tuple[Pair, ...]
    report: PairingReport


@dataclass(slots=True)
class _Issues:
    """Collects issues in a stable order for the agent to read top to bottom."""

    items: list[PairingIssue] = field(default_factory=list)

    def add(
        self,
        kind: PairingIssueKind,
        message: str,
        *,
        filename: str | None = None,
        row_number: int | None = None,
    ) -> None:
        self.items.append(
            PairingIssue(kind=kind, message=message, filename=filename, row_number=row_number)
        )


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def basename(name: str) -> str:
    """The final path component, whichever separator the client used.

    A browser folder upload sends `webkitRelativePath`-style names such as
    `import-0412/labels/old_tom.jpg`, while the manifest usually names just
    `old_tom.jpg`. A Windows user may also paste `C:\\labels\\old_tom.jpg` into
    the spreadsheet. Both separators are stripped so that all of these pair.
    """
    return re.split(r"[\\/]", name.strip())[-1].strip()


def is_os_clutter(name: str) -> bool:
    """Whether a file is operating-system debris rather than something the agent chose."""
    base = basename(name).casefold()
    return base in _OS_CLUTTER or base.startswith("._")


def has_accepted_extension(name: str) -> bool:
    """Whether a filename looks like a raster image this service can read."""
    base = basename(name).casefold()
    dot = base.rfind(".")
    return dot > 0 and base[dot:] in ACCEPTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    """Decode manifest bytes, tolerating what spreadsheet software writes.

    UTF-8 with or without a byte order mark is the expected case (`utf-8-sig`
    strips the mark when present; left in, it would glue itself to the first
    header and make `filename` look missing). UTF-16 with a BOM is what Excel
    writes for "Unicode Text". Windows-1252 is the fallback because Excel's
    plain "CSV" export on Windows still uses it, and a brand name like "Añejo"
    must not turn a whole manifest into a decoding error.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def normalise_header(name: str) -> str:
    """Map a header cell to a column key: case, spacing and hyphens ignored.

    "Brand Name", " brand_name ", "BRAND-NAME" all mean `brand_name`. Agents
    retype headers, and a manifest rejected because of a capital letter would
    teach them that the tool is fussy about the wrong things.
    """
    return re.sub(r"[\s\-]+", "_", name.strip().casefold()).strip("_")


def _plain_list(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _parse_width(raw: str) -> tuple[float | None, str | None]:
    """Parse `label_width_mm`, returning (value, problem).

    A trailing "mm" is accepted because people type units. Anything else that
    is not a positive finite number is a problem described in words.
    """
    text = raw.strip()
    if not text:
        return None, None
    text = re.sub(r"\s*mm\s*$", "", text, flags=re.IGNORECASE)
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None, f'the label width "{raw}" is not a number of millimetres'
    if not math.isfinite(value) or value <= 0:
        return None, f'the label width "{raw}" must be a positive number of millimetres'
    return value, None


def parse_manifest(
    data: bytes, *, beverage_classes: Collection[str] | None = None
) -> ParsedManifest:
    """Read a manifest CSV into validated rows and plain-language issues.

    Args:
        data: The manifest file as uploaded.
        beverage_classes: Class ids the rule set supports. When given, a row
            naming another class is reported here, before upload, instead of
            failing later as a per-label error.

    Returns:
        The valid rows, the count of all non-blank data rows, and any issues.
        Every problem an agent could have made in a spreadsheet row comes
        back as an issue.

    Raises:
        ManifestRejected: the file as a whole cannot be used — not CSV, or
            over the row or cell-size limits.
    """
    issues = _Issues()
    text = _decode(data)
    records = _read_records(text)

    # The header is the first non-blank record; blank lines above it are
    # tolerated but still counted, so row numbers keep matching the sheet.
    header_index = next(
        (i for i, record in enumerate(records) if any(cell.strip() for cell in record)), None
    )
    if header_index is None:
        for column in MANIFEST_REQUIRED_COLUMNS:
            issues.add(
                PairingIssueKind.MISSING_COLUMN,
                f'The manifest is empty. It needs a header row with a "{column}" column.',
            )
        return ParsedManifest(
            rows=(),
            total_rows=0,
            issues=tuple(issues.items),
            missing_columns=MANIFEST_REQUIRED_COLUMNS,
        )

    header = [normalise_header(cell) for cell in records[header_index]]
    positions: dict[str, int] = {}
    for position, key in enumerate(header):
        # First occurrence wins: a duplicated header column is almost always a
        # copy-paste leftover, and reading the leftmost matches what a person
        # scanning the sheet would take as "the" column.
        if key in MANIFEST_COLUMNS and key not in positions:
            positions[key] = position

    missing = tuple(c for c in MANIFEST_REQUIRED_COLUMNS if c not in positions)
    for column in missing:
        issues.add(
            PairingIssueKind.MISSING_COLUMN,
            f'The manifest has no "{column}" column. Download the template to see the '
            "expected headings.",
        )
    if missing:
        total = sum(1 for record in records[header_index + 1 :] if any(c.strip() for c in record))
        return ParsedManifest(
            rows=(), total_rows=total, issues=tuple(issues.items), missing_columns=missing
        )

    rows: list[ManifestRow] = []
    total_rows = 0

    for index in range(header_index + 1, len(records)):
        record = records[index]
        # Spreadsheet row numbering: the header record sits at row
        # header_index + 1, and every record after it — blank or not — is one
        # row further down, exactly as Excel numbers them.
        row_number = index + 1
        if not any(cell.strip() for cell in record):
            continue  # blank rows, trailing or not, are layout rather than data
        total_rows += 1

        def cell(column: str, record: list[str] = record) -> str:
            position = positions.get(column)
            if position is None or position >= len(record):
                return ""
            return record[position].strip()

        problems: list[str] = []
        empty = [_COLUMN_WORDS[c] for c in MANIFEST_REQUIRED_COLUMNS if not cell(c)]
        if empty:
            problems.append(f"the {_plain_list(empty)} {'is' if len(empty) == 1 else 'are'} blank")

        filename = basename(cell("filename"))
        if cell("filename") and not filename:
            problems.append(f'"{cell("filename")}" is a folder, not a file name')

        beverage_class = cell("beverage_class") or "distilled_spirits"
        if beverage_classes is not None and beverage_class not in beverage_classes:
            supported = ", ".join(sorted(beverage_classes))
            problems.append(
                f'the beverage class "{beverage_class}" is not supported (supported: {supported})'
            )

        width, width_problem = _parse_width(cell("label_width_mm"))
        if width_problem:
            problems.append(width_problem)

        if not problems:
            try:
                rows.append(
                    ManifestRow(
                        row_number=row_number,
                        filename=filename,
                        brand_name=cell("brand_name"),
                        class_type=cell("class_type"),
                        alcohol_content=cell("alcohol_content"),
                        net_contents=cell("net_contents"),
                        beverage_class=beverage_class,
                        label_width_mm=width,
                    )
                )
                continue
            except ValidationError as exc:  # pragma: no cover - guarded above
                problems.append("; ".join(e["msg"] for e in exc.errors()))

        issues.add(
            PairingIssueKind.INVALID_ROW,
            f"Row {row_number} was skipped: {'; '.join(problems)}.",
            filename=filename or None,
            row_number=row_number,
        )

    if total_rows == 0:
        issues.add(
            PairingIssueKind.INVALID_ROW,
            "The manifest has a header row but no label rows beneath it.",
        )

    return ParsedManifest(rows=tuple(rows), total_rows=total_rows, issues=tuple(issues.items))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One file offered for pairing.

    `problem` is set when the file was received but cannot be used — wrong
    type, empty, too large. The check endpoint judges type from the name; the
    upload endpoint also knows the content type and size, and says why.
    """

    name: str
    problem: str | None = None


def _rows_phrase(numbers: Iterable[int]) -> str:
    ordered = sorted(numbers)
    return "Rows " + _plain_list([str(n) for n in ordered])


def pair_files(parsed: ParsedManifest, files: Sequence[str | FileEntry]) -> Pairing:
    """Match manifest rows to files by name and report everything that did not match.

    Matching is by basename. An exact match is taken first; a row with no
    exact match falls back to a case-insensitive match only when that is
    unambiguous in both directions — one row, one file. Camera software writes
    `IMG_0412.JPG`, people type `img_0412.jpg`, and that difference is not
    worth a round trip; but when case is the *only* thing distinguishing two
    candidates, the tool cannot know which one was meant, so it does not pick.

    Args:
        parsed: Output of `parse_manifest`.
        files: Filenames in upload order, or `FileEntry`s carrying a reason a
            file cannot be used. Operating-system clutter is dropped silently.

    Returns:
        The runnable pairs, in upload order, and the report for the agent.
    """
    issues = _Issues()
    issues.items.extend(parsed.issues)

    entries = [f if isinstance(f, FileEntry) else FileEntry(name=f) for f in files]

    # -- files: drop clutter, flag unusable, find duplicate names -------------
    usable: dict[str, list[int]] = defaultdict(list)  # basename -> indexes
    total_images = 0
    unsupported_names: dict[str, str] = {}  # casefolded basename -> problem

    for index, entry in enumerate(entries):
        name = basename(entry.name)
        if not name or is_os_clutter(entry.name):
            continue
        total_images += 1
        problem = entry.problem
        if problem is None and not has_accepted_extension(name):
            problem = (
                "is not a supported image type. Use JPEG, PNG, WebP, TIFF or BMP; "
                "PDF artwork is not handled by this tool"
            )
        if problem is not None:
            unsupported_names[name.casefold()] = problem
            issues.add(
                PairingIssueKind.UNSUPPORTED_FILE,
                f"{name} {problem}.",
                filename=name,
            )
            continue
        usable[name].append(index)

    duplicate_images = {name for name, indexes in usable.items() if len(indexes) > 1}
    for name in sorted(duplicate_images):
        issues.add(
            PairingIssueKind.DUPLICATE_IMAGE,
            f"{len(usable[name])} images named {name} were selected (probably from different "
            "folders). None of them was checked, because there is no way to tell which one "
            "the manifest means. Rename or remove the extras.",
            filename=name,
        )

    # -- rows: find filenames named more than once ---------------------------
    rows_by_name: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in parsed.rows:
        rows_by_name[row.filename].append(row)

    duplicate_rows = {name for name, rows in rows_by_name.items() if len(rows) > 1}
    for name in sorted(duplicate_rows, key=lambda n: rows_by_name[n][0].row_number):
        numbers = [r.row_number for r in rows_by_name[name]]
        issues.add(
            PairingIssueKind.DUPLICATE_ROW,
            f"{_rows_phrase(numbers)} all name {name}. None of them was used, because there "
            "is no way to tell which application record belongs to the image. Remove the "
            "extra rows.",
            filename=name,
            row_number=numbers[0],
        )

    # Only single-row, single-image names are candidates from here on.
    open_images = {name: idx[0] for name, idx in usable.items() if name not in duplicate_images}
    open_rows = {name: rows[0] for name, rows in rows_by_name.items() if name not in duplicate_rows}

    matched: dict[str, tuple[ManifestRow, int, str]] = {}  # row filename -> pair

    # Exact matches first.
    for name, row in open_rows.items():
        if name in open_images:
            matched[name] = (row, open_images[name], name)

    # Case-insensitive fallback, unambiguous both ways.
    taken_images = {image_name for _, _, image_name in matched.values()}
    leftover_images: dict[str, list[str]] = defaultdict(list)
    for name in open_images:
        if name not in taken_images:
            leftover_images[name.casefold()].append(name)
    leftover_rows: dict[str, list[str]] = defaultdict(list)
    for name in open_rows:
        if name not in matched:
            leftover_rows[name.casefold()].append(name)

    ambiguous_case: set[str] = set()
    for folded, row_names in leftover_rows.items():
        image_names = leftover_images.get(folded, [])
        if len(row_names) == 1 and len(image_names) == 1:
            image_name = image_names[0]
            matched[row_names[0]] = (open_rows[row_names[0]], open_images[image_name], image_name)
            taken_images.add(image_name)
        elif image_names:
            ambiguous_case.add(folded)

    # -- rows left without an image -------------------------------------------
    folded_duplicate_images = {n.casefold() for n in duplicate_images}
    for name, row in sorted(open_rows.items(), key=lambda item: item[1].row_number):
        if name in matched:
            continue
        folded = name.casefold()
        if folded in unsupported_names or folded in folded_duplicate_images:
            continue  # already explained by the file-side issue
        if folded in ambiguous_case:
            message = (
                f"Row {row.row_number} names {name}. More than one selected image has that "
                "name apart from capital letters, so none was used. Make the filename in the "
                "manifest match one image exactly."
            )
        else:
            message = (
                f"Row {row.row_number} names {name}, but no image with that name was selected."
            )
        issues.add(
            PairingIssueKind.ROW_WITHOUT_IMAGE, message, filename=name, row_number=row.row_number
        )

    # -- images left without a row --------------------------------------------
    folded_duplicate_rows = {n.casefold() for n in duplicate_rows}
    invalid_row_names = {
        i.filename.casefold()
        for i in parsed.issues
        if i.kind is PairingIssueKind.INVALID_ROW and i.filename
    }
    for name, _index in sorted(open_images.items(), key=lambda item: item[1]):
        if name in taken_images:
            continue
        folded = name.casefold()
        if folded in folded_duplicate_rows or folded in invalid_row_names:
            continue  # the row-side issue already says why this image was not used
        if folded in ambiguous_case:
            message = (
                f"{name} was not checked: more than one manifest row names it apart from "
                "capital letters. Make one row match the filename exactly."
            )
        else:
            message = f"{name} was selected, but no row in the manifest names it."
        issues.add(PairingIssueKind.IMAGE_WITHOUT_ROW, message, filename=name)

    pairs = sorted(
        (
            Pair(row=row, image_index=index, filename=image_name)
            for row, index, image_name in matched.values()
        ),
        key=lambda p: p.image_index,
    )

    blocking = bool(parsed.missing_columns) or not pairs
    report = PairingReport(
        matched=len(pairs),
        total_images=total_images,
        total_rows=parsed.total_rows,
        issues=tuple(issues.items),
        blocking=blocking,
    )
    return Pairing(pairs=tuple(pairs), report=report)


def template_csv() -> str:
    """The downloadable manifest template: header plus the brief's example row."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(MANIFEST_COLUMNS)
    writer.writerow([TEMPLATE_EXAMPLE_ROW[c] for c in MANIFEST_COLUMNS])
    return buffer.getvalue()
