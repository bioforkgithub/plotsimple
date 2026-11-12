"""Reading data and turning what the user gives us into a pandas table.

Charts always draw from pandas. Polars tables and big files are read with Polars
(fast mode) and handed over as pandas.

Files can be read one at a time, several at once (a folder, or a pattern such as
"data/*.csv", stacked with a "file" column), or from a web address.
"""

from __future__ import annotations

import codecs
import csv
import glob
import io
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ._errors import SimplydataError, column_error, suggest

BIG_FILE_BYTES = 100 * 1024 * 1024  # 100 MB: above this, fast mode is used automatically
WEB_TIMEOUT_SECONDS = 30

_READERS = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "tsv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".json": "json",
    ".docx": "word",   # optional extra: python-docx
    ".sav": "spss",    # optional extra: pyreadstat
    ".zsav": "spss",
    ".dta": "stata",
}

_KIND_NAMES = {
    "csv": "CSV and text", "tsv": "TSV", "excel": "Excel", "parquet": "Parquet", "json": "JSON",
    "word": "Word", "spss": "SPSS", "stata": "Stata",
}

# Which reading options make sense for which kind of file.
_OPTIONS_FOR = {
    "separator": ("csv", "tsv"),
    "encoding": ("csv", "tsv"),
    "decimal_mark": ("csv", "tsv", "excel", "word"),
    "thousands_mark": ("csv", "tsv", "excel", "word"),
    "empty_values": ("csv", "tsv", "excel", "word"),
    "skip_rows": ("csv", "tsv", "excel", "word"),
    "table_number": ("word",),
}

READING_OPTIONS = tuple(_OPTIONS_FOR)

_FILES_HINT = "Install it with:  pip install simplydata[files]"


def _is_polars(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "polars"


def _polars_to_pandas(obj) -> pd.DataFrame:
    if hasattr(obj, "collect"):  # LazyFrame
        obj = obj.collect()
    if hasattr(obj, "to_frame") and not hasattr(obj, "columns"):  # Series
        obj = obj.to_frame()
    return obj.to_pandas()


def _missing_file_error(path: Path) -> SimplydataError:
    lines = [f'File "{path}" not found.']
    folder = path.parent if str(path.parent) not in ("", ".") else Path(".")
    try:
        nearby = [p.name for p in folder.iterdir() if p.is_file()]
    except OSError:
        nearby = []
    guess = suggest(path.name, nearby, n=1)
    if guess:
        lines.append(f'Did you mean "{folder / guess[0]}"?')
    lines.append(f"simplydata is looking in: {Path(folder).resolve()}")
    return SimplydataError("\n".join(lines))


# ---------------------------------------------------------------- reading options
def reading_options(separator=None, decimal_mark=None, thousands_mark=None, empty_values=None,
                    skip_rows=None, encoding=None, table_number=None) -> dict:
    """Check the reading options at once and give back only the ones that were given.

    sdata.table(...) and sdata.chart(...) pass their options through here, so every
    reading word checks them the same way.
    """
    options = {}
    if separator is not None:
        if not isinstance(separator, str) or separator == "":
            raise SimplydataError(
                "separator= is the character between the values on each line.\n"
                'For example:  sdata.table("lab.csv", separator=";")   or   separator="tab"'
            )
        options["separator"] = {"tab": "\t", "space": " "}.get(separator.lower(), separator)
    if decimal_mark is not None:
        if decimal_mark not in (".", ","):
            raise SimplydataError(
                f'decimal_mark= should be "." or ",", not {decimal_mark!r}.\n'
                'For example, when 3,5 means three and a half:  sdata.table("lab.csv", decimal_mark=",")'
            )
        options["decimal_mark"] = decimal_mark
    if thousands_mark is not None:
        if not isinstance(thousands_mark, str) or len(thousands_mark) != 1:
            raise SimplydataError(
                "thousands_mark= is the one character that groups thousands, as in 1.000.000 or 1 000 000.\n"
                'For example:  sdata.table("lab.csv", decimal_mark=",", thousands_mark=".")'
            )
        if thousands_mark == (decimal_mark or "."):
            raise SimplydataError(
                f'thousands_mark="{thousands_mark}" is the same as the decimal mark, so simplydata cannot '
                "tell 1.500 (one and a half) from 1.500 (fifteen hundred).\n"
                'Give both, for example:  sdata.table("lab.csv", decimal_mark=",", thousands_mark=".")'
            )
        options["thousands_mark"] = thousands_mark
    if empty_values is not None:
        if isinstance(empty_values, (str, int, float)) and not isinstance(empty_values, bool):
            empty_values = [empty_values]
        if (not isinstance(empty_values, (list, tuple, set))
                or not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in empty_values)):
            raise SimplydataError(
                'empty_values= is a list of the markers that mean "no value" in your file.\n'
                'For example:  sdata.table("lab.csv", empty_values=["NA", "-", "n/a"])'
            )
        options["empty_values"] = [str(v) for v in empty_values]
    if skip_rows is not None:
        if isinstance(skip_rows, bool) or not isinstance(skip_rows, (int, np.integer)) or skip_rows < 0:
            raise SimplydataError(
                "skip_rows= is how many lines to skip at the top, before the column names: a whole number.\n"
                'For example:  sdata.table("lab.csv", skip_rows=2)'
            )
        options["skip_rows"] = int(skip_rows)
    if encoding is not None:
        try:
            codecs.lookup(str(encoding))
        except LookupError:
            raise SimplydataError(
                f'simplydata does not know the encoding "{encoding}".\n'
                'Common ones: "utf-8" (most files), "latin-1" or "cp1252" (older Excel on Windows), "utf-16".\n'
                'For example:  sdata.table("lab.csv", encoding="latin-1")'
            ) from None
        options["encoding"] = str(encoding)
    if table_number is not None:
        good = table_number == "all" or (
            isinstance(table_number, (int, np.integer)) and not isinstance(table_number, bool) and table_number >= 1)
        if not good:
            raise SimplydataError(
                "table_number= picks a table in a Word file: 1 for the first, 2 for the second, "
                'or "all" to stack them.\nFor example:  sdata.table("report.docx", table_number=2)'
            )
        options["table_number"] = table_number if table_number == "all" else int(table_number)
    return options


def _check_options_fit(kind: str, options: dict, label: str) -> None:
    for option in options:
        kinds = _OPTIONS_FOR[option]
        if kind not in kinds:
            works = ", ".join(_KIND_NAMES[k] for k in kinds)
            raise SimplydataError(
                f'{option}= is for {works} files, so it cannot be used for "{label}" '
                f"({_KIND_NAMES[kind]}).\nLeave {option}= out for this file."
            )


def _check_reading_source(source, options) -> None:
    if options and not isinstance(source, (str, os.PathLike)):
        names = ", ".join(f"{k}=" for k in options)
        raise SimplydataError(
            f"{names} {'is' if len(options) == 1 else 'are'} for reading files, "
            "and this data is already a table.\n"
            'Use it with a file name, for example:  sdata.table("lab.csv", decimal_mark=",")'
        )


def _kind_of(suffix: str, label: str) -> str:
    kind = _READERS.get(suffix.lower())
    if kind is None:
        raise SimplydataError(
            f'simplydata cannot read "{suffix or label}" files yet.\n'
            "It can read: " + ", ".join(sorted(_READERS))
        )
    return kind


def _guess_separator(source) -> str:
    """The character between values in a text file (for the fast engine)."""
    try:
        if isinstance(source, io.BytesIO):
            sample = source.getvalue()[:65536]
        else:
            with open(source, "rb") as fh:
                sample = fh.read(65536)
        text = sample.decode("utf-8", errors="ignore")
        return csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except (csv.Error, OSError):
        return ","


def _rewind(source):
    if isinstance(source, io.BytesIO):
        source.seek(0)
    return source


# ---------------------------------------------------------------- one file
def read_file(path, sheet=None, fast: bool = False, keep_fast: bool = False, options: dict | None = None):
    """Read a file, a folder of files, a file pattern such as "data/*.csv", or a web address.

    Big files use the fast engine; keep_fast=True keeps its Polars table.
    """
    options = options or {}
    text = str(path)
    if _is_web_address(text):
        return _read_web(text, sheet=sheet, fast=fast, keep_fast=keep_fast, options=options)
    path = Path(path)
    if path.is_dir():
        return _read_many(_files_in_folder(path), sheet=sheet, fast=fast, keep_fast=keep_fast,
                          options=options, asked_for=text)
    if not path.exists() and _has_wildcards(text):
        return _read_many(_files_matching(text), sheet=sheet, fast=fast, keep_fast=keep_fast,
                          options=options, asked_for=text)
    if not path.exists():
        raise _missing_file_error(path)
    kind = _kind_of(path.suffix, path.name)
    return _read_one(path, kind, path.stat().st_size, path.name, sheet=sheet, fast=fast,
                     keep_fast=keep_fast, options=options)


def _read_one(source, kind, size, label, sheet=None, fast=False, keep_fast=False, options=None):
    """Read one file (a path, or the bytes of a downloaded file)."""
    options = options or {}
    _check_options_fit(kind, options, label)

    asked = fast
    fast = asked or size > BIG_FILE_BYTES
    polars_can = ("thousands_mark" not in options
                  and str(options.get("encoding", "utf-8")).lower().replace("_", "-") in ("utf-8", "utf8"))
    if fast and kind in ("csv", "tsv", "parquet") and polars_can:
        try:
            import polars as pl
        except ImportError:
            if asked:
                raise SimplydataError(
                    "fast=True needs the fast engine.\n"
                    "Install it with:  pip install simplydata[fast]"
                ) from None
        else:
            if not asked:
                print("This file is large, so simplydata is using its fast mode.")
            if kind == "parquet":
                fast_table = pl.read_parquet(_rewind(source))
            else:
                sep = options.get("separator") or ("\t" if kind == "tsv" else _guess_separator(source))
                extra = {}
                if options.get("decimal_mark") == ",":
                    extra["decimal_comma"] = True
                if "empty_values" in options:
                    extra["null_values"] = options["empty_values"]
                if "skip_rows" in options:
                    extra["skip_rows"] = options["skip_rows"]
                fast_table = pl.read_csv(_rewind(source), separator=sep, infer_schema_length=10000, **extra)
            return fast_table if keep_fast else fast_table.to_pandas()

    if kind in ("csv", "tsv"):
        return _read_text(source, kind, label, options)
    if kind == "excel":
        return _read_excel(source, label, sheet, options)
    if kind == "parquet":
        return pd.read_parquet(_rewind(source))
    if kind == "json":
        return pd.read_json(_rewind(source))
    if kind == "word":
        return _read_word(source, label, options)
    if kind == "spss":
        return _read_spss(source)
    return pd.read_stata(_rewind(source))


def _pandas_text_options(options: dict) -> dict:
    extra = {}
    if "decimal_mark" in options:
        extra["decimal"] = options["decimal_mark"]
    if "thousands_mark" in options:
        extra["thousands"] = options["thousands_mark"]
    if "empty_values" in options:
        extra["na_values"] = options["empty_values"]
    if "skip_rows" in options:
        extra["skiprows"] = options["skip_rows"]
    return extra


def _read_text(source, kind, label, options):
    extra = _pandas_text_options(options)
    if "encoding" in options:
        extra["encoding"] = options["encoding"]
    sep = options.get("separator") or ("\t" if kind == "tsv" else None)
    try:
        if sep is None:
            return pd.read_csv(_rewind(source), sep=None, engine="python", **extra)
        engine = {} if len(sep) == 1 else {"engine": "python"}
        return pd.read_csv(_rewind(source), sep=sep, **engine, **extra)
    except UnicodeDecodeError:
        raise SimplydataError(
            f'"{label}" is not saved as {options.get("encoding", "utf-8")} text.\n'
            "Tell simplydata how it is saved, for example:  "
            f'sdata.table("{label}", encoding="latin-1")\n'
            '"latin-1" or "cp1252" fit most files from older Excel on Windows.'
        ) from None
    except (pd.errors.ParserError, csv.Error) as problem:
        first_line = (str(problem).strip().splitlines() or [""])[0]
        raise SimplydataError(
            f'simplydata could not split "{label}" into columns ({first_line}).\n'
            "Tell it the character between the values, for example:  "
            f'sdata.table("{label}", separator=";")\n'
            "If there are notes above the column names, skip them:  "
            f'sdata.table("{label}", skip_rows=2)'
        ) from None


def _read_excel(source, label, sheet, options):
    extra = _pandas_text_options(options)
    if sheet is None:
        return pd.read_excel(_rewind(source), **extra)
    if isinstance(sheet, str) and sheet.lower() == "all":
        sheets = pd.read_excel(_rewind(source), sheet_name=None, **extra)
        parts = []
        for name, frame in sheets.items():
            if len(frame.columns) == 0:
                print(f'Sheet "{name}" is empty, so it is left out.')
                continue
            parts.append((str(name), frame))
        if not parts:
            raise SimplydataError(f'Every sheet in "{label}" is empty.')
        return _stack(parts, column="sheet", what="sheet",
                      example=f'sdata.table("{label}", sheet="{parts[0][0]}")', keep_fast=False)
    try:
        return pd.read_excel(_rewind(source), sheet_name=sheet, **extra)
    except ValueError:
        sheets = pd.ExcelFile(_rewind(source)).sheet_names
        lines = [f'Sheet "{sheet}" not found in {label}.']
        guess = suggest(sheet, sheets, n=1)
        if guess:
            lines.append(f'Did you mean "{guess[0]}"?')
        lines.append("Sheets you have: " + ", ".join(sheets))
        lines.append('sheet="all" reads every sheet and stacks them.')
        raise SimplydataError("\n".join(lines)) from None


def _read_word(source, label, options):
    """The tables in a Word file (.docx). Needs python-docx (an optional extra)."""
    try:
        import docx
    except ImportError:
        raise SimplydataError(f"Reading Word files needs python-docx.\n{_FILES_HINT}") from None
    tables = docx.Document(_rewind(source)).tables
    if not tables:
        raise SimplydataError(f'"{label}" has no tables in it. simplydata reads the tables of a Word file.')
    number = options.get("table_number", 1)

    def as_frame(word_table):
        # The cells go through the CSV reader, so numbers, decimal marks and empty values
        # are understood exactly as in a CSV file.
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t")
        for row in word_table.rows:
            writer.writerow([cell.text.strip() for cell in row.cells])
        buffer.seek(0)
        return pd.read_csv(buffer, sep="\t", **_pandas_text_options(options))

    if number == "all":
        parts = [(i, as_frame(t)) for i, t in enumerate(tables, 1)]
        return _stack(parts, column="table", what="table",
                      example=f'sdata.table("{label}", table_number=1)', keep_fast=False)
    if number > len(tables):
        raise SimplydataError(
            f'"{label}" has {len(tables)} table{"s" if len(tables) != 1 else ""}, so there is no table {number}.\n'
            f'For example:  sdata.table("{label}", table_number={len(tables)})'
        )
    if len(tables) > 1 and "table_number" not in options:
        print(f'"{label}" has {len(tables)} tables; reading table 1. '
              'Use table_number=2 for the next one, or table_number="all" to stack them.')
    return as_frame(tables[number - 1])


def _read_spss(source):
    """An SPSS file (.sav). pandas reads it with pyreadstat (an optional extra)."""
    try:
        import pyreadstat  # noqa: F401
    except ImportError:
        raise SimplydataError(f"Reading SPSS files needs pyreadstat.\n{_FILES_HINT}") from None
    if isinstance(source, io.BytesIO):  # pyreadstat needs a real file
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder) / "download.sav"
            temporary.write_bytes(source.getvalue())
            return pd.read_spss(temporary)
    return pd.read_spss(source)


# ---------------------------------------------------------------- several files
def _has_wildcards(text: str) -> bool:
    return any(c in text for c in "*?[")


def _natural_key(path: Path):
    """Sort file2 before file10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _readable(path: Path) -> bool:
    return path.is_file() and not path.name.startswith((".", "~$"))


def _list_some(names) -> str:
    names = sorted(names)
    return ", ".join(names[:15]) + (" ..." if len(names) > 15 else "")


def _files_in_folder(folder: Path) -> list[Path]:
    files = sorted((p for p in folder.iterdir() if _readable(p) and p.suffix.lower() in _READERS), key=_natural_key)
    if not files:
        there = [p.name for p in folder.iterdir() if p.is_file()]
        lines = [f'The folder "{folder}" has no files simplydata can read.',
                 "It can read: " + ", ".join(sorted(_READERS))]
        lines.append("Files in the folder: " + _list_some(there) if there else "The folder is empty.")
        raise SimplydataError("\n".join(lines))
    return files


def _files_matching(pattern: str) -> list[Path]:
    files = sorted((Path(p) for p in glob.glob(pattern, recursive=True) if _readable(Path(p))), key=_natural_key)
    if not files:
        folder = Path(pattern).parent
        while _has_wildcards(str(folder)):
            folder = folder.parent
        lines = [f'No files match "{pattern}".']
        if folder.is_dir():
            there = [p.name for p in folder.iterdir() if p.is_file()]
            lines.append(f'Files in "{folder}": ' + _list_some(there) if there else f'The folder "{folder}" is empty.')
        else:
            lines.append(f'The folder "{folder}" does not exist.')
            lines.append(f"simplydata is looking in: {Path('.').resolve()}")
        raise SimplydataError("\n".join(lines))
    return files


def _labels(files: list[Path]) -> list[str]:
    """Each file's name, with its folder only when needed to tell files apart."""
    full = [p.resolve() for p in files]
    common = Path(os.path.commonpath([p.parent for p in full]))
    return [p.relative_to(common).as_posix() for p in full]


def _read_many(files, sheet, fast, keep_fast, options, asked_for):
    labels = _labels(files)
    parts = []
    for path, label in zip(files, labels):
        kind = _kind_of(path.suffix, path.name)
        parts.append((label, _read_one(path, kind, path.stat().st_size, label, sheet=sheet, fast=fast,
                                       keep_fast=keep_fast, options=options)))
    is_folder = Path(asked_for).is_dir()
    folder = Path(asked_for) if is_folder else files[0].parent
    example = f'sdata.table("{(files[0].parent / files[0].name).as_posix()}")'
    if is_folder:
        example += (", or pick files with a pattern, for example "
                    f'sdata.table("{folder.as_posix()}/*{files[0].suffix}")')
    return _stack(parts, column="file", what="file", example=example, keep_fast=keep_fast)


def _is_number_column(native, name):
    """True for numbers, False for text, None when the column is all empty."""
    column = native[name]
    if _is_polars(native):
        if column.null_count() == len(column):
            return None
        return column.dtype.is_numeric()
    if column.isna().all():
        return None
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def _stack(parts, column, what, example, keep_fast):
    """Stack tables with the same columns, with a column saying where each row came from.

    parts is a list of (label, table); what is "file", "sheet" or "table".
    """
    def called(label):
        return {"file": f"{label}", "sheet": f'sheet "{label}"', "table": f"table {label}"}[what]

    first_label, first = parts[0]
    reference = list(first.columns)
    problems = []
    for label, frame in parts[1:]:
        names = list(frame.columns)
        problems += [f'  {called(label)} has no column "{n}"' for n in reference if n not in names]
        problems += [f'  {called(label)} has an extra column "{n}"' for n in names if n not in reference]
    if problems:
        shown = problems[:12] + ([f"  ... and {len(problems) - 12} more differences"] if len(problems) > 12 else [])
        raise SimplydataError(
            f"These {what}s do not all have the same columns, so simplydata cannot stack them:\n"
            + "\n".join(shown)
            + f"\nColumns in {called(first_label)}: " + ", ".join(map(str, reference))
            + f"\nMake the columns match, or read one {what} at a time, for example {example}."
        )

    name = column
    if name in reference:
        name = f"from_{column}"
        print(f'Your {what}s already have a column called "{column}", so the {what} names go in "{name}".')

    # A column with numbers in some parts and text in others becomes text everywhere.
    mixed = [col for col in reference
             if len({_is_number_column(frame, col) for _, frame in parts} - {None}) > 1]

    if keep_fast and any(_is_polars(frame) for _, frame in parts):
        import polars as pl

        frames = []
        for label, frame in parts:
            if not _is_polars(frame):
                frame = pl.from_pandas(frame)
            if mixed:
                frame = frame.with_columns([pl.col(c).cast(pl.String) for c in mixed])
            frames.append(frame.select(reference).with_columns(pl.lit(label).alias(name)).select([name, *reference]))
        stacked = pl.concat(frames, how="vertical_relaxed")
    else:
        frames = []
        for label, frame in parts:
            frame = (frame.to_pandas() if _is_polars(frame) else frame)[reference].copy()
            for col in mixed:
                values = frame[col]
                frame[col] = values.astype(object).where(values.isna(), values.astype(str))
            frame.insert(0, name, label)
            frames.append(frame)
        stacked = pd.concat(frames, ignore_index=True)

    for col in mixed:
        print(f'Note: column "{col}" holds numbers in some {what}s and text in others, so it is kept as text.')
    print(f"Read {len(parts)} {what}{'s' if len(parts) != 1 else ''} and stacked them: {len(stacked)} rows. "
          f'The column "{name}" says which {what} each row came from.')
    return stacked


# ---------------------------------------------------------------- web addresses
def _is_web_address(text: str) -> bool:
    return text.lower().startswith(("http://", "https://"))


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "simplydata"})
    try:
        with urllib.request.urlopen(request, timeout=WEB_TIMEOUT_SECONDS) as answer:
            return answer.read()
    except urllib.error.HTTPError as problem:
        meaning = {401: "a login is needed", 403: "not allowed", 404: "not found",
                   500: "the website had a problem"}.get(problem.code, "an error")
        raise SimplydataError(
            f"The web address answered {problem.code} ({meaning}):\n  {url}\n"
            "Check the address: open it in a web browser and see if the file downloads."
        ) from None
    except (urllib.error.URLError, OSError, ValueError) as problem:
        reason = getattr(problem, "reason", problem)
        raise SimplydataError(
            f"simplydata could not reach this web address ({reason}):\n  {url}\n"
            "Check that you are online and that the address is right.\n"
            'Or download the file and read it from your computer, for example sdata.table("data.csv").'
        ) from None


def _size_in_words(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} bytes"


def _read_web(url, sheet, fast, keep_fast, options):
    parts = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parts.path)).name or parts.netloc
    suffix = Path(name).suffix.lower()
    if suffix not in _READERS:
        wanted = urllib.parse.parse_qs(parts.query).get("format", [""])[0].lower()  # such as ?format=csv
        suffix = "." + wanted if wanted and "." + wanted in _READERS else ""
    if not suffix:
        print("The web address does not say what kind of file it is, so simplydata reads it as CSV text.")
        suffix = ".csv"
    data = _download(url)
    print(f"Downloaded {name} ({_size_in_words(len(data))}).")
    return _read_one(io.BytesIO(data), _READERS[suffix], len(data), name, sheet=sheet, fast=fast,
                     keep_fast=keep_fast, options=options)


# ---------------------------------------------------------------- anything into a table
def load(source=None, sheet=None, fast: bool = False, options: dict | None = None) -> pd.DataFrame:
    """Turn anything a beginner might give us into a pandas table.

    options are the checked reading options from reading_options(...); they are for files only.
    """
    _check_reading_source(source, options)
    if source is None:
        return pd.DataFrame()
    if getattr(source, "_is_simplydata_table", False):
        return load(source._frame.to_native())
    if _is_simplydata_numbers(source):
        return source._as_pandas(for_chart=True)
    if isinstance(source, (str, os.PathLike)):
        return read_file(source, sheet=sheet, fast=fast, options=options)
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, pd.Series):
        return source.to_frame()
    if _is_polars(source):
        return _polars_to_pandas(source)
    if isinstance(source, dict):
        try:
            return pd.DataFrame(source)
        except ValueError:
            raise SimplydataError(
                "The lists in your table have different lengths.\n"
                + ", ".join(f"{k}: {len(v)}" for k, v in source.items() if hasattr(v, "__len__"))
            ) from None
    if isinstance(source, (list, tuple, np.ndarray)):
        arr = np.asarray(source)
        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        if arr.ndim == 2:
            return pd.DataFrame(arr)
    raise SimplydataError(
        f"simplydata does not know how to use a {type(source).__name__}.\n"
        "Give it a file name, a table, or a list of numbers."
    )


def _is_simplydata_numbers(obj) -> bool:
    """sdata.numbers(...) or sdata.number_grid(...)."""
    return bool(getattr(obj, "_is_simplydata_numbers", False) or getattr(obj, "_is_simplydata_grid", False))


def is_listlike(value) -> bool:
    return (isinstance(value, (list, tuple, np.ndarray, pd.Series, range)) or _is_polars(value)
            or bool(getattr(value, "_is_simplydata_numbers", False)))


def check_columns(data: pd.DataFrame, **roles) -> None:
    """Check at call time that every named column exists (friendly error if not)."""
    for value in roles.values():
        if isinstance(value, str) and len(data.columns) and value not in data.columns:
            raise column_error(value, data.columns)
        if isinstance(value, str) and not len(data.columns):
            raise column_error(value, data.columns)


def frame_for(data: pd.DataFrame, **roles):
    """Build the table a drawing step will use.

    Each role (x=, y=, colour=, ...) can be a column name, a list of values, or None.
    Lists become new columns named after the role. Returns (table, names) where
    names maps each role to the column to use (or None).
    """
    lists = {k: v for k, v in roles.items() if v is not None and is_listlike(v)}
    names = {}
    if lists:
        lengths = {k: len(v) for k, v in lists.items()}
        if len(set(lengths.values())) > 1:
            raise SimplydataError(
                "Your lists have different lengths: "
                + ", ".join(f"{k} has {n}" for k, n in lengths.items())
            )
        n = next(iter(lengths.values()))
        if len(data.columns) and len(data) == n:
            table = data.copy()
        elif len(data.columns) and any(isinstance(v, str) for v in roles.values() if v is not None):
            raise SimplydataError(
                f"Your list has {n} values but your table has {len(data)} rows.\n"
                "Use column names, or lists of the same length as the table."
            )
        else:
            table = pd.DataFrame(index=range(n))
        for role, values in lists.items():
            if _is_polars(values):
                values = values.to_list()
            table[role] = list(values)
            names[role] = role
    else:
        table = data
    for role, value in roles.items():
        if role in names:
            continue
        if value is None:
            names[role] = None
        elif isinstance(value, str):
            if value not in table.columns:
                raise column_error(value, table.columns)
            names[role] = value
        else:
            raise SimplydataError(
                f"{role}= should be a column name or a list of values, not {value!r}."
            )
    return table, names


def load_native(source=None, sheet=None, fast: bool = False, options: dict | None = None):
    """Like load(), but Polars tables (and big files) stay in Polars. Used by tables."""
    _check_reading_source(source, options)
    if getattr(source, "_is_simplydata_table", False):
        return source._frame.to_native()
    if _is_simplydata_numbers(source):
        return source._as_pandas()
    if _is_polars(source):
        if hasattr(source, "collect"):  # LazyFrame
            source = source.collect()
        if not hasattr(source, "columns"):  # Series
            source = source.to_frame()
        return source
    if isinstance(source, (str, os.PathLike)):
        return read_file(source, sheet=sheet, fast=fast, keep_fast=True, options=options)
    return load(source, sheet=sheet, fast=fast)
