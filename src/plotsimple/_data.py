"""Reading data and turning what the user gives us into a pandas table.

Charts always draw from pandas. Polars tables and big files are read with Polars
(fast mode) and handed over as pandas.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._errors import PlotsimpleError, column_error, suggest

BIG_FILE_BYTES = 100 * 1024 * 1024  # 100 MB: above this, fast mode is used automatically

_READERS = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "tsv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".json": "json",
}


def _is_polars(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "polars"


def _polars_to_pandas(obj) -> pd.DataFrame:
    if hasattr(obj, "collect"):  # LazyFrame
        obj = obj.collect()
    if hasattr(obj, "to_frame") and not hasattr(obj, "columns"):  # Series
        obj = obj.to_frame()
    return obj.to_pandas()


def _missing_file_error(path: Path) -> PlotsimpleError:
    lines = [f'File "{path}" not found.']
    folder = path.parent if str(path.parent) not in ("", ".") else Path(".")
    try:
        nearby = [p.name for p in folder.iterdir() if p.is_file()]
    except OSError:
        nearby = []
    guess = suggest(path.name, nearby, n=1)
    if guess:
        lines.append(f'Did you mean "{folder / guess[0]}"?')
    lines.append(f"plotsimple is looking in: {Path(folder).resolve()}")
    return PlotsimpleError("\n".join(lines))


def read_file(path, sheet=None, fast: bool = False, keep_fast: bool = False):
    """Read a file. Big files use the fast engine; keep_fast=True keeps its Polars table."""
    path = Path(path)
    if not path.exists():
        raise _missing_file_error(path)
    kind = _READERS.get(path.suffix.lower())
    if kind is None:
        raise PlotsimpleError(
            f'plotsimple cannot read "{path.suffix}" files yet.\n'
            "It can read: " + ", ".join(sorted(_READERS))
        )

    asked = fast
    fast = asked or path.stat().st_size > BIG_FILE_BYTES
    if fast and kind in ("csv", "tsv", "parquet"):
        try:
            import polars as pl
        except ImportError:
            if asked:
                raise PlotsimpleError(
                    "fast=True needs the fast engine.\n"
                    "Install it with:  pip install plotsimple[fast]"
                ) from None
        else:
            if not asked:
                print("This file is large, so plotsimple is using its fast mode.")
            if kind == "parquet":
                fast_table = pl.read_parquet(path)
            else:
                sep = "\t" if kind == "tsv" else ","
                fast_table = pl.read_csv(path, separator=sep, infer_schema_length=10000)
            return fast_table if keep_fast else fast_table.to_pandas()

    if kind == "csv":
        return pd.read_csv(path, sep=None, engine="python")
    if kind == "tsv":
        return pd.read_csv(path, sep="\t")
    if kind == "excel":
        if sheet is None:
            return pd.read_excel(path)
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            sheets = pd.ExcelFile(path).sheet_names
            lines = [f'Sheet "{sheet}" not found in {path.name}.']
            guess = suggest(sheet, sheets, n=1)
            if guess:
                lines.append(f'Did you mean "{guess[0]}"?')
            lines.append("Sheets you have: " + ", ".join(sheets))
            raise PlotsimpleError("\n".join(lines)) from None
    if kind == "parquet":
        return pd.read_parquet(path)
    return pd.read_json(path)


def load(source=None, sheet=None, fast: bool = False) -> pd.DataFrame:
    """Turn anything a beginner might give us into a pandas table."""
    if source is None:
        return pd.DataFrame()
    if getattr(source, "_is_plotsimple_table", False):
        return load(source._frame.to_native())
    if _is_plotsimple_numbers(source):
        return source._as_pandas(for_chart=True)
    if isinstance(source, (str, os.PathLike)):
        return read_file(source, sheet=sheet, fast=fast)
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
            raise PlotsimpleError(
                "The lists in your table have different lengths.\n"
                + ", ".join(f"{k}: {len(v)}" for k, v in source.items() if hasattr(v, "__len__"))
            ) from None
    if isinstance(source, (list, tuple, np.ndarray)):
        arr = np.asarray(source)
        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        if arr.ndim == 2:
            return pd.DataFrame(arr)
    raise PlotsimpleError(
        f"plotsimple does not know how to use a {type(source).__name__}.\n"
        "Give it a file name, a table, or a list of numbers."
    )


def _is_plotsimple_numbers(obj) -> bool:
    """ps.numbers(...) or ps.number_grid(...)."""
    return bool(getattr(obj, "_is_plotsimple_numbers", False) or getattr(obj, "_is_plotsimple_grid", False))


def is_listlike(value) -> bool:
    return (isinstance(value, (list, tuple, np.ndarray, pd.Series, range)) or _is_polars(value)
            or bool(getattr(value, "_is_plotsimple_numbers", False)))


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
            raise PlotsimpleError(
                "Your lists have different lengths: "
                + ", ".join(f"{k} has {n}" for k, n in lengths.items())
            )
        n = next(iter(lengths.values()))
        if len(data.columns) and len(data) == n:
            table = data.copy()
        elif len(data.columns) and any(isinstance(v, str) for v in roles.values() if v is not None):
            raise PlotsimpleError(
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
            raise PlotsimpleError(
                f"{role}= should be a column name or a list of values, not {value!r}."
            )
    return table, names


def load_native(source=None, sheet=None, fast: bool = False):
    """Like load(), but Polars tables (and big files) stay in Polars. Used by tables."""
    if getattr(source, "_is_plotsimple_table", False):
        return source._frame.to_native()
    if _is_plotsimple_numbers(source):
        return source._as_pandas()
    if _is_polars(source):
        if hasattr(source, "collect"):  # LazyFrame
            source = source.collect()
        if not hasattr(source, "columns"):  # Series
            source = source.to_frame()
        return source
    if isinstance(source, (str, os.PathLike)):
        return read_file(source, sheet=sheet, fast=fast, keep_fast=True)
    return load(source, sheet=sheet, fast=fast)
