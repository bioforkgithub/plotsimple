"""The heart of simplydata tables.

A table holds its data in a narwhals frame, so every word is written once and runs on
both pandas (normal files) and Polars (big files, Polars tables, fast=True).

Every step that changes the table goes through self._change(...), which remembers the
table as it was before. That gives undo(), steps() and start_again() for free.
Summaries never change the table; they return a new, small table.
"""

from __future__ import annotations

from pathlib import Path

import narwhals as nw

from .._data import load_native
from .._errors import (SimplydataError, choice_error, column_error, friendly, missing_word_error, old_word,
                       suggest)

HISTORY_LIMIT = 30  # how many steps undo() can take back
SAVE_TYPES = ("csv", "tsv", "xlsx", "parquet", "json")


def show(native):
    """Show a table nicely: a real table in Jupyter/VS Code, text elsewhere."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            display(native)
            return
    except ImportError:
        pass
    print(native)


class BaseTable:
    """A table. Make one with sdata.table(...)."""

    _is_simplydata_table = True

    # Old names that still work for one release (with a note the first time): old -> new.
    _OLD_WORDS = {
        "show_first": "show_first_rows",
        "show_last": "show_last_rows",
        "count_empty": "count_empty_cells",
        "remove_empty_rows": "remove_rows_with_empty_cells",
        "fill_empty": "fill_empty_cells",
        "done": ("as_pandas", "_done"),  # done() gave back Polars for Polars tables, so it still does
    }

    def __init__(self, data=None, sheet=None, fast: bool = False, _frame=None):
        if _frame is None:
            native = load_native(data, sheet=sheet, fast=fast)
            if fast and type(native).__module__.split(".")[0] == "pandas":
                try:
                    import polars as pl
                except ImportError:
                    raise SimplydataError(
                        "fast=True needs the fast engine.\nInstall it with:  pip install simplydata[fast]"
                    ) from None
                native = pl.from_pandas(native)
            _frame = nw.from_native(native, eager_only=True)
        object.__setattr__(self, "_frame", _frame)
        object.__setattr__(self, "_original", _frame)
        object.__setattr__(self, "_history", [])  # list of (what was done, table before)

    # ------------------------------------------------------------------ helpers for words
    @property
    def _columns(self) -> list[str]:
        return list(self._frame.columns)

    def _need_columns(self, *names):
        """Friendly error at once if a column does not exist."""
        for name in names:
            if name not in self._frame.columns:
                raise column_error(name, self._frame.columns)

    @staticmethod
    def _check_choice(step: str, option: str, value, choices):
        if value not in choices:
            raise choice_error(step, option, value, list(choices))

    def _change(self, what: str, frame):
        """Replace the table with a changed one and remember the old one for undo().

        pandas tables are numbered again 0, 1, 2 ... after every step, so a filtered table
        never shows gaps such as 0, 3, 5. (Polars tables have no row numbers.) This is the
        only place that does it, and undo() gives back a table that was numbered the same way.
        """
        frame = nw.maybe_reset_index(frame)
        self._history.append((what, self._frame))
        if len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)
        object.__setattr__(self, "_frame", frame)
        return self

    def _new_table(self, frame):
        """A new table (for summaries), using the same engine."""
        return type(self)(_frame=frame)

    @property
    def _engine(self) -> str:
        return "Polars" if self._frame.implementation.is_polars() else "pandas"

    # ------------------------------------------------------------------ history words
    @friendly
    def undo(self):
        """Take the last step back. table.undo()"""
        if not self._history:
            print("Nothing to undo: this table has no steps yet.")
            return self
        what, before = self._history.pop()
        object.__setattr__(self, "_frame", before)
        print(f"Undone: {what}")
        return self

    @friendly
    def steps(self):
        """List every step you did to this table, in order. table.steps()"""
        if not self._history:
            print("No steps yet.")
        else:
            for i, (what, _) in enumerate(self._history, 1):
                print(f"{i}. {what}")
        return self

    @friendly
    def start_again(self):
        """Go back to the table as it was at the start. table.start_again()"""
        return self._change("start again", self._original)

    # ------------------------------------------------------------------ finishing words
    @friendly
    def as_pandas(self):
        """Give back the table as a plain pandas table, for other Python tools. df = table.as_pandas()

        Only needed when another library wants pandas: simplydata words never need it.
        """
        return self._frame.to_pandas()

    @friendly
    def as_polars(self):
        """Give back the table as a plain Polars table, for other Python tools. df = table.as_polars()

        Needs Polars (pip install simplydata[fast]). simplydata words never need it.
        """
        try:
            import polars  # noqa: F401
        except ImportError:
            raise SimplydataError(
                "as_polars() needs Polars, the fast engine.\n"
                "Install it with:  pip install simplydata[fast]\n"
                "Or use table.as_pandas() to get a pandas table instead."
            ) from None
        return self._frame.to_polars()

    def _done(self):
        """The old done(): the table as it is stored (pandas, or Polars for Polars tables)."""
        return self._frame.to_native()

    @friendly
    def chart(self):
        """Start a chart from this table. chart = table.chart()"""
        from ..chart import chart

        return chart(self)

    @friendly
    def save(self, file_name):
        """Save the table; the end of the file name decides the type. table.save("clean.xlsx")

        File types: .xlsx (Excel), .csv, .tsv, .parquet and .json.
        """
        path = Path(file_name)
        kind = path.suffix.lower().lstrip(".")
        if not kind:
            raise SimplydataError(
                f'Add a file type to the name, for example "{path.name}.xlsx" or "{path.name}.csv".'
            )
        if kind not in SAVE_TYPES:
            raise choice_error("save", "file type", kind, SAVE_TYPES)
        if path.parent and not path.parent.exists():
            raise SimplydataError(f'The folder "{path.parent}" does not exist. Create it first.')
        if kind == "csv":
            self._frame.write_csv(str(path))
        elif kind == "parquet":
            self._frame.write_parquet(str(path))
        else:
            pandas_table = self._frame.to_pandas()
            if kind == "tsv":
                pandas_table.to_csv(path, sep="\t", index=False)
            elif kind == "xlsx":
                pandas_table.to_excel(path, index=False)
            else:
                pandas_table.to_json(path, orient="records", indent=2)
        return self

    # ------------------------------------------------------------------ looking words
    @friendly
    def show_first_rows(self, rows=5):
        """Show the first rows. table.show_first_rows(5)"""
        show(self._frame.head(int(rows)).to_native())
        return None

    @friendly
    def show_last_rows(self, rows=5):
        """Show the last rows. table.show_last_rows(5)"""
        show(self._frame.tail(int(rows)).to_native())
        return None

    @friendly
    def column_names(self):
        """The names of all columns. table.column_names()"""
        return self._columns

    # ------------------------------------------------------------------ Python details
    def __len__(self):
        return len(self._frame)

    def __repr__(self):
        rows, cols = self._frame.shape
        return f"simplydata table: {rows} rows, {cols} columns\n{self._frame.head(10).to_native()!r}"

    def _repr_html_(self):
        native = self._frame.to_native()
        html = getattr(native, "_repr_html_", None)
        return html() if html else None

    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._OLD_WORDS:
            return old_word(self, name, self._OLD_WORDS)
        raise missing_word_error(name, self._word_names())

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        guesses = suggest(name, self._word_names())
        hint = f"table.{guesses[0]}(...)" if guesses else 'a step such as table.rename_column("old", to="new")'
        raise SimplydataError(f"Steps are written with brackets, not with =.\nInstead of table.{name} = ..., write {hint}")
