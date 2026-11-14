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
from .._errors import (MissingWordError, SimplydataError, choice_error, column_error, friendly, missing_word_error,
                       old_word, other_name_error, suggest)

HISTORY_LIMIT = 30  # how many steps undo() can take back
SHOW_LIMIT = 1000  # table.show() with no rows= shows at most this many rows
REPR_WHOLE_LIMIT = 10_000  # a Polars table up to this size is shown whole in notebooks
REPR_ENDS = 30  # a bigger one shows this many rows from the start and the end
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


def numbered(part, first: int = 0):
    """Rows of a table ready to show, numbered from 1 the way people count (display only).

    part is a narwhals frame of rows taken from a table; first is the position (from 0) of its
    first row in the whole table, so the last 3 rows of a 54-row table are shown as 52, 53, 54.
    The stored table is never changed: pandas tables keep their own index (as_pandas() gives
    0, 1, 2 ... as pandas users expect), and Polars tables, which have no row labels, are shown
    through pandas with the same numbers.
    """
    import pandas as pd

    native = part.to_native()
    if type(native).__module__.split(".")[0] == "pandas":
        shown = native.copy(deep=False)
    else:
        shown = part.to_pandas()
    shown.index = pd.RangeIndex(first + 1, first + 1 + len(shown))
    return shown


def numbered_at(part, positions):
    """Rows of a table ready to show, each numbered with its own position (from 0) in the whole table."""
    import pandas as pd

    shown = numbered(part)
    shown.index = pd.Index([int(p) + 1 for p in positions])
    return shown


def _show_whole(native):
    """Show a table without hiding any rows or columns behind "..."."""
    import pandas as pd

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        show(native)


class PickingError(SimplydataError, TypeError):
    """table[...] and loops over a table: a friendly error that is still the TypeError Python tools expect."""


def _slice_text(key: slice) -> str:
    return f"{'' if key.start is None else key.start}:{'' if key.stop is None else key.stop}"


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
        "fit_straight_line": "fit_line",  # the same word as chart.fit_line
    }

    def __init__(self, data=None, sheet=None, fast: bool = False, _frame=None, _reading=None):
        if _frame is None:
            native = load_native(data, sheet=sheet, fast=fast, options=_reading)
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
    def show(self, rows=None):
        """Show the whole table, every row and column. table.show()

        rows=10 shows only the first 10 rows. A table with more than 1000 rows shows its
        first 1000; give rows= to see more. Try it: sdata.table(sdata.example("marks")).show()
        """
        total = len(self._frame)
        if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0):
            raise SimplydataError(
                f"rows= is how many rows to show: a whole number, not {rows!r}.\n"
                'For example:  sdata.table(sdata.example("marks")).show(rows=10)'
            )
        if rows is None:
            rows = min(total, SHOW_LIMIT)
            if total > SHOW_LIMIT:
                print(f"This table has {total} rows; showing the first {SHOW_LIMIT}. "
                      f"table.show(rows={total}) shows them all.")
        rows = min(rows, total)
        _show_whole(numbered(self._frame.head(rows)))
        if rows < total:
            print(f"Showing {rows} of {total} rows, {len(self._columns)} columns.")
        else:
            print(f"{total} rows, {len(self._columns)} columns.")
        return None

    @friendly
    def show_first_rows(self, rows=5):
        """Show the first rows, numbered from 1. table.show_first_rows(5)"""
        show(numbered(self._frame.head(int(rows))))
        return None

    @friendly
    def show_last_rows(self, rows=5):
        """Show the last rows, with their row numbers in the whole table. table.show_last_rows(5)

        The last 3 rows of a table of 54 rows are numbered 52, 53 and 54, the numbers that
        table.keep_rows_by_position(52, to=54) takes.
        """
        part = self._frame.tail(int(rows))
        show(numbered(part, len(self._frame) - len(part)))
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
        return f"simplydata table: {rows} rows, {cols} columns\n{numbered(self._frame.head(10))!r}"

    def _repr_html_(self):
        """The table in Jupyter and VS Code, with rows numbered from 1 (as everywhere in simplydata)."""
        import pandas as pd

        total = len(self._frame)
        if self._frame.implementation.is_pandas() or total <= REPR_WHOLE_LIMIT:
            shown = numbered(self._frame)
        else:  # a big Polars table: only the rows a notebook shows are turned into pandas
            k = REPR_ENDS
            shown = pd.concat([numbered(self._frame.head(k)), numbered(self._frame.tail(k), total - k)])
        return shown._repr_html_()

    def __getitem__(self, key):
        raise PickingError(self._picking_help(key))

    def __iter__(self):
        raise PickingError(
            "A table is not gone through row by row with a loop: simplydata words work on every row at once.\n"
            "To look at one row write table.row(3) (rows count from 1); for one column, table.column(\"marks\")."
        )

    def _picking_help(self, key) -> str:
        """Why table[...] does not work, and the words that do (with the user's own column or rows)."""
        columns = self._columns
        lines = []
        if isinstance(key, str) and key in columns:
            lines.append(f'For the column "{key}" on its own write table.column("{key}").')
        elif isinstance(key, str):
            guess = suggest(key, columns, n=1)
            lines.append(f'There is no column "{key}".' + (f' Did you mean table.column("{guess[0]}")?' if guess else ""))
        elif isinstance(key, (list, tuple)) and key and all(isinstance(k, str) for k in key):
            lines.append("To keep these columns write table.keep_columns("
                         + ", ".join(f'"{k}"' for k in key) + ").")
        elif isinstance(key, int) and not isinstance(key, bool) and key >= 0:
            lines.append(f"In Python [{key}] is row {key + 1} counting from 1: table.row({key + 1}) shows it.")
        elif isinstance(key, int) and key == -1:
            lines.append('In Python [-1] is the last row: table.row("last") shows it.')
        elif isinstance(key, slice) and key.step in (None, 1) and all(
                v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0) for v in (key.start, key.stop)):
            first = (key.start or 0) + 1
            last = "last" if key.stop is None else key.stop
            if last == "last" or last >= first:
                shown = '"last"' if last == "last" else last
                lines.append(f"In Python [{_slice_text(key)}] means rows {first} to {last} counting from 1: "
                             f"table.keep_rows_by_position({first}, to={shown}).")
        lines.append(
            'Parts of a table are chosen with words, and positions count from 1:\n'
            '    table.column("marks")                     one column, as numbers\n'
            '    table.keep_columns("name", "marks")       some columns\n'
            '    table.keep_rows("marks > 50")             rows that meet a condition\n'
            '    table.keep_rows_by_position(10, to=20)    rows 10 to 20\n'
            '    table.cell_at(row=3, column="marks")      one cell\n'
            "Python's [ ] counts from 0, so simplydata does not use it. For a pandas table, use table.as_pandas()."
        )
        return "\n".join(lines)

    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n)))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._OLD_WORDS:
            return old_word(self, name, self._OLD_WORDS)
        frame = self.__dict__.get("_frame")
        if frame is not None and name in frame.columns:
            if frame.schema[name].is_numeric():
                what, rows = "its numbers", f'table.keep_rows("{name} > 50")'
            else:
                what, rows = "its values", f'table.keep_rows("{name}", equal_to="...")'
            raise MissingWordError(
                f'{name} is a column. For {what} write table.column("{name}"); to choose rows write {rows}.\n'
                "Columns are named in quotes inside a word, never after a dot."
            )
        raise other_name_error("table", name, self._word_names()) or missing_word_error(name, self._word_names())

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        guesses = suggest(name, self._word_names())
        hint = f"table.{guesses[0]}(...)" if guesses else 'a step such as table.rename_column("old", to="new")'
        raise SimplydataError(f"Steps are written with brackets, not with =.\nInstead of table.{name} = ..., write {hint}")
