"""Looking, choosing columns and rows, empty cells, fixing columns.

Every word is written with narwhals expressions, so it works the same on pandas and Polars.
Only make_dates() needs a little native code (text-to-date parsing that forgives bad cells).

What counts as an "empty cell" everywhere in this file:
a missing value (None, NaN, NaT), or text that is only spaces or nothing at all.
Empty cells never match a condition in keep_rows()/remove_rows(), except is_empty=.
"""

from __future__ import annotations

import datetime as _dt
import numbers

import narwhals as nw

from .._errors import PlotsimpleError, friendly


class _NotGiven:
    def __repr__(self):
        return "not given"


_NOT_GIVEN = _NotGiven()

# Conditions for keep_rows() / remove_rows(), in the order they are listed in messages.
CONDITIONS = (
    "above", "below", "at_least", "at_most", "equal_to", "not_equal_to",
    "between", "one_of", "contains", "starts_with", "ends_with", "is_empty",
)
COMPARE = ("above", "below", "at_least", "at_most")
TEXT_ONLY = ("contains", "starts_with", "ends_with")
FILL_CHOICES = ("average", "median", "value above", "value below")

# Text-to-date formats, tried in this order. Day comes before month (03/04/2026 is 3 April);
# month-first is only used when day-first cannot work (12/25/2026).
DATE_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y/%m/%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%d/%m/%Y", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d.%m.%Y", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d-%m-%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%m/%d/%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
)

_NUMBER = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
_THOUSANDS = r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$"  # 1,200 or 12,345.6
_DECIMAL_COMMA = r"^[+-]?\d*,\d+$"  # 12,5
_NEW = "__plotsimple_new__"


def _example(code: str, name: str = "marks") -> str:
    return f'\nFor example:  {code}   (with ps.example("{name}"))'


def _kind(dtype) -> str:
    """The kind of a column in plain words."""
    if dtype == nw.Boolean:
        return "true/false"
    if dtype.is_numeric():
        return "number"
    if dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum:
        return "text"
    if dtype == nw.Date or dtype == nw.Datetime:
        return "date"
    if dtype == nw.Object:
        return "mixed"
    return "other"


def _empty_in(frame, column):
    """True for every empty cell in the column (missing, NaN, or blank text)."""
    dtype = frame.schema[column]
    col = nw.col(column)
    empty = col.is_null()
    if dtype.is_float():
        empty = empty | col.is_nan()
    elif dtype == nw.String:
        empty = empty | (col.str.strip_chars() == "")
    return empty


def _count_in(frame, expr) -> int:
    """How many rows an expression is true for."""
    if len(frame) == 0:
        return 0
    return int(frame.select(expr.fill_null(False).cast(nw.Int64).sum()).item() or 0)


def _say(value) -> str:
    """A value written the way steps() shows it."""
    if isinstance(value, str):
        return f'"{value}"'
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, _dt.datetime):
        if value.hour or value.minute or value.second:
            return value.isoformat(sep=" ")
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if value is None:
        return "empty"
    return str(value)


def _nice(value) -> str:
    """A number or date written for people: 61, 61.5, 2026-03-15."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return str(round(value, 2))
    if isinstance(value, (_dt.date, _dt.datetime)) or hasattr(value, "to_pydatetime"):
        return _say(value)
    return str(value)


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _looks_like_number(text) -> bool:
    try:
        float(str(text).replace(",", "."))
        return True
    except ValueError:
        return False


def _rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


def _cells(n: int) -> str:
    return f"{n} cell" if n == 1 else f"{n} cells"


def _as_list(names) -> list:
    """Accept keep_columns("a", "b") as well as keep_columns(["a", "b"])."""
    out = []
    for name in names:
        if isinstance(name, (list, tuple)):
            out.extend(name)
        else:
            out.append(name)
    return out


def _check_count(step: str, n, example: str) -> int:
    if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 0:
        raise PlotsimpleError(f".{step}() needs a whole number of rows, like 5." + _example(example))
    return int(n)


def _parse_dates(native):
    """Parse a native text Series (pandas or Polars) into dates, trying DATE_FORMATS in order.

    Native code because narwhals' to_datetime stops at the first bad cell; here bad cells
    simply become empty. The same formats in the same order are used for both engines.
    """
    if type(native).__module__.split(".")[0] == "polars":
        import polars as pl

        return (
            pl.DataFrame({"s": native})
            .select(pl.coalesce([pl.col("s").str.strptime(pl.Datetime("us"), f, strict=False) for f in DATE_FORMATS]))
            .to_series()
            .rename(native.name)
        )
    import pandas as pd

    out = pd.Series(pd.NaT, index=native.index, dtype="datetime64[us]", name=native.name)
    for f in DATE_FORMATS:
        missing = out.isna()
        if not missing.any():
            break
        found = pd.to_datetime(native.where(missing), format=f, errors="coerce").astype("datetime64[us]")
        out = out.fillna(found)
    return out


class RowWords:
    # ================================================================== helpers
    def _kind_of(self, column) -> str:
        return _kind(self._frame.schema[column])

    def _empty(self, column):
        return _empty_in(self._frame, column)

    def _count(self, expr) -> int:
        return _count_in(self._frame, expr)

    def _need_one_column(self, step: str, column, example: str):
        if not isinstance(column, str):
            raise PlotsimpleError(
                f".{step}() needs a column name in quotes as the first thing." + _example(example)
            )
        self._need_columns(column)

    def _need_text(self, step: str, column, example: str):
        """Friendly error unless the column holds text. Gives the column as text."""
        self._need_one_column(step, column, example)
        kind = self._kind_of(column)
        if kind == "text":
            return nw.col(column).cast(nw.String)
        if kind == "mixed":
            raise PlotsimpleError(
                f'"{column}" holds a mix of things. Turn it into text first with '
                f'table.make_text("{column}").' + _example(example)
            )
        raise PlotsimpleError(
            f'.{step}() works on text, but "{column}" is a {kind} column.' + _example(example)
        )

    def _value_for(self, option: str, column: str, value, example: str):
        """Check one value against the column's kind and give it back ready to compare."""
        kind = self._kind_of(column)
        if kind == "number":
            if _is_number(value):
                return value
            if isinstance(value, str) and _looks_like_number(value):
                raise PlotsimpleError(
                    f'"{column}" is a number column, so write the number without quotes: '
                    f"{option}={value.strip()}" + _example(example)
                )
            raise PlotsimpleError(
                f'{option}= needs a number here, because "{column}" is a number column '
                f"(you gave {_say(value)})." + _example(example)
            )
        if kind == "text":
            if isinstance(value, str):
                return value
            hint = ""
            if _is_number(value):
                hint = (f' If "{column}" holds numbers written as text, first use '
                        f'table.make_numbers("{column}").')
            raise PlotsimpleError(
                f'"{column}" is a text column, so {option}= needs text in quotes, like '
                f'{option}="{value}".{hint}' + _example('table.keep_rows("class", equal_to="A")')
            )
        if kind == "true/false":
            if isinstance(value, bool):
                return value
            raise PlotsimpleError(
                f'"{column}" is a true/false column, so {option}= needs True or False (no quotes).'
            )
        if kind == "date":
            return self._date_value(option, column, value)
        return value

    def _date_value(self, option: str, column: str, value):
        is_datetime = self._frame.schema[column] == nw.Datetime
        if isinstance(value, str):
            try:
                value = _dt.datetime.fromisoformat(value.strip())
            except ValueError:
                raise PlotsimpleError(
                    f'"{column}" is a date column. Write dates as year-month-day, like '
                    f'{option}="2026-03-15".'
                ) from None
        if hasattr(value, "to_pydatetime"):  # a pandas Timestamp
            value = value.to_pydatetime()
        if isinstance(value, _dt.datetime):
            return value if is_datetime else value.date()
        if isinstance(value, _dt.date):
            return _dt.datetime.combine(value, _dt.time()) if is_datetime else value
        raise PlotsimpleError(
            f'"{column}" is a date column, so {option}= needs a date like "2026-03-15" (you gave {_say(value)}).'
        )

    def _condition(self, step: str, column, options: dict, extra: tuple):
        """Build the row condition for keep_rows()/remove_rows(). Gives (expression, words)."""
        example = f'table.{step}("marks", above=50)'
        if extra:
            raise PlotsimpleError(
                f".{step}() needs the column and then one plain word saying which rows, "
                f'like above=50 or equal_to="A".' + _example(example)
            )
        if column is None:
            raise PlotsimpleError(f".{step}() needs a column name first." + _example(example))
        self._need_one_column(step, column, example)
        given = {k: v for k, v in options.items() if v is not _NOT_GIVEN}
        if not given:
            raise PlotsimpleError(
                f".{step}() needs one condition saying which rows. You can use: "
                + ", ".join(f"{c}=" for c in CONDITIONS) + _example(example)
            )
        if len(given) > 1:
            names = " and ".join(f"{c}=" for c in given)
            raise PlotsimpleError(
                f".{step}() takes one condition at a time, but you gave {names}. "
                f"Write one step per condition, or use between=[low, high]."
                + _example(f'table.{step}("marks", at_least=40)   then   table.{step}("marks", at_most=80)')
            )
        option, value = next(iter(given.items()))
        kind = self._kind_of(column)
        col = nw.col(column)
        empty = self._empty(column)

        if option == "is_empty":
            if not isinstance(value, bool):
                raise PlotsimpleError(
                    "is_empty= needs True or False (no quotes)." + _example(f'table.{step}("marks", is_empty=True)')
                )
            return (empty if value else ~empty), ("is empty" if value else "is not empty")

        if (option in COMPARE or option == "between") and kind not in ("number", "date"):
            raise PlotsimpleError(
                f'{option}= compares numbers or dates, but "{column}" is a {kind} column. '
                f"For text use equal_to=, one_of=, contains=, starts_with= or ends_with=."
                + _example(f'table.{step}("class", equal_to="A")')
            )

        if option in TEXT_ONLY:
            if kind != "text":
                raise PlotsimpleError(
                    f'{option}= looks inside text, but "{column}" is a {kind} column. '
                    f"For numbers use above=, below=, between= or equal_to=." + _example(example)
                )
            if not isinstance(value, str):
                raise PlotsimpleError(
                    f'{option}= needs text in quotes, like {option}="{value}".'
                    + _example(f'table.{step}("name", {option}="A")')
                )
            text = col.cast(nw.String)
            if option == "contains":
                cond, words = text.str.contains(value, literal=True), f"contains {_say(value)}"
            elif option == "starts_with":
                cond, words = text.str.starts_with(value), f"starts with {_say(value)}"
            else:
                cond, words = text.str.ends_with(value), f"ends with {_say(value)}"
        elif option in COMPARE:
            value = self._value_for(option, column, value, example)
            cond = {"above": col > value, "below": col < value,
                    "at_least": col >= value, "at_most": col <= value}[option]
            words = f"is {option.replace('_', ' ')} {_say(value)}"
        elif option == "between":
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 2:
                raise PlotsimpleError(
                    "between= needs two values in square brackets: the lowest and the highest."
                    + _example(f'table.{step}("marks", between=[40, 60])')
                )
            low, high = sorted(self._value_for(option, column, v, example) for v in value)
            cond, words = (col >= low) & (col <= high), f"is between {_say(low)} and {_say(high)}"
        elif option == "one_of":
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
                raise PlotsimpleError(
                    "one_of= needs a list in square brackets. For a single value use equal_to=."
                    + _example(f'table.{step}("subject", one_of=["Maths", "Science"])')
                )
            values = [self._value_for(option, column, v, example) for v in value]
            if not values:
                raise PlotsimpleError(
                    "one_of= needs at least one value in the list."
                    + _example(f'table.{step}("subject", one_of=["Maths", "Science"])')
                )
            cond, words = col.is_in(values), "is one of " + ", ".join(_say(v) for v in values)
        else:  # equal_to, not_equal_to
            value = self._value_for(option, column, value, f'table.{step}("class", equal_to="A")')
            if option == "equal_to":
                cond, words = col == value, f"is {_say(value)}"
            else:
                cond, words = col != value, f"is not {_say(value)}"
        return cond.fill_null(False) & ~empty, words

    def _filter(self, what: str, keep_expr, kept_word: bool = True):
        before = len(self._frame)
        new = self._frame.filter(keep_expr)
        after = len(new)
        if kept_word:
            print(f"Kept {after} of {_rows(before)}.")
        else:
            print(f"Removed {before - after} of {_rows(before)}. {_rows(after)} left.")
        return self._change(what, new)

    def _report_failed(self, column: str, kind_word: str, new_values) -> int:
        """Print how many filled cells could not be changed, with a few of them as examples."""
        marked = self._frame.with_columns(new_values.alias(_NEW))
        failed = marked.filter(~self._empty(column) & nw.col(_NEW).is_null())
        n = len(failed)
        if n:
            samples = failed[column].cast(nw.String).unique(maintain_order=True).head(3).to_list()
            verb = "is" if n == 1 else "are"
            print(f"{_cells(n)} could not become {kind_word} and {verb} now empty, for example: "
                  + ", ".join(f'"{s}"' for s in samples))
        return n

    # ================================================================== looking
    @friendly
    def describe_columns(self):
        """Describe every column: its kind, empty cells, smallest, biggest and average.

        table.describe_columns()   (with ps.example("marks"))
        """
        import pandas as pd

        rows_out = []
        for name in self._columns:
            kind = self._kind_of(name)
            empty = self._empty(name)
            filled = self._frame.filter(~empty)
            row = {"column": name, "kind": kind, "empty cells": self._count(empty),
                   "smallest": "", "biggest": "", "average": "", "different values": ""}
            col = nw.col(name)
            if len(filled) and kind == "number":
                low, high, mean = filled.select(col.min(), col.max().alias("b"), col.mean().alias("c")).row(0)
                row.update(smallest=_nice(low), biggest=_nice(high), average=_nice(float(mean)))
            elif len(filled) and kind == "date":
                low, high = filled.select(col.min(), col.max().alias("b")).row(0)
                row.update(smallest=_nice(low), biggest=_nice(high))
            if kind in ("text", "true/false", "mixed"):
                row["different values"] = str(filled.select(col.n_unique()).item() if len(filled) else 0)
            rows_out.append(row)
        rows, cols = self._frame.shape
        print(f"The table has {_rows(rows)} and {cols} columns.\n")
        print(pd.DataFrame(rows_out).to_string(index=False))
        return None

    @friendly
    def count_empty_cells(self):
        """Show how many empty cells each column has.

        table.count_empty_cells()   (with ps.example("marks"))
        """
        counts = {name: self._count(self._empty(name)) for name in self._columns}
        width = max((len(str(n)) for n in counts), default=0)
        print("Empty cells in each column:")
        for name, n in counts.items():
            print(f"  {str(name).ljust(width)}  {n}")
        total = sum(counts.values())
        print(f"Total: {_cells(total)} empty." if total else "No empty cells.")
        return None

    # ================================================================== choosing columns
    @friendly
    def keep_columns(self, *names):
        """Keep only these columns, in the order you list them.

        table.keep_columns("name", "marks")   (with ps.example("marks"))
        """
        names = list(dict.fromkeys(_as_list(names)))
        if not names:
            raise PlotsimpleError(
                ".keep_columns() needs the names of the columns to keep."
                + _example('table.keep_columns("name", "marks")')
            )
        self._need_columns(*names)
        return self._change("keep columns " + ", ".join(names), self._frame.select(names))

    @friendly
    def remove_columns(self, *names):
        """Remove these columns.

        table.remove_columns("height")   (with ps.example("marks"))
        """
        names = list(dict.fromkeys(_as_list(names)))
        if not names:
            raise PlotsimpleError(
                ".remove_columns() needs the names of the columns to remove."
                + _example('table.remove_columns("height")')
            )
        self._need_columns(*names)
        if len(names) == len(self._columns):
            raise PlotsimpleError(
                "That would remove every column. Keep at least one." + _example('table.remove_columns("height")')
            )
        return self._change("remove columns " + ", ".join(names), self._frame.drop(names))

    @friendly
    def rename_column(self, old=None, to=None):
        """Give a column a new name.

        table.rename_column("marks", to="score")   (with ps.example("marks"))
        """
        example = 'table.rename_column("marks", to="score")'
        self._need_one_column("rename_column", old, example)
        if not isinstance(to, str) or not to.strip():
            raise PlotsimpleError(".rename_column() needs the new name in quotes after to=." + _example(example))
        if to == old:
            raise PlotsimpleError(f'The column is already called "{old}".')
        if to in self._columns:
            raise PlotsimpleError(
                f'There is already a column called "{to}". Pick another name, or remove that column first.'
                + _example(example)
            )
        return self._change(f'rename column "{old}" to "{to}"', self._frame.rename({old: to}))

    # ================================================================== choosing rows
    @friendly
    def keep_rows(self, column=None, *extra, above=_NOT_GIVEN, below=_NOT_GIVEN, at_least=_NOT_GIVEN,
                  at_most=_NOT_GIVEN, equal_to=_NOT_GIVEN, not_equal_to=_NOT_GIVEN, between=_NOT_GIVEN,
                  one_of=_NOT_GIVEN, contains=_NOT_GIVEN, starts_with=_NOT_GIVEN, ends_with=_NOT_GIVEN,
                  is_empty=_NOT_GIVEN):
        """Keep only the rows that match one condition.

        table.keep_rows("marks", above=50)   (with ps.example("marks"))
        Conditions: above, below, at_least, at_most, equal_to, not_equal_to,
        between=[low, high] (both ends included), one_of=[...], contains, starts_with,
        ends_with (capital letters matter), is_empty=True. Empty cells only match is_empty.
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, contains=contains,
                       starts_with=starts_with, ends_with=ends_with, is_empty=is_empty)
        cond, words = self._condition("keep_rows", column, options, extra)
        return self._filter(f"keep rows where {column} {words}", cond)

    @friendly
    def remove_rows(self, column=None, *extra, above=_NOT_GIVEN, below=_NOT_GIVEN, at_least=_NOT_GIVEN,
                    at_most=_NOT_GIVEN, equal_to=_NOT_GIVEN, not_equal_to=_NOT_GIVEN, between=_NOT_GIVEN,
                    one_of=_NOT_GIVEN, contains=_NOT_GIVEN, starts_with=_NOT_GIVEN, ends_with=_NOT_GIVEN,
                    is_empty=_NOT_GIVEN):
        """Remove the rows that match one condition (the opposite of keep_rows).

        table.remove_rows("subject", equal_to="English")   (with ps.example("marks"))
        Same conditions as keep_rows; between=[low, high] includes both ends.
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, contains=contains,
                       starts_with=starts_with, ends_with=ends_with, is_empty=is_empty)
        cond, words = self._condition("remove_rows", column, options, extra)
        return self._filter(f"remove rows where {column} {words}", ~cond, kept_word=False)

    @friendly
    def keep_first_rows(self, n=5):
        """Keep only the first rows.

        table.keep_first_rows(10)   (with ps.example("marks"))
        """
        n = _check_count("keep_first_rows", n, "table.keep_first_rows(10)")
        before = len(self._frame)
        new = self._frame.head(n)
        print(f"Kept {len(new)} of {_rows(before)}.")
        return self._change(f"keep the first {_rows(n)}", new)

    @friendly
    def keep_last_rows(self, n=5):
        """Keep only the last rows.

        table.keep_last_rows(10)   (with ps.example("marks"))
        """
        n = _check_count("keep_last_rows", n, "table.keep_last_rows(10)")
        before = len(self._frame)
        new = self._frame.tail(n)
        print(f"Kept {len(new)} of {_rows(before)}.")
        return self._change(f"keep the last {_rows(n)}", new)

    @friendly
    def keep_random_rows(self, n=5, seed=None):
        """Keep some rows picked at random; give a seed to get the same rows every time.

        table.keep_random_rows(10, seed=1)   (with ps.example("marks"))
        """
        example = "table.keep_random_rows(10, seed=1)"
        n = _check_count("keep_random_rows", n, example)
        if seed is not None and (not isinstance(seed, numbers.Integral) or isinstance(seed, bool) or seed < 0):
            raise PlotsimpleError(
                "seed= needs a whole number, like seed=1. The same seed picks the same rows again."
                + _example(example)
            )
        before = len(self._frame)
        if n > before:
            print(f"The table has only {_rows(before)}, so all of them are kept.")
            n = before
        new = self._frame.sample(n=n, seed=None if seed is None else int(seed))
        print(f"Kept {len(new)} of {_rows(before)}.")
        what = f"keep {n} random rows" + (f" (seed {seed})" if seed is not None else "")
        return self._change(what, new)

    @friendly
    def remove_duplicate_rows(self, columns=None):
        """Remove rows that repeat an earlier row, or that repeat it in the columns you name.

        table.remove_duplicate_rows(columns=["name"])   (with ps.example("marks"))
        """
        if columns is None:
            subset, what = None, "remove duplicate rows"
        else:
            subset = [columns] if isinstance(columns, str) else _as_list([columns])
            if not subset:
                raise PlotsimpleError(
                    "columns= needs one or more column names."
                    + _example('table.remove_duplicate_rows(columns=["name"])')
                )
            self._need_columns(*subset)
            what = "remove rows that repeat " + ", ".join(subset)
        before = len(self._frame)
        new = self._frame.unique(subset=subset, keep="first", maintain_order=True)
        print(f"Removed {before - len(new)} of {_rows(before)}. {_rows(len(new))} left.")
        return self._change(what, new)

    # ================================================================== empty cells
    @friendly
    def remove_rows_with_empty_cells(self, column=None):
        """Remove rows with an empty cell (in any column, or only in the column you name).

        table.remove_rows_with_empty_cells("marks")   (with ps.example("marks"))
        """
        if column is None:
            names, what = self._columns, "remove rows with an empty cell"
        else:
            names = [column] if isinstance(column, str) else _as_list([column])
            self._need_columns(*names)
            what = f"remove rows where {', '.join(names)} is empty"
        any_empty = nw.any_horizontal(*[self._empty(c) for c in names], ignore_nulls=True)
        return self._filter(what, ~any_empty.fill_null(False), kept_word=False)

    @friendly
    def fill_empty_cells(self, column=None, with_value=None, with_the=None):
        """Fill the empty cells of a column with a value, or with the average, median, value above or value below.

        table.fill_empty_cells("marks", with_the="average")   (with ps.example("marks"))
        """
        example = 'table.fill_empty_cells("marks", with_the="average")'
        self._need_one_column("fill_empty_cells", column, example)
        if (with_value is None) == (with_the is None):
            raise PlotsimpleError(
                ".fill_empty_cells() needs exactly one of with_value= or with_the=."
                + _example('table.fill_empty_cells("marks", with_value=0)')
                + _example(example)
            )
        empty = self._empty(column)
        kind = self._kind_of(column)
        col = nw.col(column)
        if with_value is not None:
            value = self._value_for("with_value", column, with_value, 'table.fill_empty_cells("marks", with_value=0)')
            new_col, how = nw.when(empty).then(nw.lit(value)).otherwise(col), _say(value)
        else:
            choice = str(with_the).strip().lower().replace("_", " ")
            self._check_choice("fill_empty_cells", "with_the", choice, FILL_CHOICES)
            if choice in ("average", "median"):
                if kind != "number":
                    raise PlotsimpleError(
                        f'with_the="{choice}" needs numbers, but "{column}" is a {kind} column. '
                        f'Try with_the="value above" or with_value= instead.'
                        + _example('table.fill_empty_cells("subject", with_the="value above")')
                    )
                filled = self._frame.filter(~empty)
                if len(filled) == 0:
                    raise PlotsimpleError(f'"{column}" has no numbers at all, so there is no {choice} to fill with.')
                value = filled.select(col.mean() if choice == "average" else col.median()).item()
                new_col, how = nw.when(empty).then(nw.lit(float(value))).otherwise(col), f"the {choice}"
            else:
                strategy = "forward" if choice == "value above" else "backward"
                new_col, how = nw.when(~empty).then(col).fill_null(strategy=strategy), f"the {choice}"
        before = self._count(empty)
        new = self._frame.with_columns(new_col.alias(column))
        after = _count_in(new, _empty_in(new, column))
        print(f"Filled {_cells(before - after)} in {column}."
              + (f" {_cells(after)} stayed empty (nothing to copy from)." if after else ""))
        return self._change(f"fill empty cells in {column} with {how}", new)

    # ================================================================== fixing columns
    @friendly
    def make_numbers(self, column=None):
        """Turn a column into numbers; cells that are not numbers become empty.

        table.make_numbers("marks")   (with ps.example("marks"))
        Understands 12.5, -3, 1e3, 1 200 and 1,200 (thousands) and 12,5 (decimal comma).
        """
        example = 'table.make_numbers("marks")'
        self._need_one_column("make_numbers", column, example)
        kind = self._kind_of(column)
        col = nw.col(column)
        if kind == "number":
            print(f"{column} is already numbers.")
            return self
        if kind == "true/false":
            new = self._frame.with_columns(col.cast(nw.Int64))
            print(f"{column} is now numbers: True is 1 and False is 0.")
            return self._change(f"make {column} into numbers", new)
        if kind == "date":
            raise PlotsimpleError(f'"{column}" is a date column, so it cannot become numbers.' + _example(example))
        text = col.cast(nw.String).str.replace_all(r"\s", "", literal=False)
        text = nw.when(text.str.contains(_THOUSANDS).fill_null(False)).then(
            text.str.replace_all(",", "", literal=True)
        ).otherwise(
            nw.when(text.str.contains(_DECIMAL_COMMA).fill_null(False)).then(
                text.str.replace(",", ".", literal=True)
            ).otherwise(text)
        )
        number = nw.when(text.str.contains(_NUMBER).fill_null(False)).then(text).cast(nw.Float64)
        values = self._frame.select(number.alias(column))[column]
        self._report_failed(column, "numbers", values)
        print(f"{column} is now numbers.")
        return self._change(f"make {column} into numbers", self._frame.with_columns(values))

    @friendly
    def make_dates(self, column=None):
        """Turn a column into dates; cells that are not dates become empty.

        table.make_dates("year")   (with ps.example("growth"))
        Understands 2026-03-15, 15/03/2026, 15.03.2026, 15 March 2026, March 15, 2026, 2026-03 and 2026.
        03/04/2026 is read day first, as 3 April.
        """
        example = 'table.make_dates("year")'
        self._need_one_column("make_dates", column, example)
        kind = self._kind_of(column)
        col = nw.col(column)
        if kind == "date":
            print(f"{column} is already dates.")
            return self
        if kind == "true/false":
            raise PlotsimpleError(f'"{column}" is a true/false column, so it cannot become dates.')
        if kind == "number" and not self._frame.schema[column].is_integer():
            raise PlotsimpleError(
                f'make_dates() can turn whole years like 2026 into dates, but "{column}" has decimals.'
                + _example(example, "growth")
            )
        text = col.cast(nw.String).str.strip_chars()
        text = nw.when(text.str.contains(r"^\d{4}$").fill_null(False)).then(text + nw.lit("-01-01")).otherwise(
            nw.when(text.str.contains(r"^\d{4}-\d{1,2}$").fill_null(False)).then(text + nw.lit("-01")).otherwise(text)
        )
        cleaned = self._frame.select(text.alias(column))[column]
        values = nw.from_native(_parse_dates(cleaned.to_native()), series_only=True).alias(column)
        self._report_failed(column, "dates", values)
        print(f"{column} is now dates.")
        return self._change(f"make {column} into dates", self._frame.with_columns(values))

    @friendly
    def make_text(self, column=None):
        """Turn a column into text.

        table.make_text("month_number")   (with ps.example("weather"))
        """
        example = 'table.make_text("month_number")'
        self._need_one_column("make_text", column, example)
        kind = self._kind_of(column)
        dtype = self._frame.schema[column]
        col = nw.col(column)
        empty = self._empty(column)
        if dtype == nw.String:
            print(f"{column} is already text.")
            return self
        if kind == "true/false":
            text = nw.when(col).then(nw.lit("True")).otherwise(nw.lit("False"))
        elif kind == "date":
            fmt = "%Y-%m-%d"
            filled = self._frame.filter(~empty)
            if dtype == nw.Datetime and len(filled):
                has_time = filled.select(((col.dt.hour() + col.dt.minute() + col.dt.second()) > 0).any()).item()
                if has_time:
                    fmt = "%Y-%m-%d %H:%M:%S"
            text = col.dt.to_string(fmt)
        else:
            text = col.cast(nw.String)
        new = self._frame.with_columns(nw.when(~empty).then(text).alias(column))
        print(f"{column} is now text.")
        return self._change(f"make {column} into text", new)

    def _text_step(self, step: str, column, what: str, change):
        text = self._need_text(step, column, f'table.{step}("name")')
        return self._change(what, self._frame.with_columns(change(text).alias(column)))

    @friendly
    def make_lowercase(self, column=None):
        """Write all the text in a column in small letters.

        table.make_lowercase("name")   (with ps.example("marks"))
        """
        return self._text_step("make_lowercase", column, f"make {column} lowercase", lambda t: t.str.to_lowercase())

    @friendly
    def make_uppercase(self, column=None):
        """Write all the text in a column in capital letters.

        table.make_uppercase("name")   (with ps.example("marks"))
        """
        return self._text_step("make_uppercase", column, f"make {column} uppercase", lambda t: t.str.to_uppercase())

    @friendly
    def remove_extra_spaces(self, column=None):
        """Remove spaces at the start and end of the text, and double spaces inside it.

        table.remove_extra_spaces("name")   (with ps.example("marks"))
        """
        return self._text_step(
            "remove_extra_spaces", column, f"remove extra spaces in {column}",
            lambda t: t.str.replace_all(r"\s+", " ", literal=False).str.strip_chars(),
        )

    @friendly
    def replace_values(self, column=None, old=_NOT_GIVEN, with_value=_NOT_GIVEN):
        """Replace a value (or a list of values) in a column; with_value=None makes those cells empty.

        table.replace_values("subject", "Maths", with_value="Math")   (with ps.example("marks"))
        """
        example = 'table.replace_values("subject", "Maths", with_value="Math")'
        self._need_one_column("replace_values", column, example)
        if old is _NOT_GIVEN or with_value is _NOT_GIVEN:
            raise PlotsimpleError(
                ".replace_values() needs the column, the old value, and with_value= the new value."
                + _example(example)
            )
        olds = list(old) if isinstance(old, (list, tuple, set)) else [old]
        olds = [self._value_for("old value", column, v, example) for v in olds]
        col = nw.col(column)
        match = col.is_in(olds).fill_null(False)
        if with_value is None:
            new_col = nw.when(~match).then(col)
        else:
            new_value = self._value_for("with_value", column, with_value, example)
            new_col = nw.when(match).then(nw.lit(new_value)).otherwise(col)
        n = self._count(match)
        new = self._frame.with_columns(new_col.alias(column))
        print(f"Replaced {_cells(n)} in {column}.")
        olds_words = " or ".join(_say(v) for v in olds)
        return self._change(f"replace {olds_words} in {column} with {_say(with_value)}", new)
