"""Looking, choosing columns and rows, empty cells, fixing columns.

Every word is written with narwhals expressions, so it works the same on pandas and Polars.
Only make_dates() needs a little native code (text-to-date parsing that forgives bad cells).

What counts as an "empty cell" everywhere in this file:
a missing value (None, NaN, NaT), or text that is only spaces or nothing at all.
Empty cells never match a condition in keep_rows()/remove_rows(), except is_empty=.

keep_rows()/remove_rows() also read a written condition, "marks > 50 and class == 'A'".
The condition is read by the shared reader in conditions.py (also used by
add_group_column(rules=)): the text is never run, and a comparison with an empty cell is
never true and never false (as in SQL), so "not" never brings back empty cells either.
"""

from __future__ import annotations

import datetime as _dt
import math
import numbers
import re

import narwhals as nw

from .._errors import SimplydataError, friendly
from ._base import show as _show_native
from .conditions import (
    _NOT_GIVEN, _ConditionReader, _condition_error, _count_in, _empty_in, _is_number, _kind,
    _looks_like_number, _rows, _say, empty_cells_note,
)

# Conditions for keep_rows() / remove_rows(), in the order they are listed in messages.
CONDITIONS = (
    "above", "below", "at_least", "at_most", "equal_to", "not_equal_to",
    "between", "one_of", "contains", "starts_with", "ends_with", "is_empty", "after", "before",
)
COMPARE = ("above", "below", "at_least", "at_most")
TEXT_ONLY = ("contains", "starts_with", "ends_with")
FILL_CHOICES = ("average", "median", "value above", "value below", "line between neighbours")

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


def _with_times(*days):
    """Each day format, then the same with a time after it."""
    return tuple(f + t for f in days for t in ("", " %H:%M:%S", " %H:%M"))


_ISO = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")
# make_dates(order=...): only the formats that read the parts in that order (a year with four
# digits written first is always year-month-day, so those formats are kept for every order).
DATE_ORDERS = {
    "day month year": _ISO + _with_times("%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y")
    + ("%d %B %Y", "%d %b %Y"),
    "month day year": _ISO + _with_times("%Y/%m/%d", "%m/%d/%Y", "%m.%d.%Y", "%m-%d-%Y")
    + ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"),
    "year month day": _ISO + _with_times("%Y/%m/%d", "%Y.%m.%d") + ("%Y%m%d", "%Y %B %d", "%Y %b %d"),
}

_NUMBER = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
_THOUSANDS = r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$"  # 1,200 or 12,345.6
_DECIMAL_COMMA = r"^[+-]?\d*,\d+$"  # 12,5
_NEW = "__simplydata_new__"
SHOW_LIMIT = 1000  # rows that table.show() shows when rows= is not given


def _example(code: str, name: str = "marks") -> str:
    return f'\nFor example:  {code}   (with sdata.example("{name}"))'



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
        raise SimplydataError(f".{step}() needs a whole number of rows, like 5." + _example(example))
    return int(n)


def _parse_dates(native, formats=DATE_FORMATS):
    """Parse a native text Series (pandas or Polars) into dates, trying the formats in order.

    Native code because narwhals' to_datetime stops at the first bad cell; here bad cells
    simply become empty. The same formats in the same order are used for both engines.
    """
    if type(native).__module__.split(".")[0] == "polars":
        import polars as pl

        return (
            pl.DataFrame({"s": native})
            .select(pl.coalesce([pl.col("s").str.strptime(pl.Datetime("us"), f, strict=False) for f in formats]))
            .to_series()
            .rename(native.name)
        )
    import pandas as pd

    out = pd.Series(pd.NaT, index=native.index, dtype="datetime64[us]", name=native.name)
    for f in formats:
        missing = out.isna()
        if not missing.any():
            break
        found = pd.to_datetime(native.where(missing), format=f, errors="coerce").astype("datetime64[us]")
        out = out.fillna(found)
    return out



def _rule_of(step: str, rule, example: str):
    """Read rule= for the outlier words: (how many, "IQR" or "standard deviations")."""
    text = str(rule).strip().lower().replace("×", "x") if isinstance(rule, str) else ""
    found = re.match(
        r"^(\d+(?:\.\d+)?)\s*(?:x|\*|times)?\s*(iqrs?|interquartile ranges?|standard deviations?|sds?|std)$", text
    )
    if not found or float(found.group(1)) <= 0:
        raise SimplydataError(
            f'.{step}() does not know rule={_say(rule)}. You can use:\n'
            '  rule="1.5 IQR"                 Tukey\'s rule, the usual one (the default)\n'
            '  rule="3 IQR"                   only values very far out\n'
            '  rule="3 standard deviations"   further than 3 standard deviations from the average'
            + _example(example)
        )
    how = "IQR" if found.group(2).startswith(("iqr", "inter")) else "standard deviations"
    return float(found.group(1)), how


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
            raise SimplydataError(
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
            raise SimplydataError(
                f'"{column}" holds a mix of things. Turn it into text first with '
                f'table.make_text("{column}").' + _example(example)
            )
        raise SimplydataError(
            f'.{step}() works on text, but "{column}" is a {kind} column.' + _example(example)
        )

    def _value_for(self, option: str, column: str, value, example: str):
        """Check one value against the column's kind and give it back ready to compare."""
        kind = self._kind_of(column)
        if kind == "number":
            if _is_number(value):
                return value
            if isinstance(value, str) and _looks_like_number(value):
                raise SimplydataError(
                    f'"{column}" is a number column, so write the number without quotes: '
                    f"{option}={value.strip()}" + _example(example)
                )
            raise SimplydataError(
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
            raise SimplydataError(
                f'"{column}" is a text column, so {option}= needs text in quotes, like '
                f'{option}="{value}".{hint}' + _example('table.keep_rows("class", equal_to="A")')
            )
        if kind == "true/false":
            if isinstance(value, bool):
                return value
            raise SimplydataError(
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
                raise SimplydataError(
                    f'"{column}" is a date column. Write dates as year-month-day, like '
                    f'{option}="2026-03-15".'
                ) from None
        if hasattr(value, "to_pydatetime"):  # a pandas Timestamp
            value = value.to_pydatetime()
        if isinstance(value, _dt.datetime):
            return value if is_datetime else value.date()
        if isinstance(value, _dt.date):
            return _dt.datetime.combine(value, _dt.time()) if is_datetime else value
        raise SimplydataError(
            f'"{column}" is a date column, so {option}= needs a date like "2026-03-15" (you gave {_say(value)}).'
        )

    def _condition(self, step: str, column, options: dict, extra: tuple):
        """Build the row condition for keep_rows()/remove_rows(). Gives (expression, words)."""
        example = f'table.{step}("marks", above=50)'
        if extra:
            raise SimplydataError(
                f".{step}() needs the column and then one plain word saying which rows, "
                f'like above=50 or equal_to="A".' + _example(example)
            )
        if column is None:
            raise SimplydataError(f".{step}() needs a column name first." + _example(example))
        self._need_one_column(step, column, example)
        given = {k: v for k, v in options.items() if v is not _NOT_GIVEN}
        if not given:
            raise SimplydataError(
                f".{step}() needs one condition saying which rows. You can use: "
                + ", ".join(f"{c}=" for c in CONDITIONS)
                + f"\nor write the whole condition: table.{step}(\"marks > 50 and class == 'A'\")"
                + _example(example)
            )
        if len(given) > 1:
            names = " and ".join(f"{c}=" for c in given)
            raise SimplydataError(
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
                raise SimplydataError(
                    "is_empty= needs True or False (no quotes)." + _example(f'table.{step}("marks", is_empty=True)')
                )
            return (empty if value else ~empty), ("is empty" if value else "is not empty")

        if option in ("after", "before"):
            if kind != "date":
                other = "above" if option == "after" else "below"
                raise SimplydataError(
                    f'{option}= is for dates, but "{column}" is a {kind} column. '
                    f"For numbers use {other}=." + _example(f'table.{step}("marks", {other}=50)')
                )
            given_time = isinstance(value, _dt.datetime) or hasattr(value, "to_pydatetime") or (
                isinstance(value, str) and any(sign in value for sign in (":", "T", " ")) and len(value.strip()) > 10
            )
            day = self._date_value(option, column, value)
            if self._frame.schema[column] == nw.Datetime and not given_time:
                # a day without a time: "after 1 March" starts on 2 March, "before 1 March" ends on 28 February
                start = day if option == "before" else day + _dt.timedelta(days=1)
                cond = col < start if option == "before" else col >= start
                words = f"is {option} {_say(day)}"
            else:
                cond = col > day if option == "after" else col < day
                words = f"is {option} {_say(day)}"
            return cond.fill_null(False) & ~empty, words

        if (option in COMPARE or option == "between") and kind not in ("number", "date"):
            raise SimplydataError(
                f'{option}= compares numbers or dates, but "{column}" is a {kind} column. '
                f"For text use equal_to=, one_of=, contains=, starts_with= or ends_with=."
                + _example(f'table.{step}("class", equal_to="A")')
            )

        if option in TEXT_ONLY:
            if kind != "text":
                raise SimplydataError(
                    f'{option}= looks inside text, but "{column}" is a {kind} column. '
                    f"For numbers use above=, below=, between= or equal_to=." + _example(example)
                )
            if not isinstance(value, str):
                raise SimplydataError(
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
                raise SimplydataError(
                    "between= needs two values in square brackets: the lowest and the highest."
                    + _example(f'table.{step}("marks", between=[40, 60])')
                )
            low, high = sorted(self._value_for(option, column, v, example) for v in value)
            cond, words = (col >= low) & (col <= high), f"is between {_say(low)} and {_say(high)}"
        elif option == "one_of":
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
                raise SimplydataError(
                    "one_of= needs a list in square brackets. For a single value use equal_to=."
                    + _example(f'table.{step}("subject", one_of=["Maths", "Science"])')
                )
            values = [self._value_for(option, column, v, example) for v in value]
            if not values:
                raise SimplydataError(
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

    def _filter(self, what: str, keep_expr, kept_word: bool = True, note: str = ""):
        before = len(self._frame)
        new = self._frame.filter(keep_expr)
        after = len(new)
        if kept_word:
            print(f"Kept {after} of {_rows(before)}.")
        else:
            print(f"Removed {before - after} of {_rows(before)}. {_rows(after)} left.")
        if note:
            print(note)
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

        table.describe_columns()   (with sdata.example("marks"))
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

        table.count_empty_cells()   (with sdata.example("shop"))
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

        table.keep_columns("name", "marks")   (with sdata.example("marks"))
        """
        names = list(dict.fromkeys(_as_list(names)))
        if not names:
            raise SimplydataError(
                ".keep_columns() needs the names of the columns to keep."
                + _example('table.keep_columns("name", "marks")')
            )
        self._need_columns(*names)
        return self._change("keep columns " + ", ".join(names), self._frame.select(names))

    @friendly
    def remove_columns(self, *names):
        """Remove these columns.

        table.remove_columns("height")   (with sdata.example("marks"))
        """
        names = list(dict.fromkeys(_as_list(names)))
        if not names:
            raise SimplydataError(
                ".remove_columns() needs the names of the columns to remove."
                + _example('table.remove_columns("height")')
            )
        self._need_columns(*names)
        if len(names) == len(self._columns):
            raise SimplydataError(
                "That would remove every column. Keep at least one." + _example('table.remove_columns("height")')
            )
        return self._change("remove columns " + ", ".join(names), self._frame.drop(names))

    @friendly
    def rename_column(self, old=None, to=None):
        """Give a column a new name.

        table.rename_column("marks", to="score")   (with sdata.example("marks"))
        """
        example = 'table.rename_column("marks", to="score")'
        self._need_one_column("rename_column", old, example)
        if not isinstance(to, str) or not to.strip():
            raise SimplydataError(".rename_column() needs the new name in quotes after to=." + _example(example))
        if to == old:
            raise SimplydataError(f'The column is already called "{old}".')
        if to in self._columns:
            raise SimplydataError(
                f'There is already a column called "{to}". Pick another name, or remove that column first.'
                + _example(example)
            )
        return self._change(f'rename column "{old}" to "{to}"', self._frame.rename({old: to}))

    # ================================================================== choosing rows
    def _where(self, step: str, column, options: dict, extra: tuple):
        """(condition, words, note) for keep_rows()/remove_rows(): a written condition or one plain word."""
        given = any(v is not _NOT_GIVEN for v in options.values())
        if isinstance(column, str) and column not in self._frame.columns:
            text = column.strip()
            if text and not text.isidentifier():  # "marks > 50 and class == 'A'"
                if given or extra:
                    raise SimplydataError(
                        f".{step}() takes either a written condition or a column with one plain word, not both."
                        + _example(f'table.{step}("marks > 50")') + _example(f'table.{step}("marks", above=50)')
                    )
                true, note = self._read_condition(step, text)
                return true, text, note
        is_yes_no = isinstance(column, str) and column in self._frame.columns and self._kind_of(column) == "true/false"
        if is_yes_no and not given and not extra:  # keep_rows("passed")
            true, note = self._read_condition(step, f"`{column}`")
            return true, f"{column} is True", note
        cond, words = self._condition(step, column, options, extra)
        return cond, f"{column} {words}", ""

    def _read_condition(self, step: str, text: str):
        """(where the condition is surely true, a note about rows it could not check)."""
        reader = _ConditionReader(text, self._frame)
        true, false = reader.read()
        try:
            self._count(true)  # check now that it can be worked out, so errors come at once
        except Exception as error:  # noqa: BLE001 - shown to the user in plain words
            raise _condition_error(
                f'simplydata could not check the condition "{text}" ({type(error).__name__}).'
            ) from None
        n, note = empty_cells_note(self._frame, true, false, reader.used)
        if n:
            gone = "they were left out" if step == "keep_rows" else "they were not removed"
            if n == 1:
                gone = "it was left out" if step == "keep_rows" else "it was not removed"
            note = note[:-1] + f", so {gone}."
        return true, note

    @friendly
    def keep_rows(self, column=None, *extra, above=_NOT_GIVEN, below=_NOT_GIVEN, at_least=_NOT_GIVEN,
                  at_most=_NOT_GIVEN, equal_to=_NOT_GIVEN, not_equal_to=_NOT_GIVEN, between=_NOT_GIVEN,
                  one_of=_NOT_GIVEN, contains=_NOT_GIVEN, starts_with=_NOT_GIVEN, ends_with=_NOT_GIVEN,
                  is_empty=_NOT_GIVEN, after=_NOT_GIVEN, before=_NOT_GIVEN):
        """Keep only the rows that match a condition.

        table.keep_rows("marks", above=50)   (with sdata.example("marks"))
        table.keep_rows("marks > 50 and class == 'A'")   a written condition
        Plain words: above, below, at_least, at_most, equal_to, not_equal_to,
        between=[low, high] (both ends included), one_of=[...], contains, starts_with,
        ends_with (capital letters matter), is_empty=True, and after= / before= for dates.
        Written conditions use == != < <= > >=, and, or, not, brackets, in [...], is empty;
        text and dates go in quotes. Empty cells only match is_empty / is empty.
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, contains=contains,
                       starts_with=starts_with, ends_with=ends_with, is_empty=is_empty, after=after, before=before)
        cond, words, note = self._where("keep_rows", column, options, extra)
        return self._filter(f"keep rows where {words}", cond, note=note)

    @friendly
    def remove_rows(self, column=None, *extra, above=_NOT_GIVEN, below=_NOT_GIVEN, at_least=_NOT_GIVEN,
                    at_most=_NOT_GIVEN, equal_to=_NOT_GIVEN, not_equal_to=_NOT_GIVEN, between=_NOT_GIVEN,
                    one_of=_NOT_GIVEN, contains=_NOT_GIVEN, starts_with=_NOT_GIVEN, ends_with=_NOT_GIVEN,
                    is_empty=_NOT_GIVEN, after=_NOT_GIVEN, before=_NOT_GIVEN):
        """Remove the rows that match a condition (the opposite of keep_rows).

        table.remove_rows("subject", equal_to="English")   (with sdata.example("marks"))
        table.remove_rows("marks < 40 or hours < 2")   a written condition
        Same conditions as keep_rows; between=[low, high] includes both ends. A row whose
        cell is empty is never removed by a comparison on that cell (use is_empty=True).
        """
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, contains=contains,
                       starts_with=starts_with, ends_with=ends_with, is_empty=is_empty, after=after, before=before)
        cond, words, note = self._where("remove_rows", column, options, extra)
        return self._filter(f"remove rows where {words}", ~cond, kept_word=False, note=note)

    @friendly
    def keep_first_rows(self, n=5):
        """Keep only the first rows.

        table.keep_first_rows(10)   (with sdata.example("marks"))
        """
        n = _check_count("keep_first_rows", n, "table.keep_first_rows(10)")
        before = len(self._frame)
        new = self._frame.head(n)
        print(f"Kept {len(new)} of {_rows(before)}.")
        return self._change(f"keep the first {_rows(n)}", new)

    @friendly
    def keep_last_rows(self, n=5):
        """Keep only the last rows.

        table.keep_last_rows(10)   (with sdata.example("marks"))
        """
        n = _check_count("keep_last_rows", n, "table.keep_last_rows(10)")
        before = len(self._frame)
        new = self._frame.tail(n)
        print(f"Kept {len(new)} of {_rows(before)}.")
        return self._change(f"keep the last {_rows(n)}", new)

    @friendly
    def keep_random_rows(self, n=5, seed=None):
        """Keep some rows picked at random; give a seed to get the same rows every time.

        table.keep_random_rows(10, seed=1)   (with sdata.example("marks"))
        """
        example = "table.keep_random_rows(10, seed=1)"
        n = _check_count("keep_random_rows", n, example)
        if seed is not None and (not isinstance(seed, numbers.Integral) or isinstance(seed, bool) or seed < 0):
            raise SimplydataError(
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

        table.remove_duplicate_rows(columns=["name"])   (with sdata.example("marks"))
        """
        if columns is None:
            subset, what = None, "remove duplicate rows"
        else:
            subset = [columns] if isinstance(columns, str) else _as_list([columns])
            if not subset:
                raise SimplydataError(
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

        table.remove_rows_with_empty_cells("sales")   (with sdata.example("shop"))
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
    def fill_empty_cells(self, column=None, with_value=None, with_the=None, sorted_by=None, for_each=None):
        """Fill the empty cells of a column with a value, or with the average, median, value above or value below.

        table.fill_empty_cells("customers", with_the="median")   (with sdata.example("shop"))
        with_the="line between neighbours" fills each gap on the straight line between the
        filled cells just before and just after it (linear interpolation). It follows the rows
        in table order, or sorted_by= a column of numbers or dates (then the line is spaced by
        those values, so a missing day counts as one day); for_each= fills each group on its
        own. Empty cells at the start or end have only one neighbour, so they stay empty.
          table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date", for_each="shop")
        """
        example = 'table.fill_empty_cells("marks", with_the="average")'
        self._need_one_column("fill_empty_cells", column, example)
        if (with_value is None) == (with_the is None):
            raise SimplydataError(
                ".fill_empty_cells() needs exactly one of with_value= or with_the=."
                + _example('table.fill_empty_cells("marks", with_value=0)')
                + _example(example)
            )
        if with_the is not None:
            choice = str(with_the).strip().lower().replace("_", " ").replace("neighbors", "neighbours")
            if choice == "line between neighbours":
                return self._fill_by_line(column, sorted_by, for_each)
        if sorted_by is not None or for_each is not None:
            raise SimplydataError(
                'sorted_by= and for_each= go with with_the="line between neighbours".'
                + _example('table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date")',
                           "shop")
            )
        empty = self._empty(column)
        kind = self._kind_of(column)
        col = nw.col(column)
        if with_value is not None:
            value = self._value_for("with_value", column, with_value, 'table.fill_empty_cells("marks", with_value=0)')
            new_col, how = nw.when(empty).then(nw.lit(value)).otherwise(col), _say(value)
        else:
            self._check_choice("fill_empty_cells", "with_the", choice, FILL_CHOICES)
            if choice in ("average", "median"):
                if kind != "number":
                    raise SimplydataError(
                        f'with_the="{choice}" needs numbers, but "{column}" is a {kind} column. '
                        f'Try with_the="value above" or with_value= instead.'
                        + _example('table.fill_empty_cells("subject", with_the="value above")')
                    )
                filled = self._frame.filter(~empty)
                if len(filled) == 0:
                    raise SimplydataError(f'"{column}" has no numbers at all, so there is no {choice} to fill with.')
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

    def _fill_by_line(self, column, sorted_by, for_each):
        """fill_empty_cells(with_the="line between neighbours"): see line_between_neighbours()."""
        from .._stats import line_between_neighbours

        example = 'table.fill_empty_cells("sales", with_the="line between neighbours", sorted_by="date")'
        kind = self._kind_of(column)
        if kind != "number":
            raise SimplydataError(
                f'with_the="line between neighbours" needs numbers, but "{column}" is a {kind} column. '
                f'Try with_the="value above" or with_value= instead.' + _example(example, "shop")
            )
        keys = [for_each] if isinstance(for_each, str) else list(for_each or [])
        if for_each is not None and (not keys or not all(isinstance(k, str) for k in keys)):
            raise SimplydataError("for_each= should be a column name or a list of column names." + _example(example, "shop"))
        if sorted_by is not None and not isinstance(sorted_by, str):
            raise SimplydataError("sorted_by= should be one column name, such as a date." + _example(example, "shop"))
        self._need_columns(*keys, *([sorted_by] if sorted_by else []))
        frame = self._frame
        values = frame[column].cast(nw.Float64).to_numpy()
        x = None
        if sorted_by is not None:
            by_kind = self._kind_of(sorted_by)
            if by_kind not in ("number", "date"):
                raise SimplydataError(
                    f'sorted_by= puts the rows in order of numbers or dates, but "{sorted_by}" is a {by_kind} column.'
                    + _example(example, "shop")
                )
            raw = frame[sorted_by].to_numpy()
            if by_kind == "date":
                raw = raw.astype("datetime64[us]")
                missing = raw != raw  # NaT
                x = raw.astype("int64").astype(float) / 86_400_000_000  # in days
                x[missing] = float("nan")
            else:
                x = frame[sorted_by].cast(nw.Float64).to_numpy()
        groups = None
        if keys:
            columns = [frame[k].to_list() for k in keys]
            place = {}
            groups = [place.setdefault(key, len(place)) for key in zip(*columns)]
        new_values = line_between_neighbours(values, x, groups)
        before = self._count(self._empty(column))
        new = frame.with_columns(nw.new_series(column, new_values, nw.Float64, backend=frame.implementation))
        after = _count_in(new, _empty_in(new, column))
        order = f" in order of {sorted_by}" if sorted_by else " in table order"
        each = f", for each {', '.join(keys)} on its own" if keys else ""
        print(f"Filled {_cells(before - after)} in {column} on the line between neighbours{order}{each}.")
        no_place = _count_in(frame, self._empty(column) & self._empty(sorted_by)) if sorted_by else 0
        if after - no_place:
            ends = "the start or end of a group" if keys else "the start or end"
            n = after - no_place
            print(f"Note: {_cells(n)} stayed empty: {'it is' if n == 1 else 'they are'} at {ends}, "
                  "with a filled neighbour on one side only.")
        if no_place:
            print(f"Note: {_cells(no_place)} stayed empty because {sorted_by} is empty in "
                  f"{'that row' if no_place == 1 else 'those rows'}.")
        what = f"fill empty cells in {column} with the line between neighbours{order}{each}"
        return self._change(what, new)

    # ================================================================== fixing columns
    @friendly
    def make_numbers(self, column=None):
        """Turn a column into numbers; cells that are not numbers become empty.

        table.make_numbers("marks")   (with sdata.example("marks"))
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
            raise SimplydataError(f'"{column}" is a date column, so it cannot become numbers.' + _example(example))
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
    def make_dates(self, column=None, order=None):
        """Turn a column into dates; cells that are not dates become empty.

        table.make_dates("year")   (with sdata.example("growth"))
        Understands 2026-03-15, 15/03/2026, 15.03.2026, 15 March 2026, March 15, 2026, 2026-03 and 2026.
        03/04/2026 is read day first, as 3 April (month first only when day first cannot work).
        order= says how the dates are written, so nothing is guessed: "day month year"
        (15/03/2026), "month day year" (03/15/2026) or "year month day" (2026/03/15).
          table.make_dates("when", order="month day year")
        """
        example = 'table.make_dates("year")'
        self._need_one_column("make_dates", column, example)
        if order is not None:
            if isinstance(order, str):
                order = " ".join(order.lower().replace(",", " ").replace("-", " ").split())
            self._check_choice("make_dates", "order", order, DATE_ORDERS)
        kind = self._kind_of(column)
        col = nw.col(column)
        if kind == "date":
            print(f"{column} is already dates.")
            return self
        if kind == "true/false":
            raise SimplydataError(f'"{column}" is a true/false column, so it cannot become dates.')
        if kind == "number" and not self._frame.schema[column].is_integer():
            raise SimplydataError(
                f'make_dates() can turn whole years like 2026 into dates, but "{column}" has decimals.'
                + _example(example, "growth")
            )
        text = col.cast(nw.String).str.strip_chars()
        text = nw.when(text.str.contains(r"^\d{4}$").fill_null(False)).then(text + nw.lit("-01-01")).otherwise(
            nw.when(text.str.contains(r"^\d{4}-\d{1,2}$").fill_null(False)).then(text + nw.lit("-01")).otherwise(text)
        )
        cleaned = self._frame.select(text.alias(column))[column]
        formats = DATE_FORMATS if order is None else DATE_ORDERS[order]
        values = nw.from_native(_parse_dates(cleaned.to_native(), formats), series_only=True).alias(column)
        self._report_failed(column, "dates", values)
        read_as = "" if order is None else f" (read as {order})"
        print(f"{column} is now dates{read_as}.")
        what = f"make {column} into dates" + ("" if order is None else f', order="{order}"')
        return self._change(what, self._frame.with_columns(values))

    @friendly
    def make_text(self, column=None):
        """Turn a column into text.

        table.make_text("month_number")   (with sdata.example("weather"))
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

        table.make_lowercase("name")   (with sdata.example("marks"))
        """
        return self._text_step("make_lowercase", column, f"make {column} lowercase", lambda t: t.str.to_lowercase())

    @friendly
    def make_uppercase(self, column=None):
        """Write all the text in a column in capital letters.

        table.make_uppercase("name")   (with sdata.example("marks"))
        """
        return self._text_step("make_uppercase", column, f"make {column} uppercase", lambda t: t.str.to_uppercase())

    @friendly
    def remove_extra_spaces(self, column=None):
        """Remove spaces at the start and end of the text, and double spaces inside it.

        table.remove_extra_spaces("name")   (with sdata.example("marks"))
        """
        return self._text_step(
            "remove_extra_spaces", column, f"remove extra spaces in {column}",
            lambda t: t.str.replace_all(r"\s+", " ", literal=False).str.strip_chars(),
        )

    @friendly
    def replace_values(self, column=None, old=_NOT_GIVEN, with_value=_NOT_GIVEN):
        """Replace a value (or a list of values) in a column; with_value=None makes those cells empty.

        table.replace_values("subject", "Maths", with_value="Math")   (with sdata.example("marks"))
        table.replace_values("class", {"A": "Class A", "B": "Class B"})   several at once, old: new
        With {old: new}, every cell is looked up once in the original column, so
        {"A": "B", "B": "A"} swaps A and B. A new value of None makes those cells empty.
        """
        example = 'table.replace_values("subject", "Maths", with_value="Math")'
        self._need_one_column("replace_values", column, example)
        if isinstance(old, dict):
            return self._replace_many(column, old, with_value)
        if old is _NOT_GIVEN or with_value is _NOT_GIVEN:
            raise SimplydataError(
                ".replace_values() needs the column, the old value, and with_value= the new value,"
                " or the column and {old: new} pairs." + _example(example)
                + _example('table.replace_values("class", {"A": "Class A", "B": "Class B"})')
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

    def _replace_many(self, column, pairs: dict, with_value):
        example = 'table.replace_values("class", {"A": "Class A", "B": "Class B"})'
        if with_value is not _NOT_GIVEN:
            raise SimplydataError(
                "With {old: new} pairs the new values are already in the brackets, so leave out with_value=."
                + _example(example)
            )
        if not pairs:
            raise SimplydataError("The {old: new} brackets are empty. Write at least one pair." + _example(example))
        kind = self._kind_of(column)
        checked = []
        for old_value, new_value in pairs.items():
            old_value = self._value_for("old value", column, old_value, example)
            if new_value is not None:
                try:
                    new_value = self._value_for("new value", column, new_value, example)
                except SimplydataError:
                    raise SimplydataError(
                        f'The new values must be the same kind as "{column}", a {kind} column '
                        f"(you gave {_say(new_value)}). To change the kind, replace first and then use "
                        f"make_numbers(), make_text() or make_dates()."
                        + _example('table.replace_values("class", {"A": "1", "B": "2"}).make_numbers("class")')
                    ) from None
            checked.append((old_value, new_value))
        col = nw.col(column)
        new_col = col
        for old_value, new_value in reversed(checked):  # every test looks at the original column
            if new_value is not None:
                new_col = nw.when((col == old_value).fill_null(False)).then(nw.lit(new_value)).otherwise(new_col)
        to_empty = [o for o, n in checked if n is None]
        if to_empty:
            new_col = nw.when(~col.is_in(to_empty).fill_null(False)).then(new_col)
        counts = [self._count((col == o).fill_null(False)) for o, _ in checked]
        new = self._frame.with_columns(new_col.alias(column))
        said = ", ".join(f"{_say(o)} became {_say(n)} ({_cells(k)})" for (o, n), k in zip(checked, counts) if k)
        print(f"Replaced {_cells(sum(counts))} in {column}" + (f": {said}." if said else "."))
        missing = [_say(o) for (o, _), k in zip(checked, counts) if not k]
        if missing:
            print(f"Note: {', '.join(missing)} {'was' if len(missing) == 1 else 'were'} not found in {column}.")
        what = ", ".join(f"{_say(o)} with {_say(n)}" for o, n in checked)
        return self._change(f"replace {what} in {column}", new)

    @friendly
    def round_to(self, column=None, decimals=_NOT_GIVEN):
        """Round numbers to this many decimal places: in one column, a list of columns, or every number column.

        table.round_to("hours", 0)   table.round_to(["hours", "marks"], 1)   table.round_to(1)
        (with sdata.example("marks"))
        A number exactly halfway goes to the even neighbour (2.5 becomes 2, 3.5 becomes 4),
        like R and NumPy; this stops a run of halves from always pushing totals up.
        round_to(0) gives whole numbers; a negative number of places rounds to tens (-1), hundreds (-2).
        """
        example = 'table.round_to("hours", 0)'
        if decimals is _NOT_GIVEN and isinstance(column, numbers.Integral) and not isinstance(column, bool):
            column, decimals = None, column  # table.round_to(2): every number column
        if decimals is _NOT_GIVEN:
            decimals = 0
        if not isinstance(decimals, numbers.Integral) or isinstance(decimals, bool):
            raise SimplydataError(".round_to() needs a whole number of decimal places, like 2." + _example(example))
        decimals = int(decimals)
        number_columns = [c for c in self._columns if self._kind_of(c) == "number"]
        if column is None:
            names = number_columns
            if not names:
                raise SimplydataError("This table has no columns of numbers to round.")
        else:
            names = [column] if isinstance(column, str) else _as_list([column])
            if not names or not all(isinstance(n, str) for n in names):
                raise SimplydataError(
                    ".round_to() needs column names in quotes, then the decimal places." + _example(example)
                )
            self._need_columns(*names)
            for name in names:
                if name not in number_columns:
                    raise SimplydataError(
                        f'.round_to() rounds numbers, but "{name}" is a {self._kind_of(name)} column.\n'
                        f"Columns with numbers: {', '.join(number_columns) or 'none'}" + _example(example)
                    )
        changes = []
        for name in names:
            dtype = self._frame.schema[name]
            col = nw.col(name)
            if dtype.is_integer() and decimals >= 0:
                continue  # whole numbers stay as they are
            if decimals >= 0:
                changes.append(col.round(decimals).alias(name))
            else:
                size = 10 ** -decimals
                rounded = (col.cast(nw.Float64) / size).round(0) * size
                changes.append((rounded.cast(dtype) if dtype.is_integer() else rounded).alias(name))
        if decimals >= 0:
            to = f"{decimals} decimal place" + ("" if decimals == 1 else "s")
        else:
            to = f"the nearest {10 ** -decimals}"
        where = ", ".join(names)
        print(f"Rounded {where} to {to}.")
        new = self._frame.with_columns(*changes) if changes else self._frame
        return self._change(f"round {where} to {to}", new)

    @friendly
    def show(self, rows=None):
        """Show the table, every row and every column (or only the first rows=).

        table.show()   table.show(rows=10)   (with sdata.example("marks"))
        A table with more than 1,000 rows shows its first 1,000; choose another number with rows=.
        """
        import pandas as pd

        if rows is not None:
            rows = _check_count("show", rows, "table.show(rows=10)")
        total = len(self._frame)
        part = self._frame.head(SHOW_LIMIT if rows is None else rows)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
            _show_native(part.to_pandas())
        if len(part) < total:
            print(f"(the first {_rows(len(part))} of {total}; table.show(rows={total}) shows them all)")
        return None

    # ================================================================== biggest and smallest
    def _for_each(self, step: str, for_each, column, example: str) -> list:
        """The for_each= columns as a list (empty when not given)."""
        if for_each is None:
            return []
        groups = [for_each] if isinstance(for_each, str) else _as_list([for_each])
        if not groups or not all(isinstance(g, str) for g in groups):
            raise SimplydataError('for_each= needs a column name in quotes, like for_each="class".' + _example(example))
        self._need_columns(*groups)
        if column in groups:
            raise SimplydataError(
                f'"{column}" cannot be both the column to look at and the for_each= column.' + _example(example)
            )
        return groups

    def _keep_extreme(self, step: str, n, by, for_each, biggest: bool):
        word = "biggest" if biggest else "smallest"
        example = f'table.{step}(3, by="marks", for_each="subject")'
        if n is None:
            raise SimplydataError(f".{step}() needs how many rows to keep, like 3." + _example(example))
        n = _check_count(step, n, example)
        if by is None:
            raise SimplydataError(f'.{step}() needs by= the column to look at, like by="marks".' + _example(example))
        self._need_one_column(step, by, example)
        kind = self._kind_of(by)
        if kind not in ("number", "date"):
            raise SimplydataError(
                f'.{step}() looks for the {word} numbers or dates, but "{by}" is a {kind} column.' + _example(example)
            )
        groups = self._for_each(step, for_each, by, example)
        empty = self._empty(by)
        n_empty = self._count(empty)
        frame = self._frame.filter(~empty)
        place = nw.col(by).rank("ordinal", descending=biggest)
        tied = nw.col(by).rank("min", descending=biggest)
        if groups:
            place, tied = place.over(*groups), tied.over(*groups)
        rank, tie = _NEW, _NEW + "tie"
        frame = frame.with_columns(place.alias(rank), tied.alias(tie))
        cut_ties = _count_in(frame, (nw.col(rank) > n) & (nw.col(tie) <= n))
        kept = frame.filter(nw.col(rank) <= n).sort([*groups, rank], nulls_last=True).drop(rank, tie)
        before = len(self._frame)
        each = f" for each {', '.join(groups)}" if groups else ""
        print(f"Kept the {_rows(n)} with the {word} {by}{each}: {len(kept)} of {_rows(before)}.")
        if cut_ties:
            print(f"Note: {_rows(cut_ties)} had the same {by} as the last row kept and "
                  f"{'was' if cut_ties == 1 else 'were'} left out; rows that tie are kept in table order.")
        if n_empty:
            print(f"Note: {_rows(n_empty)} with an empty {by} cell {'was' if n_empty == 1 else 'were'} left out.")
        return self._change(f"keep the {_rows(n)} with the {word} {by}{each}", kept)

    @friendly
    def keep_biggest_rows(self, n=None, by=None, for_each=None):
        """Keep the n rows with the biggest values in a column (also for each group).

        table.keep_biggest_rows(3, by="marks", for_each="subject")   (with sdata.example("marks"))
        The rows come biggest first. Exactly n rows are kept: when rows tie at the cut, the
        ones higher up in the table are kept (a note says so). Empty cells are never kept.
        """
        return self._keep_extreme("keep_biggest_rows", n, by, for_each, biggest=True)

    @friendly
    def keep_smallest_rows(self, n=None, by=None, for_each=None):
        """Keep the n rows with the smallest values in a column (also for each group).

        table.keep_smallest_rows(3, by="marks", for_each="subject")   (with sdata.example("marks"))
        The rows come smallest first. Exactly n rows are kept: when rows tie at the cut, the
        ones higher up in the table are kept (a note says so). Empty cells are never kept.
        """
        return self._keep_extreme("keep_smallest_rows", n, by, for_each, biggest=False)

    # ================================================================== outliers
    def _outliers(self, step: str, column, for_each, rule):
        """Print the rule and its limits; give the table with an outlier column (_NEW) and the helper names."""
        example = f'table.{step}("marks", for_each="subject")'
        self._need_one_column(step, column, example)
        kind = self._kind_of(column)
        if kind != "number":
            raise SimplydataError(
                f'.{step}() looks for outliers among numbers, but "{column}" is a {kind} column.' + _example(example)
            )
        k, how = _rule_of(step, rule, example)
        groups = self._for_each(step, for_each, column, example)
        value, mid, spread, low, high = (_NEW + p for p in ("value", "mid", "spread", "low", "high"))
        frame = self._frame.with_columns(
            nw.when(~self._empty(column)).then(nw.col(column).cast(nw.Float64)).alias(value)
        )
        v = nw.col(value)

        def each(expr):
            return expr.over(*groups) if groups else expr

        if how == "IQR":  # quartiles with linear interpolation, as NumPy, pandas and R's default (type 7)
            frame = frame.with_columns(each(v.quantile(0.25, "linear")).alias(mid),
                                       each(v.quantile(0.75, "linear")).alias(spread))
            iqr = nw.col(spread) - nw.col(mid)
            frame = frame.with_columns((nw.col(mid) - k * iqr).alias(low), (nw.col(spread) + k * iqr).alias(high))
        else:  # average and sample standard deviation (n - 1), as pandas .std() and R sd()
            frame = frame.with_columns(each(v.mean()).alias(mid), each(v.std(ddof=1)).alias(spread))
            frame = frame.with_columns((nw.col(mid) - k * nw.col(spread)).alias(low),
                                       (nw.col(mid) + k * nw.col(spread)).alias(high))
        frame = frame.with_columns(((v < nw.col(low)) | (v > nw.col(high))).fill_null(False).alias(_NEW))
        frame = frame.with_columns(nw.col(_NEW).cast(nw.Int64).alias(_NEW + "count"))

        kk = _nice(k)
        if how == "IQR":
            print(f'Outliers in "{column}" by Tukey\'s rule: values more than {kk} x IQR below the first '
                  f"quartile (Q1) or above the third quartile (Q3).")
            print("IQR, the interquartile range, is Q3 - Q1: the spread of the middle half of the values.")
            names = ("Q1", "Q3")
        else:
            print(f'Outliers in "{column}": values more than {kk} standard deviations below or above the average.')
            names = ("average", "standard deviation")
        summary = [nw.col(low).min().alias("low"), nw.col(high).min().alias("high"),
                   nw.col(mid).min().alias("mid"), nw.col(spread).min().alias("spread"),
                   v.count().alias("n"), nw.col(_NEW + "count").sum().alias("found")]
        if groups:
            limits = frame.group_by(groups, drop_null_keys=False).agg(*summary).sort(groups, nulls_last=True)
        else:
            limits = frame.select(*summary)

        def limit_words(row):
            if row["low"] is None or row["low"] != row["low"]:
                return "too few numbers to judge"
            return (f"below {_nice(float(row['low']))} or above {_nice(float(row['high']))} "
                    f"({names[0]} = {_nice(float(row['mid']))}, {names[1]} = {_nice(float(row['spread']))})")

        rows = limits.rows(named=True)
        if not groups:
            print(f"Here that means {limit_words(rows[0])}.")
        else:
            print(f"For each {', '.join(groups)}:")
            for row in rows[:20]:
                label = ", ".join("(empty)" if row[g] is None else _nice(row[g]) for g in groups)
                print(f"  {label}: {limit_words(row)}; {_rows(int(row['found'] or 0))} found")
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more groups")
        if how != "IQR":
            small = [m for m in (int(r["n"] or 0) for r in rows) if m >= 2 and (m - 1) / math.sqrt(m) <= k]
            if small:
                print(f"Note: with {min(small)} numbers{' in a group' if groups else ''} no value can be more than "
                      f"{kk} standard deviations from the average, because an outlier also makes the standard "
                      f'deviation bigger. For small groups use rule="1.5 IQR".')
        words = f"{kk} x IQR" if how == "IQR" else f"{kk} standard deviations"
        if groups:
            words += f", for each {', '.join(groups)}"
        return frame, frame.columns[len(self._columns):], words

    @friendly
    def show_rows_with_outliers(self, column=None, for_each=None, rule="1.5 IQR"):
        """Show the rows whose value in a column is an outlier, and say why (the table does not change).

        table.show_rows_with_outliers("marks", for_each="subject")   (with sdata.example("marks"))
        rule="1.5 IQR" (the default) is Tukey's rule, the one box plots use: below Q1 - 1.5 x IQR
        or above Q3 + 1.5 x IQR. Also rule="3 IQR" or rule="3 standard deviations".
        Empty cells are never outliers.
        """
        frame, helpers, _ = self._outliers("show_rows_with_outliers", column, for_each, rule)
        found = frame.filter(nw.col(_NEW)).drop(helpers)
        print(f"Found {_rows(len(found))} with an outlier, of {_rows(len(frame))}.")
        if len(found):
            _show_native(nw.maybe_reset_index(found).to_native())
        return None

    @friendly
    def remove_rows_with_outliers(self, column=None, for_each=None, rule="1.5 IQR"):
        """Remove the rows whose value in a column is an outlier, and say how many and why.

        table.remove_rows_with_outliers("marks", for_each="subject")   (with sdata.example("marks"))
        rule="1.5 IQR" (the default) is Tukey's rule, the one box plots use: below Q1 - 1.5 x IQR
        or above Q3 + 1.5 x IQR. Also rule="3 IQR" or rule="3 standard deviations".
        Rows with an empty cell stay. Look first with show_rows_with_outliers(), and say in your
        report which rule removed how many rows.
        """
        frame, helpers, words = self._outliers("remove_rows_with_outliers", column, for_each, rule)
        new = frame.filter(~nw.col(_NEW)).drop(helpers)
        before = len(frame)
        print(f"Removed {before - len(new)} of {_rows(before)}. {_rows(len(new))} left.")
        return self._change(f"remove rows with outliers in {column} ({words})", new)
