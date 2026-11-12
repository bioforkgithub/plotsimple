"""Summaries: averages, totals, counts, correlation. They never change the table.

Rules (the same as the rest of plotsimple, see _stats.py):

* empty cells are skipped, and a short note says how many were skipped
* standard deviation uses the sample formula (divide by n - 1), like pandas and R
* without for_each= you get one plain number; with for_each= you get a NEW small table
  with one row per group, sorted by group. Rows with an empty group cell form their own
  group, shown last.

Correlation strength words (size of r, whichever direction), after Evans (1996):

    below 0.2   very weak
    0.2 - 0.4   weak
    0.4 - 0.6   medium
    0.6 - 0.8   strong
    0.8 - 1     very strong

The p-value words: below 0.001 "very unlikely to be chance", below 0.05 "unlikely to be
chance", otherwise "could easily be chance".
"""

from __future__ import annotations

import math

import narwhals as nw

from .._errors import PlotsimpleError, friendly

# word -> (start of the result column name, narwhals calculation)
_SUMMARIES = {
    "average_of": ("average", lambda c: nw.col(c).mean()),
    "median_of": ("median", lambda c: nw.col(c).median()),
    "total_of": ("total", lambda c: nw.col(c).sum()),
    "smallest_of": ("smallest", lambda c: nw.col(c).min()),
    "biggest_of": ("biggest", lambda c: nw.col(c).max()),
    "standard_deviation_of": ("standard deviation", lambda c: nw.col(c).std(ddof=1)),
}

CORRELATION_METHODS = ("pearson", "spearman")

# lower edge of each band, biggest first
STRENGTH_WORDS = ((0.8, "very strong"), (0.6, "strong"), (0.4, "medium"), (0.2, "weak"), (0.0, "very weak"))


def _plain(value):
    """A plain Python number (not a NumPy one); empty becomes nan."""
    if value is None:
        return math.nan
    if hasattr(value, "item"):
        value = value.item()
    return value


def _as_list(step: str, option: str, value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    raise PlotsimpleError(
        f"{step}() needs {option} as a column name in quotes, or a list of names.\n"
        f'For example, with ps.example("marks"): table.{step}("marks", for_each="class")\n'
        f'or table.{step}(["marks", "hours"], for_each=["class", "subject"])'
    )


def _strength(r: float) -> str:
    r = round(r, 2)  # the words always match the r that is printed
    size = abs(r)
    word = next(w for edge, w in STRENGTH_WORDS if size >= edge)
    if r > 0:
        return f"{word}, going up"
    if r < 0:
        return f"{word}, going down"
    return "no link at all"


def _p_words(p: float) -> str:
    if p < 0.0001:
        shown = "p < 0.0001"
    elif p < 0.001:
        shown = f"p = {p:.4f}"
    else:
        shown = f"p = {p:.3f}"
    if p < 0.001:
        return f"very unlikely to be chance ({shown})"
    if p < 0.05:
        return f"unlikely to be chance ({shown})"
    return f"could easily be chance ({shown})"


class SummaryWords:
    # ------------------------------------------------------------------ helpers
    def _number_frame(self, step: str, columns: list[str]):
        """Check the columns hold numbers; give back the table with NaN made empty."""
        self._need_columns(*columns)
        schema = self._frame.schema
        for name in columns:
            dtype = schema[name]
            if dtype.is_numeric():
                continue
            if dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum:
                what = "text"
            elif dtype == nw.Boolean:
                what = "True/False values"
            elif dtype.is_temporal():
                what = "dates or times"
            else:
                what = "no numbers"
            numbers = [c for c, d in schema.items() if d.is_numeric()]
            hint = f"Columns with numbers: {', '.join(numbers)}" if numbers else "This table has no columns of numbers."
            raise PlotsimpleError(
                f'{step} needs a column of numbers; "{name}" has {what}.\n'
                f"{hint}\n"
                f'For example, with ps.example("marks"): table.{step}("marks")'
            )
        floats = [c for c in columns if schema[c].is_float()]
        if floats:
            return self._frame.with_columns(nw.col(c).fill_nan(None) for c in floats)
        return self._frame

    def _group_columns(self, step: str, for_each, columns=()) -> list[str]:
        groups = _as_list(step, "for_each=", for_each)
        self._need_columns(*groups)
        for name in groups:
            if name in columns:
                raise PlotsimpleError(
                    f'"{name}" cannot be both the column to summarise and the for_each= column.\n'
                    f'For example, with ps.example("marks"): table.{step}("marks", for_each="class")'
                )
        return groups

    @staticmethod
    def _note_empty(frame, columns: list[str]):
        counts = frame.select(nw.col(c).null_count().alias(c) for c in columns).row(0)
        skipped = [(c, int(n)) for c, n in zip(columns, counts) if n]
        if not skipped:
            return
        if len(skipped) == 1:
            name, n = skipped[0]
            cells = "empty cell" if n == 1 else "empty cells"
            verb = "was" if n == 1 else "were"
            print(f'Note: {n} {cells} in "{name}" {verb} skipped.')
        else:
            parts = ", ".join(f'{n} in "{c}"' for c, n in skipped)
            print(f"Note: empty cells were skipped: {parts}.")

    def _grouped(self, frame, groups: list[str], exprs):
        result = frame.group_by(groups, drop_null_keys=False).agg(*exprs).sort(groups, nulls_last=True)
        return self._new_table(nw.maybe_reset_index(result))

    def _summary(self, step: str, column, for_each):
        start, make = _SUMMARIES[step]
        columns = _as_list(step, "the column", column)
        groups = self._group_columns(step, for_each, columns) if for_each is not None else None
        frame = self._number_frame(step, columns)
        exprs = [make(c).alias(f"{start} {c}") for c in columns]
        if groups is None:
            result = frame.select(*exprs)
            self._note_empty(frame, columns)
            if isinstance(column, str):
                return _plain(result.item())
            return self._new_table(result)
        result = self._grouped(frame, groups, exprs)
        self._note_empty(frame, columns)
        return result

    # ------------------------------------------------------------------ number words
    @friendly
    def average_of(self, column, for_each=None):
        """The average (mean) of a column; empty cells are skipped.

        table.average_of("marks", for_each="class")   (with ps.example("marks"))
        """
        return self._summary("average_of", column, for_each)

    @friendly
    def median_of(self, column, for_each=None):
        """The middle number of a column once it is sorted; empty cells are skipped.

        table.median_of("marks", for_each="class")   (with ps.example("marks"))
        """
        return self._summary("median_of", column, for_each)

    @friendly
    def total_of(self, column, for_each=None):
        """All the numbers in a column added up; empty cells are skipped.

        table.total_of("count", for_each="shop")   (with ps.example("fruit"))
        """
        return self._summary("total_of", column, for_each)

    @friendly
    def smallest_of(self, column, for_each=None):
        """The smallest number in a column; empty cells are skipped.

        table.smallest_of("temperature", for_each="city")   (with ps.example("weather"))
        """
        return self._summary("smallest_of", column, for_each)

    @friendly
    def biggest_of(self, column, for_each=None):
        """The biggest number in a column; empty cells are skipped.

        table.biggest_of("temperature", for_each="city")   (with ps.example("weather"))
        """
        return self._summary("biggest_of", column, for_each)

    @friendly
    def standard_deviation_of(self, column, for_each=None):
        """How spread out the numbers are: the sample standard deviation (divide by n - 1).

        It is the same as pandas .std() and R sd(). Empty cells are skipped.
        table.standard_deviation_of("marks", for_each="class")   (with ps.example("marks"))
        """
        return self._summary("standard_deviation_of", column, for_each)

    # ------------------------------------------------------------------ counting words
    @friendly
    def count_rows(self, for_each=None):
        """How many rows the table has, or how many in each group.

        table.count_rows(for_each="class")   (with ps.example("marks"))
        """
        if for_each is None:
            return len(self._frame)
        groups = self._group_columns("count_rows", for_each)
        summary = self._grouped(self._frame, groups, [nw.len().alias("count")])
        return self._new_table(summary._frame.with_columns(nw.col("count").cast(nw.Int64)))

    @friendly
    def count_each(self, column):
        """Each different value in a column, how many times it appears, and its percent.

        Biggest count first. Empty cells count as a value of their own, so the percents
        are out of all rows. Percents are rounded to 1 decimal place.
        table.count_each("fruit")   (with ps.example("fruit"))
        """
        if not isinstance(column, str):
            raise PlotsimpleError(
                "count_each() needs one column name in quotes.\n"
                'For example, with ps.example("fruit"): table.count_each("fruit")'
            )
        self._need_columns(column)
        if column in ("count", "percent"):
            raise PlotsimpleError(
                f'count_each() makes columns called "count" and "percent", so it cannot count a column called "{column}".\n'
                f'Rename it first, for example table.rename_column("{column}", to="number").'
            )
        total = len(self._frame)
        result = (
            self._frame.group_by([column], drop_null_keys=False)
            .agg(nw.len().alias("count"))
            .with_columns(nw.col("count").cast(nw.Int64))
            .with_columns((nw.col("count") * 100.0 / max(total, 1)).round(1).alias("percent"))
            .sort(["count", column], descending=[True, False], nulls_last=True)
        )
        return self._new_table(nw.maybe_reset_index(result))

    # ------------------------------------------------------------------ correlation
    @friendly
    def correlation_between(self, column_a, column_b, method="pearson"):
        """How strongly two columns of numbers go up or down together, from -1 to 1.

        method="pearson" (a straight-line link, the usual one) or "spearman" (uses ranks,
        for any link that keeps going one way). Prints the strength in words and the
        p-value from SciPy, and gives back the number r. Rows with an empty cell in either
        column are skipped.
        Strength words for the size of r: below 0.2 very weak, from 0.2 weak, from 0.4
        medium, from 0.6 strong, from 0.8 very strong (after Evans 1996).
        p below 0.001: "very unlikely to be chance"; below 0.05: "unlikely to be chance";
        otherwise "could easily be chance".
        A correlation does not show that one thing causes the other.
        table.correlation_between("hours", "marks")   (with ps.example("marks"))
        """
        step = "correlation_between"
        for value in (column_a, column_b):
            if not isinstance(value, str):
                raise PlotsimpleError(
                    f"{step}() needs two column names in quotes.\n"
                    f'For example, with ps.example("marks"): table.{step}("hours", "marks")'
                )
        self._check_choice(step, "method", method, CORRELATION_METHODS)
        if column_a == column_b:
            raise PlotsimpleError(
                f"{step}() needs two different columns.\n"
                f'For example, with ps.example("marks"): table.{step}("hours", "marks")'
            )
        frame = self._number_frame(step, [column_a, column_b])
        pairs = frame.select(column_a, column_b).drop_nulls()
        skipped = len(frame) - len(pairs)
        if len(pairs) < 3:
            raise PlotsimpleError(
                f'{step}() needs at least 3 rows where both "{column_a}" and "{column_b}" have a number; '
                f"this table has {len(pairs)}."
            )
        a = pairs[column_a].to_numpy().astype(float)
        b = pairs[column_b].to_numpy().astype(float)
        for name, values in ((column_a, a), (column_b, b)):
            if values.min() == values.max():
                raise PlotsimpleError(
                    f'{step}() needs numbers that change, but every number in "{name}" is the same, '
                    "so there is no correlation to find."
                )
        from scipy import stats

        if method == "pearson":
            r, p = stats.pearsonr(a, b)
            name = "Pearson"
        else:
            r, p = stats.spearmanr(a, b)
            name = "Spearman"
        r, p = float(r), float(p)
        if skipped:
            rows = "row" if skipped == 1 else "rows"
            verb = "was" if skipped == 1 else "were"
            print(f'Note: {skipped} {rows} with an empty cell in "{column_a}" or "{column_b}" {verb} skipped.')
        print(
            f'The correlation between "{column_a}" and "{column_b}" is r = {r:.2f} ({name}, {len(pairs)} rows): '
            f"{_strength(r)}, and {_p_words(p)}."
        )
        return r
