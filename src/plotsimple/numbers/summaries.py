"""Summaries: average, median, total ... They never change the numbers; they give back an answer.

Naming: the same words as tables, without "_of". A table needs to be told which column
(table.average_of("marks")); numbers are already one set, so there is nothing to name:
nums.average(). Grids take for_each="row" or for_each="column", like tables take
for_each="class", and then give back one answer per row or column as new numbers.

Rules (the same as tables, see table/summaries.py):

* empty values are skipped, and a short note says how many were skipped
* standard deviation uses the sample formula (divide by n - 1), like pandas and R
* percentile uses NumPy's usual "linear" method, the same as pandas and R's default (type 7)
"""

from __future__ import annotations

import warnings

import numpy as np

from .._errors import PlotsimpleError, choice_error, friendly
from ._base import amount, is_number, plain

FOR_EACH = ("row", "column")


def _standard_deviation(values, axis=None):
    return np.nanstd(values, ddof=1, axis=axis)


# word -> (NumPy calculation, what it means, example for a list, example for a grid)
SUMMARIES = {
    "average": (np.nanmean, "The average (mean) of the numbers; empty values are skipped."),
    "median": (np.nanmedian, "The middle number once the numbers are sorted; empty values are skipped."),
    "total": (np.nansum, "All the numbers added up; empty values are skipped."),
    "smallest": (np.nanmin, "The smallest number; empty values are skipped."),
    "biggest": (np.nanmax, "The biggest number; empty values are skipped."),
    "standard_deviation": (
        _standard_deviation,
        "How spread out the numbers are: the sample standard deviation (divide by n - 1).\n\n"
        "        It is the same as pandas .std() and R sd(). Empty values are skipped.",
    ),
}


def _quiet(func, *args, **kwargs):
    """Run a NumPy summary without its warnings: empty answers are explained by our own notes."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args, **kwargs)


class SummaryCore:
    """The shared engine; the words themselves are made below for Numbers and for grids."""

    def _note_skipped(self):
        n = int(self._empty_mask().sum())
        if n:
            print(f"Note: {amount(n, 'empty value')} {'was' if n == 1 else 'were'} skipped.")

    def _summary(self, step: str, func, for_each=None, **more):
        values = self._values
        if for_each is not None:
            if for_each not in FOR_EACH:
                raise choice_error(step, "for_each", for_each, FOR_EACH)
            axis = 1 if for_each == "row" else 0
            if values.size == 0:
                result = np.full(values.shape[1 - axis], np.nan)
            else:
                result = _quiet(func, values, axis=axis, **more)
            self._note_skipped()
            return self._wrap(np.asarray(result))
        filled = values[~self._empty_mask()]
        if filled.size == 0:
            if func is np.nansum:
                return 0
            print(f"Note: there are no numbers here, so the {step.replace('_', ' ')} is empty (nan).")
            return float("nan")
        if func is _standard_deviation and filled.size < 2:
            print("Note: the standard deviation needs at least 2 numbers, so it is empty (nan).")
            return float("nan")
        self._note_skipped()
        return plain(_quiet(func, filled, **more))

    def _check_percent(self, percent):
        if not is_number(percent) or not 0 <= percent <= 100:
            raise PlotsimpleError(
                ".percentile() needs a number from 0 to 100: percentile(90) is the number that 90 % "
                "of the numbers are below." + self._example("percentile(90)")
            )
        return float(percent)


def _make(step: str, func, doc: str, grid: bool):
    if grid:
        def word(self, for_each=None):
            return self._summary(step, func, for_each)
        word.__doc__ = (f"{doc}\n\n        grid.{step}()   or one answer per row or column:   "
                        f'grid.{step}(for_each="row")\n        ')
    else:
        def word(self):
            return self._summary(step, func)
        word.__doc__ = f"{doc}\n\n        nums.{step}()\n        "
    word.__name__ = word.__qualname__ = step
    return friendly(word)


class SummaryWords(SummaryCore):
    """Summaries for a list of numbers."""

    average = _make("average", *SUMMARIES["average"], grid=False)
    median = _make("median", *SUMMARIES["median"], grid=False)
    total = _make("total", *SUMMARIES["total"], grid=False)
    smallest = _make("smallest", *SUMMARIES["smallest"], grid=False)
    biggest = _make("biggest", *SUMMARIES["biggest"], grid=False)
    standard_deviation = _make("standard_deviation", *SUMMARIES["standard_deviation"], grid=False)

    @friendly
    def percentile(self, percent=None):
        """The number that this percent of the numbers are below: percentile(50) is the median.

        nums.percentile(90)
        Uses NumPy's usual "linear" method, the same as pandas and R. Empty values are skipped.
        """
        return self._summary("percentile", np.nanpercentile, q=self._check_percent(percent))

    @friendly
    def count_each(self):
        """Each different number, how many times it appears, and its percent, as a table.

        Biggest count first. Empty values count as a value of their own, so the percents are
        out of all numbers. Percents are rounded to 1 decimal place.
        nums.count_each()
        """
        import pandas as pd

        from ..table import Table

        series = pd.Series(self._values, name="number")
        counts = series.value_counts(dropna=False).rename("count").reset_index()
        counts["count"] = counts["count"].astype("int64")
        counts["percent"] = (counts["count"] * 100.0 / max(len(series), 1)).round(1)
        counts = counts.sort_values(["count", "number"], ascending=[False, True], na_position="last",
                                    kind="stable").reset_index(drop=True)
        return Table(counts)


class GridSummaryWords(SummaryCore):
    """Summaries for a grid: one answer, or one per row or column with for_each=."""

    average = _make("average", *SUMMARIES["average"], grid=True)
    median = _make("median", *SUMMARIES["median"], grid=True)
    total = _make("total", *SUMMARIES["total"], grid=True)
    smallest = _make("smallest", *SUMMARIES["smallest"], grid=True)
    biggest = _make("biggest", *SUMMARIES["biggest"], grid=True)
    standard_deviation = _make("standard_deviation", *SUMMARIES["standard_deviation"], grid=True)

    @friendly
    def percentile(self, percent=None, for_each=None):
        """The number that this percent of the numbers are below: percentile(50) is the median.

        grid.percentile(90)   or   grid.percentile(90, for_each="column")
        Uses NumPy's usual "linear" method, the same as pandas and R. Empty values are skipped.
        """
        return self._summary("percentile", np.nanpercentile, for_each, q=self._check_percent(percent))
