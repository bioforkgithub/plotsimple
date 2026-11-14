"""Summaries: averages, totals, counts, correlation. They never change the table.

Rules (the same as the rest of simplydata, see _stats.py):

* empty cells are skipped, and a short note says how many were skipped
* standard deviation uses the sample formula (divide by n - 1), like pandas and R
* without for_each= you get one plain number; with for_each= you get a NEW small table
  with one row per group, sorted by group. Rows with an empty group cell form their own
  group, shown last.
* every summary also prints its answer as one short sentence, so a script needs no print()
* every="month" (or "hour", "day", "week", "quarter", "year") groups a date column in
  for_each= into periods; each period is shown in the new table by its first day (a week
  starts on Monday, as ISO 8601)
* across="subject" gives a cross table: one row per for_each= group, one column per value
  of across= (pandas pivot_table / crosstab)
* percents are numbers from 0 to 100 (never fractions, never text); tables keep every
  digit, sentences show one decimal: 41.7 %. Percent intervals are Clopper-Pearson exact
  95 % intervals (R binom.test)
* the interquartile range uses the usual linear rule for quartiles (R type 7, NumPy and
  pandas default)

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

import datetime as _dt
import math

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, choice_error, friendly
from .._stats import _plain_name
from .conditions import _empty_in
from .rows import _NOT_GIVEN

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
CORRELATION_SHOW = ("r", "p-value", "each pair")
STRONGEST_PAIRS = 3  # a correlation table prints this many of its strongest pairs

# every= : the time step -> narwhals truncate code (weeks are done by hand: Monday start)
EVERY = {"hour": "1h", "day": "1d", "week": None, "month": "1mo", "quarter": "1q", "year": "1y"}
EVERY_EXAMPLE = ('table.make_dates("year")\n'
                 '              table.average_of("population", for_each="year", every="year")'
                 '   (with sdata.example("growth"))')

# summary_of(show=[...]): the measure names, the same as the _of words and chart show=/error=
MEASURES = ("count", "average", "median", "total", "smallest", "biggest", "standard deviation", "variance",
            "standard error", "confidence interval", "range", "interquartile range", "empty cells",
            "geometric mean", "skewness", "kurtosis", "coefficient of variation", "median absolute deviation",
            "trimmed average", "most common", "percentile")
MEASURE_ALIASES = {"mean": "average", "sum": "total", "sd": "standard deviation", "se": "standard error",
                   "ci": "confidence interval", "iqr": "interquartile range", "geometric average": "geometric mean",
                   "gmean": "geometric mean", "skew": "skewness", "cv": "coefficient of variation",
                   "mad": "median absolute deviation", "trimmed mean": "trimmed average", "mode": "most common",
                   "quantile": "percentile"}
# measures worked out with NumPy / SciPy on each group's numbers (the others come from narwhals)
ARRAY_MEASURES = ("geometric mean", "skewness", "kurtosis", "coefficient of variation", "median absolute deviation",
                  "trimmed average", "most common", "percentile")
TRIM_PERCENT = 10  # "trimmed average": 10 % cut from each end, unless another percent is given
MAD_CONSTANT = 1.4826  # R mad(): makes the median absolute deviation match the sd for bell-shaped values
LEAST_VALUES = {"skewness": 3, "kurtosis": 4}  # fewer values than this give an empty cell
DEFAULT_MEASURES = ("count", "average", "standard deviation", "smallest", "median", "biggest")
SUMMARY_LINES = 12  # summary_of prints at most this many rows; the table has them all


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
    raise SimplydataError(
        f"{step}() needs {option} as a column name in quotes, or a list of names.\n"
        f'For example, with sdata.example("marks"): table.{step}("marks", for_each="class")\n'
        f'or table.{step}(["marks", "hours"], for_each=["class", "subject"])'
    )


GROUPS_IN_SENTENCE = 8  # a sentence lists at most this many groups; the table has them all


def _say_number(value) -> str:
    """A number for a sentence: 61, 61.24, 0.0042 (never more digits than people need)."""
    value = _plain(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "empty"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 0.01:
        return f"{round(value, 2):g}" if abs(value) < 1e6 else f"{value:.4g}"
    return f"{value:.2g}"


def _missing(value) -> bool:
    """True for None, NaN and NaT (the empty values a table can give back)."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _as_datetime(value):
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        return value.to_pydatetime()
    return value


def _say_label(value, every=None) -> str:
    """A group name for a sentence; dates as 2026-03-15, a period as 2026-03 (every="month")."""
    if _missing(value):
        return "(empty)"
    value = _plain(_as_datetime(value))
    if isinstance(value, _dt.date):
        return _say_period(value, every)
    return str(value)


def _say_period(value, every=None) -> str:
    if every == "year":
        return f"{value.year}"
    if every == "quarter":
        return f"{value.year} Q{(value.month - 1) // 3 + 1}"
    if every == "month":
        return f"{value.year}-{value.month:02d}"
    if every == "week":
        return f"week of {value:%Y-%m-%d}"
    if isinstance(value, _dt.datetime) and (every == "hour" or (every is None and (value.hour or value.minute
                                                                                     or value.second))):
        return f"{value:%Y-%m-%d %H:%M}"
    return f"{value:%Y-%m-%d}"


def _week_start(value):
    """The Monday that starts the week of a date (ISO 8601 weeks), at midnight."""
    value = _as_datetime(value)
    if isinstance(value, _dt.datetime):
        value = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return value - _dt.timedelta(days=value.weekday())


def _say_percent(value) -> str:
    """41.7 % (one decimal, a space before the sign, as everywhere in simplydata)."""
    if _missing(value):
        return "empty"
    return f"{float(value):.1f} %"


def _order(value):
    """Sort key for group names: numbers, then everything else, then empty last."""
    if _missing(value):
        return (2, "", 0)
    value = _plain(_as_datetime(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, "", value)
    return (1, type(value).__name__, value)


def _key(row) -> tuple:
    """A row of group values as a dictionary key: empty values all become None."""
    return tuple(None if _missing(v) else _plain(_as_datetime(v)) for v in row)


def _say_groups(frame, groups: list[str], column: str, periods=None, say=None) -> str:
    """ "A 62.1, B 58.3, C 65" from a small table: one item per row, the groups then a value."""
    periods = periods or {}
    say = say or _say_number
    rows = frame.select(*groups, column).rows()
    items = [" / ".join(_say_label(v, periods.get(g)) for g, v in zip(groups, row[:-1])) + " " + say(row[-1])
             for row in rows[:GROUPS_IN_SENTENCE]]
    text = ", ".join(items)
    if len(rows) > GROUPS_IN_SENTENCE:
        text += f" ... and {len(rows) - GROUPS_IN_SENTENCE} more (the new table has them all)"
    return text


def _names(names) -> str:
    """ "class" or "class" and "subject" """
    quoted = [f'"{n}"' for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _each(groups, periods=None) -> str:
    """ "class" and "subject", or month of "date" (for every="month") """
    periods = periods or {}
    return " and ".join(f'{periods[g]} of "{g}"' if g in periods else f'"{g}"' for g in groups)


def _period_note(periods) -> None:
    for every in dict.fromkeys(periods.values()):
        first = {"week": "its Monday", "hour": "the start of the hour"}.get(every, "its first day")
        print(f"Note: in the new table each {every} is shown by {first}.")


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


def _share(hits: int, rows: int):
    """Percent and its 95 % Clopper-Pearson exact interval (R binom.test), all from 0 to 100."""
    if rows == 0:
        return (None, None, None)
    from scipy import stats

    low, high = stats.binomtest(hits, rows).proportion_ci(confidence_level=0.95, method="exact")
    return (hits * 100.0 / rows, float(low) * 100.0, float(high) * 100.0)


def _ordinal(number: float) -> str:
    """90 -> "90th", 1 -> "1st", 2.5 -> "2.5th"."""
    text = f"{number:g}"
    if not float(number).is_integer():
        return text + "th"
    whole = int(number)
    if 10 <= whole % 100 <= 20:
        return text + "th"
    return text + {1: "st", 2: "nd", 3: "rd"}.get(whole % 10, "th")


def _check_percent(step: str, percent, option="percent", example=None) -> float:
    example = example or f'table.{step}("marks", 90, for_each="class")   (with sdata.example("marks"))'
    if not isinstance(percent, (int, float)) or isinstance(percent, bool) or not math.isfinite(percent):
        raise SimplydataError(f"{step}() needs the {option} as a number from 0 to 100.\nFor example:  {example}")
    if not 0 <= percent <= 100:
        raise SimplydataError(f"The {option} should be from 0 to 100 (it is {percent:g}).\nFor example:  {example}")
    return float(percent)


def _measure_name(step: str, item):
    """The measure a show= item names: "percentile 90" and "90th percentile" both give
    "percentile 90"; "trimmed average" gives "trimmed average 10 %" (R mean(trim = 0.1))."""
    import re

    name = _plain_name(item)
    if not isinstance(name, str):
        raise choice_error(step, "show", item, list(MEASURES))
    example = f'table.{step}("marks", show=["percentile 90", "trimmed average 20"])   (with sdata.example("marks"))'
    found = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*(?:percentile|quantile)"
                         r"|(?:percentile|quantile)\s*(\d+(?:\.\d+)?))", name)
    if found:
        number = float(found.group(1) or found.group(2))
        if number > 100:
            raise SimplydataError(f"A percentile goes from 0 to 100, but show= asks for {number:g}.\n"
                                  f"For example:  {example}")
        return f"percentile {number:g}"
    found = re.fullmatch(r"trimmed (?:average|mean)\s*(\d+(?:\.\d+)?)?\s*%?", name)
    if found:
        number = float(found.group(1)) if found.group(1) else float(TRIM_PERCENT)
        if number >= 50:
            raise SimplydataError(
                f"A trimmed average leaves out a percent from each end, so it must be below 50 (it is {number:g}).\n"
                f"For example:  {example}"
            )
        return f"trimmed average {number:g} %"
    name = MEASURE_ALIASES.get(name, name)
    if name == "percentile":
        raise SimplydataError(f'show="percentile" needs the percent, for example "percentile 90".\nFor example:  {example}')
    if name not in MEASURES:
        raise choice_error(step, "show", item, list(MEASURES))
    return name


def _is_array_measure(name) -> bool:
    return isinstance(name, str) and (name in ARRAY_MEASURES or name.startswith(("percentile ", "trimmed average ")))


def _array_measure(name: str, values):
    """One group's value of a NumPy / SciPy measure: (value, why it is empty, values as common).

    Definitions (as R, so results can be checked there):
      geometric mean            exp(mean(log(x))); only for numbers above 0
      skewness                  bias-corrected sample skewness G1 (SciPy skew(bias=False),
                                e1071 skewness(type = 2), SAS, SPSS, Excel SKEW)
      kurtosis                  bias-corrected excess kurtosis G2, 0 for a bell shape (SciPy
                                kurtosis(bias=False), e1071 kurtosis(type = 2), Excel KURT)
      coefficient of variation  sd / average, sd with n - 1 (SciPy variation(ddof=1)); a ratio
      median absolute deviation 1.4826 * median(|x - median(x)|), as R mad()
      trimmed average p %       floor(n * p / 100) values cut from each end, as R mean(trim =)
      most common               the value that turns up most often; the smallest when tied
      percentile p              the usual linear rule (R quantile type 7, NumPy default)
    """
    from scipy import stats

    n = len(values)
    if n == 0:
        return None, None, None
    if name == "geometric mean":
        if (values <= 0).any():
            return None, "needs numbers above 0", None
        return float(np.exp(np.mean(np.log(values)))), None, None
    if name in LEAST_VALUES and n < LEAST_VALUES[name]:
        return None, f"needs at least {LEAST_VALUES[name]} values", None
    if name in ("skewness", "kurtosis") and np.ptp(values) == 0:
        return None, "has no spread (every value is the same)", None
    if name == "skewness":
        return float(stats.skew(values, bias=False)), None, None
    if name == "kurtosis":
        return float(stats.kurtosis(values, fisher=True, bias=False)), None, None
    if name == "coefficient of variation":
        if n < 2:
            return None, "needs at least 2 values", None
        mean = float(values.mean())
        if mean == 0:
            return None, "is not defined when the average is 0", None
        return float(values.std(ddof=1)) / mean, None, None
    if name == "median absolute deviation":
        return MAD_CONSTANT * float(np.median(np.abs(values - np.median(values)))), None, None
    if name == "most common":
        found, counts = np.unique(values, return_counts=True)
        top = counts.max()
        return float(found[counts == top][0]), None, found[counts == top]
    number = float(name.split()[-2] if name.endswith("%") else name.split()[-1])
    if name.startswith("percentile"):
        return float(np.percentile(values, number)), None, None
    cut = math.floor(n * number / 100)
    return float(np.sort(values)[cut:n - cut].mean()), None, None


def _measure_values(names, raw: dict) -> dict:
    """The summary_of measures of one group, from its raw pieces (count, mean, sd ...)."""
    n = int(raw["n"] or 0)

    def number(value):
        return None if _missing(value) else float(value)

    sd = number(raw["sd"]) if n >= 2 else None
    out = {}
    for name in names:
        if name == "count":
            out[name] = n
        elif name == "empty cells":
            out[name] = int(raw["empty"] or 0)
        elif n == 0:
            out[name] = None if name not in ("total",) else 0.0
            if name == "confidence interval":
                out["average 95 % CI low"] = out["average 95 % CI high"] = None
        elif name == "average":
            out[name] = number(raw["mean"])
        elif name == "median":
            out[name] = number(raw["median"])
        elif name == "total":
            out[name] = number(raw["sum"])
        elif name == "smallest":
            out[name] = number(raw["min"])
        elif name == "biggest":
            out[name] = number(raw["max"])
        elif name == "standard deviation":
            out[name] = sd
        elif name == "variance":
            out[name] = number(raw["var"]) if n >= 2 else None
        elif name == "standard error":
            out[name] = sd / math.sqrt(n) if sd is not None else None
        elif name == "confidence interval":
            if sd is None:
                out["average 95 % CI low"] = out["average 95 % CI high"] = None
            else:
                from scipy import stats

                half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
                mean = float(raw["mean"])
                out["average 95 % CI low"], out["average 95 % CI high"] = mean - half, mean + half
        elif name == "range":
            out[name] = number(raw["max"]) - number(raw["min"])
        elif name == "interquartile range":
            out[name] = number(raw["q3"]) - number(raw["q1"])
    return out


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
            raise SimplydataError(
                f'{step} needs a column of numbers; "{name}" has {what}.\n'
                f"{hint}\n"
                f'For example, with sdata.example("marks"): table.{step}("marks")'
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
                raise SimplydataError(
                    f'"{name}" cannot be both the column to summarise and the for_each= column.\n'
                    f'For example, with sdata.example("marks"): table.{step}("marks", for_each="class")'
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

    def _periods(self, step: str, frame, groups, every):
        """every="month": the date columns in for_each= become the first day of their month.

        Gives back (frame, {date column: "month"}). Without every= nothing changes."""
        if every is None:
            return frame, {}
        name = _plain_name(every)
        if name not in EVERY:
            raise choice_error(step, "every", every, list(EVERY))
        if not groups:
            raise SimplydataError(
                f"every= needs the column of dates in for_each=, to say which dates to group.\n"
                f"For example:  {EVERY_EXAMPLE}"
            )
        schema = frame.schema
        dates = [g for g in groups if schema[g] == nw.Date or schema[g] == nw.Datetime]
        if not dates:
            kinds = ", ".join(f'"{g}"' for g in groups)
            text = [g for g in groups if schema[g] == nw.String]
            hint = (f'If {_names(text)} holds dates written as text, make them dates first: '
                    f'table.make_dates("{text[0]}").\n' if text else "")
            raise SimplydataError(
                f"every= groups dates into periods, but none of the for_each= columns ({kinds}) holds dates.\n"
                f"{hint}For example:  {EVERY_EXAMPLE}"
            )
        frame = nw.maybe_reset_index(frame)
        for g in dates:
            if name == "week":
                values = [None if _missing(v) else _week_start(v) for v in frame[g].to_list()]
                frame = frame.with_columns(
                    nw.new_series(g, values, schema[g], backend=frame.implementation))
            elif schema[g] == nw.Date and name == "hour":
                continue  # a date has no hours: each day stays as it is
            else:
                frame = frame.with_columns(nw.col(g).dt.truncate(EVERY[name]))
        return frame, {g: name for g in dates}

    def _cross(self, frame, groups: list[str], across: str, expr, fill=None, periods=None):
        """A cross table: one row per group, one column per value of across=, expr in each cell."""
        long = frame.group_by([*groups, across], drop_null_keys=False).agg(expr.alias("__value"))
        rows = long.select(*groups, across, "__value").rows()
        row_keys = sorted({_key(r[:-2]) for r in rows}, key=lambda k: tuple(_order(v) for v in k))
        col_keys = sorted({_key(r[-2:-1])[0] for r in rows}, key=_order)
        cells = {(_key(r[:-2]), _key(r[-2:-1])[0]): _plain(r[-1]) for r in rows}
        data = {g: [k[i] for k in row_keys] for i, g in enumerate(groups)}
        for c in col_keys:
            label = _say_label(c, (periods or {}).get(across))
            if label in data:
                label = f"{across} {label}"
            data[label] = [cells.get((k, c), fill) for k in row_keys]
        result = nw.from_dict(data, backend=frame.implementation)
        return self._new_table(result), [str(c) for c in data if c not in groups]

    @staticmethod
    def _say_cross(table, groups, columns, say=None, periods=None):
        say = say or _say_number
        rows = table._frame.select(*groups, *columns).rows()
        for row in rows[:GROUPS_IN_SENTENCE]:
            name = " / ".join(_say_label(v, (periods or {}).get(g)) for g, v in zip(groups, row))
            cells = ", ".join(f"{c} {say(v)}" for c, v in zip(columns, row[len(groups):]))
            print(f"  {name}: {cells}")
        if len(rows) > GROUPS_IN_SENTENCE:
            print(f"  ... and {len(rows) - GROUPS_IN_SENTENCE} more rows (the new table has them all)")

    def _check_across(self, step: str, across, groups, columns=()):
        example = f'table.{step}("marks", for_each="class", across="subject")   (with sdata.example("marks"))'
        if step == "count_rows":
            example = 'table.count_rows(for_each="class", across="subject")   (with sdata.example("marks"))'
        if not isinstance(across, str):
            raise SimplydataError(f"{step}() needs across= as one column name in quotes.\nFor example:  {example}")
        if groups is None:
            raise SimplydataError(
                f"across= makes a cross table: for_each= gives its rows and across= its columns.\n"
                f"For example:  {example}"
            )
        self._need_columns(across)
        if across in groups or across in columns:
            raise SimplydataError(
                f'"{across}" cannot be both the across= column and a for_each= column or the column to summarise.\n'
                f"For example:  {example}"
            )

    def _summary(self, step: str, column, for_each, every=None, across=None, measure=None):
        """measure=(start of the column name, calculation, words for sentences) for words not in _SUMMARIES."""
        start, make = _SUMMARIES[step] if measure is None else measure[:2]
        said = start if measure is None else measure[2]
        columns = _as_list(step, "the column", column)
        groups = self._group_columns(step, for_each, columns) if for_each is not None else None
        if across is not None:
            self._check_across(step, across, groups, columns)
            if len(columns) > 1:
                raise SimplydataError(
                    f"across= makes a cross table of one column. Give one column, or use summary_of() for several.\n"
                    f'For example:  table.{step}("marks", for_each="class", across="subject")'
                    '   (with sdata.example("marks"))'
                )
        frame = self._number_frame(step, columns)
        frame, periods = self._periods(step, frame, groups, every)
        exprs = [make(c).alias(f"{start} of {c}") for c in columns]
        if across is not None:
            result, labels = self._cross(frame, groups, across, make(columns[0]), periods=periods)
            self._note_empty(frame, columns)
            print(f'The {said} of "{columns[0]}" for each {_each(groups, periods)} (rows) and each "{across}" '
                  f"(columns): a new table with {len(result)} rows and {len(labels)} columns of numbers.")
            self._say_cross(result, groups, labels, periods=periods)
            _period_note(periods)
            return result
        if groups is None:
            result = frame.select(*exprs)
            self._note_empty(frame, columns)
            counts = frame.select(nw.col(c).count().alias(c) for c in columns).row(0)
            sentence = "; ".join(f'the {said} of "{c}" is {_say_number(v)} ({n} values)'
                                 for c, v, n in zip(columns, result.row(0), counts))
            print(sentence[0].upper() + sentence[1:] + ".")
            if isinstance(column, str):
                return _plain(result.item())
            return self._new_table(result)
        result = self._grouped(frame, groups, exprs)
        self._note_empty(frame, columns)
        each = _each(groups, periods)
        if len(columns) == 1:
            values = _say_groups(result._frame, groups, f"{start} of {columns[0]}", periods)
            print(f'The {said} of "{columns[0]}" for each {each}: {values}.')
        else:
            print(f"The {said} of {_names(columns)} for each {each}: a new table with {len(result)} rows.")
        _period_note(periods)
        return result

    # ------------------------------------------------------------------ number words
    @friendly
    def average_of(self, column, for_each=None, every=None, across=None):
        """The average (mean) of a column; empty cells are skipped.

        table.average_of("marks", for_each="class")   (with sdata.example("marks"))
        across= gives a cross table, one column per subject:
        table.average_of("marks", for_each="class", across="subject")
        every= groups a date column in for_each= into "hour", "day", "week", "month",
        "quarter" or "year" periods:
        table.make_dates("year")
        table.average_of("population", for_each="year", every="year")   (with sdata.example("growth"))
        """
        return self._summary("average_of", column, for_each, every, across)

    @friendly
    def median_of(self, column, for_each=None, every=None, across=None):
        """The middle number of a column once it is sorted; empty cells are skipped.

        table.median_of("marks", for_each="class")   (with sdata.example("marks"))
        across= and every= work as in average_of.
        """
        return self._summary("median_of", column, for_each, every, across)

    @friendly
    def total_of(self, column, for_each=None, every=None, across=None):
        """All the numbers in a column added up; empty cells are skipped.

        table.total_of("count", for_each="shop")   (with sdata.example("fruit"))
        table.total_of("count", for_each="shop", across="fruit")   a cross table
        every="month" with a date column in for_each= gives the total of each month.
        """
        return self._summary("total_of", column, for_each, every, across)

    @friendly
    def smallest_of(self, column, for_each=None, every=None, across=None):
        """The smallest number in a column; empty cells are skipped.

        table.smallest_of("temperature", for_each="city")   (with sdata.example("weather"))
        across= and every= work as in average_of.
        """
        return self._summary("smallest_of", column, for_each, every, across)

    @friendly
    def biggest_of(self, column, for_each=None, every=None, across=None):
        """The biggest number in a column; empty cells are skipped.

        table.biggest_of("temperature", for_each="city")   (with sdata.example("weather"))
        across= and every= work as in average_of.
        """
        return self._summary("biggest_of", column, for_each, every, across)

    @friendly
    def standard_deviation_of(self, column, for_each=None, every=None, across=None):
        """How spread out the numbers are: the sample standard deviation (divide by n - 1).

        It is the same as pandas .std() and R sd(). Empty cells are skipped.
        table.standard_deviation_of("marks", for_each="class")   (with sdata.example("marks"))
        across= and every= work as in average_of.
        """
        return self._summary("standard_deviation_of", column, for_each, every, across)

    @friendly
    def percentile_of(self, column=None, percent=None, for_each=None, every=None, across=None):
        """The value below which a given percent of the numbers lie; empty cells are skipped.

        table.percentile_of("marks", 90, for_each="class")   (with sdata.example("marks"))
        percentile_of("marks", 50) is the median, 25 and 75 the quartiles. Between two
        numbers it uses the usual linear rule (R quantile(type = 7), NumPy and pandas
        default). across= and every= work as in average_of.
        """
        step = "percentile_of"
        if column is None:
            raise SimplydataError(f"{step}() needs the column first, then the percent.\n"
                                  f'For example:  table.{step}("marks", 90, for_each="class")   (with sdata.example("marks"))')
        number = _check_percent(step, percent)
        share = number / 100
        measure = (f"percentile {number:g}", lambda c: nw.col(c).quantile(share, interpolation="linear"),
                   f"{_ordinal(number)} percentile")
        return self._summary(step, column, for_each, every, across, measure=measure)

    @friendly
    def most_common_of(self, column=None, for_each=None, every=None):
        """The value that turns up most often (the mode), for numbers or text; empty cells are skipped.

        table.most_common_of("days_absent", for_each="school")   (with sdata.example("school"))
        When values are tied, the smallest (first in order) is given and a note names the
        others. Without for_each= you get the value; with it a new table with, for each
        group, the most common value, how many rows have it, its percent of the group's
        filled cells, and how many values share the top count.
        """
        step = "most_common_of"
        example = 'table.most_common_of("days_absent", for_each="school")   (with sdata.example("school"))'
        if not isinstance(column, str):
            raise SimplydataError(f"{step}() needs one column name in quotes.\nFor example:  {example}")
        self._need_columns(column)
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else []
        frame = self._frame
        empty = self._empty(column)
        n_empty = int(frame.select(empty.cast(nw.Int64).sum()).item())
        frame, periods = self._periods(step, frame, groups or None, every)
        kept = frame.filter(~_empty_in(frame, column)) if n_empty else frame
        if n_empty:
            print(f'Note: {n_empty} empty cell{"" if n_empty == 1 else "s"} in "{column}" '
                  f'{"was" if n_empty == 1 else "were"} skipped.')
        values = [_plain(_as_datetime(v)) for v in kept[column].to_list()]
        keys = [_key(r) for r in kept.select(groups).rows()] if groups else [()] * len(values)
        by_group: dict = {}
        for key, v in zip(keys, values):
            counts = by_group.setdefault(key, {})
            counts[v] = counts.get(v, 0) + 1
        if not by_group:
            raise SimplydataError(f'"{column}" has no values to count: every cell is empty.')
        found = {}
        for key, counts in by_group.items():
            top = max(counts.values())
            tied = sorted((v for v, c in counts.items() if c == top), key=_order)
            found[key] = (tied, top, sum(counts.values()))
        order = sorted(found, key=lambda k: tuple(_order(v) for v in k))
        ties = []
        for key in order:
            tied, top, total = found[key]
            if len(tied) > 1:
                where = " / ".join(_say_label(v, periods.get(g)) for g, v in zip(groups, key)) or f'"{column}"'
                ties.append(f"{where} ({', '.join(_say_label(t) for t in tied[:6])}{' ...' if len(tied) > 6 else ''})")
        if not groups:
            tied, top, total = found[()]
            share = top * 100.0 / total
            if len(tied) > 1:
                listed = ", ".join(f'"{_say_label(t)}"' for t in tied[:6])
                print(f'The most common values of "{column}" are {listed}: each turns up {top} times of {total} '
                      f"({_say_percent(share)}). The first, {_say_label(tied[0])!r}, is given back.")
            else:
                print(f'The most common value of "{column}" is "{_say_label(tied[0])}": {top} of {total} filled cells '
                      f"({_say_percent(share)}).")
            return tied[0]
        name = f"most common {column}"
        data = {g: [k[i] for k in order] for i, g in enumerate(groups)}
        if name in data or any(c in data for c in ("count", "percent", "values with that count")):
            raise SimplydataError(f'A for_each= column has the name of a result column ("{name}", "count" or '
                                  '"percent"). Rename it first.')
        data[name] = [found[k][0][0] for k in order]
        data["count"] = [found[k][1] for k in order]
        data["percent"] = [found[k][1] * 100.0 / found[k][2] for k in order]
        data["values with that count"] = [len(found[k][0]) for k in order]
        result = self._new_table(nw.from_dict(data, backend=self._frame.implementation))
        result = self._new_table(result._frame.with_columns(nw.col("count").cast(nw.Int64),
                                                             nw.col("values with that count").cast(nw.Int64)))
        shown = _say_groups(result._frame, groups, name, periods, say=_say_label)
        print(f'The most common "{column}" for each {_each(groups, periods)}: {shown}.')
        if ties:
            print(f"Note: more than one value is the most common in {', '.join(ties[:GROUPS_IN_SENTENCE])}"
                  f"{' ...' if len(ties) > GROUPS_IN_SENTENCE else ''}; the table shows the first of them.")
        _period_note(periods)
        return result

    # ------------------------------------------------------------------ counting words
    @friendly
    def count_rows(self, for_each=None, across=None, every=None, in_percent=False):
        """How many rows the table has, or how many in each group.

        table.count_rows(for_each="class")   (with sdata.example("marks"))
        across= gives a cross table (pandas crosstab), one column per value:
        table.count_rows(for_each="shop", across="fruit")   (with sdata.example("fruit"))
        in_percent=True gives percents (0 to 100) instead: of all rows, or with across= of
        each row of the cross table, so every row adds up to 100 %.
        every="month" with a date column in for_each= counts the rows of each month.
        """
        step = "count_rows"
        if not isinstance(in_percent, bool):
            raise SimplydataError("in_percent= should be True or False.")
        if for_each is None:
            if across is not None:
                self._check_across(step, across, None)
            if every is not None:
                self._periods(step, self._frame, None, every)
            if in_percent:
                raise SimplydataError(
                    "in_percent=True needs groups to share out the rows.\n"
                    'For example:  table.count_rows(for_each="class", in_percent=True)   (with sdata.example("marks"))'
                )
            n = len(self._frame)
            print(f"The table has {n} row{'' if n == 1 else 's'}.")
            return n
        groups = self._group_columns(step, for_each)
        if across is not None:
            self._check_across(step, across, groups)
        frame, periods = self._periods(step, self._frame, groups, every)
        each = _each(groups, periods)
        if across is not None:
            result, labels = self._cross(frame, groups, across, nw.len(), fill=0, periods=periods)
            say = _say_number
            if in_percent:
                shares = result._frame.with_columns(nw.sum_horizontal(*labels).alias("__total")).with_columns(
                    *(nw.col(c).cast(nw.Float64) * 100.0 / nw.col("__total") for c in labels)).drop("__total")
                result = self._new_table(shares)
                say = _say_percent
                print(f'Percent of rows in each "{across}" (columns) for each {each} (rows); each row adds up to 100 %:')
            else:
                result = self._new_table(result._frame.with_columns(nw.col(c).cast(nw.Int64) for c in labels))
                print(f'Rows for each {each} (rows) and each "{across}" (columns):')
            self._say_cross(result, groups, labels, say=say, periods=periods)
            _period_note(periods)
            return result
        summary = self._grouped(frame, groups, [nw.len().alias("count")])
        result = self._new_table(summary._frame.with_columns(nw.col("count").cast(nw.Int64)))
        if in_percent:
            total = max(len(self._frame), 1)
            result = self._new_table(result._frame.with_columns((nw.col("count") * 100.0 / total).alias("percent")))
            print(f"Percent of all {len(self._frame)} rows in each {each}: "
                  f"{_say_groups(result._frame, groups, 'percent', periods, _say_percent)}.")
        else:
            print(f"Rows for each {each}: {_say_groups(result._frame, groups, 'count', periods)}.")
        _period_note(periods)
        return result

    @friendly
    def count_each(self, column):
        """Each different value in a column, how many times it appears, and its percent.

        Biggest count first. Empty cells count as a value of their own, so the percents
        are out of all rows. Percents are rounded to 1 decimal place.
        table.count_each("fruit")   (with sdata.example("fruit"))
        """
        if not isinstance(column, str):
            raise SimplydataError(
                "count_each() needs one column name in quotes.\n"
                'For example, with sdata.example("fruit"): table.count_each("fruit")'
            )
        self._need_columns(column)
        if column in ("count", "percent"):
            raise SimplydataError(
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
        kinds = len(result)
        print(f'How many rows have each "{column}" ({kinds} different value{"" if kinds == 1 else "s"}): '
              f"{_say_groups(result, [column], 'count')}.")
        return self._new_table(nw.maybe_reset_index(result))

    @friendly
    def count_different_values(self, column=None, for_each=None, every=None):
        """How many different values a column has, or how many in each group.

        table.count_different_values("name", for_each="class")   (with sdata.example("marks"))
        Empty cells are not counted (as pandas nunique()); a note says how many there were.
        Capital letters matter: "Apple" and "apple" are two values.
        """
        step = "count_different_values"
        if not isinstance(column, str):
            raise SimplydataError(
                f"{step}() needs one column name in quotes.\n"
                f'For example:  table.{step}("name", for_each="class")   (with sdata.example("marks"))'
            )
        self._need_columns(column)
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else None
        empty = self._empty(column)
        frame = self._frame.with_columns(nw.when(~empty).then(nw.col(column)).alias("__value"),
                                         empty.cast(nw.Int64).alias("__empty"))
        frame, periods = self._periods(step, frame, groups, every)
        n_empty = int(frame.select(nw.col("__empty").sum()).item())
        found = [nw.col("__value").n_unique().alias("__unique"), nw.col("__empty").sum().alias("__empties")]
        if n_empty:
            print(f'Note: {n_empty} empty cell{"" if n_empty == 1 else "s"} in "{column}" '
                  f'{"was" if n_empty == 1 else "were"} not counted.')
        if groups is None:
            unique, empties = frame.select(*found).row(0)
            n = int(unique) - (1 if empties else 0)  # n_unique counts the empty value once
            print(f'"{column}" has {n} different value{"" if n == 1 else "s"}.')
            return n
        summary = self._grouped(frame, groups, found)._frame
        summary = summary.with_columns(
            (nw.col("__unique") - nw.when(nw.col("__empties") > 0).then(1).otherwise(0)).cast(nw.Int64)
            .alias("different values")).select(*groups, "different values")
        result = self._new_table(summary)
        print(f'Different values of "{column}" for each {_each(groups, periods)}: '
              f"{_say_groups(summary, groups, 'different values', periods)}.")
        _period_note(periods)
        return result

    # ------------------------------------------------------------------ percents
    @friendly
    def percent_of(self, column=None, *extra, for_each=None, every=None, above=_NOT_GIVEN, below=_NOT_GIVEN,
                   at_least=_NOT_GIVEN, at_most=_NOT_GIVEN, equal_to=_NOT_GIVEN, not_equal_to=_NOT_GIVEN,
                   between=_NOT_GIVEN, one_of=_NOT_GIVEN, contains=_NOT_GIVEN, starts_with=_NOT_GIVEN,
                   ends_with=_NOT_GIVEN, is_empty=_NOT_GIVEN):
        """What percent of the rows meet a condition, or each group's share.

        table.percent_of("marks", above=60, for_each="class")   (with sdata.example("marks"))
        With a condition (the same words as keep_rows: above, below, at_least, at_most,
        equal_to, not_equal_to, between, one_of, contains, starts_with, ends_with, is_empty):
        the percent of rows that meet it, out of the rows with a value in the column, with
        its 95 % confidence interval (Clopper-Pearson exact interval, as R binom.test).
        Without a condition:
        table.percent_of(for_each="class")   each group's percent of all rows
        table.percent_of("count", for_each="shop")   each group's share of the total of a
        column   (with sdata.example("fruit"))
        Percents go from 0 to 100. Without for_each= you get one number; with it a new table.
        """
        step = "percent_of"
        options = dict(above=above, below=below, at_least=at_least, at_most=at_most, equal_to=equal_to,
                       not_equal_to=not_equal_to, between=between, one_of=one_of, contains=contains,
                       starts_with=starts_with, ends_with=ends_with, is_empty=is_empty)
        given = {k: v for k, v in options.items() if v is not _NOT_GIVEN}
        example = 'table.percent_of("marks", above=60, for_each="class")   (with sdata.example("marks"))'
        if extra:
            raise SimplydataError(
                f"{step}() needs the column and then one plain word saying which rows, like above=60 or "
                f'equal_to="A".\nFor example:  {example}'
            )
        if column is None:
            if given:
                raise SimplydataError(f"{step}() needs the column name first.\nFor example:  {example}")
            if for_each is None:
                raise SimplydataError(
                    f"{step}() needs a column with a condition, or groups to share out the rows.\n"
                    f"For example:  {example}\n"
                    'or:           table.percent_of(for_each="class")'
                )
            return self._percent_of_rows(step, for_each, every)
        if not isinstance(column, str):
            raise SimplydataError(f"{step}() needs one column name in quotes.\nFor example:  {example}")
        self._need_columns(column)
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else None
        if not given:
            return self._percent_of_total(step, column, groups, every)
        if len(given) > 1:
            names = " and ".join(f"{c}=" for c in given)
            raise SimplydataError(
                f"{step}() takes one condition at a time, but you gave {names}. "
                "Use between=[low, high] for a range.\n"
                'For example:  table.percent_of("marks", between=[40, 80])   (with sdata.example("marks"))'
            )
        cond, words = self._condition(step, column, options, ())
        counted = nw.lit(True) if "is_empty" in given else ~self._empty(column)
        frame = self._frame.with_columns(cond.cast(nw.Int64).alias("__hit"), counted.cast(nw.Int64).alias("__counted"))
        frame, periods = self._periods(step, frame, groups, every)
        sums = [nw.col("__hit").sum().alias("matching rows"), nw.col("__counted").sum().alias("rows")]
        left_out = len(frame) - int(frame.select(nw.col("__counted").sum()).item())
        if left_out:
            print(f'Note: {left_out} row{"" if left_out == 1 else "s"} with an empty cell in "{column}" '
                  f'{"was" if left_out == 1 else "were"} left out of the percent.')
        if groups is None:
            hits, rows = (int(v) for v in frame.select(*sums).row(0))
            share, low, high = _share(hits, rows)
            print(f'"{column}" {words} in {_say_percent(share)} of the rows ({hits} of {rows}; 95 % CI '
                  f"{_say_percent(low)} to {_say_percent(high)}, exact Clopper-Pearson interval).")
            return share
        summary = self._grouped(frame, groups, sums)._frame
        result = self._with_shares(summary, "matching rows", "rows")
        items = []
        for row in result._frame.select(*groups, "matching rows", "rows", "percent").rows():
            name = " / ".join(_say_label(v, periods.get(g)) for g, v in zip(groups, row))
            items.append(f"{name} {_say_percent(row[-1])} ({row[-3]} of {row[-2]})")
        text = ", ".join(items[:GROUPS_IN_SENTENCE])
        if len(items) > GROUPS_IN_SENTENCE:
            text += f" ... and {len(items) - GROUPS_IN_SENTENCE} more (the new table has them all)"
        print(f'Percent of rows where "{column}" {words}, for each {_each(groups, periods)}: {text}. '
              "(The new table has exact 95 % confidence intervals.)")
        _period_note(periods)
        return result

    def _with_shares(self, summary, hits: str, rows: str):
        """Add percent and its Clopper-Pearson 95 % interval to a small table of counts."""
        summary = nw.maybe_reset_index(summary.with_columns(nw.col(hits).cast(nw.Int64), nw.col(rows).cast(nw.Int64)))
        found = [_share(int(h), int(n)) for h, n in summary.select(hits, rows).rows()]
        backend = summary.implementation
        columns = [nw.new_series(name, [f[i] for f in found], nw.Float64, backend=backend)
                   for i, name in enumerate(("percent", "percent 95 % CI low", "percent 95 % CI high"))]
        return self._new_table(summary.with_columns(*columns))

    def _percent_of_rows(self, step, for_each, every):
        groups = self._group_columns(step, for_each)
        frame, periods = self._periods(step, self._frame, groups, every)
        summary = self._grouped(frame, groups, [nw.len().alias("count")])._frame
        summary = summary.with_columns(nw.lit(len(frame)).alias("__all"))
        result = self._with_shares(summary, "count", "__all")
        result = self._new_table(result._frame.drop("__all"))
        print(f"Percent of all {len(frame)} rows in each {_each(groups, periods)}: "
              f"{_say_groups(result._frame, groups, 'percent', periods, _say_percent)}. "
              "(The new table has exact 95 % confidence intervals.)")
        _period_note(periods)
        return result

    def _percent_of_total(self, step, column, groups, every):
        if groups is None:
            raise SimplydataError(
                f'{step}("{column}") needs for_each= to share out the total of "{column}", or a condition '
                "such as above=60 to count rows.\n"
                'For example:  table.percent_of("count", for_each="shop")   (with sdata.example("fruit"))'
            )
        frame = self._number_frame(step, [column])
        frame, periods = self._periods(step, frame, groups, every)
        grand = _plain(frame.select(nw.col(column).sum()).item())
        if not grand:
            raise SimplydataError(f'The total of "{column}" is 0, so there is no total to share out.')
        if frame.select((nw.col(column) < 0).any()).item():
            print(f'Note: "{column}" has numbers below 0, so a share can be below 0 % or above 100 %.')
        self._note_empty(frame, [column])
        total = f"total of {column}"
        summary = self._grouped(frame, groups, [nw.col(column).sum().alias(total)])._frame
        result = self._new_table(summary.with_columns((nw.col(total) * 100.0 / grand).alias("percent")))
        print(f'Share of the total of "{column}" ({_say_number(grand)}) for each {_each(groups, periods)}: '
              f"{_say_groups(result._frame, groups, 'percent', periods, _say_percent)}.")
        _period_note(periods)
        return result

    # ------------------------------------------------------------------ several measures at once
    @staticmethod
    def _measures(step, show):
        if show is None:
            return list(DEFAULT_MEASURES)
        items = show if isinstance(show, (list, tuple)) else [show]
        if not items:
            raise SimplydataError(
                f"show= needs at least one measure.\n"
                f'For example:  table.{step}("marks", show=["average", "standard deviation", "count"])'
                '   (with sdata.example("marks"))'
            )
        measures = []
        for item in items:
            if callable(item):
                measures.append(item)
                continue
            name = _measure_name(step, item)
            if name not in measures:
                measures.append(name)
        return measures

    @friendly
    def summary_of(self, column=None, show=None, for_each=None, every=None):
        """Several measures of one or more columns at once, in one tidy table.

        table.summary_of("marks", for_each="class", show=["average", "standard deviation", "count"])
        (with sdata.example("marks")). Several columns: table.summary_of(["marks", "hours"]).
        show= takes any of: "count" (numbers, empty cells left out), "average", "median",
        "total", "smallest", "biggest", "standard deviation" (sample, n - 1), "variance"
        (sample, n - 1), "standard error" (of the average), "confidence interval" (95 % t
        interval of the average, as R t.test), "range" (biggest minus smallest),
        "interquartile range" (upper quartile minus lower quartile, R type 7),
        "empty cells", and these, with the definitions R uses:
        "geometric mean" (numbers above 0), "skewness" and "kurtosis" (the bias-corrected
        sample values G1 and G2, as SciPy skew/kurtosis(bias=False), R e1071 type 2 and
        Excel; kurtosis is excess kurtosis, 0 for a bell shape), "coefficient of variation"
        (standard deviation / average, a ratio), "median absolute deviation" (as R mad(),
        times 1.4826), "trimmed average" (10 % cut from each end, as R mean(trim = 0.1);
        "trimmed average 20" cuts 20 %), "most common" (the smallest if tied) and
        "percentile 90" (any percent, R quantile type 7).
        Without show=: count, average, standard deviation, smallest, median,
        biggest. A measure of your own also works: a function that gets one group's
        numbers (a NumPy array) and gives back one number, show=["average", my_measure].
        The new table has one row for each column (and each group), so it can be saved
        or charted. every= groups a date column in for_each= by "week", "month", "year" ...
        """
        step = "summary_of"
        if column is None:
            raise SimplydataError(
                f"{step}() needs the column to summarise.\n"
                f'For example:  table.{step}("marks", for_each="class", show=["average", "count"])'
                '   (with sdata.example("marks"))'
            )
        columns = _as_list(step, "the column", column)
        measures = self._measures(step, show)
        groups = self._group_columns(step, for_each, columns) if for_each is not None else []
        if "column" in groups:
            raise SimplydataError(
                'summary_of() makes a column called "column", so a for_each= column cannot have that name.\n'
                'Rename it first, for example table.rename_column("column", to="group").'
            )
        frame = self._number_frame(step, columns)
        frame, periods = self._periods(step, frame, groups or None, every)
        frame = frame.with_columns(nw.col(c).is_null().cast(nw.Int64).alias(f"__empty {i}")
                                   for i, c in enumerate(columns))
        names = [m for m in measures if isinstance(m, str) and not _is_array_measure(m)]
        arrays = [m for m in measures if _is_array_measure(m)]
        own = [m for m in measures if not isinstance(m, str)]
        problems: dict = {}  # (measure, why) -> [where]
        data = {"column": [], **{g: [] for g in groups}}
        out_names, own_names = [], []
        for m in measures:
            if m == "confidence interval":
                out_names += ["average 95 % CI low", "average 95 % CI high"]
            elif isinstance(m, str):
                out_names.append(m)
            else:
                label = getattr(m, "__name__", "") or ""
                label = "own measure" if not label or label.startswith("<") else label
                while label in out_names or label in data:
                    label += " (own)"
                own_names.append(label)
                out_names.append(label)
        for name in out_names:
            if name in data:
                raise SimplydataError(f'A for_each= column is called "{name}", like a measure. Rename it first.')
            data[name] = []
        for i, c in enumerate(columns):
            col = nw.col(c)
            raw = {"n": col.count(), "mean": col.mean(), "median": col.median(), "sum": col.sum(), "min": col.min(),
                   "max": col.max(), "sd": col.std(ddof=1), "var": col.var(ddof=1),
                   "q1": col.quantile(0.25, interpolation="linear"), "q3": col.quantile(0.75, interpolation="linear"),
                   "empty": nw.col(f"__empty {i}").sum()}
            exprs = [e.alias(f"__{k}") for k, e in raw.items()]
            if groups:
                agg = frame.group_by(groups, drop_null_keys=False).agg(*exprs).sort(groups, nulls_last=True)
            else:
                agg = frame.select(*exprs)
            own_values = self._own_measures(frame, groups, c, own, own_names) if own else {}
            array_values = self._array_measures(frame, groups, c, arrays, problems, len(columns) > 1) if arrays else {}
            for row in agg.rows(named=True):
                key = _key([row[g] for g in groups])
                data["column"].append(c)
                for g, v in zip(groups, key):
                    data[g].append(v)
                values = _measure_values(names, {k: _plain(row[f"__{k}"]) for k in raw})
                for name in out_names:
                    if name in values:
                        data[name].append(values[name])
                    elif (key, name) in array_values:
                        data[name].append(array_values[(key, name)])
                    else:
                        data[name].append(own_values.get((key, name)))
        result = nw.from_dict(data, backend=self._frame.implementation)
        float_names = [n for n in out_names if _is_array_measure(n)]
        if float_names:
            result = result.with_columns(nw.col(n).cast(nw.Float64) for n in float_names)
        result = result.with_columns(nw.col(n).cast(nw.Int64) for n in ("count", "empty cells") if n in out_names)
        table = self._new_table(result)
        self._note_empty(frame, columns)
        self._say_summary(table, columns, groups, out_names, periods)
        for (name, why), places in problems.items():
            where = ", ".join(places[:GROUPS_IN_SENTENCE]) + (" ..." if len(places) > GROUPS_IN_SENTENCE else "")
            if why.startswith("tied:"):
                print(f"Note: more than one value is the most common in {where}; the table shows the smallest.")
            else:
                print(f"Note: the {name} {why}, so it is empty for {where}.")
        for name in out_names:
            if name.startswith("trimmed average "):
                share = float(name.split()[-2])
                place = " in each group" if groups else ""
                print(f"Note: {name} leaves out the lowest and highest {share:g} % of the values{place}, "
                      f"as R mean(x, trim = {share / 100:g}).")
        _period_note(periods)
        return table

    @staticmethod
    def _group_arrays(frame, groups, column) -> dict:
        """{group key: the group's numbers (empty cells left out)}, in the order the groups turn up."""
        values = frame.select(nw.col(column).cast(nw.Float64))[column].to_numpy()
        keys = [_key(r) for r in frame.select(groups).rows()] if groups else [()] * len(values)
        by_group: dict = {}
        for key, v in zip(keys, values):
            by_group.setdefault(key, [])
            if not math.isnan(v):
                by_group[key].append(v)
        return {key: np.asarray(numbers, dtype=float) for key, numbers in by_group.items()}

    def _array_measures(self, frame, groups, column, names, problems, several) -> dict:
        """{(group key, measure): value} for the measures worked out with NumPy / SciPy."""
        out = {}
        for key, array in self._group_arrays(frame, groups, column).items():
            label = [f'"{column}"'] if several or not groups else []
            label += [_say_label(v) for v in key]
            where = " / ".join(label) if groups else f'"{column}"'
            for name in names:
                value, why, tied = _array_measure(name, array)
                out[(key, name)] = value
                if why:
                    problems.setdefault((name, why), []).append(where)
                if tied is not None and len(tied) > 1:
                    problems.setdefault((name, "tied:"), []).append(
                        f"{where} ({', '.join(_say_number(t) for t in tied[:6])})")
        return out

    def _own_measures(self, frame, groups, column, functions, labels):
        """{(group key, label): value} for measures written by the user."""
        out = {}
        for key, array in self._group_arrays(frame, groups, column).items():
            for function, label in zip(functions, labels):
                where = f'"{column}"' + (f" in group {' / '.join(_say_label(v) for v in key)}" if groups else "")
                try:
                    value = function(array)
                    value = None if value is None else float(value)
                except Exception as error:  # the user's own code: say what went wrong in plain words
                    raise SimplydataError(
                        f'Your measure "{label}" did not work on {where}: {error}\n'
                        "A measure of your own gets the numbers of one group (a NumPy array, empty cells left out) "
                        "and gives back one number, for example:\n"
                        "    def spread(values):\n        return values.max() - values.min()\n"
                        '    table.summary_of("marks", show=["average", spread])   (with sdata.example("marks"))'
                    ) from None
                out[(key, label)] = value
        return out

    @staticmethod
    def _say_summary(table, columns, groups, out_names, periods):
        header = f"Summary of {_names(columns)}"
        if groups:
            header += f" for each {_each(groups, periods)}"
        print(header + ":")
        rows = table._frame.rows(named=True)
        for row in rows[:SUMMARY_LINES]:
            label = [f'"{row["column"]}"'] if len(columns) > 1 else []
            label += [_say_label(row[g], periods.get(g)) for g in groups]
            items = []
            for name in out_names:
                if name == "average 95 % CI high":
                    continue
                if name == "average 95 % CI low":
                    items.append(f"95 % CI {_say_number(row[name])} to {_say_number(row['average 95 % CI high'])}")
                else:
                    items.append(f"{name} {_say_number(row[name])}")
            start = "  " + " / ".join(label) + ": " if label else "  "
            print(start + ", ".join(items))
        if len(rows) > SUMMARY_LINES:
            print(f"  ... and {len(rows) - SUMMARY_LINES} more rows (the new table has them all)")

    # ------------------------------------------------------------------ one column on its own
    @friendly
    def column(self, name=None):
        """One column on its own: a column of numbers becomes sdata.numbers, any other a plain list.

        marks = table.column("marks")   (with sdata.example("marks"))
        then marks.average(), marks.chart() ... like any sdata.numbers. Empty cells become
        empty numbers. The table does not change.
        """
        if not isinstance(name, str):
            raise SimplydataError(
                "column() needs one column name in quotes.\n"
                'For example:  marks = table.column("marks")   (with sdata.example("marks"))'
            )
        self._need_columns(name)
        dtype = self._frame.schema[name]
        series = self._frame[name]
        if dtype.is_numeric():
            from ..numbers import numbers

            if series.null_count() or dtype.is_float():
                values = series.cast(nw.Float64).to_numpy()
            else:
                values = series.cast(nw.Int64).to_numpy()
            return numbers(values)
        kind = "text" if dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum else (
            "True/False values" if dtype == nw.Boolean else "dates" if dtype.is_temporal() else "values")
        print(f'Note: "{name}" holds {kind}, not numbers, so it comes back as a plain Python list.')
        return [None if _missing(v) else _plain(v) for v in series.to_list()]

    # ------------------------------------------------------------------ correlation
    @friendly
    def correlation_between(self, *columns, method="pearson", show="r"):
        """How strongly columns of numbers go up or down together, from -1 to 1.

        table.correlation_between("hours", "marks")   (with sdata.example("marks"))
        method="pearson" (a straight-line link, the usual one) or "spearman" (uses ranks,
        for any link that keeps going one way). Prints the strength in words and the
        p-value from SciPy. With two columns it gives back the number r; rows with an
        empty cell in either column are skipped.
        With three or more columns it gives back a correlation table (as pandas .corr()),
        and prints the strongest pairs:
        table.correlation_between("hours", "marks", "height")
        Each pair uses the rows where both columns have a number. show="p-value" gives the
        table of p-values instead, and show="each pair" one row per pair (r, p-value,
        rows), ready for add_adjusted_p_value_column(): testing many pairs makes a chance
        finding more likely.
        Strength words for the size of r: below 0.2 very weak, from 0.2 weak, from 0.4
        medium, from 0.6 strong, from 0.8 very strong (after Evans 1996).
        p below 0.001: "very unlikely to be chance"; below 0.05: "unlikely to be chance";
        otherwise "could easily be chance".
        A correlation does not show that one thing causes the other.
        """
        step = "correlation_between"
        names = []
        for value in columns:
            names.extend(value if isinstance(value, (list, tuple)) else [value])
        if len(names) < 2 or not all(isinstance(v, str) for v in names):
            raise SimplydataError(
                f"{step}() needs two column names in quotes (or more, for a correlation table).\n"
                f'For example, with sdata.example("marks"): table.{step}("hours", "marks")'
            )
        self._check_choice(step, "method", method, CORRELATION_METHODS)
        shown = _plain_name(show)
        if shown not in CORRELATION_SHOW:
            raise choice_error(step, "show", show, list(CORRELATION_SHOW))
        if len(set(names)) < len(names):
            if len(names) == 2:
                raise SimplydataError(
                    f"{step}() needs two different columns.\n"
                    f'For example, with sdata.example("marks"): table.{step}("hours", "marks")'
                )
            twice = next(n for n in names if names.count(n) > 1)
            raise SimplydataError(f'"{twice}" is listed twice. Give each column once.')
        if len(names) == 2 and shown != "r":
            raise SimplydataError(
                f'show="{show}" is for a correlation table of three or more columns. With two columns you get r, '
                "and the sentence gives its p-value."
            )
        frame = self._number_frame(step, names)
        if len(names) == 2:
            return self._one_correlation(step, frame, names[0], names[1], method)
        return self._correlation_table(step, frame, names, method, shown)

    @staticmethod
    def _correlate(step, frame, column_a, column_b, method):
        """r, p and the number of rows used, for one pair (rows with an empty cell skipped)."""
        pairs = frame.select(column_a, column_b).drop_nulls()
        if len(pairs) < 3:
            raise SimplydataError(
                f'{step}() needs at least 3 rows where both "{column_a}" and "{column_b}" have a number; '
                f"this table has {len(pairs)}."
            )
        a = pairs[column_a].to_numpy().astype(float)
        b = pairs[column_b].to_numpy().astype(float)
        for name, values in ((column_a, a), (column_b, b)):
            if values.min() == values.max():
                where = ""
                if frame[name].drop_nulls().n_unique() > 1:
                    where = f' (in the rows where "{column_a}" and "{column_b}" both have a number)'
                raise SimplydataError(
                    f'{step}() needs numbers that change, but every number in "{name}" is the same{where}, '
                    "so there is no correlation to find."
                )
        from scipy import stats

        result = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
        return float(result[0]), float(result[1]), len(pairs)

    def _one_correlation(self, step, frame, column_a, column_b, method):
        r, p, n = self._correlate(step, frame, column_a, column_b, method)
        skipped = len(frame) - n
        name = "Pearson" if method == "pearson" else "Spearman"
        if skipped:
            rows = "row" if skipped == 1 else "rows"
            verb = "was" if skipped == 1 else "were"
            print(f'Note: {skipped} {rows} with an empty cell in "{column_a}" or "{column_b}" {verb} skipped.')
        print(
            f'The correlation between "{column_a}" and "{column_b}" is r = {r:.2f} ({name}, {n} rows): '
            f"{_strength(r)}, and {_p_words(p)}."
        )
        return r

    def _correlation_table(self, step, frame, names, method, shown):
        found = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                found[(a, b)] = self._correlate(step, frame, a, b, method)
        method_name = "Pearson" if method == "pearson" else "Spearman"
        if shown == "each pair":
            pairs = list(found.items())
            data = {"column 1": [a for (a, _), _ in pairs], "column 2": [b for (_, b), _ in pairs],
                    "method": [method_name] * len(pairs), "r": [v[0] for _, v in pairs],
                    "p-value": [v[1] for _, v in pairs], "rows": [v[2] for _, v in pairs],
                    "strength": [_strength(v[0]) for _, v in pairs]}
        else:
            index = 0 if shown == "r" else 1
            data = {"column": list(names)}
            for b in names:
                data[b] = [(1.0 if index == 0 else None) if a == b else found.get((a, b), found.get((b, a)))[index]
                           for a in names]
        result = self._new_table(nw.from_dict(data, backend=self._frame.implementation))
        n_pairs = len(found)
        strongest = sorted(found.items(), key=lambda item: -abs(round(item[1][0], 2)))
        print(f"Correlations between {len(names)} columns ({method_name}, {n_pairs} pairs; each pair uses the rows "
              "where both columns have a number). The strongest:")
        for (a, b), (r, p, n) in strongest[:STRONGEST_PAIRS]:
            print(f'  "{a}" and "{b}": r = {r:.2f} ({n} rows), {_strength(r)}, and {_p_words(p)}.')
        if n_pairs > STRONGEST_PAIRS:
            print(f"  ... and {n_pairs - STRONGEST_PAIRS} more pairs (the new table has them all).")
        gaps = len(frame) - len(frame.select(*names).drop_nulls())
        if gaps:
            print(f"Note: {gaps} row{'' if gaps == 1 else 's'} {'has' if gaps == 1 else 'have'} an empty cell in at "
                  "least one of these columns; each pair skipped only the rows it could not use.")
        listed = ", ".join(f'"{n}"' for n in names)
        print(f"Note: these p-values are not adjusted for testing {n_pairs} pairs. To adjust them: "
              f'results = table.correlation_between({listed}, show="each pair"), then '
              'results.add_adjusted_p_value_column(from_column="p-value").')
        return result
