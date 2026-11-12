"""Statistics choices, written down in one place.

The numbers come from NumPy, pandas, SciPy and seaborn. simplydata's job is to choose the
right calculation and say which one it used. Decisions:

* standard deviation uses the sample formula (divide by n - 1), like pandas and R
* empty cells are skipped
* error="confidence interval" is a 95 % confidence interval of the mean from the t
  distribution (same as R's t.test). It gives the same answer every time, unlike bootstrapping.
* measure names are the same everywhere: show="biggest" on a chart is table.biggest_of,
  error="standard deviation" is table.standard_deviation_of.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._errors import SimplydataError, choice_error

# show= on bars and lines: the name people write -> the calculation pandas/seaborn does.
# The taught names match the table words (average_of, median_of ...); "mean" and "sum" also work.
SHOW = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "total": "sum",
    "sum": "sum",
    "count": "count",
    "smallest": "min",
    "biggest": "max",
    "standard deviation": "std",
}
SHOW_NAMES = ("average", "median", "total", "count", "smallest", "biggest", "standard deviation")

# error= and error_bars(): the long names are taught; "sd", "se" and "ci" are the usual
# abbreviations and keep working.
ERRORS = {
    "standard deviation": "sd",
    "standard error": "se",
    "confidence interval": "ci",
    "sd": "sd",
    "se": "se",
    "ci": "ci",
}
ERROR_NAMES = ("standard deviation", "standard error", "confidence interval")
ERROR_LONG = {"sd": "standard deviation", "se": "standard error", "ci": "confidence interval"}
ERROR_MEANING = {
    "sd": "the standard deviation of the values (how spread out they are around the average)",
    "se": "the standard error of the average (how sure we are about the average itself)",
    "ci": "the 95 % confidence interval of the average (where the true average very likely is)",
}


def _plain_name(value):
    """ "Standard_Deviation" -> "standard deviation"; anything that is not text stays as it is."""
    if isinstance(value, str):
        return " ".join(value.strip().lower().replace("_", " ").split())
    return value


def t_interval(values, level: float = 0.95):
    """95 % confidence interval of the mean, from the t distribution."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    m = v.mean()
    if len(v) < 2:
        return (m, m)
    half = stats.t.ppf((1 + level) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (m - half, m + half)


def check_show(step: str, show) -> str:
    """The show= name, written the usual way ("Biggest" -> "biggest"), or a friendly error."""
    name = _plain_name(show)
    if name not in SHOW:
        raise choice_error(step, "show", show, list(SHOW_NAMES))
    return name


def error_code(step: str, error, option: str = "error"):
    """"standard error" (or "se") -> "se". None stays None."""
    if error is None:
        return None
    name = _plain_name(error)
    if name not in ERRORS:
        raise choice_error(step, option, error, list(ERROR_NAMES))
    return ERRORS[name]


def say_error(code: str, what: str = "Error bars") -> None:
    """One line that says which error was drawn, so the choice is never hidden."""
    print(f"{what}: {ERROR_MEANING[code]}.")


def check_error(step: str, error, show: str = "average"):
    """Return what seaborn's errorbar= option needs for this error choice."""
    code = error_code(step, error)
    if code is None:
        return None
    if SHOW.get(_plain_name(show)) != "mean":
        raise SimplydataError(
            f'error="{ERROR_LONG[code]}" goes with show="average".\n'
            f'You asked for show="{show}". Remove error= or use show="average".'
        )
    if code == "ci":
        return t_interval
    return code  # seaborn's "sd" and "se" use n - 1, matching our rule


def line_between_neighbours(values, x=None, groups=None):
    """Fill empty values (NaN) on the straight line between the nearest filled neighbours.

    Linear interpolation, as pandas Series.interpolate(limit_area="inside"): x gives the place
    of each value (row positions when None; numbers or dates turned into numbers when the
    rows follow a column such as a date, so the line is spaced by those values, as pandas
    method="index"). groups (one code per value) fills each group on its own. Empty values
    before the first or after the last filled value of a group have only one neighbour, so
    they stay empty, as do values whose x is empty. Filled values with the same x are
    averaged first. Gives a new float array.
    """
    values = np.asarray(values, dtype=float)
    out = values.copy()
    x = np.arange(len(values), dtype=float) if x is None else np.asarray(x, dtype=float)
    codes = np.zeros(len(values), dtype=int) if groups is None else np.asarray(groups)
    for code in np.unique(codes):
        where = np.flatnonzero(codes == code)
        xs, ys = x[where], values[where]
        known = ~np.isnan(ys) & ~np.isnan(xs)
        if not known.any():
            continue
        kx, inverse = np.unique(xs[known], return_inverse=True)
        ky = np.bincount(inverse, weights=ys[known]) / np.bincount(inverse)
        wanted = np.isnan(ys) & ~np.isnan(xs) & (xs >= kx[0]) & (xs <= kx[-1])
        out[where[wanted]] = np.interp(xs[wanted], kx, ky)
    return out
