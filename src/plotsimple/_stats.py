"""Statistics choices, written down in one place.

The numbers come from NumPy, pandas, SciPy and seaborn. plotsimple's job is to choose the
right calculation and say which one it used. Decisions:

* standard deviation uses the sample formula (divide by n - 1), like pandas and R
* empty cells are skipped
* error="ci" is a 95 % confidence interval of the mean from the t distribution
  (same as R's t.test). It gives the same answer every time, unlike bootstrapping.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._errors import PlotsimpleError, choice_error

SHOW = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "total": "sum",
    "sum": "sum",
    "count": "count",
}

ERRORS = ("sd", "se", "ci")


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


def check_show(step: str, show: str) -> str:
    if show not in SHOW:
        raise choice_error(step, "show", show, list(SHOW))
    return SHOW[show]


def check_error(step: str, error, show: str = "average"):
    """Return what seaborn's errorbar= option needs for this error choice."""
    if error is None:
        return None
    if error not in ERRORS:
        raise choice_error(step, "error", error, ERRORS)
    if SHOW.get(show) != "mean":
        raise PlotsimpleError(
            f'error="{error}" goes with show="average".\n'
            f'You asked for show="{show}". Remove error= or use show="average".'
        )
    if error == "ci":
        return t_interval
    return error  # seaborn's "sd" and "se" use n - 1, matching our rule
