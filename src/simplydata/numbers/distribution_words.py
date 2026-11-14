"""Numbers words for distributions: nums.fit_distribution("gamma")."""

from __future__ import annotations

import numpy as np

from .. import distributions as _d
from .._errors import friendly


class DistributionNumberWords:
    @friendly
    def fit_distribution(self, distribution="normal", fixed=None):
        """Fit a distribution to the numbers by maximum likelihood; gives back the fitted distribution.

        fit = nums.fit_distribution("gamma")   then fit.chance(above=10), fit.table()
        As R MASS::fitdistr(x, "gamma"); see table.fit_distribution. Empty values are skipped.
        """
        values = np.asarray(self._values, dtype=float).ravel()
        empty = int(np.isnan(values).sum())
        if empty:
            print(f"Note: {empty} empty value{'' if empty == 1 else 's'} {'was' if empty == 1 else 'were'} skipped.")
        return _d.fit(values[~np.isnan(values)], distribution, fixed=fixed, say=True)
