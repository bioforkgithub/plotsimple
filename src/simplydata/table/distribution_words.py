"""Table words for distributions: fit_distribution, compare_distributions, and the shape_test engine.

The numbers come from simplydata.distributions (maximum likelihood, as R MASS::fitdistr and
fitdistrplus); see that module for every choice.
"""

from __future__ import annotations

import math

import narwhals as nw
import numpy as np

from .. import distributions as _d
from .._errors import SimplydataError, friendly
from .summaries import GROUPS_IN_SENTENCE, _key, _order, _say_label

DEFAULT_COMPARED = ("normal", "lognormal", "gamma", "weibull")


def shape_test(values, name, seed=_d.SEED, n_mc_samples=9999, column=None) -> dict:
    """compare_shape_to(column, shape) for any shape but "normal": see distributions.shape_test."""
    return _d.shape_test(values, name, seed=seed, n_mc_samples=n_mc_samples, column=column)


class DistributionTableWords:
    # ------------------------------------------------------------------ helpers
    def _distribution_groups(self, step, column, for_each, example):
        """(group column names, [(key, label, values)]) with empty cells skipped (with a note)."""
        if not isinstance(column, str):
            raise SimplydataError(f"{step}() needs the column as a name in quotes.\nFor example, {example}")
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else []
        frame = self._number_frame(step, [column])
        values = frame.select(nw.col(column).cast(nw.Float64))[column].to_numpy().astype(float)
        keys = [_key(r) for r in frame.select(groups).rows()] if groups else [()] * len(values)
        by_group: dict = {}
        for key, v in zip(keys, values):
            by_group.setdefault(key, [])
            if not math.isnan(v):
                by_group[key].append(v)
        skipped = int(np.isnan(values).sum())
        if skipped:
            rows = "row" if skipped == 1 else "rows"
            verb = "was" if skipped == 1 else "were"
            print(f'Note: {skipped} {rows} with an empty cell in "{column}" {verb} skipped.')
        order = sorted(by_group, key=lambda k: tuple(_order(v) for v in k))
        out = [(key, " / ".join(_say_label(x) for x in key), np.asarray(by_group[key], dtype=float)) for key in order]
        return groups, out

    def _distribution_rows_table(self, rows):
        columns = list(rows[0])
        data = {c: [r.get(c) for r in rows] for c in columns}
        return self._new_table(nw.from_dict(data, backend=self._frame.implementation))

    # ------------------------------------------------------------------ fit_distribution
    @friendly
    def fit_distribution(self, column=None, distribution="normal", for_each=None, fixed=None):
        """Fit a distribution to a column by maximum likelihood; gives back the fitted distribution.

        fit = table.fit_distribution("sales", "lognormal")   (with sdata.example("shop"))
        then fit.chance(above=600), fit.percentile(90), fit.chart(), and fit.table() for the
        result table (values, standard errors, log-likelihood, AIC, BIC, rows). As R
        MASS::fitdistr(x, "gamma") (fitdistrplus::fitdist for beta and others); standard
        errors from the observed information. Families without a location in R (gamma,
        weibull, lognormal, exponential, beta) keep it at 0, as R. fixed={"shape": 2} holds a
        parameter. With for_each= it gives back a result table, one row per group.
        Empty cells are skipped.
        """
        step = "fit_distribution"
        example = 'with sdata.example("trial"): table.fit_distribution("days", "gamma")'
        groups, parts = self._distribution_groups(step, column, for_each, example)
        backend = self._frame.implementation
        if not groups:
            values = parts[0][2] if parts else np.asarray([])
            return _d.fit(values, distribution, fixed=fixed, say=True, column=column, backend=backend)
        rows, lines, fits = [], [], []
        for key, label, values in parts:
            dist = _d.fit(values, distribution, fixed=fixed, column=column, backend=backend)
            fits.append(dist)
            info = dist._fit_info
            row = {g: k for g, k in zip(groups, key)}
            row.update({"distribution": dist.name, "column": column, **info["row"]})
            rows.append(row)
            lines.append(f"  {label} ({info['rows']} values): {_d._fit_parts(dist)}; log-likelihood "
                         f"{info['log-likelihood']:.1f}, AIC {info['AIC']:.1f}")
        names = " and ".join(f'"{g}"' for g in groups)
        title = fits[0]._family.title if isinstance(fits[0]._family, _d.Family) else f"SciPy {fits[0].name}"
        article = "an" if title[0].lower() in "aeiou" else "a"
        print(f'Fitted {article} {title} distribution to "{column}" for each {names}:')
        for line in lines[:GROUPS_IN_SENTENCE]:
            print(line)
        if len(lines) > GROUPS_IN_SENTENCE:
            print(f"  ... and {len(lines) - GROUPS_IN_SENTENCE} more (the result table has them all)")
        print(f"Method: maximum likelihood, as {_d._r_fit_text(fits[0])}, for each group. For one group's "
              "distribution (to ask chances), keep its rows first and fit again.")
        if fits[0].name == "normal" and not fits[0]._fit_info["held"].get("standard_deviation"):
            print("Note: the maximum-likelihood standard deviation divides by n, so it is a little smaller than R sd().")
        return self._distribution_rows_table(rows)

    # ------------------------------------------------------------------ compare_distributions
    @friendly
    def compare_distributions(self, column=None, distributions=DEFAULT_COMPARED, for_each=None):
        """Fit several distributions to a column and rank them by AIC (smaller is better).

        table.compare_distributions("sales")   (with sdata.example("shop"))
        distributions=("normal", "lognormal", "gamma", "weibull") by default. Each is fitted by
        maximum likelihood (as fit_distribution); the result table has one row per
        distribution with its values, log-likelihood, AIC, BIC, the AIC difference from the
        best, and the Anderson-Darling, Kolmogorov-Smirnov and Cramer-von Mises statistics, as
        R fitdistrplus::gofstat(). An AIC difference below 2 is too small to choose between them.
        """
        step = "compare_distributions"
        example = 'with sdata.example("trial"): table.compare_distributions("days")'
        if isinstance(distributions, str):
            distributions = [distributions]
        groups, parts = self._distribution_groups(step, column, for_each, example)
        rows, sentences = [], []
        for key, label, values in parts:
            ranked, notes = _d.compare_fits(values, distributions, column=column)
            for note in notes:
                sentences.append(note if not groups else f"  {label}: {note}")
            sentence = _d._ranking_sentence(ranked)
            sentences.append(sentence if not groups else f"  {label} ({len(values)} values): {sentence}")
            for r in ranked:
                row = {g: k for g, k in zip(groups, key)}
                row.update({c: r[c] for c in _d.COMPARE_COLUMNS})
                rows.append(row)
        if groups:
            names = " and ".join(f'"{g}"' for g in groups)
            print(f'Distributions of "{column}" ranked by AIC for each {names} (smaller is better):')
            shown = [s for s in sentences]
            for line in shown[:GROUPS_IN_SENTENCE * 2]:
                print(line)
            if len(parts) > GROUPS_IN_SENTENCE:
                print("  ... (the result table has every group)")
        else:
            for line in sentences:
                print(line)
        print("A difference below 2 is too small to choose between them. Method: maximum likelihood for each, "
              "ranked by AIC, as R fitdistrplus::gofstat().")
        return self._distribution_rows_table(rows)
