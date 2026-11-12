"""Models that explain one column from several others. They never change the table
(except add_predicted_column, which adds one column and can be undone).

table.explain("marks", using=["hours", "breakfast"]) fits one model and prints, for each
column used, how big its effect is, with a 95 % confidence interval and a p-value in plain
words. It gives back a table with one row per term (the model-level values are printed).

Choices (the same defaults as R, so results can be checked there):

* using=: column names. A column of numbers gets a slope. A text (or True/False) column is a
  group column: each group is compared with the first group in sorted order, the baseline
  (dummy coding, as R treatment contrasts). "hours * breakfast" means both columns and their
  interaction, as in an R formula; "hours : breakfast" means the interaction only. The
  coding of a group column inside an interaction follows R's rule (full coding when the rest
  of the term is not in the model), so the columns are the same as R model.matrix().
* outcome="numbers": linear model by least squares (R lm()), t-based 95 % intervals,
  R squared, adjusted R squared and the overall F test (R summary.lm()).
* outcome="yes or no": logistic regression (R glm(family = binomial)), fitted by iteratively
  reweighted least squares. Odds ratios with 95 % Wald intervals (R confint.default()) and
  Wald z p-values (R summary.glm()). The model as a whole: likelihood-ratio test against the
  model with no columns.
* outcome="counts": Poisson regression (R glm(family = poisson)), rate ratios. When the
  residual deviance is more than 1.5 times its degrees of freedom a note says the counts
  vary more than Poisson allows and suggests outcome="counts (overdispersed)": quasi-Poisson
  (R glm(family = quasipoisson)), the same rate ratios with standard errors scaled by the
  Pearson dispersion, t-based p-values as R summary.glm(), Wald intervals as
  R confint.default(). simplydata never switches by itself.
* Group columns with 3 or more groups (and interactions with them) also get one test for the
  whole column, as R drop1(): an F test for least squares and quasi-Poisson
  (drop1(test = "F")), a likelihood-ratio chi-square test for logistic and Poisson models
  (drop1(test = "Chisq")). Like drop1(), a column that is also part of an interaction is not
  tested alone. (This equals car::Anova() type II when there are no interactions.)
* grouped_by="school": linear mixed model with a random intercept for each group, fitted by
  REML (R lme4::lmer(REML = TRUE)): statsmodels MixedLM finds the two variances, and the fixed
  effects and their standard errors follow by generalised least squares, as lme4 fixef() and
  vcov(). P-values are Wald z-tests and intervals Wald intervals
  (R confint(method = "Wald")); R lmerTest gives Satterthwaite
  t-tests instead, which give slightly larger p-values when there are few groups. Group
  columns with 3 or more groups get a Wald chi-square test. The share of the differences
  that lies between the groups is the intraclass correlation (ICC). Mixed models for yes/no
  or counts are not available yet.
* empty_cells="multiple imputation": empty cells in number columns are filled in 20 times by
  chained equations with predictive mean matching (statsmodels MICEData, 5 donors as R
  mice()), the model is fitted to each filled-in table, and the results are pooled with
  Rubin's rules, with the Barnard-Rubin degrees of freedom (as R mice::pool()). The filling
  is random: seed= fixes it.

Notes, never silent switches: residuals not bell-shaped (least squares), strongly linked
columns (variance inflation factor above 5), too few rows (or too few yes/no events) per
effect, complete separation (logistic), overdispersion (Poisson).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import warnings

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, column_error, friendly
from .stat_tests import _label, _list_words, _num, _shape_notes, _skipped_note, _whole
from .summaries import _p_words

OUTCOMES = ("numbers", "yes or no", "counts", "counts (overdispersed)")
EMPTY_CELLS = ("leave out", "multiple imputation")
IMPUTATIONS = 20  # filled-in tables for multiple imputation
PMM_DONORS = 5  # as R mice() predictive mean matching
BURN_IN, BETWEEN = 10, 3  # imputation cycles before the first filled-in table, and between tables
VIF_LIMIT = 5.0
ROWS_PER_EFFECT = 10
OVERDISPERSION = 1.5
Z95 = 1.959963984540054
YES_NO_PAIRS = (("yes", "no"), ("true", "false"), ("1", "0"), ("1.0", "0.0"), ("y", "n"))
EXAMPLE = 'with sdata.example("school"): table.explain("marks", using=["hours", "breakfast"])'
INTERCEPT = "(intercept)"
KINDS = {"numbers": "lm", "yes or no": "binomial", "counts": "poisson", "counts (overdispersed)": "quasipoisson"}


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _sort_key(value):
    return (str(type(value)), value)


def _quoted(name) -> str:
    return f'"{name}"'


def _r_name(name: str) -> str:
    plain = name.replace("_", "").replace(".", "").isalnum() and not name[0].isdigit()
    return name if plain else f"`{name}`"


def _quiet(func, *args):
    """Run func without printing (the filled-in tables of multiple imputation)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


# ---------------------------------------------------------------------------- reading using=
def _read_using(step: str, using, columns) -> tuple[list[tuple], list[str]]:
    """using= as a list of terms (tuples of column names, in R order) and the R formula pieces."""
    if isinstance(using, str):
        items = [using]
    elif isinstance(using, (list, tuple)) and using:
        items = list(using)
    else:
        raise SimplydataError(f"{step}() needs using= with the columns that explain it.\nFor example, {EXAMPLE}")
    terms, pieces = [], []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise SimplydataError(f"Each column in using= is a name in quotes, not {item!r}.\nFor example, {EXAMPLE}")
        text = item.strip()
        if text in columns:
            parts, kind = [text], "main"
        elif "*" in text:
            parts, kind = [p.strip().strip("`") for p in text.split("*")], "cross"
        elif ":" in text:
            parts, kind = [p.strip().strip("`") for p in text.split(":")], "only"
        else:
            parts, kind = [text.strip("`")], "main"
        for part in parts:
            if part not in columns:
                raise column_error(part, columns)
        if len(set(parts)) < len(parts):
            raise SimplydataError(f'"{item}" names the same column twice.')
        sign = " * " if kind == "cross" else ":"
        pieces.append(sign.join(_r_name(p) for p in parts))
        if kind == "cross":
            for size in range(1, len(parts) + 1):
                terms.extend(itertools.combinations(parts, size))
        else:
            terms.append(tuple(parts))
    unique, seen = [], set()
    for term in terms:
        if frozenset(term) not in seen:
            seen.add(frozenset(term))
            unique.append(term)
    unique.sort(key=len)  # R puts main effects first, then two-way, then three-way terms
    return unique, pieces


def _design(data: dict, kinds: dict, levels: dict, terms: list[tuple]):
    """The model matrix exactly as R model.matrix() with treatment contrasts.

    Gives back X, the column labels, which term owns each column (-1 for the intercept)
    and, per column, its parts (label, (group column, group) or None) for the sentences."""
    n = len(next(iter(data.values()))) if data else 0
    term_set = {frozenset(t) for t in terms}
    columns, labels, owner, parts_of = [np.ones(n)], [INTERCEPT], [-1], [[]]
    for index, term in enumerate(terms):
        pieces = []
        for name in term:
            if kinds[name] == "number":
                pieces.append([(name, data[name], None)])
                continue
            rest = frozenset(term) - {name}
            used = levels[name] if (rest and rest not in term_set) else levels[name][1:]
            values = data[name]
            pieces.append([(f"{name} ({_label(level)})", np.array([v == level for v in values], dtype=float),
                            (name, level)) for level in used])
        for combo in itertools.product(*reversed(pieces)):  # the first column varies fastest, as R
            combo = combo[::-1]
            vector = np.ones(n)
            for _, values, _ in combo:
                vector = vector * values
            columns.append(vector)
            labels.append(" : ".join(c[0] for c in combo))
            owner.append(index)
            parts_of.append([(c[0], c[2]) for c in combo])
    return np.column_stack(columns), labels, np.array(owner), parts_of


# ---------------------------------------------------------------------------- the fits
def _least_squares(X, y):
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    fitted = X @ beta
    rss = float(((y - fitted) ** 2).sum())
    r_inv = np.linalg.inv(r)
    return beta, r_inv @ r_inv.T, fitted, rss


def _family(kind):
    """link, inverse link, d mu / d eta, variance, start values, exactly as R binomial() and poisson()."""
    tiny = np.finfo(float).eps
    if kind == "binomial":
        def inverse(eta):  # R C_logit_linkinv: beyond |eta| = 30 the chance is taken as certain
            with np.errstate(over="ignore"):
                odds = np.where(eta < -30, tiny, np.where(eta > 30, 1 / tiny, np.exp(np.clip(eta, -30, 30))))
            return odds / (1 + odds)

        def slope(eta):  # R C_logit_mu_eta
            e = np.exp(np.clip(eta, -30, 30))
            return np.where(np.abs(eta) > 30, tiny, e / (1 + e) ** 2)

        return (lambda mu: np.log(mu / (1 - mu)), inverse, slope, lambda mu: mu * (1 - mu), lambda y: (y + 0.5) / 2)
    return (np.log, lambda eta: np.maximum(np.exp(eta), tiny), lambda eta: np.maximum(np.exp(eta), tiny),
            lambda mu: mu, lambda y: y + 0.1)


def _deviance(kind, y, mu) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "binomial":
            terms = (np.where(y > 0, y * np.log(y / mu), 0.0)
                     + np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0))
        else:
            terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2 * terms.sum())


def _log_likelihood(kind, y, mu) -> float:
    from scipy.special import gammaln

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "binomial":
            return float(np.where(y > 0, y * np.log(mu), 0.0).sum()
                         + np.where(y < 1, (1 - y) * np.log(1 - mu), 0.0).sum())
        return float((np.where(y > 0, y * np.log(mu), 0.0) - mu - gammaln(y + 1)).sum())


def _irls(X, y, kind):
    """Maximum likelihood by iteratively reweighted least squares, step for step as R glm.fit():
    the same start values, the same stopping rule (relative change in deviance below 1e-8, at most
    25 steps) and the standard errors from the weights of the last step, so the numbers agree with
    R to many digits."""
    link, inverse, slope, variance, start = _family(kind)
    mu = start(y)
    eta = link(mu)
    old = _deviance(kind, y, mu)
    beta = np.zeros(X.shape[1])
    converged = False
    for _ in range(25):
        d = slope(eta)
        z = eta + (y - mu) / d
        weights = d ** 2 / variance(mu)
        w = np.sqrt(weights)
        q, r = np.linalg.qr(X * w[:, None])
        beta = np.linalg.solve(r, q.T @ (z * w))
        eta = X @ beta
        mu = inverse(eta)
        deviance = _deviance(kind, y, mu)
        if abs(deviance - old) / (abs(deviance) + 0.1) < 1e-8:
            converged = True
            break
        old = deviance
    r_inv = np.linalg.inv(r)
    return beta, r_inv @ r_inv.T, mu, deviance, converged


def _gls(X, y, groups, between, within):
    """Fixed effects, their covariance and the fitted values (with each group's own level) for
    given variances, by generalised least squares. This is how lme4 vcov() and fitted() work."""
    k = X.shape[1]
    xvx, xvy = np.zeros((k, k)), np.zeros(k)
    blocks = []
    for name in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == name)
        n = len(rows)
        v = within * np.eye(n) + between * np.ones((n, n))
        xv = np.linalg.solve(v, X[rows]).T
        xvx += xv @ X[rows]
        xvy += xv @ y[rows]
        blocks.append((rows, n))
    cov = np.linalg.inv(xvx)
    beta = cov @ xvy
    fitted = X @ beta
    for rows, n in blocks:
        shrink = between * n / (within + between * n)
        fitted[rows] += shrink * float((y[rows] - fitted[rows]).mean())
    return beta, cov, fitted


# ---------------------------------------------------------------------------- the words
class ModelWords:
    # ------------------------------------------------------------------ gathering the data
    def _model_parts(self, step, column, using, outcome, grouped_by, empty_cells):
        """Check every input and gather the columns (empty cells kept, as masks)."""
        self._need_name(step, "the column to explain", column, EXAMPLE)
        self._need_columns(column)
        self._check_choice(step, "outcome", outcome, OUTCOMES)
        self._check_choice(step, "empty_cells", empty_cells, EMPTY_CELLS)
        terms, pieces = _read_using(step, using, self._columns)
        variables = list(dict.fromkeys(name for term in terms for name in term))
        if column in variables:
            raise SimplydataError(f'"{column}" cannot both be explained and be in using=.\nFor example, {EXAMPLE}')
        if grouped_by is not None:
            self._need_name(step, "grouped_by=", grouped_by, EXAMPLE)
            self._need_columns(grouped_by)
            if grouped_by == column or grouped_by in variables:
                raise SimplydataError(f'"{grouped_by}" cannot be the grouped_by= column and also be explained or '
                                      "be in using=.")
            if outcome != "numbers":
                raise SimplydataError(
                    f'grouped_by= with outcome="{outcome}" (a generalised mixed model, as R lme4::glmer()) is not '
                    'in simplydata yet. grouped_by= works with outcome="numbers".'
                )
            if empty_cells != "leave out":
                raise SimplydataError(
                    'empty_cells="multiple imputation" cannot be used with grouped_by= yet: filling in empty cells '
                    'would need to respect the groups. Use the default empty_cells="leave out".'
                )
        frame = self._frame
        names = [column] + variables + ([grouped_by] if grouped_by is not None else [])
        raw, missing, kinds = {}, {}, {}
        for name in names:
            dtype = frame.schema[name]
            if dtype.is_temporal():
                raise SimplydataError(f'"{name}" holds dates or times, which {step}() cannot use. Make a number '
                                      "from them first, for example with table.add_date_part_column().")
            if dtype.is_numeric():
                values = frame[name].cast(nw.Float64).to_numpy().astype(float)
                raw[name], missing[name], kinds[name] = values, np.isnan(values), "number"
            else:
                values = frame[name].to_list()
                gone = np.array([_missing(v) for v in values], dtype=bool)
                raw[name], missing[name], kinds[name] = np.array(values, dtype=object), gone, "group"
        if grouped_by is not None:
            kinds[grouped_by] = "group"
        return terms, pieces, variables, raw, missing, kinds

    @staticmethod
    def _outcome_values(column, outcome, values, kind):
        """The outcome as floats (yes = 1) and the words saying which value counts as yes."""
        if outcome != "yes or no":
            if kind != "number":
                hint = ' For a yes/no column use outcome="yes or no".' if len(set(values.tolist())) == 2 else ""
                raise SimplydataError(f'outcome="{outcome}" needs "{column}" to hold numbers; it holds text.{hint}')
            y = values.astype(float)
            if outcome != "numbers" and (np.any(y < 0) or np.any(y != np.round(y))):
                raise SimplydataError(f'outcome="{outcome}" needs whole numbers of 0 or more in "{column}" (counts).')
            if outcome == "numbers" and set(np.unique(y).tolist()) == {0.0, 1.0}:
                print(f'Note: "{column}" holds only 0 and 1; for a yes/no outcome use outcome="yes or no".')
            return y, None
        distinct = sorted(set(values.tolist()), key=_sort_key)
        if len(distinct) != 2:
            found = ("only " + _list_words(f'"{_label(v)}"' for v in distinct)) if len(distinct) < 2 else (
                f"{len(distinct)} different values")
            raise SimplydataError(f'outcome="yes or no" needs "{column}" to hold exactly 2 different values; it has '
                                  f"{found}.")
        lowered = {str(v).strip().lower(): v for v in distinct}
        yes = None
        for word, other in YES_NO_PAIRS:
            if set(lowered) == {word, other}:
                yes = lowered[word]
                why = f'counted as yes: "{column}" = "{_label(yes)}"'
        if yes is None:
            yes = distinct[1]
            why = (f'counted as yes: "{column}" = "{_label(yes)}" (the second value in sorted order, as R; '
                   f'"{_label(distinct[0])}" is no)')
        return np.array([v == yes for v in values], dtype=float), why

    # ------------------------------------------------------------------ fitting one model
    @staticmethod
    def _fit_one(kind, X, y, groups=None):
        if kind == "lm":
            beta, unscaled, fitted, rss = _least_squares(X, y)
            df = len(y) - X.shape[1]
            return {"beta": beta, "cov": unscaled * (rss / df), "fitted": fitted, "rss": rss, "df": df}
        if kind == "mixed":
            from statsmodels.regression.mixed_linear_model import MixedLM
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Powell first, then L-BFGS: reaches the REML optimum as closely as lme4 does
                result = MixedLM(y, X, groups=groups).fit(reml=True, method=["powell", "lbfgs"])
            trouble = any(issubclass(w.category, ConvergenceWarning) for w in caught) or not result.converged
            between, within = max(float(np.asarray(result.cov_re)[0, 0]), 0.0), float(result.scale)
            beta, cov, fitted = _gls(X, y, groups, between, within)
            return {"beta": beta, "cov": cov, "fitted": fitted, "between": between, "within": within,
                    "trouble": trouble}
        family = "binomial" if kind == "binomial" else "poisson"
        beta, cov, mu, deviance, converged = _irls(X, y, family)
        df = len(y) - X.shape[1]
        out = {"beta": beta, "cov": cov, "fitted": mu, "deviance": deviance, "df": df, "converged": converged,
               "loglik": _log_likelihood(family, y, mu)}
        if kind == "quasipoisson":
            out["dispersion"] = float(((y - mu) ** 2 / mu).sum() / df)
            out["cov"] = cov * out["dispersion"]
        return out

    def _fit_model(self, step, column, outcome, grouped_by, terms, variables, raw, kinds, keep):
        """Fit the model on the rows in keep; a dict with everything the printing needs."""
        n = int(keep.sum())
        if n == 0:
            raise SimplydataError(f"{step}() found no rows with a value in every column it needs.")
        y, yes_words = self._outcome_values(column, outcome, raw[column][keep], kinds[column])
        data = {name: raw[name][keep] for name in variables}
        levels = {}
        for name in variables:
            if kinds[name] == "group":
                found = sorted(set(data[name].tolist()), key=_sort_key)
                if len(found) < 2:
                    which = f' ("{_label(found[0])}")' if found else ""
                    raise SimplydataError(f'The group column "{name}" has only one group{which} in the rows used, so '
                                          "it cannot explain anything. Leave it out of using=.")
                levels[name] = found
        X, labels, owner, parts_of = _design(data, kinds, levels, terms)
        p = X.shape[1]
        if n <= p:
            raise SimplydataError(f"{step}() needs more rows ({n}) than values to estimate ({p}). "
                                  "Use fewer columns in using=, or groups with more rows.")
        if np.linalg.matrix_rank(X) < p:
            raise SimplydataError(self._rank_problem(X, labels))
        kind, groups = KINDS[outcome], None
        if grouped_by is not None:
            kind = "mixed"
            groups = np.array([_label(g) for g in raw[grouped_by][keep]], dtype=object)
            if len(set(groups.tolist())) < 2:
                raise SimplydataError(f'grouped_by="{grouped_by}" needs at least 2 groups.')
            self._need_statsmodels()
        fit = self._fit_one(kind, X, y, groups)
        return dict(kind=kind, X=X, y=y, labels=labels, owner=owner, parts_of=parts_of, terms=terms, levels=levels,
                    kinds=kinds, fit=fit, n=n, yes_words=yes_words, groups=groups, keep=keep, outcome=outcome,
                    variables=variables, column=column)

    @staticmethod
    def _need_statsmodels():
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            raise SimplydataError('This needs the statsmodels package: pip install "simplydata[stats]"') from None

    @staticmethod
    def _rank_problem(X, labels) -> str:
        """Name the columns that can be made exactly from the others."""
        bad = []
        for j in range(1, X.shape[1]):
            others = np.delete(X, j, axis=1)
            coef, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
            if np.allclose(others @ coef, X[:, j], atol=1e-9 * max(1.0, float(np.abs(X[:, j]).max()))):
                bad.append(labels[j])
        named = _list_words(_quoted(b) for b in bad) if bad else "some columns"
        return (f"The model cannot separate the effects of {named}: one can be made exactly from the others "
                "(for example the same thing measured twice, or a combination of groups with no rows). "
                "Leave one of them out of using=.")

    # ------------------------------------------------------------------ numbers per term
    @staticmethod
    def _coefficients(model):
        """Per model column: estimate, standard error, statistic, df, p and 95 % CI (log scale for ratios)."""
        from scipy import stats

        fit, kind = model["fit"], model["kind"]
        beta = fit["beta"]
        se = np.sqrt(np.diag(fit["cov"]))
        stat = beta / se
        if kind in ("lm", "quasipoisson"):
            df = float(fit["df"])
            p = 2 * stats.t.sf(np.abs(stat), df)
            q = float(stats.t.ppf(0.975, df)) if kind == "lm" else Z95  # quasi-Poisson: R confint.default()
            low, high = beta - q * se, beta + q * se
            name, dfs = "t", np.full(len(beta), df)
        else:
            p = 2 * stats.norm.sf(np.abs(stat))
            low, high = beta - Z95 * se, beta + Z95 * se
            name, dfs = "z", np.full(len(beta), math.nan)
        return dict(beta=beta, se=se, stat=stat, p=p, low=low, high=high, name=name, df=dfs)

    @staticmethod
    def _whole_column_tests(model):
        """One test per term with more than one column, as R drop1() (mixed models: Wald chi-square)."""
        from scipy import stats

        terms, owner, kind, fit = model["terms"], model["owner"], model["kind"], model["fit"]
        X, y = model["X"], model["y"]
        out = []
        for index, term in enumerate(terms):
            columns = np.flatnonzero(owner == index)
            if len(columns) < 2 or any(set(term) < set(other) for other in terms):
                continue  # one column (its own t or z test is the same), or part of an interaction, as R drop1()
            label = " : ".join(term)
            k = len(columns)
            if kind == "mixed":
                b = fit["beta"][columns]
                chi = float(b @ np.linalg.solve(fit["cov"][np.ix_(columns, columns)], b))
                out.append(dict(term=label, name="chi-square", stat=chi, df=float(k), p=float(stats.chi2.sf(chi, k)),
                                test="Wald chi-square test"))
                continue
            reduced = np.delete(X, columns, axis=1)
            if kind == "lm":
                rss0 = _least_squares(reduced, y)[3]
                f = ((rss0 - fit["rss"]) / k) / (fit["rss"] / fit["df"])
                out.append(dict(term=label, name="F", stat=f, df=float(k), df2=float(fit["df"]),
                                p=float(stats.f.sf(f, k, fit["df"])), test="F test"))
                continue
            family = "binomial" if kind == "binomial" else "poisson"
            change = _irls(reduced, y, family)[3] - fit["deviance"]
            if kind == "quasipoisson":
                f = (change / k) / (fit["deviance"] / fit["df"])  # as R drop1(test = "F")
                out.append(dict(term=label, name="F", stat=f, df=float(k), df2=float(fit["df"]),
                                p=float(stats.f.sf(f, k, fit["df"])), test="F test"))
            else:
                out.append(dict(term=label, name="chi-square", stat=change, df=float(k),
                                p=float(stats.chi2.sf(change, k)), test="likelihood-ratio test"))
        return out

    @staticmethod
    def _effect_name(outcome):
        return {"yes or no": "odds ratio", "counts": "rate ratio", "counts (overdispersed)": "rate ratio"}.get(
            outcome, "effect")

    def _term_table(self, model, coef, tests, extra=None):
        """The result table: one row per model column, then one per whole-column test."""
        effect = self._effect_name(model["outcome"])
        ratio = effect != "effect"
        rows = []
        for j, label in enumerate(model["labels"]):
            b, low, high = (float(coef["beta"][j]), float(coef["low"][j]), float(coef["high"][j]))
            row = {"term": label}
            if ratio:
                row.update({effect: math.exp(b), "95 % CI low": math.exp(low), "95 % CI high": math.exp(high),
                            f"log {effect}": b})
            else:
                row.update({effect: b, "95 % CI low": low, "95 % CI high": high})
            row.update({"standard error": float(coef["se"][j]), "statistic name": coef["name"],
                        "statistic": float(coef["stat"][j]), "degrees of freedom": float(coef["df"][j]),
                        "p-value": float(coef["p"][j])})
            if extra:
                row.update({key: float(values[j]) for key, values in extra.items()})
            rows.append(row)
        for test in tests:
            row = {key: math.nan for key in rows[0]}
            row.update({"term": f"{test['term']} (whole column)", "statistic name": test["name"],
                        "statistic": test["stat"], "degrees of freedom": test["df"], "p-value": test["p"]})
            rows.append(row)
        data = {key: [r[key] for r in rows] for key in rows[0]}
        return self._new_table(nw.from_dict(data, backend=self._frame.implementation))

    # ------------------------------------------------------------------ sentences
    @staticmethod
    def _term_sentence(model, j, beta, low, high, p) -> str:
        column, outcome = model["column"], model["outcome"]
        parts = model["parts_of"][j]
        words = _p_words(p)
        has_groups = any(model["kinds"][v] == "group" for v in model["variables"])
        at_base = " and every group column is at its baseline" if has_groups else ""
        if model["labels"][j] == INTERCEPT:
            if outcome == "numbers":
                return f'  intercept: {_num(beta)}, the expected "{column}" when every number column is 0{at_base}.'
            what = "odds of yes" if outcome == "yes or no" else "expected count"
            return f"  intercept: {_num(math.exp(beta))}, the {what} when every number column is 0{at_base}."
        if outcome == "numbers":
            interval = f"95 % CI {_num(low)} to {_num(high)}"
            if len(parts) == 1 and parts[0][1] is None:
                name = parts[0][0]
                more = "more" if beta >= 0 else "less"
                return (f'  {name}: each 1 more "{name}" goes with {_num(abs(beta))} {more} "{column}" '
                        f"({interval}); {words}.")
            if len(parts) == 1:
                var, level = parts[0][1]
                way = "higher" if beta >= 0 else "lower"
                return (f'  {var} "{_label(level)}" against "{_label(model["levels"][var][0])}": "{column}" is '
                        f"{_num(abs(beta))} {way} ({interval}); {words}.")
            return (f"  {model['labels'][j]} (interaction): {_num(beta)} ({interval}), how much the effect of one "
                    f"changes with the other; {words}.")
        thing = "the odds of yes" if outcome == "yes or no" else "the expected count"
        ratio = math.exp(beta)
        interval = f"95 % CI {_num(math.exp(low))} to {_num(math.exp(high))}"
        if len(parts) == 1 and parts[0][1] is None:
            name = parts[0][0]
            return f'  {name}: each 1 more "{name}" multiplies {thing} by {_num(ratio)} ({interval}); {words}.'
        if len(parts) == 1:
            var, level = parts[0][1]
            return (f'  {var} "{_label(level)}" against "{_label(model["levels"][var][0])}": {thing} is '
                    f"{_num(ratio)} times as high ({interval}); {words}.")
        return (f"  {model['labels'][j]} (interaction): {_num(ratio)} ({interval}), how many times the effect of one "
                f"is multiplied by the other; {words}.")

    def _print_terms(self, model, coef, rows_text):
        what = {"lm": "effect", "mixed": "effect", "binomial": "odds ratio", "poisson": "rate ratio",
                "quasipoisson": "rate ratio"}[model["kind"]]
        variables = _list_words(_quoted(v) for v in model["variables"])
        print(f'Explaining "{model["column"]}" from {variables} ({rows_text}). Each {what} below holds the other '
              "columns fixed:")
        for j in range(len(model["labels"])):
            print(self._term_sentence(model, j, coef["beta"][j], coef["low"][j], coef["high"][j], coef["p"][j]))

    def _report(self, model, pieces, grouped_by):
        """Print everything about one fitted model; gives back (coefficients, whole-column tests)."""
        from scipy import stats

        coef = self._coefficients(model)
        tests = self._whole_column_tests(model)
        fit, kind, n, column = model["fit"], model["kind"], model["n"], model["column"]
        formula = f"{_r_name(column)} ~ " + " + ".join(pieces)
        self._print_terms(model, coef, f"{n} rows")
        for test in tests:
            extra = f" and {_whole(test['df2'])}" if "df2" in test else ""
            print(f'  {test["term"]}, all groups together: {test["test"]} ({test["name"]} = {_num(test["stat"])}, '
                  f"df = {_whole(test['df'])}{extra}); {_p_words(test['p'])}.")
        if kind == "lm":
            print(f"Method: linear model by least squares, as R lm({formula}).")
        elif kind == "binomial":
            print(f"Method: logistic regression, as R glm({formula}, family = binomial); {model['yes_words']}. "
                  "Wald 95 % intervals and z-tests, as R confint.default() and summary().")
        elif kind == "poisson":
            print(f"Method: Poisson regression, as R glm({formula}, family = poisson). Wald 95 % intervals and "
                  "z-tests, as R confint.default() and summary().")
        elif kind == "quasipoisson":
            print(f"Method: quasi-Poisson regression, as R glm({formula}, family = quasipoisson): Poisson rate "
                  f"ratios with standard errors widened for counts that vary more than Poisson allows (dispersion "
                  f"{_num(fit['dispersion'])}); t-tests as R summary(), Wald 95 % intervals as R confint.default().")
        else:
            print(f'Method: linear mixed model with a random intercept for each "{grouped_by}", fitted by REML, as R '
                  f"lme4::lmer({formula} + (1 | {_r_name(grouped_by)})).")
        k = model["X"].shape[1] - 1
        if kind == "lm":
            y = model["y"]
            r2 = 1 - fit["rss"] / float(((y - y.mean()) ** 2).sum())
            adj = 1 - (1 - r2) * (n - 1) / fit["df"]
            line = (f"R squared = {r2:.3f} (adjusted {adj:.3f}): the model explains about {round(r2 * 100)} % of the "
                    f'differences in "{column}".')
            if k > 0:
                f = (r2 / k) / ((1 - r2) / fit["df"])
                p = float(stats.f.sf(f, k, fit["df"]))
                line += f" The model as a whole (F = {_num(f)}, df = {k} and {fit['df']}): {_p_words(p)}."
            print(line)
        elif kind != "mixed":
            family = "binomial" if kind == "binomial" else "poisson"
            y = model["y"]
            null = _deviance(family, y, np.full(len(y), y.mean()))
            line = (f"Residual deviance {_num(fit['deviance'])} on {fit['df']} df (with no columns: {_num(null)} on "
                    f"{n - 1} df).")
            if kind != "quasipoisson" and k > 0:
                change = null - fit["deviance"]
                p = float(stats.chi2.sf(change, k))
                aic = -2 * fit["loglik"] + 2 * (k + 1)
                line += (f" The model as a whole (likelihood-ratio chi-square = {_num(change)}, df = {k}): "
                         f"{_p_words(p)}. AIC = {_num(aic)}.")
            print(line)
        else:
            between, within = fit["between"], fit["within"]
            icc = between / (between + within)
            groups = len(set(model["groups"].tolist()))
            print(f'{groups} groups in "{grouped_by}". Variance between groups {_num(between)}, within groups '
                  f'{_num(within)}: about {round(icc * 100)} % of the differences in "{column}" that the columns do '
                  f"not explain lies between the groups (intraclass correlation, ICC = {icc:.3f}).")
            print("P-values: Wald z-tests, with Wald 95 % intervals (standard errors as R vcov(), intervals as R "
                  'confint(method = "Wald")); R lmerTest gives Satterthwaite t-tests instead, which give slightly '
                  "larger p-values when there are few groups.")
            if between <= 1e-6 * within:
                print(f'Note: the variance between groups is about 0 (R calls this a singular fit): the groups in '
                      f'"{grouped_by}" differ no more than chance, so the model is the same as without grouped_by=.')
            if groups < 5:
                print(f"Note: only {groups} groups; with fewer than 5 the variance between groups is poorly estimated.")
            if fit["trouble"]:
                print("Note: the fit did not settle fully (statsmodels warned about convergence); check the results "
                      "or use fewer columns.")
        for note in self._model_notes(model):
            print(note)
        self._closing(coef["p"][1:])
        return coef, tests

    @staticmethod
    def _closing(p_values):
        if any(p >= 0.05 for p in p_values):
            print("A large p-value does not show that there is no effect, only that the data cannot tell. "
                  "A link does not show that one thing causes the other.")
        else:
            print("A link does not show that one thing causes the other.")

    @staticmethod
    def _model_notes(model) -> list[str]:
        notes = []
        kind, fit, n = model["kind"], model["fit"], model["n"]
        X, labels = model["X"], model["labels"]
        effects = X.shape[1] - 1
        if kind == "lm":
            residuals = model["y"] - fit["fitted"]
            for note in _shape_notes([("this model", residuals)], what="the residuals (distances from the model)"):
                notes.append(note.replace('a safer test here: based_on="ranks".',
                                          "the p-values and confidence intervals may be off."))
        if kind == "binomial":
            events = int(min(model["y"].sum(), len(model["y"]) - model["y"].sum()))
            if effects and events / effects < ROWS_PER_EFFECT:
                notes.append(f"Note: only {events} rows in the smaller of yes and no for {effects} effects; with fewer "
                             f"than {ROWS_PER_EFFECT} per effect the results are unstable. Use fewer columns.")
            mu = fit["fitted"]
            if np.any(mu < 1e-8) or np.any(mu > 1 - 1e-8) or not fit["converged"]:
                notes.append("Note: some rows are predicted as certainly yes or certainly no (complete separation; R "
                             'warns "fitted probabilities numerically 0 or 1 occurred"), so the odds ratios and their '
                             "intervals cannot be trusted. Use fewer columns or merge groups.")
        elif effects and n / effects < ROWS_PER_EFFECT:
            notes.append(f"Note: only {n} rows for {effects} effects; with fewer than {ROWS_PER_EFFECT} rows per "
                         "effect the results are unstable. Use fewer columns.")
        if kind == "poisson" and fit["df"] > 0:
            ratio = fit["deviance"] / fit["df"]
            if ratio > OVERDISPERSION:
                notes.append(f"Note: the counts vary more than a Poisson model allows (residual deviance / df = "
                             f"{_num(ratio)}, above {OVERDISPERSION}), so the intervals are too narrow and the "
                             'p-values too small. Safer: outcome="counts (overdispersed)" (quasi-Poisson).')
        main = [j for j in range(1, X.shape[1]) if len(model["parts_of"][j]) == 1]
        if len(main) >= 2:
            with np.errstate(all="ignore"):
                corr = np.corrcoef(X[:, main], rowvar=False)
            if np.all(np.isfinite(corr)):
                try:
                    vif = np.diag(np.linalg.inv(corr))
                except np.linalg.LinAlgError:
                    vif = np.full(len(main), np.inf)
                high = [(labels[j], v) for j, v in zip(main, vif) if v > VIF_LIMIT]
                if high:
                    named = _list_words(f'"{name}" ({_num(v)})' for name, v in high)
                    notes.append(f"Note: some columns are strongly linked with the others (variance inflation factor "
                                 f"above {VIF_LIMIT:g}: {named}), so their separate effects are hard to tell apart and "
                                 "their intervals are wide. Consider leaving one out.")
        return notes

    # ------------------------------------------------------------------ explain
    @friendly
    def explain(self, column=None, using=None, outcome="numbers", grouped_by=None, empty_cells="leave out", seed=1):
        """Explain one column from several others with a model, and say how big each effect is.

        table.explain("marks", using=["hours", "breakfast"])   (with sdata.example("school"))
        outcome="numbers" (the default): linear model by least squares (R lm()).
        outcome="yes or no": logistic regression, odds ratios (R glm(family = binomial)).
        outcome="counts": Poisson regression, rate ratios (R glm(family = poisson));
        outcome="counts (overdispersed)": quasi-Poisson (R glm(family = quasipoisson)).
        A text column in using= is a group column: each group is compared with the first group
        in sorted order (the baseline). "hours * breakfast" adds both columns and their
        interaction (does the effect of hours depend on breakfast?).
        grouped_by="school": a mixed model with its own starting level for each school
        (R lme4::lmer()), for rows that belong together. empty_cells="multiple imputation"
        fills in empty cells 20 times (R mice()) instead of leaving those rows out; seed= fixes
        the random filling. Prints each effect with its 95 % confidence interval and p-value in
        words, and how well the model fits; gives back a table with one row per term.
        A link does not show that one thing causes the other.
        """
        step = "explain"
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SimplydataError("seed= is a whole number that fixes the random filling-in, for example seed=1.")
        terms, pieces, variables, raw, missing, kinds = self._model_parts(
            step, column, using, outcome, grouped_by, empty_cells)
        names = [column] + variables + ([grouped_by] if grouped_by is not None else [])
        if empty_cells == "multiple imputation":
            return self._explain_imputed(step, column, outcome, terms, pieces, variables, raw, missing, kinds, seed)
        keep = ~np.any([missing[nm] for nm in names], axis=0)
        _skipped_note(int((~keep).sum()), names)
        model = self._fit_model(step, column, outcome, grouped_by, terms, variables, raw, kinds, keep)
        coef, tests = self._report(model, pieces, grouped_by)
        return self._term_table(model, coef, tests)

    # ------------------------------------------------------------------ multiple imputation
    def _explain_imputed(self, step, column, outcome, terms, pieces, variables, raw, missing, kinds, seed):
        from scipy import stats

        self._need_statsmodels()
        import pandas as pd
        from statsmodels.imputation.mice import MICEData

        names = [column] + variables
        size = len(raw[column])
        text_cols = [nm for nm in names if kinds[nm] == "group" and nm != column]
        number_cols = [nm for nm in names if kinds[nm] == "number"]
        text_gap = np.any([missing[nm] for nm in text_cols], axis=0) if text_cols else np.zeros(size, dtype=bool)
        filled_cols = list(dict.fromkeys(number_cols + [column]))
        every_gap = np.all([missing[nm] for nm in filled_cols], axis=0)
        keep = ~text_gap & ~every_gap
        if text_gap.any():
            print(f"Note: {int(text_gap.sum())} rows with an empty cell in a text column "
                  f"({_list_words(_quoted(g) for g in text_cols)}) were left out: multiple imputation here fills in "
                  "number columns (and a yes/no outcome) only.")
        if every_gap.any():
            _skipped_note(int((every_gap & ~text_gap).sum()), filled_cols)
        gaps = np.any([missing[nm][keep] for nm in filled_cols], axis=0)
        rows_with_gaps = int(gaps.sum())
        if rows_with_gaps == 0:
            print("There are no empty cells to fill in, so this is the ordinary fit.")
            model = self._fit_model(step, column, outcome, None, terms, variables, raw, kinds, keep)
            coef, tests = self._report(model, pieces, None)
            return self._term_table(model, coef, tests)
        work = {nm: raw[nm][keep] for nm in names}
        yes_words = None
        if outcome == "yes or no":  # filled in as 0 and 1
            given = ~missing[column][keep]
            mapped, yes_words = self._outcome_values(column, outcome, work[column][given], kinds[column])
            yes_value = work[column][given][int(np.argmax(mapped))]
            outcome_numbers = np.array([math.nan if _missing(v) else float(v == yes_value) for v in work[column]])
        else:
            outcome_numbers = work[column].astype(float)
        frame, safe = {}, {}
        for i, nm in enumerate(names):
            key = f"v{i}"
            if nm == column or kinds[nm] == "number":
                frame[key] = outcome_numbers if nm == column else work[nm].astype(float)
                safe[nm] = key
            else:
                levels = sorted(set(work[nm].tolist()), key=_sort_key)
                for j, level in enumerate(levels[1:]):
                    frame[f"{key}_{j}"] = np.array([v == level for v in work[nm]], dtype=float)
        imputer = MICEData(pd.DataFrame(frame), k_pmm=PMM_DONORS, rng=seed)
        filled = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer.update_all(BURN_IN)
            for _ in range(IMPUTATIONS):
                imputer.update_all(BETWEEN)
                filled.append(imputer.data.copy())
        fit_kinds = dict(kinds)
        fit_kinds[column] = "number" if outcome != "yes or no" else "group"
        estimates, variances, model = [], [], None
        every_row = np.ones(len(filled[0]), dtype=bool)
        for data in filled:
            filled_raw = dict(work)
            for nm, key in safe.items():
                filled_raw[nm] = data[key].to_numpy(dtype=float)
            if outcome == "yes or no":
                filled_raw[column] = np.where(filled_raw[column] > 0.5, "yes", "no").astype(object)
            model = _quiet(self._fit_model, step, column, outcome, None, terms, variables, filled_raw, fit_kinds,
                           every_row)
            estimates.append(model["fit"]["beta"])
            variances.append(np.diag(model["fit"]["cov"]))
        m = IMPUTATIONS
        q, u = np.array(estimates), np.array(variances)
        qbar, ubar, b = q.mean(axis=0), u.mean(axis=0), q.var(axis=0, ddof=1)
        total = ubar + (1 + 1 / m) * b
        se = np.sqrt(total)
        lam = (1 + 1 / m) * b / total
        dfcom = float(model["n"] - model["X"].shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            df_old = np.where(lam > 0, (m - 1) / lam ** 2, np.inf)
            df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
            df = np.where(np.isinf(df_old), df_obs, df_old * df_obs / (df_old + df_obs))
        stat = qbar / se
        p = 2 * stats.t.sf(np.abs(stat), df)
        tq = stats.t.ppf(0.975, df)
        coef = dict(beta=qbar, se=se, stat=stat, p=p, low=qbar - tq * se, high=qbar + tq * se, name="t", df=df)
        self._print_terms(model, coef, f"{model['n']} rows, {rows_with_gaps} of them with empty cells filled in")
        formula = f"{_r_name(column)} ~ " + " + ".join(pieces)
        call = {"numbers": f"lm({formula})", "yes or no": f"glm({formula}, family = binomial)",
                "counts": f"glm({formula}, family = poisson)",
                "counts (overdispersed)": f"glm({formula}, family = quasipoisson)"}[outcome]
        if yes_words:
            print(f"Outcome {yes_words}.")
        print(f"Method: multiple imputation by chained equations (predictive mean matching with {PMM_DONORS} donors, "
              f"seed={seed}): the empty cells were filled in {m} times, the model was fitted to each of the {m} "
              "filled-in tables, and the results were pooled with Rubin's rules (t-based intervals with Barnard-Rubin "
              f"degrees of freedom), as R pool(with(mice(...), {call})).")
        print(f"The filling-in adds uncertainty: up to {round(float(np.max(lam)) * 100)} % of the variance of an "
              "effect comes from the differences between the filled-in tables. Tests for whole group columns are not "
              "pooled here.")
        self._closing(p[1:])
        return self._term_table(model, coef, [], extra={"share of variance from empty cells": lam})

    # ------------------------------------------------------------------ add_predicted_column
    @friendly
    def add_predicted_column(self, column=None, using=None, outcome="numbers", name=None, grouped_by=None):
        """Add a column with the value the model predicts for each row.

        table.add_predicted_column("marks", using=["hours", "breakfast"])   (with sdata.example("school"))
        The same model as explain() with the same options: outcome="numbers" gives the fitted
        value (R fitted(lm(...))), outcome="yes or no" the chance of yes from 0 to 1 and
        outcome="counts" the expected count (R fitted(glm(...))). With grouped_by= the
        prediction includes each group's own level (R fitted(lmer(...))). Rows with an empty
        cell in a column used stay empty. Undo with table.undo().
        """
        step = "add_predicted_column"
        terms, pieces, variables, raw, missing, kinds = self._model_parts(
            step, column, using, outcome, grouped_by, "leave out")
        names = [column] + variables + ([grouped_by] if grouped_by is not None else [])
        if name is None:
            name = f"predicted {column}"
        if not isinstance(name, str) or not name.strip():
            raise SimplydataError(f'{step}() needs the new column\'s name in quotes, for example name="predicted".')
        if name in self._columns:
            raise SimplydataError(f'There is already a column called "{name}". Choose another name with name=.')
        keep = ~np.any([missing[nm] for nm in names], axis=0)
        model = self._fit_model(step, column, outcome, grouped_by, terms, variables, raw, kinds, keep)
        values = [None] * len(keep)
        for i, v in zip(np.flatnonzero(keep), model["fit"]["fitted"]):
            values[i] = float(v)
        base = nw.maybe_reset_index(self._frame)
        new = nw.new_series(name, values, nw.Float64, backend=base.implementation)
        what = {"numbers": "the fitted value", "yes or no": "the chance of yes (0 to 1)",
                "counts": "the expected count", "counts (overdispersed)": "the expected count"}[outcome]
        how = {"lm": "least squares", "binomial": "logistic regression", "poisson": "Poisson regression",
               "quasipoisson": "quasi-Poisson regression", "mixed": "linear mixed model"}[model["kind"]]
        empty = len(keep) - model["n"]
        yes = f"; {model['yes_words']}" if model["yes_words"] else ""
        print(f'Added "{name}": {what} of "{column}" from {_list_words(_quoted(v) for v in variables)} '
              f"({how}, {model['n']} rows{yes})." + (f" {empty} rows with an empty cell stay empty." if empty else ""))
        return self._change(f'add predicted column "{name}" ({column} from {", ".join(pieces)}, {outcome})',
                            base.with_columns(new))
