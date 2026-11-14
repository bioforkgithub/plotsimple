"""Fitting lines, curves and formulas to the columns of a table. They never change the table
(except add_predicted_column with a formula, which adds one column and can be undone).

Choices (the same defaults as R, so results can be checked there):

* fit_line(x=, y=): ordinary least squares (R lm(y ~ x)), t-based 95 % intervals. Its old
  name fit_straight_line still works for one release, with a note.
* fit_curve(x=, y=, degree=2): least squares on the powers of x, as R
  lm(y ~ poly(x, 2, raw = TRUE)), solved through a QR decomposition of the matrix of powers
  (as lm, and steadier than the normal equations np.polyfit uses). t-based 95 % intervals,
  R squared and adjusted R squared as R summary.lm(). weights= as R lm(weights =).
* fit_formula("y = a * exp(b * x)"): nonlinear least squares, as R nls().
  - Without between=: Levenberg-Marquardt (SciPy curve_fit method "lm"; R nls() uses
    Gauss-Newton and minpack.lm::nlsLM() uses Levenberg-Marquardt: both land on the same
    least-squares answer).
  - With between=: trust region reflective, which keeps every value inside its limits
    (curve_fit method "trf"), as R nls(algorithm = "port", lower =, upper =).
  - method= lets an expert choose "levenberg-marquardt", "trust region" or "dogbox".
  - weights=: weighted least squares (curve_fit sigma = 1 / sqrt(weights)), as R
    nls(weights =).
  - search="whole range": differential evolution over the whole range of between= (SciPy
    differential_evolution, seed 1, so the result is the same every time), then nonlinear
    least squares from its best point. Useful when start values are hard to guess or the
    fit has several dips.
  Standard errors, t and p-values as R summary.nls(): the covariance is the residual
  variance times (J'WJ)^-1 at the answer. The 95 % intervals are estimate +/- t x standard
  error (Wald intervals); R confint() on an nls fit profiles the likelihood instead and
  can differ a little. Residual standard error, degrees of freedom and AIC as R sigma(),
  df.residual() and AIC() (R logLik.nls, which counts the residual variance as one more
  value). R squared is kept as a rough guide only: for a curve it has no exact meaning.
  There is no add_residual_column: the residuals are the fitted column minus the predicted
  one, one add_column away.
* add_predicted_column("y = a * exp(b * x)"): fits the formula as fit_formula() (or takes
  the values from a fit_formula() result with using=) and adds the value of the formula
  for every row, as R predict(nls(...), newdata).

for_each= fits each group on its own (one row, or one set of rows, per group in the result).
Problems are printed notes, never silent switches.
"""

from __future__ import annotations

import math
import warnings

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, choice_error, friendly

DEFAULT_START = 1.0
MAX_DEGREE = 10  # as chart.fit_curve
WIGGLY_DEGREE = 4  # from this degree a note says the curve may wiggle between points
SEARCHES = ("from start", "whole range")
SEARCH_SPELLINGS = {"from the start": "from start", "start": "from start", "whole": "whole range",
                    "global": "whole range", "the whole range": "whole range"}
# method= words -> SciPy curve_fit method
METHODS = {"levenberg-marquardt": "lm", "trust region": "trf", "dogbox": "dogbox"}
METHOD_SPELLINGS = {"lm": "levenberg-marquardt", "levenberg marquardt": "levenberg-marquardt",
                    "marquardt": "levenberg-marquardt", "trf": "trust region", "trust-region": "trust region",
                    "trust region reflective": "trust region", "port": "trust region"}
METHOD_WORDS = {"lm": "Levenberg-Marquardt", "trf": "trust region reflective", "dogbox": "dogbox"}
SEED = 1  # the whole-range search is random; this seed makes it the same every time
LIMIT_SHARE = 1e-8  # a value this close to a limit of between= is "at the limit"


def _st():
    """The statistics module (imported when needed: it imports this module)."""
    from . import stat_tests

    return stat_tests


def _plain_value(value):
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------------------- the formula
class ParsedFormula:
    """A formula to fit, read safely (with ast, never run): "population = a * exp(b * year)"."""

    def __init__(self, text, left, model, names, used):
        self.text, self.left, self.model, self.names, self.used = text, left, model, names, used

    @property
    def columns_used(self) -> list[str]:
        return [self.left] + list(self.used)


def read_fit_formula(formula, columns, step="fit_formula") -> ParsedFormula:
    """Read "y = ... " with the safe formula reader; columns are data, other names are values to find."""
    st = _st()
    if not isinstance(formula, str) or formula.count("=") != 1 or "==" in formula:
        raise st._fit_error(
            f'{step}() needs a formula with one = sign: the column to fit on the left, for example\n'
            f'table.{step}("marks = a * hours + b").'
        )
    left, right = (part.strip() for part in formula.split("="))
    if left.startswith("`") and left.endswith("`"):
        left = left[1:-1]
    columns = list(columns)
    if left not in columns:
        from .._errors import column_error

        raise column_error(left, columns)
    reader = st._FitReader(right, columns)
    try:
        model = reader.read()
    except SimplydataError as error:
        if reader.finished and not reader.used:
            raise st._fit_error(
                f'The right side of the formula has no column in it, so there is nothing to fit "{left}" to.'
            ) from None
        raise SimplydataError(str(error).replace(st.FORMULA_HELP, st.FIT_HELP)) from None
    names = list(reader.values_to_find)
    if not names:
        raise st._fit_error("The formula has no value to find. Use a name that is not a column, such as a or b.")
    if left in reader.used:
        raise st._fit_error(f'"{left}" is on both sides of the formula.')
    return ParsedFormula(formula, left, model, names, list(reader.used))


# ---------------------------------------------------------------------------- options
def _read_start(step, start, names) -> dict:
    st = _st()
    if start is None:
        return {}
    if st._is_number(start):
        if len(names) != 1:
            raise SimplydataError(
                f"start= is one number, but the formula has {len(names)} values to find ({', '.join(names)}).\n"
                "Give one for each, for example start={" + ", ".join(f'"{k}": 1' for k in names) + "}."
            )
        return {names[0]: float(start)}
    if not isinstance(start, dict) or not all(st._is_number(v) for v in start.values()):
        raise SimplydataError('start= should look like {"a": 10, "b": 0.5}: a name and a number for each value.')
    unknown = [k for k in start if k not in names]
    if unknown:
        raise SimplydataError(f"start= has {st._list_words(repr(k) for k in unknown)}, which "
                              f"{'is' if len(unknown) == 1 else 'are'} not in the formula. "
                              f"Values to find: {', '.join(names)}.")
    return {k: float(v) for k, v in start.items()}


def _limit(value, side, name):
    st = _st()
    if value is None:
        return -math.inf if side == 0 else math.inf
    if not st._is_number(value) or math.isnan(value):
        raise SimplydataError(f"The limits for {name} in between= should be numbers (or None for no limit), "
                              f"not {value!r}.")
    return float(value)


def _read_between(step, between, names):
    """(low, high) arrays from between=[low, high] (one value) or {"a": [low, high], ...}."""
    low = np.full(len(names), -math.inf)
    high = np.full(len(names), math.inf)
    if between is None:
        return low, high
    example = "between={" + ", ".join(f'"{k}": [0, None]' for k in names) + "}"
    if isinstance(between, (list, tuple)):
        if len(names) != 1:
            raise SimplydataError(
                f"between=[low, high] is for one value to find, but the formula has {len(names)} "
                f"({', '.join(names)}).\nGive limits by name, for example {example} (None means no limit)."
            )
        between = {names[0]: between}
    if not isinstance(between, dict):
        raise SimplydataError(f"between= should give a low and a high limit for each value, for example {example}.")
    unknown = [k for k in between if k not in names]
    if unknown:
        raise SimplydataError(f"between= has {_st()._list_words(repr(k) for k in unknown)}, which "
                              f"{'is' if len(unknown) == 1 else 'are'} not in the formula. "
                              f"Values to find: {', '.join(names)}.")
    for name, pair in between.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise SimplydataError(f"between= needs two limits for {name}, [low, high], for example {example}.")
        i = names.index(name)
        low[i], high[i] = _limit(pair[0], 0, name), _limit(pair[1], 1, name)
        if not low[i] < high[i]:
            raise SimplydataError(f"The low limit for {name} in between= should be below the high limit "
                                  f"(it is {pair[0]} to {pair[1]}).")
    return low, high


def _read_search(step, search) -> str:
    if isinstance(search, str):
        key = search.strip().lower()
        key = SEARCH_SPELLINGS.get(key, key)
        if key in SEARCHES:
            return key
    raise choice_error(step, "search", search, SEARCHES)


def _read_method(step, method, search, low, high) -> str:
    """The curve_fit method: method= if given, else Levenberg-Marquardt, or trust region with limits."""
    limited = bool(np.any(np.isfinite(low)) or np.any(np.isfinite(high)))
    if method is None:
        return "trf" if limited or search == "whole range" else "lm"
    key = method.strip().lower() if isinstance(method, str) else method
    key = METHOD_SPELLINGS.get(key, key)
    if key not in METHODS:
        raise choice_error(step, "method", method, list(METHODS))
    if METHODS[key] == "lm" and (limited or search == "whole range"):
        raise SimplydataError(
            "Levenberg-Marquardt cannot keep values between limits. With between= use method=\"trust region\" "
            "(or leave method= out), or leave out between= to use Levenberg-Marquardt."
        )
    return METHODS[key]


def _default_start(low, high) -> float:
    if low <= DEFAULT_START <= high:
        return DEFAULT_START
    if math.isfinite(low) and math.isfinite(high):
        return (low + high) / 2
    return low + 1 if math.isfinite(low) else high - 1


# ---------------------------------------------------------------------------- the fit
class FitResult:
    """What fit_formula_core found. predict(data) gives the formula's value for new data."""

    def __init__(self, parsed, values, standard_errors, cov, n, df, sigma, rss, r2, aic, converged,
                 method_sentence, method_name, notes, skipped):
        self.parsed = parsed
        self.formula = parsed.text
        self.left = parsed.left
        self.names = list(parsed.names)
        self.columns_used = list(parsed.used)
        self.values = np.asarray(values, dtype=float)
        self.standard_errors = np.asarray(standard_errors, dtype=float)
        self.cov = cov
        self.n, self.df, self.sigma, self.rss, self.r2, self.aic = n, df, sigma, rss, r2, aic
        self.converged = converged
        self.method_sentence = method_sentence
        self.method_name = method_name
        self.notes = list(notes)
        self.skipped = skipped

    @property
    def value_dict(self) -> dict:
        return dict(zip(self.names, self.values.tolist()))

    @property
    def t(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.values / self.standard_errors

    @property
    def p_values(self):
        from scipy import stats

        return 2 * stats.t.sf(np.abs(self.t), self.df)

    @property
    def intervals(self):
        from scipy import stats

        q = float(stats.t.ppf(0.975, self.df))
        return self.values - q * self.standard_errors, self.values + q * self.standard_errors

    def predict(self, data) -> np.ndarray:
        """The formula with the values found, for data = {column: numbers}."""
        return predict_formula(self.parsed, self.value_dict, data)


def predict_formula(parsed, values: dict, data) -> np.ndarray:
    arrays = {c: np.asarray(data[c], dtype=float) for c in parsed.used}
    size = len(next(iter(arrays.values()))) if arrays else 1
    with np.errstate(all="ignore"):
        out = parsed.model(values, arrays)
    return np.array(np.broadcast_to(np.asarray(out, dtype=float), (size,)))


def _method_sentence(key, search, limited, weights_name):
    words = METHOD_WORDS[key]
    weighted = "weighted " if weights_name is not None else ""
    r_weights = f", weights = {weights_name}" if weights_name is not None else ""
    if search == "whole range":
        how = (f"differential evolution over the whole range of between= (seed {SEED}), then {weighted}nonlinear "
               f"least squares ({words}) from its best point, as R nls(algorithm = \"port\", lower =, upper ="
               f"{r_weights}) from a good start")
    elif limited:
        how = (f"{weighted}nonlinear least squares within the limits of between= ({words}), as R "
               f"nls(algorithm = \"port\", lower =, upper ={r_weights})")
    else:
        r_call = "R nls()" if not r_weights else f"R nls({r_weights[2:]})"
        how = f"{weighted}nonlinear least squares ({words}), as {r_call}"
    return (f"Method: {how}; intervals are estimate +/- t x standard error (R confint() on nls profiles the "
            "likelihood and can differ a little).")


def fit_formula_core(formula, data, columns, start=None, between=None, weights=None, search="from start",
                     method=None, step="fit_formula", weights_name=None) -> FitResult:
    """Fit "y = formula" to data = {column: numbers}; rows with an empty or impossible cell are left out.

    formula is the text (or a ParsedFormula); columns are all the column names (the reader needs
    them to tell data from values to find). weights is an array (or None). Raises SimplydataError
    with a plain explanation when the fit fails.
    """
    from scipy import optimize

    st = _st()
    parsed = formula if isinstance(formula, ParsedFormula) else read_fit_formula(formula, columns, step)
    names = parsed.names
    start = _read_start(step, start, names)
    low, high = _read_between(step, between, names)
    search = _read_search(step, search)
    key = _read_method(step, method, search, low, high)
    limited = bool(np.any(np.isfinite(low)) or np.any(np.isfinite(high)))

    y = np.asarray(data[parsed.left], dtype=float)
    xs = {c: np.asarray(data[c], dtype=float) for c in parsed.used}
    keep = np.isfinite(y)
    for values in xs.values():
        keep &= np.isfinite(values)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        keep &= np.isfinite(w)
        if np.any(w[keep] <= 0):
            raise SimplydataError(
                "weights= needs a number above 0 in every row used (a row with weight 0 would count for "
                "nothing). Leave such rows out first, for example with table.keep_rows(...)."
            )
        w = w[keep]
    skipped = int(len(y) - keep.sum())
    y = y[keep]
    xs = {c: v[keep] for c, v in xs.items()}
    n, p = len(y), len(names)
    if n <= p:
        raise SimplydataError(f"{step}() needs more rows ({n}) than values to find ({p}).")

    p0 = [start.get(k, _default_start(lo, hi)) for k, lo, hi in zip(names, low, high)]
    outside = [k for k, v, lo, hi in zip(names, p0, low, high) if not lo <= v <= hi]
    if outside:
        raise SimplydataError(f"The start value for {st._list_words(outside)} is outside its limits in between=. "
                              "Give a start value inside the limits.")
    weight = np.ones(n) if w is None else w

    def curve(_, *theta):
        with np.errstate(all="ignore"):
            out = parsed.model(dict(zip(names, theta)), xs)
        return np.broadcast_to(np.asarray(out, dtype=float), y.shape)

    start_text = "start={" + ", ".join(f'"{k}": {v:g}' for k, v in zip(names, p0)) + "}"
    start_hint = "start={" + ", ".join(f'"{k}": ...' for k in names) + "}"
    notes = []
    converged = True
    if search == "whole range":
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            missing = [k for k, lo, hi in zip(names, low, high) if not (math.isfinite(lo) and math.isfinite(hi))]
            raise SimplydataError(
                f'search="whole range" needs a low and a high limit for every value to find; '
                f"{st._list_words(missing)} {'has' if len(missing) == 1 else 'have'} none.\n"
                "For example between={" + ", ".join(f'"{k}": [0, 10]' for k in names) + "}."
            )

        def badness(theta):
            r = y - curve(None, *theta)
            total = float(np.sum(weight * r * r))
            return total if math.isfinite(total) else 1e300

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searched = optimize.differential_evolution(badness, list(zip(low, high)), rng=SEED, polish=False,
                                                       tol=1e-8, maxiter=2000)
        if not searched.success:
            notes.append("Note: the search over the whole range did not settle; the best point it found was "
                         "used as the start. Try narrower limits in between=.")
            converged = False
        p0 = list(searched.x)
    elif not np.all(np.isfinite(curve(None, *p0))):
        raise SimplydataError(
            f"With the start values ({start_text}) the formula gives impossible numbers, such as the log of a "
            "negative number or a division by 0.\nGive start values close to what you expect, for example "
            f"table.{step}(\"{parsed.text}\", {start_hint})."
        )
    else:
        with np.errstate(all="ignore"):
            start_rss = float(np.sum(weight * (y - curve(None, *p0)) ** 2))
        if not math.isfinite(start_rss):  # a search from overflow ends in a different place on each computer
            raise SimplydataError(
                f"With the start values ({start_text}) the formula gives impossible numbers (too big to work "
                f"with).\nGive start values close to what you expect, for example table.{step}(\"{parsed.text}\", "
                f"{start_hint}), or limits with between=."
            )

    options = {"p0": p0, "method": key, "full_output": True}
    if w is not None:
        options["sigma"] = 1 / np.sqrt(w)
    if key == "lm":
        options.update(maxfev=20000, xtol=1e-12, ftol=1e-12)
    else:
        options.update(max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12, bounds=(low, high))
    uncertain = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            found, cov, *_ = optimize.curve_fit(curve, np.zeros(n), y, **options)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            raise SimplydataError(
                f"simplydata could not fit the formula: the search for {st._list_words(names)} did not settle "
                f"(it began at {start_text}).\nTry start values close to what you expect, for example "
                f"{start_hint}, or limits with between= and search=\"whole range\", or check the formula."
            ) from None
        uncertain = any(issubclass(c.category, optimize.OptimizeWarning) for c in caught)
    found = np.asarray(found, dtype=float)
    fitted = curve(None, *found)
    if not np.all(np.isfinite(fitted)):
        raise SimplydataError("The fit ended on values that give impossible numbers. Try other start values.")
    cov = np.asarray(cov, dtype=float) if cov is not None else np.full((p, p), np.inf)
    if not uncertain and _not_pinned_down(curve, found, weight):
        uncertain = True  # as SciPy does when it notices: no standard errors
        cov = np.full((p, p), np.inf)
    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(cov))
    residuals = y - fitted
    with np.errstate(all="ignore"):
        rss = float(np.sum(weight * residuals ** 2))
    if not math.isfinite(rss):
        raise SimplydataError("The fit ended on values that give impossible numbers (too big to work with). "
                              f"Try other start values, for example {start_hint}, or limits with between=.")
    df = n - p
    sigma = math.sqrt(rss / df)
    centre = float(np.sum(weight * y) / np.sum(weight))
    tss = float(np.sum(weight * (y - centre) ** 2))
    r2 = 1 - rss / tss if tss > 0 else math.nan
    # R logLik.nls: -N/2 (log(2 pi) + 1 - log(N) - sum(log(w)) / N + log(sum(w r^2))), with p + 1 values
    log_lik = -n / 2 * (math.log(2 * math.pi) + 1 - math.log(n) - float(np.sum(np.log(weight))) / n
                        + math.log(rss)) if rss > 0 else math.inf
    aic = -2 * log_lik + 2 * (p + 1)
    if uncertain or not np.all(np.isfinite(errors)):
        notes.append("Note: the standard errors could not be worked out well: the data do not pin down every "
                     "value. Try a simpler formula or more data.")
    at_limit = [f"{k} ended at its limit ({(lo if abs(v - lo) <= LIMIT_SHARE * max(1, abs(lo)) else hi):g})"
                for k, v, lo, hi in zip(names, found, low, high)
                if (math.isfinite(lo) and abs(v - lo) <= LIMIT_SHARE * max(1, abs(lo)))
                or (math.isfinite(hi) and abs(v - hi) <= LIMIT_SHARE * max(1, abs(hi)))]
    if at_limit:
        notes.append(f"Note: {st._list_words(at_limit)}: the best fit may lie beyond it, and the standard error "
                     "and interval there are only a rough guide. Widen between= if the limit is not a real one.")
    sentence = _method_sentence(key, search, limited, weights_name)
    name = "differential evolution, then " + METHOD_WORDS[key] if search == "whole range" else METHOD_WORDS[key]
    return FitResult(parsed, found, errors, cov, n, df, sigma, rss, r2, aic, converged, sentence, name, notes,
                     skipped)



def _not_pinned_down(curve, found, weight) -> bool:
    """True when the data cannot pin down every value: the gradient has lower rank (R nls(): "singular gradient").

    SciPy only notices this when the columns of its numerical gradient come out exactly in
    proportion, which depends on the computer's rounding. Here the gradient is worked out by
    central differences, each column scaled to length 1, and the rank checked with R qr()'s
    tolerance (1e-7).
    """
    theta = np.asarray(found, dtype=float)
    root_w = np.sqrt(np.asarray(weight, dtype=float))
    columns = []
    with np.errstate(all="ignore"):
        for j in range(len(theta)):
            h = np.finfo(float).eps ** (1 / 3) * max(abs(theta[j]), 1.0)
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            columns.append(root_w * (curve(None, *up) - curve(None, *down)) / (2 * h))
    jac = np.column_stack(columns)
    if not np.all(np.isfinite(jac)):
        return False
    norms = np.linalg.norm(jac, axis=0)
    if np.any(norms == 0):
        return True
    singular = np.linalg.svd(jac / norms, compute_uv=False)
    return bool(singular[-1] < 1e-7 * singular[0])


# ---------------------------------------------------------------------------- polynomial
def fit_curve_core(x, y, degree, weights=None) -> dict:
    """Least squares on 1, x, x**2, ... x**degree (QR), as R lm(y ~ poly(x, degree, raw = TRUE))."""
    from scipy import stats

    from .models import _least_squares

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    n, k = len(y), degree + 1
    X = np.vander(x, k, increasing=True)
    root = np.sqrt(w)
    beta, unscaled, _, rss = _least_squares(X * root[:, None], y * root)
    df = n - k
    sigma2 = rss / df
    se = np.sqrt(np.diag(unscaled) * sigma2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    q = float(stats.t.ppf(0.975, df))
    fitted = X @ beta
    centre = float(np.sum(w * fitted) / np.sum(w))  # R summary.lm with weights: the weighted mean
    mss = float(np.sum(w * (fitted - centre) ** 2))
    r2 = mss / (mss + rss) if mss + rss > 0 else math.nan
    adjusted = 1 - (1 - r2) * (n - 1) / df
    return {"beta": beta, "se": se, "t": t, "p": p, "low": beta - q * se, "high": beta + q * se, "df": df,
            "r2": r2, "adjusted": adjusted, "n": n, "sigma": math.sqrt(sigma2), "fitted": fitted}


def _curve_text(y, x, beta) -> str:
    """marks = 31.0 + 8.26 * hours - 0.297 * hours**2"""
    st = _st()
    parts = []
    for power, b in enumerate(beta):
        number = st._num(abs(b))
        term = number if power == 0 else f"{number} * {x}" if power == 1 else f"{number} * {x}**{power}"
        if not parts:
            parts.append(("-" if b < 0 else "") + term)
        else:
            parts.append(("- " if b < 0 else "+ ") + term)
    return f"{y} = " + " ".join(parts)


# ---------------------------------------------------------------------------- the words
class FittingWords:
    """fit_line, fit_curve, fit_formula (and the formula part of add_predicted_column)."""

    # fit_line was fit_straight_line (renamed to match chart.fit_line): the old name works for
    # one release, with a note the first time (BaseTable._OLD_WORDS in table/_base.py).

    # ------------------------------------------------------------------ helpers
    def _rows_table(self, rows: list[dict]):
        """A result table with one row per item, on the same engine as this table."""
        keys = list(dict.fromkeys(k for r in rows for k in r))
        data = {k: [_plain_value(r.get(k)) for r in rows] for k in keys}
        return self._new_table(nw.from_dict(data, backend=self._frame.implementation))

    def _fit_data(self, step, columns, for_each, weights, example):
        """[(group label, group value, {column: numbers})], empty cells skipped with a note.

        Without for_each= there is one item, with label None."""
        st = _st()
        needed = list(dict.fromkeys(list(columns) + ([weights] if weights is not None else [])))
        if weights is not None:
            self._need_name(step, "weights=", weights, example)
        if for_each is not None:
            self._need_name(step, "for_each=", for_each, example)
            self._need_columns(for_each)
            if for_each in needed:
                raise SimplydataError(f'"{for_each}" cannot be both a column of the fit and the for_each= column.\n'
                                      f"For example, {example}")
        frame = self._number_frame(step, needed)
        if for_each is not None and frame.schema[for_each].is_float():
            frame = frame.with_columns(nw.col(for_each).fill_nan(None))
        groups = [for_each] if for_each is not None else []
        kept = frame.select(*groups, *needed).drop_nulls()
        st._skipped_note(len(frame) - len(kept), needed + groups)
        arrays = {c: kept[c].to_numpy().astype(float) for c in needed}
        if for_each is None:
            return [(None, None, arrays)]
        keys = np.asarray(kept[for_each].to_list(), dtype=object)
        order = sorted(set(keys.tolist()), key=lambda k: (str(type(k)), k))
        if not order:
            raise SimplydataError(f'{step}() found no rows with a number in every column used.')
        return [(st._label(k), k, {c: a[keys == k] for c, a in arrays.items()}) for k in order]

    # ------------------------------------------------------------------ fit_line
    @friendly
    def fit_line(self, x=None, y=None):
        """Fit the straight line y = slope * x + intercept that best follows the points.

        table.fit_line(x="hours", y="marks")   (with sdata.example("marks"))
        Ordinary least squares (the same as R lm(y ~ x)). Prints how much y goes up for each
        1 more of x, with its 95 % confidence interval and p-value, and R squared: the share
        of the differences in y that the line explains. Rows with an empty cell in x or y
        are skipped. A link does not show that x causes y. Gives back a one-row table.
        The same line as chart.fit_line(); its old name fit_straight_line still works.
        For a curve use fit_curve(), for any formula fit_formula().
        """
        st = _st()
        step = "fit_line"
        example = 'with sdata.example("marks"): table.fit_line(x="hours", y="marks")'
        self._need_name(step, "x=", x, example)
        self._need_name(step, "y=", y, example)
        if x == y:
            raise SimplydataError(f"{step}() needs two different columns, for example {example}.")
        xs, ys = self._number_values(step, [x, y])
        if len(xs) < 3:
            raise SimplydataError(f'{step}() needs at least 3 rows with a number in both "{x}" and "{y}".')
        if np.ptp(xs) == 0:
            raise SimplydataError(f'Every number in "{x}" is the same, so no line can be fitted.')
        from scipy import stats

        fit = stats.linregress(xs, ys)
        n = len(xs)
        df = n - 2
        tq = float(stats.t.ppf(0.975, df))
        slope, intercept = float(fit.slope), float(fit.intercept)
        se_s, se_i = float(fit.stderr), float(fit.intercept_stderr)
        r2 = float(fit.rvalue) ** 2
        p = float(fit.pvalue)
        row = {"test": "straight-line fit (least squares)", "what it compares": f'"{y}" against "{x}"',
               "slope": slope, "slope 95 % CI low": slope - tq * se_s, "slope 95 % CI high": slope + tq * se_s,
               "slope standard error": se_s, "intercept": intercept,
               "intercept 95 % CI low": intercept - tq * se_i, "intercept 95 % CI high": intercept + tq * se_i,
               "intercept standard error": se_i, "statistic name": "t",
               "statistic": slope / se_s if se_s > 0 else math.inf, "degrees of freedom": float(df),
               "p-value": p, "R squared": r2, "rows": n}
        direction = "up" if slope >= 0 else "down"
        num = st._num
        print(f'"{y}" goes {direction} by {num(abs(slope))} for each 1 more "{x}" (95 % CI '
              f'{num(row["slope 95 % CI low"])} to {num(row["slope 95 % CI high"])}); this slope {st._chance(p)}. '
              f'R squared = {r2:.2f}: "{x}" explains about {round(r2 * 100)} % of the differences in "{y}".')
        print(f"Line: {y} = {num(slope)} * {x} + {num(intercept)}, by least squares over {n} rows. "
              "A link does not show that one thing causes the other.")
        residuals = ys - (intercept + slope * xs)
        for note in st._shape_notes([("the line", residuals)], what="the distances from the line"):
            print(note.replace('a safer test here: based_on="ranks".',
                               "the p-value and confidence intervals may be off."))
        return self._result_table(row)

    # ------------------------------------------------------------------ fit_curve
    @friendly
    def fit_curve(self, x=None, y=None, degree=2, weights=None, for_each=None):
        """Fit a curve y = b0 + b1 * x + b2 * x**2 ... that best follows the points.

        table.fit_curve(x="hours", y="marks", degree=2)   (with sdata.example("marks"))
        degree=2 is a curve with one bend, degree=3 can bend twice, and so on (up to 10), the
        same curve as chart.fit_curve(degree=). Least squares on the powers of x, as R
        lm(y ~ poly(x, 2, raw = TRUE)). Prints the curve and R squared; gives back one row
        per term (the intercept, x, x**2 ...) with its estimate, standard error, t, p-value
        and 95 % confidence interval. weights="column" counts some rows more (as R
        lm(weights =)); for_each="column" fits each group on its own. Rows with an empty
        cell are skipped.
        """
        st = _st()
        step = "fit_curve"
        example = 'with sdata.example("marks"): table.fit_curve(x="hours", y="marks", degree=2)'
        self._need_name(step, "x=", x, example)
        self._need_name(step, "y=", y, example)
        if x == y:
            raise SimplydataError(f"{step}() needs two different columns, for example {example}.")
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 1 <= degree <= MAX_DEGREE:
            raise SimplydataError(
                f"degree= should be a whole number from 1 to {MAX_DEGREE}, not {degree!r}.\n"
                "For example degree=2 for a curve with one bend."
            )
        degree = int(degree)
        groups = self._fit_data(step, [x, y], for_each, weights, example)
        rows, lines, failed = [], [], []
        for label, key, arrays in groups:
            xs, ys = arrays[x], arrays[y]
            w = arrays[weights] if weights is not None else None
            where = f' in group "{label}"' if label is not None else ""
            try:
                if w is not None and np.any(w <= 0):
                    raise SimplydataError("weights= needs a number above 0 in every row used.")
                if len(np.unique(xs)) <= degree:
                    raise SimplydataError(
                        f'A curve of degree {degree} needs at least {degree + 1} different values of "{x}"{where}, '
                        f"and more rows than that to say anything about its fit; there are {len(np.unique(xs))}.")
                if len(xs) <= degree + 1:
                    raise SimplydataError(f"{step}() needs more rows ({len(xs)}){where} than terms in the curve "
                                          f"({degree + 1}).")
                fit = fit_curve_core(xs, ys, degree, w)
            except SimplydataError as error:
                if label is None:
                    raise
                failed.append((label, str(error).split("\n")[0]))
                continue
            terms = ["(intercept)", x] + [f"{x}**{k}" for k in range(2, degree + 1)]
            for j, term in enumerate(terms):
                row = {} if label is None else {for_each: key}
                row.update({"term": term, "estimate": float(fit["beta"][j]),
                            "standard error": float(fit["se"][j]), "statistic name": "t",
                            "statistic": float(fit["t"][j]), "degrees of freedom": float(fit["df"]),
                            "p-value": float(fit["p"][j]), "95 % CI low": float(fit["low"][j]),
                            "95 % CI high": float(fit["high"][j]), "R squared": fit["r2"],
                            "adjusted R squared": fit["adjusted"], "rows": fit["n"]})
                rows.append(row)
            lines.append((label, fit))
        if not rows:
            raise SimplydataError(f"{step}() could not fit a curve in any group: {failed[0][1]}")
        r_weights = f", weights = {weights}" if weights is not None else ""
        if for_each is None:
            fit = lines[0][1]
            print(f"Curve: {_curve_text(y, x, fit['beta'])} (each with its 95 % CI in the table); "
                  f"R squared = {fit['r2']:.2f}, adjusted {fit['adjusted']:.2f}.")
        else:
            print(f'Curves of degree {degree} for each "{for_each}" (each value with its 95 % CI in the table):')
            for label, fit in lines[:st.GROUPS_SHOWN]:
                print(f"  {label} ({fit['n']} rows): {_curve_text(y, x, fit['beta'])}; "
                      f"R squared = {fit['r2']:.2f}, adjusted {fit['adjusted']:.2f}.")
            if len(lines) > st.GROUPS_SHOWN:
                print(f"  ... and {len(lines) - st.GROUPS_SHOWN} more groups (the table has them all)")
        weighted = f'weighted least squares (weights "{weights}")' if weights is not None else "least squares"
        print(f"Method: {weighted} on powers of {x}, as R lm({y} ~ poly({x}, {degree}, raw = TRUE){r_weights}).")
        for label, problem in failed:
            print(f'Note: group "{label}" was not fitted: {problem}')
        if degree >= WIGGLY_DEGREE:
            print('Note: high degrees wiggle between points; a smooth curve (values_at(..., smoothness="auto")) '
                  "or a formula (fit_formula) may describe the data better.")
        return self._rows_table(rows)

    # ------------------------------------------------------------------ fit_formula
    def _fit_formula_groups(self, step, formula, start, between, weights, for_each, search, method, example):
        """[(label, group value, FitResult or error text)] and the parsed formula."""
        parsed = read_fit_formula(formula, self._columns, step)
        groups = self._fit_data(step, parsed.columns_used, for_each, weights, example)
        fits = []
        for label, key, arrays in groups:
            try:
                fit = fit_formula_core(parsed, arrays, self._columns, start=start, between=between,
                                       weights=None if weights is None else arrays[weights], search=search,
                                       method=method, step=step, weights_name=weights)
            except SimplydataError as error:
                if label is None or len(groups) == 1:
                    raise
                fits.append((label, key, str(error).split("\n")[0]))
                continue
            fits.append((label, key, fit))
        if all(isinstance(f, str) for _, _, f in fits):
            raise SimplydataError(f"{step}() could not fit the formula in any group. In the first: {fits[0][2]}")
        return parsed, fits

    @friendly
    def fit_formula(self, formula=None, start=None, between=None, weights=None, for_each=None,
                    search="from start", method=None):
        """Find the values in a formula that make it follow the data best.

        table.fit_formula("population = a * exp(b * (year - 2000))")   (with sdata.example("growth"))
        Write the column to fit on the left of =. On the right, names that are columns are
        data; any other name (such as a, b or rate) is a value to find. Uses nonlinear least
        squares, as R nls() (Levenberg-Marquardt). Every value to find starts at 1; give
        better start values with start={"a": 10, "b": 0.03} if the fit fails.
        between={"b": [0, None]} keeps a value between limits (None: no limit), as R
        nls(algorithm = "port"); search="whole range" searches the whole range of between=
        before fitting (differential evolution, seed 1), for when start values are hard to
        guess. weights="column" counts some rows more (as R nls(weights =)). for_each="column"
        fits each group on its own. method= lets an expert choose "levenberg-marquardt",
        "trust region" or "dogbox".
        Prints each value with its 95 % confidence interval, the residual standard error and
        R squared (only a rough guide for a curve). Gives back one row (one per group) with
        each value, its standard error, t, p-value and 95 % CI, the residual standard error,
        degrees of freedom, R squared, AIC, rows and whether the fit converged. Residuals:
        add_predicted_column with the same formula, then add_column("y - predicted y").
        Rows with an empty cell are skipped.
        """
        st = _st()
        step = "fit_formula"
        example = 'with sdata.example("growth"): table.fit_formula("population = a * exp(b * (year - 2000))")'
        parsed, fits = self._fit_formula_groups(step, formula, start, between, weights, for_each, search, method,
                                                example)
        rows = []
        num = st._num
        for label, key, fit in fits:
            row = {} if label is None else {for_each: key}
            row.update({"test": "formula fit (nonlinear least squares)", "formula": parsed.text})
            if isinstance(fit, str):
                for name in parsed.names:
                    row.update({name: math.nan, f"standard error of {name}": math.nan, f"t of {name}": math.nan,
                                f"p-value of {name}": math.nan, f"95 % CI low of {name}": math.nan,
                                f"95 % CI high of {name}": math.nan})
                row.update({"residual standard error": math.nan, "degrees of freedom": math.nan,
                            "R squared": math.nan, "AIC": math.nan, "rows": 0, "converged": "no",
                            "method": ""})
                rows.append(row)
                continue
            low, high = fit.intervals
            for j, name in enumerate(fit.names):
                row.update({name: float(fit.values[j]), f"standard error of {name}": float(fit.standard_errors[j]),
                            f"t of {name}": float(fit.t[j]), f"p-value of {name}": float(fit.p_values[j]),
                            f"95 % CI low of {name}": float(low[j]), f"95 % CI high of {name}": float(high[j])})
            row.update({"residual standard error": fit.sigma, "degrees of freedom": float(fit.df),
                        "R squared": fit.r2, "AIC": fit.aic, "rows": fit.n,
                        "converged": "yes" if fit.converged else "no", "method": fit.method_name})
            rows.append(row)

        def values_text(fit):
            low, high = fit.intervals
            return ", ".join(f"{k} = {num(v)} (95 % CI {num(lo)} to {num(hi)})"
                             for k, v, lo, hi in zip(fit.names, fit.values, low, high))

        def quality(fit):
            return (f"Residual standard error {num(fit.sigma)} on {fit.df} degrees of freedom; "
                    f"R squared = {fit.r2:.2f} (only a rough guide for a curve).")

        done = [(label, fit) for label, _, fit in fits if not isinstance(fit, str)]
        if for_each is None:
            fit = done[0][1]
            print(f'Fitted "{parsed.text}" to {fit.n} rows: {values_text(fit)}. {quality(fit)}')
        else:
            print(f'Fitted "{parsed.text}" for each "{for_each}":')
            for label, fit in done[:st.GROUPS_SHOWN]:
                print(f"  {label} ({fit.n} rows): {values_text(fit)}. {quality(fit)}")
            if len(done) > st.GROUPS_SHOWN:
                print(f"  ... and {len(done) - st.GROUPS_SHOWN} more groups (the table has them all)")
        first = done[0][1]
        print(f"Values found: {', '.join(parsed.names)}. Data: {', '.join(parsed.used)}. {first.method_sentence}")
        for label, _, fit in fits:
            where = "" if label is None else f'In group "{label}": '
            if isinstance(fit, str):
                print(f'Note: group "{label}" was not fitted: {fit}')
                continue
            for note in fit.notes:
                print(note if not where else note.replace("Note: ", f"Note: {where}", 1))
        return self._rows_table(rows)

    # ------------------------------------------------------------------ add_predicted_column with a formula
    def _add_formula_column(self, formula, using, name, start, between, weights, for_each):
        """add_predicted_column("y = a * exp(b * x)"): fit (or take using=) and add the formula's value."""
        st = _st()
        step = "add_predicted_column"
        example = ('with sdata.example("growth"): table.add_predicted_column("population = a * exp(b * (year - '
                   '2000))", for_each="country")')
        parsed = read_fit_formula(formula, self._columns, step)
        if name is None:
            name = f"predicted {parsed.left}"
        if not isinstance(name, str) or not name.strip():
            raise SimplydataError(f'{step}() needs the new column\'s name in quotes, for example name="predicted".')
        if name in self._columns:
            raise SimplydataError(f'There is already a column called "{name}". Choose another name with name=.')
        if for_each is not None:
            self._need_name(step, "for_each=", for_each, example)
            self._need_columns(for_each)
        if using is not None:
            values_by_group, how = self._values_from_result(step, parsed, using, for_each, start, between, weights)
        else:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()) as quiet:
                _, fits = self._fit_formula_groups(step, formula, start, between, weights, for_each, "from start",
                                                   None, example)
            skipped = [line for line in quiet.getvalue().splitlines() if line.startswith("Note:")]
            for line in skipped:
                print(line)
            values_by_group = {key: fit.value_dict for _, key, fit in fits if not isinstance(fit, str)}
            for label, _, fit in fits:
                if isinstance(fit, str):
                    print(f'Note: group "{label}" was not fitted, so its rows stay empty: {fit}')
            done = [fit for _, _, fit in fits if not isinstance(fit, str)]
            rows = sum(fit.n for fit in done)
            how = f"fitted to {rows} rows by nonlinear least squares, as R nls()"
            for fit in done:
                for note in fit.notes:
                    print(note)
        # the formula's value for every row with a number in each column it uses
        frame = self._number_frame(step, list(parsed.used))
        data = {c: frame[c].to_numpy().astype(float) for c in parsed.used}
        size = len(frame)
        values = np.full(size, np.nan)
        if for_each is None:
            values = predict_formula(parsed, values_by_group[None], data)
        else:
            keys = frame[for_each].to_list()
            for key, found in values_by_group.items():
                rows_in = np.array([k == key for k in keys], dtype=bool)
                if rows_in.any():
                    values[rows_in] = predict_formula(parsed, found, {c: v[rows_in] for c, v in data.items()})
        out = [float(v) if math.isfinite(v) else None for v in values]
        base = nw.maybe_reset_index(self._frame)
        new = nw.new_series(name, out, nw.Float64, backend=base.implementation)
        empty = sum(v is None for v in out)
        if for_each is None:
            found = ", ".join(f"{k} = {st._num(v)}" for k, v in values_by_group[None].items())
            shown = f"with {found}"
        else:
            shown = f'with values found for each "{for_each}" ({len(values_by_group)} groups)'
        print(f'Added "{name}": "{parsed.text}" {shown} ({how}).'
              + (f" {empty} rows stay empty (an empty cell in {st._list_words(repr(c) for c in parsed.used)}"
                 + (f' or "{for_each}"' if for_each is not None else "") + ")." if empty else "")
              + " Undo with table.undo().")
        return self._change(f'add predicted column "{name}" ({parsed.text})', base.with_columns(new))

    def _values_from_result(self, step, parsed, using, for_each, start, between, weights):
        """{group value or None: {name: value}} from a fit_formula() result table."""
        if start is not None or between is not None or weights is not None:
            raise SimplydataError("using= takes the values from a fit that is already done, so start=, between= "
                                  "and weights= are not needed with it.")
        if not getattr(using, "_is_simplydata_table", False):
            raise SimplydataError(
                f'{step}(..., using=) takes the result of table.fit_formula(...), for example\n'
                'fit = table.fit_formula("population = a * exp(b * (year - 2000))")\n'
                'table.add_predicted_column("population = a * exp(b * (year - 2000))", using=fit)'
            )
        result = using._frame
        missing = [k for k in parsed.names if k not in result.columns]
        if missing:
            raise SimplydataError(f"using= has no column for {_st()._list_words(repr(k) for k in missing)}: give the "
                                  "result of fit_formula() with the same formula.")
        if "formula" in result.columns:
            texts = set(result["formula"].to_list())
            if texts and texts != {parsed.text}:
                raise SimplydataError(f"using= is the fit of {sorted(texts)[0]!r}, not of {parsed.text!r}. "
                                      "Give the same formula as in the fit.")
        records = result.rows(named=True)
        if for_each is None:
            if len(records) != 1:
                raise SimplydataError(f"using= has {len(records)} rows (one per group?). Say which column holds the "
                                      "groups with for_each=.")
            found = {None: {k: float(records[0][k]) for k in parsed.names}}
        else:
            if for_each not in result.columns:
                raise SimplydataError(f'using= has no "{for_each}" column: give the fit_formula(..., '
                                      f'for_each="{for_each}") result.')
            found = {r[for_each]: {k: float(r[k]) for k in parsed.names} for r in records
                     if all(r[k] is not None and math.isfinite(float(r[k])) for k in parsed.names)}
        rows = [r.get("rows") for r in records if r.get("rows") is not None]
        how = "values from using=" + (f", fitted to {int(sum(rows))} rows" if rows else "")
        return found, how
