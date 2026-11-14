"""The Monte Carlo part of compare_shape_to (distributions.shape_test): simulate, refit, compare.

This is the method of SciPy stats.goodness_of_fit(statistic="ad"), done here so that it is fast:

* The simulated samples are drawn exactly as goodness_of_fit draws them: all at once from the
  fitted distribution, frozen.rvs(size=(samples, n), random_state=np.random.default_rng(seed)).
  So the same seed gives the same samples as SciPy, whatever the computer.
* Each sample is fitted again by maximum likelihood, with the same parameters held (loc = 0 for
  gamma, as R). goodness_of_fit refits one sample at a time with SciPy's dist.fit, which for
  Weibull, half-normal, Gumbel, logistic, chi-square, beta and von Mises is a numerical search
  (Nelder-Mead), about 5 ms per sample. Here those families are refitted all at once, one row per
  sample in NumPy arrays: closed forms where they exist, otherwise Newton's method on the exact
  likelihood equations (safeguarded by a bracket, or by halving steps on a concave
  log-likelihood), each row stopping on its own. That gives the exact maximum (as R's uniroot()
  on the score equation), like the fit of the data itself, and is 20 to 1000 times faster. Rows
  that do not settle, and families without such a fit (t, Cauchy, F and other SciPy
  distributions), are refitted one by one with SciPy's dist.fit, as goodness_of_fit does.
* The Anderson-Darling statistic and the p-value use goodness_of_fit's formulas:
  A = -n - sum((2i - 1)/n (log F(x_i) + log S(x_(n+1-i)))) and p = (k + 1)/(samples + 1), with k
  the simulated A at or above the data's A (less a relative tolerance of 100 machine epsilons).

Each row's fit depends only on that row, so the result does not depend on how the rows are
grouped into batches. One process and one thread: parallel workers were measured and do not pay
off (threads are slower because SciPy's search holds the GIL; processes cost seconds to start on
Windows and need the main-module guard in scripts), while the vectorised refits remove the cost.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np

NOTE_SECONDS = 10  # a run expected to take longer than this gets a note before it starts
BATCH_VALUES = 2_000_000  # values refitted at once (rows x n), to keep memory modest
PILOT_ROWS_VECTOR = 500
PILOT_ROWS_ONE_BY_ONE = 10
_TOL = 1e-14  # relative step at which Newton in one parameter stops (the next step is far below rounding)
_TOL_2 = 1e-10  # the same for two parameters: Newton doubles the digits, so the step after is below rounding
_MAX_STEPS = 100


# ====================================================================== solving one equation per row
def _newton_rows(step_of, start, low, high, increasing=True, floor=0.0):
    """Solve g(t) = 0 in every row by Newton's method kept inside a bracket [low, high].

    step_of(t, rows) gives (g, slope) for the rows asked. g must be monotone (increasing or
    decreasing) in t, with one root. Rows that leave the bracket bisect (or double when the
    bracket is still open above). Each row stops on its own, when the step is below _TOL times
    max(|t|, floor) (floor=1 for a parameter on the log scale, which can be near 0). Gives (t, settled).
    """
    t = np.array(start, dtype=float)
    low = np.broadcast_to(np.asarray(low, dtype=float), t.shape).copy()
    high = np.broadcast_to(np.asarray(high, dtype=float), t.shape).copy()
    active = np.isfinite(t)
    settled = np.zeros(t.shape, dtype=bool)
    for _ in range(_MAX_STEPS):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        tr = t[rows]
        g, slope = step_of(tr, rows)
        below = g < 0 if increasing else g > 0
        above = g > 0 if increasing else g < 0
        low[rows] = np.where(below, tr, low[rows])
        high[rows] = np.where(above, tr, high[rows])
        with np.errstate(all="ignore"):
            new = tr - g / slope
        lo, hi = low[rows], high[rows]
        outside = ~((new > lo) & (new < hi)) | ~np.isfinite(new)
        fallback = np.where(np.isfinite(hi), (lo + hi) / 2, 2 * tr)
        new = np.where(outside, fallback, new)
        exact = g == 0
        new = np.where(exact, tr, new)
        done = exact | (np.abs(new - tr) <= _TOL * np.maximum(np.abs(tr), floor))
        bad = ~np.isfinite(new)
        t[rows] = new
        settled[rows[done & ~bad]] = True
        active[rows[done | bad]] = False
    return t, settled & np.isfinite(t)


# ====================================================================== the families refitted at once
def _fit_normal(x):
    return {"loc": x.mean(axis=-1), "scale": x.std(axis=-1)}, None


def _fit_lognormal(x):
    lx = np.log(x)
    return {"s": lx.std(axis=-1), "scale": np.exp(lx.mean(axis=-1))}, None


def _fit_exponential(x):
    return {"scale": x.mean(axis=-1)}, None


def _fit_half_normal(x):
    return {"scale": np.sqrt(np.mean(x * x, axis=-1))}, None


def _fit_inverse_gaussian(x):
    m = x.mean(axis=-1)
    shape = x.shape[-1] / np.sum(1 / x - 1 / m[:, None], axis=-1)
    return {"mu": m / shape, "scale": shape}, None


def _fit_pareto(x):
    scale = x.min(axis=-1)
    with np.errstate(divide="ignore"):
        b = x.shape[-1] / np.sum(np.log(x / scale[:, None]), axis=-1)
    return {"b": b, "scale": scale}, None


def _fit_laplace(x):
    med = np.median(x, axis=-1)
    return {"loc": med, "scale": np.mean(np.abs(x - med[:, None]), axis=-1)}, None


def _fit_uniform(x):
    low = x.min(axis=-1)
    return {"loc": low, "scale": x.max(axis=-1) - low}, None


def _fit_gamma(x):
    """log(a) - digamma(a) = log(mean) - mean(log): Newton in log(a) (the left side falls with a)."""
    from scipy import special

    s = np.log(x.mean(axis=-1)) - np.log(x).mean(axis=-1)
    with np.errstate(all="ignore"):
        start = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)  # Minka's start, close already

    def step_of(u, rows):
        a = np.exp(u)
        return u - special.digamma(a) - s[rows], 1 - a * special.polygamma(1, a)

    u, settled = _newton_rows(step_of, np.log(start), -40.0, 40.0, increasing=False, floor=1.0)
    a = np.exp(u)
    return {"a": a, "scale": x.mean(axis=-1) / a}, settled & (s > 0)


def _fit_weibull(x):
    """The profile likelihood: mean(x^k log x)/mean(x^k) - 1/k - mean(log x) = 0 rises with k.

    The scale is then closed, (mean x^k)^(1/k). Logs are centred and the powers taken relative to
    the largest value, so nothing overflows.
    """
    lx = np.log(x)
    centre = lx.mean(axis=-1, keepdims=True)
    z = lx - centre
    top = z.max(axis=-1, keepdims=True)
    zt = z - top
    spread = z.std(axis=-1)
    with np.errstate(divide="ignore"):
        start = 1.2825 / spread  # pi/sqrt(6): the log of a Weibull value has this spread

    def step_of(k, rows):
        w = np.exp(k[:, None] * zt[rows])
        sw = w.sum(axis=-1)
        zr = z[rows]
        m1 = (w * zr).sum(axis=-1) / sw
        m2 = (w * zr * zr).sum(axis=-1) / sw
        return m1 - 1 / k, m2 - m1 * m1 + 1 / (k * k)

    k, settled = _newton_rows(step_of, start, 0.0, np.inf)
    w = np.exp(k[:, None] * zt)
    scale = np.exp(centre[:, 0] + top[:, 0] + np.log(w.mean(axis=-1)) / k)
    return {"c": k, "scale": scale}, settled & (spread > 0)


def _fit_gumbel(x):
    """Gumbel (largest value): b - mean(z) + sum(z w)/sum(w) = 0 with w = exp(-z/b), rising with b."""
    centre = x.mean(axis=-1, keepdims=True)
    z = x - centre
    zmin = z.min(axis=-1, keepdims=True)
    zs = z - zmin
    sd = z.std(axis=-1)
    zbar = z.mean(axis=-1)

    def step_of(b, rows):
        zr, zsr = z[rows], zs[rows]
        w = np.exp(-zsr / b[:, None])  # at most 1, so it never overflows
        sw = w.sum(axis=-1)
        m1 = (w * zr).sum(axis=-1) / sw
        m2 = (w * zr * zr).sum(axis=-1) / sw
        return b - zbar[rows] + m1, 1 + (m2 - m1 * m1) / (b * b)

    b, settled = _newton_rows(step_of, sd * math.sqrt(6) / math.pi, 0.0, np.inf)
    w = np.exp(-zs / b[:, None])
    loc = centre[:, 0] - b * (np.log(w.mean(axis=-1)) - zmin[:, 0] / b)
    return {"loc": loc, "scale": b}, settled & (sd > 0)


def _fit_chi_square(x):
    """Chi-square degrees of freedom k (loc 0, scale 1): digamma(k/2) = mean(log x) - log 2."""
    from scipy import special

    target = np.log(x).mean(axis=-1) - math.log(2)
    start = np.where(target > -2.22, np.exp(target) + 0.5, -1 / (target + 0.5772156649))  # Minka

    def step_of(u, rows):
        h = np.exp(u)
        return special.digamma(h) - target[rows], h * special.polygamma(1, h)

    u, settled = _newton_rows(step_of, np.log(np.maximum(start, 1e-300)), -700.0, 700.0, floor=1.0)
    return {"df": 2 * np.exp(u)}, settled


def _fit_von_mises(x):
    """Von Mises (scale 1): the mean direction, and I1(k)/I0(k) = mean resultant length, rising with k."""
    from scipy import special

    c, s = np.cos(x).mean(axis=-1), np.sin(x).mean(axis=-1)
    loc = np.arctan2(s, c)
    length = np.hypot(c, s)

    def step_of(k, rows):
        a1 = special.ive(1, k) / special.ive(0, k)
        return a1 - length[rows], 1 - a1 / k - a1 * a1

    start = length * (2 - length * length) / (1 - length * length)  # Banerjee et al.
    with np.errstate(all="ignore"):
        k, settled = _newton_rows(step_of, np.clip(start, 1e-8, 1e12), 0.0, np.inf)
    tiny = length <= 1e-12 / 2  # simplydata's own fit stops at 1e-12 there
    k = np.where(tiny, 1e-12, k)
    return {"kappa": k, "loc": loc}, (settled | tiny) & (length < 1 - 1e-12)


def _newton_concave_2(value_grad_hess, start):
    """Maximise a concave function of two parameters in every row: Newton, halving steps that do worse."""
    p = np.array(start, dtype=float)  # shape (rows, 2)
    active = np.all(np.isfinite(p), axis=1)
    settled = np.zeros(len(p), dtype=bool)
    for _ in range(_MAX_STEPS):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        pr = p[rows]
        f, g, h = value_grad_hess(pr, rows)
        det = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] * h[:, 1, 0]
        with np.errstate(all="ignore"):
            d0 = -(h[:, 1, 1] * g[:, 0] - h[:, 0, 1] * g[:, 1]) / det
            d1 = -(-h[:, 1, 0] * g[:, 0] + h[:, 0, 0] * g[:, 1]) / det
        step = np.stack([d0, d1], axis=1)
        factor = np.ones(len(rows))
        new = pr + step
        for _ in range(60):
            fn = value_grad_hess(new, rows, value_only=True)
            worse = ~(fn >= f - 1e-12 * np.abs(f))
            if not worse.any():
                break
            factor = np.where(worse, factor / 2, factor)
            new = np.where(worse[:, None], pr + factor[:, None] * step, new)
        done = np.all(np.abs(new - pr) <= _TOL_2 * np.maximum(np.abs(pr), 1e-300), axis=1)
        bad = ~np.all(np.isfinite(new), axis=1)
        p[rows] = np.where(bad[:, None], pr, new)
        settled[rows[done & ~bad]] = True
        active[rows[done | bad]] = False
    return p, settled


def _fit_logistic(x):
    """Logistic: the log-likelihood is concave in (1/scale, loc/scale), so Newton finds its maximum."""
    n = x.shape[-1]
    centre = x.mean(axis=-1, keepdims=True)
    spread = x.std(axis=-1, keepdims=True)
    y = (x - centre) / spread  # standardised, so the numbers stay near 1

    def value_grad_hess(p, rows, value_only=False):
        a, b = p[:, 0:1], p[:, 1:2]
        yr = y[rows]
        z = a * yr - b
        # log f(z) = -z - 2 log(1 + exp(-z)) = -|z| - 2 log1p(exp(-|z|))
        value = n * np.log(a[:, 0]) + np.sum(-np.abs(z) - 2 * np.log1p(np.exp(-np.abs(z))), axis=-1)
        value = np.where(a[:, 0] > 0, value, -np.inf)
        if value_only:
            return value
        t = np.tanh(z / 2)  # d log f / dz = -tanh(z/2)
        u = (1 - t * t) / 2  # -d2 log f / dz2
        grad = np.stack([n / a[:, 0] - np.sum(t * yr, axis=-1), np.sum(t, axis=-1)], axis=1)
        h00 = -n / a[:, 0] ** 2 - np.sum(u * yr * yr, axis=-1)
        h01 = np.sum(u * yr, axis=-1)
        h11 = -np.sum(u, axis=-1)
        hess = np.stack([np.stack([h00, h01], axis=1), np.stack([h01, h11], axis=1)], axis=1)
        return value, grad, hess

    start = np.column_stack([np.full(len(x), math.pi / math.sqrt(3)), np.zeros(len(x))])
    p, settled = _newton_concave_2(value_grad_hess, start)
    scale = 1 / p[:, 0]
    return {"loc": centre[:, 0] + spread[:, 0] * p[:, 1] * scale, "scale": spread[:, 0] * scale}, \
        settled & (spread[:, 0] > 0)


def _fit_beta(x):
    """Beta (loc 0, scale 1): concave log-likelihood in (a, b); Newton from the moment estimates."""
    from scipy import special

    lx, l1x = np.log(x).mean(axis=-1), np.log1p(-x).mean(axis=-1)
    m, v = x.mean(axis=-1), x.var(axis=-1)
    common = m * (1 - m) / np.maximum(v, 1e-12) - 1
    start = np.column_stack([np.maximum(m * common, 0.1), np.maximum((1 - m) * common, 0.1)])

    def value_grad_hess(p, rows, value_only=False):
        a, b = p[:, 0], p[:, 1]
        with np.errstate(all="ignore"):
            value = (a - 1) * lx[rows] + (b - 1) * l1x[rows] - special.betaln(a, b)
        value = np.where((a > 0) & (b > 0), value, -np.inf)
        if value_only:
            return value
        dab = special.digamma(a + b)
        grad = np.stack([lx[rows] - special.digamma(a) + dab, l1x[rows] - special.digamma(b) + dab], axis=1)
        tab = special.polygamma(1, a + b)
        h00 = -special.polygamma(1, a) + tab
        h11 = -special.polygamma(1, b) + tab
        hess = np.stack([np.stack([h00, tab], axis=1), np.stack([tab, h11], axis=1)], axis=1)
        return value, grad, hess

    p, settled = _newton_concave_2(value_grad_hess, start)
    return {"a": p[:, 0], "b": p[:, 1]}, settled


# SciPy name -> (the held parameters it expects, the vectorised fit). The fit gives the fitted
# parameters in SciPy's names (arrays, one value per row) and which rows settled (None: all).
VECTOR_FITS = {
    "norm": ({}, _fit_normal),
    "lognorm": ({"loc": 0.0}, _fit_lognormal),
    "expon": ({"loc": 0.0}, _fit_exponential),
    "halfnorm": ({"loc": 0.0}, _fit_half_normal),
    "invgauss": ({"loc": 0.0}, _fit_inverse_gaussian),
    "pareto": ({"loc": 0.0}, _fit_pareto),
    "laplace": ({}, _fit_laplace),
    "uniform": ({}, _fit_uniform),
    "gamma": ({"loc": 0.0}, _fit_gamma),
    "weibull_min": ({"loc": 0.0}, _fit_weibull),
    "gumbel_r": ({}, _fit_gumbel),
    "chi2": ({"loc": 0.0, "scale": 1.0}, _fit_chi_square),
    "vonmises": ({"scale": 1.0}, _fit_von_mises),
    "logistic": ({}, _fit_logistic),
    "beta": ({"loc": 0.0, "scale": 1.0}, _fit_beta),
}


# ====================================================================== the statistic and the p-value
# SciPy's logcdf for these first finds the median by a numerical search for every value (about
# 0.3 ms each, half an hour for 9999 samples of von Mises); log(cdf) and log(sf) agree to rounding.
PLAIN_LOGS = {"vonmises"}


def anderson_darling(frozen, data, plain_logs=False):
    """A for each row of data (the last axis), as SciPy goodness_of_fit's statistic="ad"."""
    xs = np.sort(data, axis=-1)
    n = data.shape[-1]
    i = np.arange(1, n + 1)
    with np.errstate(all="ignore"):
        if plain_logs:
            logs = np.log(frozen.cdf(xs)) + np.log(frozen.sf(xs[..., ::-1]))
        else:
            logs = frozen.logcdf(xs) + frozen.logsf(xs[..., ::-1])
        si = (2 * i - 1) / n * logs
    return -n - np.sum(si, axis=-1)


def p_value(null, observed):
    """(k + 1)/(samples + 1), with k the simulated statistics at or above the observed one."""
    gamma = abs(np.finfo(float).eps * 100 * observed)
    with np.errstate(invalid="ignore"):
        k = np.sum(null >= observed - gamma)
    return float(min(max((k + 1.0) / (len(null) + 1.0), 0.0), 1.0))


# ====================================================================== the run
class MonteCarlo:
    """One shape test's simulation: draw the samples, refit them (a pilot first), give the p-value.

    scipy_dist: the SciPy distribution (not frozen); known: the held parameters; fitted: the fitted
    parameters (SciPy names); x: the data. pilot() refits the first rows and gives the expected
    seconds for the rest, so a note can be printed before the long part; finish() gives
    (statistic, p-value).
    """

    def __init__(self, scipy_dist, known, fitted, x, samples, seed):
        self.dist = scipy_dist
        self.known = dict(known)
        self.fitted = dict(fitted)
        self.x = np.asarray(x, dtype=float)
        self.samples = samples
        self.plain_logs = scipy_dist.name in PLAIN_LOGS
        frozen = scipy_dist(**self.fitted, **self.known)
        self.statistic = float(anderson_darling(frozen, self.x, self.plain_logs))
        rng = np.random.default_rng(seed)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sims = np.asarray(frozen.rvs(size=(samples, len(self.x)), random_state=rng), dtype=float)
        self.null = np.full(samples, np.nan)
        self.done = 0
        vector = VECTOR_FITS.get(scipy_dist.name)
        self.vector = vector[1] if vector is not None and vector[0] == self.known else None
        n = max(len(self.x), 1)
        self.batch = max(1, BATCH_VALUES // n) if self.vector else self.samples

    def _run(self, upto):
        while self.done < upto:
            stop = min(upto, self.done + self.batch)
            rows = self.sims[self.done:stop]
            self.null[self.done:stop] = self._statistics(rows)
            self.done = stop

    def _statistics(self, rows):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.vector is None:  # as goodness_of_fit: refit each row, then A for all rows at once
                params = np.array([self._one_by_one(r) for r in rows])
                frozen = self.dist(*(column[:, None] for column in params.T))
                return np.asarray(anderson_darling(frozen, rows, self.plain_logs), dtype=float)
            params, settled = self.vector(rows)
            frozen = self.dist(**{k: np.asarray(v)[:, None] for k, v in params.items()}, **self.known)
            out = np.asarray(anderson_darling(frozen, rows, self.plain_logs), dtype=float)
            if settled is not None:
                for i in np.flatnonzero(~settled):
                    one = self._one_by_one(rows[i])
                    out[i] = float(anderson_darling(self.dist(*one), rows[i], self.plain_logs))
            return out

    def _one_by_one(self, row):
        """Refit one sample with SciPy's dist.fit, exactly as goodness_of_fit does: all its parameters."""
        shape_names = [] if self.dist.shapes is None else self.dist.shapes.split(", ")
        guessed = dict(self.fitted)
        shapes = [guessed.pop(name) for name in shape_names if name in guessed]
        fixed = {"f" + k: v for k, v in self.known.items()}
        try:
            return np.asarray(self.dist.fit(row, *shapes, **guessed, **fixed), dtype=float)
        except Exception:  # a sample that cannot be fitted counts as not at or above the data
            return np.full(len(shape_names) + 2, np.nan)

    def pilot(self):
        """Refit the first rows; give the expected seconds for the rest."""
        rows = PILOT_ROWS_VECTOR if self.vector else PILOT_ROWS_ONE_BY_ONE
        upto = min(self.samples, rows)
        start = time.perf_counter()
        batch, self.batch = self.batch, upto  # the pilot rows as one batch; rows do not affect each other
        self._run(upto)
        self.batch = batch
        elapsed = time.perf_counter() - start
        return elapsed / upto * (self.samples - upto)

    def finish(self):
        self._run(self.samples)
        return self.statistic, p_value(self.null, self.statistic)


def time_words(seconds) -> str:
    """'about 20 seconds', 'about a minute', 'about 5 minutes', 'about 2 hours'."""
    if seconds < 45:
        return f"about {max(10, int(seconds / 10 + 0.5) * 10)} seconds"
    minutes = seconds / 60
    if minutes < 1.5:
        return "about a minute"
    if minutes < 90:
        return f"about {int(minutes + 0.5)} minutes"
    hours = minutes / 60
    return f"about {int(hours + 0.5)} hours"


def time_note(seconds, samples, groups=1):
    """The plain note printed before a long run, or None when it is quick."""
    if seconds <= NOTE_SECONDS:
        return None
    each = " for each group" if groups > 1 else ""
    return (f"This takes {time_words(seconds)}: {samples} simulated samples are refitted{each}. "
            "Use samples= for a quicker, rougher answer.")
