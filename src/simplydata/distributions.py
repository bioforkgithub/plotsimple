"""Distributions in plain words: chances, percentiles, random numbers and fitting.

    dist = sdata.distribution("normal", average=50, standard_deviation=10)
    dist.chance(above=80)          # 0.00135, printed as "0.135 %", with the R call that gives it
    dist.percentile(95)            # 66.4
    fit = table.fit_distribution("days", "gamma")   # a Distribution with the fitted values
    fit.chance(above=365)

Every word prints one plain sentence with the answer and the R function that gives the same
number (R is exact here, so the call teaches the R name and lets you check). Parameters are R's,
with plain names; R's own names are accepted as other spellings (mean= for average=, sd= for
standard_deviation=, size= for tries=, prob= for chance=).

Choices (the same as R, so results can be checked there):

* Numbers come from the classic scipy.stats distributions (pdf/pmf, cdf, sf, ppf, isf). Upper
  tails use sf and isf, so tiny chances stay accurate, as R's lower.tail = FALSE.
* Counts (discrete distributions): below=3 means 0, 1 or 2 (strict, as keep_rows), at_most=3
  includes 3 (R pbinom(3)); the sentence spells it out ("3 or fewer"). percentile() and
  cut_off() give the smallest count with at least that share at or below it (R qbinom()).
* geometric and negative binomial count the failures before the (first or size-th) success,
  as R; hypergeometric counts the marked items drawn (R dhyper(x, m, n, k)).
* Fitting is maximum likelihood (R MASS::fitdistr, fitdistrplus::fitdist). Families that have
  no location in R (gamma, weibull, lognormal, exponential, beta, ...) keep it at 0, so the
  answers match R. Closed forms are used where they exist (normal, lognormal, exponential,
  poisson, geometric, laplace, pareto, half-normal, inverse gaussian, uniform); gamma, weibull,
  gumbel, negative binomial and von Mises solve their likelihood equations exactly; the rest
  search numerically (Nelder-Mead, then BFGS). Standard errors come from the observed
  information (a numerical Hessian of the log-likelihood at the maximum, as fitdistr's
  optim(hessian = TRUE)), except the textbook closed forms that fitdistr also uses. Note: on
  some data R's fitdistr(x, "gamma") stops a little short of the maximum; simplydata's answer
  is the exact maximum (R's score equation solved with uniroot() gives the same).
* The normal fit's standard deviation divides by n (maximum likelihood), so it is a little
  smaller than R sd(), as fitdistr.
* compare_distributions ranks by AIC, with the Anderson-Darling, Kolmogorov-Smirnov and
  Cramer-von Mises statistics of each fit, as R fitdistrplus::gofstat().
* shape_test (compare_shape_to with any shape but normal): Anderson-Darling with a Monte Carlo
  p-value that allows for the fitting (the method of SciPy goodness_of_fit, seed printed); the
  simulated samples are refitted all at once where the likelihood equations allow it
  (simplydata._shape_monte_carlo).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from ._errors import MissingWordError, SimplydataError, friendly, suggest

REQUIRED = object()
SEED = 1
LIST_SHOWN = 5  # a sentence lists at most this many answers


# ====================================================================== number words
def _num(x) -> str:
    from .table.stat_tests import _num as num

    return num(x)


def _say(x) -> str:
    """A number for a sentence: whole numbers as whole, others with up to 6 significant digits."""
    x = float(x)
    if math.isinf(x):
        return "infinity" if x > 0 else "minus infinity"
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6g}"


def _answer(x) -> str:
    """An answer in a sentence: 3 significant digits (66.4, 0.0447), whole numbers as whole."""
    x = float(x)
    if math.isinf(x):
        return "infinity" if x > 0 else "minus infinity"
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return _num(x)


def _r(x) -> str:
    """A number for R code: exact enough to give the same answer (10 significant digits)."""
    x = float(x)
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.10g}"


def _r_vector(values) -> str:
    values = list(values)
    return _r(values[0]) if len(values) == 1 else "c(" + ", ".join(_r(v) for v in values) + ")"


def _percent(share: float) -> str:
    """0.00135 -> '0.135 %'; never '100 %' for a chance that is not 1."""
    share = float(share)
    if math.isnan(share):
        return "nan"
    value = share * 100
    if value == 0:
        return "0 %"
    if share == 1:
        return "100 %"
    if value >= 99.95:
        return f"{value:.6g} %" if value < 99.99995 else "more than 99.9999 %"
    if value >= 1e-3:
        return f"{value:.3g} %"
    return f"{value:.3g} %".replace("e-0", "e-")


def _words_list(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _shown_list(texts) -> str:
    texts = list(texts)
    if len(texts) > LIST_SHOWN:
        return ", ".join(texts[:LIST_SHOWN]) + f" ... and {len(texts) - LIST_SHOWN} more"
    return _words_list(texts)


def _is_number(value) -> bool:
    import numbers as _nb

    return isinstance(value, _nb.Real) and not isinstance(value, (bool, np.bool_))


def _ordinal(n: float) -> str:
    text = _say(n)
    if not float(n).is_integer():
        return text + "th"
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ====================================================================== the catalogue
@dataclass
class Param:
    """One parameter: its plain name, default, other spellings, R name and what values it takes."""

    name: str
    default: object = REQUIRED
    aliases: tuple = ()
    r: str | None = None
    kind: str = "positive"  # real, positive, non-negative, share, share-open, whole, whole-positive

    @property
    def shown(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class Family:
    name: str  # the plain name
    title: str  # as written in sentences (Poisson, von Mises)
    other_names: tuple
    discrete: bool
    params: list
    scipy: object  # function(params) -> frozen scipy distribution
    r_prefix: str | None = None  # norm -> dnorm, pnorm, qnorm, rnorm
    r_package: str = ""  # "extraDistr::"
    r_fixed: tuple = ()  # extra R arguments, e.g. (("size", "1"),) for bernoulli
    pairs: dict = field(default_factory=dict)  # between=[a, b] -> two parameters
    alternatives: dict = field(default_factory=dict)  # scale= -> rate=1/scale
    describe: object = None  # function(params) -> words after "... distribution"
    short: object = None  # function(params) -> short label
    counts: str = ""  # what a discrete distribution counts
    data: str = "any"  # the values a fit needs: any, positive, non-negative, share, counts, binary, angle
    fit: str | None = None  # how it is fitted: closed, numeric, score, or None (not fittable)
    r_fit: str = ""  # the R call that fits it
    not_fittable: str = ""
    goodness: object = None  # function(params) -> (scipy name, known params, fitted params) for shape_test
    check: object = None  # function(params) -> problem text or None

    def param(self, name):
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)


def _st():
    from scipy import stats

    return stats


def _norm(p):
    return _st().norm(loc=p["average"], scale=p["standard_deviation"])


def _lnorm(p):
    return _st().lognorm(s=p["log_standard_deviation"], scale=math.exp(p["log_average"]))


def _expon(p):
    return _st().expon(scale=1 / p["rate"])


def _gamma(p):
    return _st().gamma(a=p["shape"], scale=1 / p["rate"])


def _beta(p):
    return _st().beta(p["shape1"], p["shape2"])


def _unif(p):
    return _st().uniform(loc=p["lowest"], scale=p["highest"] - p["lowest"])


def _t(p):
    return _st().t(p["degrees_of_freedom"], loc=p.get("location", 0), scale=p.get("scale", 1))


def _chisq(p):
    return _st().chi2(p["degrees_of_freedom"])


def _f(p):
    return _st().f(p["degrees_of_freedom_1"], p["degrees_of_freedom_2"])


def _weibull(p):
    return _st().weibull_min(c=p["shape"], scale=p["scale"])


def _cauchy(p):
    return _st().cauchy(loc=p["location"], scale=p["scale"])


def _logis(p):
    return _st().logistic(loc=p["location"], scale=p["scale"])


def _laplace(p):
    return _st().laplace(loc=p["location"], scale=p["scale"])


def _gumbel(p):
    return _st().gumbel_r(loc=p["location"], scale=p["scale"])


def _pareto(p):
    return _st().pareto(b=p["shape"], scale=p["scale"])


def _triang(p):
    low, high = p["lowest"], p["highest"]
    return _st().triang(c=(p["peak"] - low) / (high - low), loc=low, scale=high - low)


def _halfnorm(p):
    return _st().halfnorm(scale=p["standard_deviation"])


def _invgauss(p):
    return _st().invgauss(mu=p["average"] / p["shape"], scale=p["shape"])


def _vonmises(p):
    return _st().vonmises(kappa=p["concentration"], loc=p["average"])


def _binom(p):
    return _st().binom(int(p["tries"]), p["chance"])


def _bernoulli(p):
    return _st().bernoulli(p["chance"])


def _pois(p):
    return _st().poisson(p["average"])


def _nbinom(p):
    size = p["size"]
    chance = p["chance"] if p.get("chance") is not None else size / (size + p["average"])
    return _st().nbinom(size, chance)


def _geom(p):
    return _st().geom(p["chance"], loc=-1)


def _hyper(p):
    m, n, k = int(p["marked"]), int(p["unmarked"]), int(p["drawn"])
    return _st().hypergeom(M=m + n, n=m, N=k)


def _dunif(p):
    return _st().randint(int(p["lowest"]), int(p["highest"]) + 1)


# --- words for each family
def _two(a, b):
    return f"{a} and {b}"


def _with(p, family):
    parts = [f"{q.shown} {_say(p[q.name])}" for q in family.params if p.get(q.name) is not None]
    return " with " + _words_list(parts) if parts else ""


def _short_default(p, family):
    parts = [f"{q.shown} {_say(p[q.name])}" for q in family.params if p.get(q.name) is not None]
    return ", ".join([family.title] + parts)


def _between_words(p):
    return f" between {_say(p['lowest'])} and {_say(p['highest'])}"


def _t_words(p):
    text = f" with {_say(p['degrees_of_freedom'])} degrees of freedom"
    if p.get("location", 0) != 0 or p.get("scale", 1) != 1:
        text += f", location {_say(p['location'])} and scale {_say(p['scale'])}"
    return text


def _t_short(p):
    text = f"t, {_say(p['degrees_of_freedom'])} degrees of freedom"
    if p.get("location", 0) != 0 or p.get("scale", 1) != 1:
        text += f", location {_say(p['location'])}, scale {_say(p['scale'])}"
    return text


def _nbinom_words(p):
    size = _say(p["size"])
    if p.get("chance") is not None:
        return (f" with size {size} and chance {_say(p['chance'])}: the failures before success number {size}, "
                "as R")
    return f" with size {size} and average {_say(p['average'])}: counts, as R (mu = average)"


def _nbinom_short(p):
    second = f"chance {_say(p['chance'])}" if p.get("chance") is not None else f"average {_say(p['average'])}"
    return f"negative binomial, size {_say(p['size'])}, {second}"


def _check_between(p, what="lowest", other="highest"):
    if not p[what] < p[other]:
        return f"the lowest value ({_say(p[what])}) must be below the highest ({_say(p[other])})"
    return None


def _check_triangle(p):
    problem = _check_between(p)
    if problem:
        return problem
    if not p["lowest"] <= p["peak"] <= p["highest"]:
        return f"the peak ({_say(p['peak'])}) must lie between {_say(p['lowest'])} and {_say(p['highest'])}"
    return None


def _check_whole_between(p):
    if not p["lowest"] <= p["highest"]:
        return f"the lowest value ({_say(p['lowest'])}) cannot be above the highest ({_say(p['highest'])})"
    return None


def _check_hyper(p):
    if p["drawn"] > p["marked"] + p["unmarked"]:
        return (f"drawn ({_say(p['drawn'])}) cannot be more than all the items "
                f"({_say(p['marked'] + p['unmarked'])} = marked + unmarked)")
    return None


def _check_nbinom(p):
    given = [k for k in ("chance", "average") if p.get(k) is not None]
    if len(given) != 1:
        return "it needs size= and one of chance= (prob) or average= (mu), not both and not neither"
    if p.get("chance") is not None and not 0 < p["chance"] <= 1:
        return "chance= must be above 0 and at most 1"
    if p.get("average") is not None and p["average"] < 0:
        return "average= cannot be below 0"
    return None


def _check_vonmises(p):
    return None


# goodness of fit: scipy name, known (held) parameters, and the fitted ones in SciPy's names
def _gof_loc0(name, shapes_of):
    def make(p):
        shapes, scale = shapes_of(p)
        return name, {"loc": 0.0}, {**shapes, "scale": scale}
    return make


_GOODNESS = {
    "lognormal": _gof_loc0("lognorm", lambda p: ({"s": p["log_standard_deviation"]}, math.exp(p["log_average"]))),
    "exponential": _gof_loc0("expon", lambda p: ({}, 1 / p["rate"])),
    "gamma": _gof_loc0("gamma", lambda p: ({"a": p["shape"]}, 1 / p["rate"])),
    "weibull": _gof_loc0("weibull_min", lambda p: ({"c": p["shape"]}, p["scale"])),
    "pareto": _gof_loc0("pareto", lambda p: ({"b": p["shape"]}, p["scale"])),
    "half-normal": _gof_loc0("halfnorm", lambda p: ({}, p["standard_deviation"])),
    "inverse gaussian": _gof_loc0("invgauss", lambda p: ({"mu": p["average"] / p["shape"]}, p["shape"])),
    "beta": lambda p: ("beta", {"loc": 0.0, "scale": 1.0}, {"a": p["shape1"], "b": p["shape2"]}),
    "chi-square": lambda p: ("chi2", {"loc": 0.0, "scale": 1.0}, {"df": p["degrees_of_freedom"]}),
    "F": lambda p: ("f", {"loc": 0.0, "scale": 1.0}, {"dfn": p["degrees_of_freedom_1"],
                                                       "dfd": p["degrees_of_freedom_2"]}),
    "normal": lambda p: ("norm", {}, {"loc": p["average"], "scale": p["standard_deviation"]}),
    "t": lambda p: ("t", {}, {"df": p["degrees_of_freedom"], "loc": p["location"], "scale": p["scale"]}),
    "cauchy": lambda p: ("cauchy", {}, {"loc": p["location"], "scale": p["scale"]}),
    "logistic": lambda p: ("logistic", {}, {"loc": p["location"], "scale": p["scale"]}),
    "laplace": lambda p: ("laplace", {}, {"loc": p["location"], "scale": p["scale"]}),
    "gumbel": lambda p: ("gumbel_r", {}, {"loc": p["location"], "scale": p["scale"]}),
    "uniform": lambda p: ("uniform", {}, {"loc": p["lowest"], "scale": p["highest"] - p["lowest"]}),
    "von mises": lambda p: ("vonmises", {"scale": 1.0}, {"kappa": p["concentration"], "loc": p["average"]}),
}

_LOC_SCALE = [Param("location", 0, (), "location", "real"), Param("scale", 1, (), "scale", "positive")]

FAMILIES = [
    Family("normal", "normal", ("bell", "gaussian", "norm"), False,
           [Param("average", 0, ("mean", "mu"), "mean", "real"),
            Param("standard_deviation", 1, ("sd", "sigma"), "sd", "positive")],
           _norm, "norm", data="any", fit="closed", r_fit='MASS::fitdistr(x, "normal")'),
    Family("lognormal", "lognormal", ("log-normal", "log normal", "lnorm"), False,
           [Param("log_average", 0, ("meanlog",), "meanlog", "real"),
            Param("log_standard_deviation", 1, ("sdlog",), "sdlog", "positive")],
           _lnorm, "lnorm", data="positive", fit="closed", r_fit='MASS::fitdistr(x, "lognormal")'),
    Family("exponential", "exponential", ("exp", "expon"), False,
           [Param("rate", 1, (), "rate", "positive")],
           _expon, "exp", data="non-negative", fit="closed", r_fit='MASS::fitdistr(x, "exponential")'),
    Family("gamma", "gamma", (), False,
           [Param("shape", REQUIRED, ("alpha",), "shape", "positive"), Param("rate", 1, (), "rate", "positive")],
           _gamma, "gamma", alternatives={"scale": "rate"}, data="positive", fit="score",
           r_fit='MASS::fitdistr(x, "gamma")'),
    Family("beta", "beta", (), False,
           [Param("shape1", REQUIRED, ("a", "alpha"), "shape1", "positive"),
            Param("shape2", REQUIRED, ("b",), "shape2", "positive")],
           _beta, "beta", data="share", fit="numeric", r_fit='fitdistrplus::fitdist(x, "beta")'),
    Family("uniform", "uniform", ("unif", "flat", "rectangular"), False,
           [Param("lowest", 0, ("min", "low"), "min", "real"), Param("highest", 1, ("max", "high"), "max", "real")],
           _unif, "unif", pairs={"between": ("lowest", "highest")}, describe=_between_words,
           short=lambda p: f"uniform, between {_say(p['lowest'])} and {_say(p['highest'])}",
           check=_check_between, fit="closed", r_fit='fitdistrplus::fitdist(x, "unif")'),
    Family("t", "t", ("student t", "student", "students t", "student's t"), False,
           [Param("degrees_of_freedom", REQUIRED, ("df",), "df", "positive"),
            Param("location", 0, ("m", "mean"), "location", "real"),
            Param("scale", 1, ("s",), "scale", "positive")],
           _t, "t", describe=_t_words, short=_t_short, fit="numeric", r_fit='MASS::fitdistr(x, "t")'),
    Family("chi-square", "chi-square", ("chi-squared", "chi square", "chi squared", "chisq", "chi2"), False,
           [Param("degrees_of_freedom", REQUIRED, ("df",), "df", "positive")],
           _chisq, "chisq", describe=lambda p: f" with {_say(p['degrees_of_freedom'])} degrees of freedom",
           short=lambda p: f"chi-square, {_say(p['degrees_of_freedom'])} degrees of freedom",
           data="positive", fit="numeric", r_fit='fitdistrplus::fitdist(x, "chisq", start = list(df = mean(x)))'),
    Family("F", "F", ("f distribution", "fisher", "snedecor"), False,
           [Param("degrees_of_freedom_1", REQUIRED, ("df1",), "df1", "positive"),
            Param("degrees_of_freedom_2", REQUIRED, ("df2",), "df2", "positive")],
           _f, "f", pairs={"degrees_of_freedom": ("degrees_of_freedom_1", "degrees_of_freedom_2")},
           describe=lambda p: (f" with {_say(p['degrees_of_freedom_1'])} and {_say(p['degrees_of_freedom_2'])} "
                               "degrees of freedom"),
           short=lambda p: (f"F, {_say(p['degrees_of_freedom_1'])} and {_say(p['degrees_of_freedom_2'])} degrees "
                            "of freedom"),
           data="positive", fit="numeric", r_fit='fitdistrplus::fitdist(x, "f", start = list(df1 = 5, df2 = 10))'),
    Family("weibull", "Weibull", ("weibull_min",), False,
           [Param("shape", REQUIRED, (), "shape", "positive"), Param("scale", 1, (), "scale", "positive")],
           _weibull, "weibull", data="positive", fit="score", r_fit='MASS::fitdistr(x, "weibull")'),
    Family("cauchy", "Cauchy", (), False, list(_LOC_SCALE), _cauchy, "cauchy", fit="numeric",
           r_fit='MASS::fitdistr(x, "cauchy")'),
    Family("logistic", "logistic", ("logis",), False, list(_LOC_SCALE), _logis, "logis", fit="numeric",
           r_fit='MASS::fitdistr(x, "logistic")'),
    Family("laplace", "Laplace", ("double exponential",), False,
           [Param("location", 0, ("mu", "m"), "mu", "real"), Param("scale", 1, ("sigma", "s"), "sigma", "positive")],
           _laplace, "laplace", "extraDistr::", fit="closed",
           r_fit='fitdistrplus::fitdist(x, "laplace") with extraDistr loaded (closed form: location = median, '
                 "scale = average distance from it)"),
    Family("gumbel", "Gumbel", ("extreme value", "gumbel_r", "gumbel r"), False,
           [Param("location", 0, ("mu", "m"), "mu", "real"), Param("scale", 1, ("sigma", "s"), "sigma", "positive")],
           _gumbel, "gumbel", "extraDistr::", fit="score",
           r_fit='fitdistrplus::fitdist(x, "gumbel", start = list(mu = 0, sigma = 1)) with extraDistr loaded'),
    Family("pareto", "Pareto", (), False,
           [Param("shape", REQUIRED, ("a", "alpha"), "a", "positive"),
            Param("scale", 1, ("b", "minimum"), "b", "positive")],
           _pareto, "pareto", "extraDistr::", data="positive", fit="closed",
           r_fit="the closed form (scale = smallest value, shape = n / sum(log(x / scale)))"),
    Family("triangular", "triangular", ("triangle", "triang"), False,
           [Param("lowest", REQUIRED, ("a", "min", "low"), "a", "real"),
            Param("highest", REQUIRED, ("b", "max", "high"), "b", "real"),
            Param("peak", REQUIRED, ("c", "mode"), "c", "real")],
           _triang, "triang", "extraDistr::", pairs={"between": ("lowest", "highest")},
           describe=lambda p: f"{_between_words(p)} with peak {_say(p['peak'])}",
           short=lambda p: f"triangular, between {_say(p['lowest'])} and {_say(p['highest'])}, peak {_say(p['peak'])}",
           check=_check_triangle,
           not_fittable="the ends of a triangular distribution have no regular maximum-likelihood fit; "
                        'give them yourself: sdata.distribution("triangular", between=[0, 10], peak=3)'),
    Family("half-normal", "half-normal", ("half normal", "halfnorm"), False,
           [Param("standard_deviation", 1, ("sd", "sigma"), "sigma", "positive")],
           _halfnorm, "hnorm", "extraDistr::", data="non-negative", fit="closed",
           r_fit="the closed form (sigma = sqrt(mean(x^2)))"),
    Family("inverse gaussian", "inverse Gaussian", ("wald", "invgauss", "inverse normal"), False,
           [Param("average", REQUIRED, ("mean", "mu"), "mean", "positive"),
            Param("shape", REQUIRED, ("lambda",), "shape", "positive")],
           _invgauss, "invgauss", "statmod::", data="positive", fit="closed",
           r_fit="the closed form (average = mean(x), shape = n / sum(1 / x - 1 / mean(x)))"),
    Family("von mises", "von Mises", ("vonmises", "circular normal"), False,
           [Param("average", 0, ("mu", "mean"), "mu", "real"),
            Param("concentration", REQUIRED, ("kappa",), "kappa", "positive")],
           _vonmises, None, describe=lambda p: f"{_with(p, FAMILY['von mises'])} (angles in radians)",
           data="angle", fit="score",
           r_fit="the closed form (average = circular mean; concentration solves A1(kappa) = mean resultant "
                 "length, as R circular::mle.vonmises())"),
    # ---- counts
    Family("binomial", "binomial", ("binom",), True,
           [Param("tries", REQUIRED, ("size", "n", "trials"), "size", "whole"),
            Param("chance", REQUIRED, ("prob", "p", "probability"), "prob", "share")],
           _binom, "binom",
           describe=lambda p: f": successes in {_say(p['tries'])} tries with a chance of {_say(p['chance'])} each",
           short=lambda p: f"binomial, {_say(p['tries'])} tries, chance {_say(p['chance'])}",
           counts="successes", data="counts", fit="closed", r_fit="the closed form (chance = mean(x) / tries)"),
    Family("bernoulli", "Bernoulli", (), True,
           [Param("chance", REQUIRED, ("prob", "p", "probability"), "prob", "share")],
           _bernoulli, "binom", r_fixed=(("size", "1"),),
           describe=lambda p: f": 1 (a success) with chance {_say(p['chance'])}, otherwise 0",
           counts="successes", data="binary", fit="closed", r_fit="the closed form (chance = mean(x))"),
    Family("poisson", "Poisson", ("pois",), True,
           [Param("average", REQUIRED, ("lambda", "mean", "mu", "rate"), "lambda", "non-negative")],
           _pois, "pois", describe=lambda p: f" with average {_say(p['average'])} (counts of events)",
           counts="events", data="counts", fit="closed", r_fit='MASS::fitdistr(x, "poisson")'),
    Family("negative binomial", "negative binomial", ("nbinom", "negbin", "neg binomial"), True,
           [Param("size", REQUIRED, (), "size", "positive"),
            Param("chance", None, ("prob", "p", "probability"), "prob", "share-open"),
            Param("average", None, ("mu", "mean"), "mu", "non-negative")],
           _nbinom, "nbinom", describe=_nbinom_words, short=_nbinom_short, check=_check_nbinom,
           counts="failures", data="counts", fit="score", r_fit='MASS::fitdistr(x, "negative binomial")'),
    Family("geometric", "geometric", ("geom",), True,
           [Param("chance", REQUIRED, ("prob", "p", "probability"), "prob", "share-open")],
           _geom, "geom",
           describe=lambda p: f" with chance {_say(p['chance'])}: failures before the first success, as R",
           counts="failures", data="counts", fit="closed", r_fit='MASS::fitdistr(x, "geometric")'),
    Family("hypergeometric", "hypergeometric", ("hyper", "hypergeom"), True,
           [Param("marked", REQUIRED, ("m",), "m", "whole"), Param("unmarked", REQUIRED, ("n",), "n", "whole"),
            Param("drawn", REQUIRED, ("k",), "k", "whole")],
           _hyper, "hyper",
           describe=lambda p: (f": marked items among {_say(p['drawn'])} drawn without putting back, from "
                               f"{_say(p['marked'])} marked and {_say(p['unmarked'])} unmarked"),
           short=lambda p: (f"hypergeometric, {_say(p['marked'])} marked, {_say(p['unmarked'])} unmarked, "
                            f"{_say(p['drawn'])} drawn"),
           check=_check_hyper, counts="marked items",
           not_fittable="the hypergeometric distribution is set by how the items were drawn, not fitted"),
    Family("whole number uniform", "whole number uniform", ("discrete uniform", "randint", "dice"), True,
           [Param("lowest", 1, ("min", "low"), "min", "integer"), Param("highest", 6, ("max", "high"), "max", "integer")],
           _dunif, "dunif", "extraDistr::", pairs={"between": ("lowest", "highest")},
           describe=lambda p: f": each whole number from {_say(p['lowest'])} to {_say(p['highest'])} equally likely",
           short=lambda p: f"whole number uniform, {_say(p['lowest'])} to {_say(p['highest'])}",
           check=_check_whole_between, counts="values", data="counts-any", fit="closed",
           r_fit="the closed form (lowest = min(x), highest = max(x))"),
]
FAMILY = {f.name: f for f in FAMILIES}
for _family in FAMILIES:
    _family.goodness = _GOODNESS.get(_family.name)


def _key(text: str) -> str:
    return re.sub(r"[\s_\-']+", " ", text.strip().lower())


_LOOKUP = {}
for _family in FAMILIES:
    for _n in (_family.name, *_family.other_names):
        _LOOKUP[_key(_n)] = _family.name
NAMES = [f.name for f in FAMILIES]

# the parameters SciPy or other tools use, pointing to the plain name (for friendly errors)
_SCIPY_PARAM_HINT = {"loc": "the location", "scale": "the scale"}


# ====================================================================== making a Distribution
def _find_family(name, step="distribution"):
    if isinstance(name, Distribution):
        return name
    if not isinstance(name, str) or not name.strip():
        raise SimplydataError(
            f"{step}() needs the name of a distribution in quotes, for example "
            'sdata.distribution("normal", average=50, standard_deviation=10).\n'
            "Names you can use: " + ", ".join(f'"{n}"' for n in NAMES) + ", or any SciPy distribution by its "
            'SciPy name, like "burr12".'
        )
    found = _LOOKUP.get(_key(name))
    if found:
        return FAMILY[found]
    stats = _st()
    scipy_name = name.strip()
    candidate = getattr(stats, scipy_name, None)
    if isinstance(candidate, (stats.rv_continuous, stats.rv_discrete)):
        return scipy_name
    every = list(_LOOKUP)
    guesses = suggest(_key(name), every, n=1)
    lines = [f'.{step}() does not know the distribution "{name}".']
    if guesses:
        lines.append(f'Did you mean "{_LOOKUP[guesses[0]]}"?')
    lines.append("Names you can use: " + ", ".join(f'"{n}"' for n in NAMES) + ", or any SciPy distribution by its "
                 'SciPy name (for example "burr12").')
    raise SimplydataError("\n".join(lines))


def _example(family) -> str:
    parts = []
    for p in family.params:
        if p.name in ("chance",) and family.name == "negative binomial":
            continue
        value = {"real": 0, "positive": 2, "non-negative": 3, "share": 0.3, "share-open": 0.3,
                 "whole": 10, "integer": 1}.get(p.kind, 1)
        if family.name == "hypergeometric":
            value = {"marked": 20, "unmarked": 30, "drawn": 10}[p.name]
        parts.append(f"{p.name}={value}")
    for pair, (low, high) in family.pairs.items():
        parts = [x for x in parts if not x.startswith((low + "=", high + "="))]
        parts.insert(0, f"{pair}=[1, 6]" if family.discrete else f"{pair}=[0, 10]" if pair == "between"
                     else f"{pair}=[3, 20]")
    if family.name == "triangular":
        parts = ["between=[0, 10]", "peak=3"]
    return f'sdata.distribution("{family.name}", ' + ", ".join(parts) + ")"


def _param_list(family) -> str:
    lines = []
    for p in family.params:
        default = "needed" if p.default is REQUIRED else ("optional" if p.default is None else f"default {_say(p.default)}")
        other = f" (R: {p.r})" if p.r and p.r != p.name else ""
        lines.append(f"  {p.name}={other}  {default}")
    for pair, (low, high) in family.pairs.items():
        lines.append(f"  {pair}=[{low}, {high}]  (the same two, as one option)")
    for alt, main in family.alternatives.items():
        lines.append(f"  {alt}=  in place of {main}= ({alt} = 1 / {main})")
    return "\n".join(lines)


def _kind_ok(kind, value) -> str | None:
    if not _is_number(value) or not math.isfinite(value):
        return "a number"
    if kind == "positive" and not value > 0:
        return "a number above 0"
    if kind == "non-negative" and value < 0:
        return "a number of 0 or more"
    if kind == "share" and not 0 <= value <= 1:
        return "a chance from 0 to 1"
    if kind == "share-open" and not 0 < value <= 1:
        return "a chance above 0 and at most 1"
    if kind == "whole" and (value < 0 or not float(value).is_integer()):
        return "a whole number (0 or more)"
    if kind == "integer" and not float(value).is_integer():
        return "a whole number"
    return None


def _read_params(family, given: dict, step="distribution") -> dict:
    """Plain parameters from what was typed (other spellings, pairs and alternatives), checked."""
    out: dict = {}
    names = {p.name for p in family.params}
    spelled = {}
    for p in family.params:
        spelled[p.name] = p.name
        spelled.update({a: p.name for a in p.aliases})
    for key, value in given.items():
        if key in family.pairs:
            low, high = family.pairs[key]
            if (isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2):
                raise SimplydataError(
                    f"{key}= needs two numbers in square brackets, like {key}=[1, 6].\nFor example:  {_example(family)}")
            for name, v in zip((low, high), list(value)):
                if name in out:
                    raise SimplydataError(f"Give {key}= or {name}=, not both.")
                out[name] = v
            continue
        if key in family.alternatives:
            main = family.alternatives[key]
            if main in out or main in given:
                raise SimplydataError(f"Give {main}= or {key}=, not both ({key} = 1 / {main}).")
            problem = _kind_ok("positive", value)
            if problem:
                raise SimplydataError(f"{key}= needs {problem}.\nFor example:  {_example(family)}")
            out[main] = 1 / float(value)
            continue
        name = spelled.get(key)
        if name is None:
            known = list(spelled) + list(family.pairs) + list(family.alternatives)
            lines = [f"The {family.title} distribution has no parameter called {key}=."]
            guess = suggest(key, known, n=1)
            if guess:
                lines.append(f"Did you mean {guess[0]}=?")
            elif key in _SCIPY_PARAM_HINT:
                lines.append(f"{key}= is SciPy's name; simplydata uses the plain (R) parameters below.")
            lines.append(f"Its parameters:\n{_param_list(family)}\nFor example:  {_example(family)}")
            raise SimplydataError("\n".join(lines))
        if name in out:
            raise SimplydataError(f"{key}= gives {name}= a second time: give it once.")
        out[name] = value
    for p in family.params:
        if p.name not in out:
            if p.default is REQUIRED:
                raise SimplydataError(
                    f"The {family.title} distribution needs {p.name}=.\nIts parameters:\n{_param_list(family)}\n"
                    f"For example:  {_example(family)}")
            out[p.name] = p.default
    for p in family.params:
        value = out[p.name]
        if value is None:
            continue
        problem = _kind_ok(p.kind, value)
        if problem:
            raise SimplydataError(f"{p.name}= needs {problem}, not {value!r}.\nFor example:  {_example(family)}")
        out[p.name] = float(value)
    if family.check:
        problem = family.check(out)
        if problem:
            raise SimplydataError(f"For a {family.title} distribution {problem}.\nFor example:  {_example(family)}")
    del names
    return {p.name: out[p.name] for p in family.params}


def _scipy_params(scipy_name, given):
    stats = _st()
    dist = getattr(stats, scipy_name)
    shapes = [s.strip() for s in (dist.shapes or "").split(",") if s.strip()]
    allowed = shapes + ["loc"] + ([] if isinstance(dist, stats.rv_discrete) else ["scale"])
    example = f'sdata.distribution("{scipy_name}", ' + ", ".join(f"{s}=2" for s in shapes) + ")"
    unknown = [k for k in given if k not in allowed]
    if unknown:
        raise SimplydataError(
            f"SciPy's {scipy_name} has no parameter called {unknown[0]}=. Its parameters (SciPy's names): "
            + ", ".join(f"{a}=" for a in allowed) + f".\nFor example:  {example}")
    missing = [s for s in shapes if s not in given]
    if missing:
        raise SimplydataError(
            f"SciPy's {scipy_name} needs {', '.join(m + '=' for m in missing)} (its shape parameters; loc= and "
            f"scale= are optional).\nFor example:  {example}")
    for k, v in given.items():
        if not _is_number(v) or not math.isfinite(v):
            raise SimplydataError(f"{k}= needs a number.\nFor example:  {example}")
    params = {s: float(given[s]) for s in shapes}
    params["loc"] = float(given.get("loc", 0))
    if "scale" in allowed:
        params["scale"] = float(given.get("scale", 1))
    frozen = _scipy_generic(scipy_name, params)
    args = [params[s] for s in shapes]
    ok = bool(np.all(dist._argcheck(*args))) and params.get("scale", 1) > 0
    if not ok:
        raise SimplydataError(f"These values are not allowed for SciPy's {scipy_name}: "
                              + ", ".join(f"{k}={_say(v)}" for k, v in params.items()) + ".")
    del frozen
    return params


def _scipy_generic(scipy_name, params):
    dist = getattr(_st(), scipy_name)
    return dist(**params)


@friendly
def distribution(name=None, *values, **parameters):
    """Make a distribution to ask chances, percentiles and random numbers of.

    dist = sdata.distribution("normal", average=50, standard_deviation=10)
    then dist.chance(above=80), dist.percentile(95), dist.random_numbers(how_many=100, seed=1).
    Names: normal, lognormal, exponential, gamma, beta, uniform, t, chi-square, F, weibull,
    cauchy, logistic, laplace, gumbel, pareto, triangular, half-normal, inverse gaussian,
    von mises; for counts binomial, bernoulli, poisson, negative binomial, geometric,
    hypergeometric, whole number uniform. Parameters are R's with plain names; R's names
    (mean=, sd=, prob=, size=, df=, ...) work too. Any other SciPy distribution by its SciPy
    name and SciPy's parameters: sdata.distribution("burr12", c=2, d=3).
    """
    family = _find_family(name)
    if isinstance(family, Distribution):
        return family
    if values:
        example = _example(family) if isinstance(family, Family) else f'sdata.distribution("{family}", c=2)'
        raise SimplydataError(f"Write the parameters with their names.\nFor example:  {example}")
    if isinstance(family, str):
        return Distribution(family, _scipy_params(family, parameters))
    return Distribution(family, _read_params(family, parameters))


# ====================================================================== the Distribution
HINTS = {"pdf": "density_at", "pmf": "chance", "cdf": "chance", "sf": "chance", "ppf": "percentile",
         "quantile": "percentile", "isf": "cut_off", "critical_value": "cut_off", "rvs": "random_numbers",
         "sample": "random_numbers", "mean": "average", "std": "standard_deviation", "sd": "standard_deviation",
         "var": "variance", "interval": "middle_range", "stats": "summary", "describe": "summary",
         "probability": "chance", "density": "density_at", "plot": "chart", "show": "summary", "fit": "fit_distribution",
         "dnorm": "density_at", "pnorm": "chance", "qnorm": "percentile", "rnorm": "random_numbers",
         "logpdf": "density_at", "logcdf": "chance", "moment": "summary", "entropy": "summary",
         "support": "summary", "params": "parameters", "args": "parameters", "kwds": "parameters"}
# How the word a HINT points to is written, with a typical value.
HINT_EXAMPLES = {"density_at": "dist.density_at(60)", "chance": "dist.chance(below=60) (or at_most=, above=, "
                 "at_least=, between=; for counts dist.chance(3) is exactly 3)",
                 "percentile": "dist.percentile(95)", "cut_off": "dist.cut_off(above=0.025)",
                 "random_numbers": "dist.random_numbers(how_many=100, seed=1)", "middle_range": "dist.middle_range(95)",
                 "summary": "dist.summary()", "average": "dist.average()",
                 "standard_deviation": "dist.standard_deviation()", "variance": "dist.variance()",
                 "chart": "dist.chart()", "parameters": "dist.parameters()"}


class Distribution:
    """A distribution with its parameters. Make one with sdata.distribution("normal", average=50, standard_deviation=10).

    Words: chance, density_at, percentile, cut_off, middle_range, average, median,
    standard_deviation, variance, summary, random_numbers, chart, table, parameters.
    """

    _OLD_WORDS: dict = {}  # no old names yet (sdata.help reads this, as for tables and charts)

    @classmethod
    def _word_names(cls) -> list[str]:
        return sorted(WORDS)

    def __init__(self, family, params, fit_info=None, backend=None):
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_params", {k: (None if v is None else float(v)) for k, v in params.items()})
        object.__setattr__(self, "_fit_info", fit_info)
        object.__setattr__(self, "_backend", backend)
        if isinstance(family, Family):
            frozen = family.scipy(self._params)
        else:
            frozen = _scipy_generic(family, self._params)
        object.__setattr__(self, "_frozen", frozen)

    def __setattr__(self, name, value):
        raise SimplydataError("A distribution cannot be changed. Make a new one with sdata.distribution(...).")

    # ------------------------------------------------------------------ for other parts of simplydata
    @property
    def is_discrete(self) -> bool:
        if isinstance(self._family, Family):
            return self._family.discrete
        return isinstance(self._frozen.dist, _st().rv_discrete)

    @property
    def name(self) -> str:
        return self._family.name if isinstance(self._family, Family) else self._family

    @property
    def label(self) -> str:
        """A short label for sentences and legends: 'normal, average 50, standard deviation 10'."""
        family = self._family
        if not isinstance(family, Family):
            return f"SciPy {family}, {self._param_text()}"
        if family.short:
            return family.short(self._params)
        return _short_default(self._params, family)

    @property
    def r_call(self) -> str:
        """The R density (or chance for counts) at x: 'dnorm(x, mean = 50, sd = 10)'."""
        return self._r("d", "x") or "no R function (a SciPy distribution)"

    # pdf, pmf and cdf are for charts (words/curves.py), so they start with _: dist.pdf() is an
    # other name that points to density_at(), never a word that runs (see HINTS).
    def _pdf(self, x):
        """Height of the density curve (continuous); for people the word is density_at()."""
        with np.errstate(all="ignore"):
            return self._frozen.pdf(np.asarray(x, dtype=float))

    def _pmf(self, k):
        """Chance of exactly k (counts); for people the word is chance(k)."""
        with np.errstate(all="ignore"):
            return self._frozen.pmf(np.asarray(k, dtype=float))

    def _cdf(self, x):
        """Chance of a value at or below x; for people the word is chance(at_most=x)."""
        with np.errstate(all="ignore"):
            return self._frozen.cdf(np.asarray(x, dtype=float))

    # ------------------------------------------------------------------ words for sentences
    def _describe(self) -> str:
        family = self._family
        if not isinstance(family, Family):
            return f"SciPy's {family} with {self._param_text()}"
        words = family.describe(self._params) if family.describe else _with(self._params, family)
        return f"{family.title} distribution{words}"

    def _r_args(self) -> str:
        family = self._family
        parts = []
        for p in family.params:
            value = self._params.get(p.name)
            if value is None:
                continue
            if family.name == "t" and p.name in ("location", "scale"):
                continue
            parts.append(f"{p.r} = {_r(value)}")
        parts += [f"{k} = {v}" for k, v in family.r_fixed]
        return ", ".join(parts)

    def _r(self, kind, x, upper=False) -> str | None:
        """R code for d (density), p (chance), q (percentile) or r (random): exact, so it can be checked."""
        family = self._family
        if not isinstance(family, Family):
            return None
        tail = ", lower.tail = FALSE" if upper and kind in ("p", "q") else ""
        p = self._params
        if family.name == "von mises":
            mu, kappa = _r(p["average"]), _r(p["concentration"])
            density = f"exp({kappa} * cos(t - {mu})) / (2 * pi * besselI({kappa}, 0))"
            if kind == "d":
                return density.replace("t - ", f"{x} - ") if x == "x" else f"(function(t) {density})({x})"
            many = x.startswith("c(")
            if kind == "p":
                area = f"integrate(function(t) {density}, {mu} - pi, {'v' if many else x})$value"
                area = f"1 - {area}" if upper else area
                return f"sapply({x}, function(v) {area})" if many else area
            if kind == "q":
                share = "s" if many else x
                share = f"(1 - {share})" if upper else share
                root = (f"uniroot(function(q) integrate(function(t) {density}, {mu} - pi, q)$value - {share}, "
                        f"c({mu} - pi, {mu} + pi), tol = 1e-12)$root")
                return f"sapply({x}, function(s) {root})" if many else root
            return None
        if family.name == "t" and (p.get("location", 0) != 0 or p.get("scale", 1) != 1):
            m, s, df = _r(p["location"]), _r(p["scale"]), _r(p["degrees_of_freedom"])
            if kind == "d":
                return f"dt(({x} - {m}) / {s}, df = {df}) / {s}"
            if kind == "p":
                return f"pt(({x} - {m}) / {s}, df = {df}{tail})"
            if kind == "q":
                return f"{m} + {s} * qt({x}, df = {df}{tail})"
            return f"{m} + {s} * rt({x}, df = {df})"
        return f"{family.r_package}{kind}{family.r_prefix}({x}, {self._r_args()}{tail})"

    def _with_r(self, code) -> str:
        return f"{self.label}; R: {code}" if code else self.label

    def __repr__(self):
        text = self._describe()
        if not text.startswith("t "):
            text = text[0].upper() + text[1:]
        code = self._r("d", "x")
        lines = [f"{text}" + (f" (R: {code})." if code else ".")]
        info = self._fit_info
        if info:
            lines.append(f"Fitted by maximum likelihood to {info['rows']} values"
                         + (f' of "{info["column"]}"' if info.get("column") else "")
                         + f": log-likelihood {info['log-likelihood']:.4g}, AIC {info['AIC']:.6g}. "
                           "fit.table() gives the standard errors.")
        return "\n".join(lines)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        known = [w for w in WORDS]
        target = HINTS.get(name)
        if target:
            if target == "fit_distribution":
                raise MissingWordError(
                    f"A distribution has no step called .{name}(). To fit one to data use "
                    'table.fit_distribution("column", "gamma") or nums.fit_distribution("gamma").')
            example = HINT_EXAMPLES.get(target, f"dist.{target}()")
            raise MissingWordError(f"A distribution has no step called .{name}().\n"
                                   f"In simplydata this is {example}.\n"
                                   "Every step for a distribution: " + ", ".join(known))
        lines = [f"A distribution has no step called .{name}()."]
        guesses = suggest(name, known)
        if guesses:
            lines.append("Did you mean " + " or ".join(f".{g}()" for g in guesses) + "?")
        lines.append("Every step for a distribution: " + ", ".join(known))
        raise MissingWordError("\n".join(lines))

    def __dir__(self):
        return sorted(set(WORDS) | {"is_discrete", "label", "r_call", "name"})

    # ------------------------------------------------------------------ input checks
    def _numbers_in(self, step, option, value, example):
        """One number -> (array, False); a list -> (array, True)."""
        from .numbers._base import BaseNumbers

        if isinstance(value, BaseNumbers):
            value = value._values
        if _is_number(value):
            if not math.isfinite(value) and not math.isinf(value):
                raise SimplydataError(f"{step}() needs {option} as a number.\nFor example:  {example}")
            return np.asarray([float(value)]), False
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0 and all(
                _is_number(v) and not math.isnan(v) for v in np.asarray(value, dtype=object).ravel()):
            return np.asarray(value, dtype=float).ravel(), True
        if isinstance(value, str):
            raise SimplydataError(f"{step}() needs {option} as a number without quotes.\nFor example:  {example}")
        raise SimplydataError(f"{step}() needs {option} as a number (or a list of numbers).\nFor example:  {example}")

    def _give(self, values, many):
        if many:
            from .numbers.numbers import Numbers

            return Numbers(np.asarray(values, dtype=float))
        return float(values[0])

    # ------------------------------------------------------------------ chance
    @friendly
    def chance(self, value=None, above=None, below=None, at_most=None, at_least=None, between=None):
        """The chance of a value in a range, as a share from 0 to 1 (printed as a percent).

        dist.chance(above=80)   dist.chance(below=60)   dist.chance(between=[40, 60])
        at_most= and at_least= include the number itself; above= and below= do not (as
        keep_rows). For counts, dist.chance(3) is the chance of exactly 3; below=3 means 0, 1
        or 2 and at_most=3 means 3 or fewer (R pbinom(3)). One lower option (above=, at_least=)
        and one upper option (below=, at_most=) can go together. A list gives Numbers.
        R: pnorm(q, lower.tail = ), dbinom.
        """
        step = "chance"
        example = "dist.chance(above=80)   or   dist.chance(between=[40, 60])"
        given = {k: v for k, v in (("above", above), ("below", below), ("at_most", at_most),
                                   ("at_least", at_least), ("between", between)) if v is not None}
        if value is not None:
            if given:
                raise SimplydataError("Give one number for the chance of exactly that value, or above=/below=/"
                                      "at_most=/at_least=/between=, not both.")
            return self._chance_exactly(value)
        if not given:
            raise SimplydataError(f"chance() needs a range: above=, below=, at_most=, at_least= or between=.\n"
                                  f"For example:  {example}")
        if "between" in given:
            if len(given) > 1:
                raise SimplydataError("Give between=[low, high] on its own (it includes both ends), "
                                      "or above=/at_least= with below=/at_most=.")
            b = between
            from .numbers._base import BaseNumbers

            if isinstance(b, BaseNumbers):
                b = list(b._values)
            if (isinstance(b, (str, bytes)) or not hasattr(b, "__len__") or len(b) != 2
                    or not all(_is_number(v) and not math.isnan(v) for v in b)):
                raise SimplydataError("between= needs two numbers in square brackets: the lowest and the highest.\n"
                                      "For example:  dist.chance(between=[40, 60])")
            low, high = sorted(float(v) for v in b)
            return self._chance_range(("at_least", low), ("at_most", high), many=False)
        lowers = [k for k in given if k in ("above", "at_least")]
        uppers = [k for k in given if k in ("below", "at_most")]
        if len(lowers) > 1 or len(uppers) > 1:
            raise SimplydataError("Give at most one of above= and at_least=, and at most one of below= and at_most=.\n"
                                  f"For example:  {example}")
        arrays = {k: self._numbers_in(step, f"{k}=", v, example) for k, v in given.items()}
        many = any(m for _, m in arrays.values())
        if lowers and uppers:
            if many:
                raise SimplydataError("A lower and an upper limit together take one number each; for several "
                                      "ranges ask once per range.")
            return self._chance_range((lowers[0], arrays[lowers[0]][0][0]), (uppers[0], arrays[uppers[0]][0][0]),
                                      many=False)
        option = (lowers or uppers)[0]
        return self._chance_one_side(option, arrays[option][0], many)

    def _check_angles(self, xs):
        if self.name == "von mises":
            mu = self._params["average"]
            if np.any((xs < mu - math.pi - 1e-12) | (xs > mu + math.pi + 1e-12)):
                raise SimplydataError(
                    f"The von Mises distribution here covers the angles from {_say(mu - math.pi)} to "
                    f"{_say(mu + math.pi)} (average minus pi to average plus pi); give values in that range.")

    def _chance_one_side(self, option, xs, many):
        self._check_angles(xs[np.isfinite(xs)])
        f = self._frozen
        discrete = self.is_discrete
        upper = option in ("above", "at_least")
        if discrete:
            if option == "below":
                q = np.ceil(xs) - 1
            elif option == "at_most":
                q = np.floor(xs)
            elif option == "above":
                q = np.floor(xs)
            else:
                q = np.ceil(xs) - 1
        else:
            q = xs
        with np.errstate(all="ignore"):
            share = f.sf(q) if upper else f.cdf(q)
        share = np.clip(np.asarray(share, dtype=float), 0, 1)
        code = self._r("p", _r_vector(q), upper=upper)
        if many:
            lead = ({"above": "more than", "below": "fewer than", "at_most": "at most", "at_least": "at least"}
                    if discrete else {"above": "a value above", "below": "a value below", "at_most": "a value of at most",
                                      "at_least": "a value of at least"})[option]
            print(f"The chances of {lead} {_shown_list(_say(x) for x in xs)} are "
                  f"{_shown_list(_percent(s) for s in share)} ({self._with_r(code)}).")
        else:
            print(f"The chance of {self._side_words(option, xs[0], discrete)} is {_percent(share[0])} "
                  f"({self._with_r(code)}).")
        return self._give(share, many)

    def _side_words(self, option, x, discrete):
        x_text = _say(x)
        if not discrete:
            return {"above": f"a value above {x_text}", "below": f"a value below {x_text}",
                    "at_most": f"a value of {x_text} or less", "at_least": f"a value of {x_text} or more"}[option]
        if option == "at_most":
            return f"{_say(math.floor(x))} or fewer" if float(x).is_integer() else \
                f"{x_text} or fewer ({_say(math.floor(x))} or fewer)"
        if option == "below":
            return f"fewer than {x_text} ({_say(math.ceil(x) - 1)} or fewer)"
        if option == "above":
            return f"more than {x_text} ({_say(math.floor(x) + 1)} or more)"
        return f"{_say(math.ceil(x))} or more" if float(x).is_integer() else \
            f"{x_text} or more ({_say(math.ceil(x))} or more)"

    def _chance_range(self, low, high, many):
        (low_option, a), (high_option, b) = low, high
        f = self._frozen
        self._check_angles(np.asarray([v for v in (a, b) if math.isfinite(v)]))
        if self.is_discrete:
            first = math.floor(a) + 1 if low_option == "above" else math.ceil(a)
            last = math.ceil(b) - 1 if high_option == "below" else math.floor(b)
            if first > last:
                share = 0.0
            else:
                with np.errstate(all="ignore"):
                    share = float(f.cdf(last) - f.cdf(first - 1))
                    if f.cdf(first - 1) > 0.5:
                        share = float(f.sf(first - 1) - f.sf(last))
            share = min(max(share, 0.0), 1.0)
            words = f"{_say(first)} to {_say(last)} (both included)" if first != last else f"exactly {_say(first)}"
            if first > last:
                words = f"a count {('above ' + _say(a)) if low_option == 'above' else ('from ' + _say(a))} and " \
                        f"{('below ' + _say(b)) if high_option == 'below' else ('up to ' + _say(b))}"
            name = self._family.r_prefix if isinstance(self._family, Family) else None
            code = None
            if name:
                code = f"{self._r('p', _r(last))} - {self._r('p', _r(first - 1))}"
            print(f"The chance of {words} is {_percent(share)} ({self._with_r(code)}).")
            return share
        if b < a:
            share = 0.0
        else:
            with np.errstate(all="ignore"):
                share = float(f.cdf(b) - f.cdf(a))
                if f.cdf(a) > 0.5:
                    share = float(f.sf(a) - f.sf(b))
        share = min(max(share, 0.0), 1.0)
        code = None
        if self._r("p", "q"):
            code = f"{self._r('p', _r(b))} - {self._r('p', _r(a))}"
        print(f"The chance of a value between {_say(a)} and {_say(b)} is {_percent(share)} ({self._with_r(code)}).")
        return share

    def _chance_exactly(self, value):
        example = "dist.chance(3)"
        if not self.is_discrete:
            raise SimplydataError(
                "For a smooth (continuous) distribution the chance of exactly one value is 0. Ask for a range, "
                "like dist.chance(between=[59.5, 60.5]), or the height of the curve with dist.density_at(60).")
        xs, many = self._numbers_in("chance", "the value", value, example)
        with np.errstate(all="ignore"):
            share = np.clip(self._frozen.pmf(xs), 0, 1)
        code = self._r("d", _r_vector(xs))
        if many:
            print(f"The chances of exactly {_shown_list(_say(x) for x in xs)} are "
                  f"{_shown_list(_percent(s) for s in share)} ({self._with_r(code)}).")
        else:
            print(f"The chance of exactly {_say(xs[0])} is {_percent(share[0])} ({self._with_r(code)}).")
        return self._give(share, many)

    # ------------------------------------------------------------------ density
    @friendly
    def density_at(self, value=None):
        """The height of the density curve at a value (continuous distributions).

        dist.density_at(60)   (R dnorm(60, mean = 50, sd = 10)). A list gives Numbers.
        It is not a chance: a chance is an area under the curve (dist.chance(between=[...])).
        """
        if self.is_discrete:
            raise SimplydataError(
                "A count distribution has no density curve: use dist.chance(3) for the chance of exactly 3.")
        if value is None:
            raise SimplydataError("density_at() needs a value.\nFor example:  dist.density_at(60)")
        xs, many = self._numbers_in("density_at", "the value", value, "dist.density_at(60)")
        with np.errstate(all="ignore"):
            heights = np.asarray(self._frozen.pdf(xs), dtype=float)
        code = self._r("d", _r_vector(xs))
        if many:
            print(f"The heights of the density curve at {_shown_list(_say(x) for x in xs)} are "
                  f"{_shown_list(_answer(h) for h in heights)} ({self._with_r(code)}).")
        else:
            print(f"The height of the density curve at {_say(xs[0])} is {_answer(heights[0])} "
                  f"({self._with_r(code)}); it is not a chance, a chance is an area under the curve.")
        return self._give(heights, many)

    # ------------------------------------------------------------------ percentiles and cut-offs
    def _quantile(self, shares, upper):
        f = self._frozen
        with np.errstate(all="ignore"):
            if self.is_discrete:
                return np.asarray(f.isf(shares) if upper else f.ppf(shares), dtype=float)
            out = np.where(shares > 0.5, f.isf(1 - shares), f.ppf(shares)) if not upper else f.isf(shares)
        return np.asarray(out, dtype=float)

    @friendly
    def percentile(self, percent=None):
        """The value that this percent of the distribution is below: percentile(95).

        dist.percentile(95)   (R qnorm(0.95, mean = 50, sd = 10)). percentile(50) is the median.
        For counts: the smallest count with at least that percent at or below it (R qbinom).
        A list of percents gives Numbers.
        """
        example = "dist.percentile(95)"
        if percent is None:
            raise SimplydataError(f"percentile() needs a percent from 0 to 100.\nFor example:  {example}")
        ps, many = self._numbers_in("percentile", "a percent", percent, example)
        if np.any((ps < 0) | (ps > 100)):
            raise SimplydataError("percentile() needs a percent from 0 to 100: percentile(95) is the value 95 % of "
                                  f"the distribution is below.\nFor example:  {example}")
        shares = ps / 100
        values = self._quantile(shares, upper=False)
        code = self._r("q", _r_vector(shares))
        if many:
            kind = ", each the smallest count with at least that share at or below it" if self.is_discrete else ""
            print(f"The {_shown_list(_ordinal(p) for p in ps)} percentiles are "
                  f"{_shown_list(_answer(v) for v in values)}{kind} ({self._with_r(code)}).")
        elif self.is_discrete:
            print(f"{_say(ps[0])} % of the distribution is at or below {_answer(values[0])}, the smallest count with "
                  f"at least that share ({self._with_r(code)}).")
        else:
            print(f"{_say(ps[0])} % of the distribution is below {_answer(values[0])} ({self._with_r(code)}).")
        return self._give(values, many)

    @friendly
    def cut_off(self, above=None, below=None):
        """The value with this chance above it (or below it): critical values.

        dist.cut_off(above=0.025)   the value with a 2.5 % chance above it (R qnorm(0.025, lower.tail = FALSE))
        dist.cut_off(below=0.05)    the value with a 5 % chance below it
        Chances are shares from 0 to 1. A list gives Numbers.
        """
        example = "dist.cut_off(above=0.025)"
        if (above is None) == (below is None):
            raise SimplydataError(f"cut_off() needs above= or below= (a chance from 0 to 1), one of them.\n"
                                  f"For example:  {example}")
        option = "above" if above is not None else "below"
        shares, many = self._numbers_in("cut_off", f"{option}=", above if above is not None else below, example)
        if np.any((shares < 0) | (shares > 1)):
            raise SimplydataError(
                f"cut_off() takes a chance from 0 to 1, like {option}=0.025 for 2.5 %. For a percentile from 0 to "
                f"100, use dist.percentile(97.5).")
        upper = option == "above"
        values = self._quantile(shares, upper=upper)
        code = self._r("q", _r_vector(shares), upper=upper)
        if self.is_discrete:
            f = self._frozen
            with np.errstate(all="ignore"):
                actual = f.sf(values) if upper else f.cdf(values)
            side = "more than" if upper else "at most"
            texts = [f"{_answer(v)} (the chance of {side} it is {_percent(a)})" for v, a in zip(values, actual)]
            rule = ("the smallest count with a chance of at most that share above it" if upper else
                    "the smallest count with at least that share at or below it")
            print(f"The cut-off for a {_shown_list(_percent(s) for s in shares)} chance {option} is "
                  f"{_shown_list(texts)}: {rule} ({self._with_r(code)}).")
        else:
            print(f"The cut-off with a {_shown_list(_percent(s) for s in shares)} chance {option} it is "
                  f"{_shown_list(_answer(v) for v in values)} ({self._with_r(code)}).")
        return self._give(values, many)

    @friendly
    def middle_range(self, percent=95):
        """The range that holds the middle percent of the distribution, as Numbers [low, high].

        dist.middle_range(95)   (R qnorm(c(0.025, 0.975), mean = 50, sd = 10))
        """
        example = "dist.middle_range(95)"
        if not _is_number(percent) or not 0 < percent < 100:
            raise SimplydataError(f"middle_range() needs a percent above 0 and below 100.\nFor example:  {example}")
        tail = (1 - percent / 100) / 2
        low = self._quantile(np.asarray([tail]), upper=False)[0]
        high = self._quantile(np.asarray([1 - tail]), upper=False)[0]
        code = self._r("q", _r_vector([tail, 1 - tail]))
        extra = ", holding at least that share" if self.is_discrete else ""
        print(f"The middle {_say(percent)} % of the distribution lies between {_answer(low)} and {_answer(high)}"
              f"{extra} ({self._with_r(code)}).")
        from .numbers.numbers import Numbers

        return Numbers(np.asarray([low, high], dtype=float))

    # ------------------------------------------------------------------ moments
    def _moment(self, what, value, formula=None):
        if math.isnan(value) or math.isinf(value):
            print(f"The distribution ({self.label}) has no finite {what} (its tails are too heavy).")
            return float(value)
        print(f"The {what} of the distribution is {_answer(value)} ({self.label}).")
        return float(value)

    @friendly
    def average(self):
        """The average (mean) of the distribution. dist.average()"""
        with np.errstate(all="ignore"):
            return self._moment("average", float(self._frozen.mean()))

    @friendly
    def median(self):
        """The median: half of the distribution is below it. dist.median()"""
        with np.errstate(all="ignore"):
            return self._moment("median", float(self._frozen.median()))

    @friendly
    def standard_deviation(self):
        """The standard deviation of the distribution. dist.standard_deviation()"""
        with np.errstate(all="ignore"):
            return self._moment("standard deviation", float(self._frozen.std()))

    @friendly
    def variance(self):
        """The variance (the standard deviation squared). dist.variance()"""
        with np.errstate(all="ignore"):
            return self._moment("variance", float(self._frozen.var()))

    def _summary_row(self) -> dict:
        f = self._frozen
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, var, skew, kurt = (float(v) for v in f.stats(moments="mvsk"))
            low, high = (float(v) for v in f.support())
            try:
                entropy = float(f.entropy())
            except Exception:  # noqa: BLE001 - SciPy cannot always integrate it
                entropy = math.nan
            median = float(f.median())
        return {"distribution": self.name, "parameters": self._param_text(), "average": mean, "median": median,
                "standard deviation": math.sqrt(var) if var >= 0 else math.nan, "variance": var,
                "skewness": skew, "excess kurtosis": kurt, "entropy": entropy,
                "lowest possible": low, "highest possible": high}

    def _param_text(self) -> str:
        if isinstance(self._family, Family):
            return ", ".join(f"{p.shown} {_say(self._params[p.name])}" for p in self._family.params
                             if self._params.get(p.name) is not None)
        return ", ".join(f"{k}={_say(v)}" for k, v in self._params.items()
                         if not (k == "loc" and v == 0) and not (k == "scale" and v == 1))

    def _table(self, row):
        import narwhals as nw

        from .table import Table

        backend = self._backend
        if backend is None:
            import pandas as pd

            return Table(pd.DataFrame({k: [v] for k, v in row.items()}))
        return Table(_frame=nw.from_dict({k: [v] for k, v in row.items()}, backend=backend))

    @friendly
    def summary(self):
        """A one-row table: average, median, standard deviation, variance, skewness, excess kurtosis,
        entropy and the lowest and highest possible values. dist.summary()"""
        row = self._summary_row()
        print(f"The distribution ({self.label}): average {_answer(row['average'])}, median {_answer(row['median'])}, "
              f"standard deviation {_answer(row['standard deviation'])}, skewness {_answer(row['skewness'])}, "
              f"excess kurtosis {_answer(row['excess kurtosis'])}; values from {_say(row['lowest possible'])} to "
              f"{_say(row['highest possible'])}.")
        return self._table(row)

    @friendly
    def parameters(self):
        """The parameters as a dictionary of plain names: {"average": 50, "standard_deviation": 10}. dist.parameters()"""
        return dict(self._params)

    # ------------------------------------------------------------------ random numbers
    @friendly
    def random_numbers(self, how_many=None, seed=SEED, rows=None, columns=None):
        """Numbers drawn at random from the distribution; the same seed gives the same numbers.

        dist.random_numbers(how_many=100, seed=1)   (rows= and columns= give a grid)
        R's rnorm() and friends use another random number generator, so their numbers differ.
        """
        from .numbers._base import check_seed, check_whole
        from .numbers.numbers import NumberGrid, Numbers

        example = "dist.random_numbers(how_many=100, seed=1)"
        seed = check_seed(seed, example)
        if how_many is not None and (rows is not None or columns is not None):
            raise SimplydataError(f"Give how_many= for a list, or rows= and columns= for a grid, not both.\n"
                                  f"For example:  {example}")
        if how_many is not None:
            size = check_whole("random_numbers", "how_many=", how_many, example)
        elif rows is not None and columns is not None:
            size = (check_whole("random_numbers", "rows=", rows, example, smallest=1),
                    check_whole("random_numbers", "columns=", columns, example, smallest=1))
        else:
            raise SimplydataError(f"random_numbers() needs how_many= (or rows= and columns= for a grid).\n"
                                  f"For example:  {example}")
        values = self._frozen.rvs(size=size, random_state=np.random.default_rng(seed))
        values = np.asarray(values, dtype=float)
        count = size if isinstance(size, int) else size[0] * size[1]
        code = self._r("r", _r(count))
        seed_text = f"seed={seed}" if seed is not None else "no seed, so different every time"
        r_text = f"; R: {code}, whose numbers differ" if code else ""
        print(f"Drew {count} random numbers from the {self.label} distribution ({seed_text}{r_text}).")
        return NumberGrid(values) if isinstance(size, tuple) else Numbers(values)

    # ------------------------------------------------------------------ chart and table
    @friendly
    def chart(self, show="density", between=None):
        """A chart of the distribution: its density curve (or bars for counts).

        dist.chart()   or   dist.chart(show="cumulative") for the chance of a value at or below x.
        """
        from ._errors import choice_error
        from .chart import chart

        if show not in ("density", "cumulative"):
            raise choice_error("chart", "show", show, ["density", "cumulative"])
        return chart().curve(self, between=between, show=show)

    @friendly
    def table(self):
        """A one-row table: the distribution, its parameters and, for a fit, the standard errors,
        log-likelihood, AIC and BIC. dist.table()"""
        info = self._fit_info
        row = {"distribution": self.name}
        if info and info.get("column"):
            row["column"] = info["column"]
        if not info:
            for k, v in self._params.items():
                if v is not None:
                    row[self._shown(k)] = v
            return self._table(row)
        return self._table({**row, **info["row"]})

    def _shown(self, name):
        return name.replace("_", " ") if isinstance(self._family, Family) else name


WORDS = ["chance", "density_at", "percentile", "cut_off", "middle_range", "average", "median",
         "standard_deviation", "variance", "summary", "random_numbers", "chart", "table", "parameters"]


# ====================================================================== fitting
def _data_problem(family, x, what) -> str | None:
    rule = family.data
    title = family.title
    if rule == "positive" and np.any(x <= 0):
        return f"a {title} distribution has only values above 0, but {what} has {int(np.sum(x <= 0))} of 0 or below"
    if rule == "non-negative" and np.any(x < 0):
        return f"a {title} distribution has no values below 0, but {what} has {int(np.sum(x < 0))}"
    if rule == "share" and np.any((x <= 0) | (x >= 1)):
        return f"a {title} distribution has values between 0 and 1 only, but {what} has some outside (0, 1)"
    if rule in ("counts", "binary", "counts-any") and np.any(x != np.round(x)):
        return f"a {title} distribution has whole numbers only, but {what} has numbers with decimals"
    if rule in ("counts", "binary") and np.any(x < 0):
        return f"a {title} distribution has no counts below 0, but {what} has some"
    if rule == "binary" and np.any(x > 1):
        return f"a {title} distribution has only 0 and 1, but {what} has other values"
    if rule == "angle" and np.any(np.abs(x) > 2 * math.pi + 1e-9):
        return f"a {title} distribution is for angles in radians, but {what} has values beyond 2 * pi"
    return None


def _loglik(family, params, x) -> float:
    try:
        frozen = family.scipy(params) if isinstance(family, Family) else _scipy_generic(family, params)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (family.discrete if isinstance(family, Family) else isinstance(frozen.dist, _st().rv_discrete)):
                values = frozen.logpmf(x)
            else:
                values = frozen.logpdf(x)
        total = float(np.sum(values))
    except (ValueError, ZeroDivisionError, OverflowError):
        return -math.inf
    return total if math.isfinite(total) else -math.inf


def _transform(kind):
    if kind in ("positive",):
        return np.log, np.exp
    if kind in ("share", "share-open"):
        return (lambda v: math.log(v / (1 - v)) if 0 < v < 1 else (-30.0 if v <= 0 else 30.0),
                lambda u: 1 / (1 + math.exp(-u)))
    return (lambda v: v), (lambda u: u)


def _maximise(loglik, start: dict, free: list, kinds: dict):
    """Largest log-likelihood over the free parameters: Nelder-Mead, then BFGS, on a log/logit scale."""
    from scipy import optimize

    to_u = {k: _transform(kinds[k])[0] for k in free}
    from_u = {k: _transform(kinds[k])[1] for k in free}

    def params_of(u):
        p = dict(start)
        for k, v in zip(free, u):
            with np.errstate(all="ignore"):
                try:
                    p[k] = float(from_u[k](v))
                except OverflowError:
                    p[k] = math.inf
        return p

    def nll(u):
        value = loglik(params_of(u))
        return -value if math.isfinite(value) else 1e300

    u0 = np.array([to_u[k](start[k]) for k in free], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Nelder-Mead first (safe far from the top), then BFGS to the top, then a last Nelder-Mead
        # look around; the tolerances stop just above floating-point noise in the log-likelihood.
        best = optimize.minimize(nll, u0, method="Nelder-Mead",
                                 options={"xatol": 1e-6, "fatol": 1e-9, "maxfev": 4000, "adaptive": len(free) > 2})
        for method, options in (("BFGS", {"gtol": 1e-7, "maxiter": 500}),
                                ("Nelder-Mead", {"xatol": 1e-9, "fatol": 1e-11, "maxfev": 2000}),
                                ("BFGS", {"gtol": 1e-8, "maxiter": 500})):
            step = optimize.minimize(nll, best.x, method=method, options=options)
            if step.fun < best.fun:
                best = step
    return params_of(best.x), bool(best.fun < 1e299)


def _hessian(func, point: np.ndarray) -> np.ndarray:
    """Second derivatives by central differences with one Richardson step (accurate to about 1e-8)."""
    k = len(point)

    def at(h):
        out = np.empty((k, k))
        steps = h * np.maximum(np.abs(point), 1e-3)
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k)
                ej = np.zeros(k)
                ei[i] = steps[i]
                ej[j] = steps[j]
                value = (func(point + ei + ej) - func(point + ei - ej) - func(point - ei + ej)
                         + func(point - ei - ej)) / (4 * steps[i] * steps[j])
                out[i, j] = out[j, i] = value
        return out

    big, small = at(2e-4), at(1e-4)
    return (4 * small - big) / 3


def _standard_errors(family, params, free, x):
    if not free:
        return {}
    point = np.array([params[k] for k in free], dtype=float)

    def ll(v):
        p = dict(params)
        p.update(zip(free, (float(t) for t in v)))
        return _loglik(family, p, x)

    with np.errstate(all="ignore"):
        h = _hessian(ll, point)
    errors = {k: math.nan for k in free}
    if not np.all(np.isfinite(h)):
        return errors
    try:
        cov = np.linalg.inv(-h)
    except np.linalg.LinAlgError:
        return errors
    diag = np.diag(cov)
    for k, v in zip(free, diag):
        errors[k] = math.sqrt(v) if v > 0 else math.nan
    return errors


def _gamma_shape(x):
    from scipy import optimize, special

    s = math.log(x.mean()) - np.log(x).mean()
    if s <= 0:
        raise SimplydataError("Every value is the same, so there is no shape to fit.")

    def score(a):
        return math.log(a) - special.digamma(a) - s

    return optimize.brentq(score, 1e-8, 1e10, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)


def _weibull_shape(x):
    from scipy import optimize

    y = x / x.max()
    ly = np.log(y)
    mean_l = ly.mean()

    def score(k):
        w = y ** k
        return float((w * ly).sum() / w.sum() - 1 / k - mean_l)

    low, high = 1e-3, 1.0
    while score(high) < 0 and high < 1e6:
        high *= 2
    while score(low) > 0 and low > 1e-12:
        low /= 2
    k = optimize.brentq(score, low, high, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    scale = float(np.mean(y ** k) ** (1 / k)) * x.max()
    return k, scale


def _gumbel_scale(x):
    from scipy import optimize

    centre = x.mean()
    z = x - centre

    def score(b):
        w = np.exp(-(z - z.min()) / b)  # at most 1, so it never overflows
        return float(b - z.mean() + (z * w).sum() / w.sum())

    sd = x.std()
    low, high = sd * 1e-3, sd * 10
    while score(low) > 0 and low > 1e-300:
        low /= 10
    while score(high) < 0:
        high *= 10
    b = optimize.brentq(score, low, high, xtol=1e-15 * sd, rtol=4 * np.finfo(float).eps, maxiter=500)
    location = centre - b * (math.log(np.mean(np.exp(-(z - z.min()) / b))) - z.min() / b)
    return location, b


def _nbinom_size(x):
    from scipy import optimize, special

    n, m = len(x), x.mean()

    def score(r):
        return float(special.digamma(x + r).sum() - n * special.digamma(r) + n * math.log(r / (r + m)))

    if x.var() <= m:
        raise SimplydataError(
            "The values spread no more than a Poisson distribution would (variance at or below the average), so a "
            'negative binomial has no best size; fit "poisson" instead.')
    low, high = 1e-6, 1.0
    while score(high) > 0 and high < 1e12:
        high *= 4
    if score(high) > 0:
        raise SimplydataError('The negative binomial fit runs off to a Poisson distribution; fit "poisson" instead.')
    return optimize.brentq(score, low, high, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)


def _vonmises_fit(x):
    from scipy import optimize, special

    c, s = np.cos(x).mean(), np.sin(x).mean()
    mu = math.atan2(s, c)
    length = math.hypot(c, s)
    if length >= 1 - 1e-12:
        raise SimplydataError("Every angle is the same, so there is no spread to fit.")

    def score(k):
        return float(special.ive(1, k) / special.ive(0, k) - length)

    high = 1.0
    while score(high) < 0:
        high *= 2
    kappa = optimize.brentq(score, 1e-12, high, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500) \
        if score(1e-12) < 0 else 1e-12
    return mu, kappa


def _closed_or_score(family, x):
    """Maximum-likelihood values without a search, and the textbook standard errors (or None)."""
    n = len(x)
    name = family.name
    if name == "normal":
        m, sd = x.mean(), x.std()
        return {"average": m, "standard_deviation": sd}, {"average": sd / math.sqrt(n),
                                                          "standard_deviation": sd / math.sqrt(2 * n)}
    if name == "lognormal":
        lx = np.log(x)
        m, sd = lx.mean(), lx.std()
        return ({"log_average": m, "log_standard_deviation": sd},
                {"log_average": sd / math.sqrt(n), "log_standard_deviation": sd / math.sqrt(2 * n)})
    if name == "exponential":
        rate = 1 / x.mean()
        return {"rate": rate}, {"rate": rate / math.sqrt(n)}
    if name == "poisson":
        lam = x.mean()
        return {"average": lam}, {"average": math.sqrt(lam / n)}
    if name == "geometric":
        p = 1 / (1 + x.mean())
        return {"chance": p}, {"chance": math.sqrt(p * p * (1 - p) / n)}
    if name == "bernoulli":
        p = x.mean()
        return {"chance": p}, {"chance": math.sqrt(p * (1 - p) / n)}
    if name == "gamma":
        a = _gamma_shape(x)
        return {"shape": a, "rate": a / x.mean()}, None
    if name == "weibull":
        k, scale = _weibull_shape(x)
        return {"shape": k, "scale": scale}, None
    if name == "gumbel":
        location, scale = _gumbel_scale(x)
        return {"location": location, "scale": scale}, None
    if name == "negative binomial":
        return {"size": _nbinom_size(x), "chance": None, "average": x.mean()}, None
    if name == "von mises":
        mu, kappa = _vonmises_fit(x)
        return {"average": mu, "concentration": kappa}, None
    if name == "laplace":
        med = float(np.median(x))
        scale = float(np.mean(np.abs(x - med)))
        return {"location": med, "scale": scale}, {"location": math.nan, "scale": scale / math.sqrt(n)}
    if name == "pareto":
        scale = float(x.min())
        shape = n / float(np.log(x / scale).sum())
        return {"shape": shape, "scale": scale}, {"shape": shape / math.sqrt(n), "scale": math.nan}
    if name == "half-normal":
        sd = math.sqrt(float(np.mean(x ** 2)))
        return {"standard_deviation": sd}, {"standard_deviation": sd / math.sqrt(2 * n)}
    if name == "inverse gaussian":
        m = x.mean()
        shape = n / float((1 / x - 1 / m).sum())
        return {"average": m, "shape": shape}, {"average": math.sqrt(m ** 3 / (shape * n)),
                                                 "shape": shape * math.sqrt(2 / n)}
    if name == "uniform":
        return {"lowest": float(x.min()), "highest": float(x.max())}, {"lowest": math.nan, "highest": math.nan}
    if name == "whole number uniform":
        return {"lowest": float(x.min()), "highest": float(x.max())}, {"lowest": math.nan, "highest": math.nan}
    return None, None


def _numeric_start(family, x):
    name = family.name
    m, sd = float(x.mean()), float(x.std())
    q1, q3 = np.percentile(x, [25, 75])
    if name == "beta":
        common = m * (1 - m) / max(x.var(), 1e-12) - 1
        return {"shape1": max(m * common, 0.1), "shape2": max((1 - m) * common, 0.1)}
    if name == "t":
        return {"degrees_of_freedom": 10.0, "location": float(np.median(x)), "scale": max(float(q3 - q1) / 1.5, 1e-6)}
    if name == "cauchy":
        return {"location": float(np.median(x)), "scale": max(float(q3 - q1) / 2, 1e-6)}
    if name == "logistic":
        return {"location": m, "scale": max(sd * math.sqrt(3) / math.pi, 1e-6)}
    if name == "chi-square":
        return {"degrees_of_freedom": max(m, 0.1)}
    if name == "F":
        d2 = 2 * m / (m - 1) if m > 1.05 else 20.0
        return {"degrees_of_freedom_1": 5.0, "degrees_of_freedom_2": min(max(d2, 4.5), 200)}
    return None


@dataclass
class _Fit:
    params: dict
    errors: dict
    free: list
    loglik: float
    settled: bool
    method: str


def _fit_values(family, x, fixed: dict):
    """The fit itself: parameters, standard errors, log-likelihood."""
    if isinstance(family, str):
        return _fit_scipy(family, x, fixed)
    if family.name == "negative binomial":
        names = ["size", "average"]
    else:
        names = [p.name for p in family.params]
    kinds = {p.name: p.kind for p in family.params}
    free = [k for k in names if k not in fixed]
    if family.name == "binomial":
        if "tries" not in fixed:
            raise SimplydataError(
                'Fitting a binomial distribution needs the number of tries, held fixed: fixed={"tries": 10}.')
        tries = fixed["tries"]
        if np.any(x > tries):
            raise SimplydataError(f"Some counts are above tries={_say(tries)}, which is impossible for a binomial.")
        if "chance" in fixed:
            raise SimplydataError("Every parameter is held fixed, so there is nothing to fit.")
        p = float(x.mean() / tries)
        params = {"tries": float(tries), "chance": p}
        errors = {"chance": math.sqrt(p * (1 - p) / (len(x) * tries))}
        return _Fit(params, errors, ["chance"], _loglik(family, params, x), True, "closed form")
    if not free:
        raise SimplydataError("Every parameter is held fixed, so there is nothing to fit.")
    if not fixed:
        params, errors = _closed_or_score(family, x)
        if params is not None:
            how = "closed form" if family.fit == "closed" else "likelihood equations solved exactly"
            if errors is None:
                errors = _standard_errors(family, params, free, x)
            return _Fit(params, errors, free, _loglik(family, params, x), True, how)
    # a search over the free parameters (with fixed ones held)
    start = _numeric_start(family, x)
    if start is None:
        start, _ = _closed_or_score(family, x) if family.fit != "numeric" else (None, None)
    if start is None:
        start = {p.name: (p.default if p.default not in (REQUIRED, None) else 1.0) for p in family.params}
    start = {k: v for k, v in start.items() if v is not None}
    start.update({k: float(v) for k, v in fixed.items()})
    if family.name == "negative binomial":
        start.pop("chance", None)
    params, ok = _maximise(lambda p: _loglik(family, p, x), start, free, kinds)
    if family.name == "negative binomial":
        params["chance"] = None
        params = {"size": params["size"], "chance": None, "average": params["average"]}
    ll = _loglik(family, params, x)
    if not math.isfinite(ll):
        raise SimplydataError(f"simplydata could not fit a {family.title} distribution to these values.")
    errors = _standard_errors(family, params, free, x)
    return _Fit(params, errors, free, ll, ok, "numerical search (Nelder-Mead, then BFGS)")


def _fit_scipy(scipy_name, x, fixed):
    dist = getattr(_st(), scipy_name)
    if isinstance(dist, _st().rv_discrete):
        raise SimplydataError(f"SciPy's {scipy_name} counts; fitting SciPy count distributions by name is not "
                              "supported. Use one of simplydata's count names, like \"poisson\".")
    shapes = [s.strip() for s in (dist.shapes or "").split(",") if s.strip()]
    names = shapes + ["loc", "scale"]
    held = {("f" + k if k in ("loc", "scale") else "f" + k): float(v) for k, v in fixed.items()}
    unknown = [k for k in fixed if k not in names]
    if unknown:
        raise SimplydataError(f"fixed= has {unknown[0]!r}, which is not a parameter of SciPy's {scipy_name} "
                              f"({', '.join(names)}).")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = dist.fit(x, **held)
    params = dict(zip(names, (float(v) for v in found)))
    free = [k for k in names if k not in fixed]
    kinds = {k: ("positive" if k == "scale" else "real") for k in names}

    def ll(p):
        return _loglik(scipy_name, p, x)

    params, _ = _maximise(ll, params, free, kinds)
    errors = _standard_errors(scipy_name, params, free, x)
    return _Fit(params, errors, free, ll(params), True, "SciPy's fit, then a numerical polish")


def _read_fixed(family, fixed):
    if fixed is None:
        return {}
    if not isinstance(fixed, dict) or not all(_is_number(v) for v in fixed.values()):
        raise SimplydataError('fixed= holds parameters at a value, like fixed={"shape": 2}.')
    if isinstance(family, str):
        return dict(fixed)
    spelled = {}
    for p in family.params:
        spelled[p.name] = p.name
        spelled.update({a: p.name for a in p.aliases})
    out = {}
    for k, v in fixed.items():
        if k in family.alternatives:
            out[family.alternatives[k]] = 1 / float(v)
            continue
        name = spelled.get(k)
        if name is None:
            raise SimplydataError(f"fixed= has {k!r}, which is not a parameter of the {family.title} distribution.\n"
                                  f"Its parameters:\n{_param_list(family)}")
        problem = _kind_ok(family.param(name).kind, v)
        if problem:
            raise SimplydataError(f"fixed= gives {name} the value {v!r}, but it needs {problem}.")
        out[name] = float(v)
    if family.name == "negative binomial" and "chance" in out:
        raise SimplydataError('A negative binomial fit finds size and average (as R fitdistr); hold one of those, '
                              'like fixed={"size": 2}.')
    return out


def fit(values, name="normal", fixed=None, say=False, column=None, backend=None) -> Distribution:
    """Fit a distribution to numbers by maximum likelihood; gives back the fitted Distribution.

    For other parts of simplydata (chart.fit_distribution) and for the table and numbers
    words. say=True prints the two sentences that table.fit_distribution prints.
    """
    family = _find_family(name, "fit_distribution")
    if isinstance(family, Distribution):
        family = family._family
    if isinstance(family, Family) and family.fit is None:
        raise SimplydataError(f"A {family.title} distribution cannot be fitted here: {family.not_fittable}.")
    x = np.asarray(values, dtype=float).ravel()
    x = x[~np.isnan(x)]
    what = f'"{column}"' if column else "the numbers"
    if len(x) < 2:
        raise SimplydataError(f"Fitting a distribution needs at least 2 values; {what} has {len(x)}.")
    if not np.all(np.isfinite(x)):
        raise SimplydataError(f"{what[0].upper()}{what[1:]} has infinite values, which no distribution can fit.")
    if isinstance(family, Family):
        problem = _data_problem(family, x, what)
        if problem:
            raise SimplydataError(f"Cannot fit: {problem}.")
        if not family.discrete and np.ptp(x) == 0:
            raise SimplydataError(f"Every value of {what} is the same, so there is no spread to fit.")
    held = _read_fixed(family, fixed)
    result = _fit_values(family, x, held)
    n = len(x)
    k = len(result.free)
    aic = -2 * result.loglik + 2 * k
    bic = -2 * result.loglik + k * math.log(n)
    params = result.params
    shown = (lambda s: s.replace("_", " ")) if isinstance(family, Family) else (lambda s: s)
    row = {}
    for key, value in params.items():
        if value is None:
            continue
        row[shown(key)] = float(value)
    for key, value in params.items():
        if value is None:
            continue
        row[f"standard error of {shown(key)}"] = float(result.errors.get(key, math.nan))
    row.update({"log-likelihood": result.loglik, "AIC": aic, "BIC": bic, "rows": n,
                "held fixed": ", ".join(shown(h) for h in held) if held else "",
                "method": "maximum likelihood (" + result.method + ")", "settled": "yes" if result.settled else "no"})
    info = {"rows": n, "column": column, "log-likelihood": result.loglik, "AIC": aic, "BIC": bic,
            "errors": result.errors, "free": result.free, "held": held, "row": row, "settled": result.settled,
            "method": result.method}
    if isinstance(family, Family):
        plain = {p.name: params.get(p.name) for p in family.params}
        dist = Distribution(family, plain, fit_info=info, backend=backend)
    else:
        dist = Distribution(family, params, fit_info=info, backend=backend)
    if say:
        _say_fit(dist, column)
    return dist


def _fit_parts(dist) -> str:
    info = dist._fit_info
    parts = []
    labelled = False
    shown = dist._shown
    for key, value in dist._params.items():
        if value is None:
            continue
        if key in info["held"]:
            parts.append(f"{shown(key)} = {_answer(value)} (held fixed)")
            continue
        error = info["errors"].get(key, math.nan)
        err = "no standard error at this edge" if math.isnan(error) else _answer(error)
        label = "" if labelled or math.isnan(error) else "standard error "
        labelled = labelled or bool(label)
        parts.append(f"{shown(key)} = {_answer(value)} ({label}{err})")
    return ", ".join(parts)


def _r_fit_text(dist) -> str:
    family = dist._family
    if not isinstance(family, Family):
        return f"SciPy {family}.fit()"
    return f"R {family.r_fit}" if family.r_fit.startswith(("MASS", "fitdistrplus")) else family.r_fit


def _say_fit(dist, column=None, check_word="table.compare_shape_to"):
    info = dist._fit_info
    family = dist._family
    title = family.title if isinstance(family, Family) else f"SciPy {family}"
    what = f'"{column}"' if column else "the numbers"
    article = "an" if title[0].lower() in "aeiou" else "a"
    print(f"Fitted {article} {title} distribution to {what} ({info['rows']} values): {_fit_parts(dist)}. "
          f"Log-likelihood {info['log-likelihood']:.1f}, AIC {info['AIC']:.1f}.")
    method = f"Method: maximum likelihood, as {_r_fit_text(dist)}."
    if info["method"] not in ("closed form",) and isinstance(family, Family) and family.fit == "numeric":
        method += " The values were found by a numerical search (Nelder-Mead, then BFGS)."
    if not dist.is_discrete and column and dist.name != "normal":
        method += f' To check whether the shape fits, use {check_word}("{column}", "{dist.name}").'
    print(method)
    if dist.name == "normal" and "standard_deviation" not in info["held"]:
        print("Note: the maximum-likelihood standard deviation divides by n, so it is a little smaller than R sd().")
    if not info["settled"]:
        print("Note: the search did not settle; the values may not be the best. Try fixed= for one parameter.")


# ====================================================================== goodness of fit
def _gof_statistics(dist, x) -> dict:
    """Anderson-Darling, Kolmogorov-Smirnov and Cramer-von Mises statistics, as R fitdistrplus::gofstat()."""
    if dist.is_discrete:
        return {"Anderson-Darling statistic": math.nan, "Kolmogorov-Smirnov statistic": math.nan,
                "Cramer-von Mises statistic": math.nan}
    xs = np.sort(x)
    n = len(xs)
    with np.errstate(all="ignore"):
        u = np.asarray(dist._frozen.cdf(xs), dtype=float)
        log_cdf = np.asarray(dist._frozen.logcdf(xs), dtype=float)
        log_sf = np.asarray(dist._frozen.logsf(xs), dtype=float)
    i = np.arange(1, n + 1)
    ad = -n - np.mean((2 * i - 1) * (log_cdf + log_sf[::-1]))
    ks = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    cvm = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
    return {"Anderson-Darling statistic": float(ad), "Kolmogorov-Smirnov statistic": float(ks),
            "Cramer-von Mises statistic": float(cvm)}


def compare_fits(x, names, column=None):
    """Fit each distribution and rank by AIC. Gives (rows sorted by AIC, notes)."""
    names = list(names) if not isinstance(names, str) else [names]
    if not names:
        raise SimplydataError('compare_distributions() needs some names, like ("normal", "gamma").')
    families = [_find_family(n, "compare_distributions") for n in names]
    kinds = {(f.discrete if isinstance(f, Family) else False) for f in families}
    if len(kinds) > 1:
        raise SimplydataError("Count (discrete) and smooth (continuous) distributions cannot be compared by AIC: "
                              "their likelihoods are on different scales. Compare counts with counts.")
    rows, notes = [], []
    for name, family in zip(names, families):
        label = family.name if isinstance(family, Family) else family
        try:
            dist = fit(x, label, column=column)
        except SimplydataError as error:
            notes.append(f"Note: {label} was left out: {str(error).splitlines()[0]}")
            continue
        info = dist._fit_info
        rows.append({"distribution": label, "parameters": _param_short(dist), "log-likelihood": info["log-likelihood"],
                     "AIC": info["AIC"], "BIC": info["BIC"], **_gof_statistics(dist, x), "rows": info["rows"],
                     "_dist": dist})
    if not rows:
        raise SimplydataError("None of the distributions could be fitted:\n" + "\n".join(notes))
    rows.sort(key=lambda r: r["AIC"])
    best = rows[0]["AIC"]
    for r in rows:
        r["AIC difference"] = r["AIC"] - best
    return rows, notes


def _param_short(dist) -> str:
    return ", ".join(f"{dist._shown(k)} {_answer(v)}" for k, v in dist._params.items() if v is not None)


def _ranking_sentence(rows) -> str:
    first = rows[0]
    text = f"Best by AIC: {first['distribution']} (AIC {first['AIC']:.1f})"
    rest = [f"{r['distribution']} (+{r['AIC difference']:.1f})" for r in rows[1:]]
    if rest:
        text += ", then " + ", ".join(rest)
    return text + "."


COMPARE_COLUMNS = ["distribution", "parameters", "log-likelihood", "AIC", "BIC", "AIC difference",
                   "Anderson-Darling statistic", "Kolmogorov-Smirnov statistic", "Cramer-von Mises statistic", "rows"]


SHAPE_TEST_SAMPLES = 9999
SAMPLES_RANGE = (99, 1_000_000)


def check_samples(samples, step="compare_shape_to"):
    """samples= for the shape test: a whole number from 99 to a million, with a friendly error."""
    low, high = SAMPLES_RANGE
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise SimplydataError(f"{step}() needs samples= as a whole number of simulated samples, such as "
                              f"samples=999 (quicker, rougher) or samples={SHAPE_TEST_SAMPLES} (the default).")
    if not low <= samples <= high:
        raise SimplydataError(f"{step}() needs samples= from {low} to {high:,}, not {samples}; "
                              f"{SHAPE_TEST_SAMPLES} (the default) gives a p-value to about 0.005, 999 a rougher one.")
    return int(samples)


def _shape_test_start(values, name, seed, samples, column):
    """Check, fit the data, and set up the simulation: (family, label, x, fitted Distribution, MonteCarlo)."""
    from ._shape_monte_carlo import MonteCarlo

    family = _find_family(name, "compare_shape_to")
    label = family.name if isinstance(family, Family) else family
    if isinstance(family, Family) and family.discrete or (
            isinstance(family, str) and isinstance(getattr(_st(), family), _st().rv_discrete)):
        raise SimplydataError(f"compare_shape_to() checks smooth (continuous) shapes; {label} is a count "
                              "distribution. For counts, compare them with compare_counts_to(column, expected=...).")
    if isinstance(family, Family) and family.goodness is None:
        raise SimplydataError(f"compare_shape_to() cannot check a {family.title} shape: "
                              f"{family.not_fittable or 'it has no fit to test against'}.")
    samples = check_samples(samples)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise SimplydataError("seed= needs a whole number, such as seed=1 (the same seed gives the same p-value).")
    x = np.asarray(values, dtype=float).ravel()
    x = x[~np.isnan(x)]
    dist = fit(x, label, column=column)
    if isinstance(family, Family):
        scipy_name, known, fitted = family.goodness(dist._params)
    else:
        scipy_name, known, fitted = family, {}, dict(dist._params)
    run = MonteCarlo(getattr(_st(), scipy_name), known, fitted, x, samples, int(seed))
    return family, x, dist, run


def shape_test(values, name, seed=SEED, samples=SHAPE_TEST_SAMPLES, column=None, _started=None) -> dict:
    """Anderson-Darling test of a fitted shape with a Monte Carlo p-value that allows for the fitting.

    The engine of table.compare_shape_to(column, "gamma") (any shape but "normal", which keeps
    Shapiro-Wilk). The parameters are fitted by maximum likelihood (as fit_distribution); the
    p-value comes from samples= simulated samples (9999) drawn from the fitted distribution
    (seed=), each fitted again by maximum likelihood with the same parameters held (loc = 0 for
    gamma, as R): the method of SciPy goodness_of_fit(statistic="ad"), with the refits done all
    at once where the likelihood equations allow it (see simplydata._shape_monte_carlo). When the
    run is expected to take longer than about 10 s, a short note says so before it starts.
    Gives back a dict: test, what it compares, statistic name, statistic, p-value, rows,
    Monte Carlo samples, seed, distribution (the fitted Distribution), sentence, method.
    """
    from ._shape_monte_carlo import time_note

    if _started is None:
        _started = _shape_test_start(values, name, seed, samples, column)
        note = time_note(_started[3].pilot(), _started[3].samples)
        if note:
            print(note)
    family, x, dist, run = _started
    a, p = run.finish()
    samples, seed = run.samples, int(seed)
    from .table.stat_tests import _chance, _p_shown

    what = f'"{column}"' if column else "the numbers"
    title = family.title if isinstance(family, Family) else f"SciPy {family}"
    fitted_words = _param_short(dist)
    if p < 0.05:
        verdict = _chance(p)[:-1]  # "is unlikely to be chance (p = 0.030"
        sentence = (f"The values of {what} do not look like a {title} distribution: a shape this far from {title} "
                    f"{verdict}; Anderson-Darling A = {a:.3f}, {len(x)} values; fitted {fitted_words}).")
    else:
        sentence = (f"The values of {what} could well follow a {title} distribution (Anderson-Darling A = {a:.3f}, "
                    f"{_p_shown(p)}, {len(x)} values; fitted {fitted_words}). This does not prove that they do.")
    method = (f"Test: Anderson-Darling test of a fitted {title} shape, with a Monte Carlo p-value from "
              f"{samples} simulated samples that are each fitted again (as SciPy goodness_of_fit, seed={seed}); "
              + (f"the statistic is R goftest::ad.test(x, \"{family.r_package}p{family.r_prefix}\", ...) with the "
                 "fitted values." if isinstance(family, Family) and family.r_prefix else
                 "the statistic is the usual Anderson-Darling A with the fitted values."))
    return {"test": "Anderson-Darling test (fitted parameters, Monte Carlo p-value)",
            "what it compares": f"the shape of {what} with a fitted {title} distribution",
            "statistic name": "A", "statistic": a, "p-value": p, "rows": len(x),
            "Monte Carlo samples": samples, "seed": seed, "distribution": dist,
            "sentence": sentence, "method": method}
