"""Statistical tests in plain words. They never change the table.

You ask the question in words; simplydata picks the standard test, prints one plain answer
that says which test was used and why, and gives back a one-row result table (which can be
saved like any table). test= lets an expert choose another test. Numbers come from SciPy.

Choices (the same defaults as R, so results can be checked there):

* compare_averages, 2 groups: Welch's t-test (R t.test() default). It does NOT assume that
  the groups have the same spread, so there is nothing to check about spread. Effect size:
  Hedges' g, the difference in pooled standard deviations, with the exact small-sample
  correction.
* compare_averages, 3 or more groups: Welch's one-way ANOVA (R oneway.test() default,
  var.equal = FALSE), written out from Welch (1951). Effect size: omega-squared, from the
  ordinary sums of squares; it is less biased upwards than eta-squared in small samples
  (Okada 2013). Which pairs differ: compare_each_pair(), pairwise Welch t-tests with Holm's
  correction (R pairwise.t.test(pool.sd = FALSE, p.adjust.method = "holm")); or with
  test="tukey" Tukey's honest significant difference (R TukeyHSD(aov(...)), SciPy
  tukey_hsd, the Tukey-Kramer form for unequal group sizes), with simultaneous 95 %
  intervals. Tukey assumes equal spreads, so a note warns when the biggest standard
  deviation is more than twice the smallest.
* compare_spreads: Levene's test centred on the medians (the Brown-Forsythe form; the
  R car::leveneTest() default, SciPy levene(center="median"); the same as a one-way ANOVA
  on the distances from each group's median). test="bartlett" (R bartlett.test()) and
  test="fligner" (R fligner.test(), Fligner-Killeen with median centring).
* based_on="ranks": Mann-Whitney U (2 groups) or Kruskal-Wallis (3 or more), with the
  rank-biserial correlation or rank epsilon-squared as effect size.
* compare_before_after: paired t-test (Cohen's d_z) or Wilcoxon signed-rank.
* compare_average_to: one-sample t-test (Cohen's d) or Wilcoxon signed-rank.
  Wilcoxon and Mann-Whitney give an exact p-value when there are fewer than 50 values and
  no ties (and no zero changes), otherwise the normal approximation with continuity and tie
  correction; zero changes are left out. This is R wilcox.test() as it was for many years,
  and R wilcox.test(exact = FALSE) when there are ties. (R 4.6 by default now gives an
  exact p-value even with ties below 50 values, which can differ a little.)
* compare_counts: chi-square test of independence, with Yates' continuity correction for
  2 x 2 tables only (R chisq.test() and SciPy default). If an expected count is below 5 a
  note says so and suggests test="fisher" (2 x 2 tables); simplydata never switches by
  itself. Effect size: Cramer's V (from the chi-square without correction).
* compare_counts_to: chi-square goodness-of-fit test against expected shares (R
  chisq.test(x, p = ...); equal shares when none are given). Effect size: Cohen's w.
* compare_percent_to: the exact binomial test of a percent against a chosen percent
  (R binom.test(x, n, p), SciPy binomtest; the two-sided p-value as R), with the exact
  Clopper-Pearson 95 % interval. Effect size: Cohen's h.
* compare_shape_to(..., "normal"): the Shapiro-Wilk test (R shapiro.test(), Royston's
  algorithm), for 3 to 5000 values per group, as R.
* add_adjusted_p_value_column: Holm (default), Bonferroni, Benjamini-Hochberg or
  Benjamini-Yekutieli, exactly as R p.adjust(); empty p-values stay empty and are not
  counted.
* The groups column is for_each= in every test (between= was its old name and still works
  for one release, with a note).
* fit_line, fit_curve and fit_formula: see fitting.py (R lm() and nls()).

Assumption checks are notes, never silent switches. For tests on averages, each group with
3 to 5000 values is checked with the Shapiro-Wilk test. With fewer than 30 values a
"not bell-shaped" result comes with a safer choice (based_on="ranks"). With 30 or more the
note says the test is usually still fine: the average of many values is close to
bell-shaped even when the values are not (the central limit theorem), and with many values
Shapiro-Wilk flags even tiny, harmless departures.

The p-value words are the same as for correlation_between: below 0.001 "very unlikely to be
chance", below 0.05 "unlikely to be chance", otherwise "could easily be chance".
"""

from __future__ import annotations

import ast
import difflib
import math

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, choice_error, friendly
from .._stats import _plain_name
from .columns import (CONSTANTS, FORMULA_HELP, FUNCTION_SPELLINGS, FUNCTIONS, SPECIAL_FUNCTIONS, _formula_error,
                      _FormulaReader, _is_number, special_function)
from .summaries import GROUPS_IN_SENTENCE as GROUPS_SHOWN
from .summaries import _key, _order, _p_words, _plain, _say_label

BASED_ON = ("averages", "ranks")
AVERAGE_TESTS = ("welch", "student", "welch anova", "anova", "mann-whitney", "kruskal-wallis")
RANK_TESTS = ("mann-whitney", "kruskal-wallis")
TWO_GROUP_TESTS = ("welch", "student", "mann-whitney")
COUNT_TESTS = ("chi-square", "fisher")
PAIR_TESTS = ("welch", "mann-whitney", "tukey")
SPREAD_TESTS = ("levene", "bartlett", "fligner")
SPREAD_ALIASES = {"brown-forsythe": "levene", "brown forsythe": "levene", "fligner-killeen": "fligner",
                  "fligner killeen": "fligner"}
SPREAD_RATIO = 2  # Tukey note: the biggest standard deviation more than twice the smallest
EXACT_LIMIT = 50  # exact p-values below 50 values without ties, as R wilcox.test()
CLT_SIZE = 30  # from this many values, a "not bell-shaped" note does not suggest switching
SHAPIRO_LIMIT = 5000  # SciPy's Shapiro-Wilk is reliable up to 5000 values
SHAPES = ("normal",)  # compare_shape_to: the shapes it can check
# add_adjusted_p_value_column: method -> (name in sentences, R p.adjust name)
ADJUST_METHODS = {
    "holm": ("Holm", "holm"),
    "bonferroni": ("Bonferroni", "bonferroni"),
    "benjamini-hochberg": ("Benjamini-Hochberg", "BH"),
    "benjamini-yekutieli": ("Benjamini-Yekutieli", "BY"),
}
ADJUST_ALIASES = {"bh": "benjamini-hochberg", "fdr": "benjamini-hochberg", "by": "benjamini-yekutieli",
                  "benjamini hochberg": "benjamini-hochberg", "benjamini yekutieli": "benjamini-yekutieli"}
ADJUST_MEANING = {
    "holm": "the chance of even one false alarm among all the tests stays at 5 %",
    "bonferroni": "the chance of even one false alarm among all the tests stays at 5 %",
    "benjamini-hochberg": "about 5 % of the findings below 0.05 are expected to be false alarms "
                          "(the false discovery rate)",
    "benjamini-yekutieli": "about 5 % of the findings below 0.05 are expected to be false alarms "
                           "(the false discovery rate), however the tests depend on each other",
}


# ---------------------------------------------------------------------------- small helpers
def _num(x) -> str:
    """A number with 3 significant digits, written plainly (no 1e+05 unless huge or tiny)."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "infinite"
    if x == 0:
        return "0"
    size = abs(x)
    if size >= 1e6 or size < 1e-4:
        return f"{x:.3g}"
    digits = max(0, 2 - math.floor(math.log10(size)))
    text = f"{x:.{digits}f}"
    return "0" if float(text) == 0 else text


def _chance(p: float, plural: bool = False) -> str:
    """'is unlikely to be chance (p = 0.01)' or 'could easily be chance (p = 0.4)'.

    A large p-value only means the data cannot tell; it never shows that there is no
    difference, so the words say that too."""
    words = _p_words(p)
    if words.startswith("could"):
        return words + "; this does not show that there is no difference"
    return ("are " if plural else "is ") + words


def _whole(x) -> str:
    """Degrees of freedom and rank sums: 2 not 2.00, 33.8 stays 33.8, 213.5 stays 213.5."""
    x = float(x)
    return str(int(x)) if x.is_integer() else (f"{x:g}" if abs(x) >= 100 else _num(x))


def _percent(share: float) -> str:
    return f"{round(share * 100)} %"


def _label(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def _list_words(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _skipped_note(skipped: int, names) -> None:
    if not skipped:
        return
    rows = "row" if skipped == 1 else "rows"
    verb = "was" if skipped == 1 else "were"
    where = " or ".join(f'"{n}"' for n in names)
    print(f"Note: {skipped} {rows} with an empty cell in {where} {verb} skipped.")


def _hedges_correction(df: float) -> float:
    """Exact small-sample correction J for Hedges' g (Hedges 1981)."""
    from scipy.special import gammaln

    return math.exp(gammaln(df / 2) - 0.5 * math.log(df / 2) - gammaln((df - 1) / 2))


def _holm(p_values) -> list[float]:
    """Holm's step-down adjustment (R p.adjust(method = "holm"))."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def _adjust_p(p_values, method: str) -> np.ndarray:
    """Adjusted p-values exactly as R p.adjust() (no empty values here).

    bonferroni  pmin(1, m * p)
    holm        pmin(1, cummax((m - i + 1) * p[o]))[ro],  o = order(p)
    BH          pmin(1, cummin(m / i * p[o]))[ro],        o = order(p, decreasing = TRUE)
    BY          the same as BH, times sum(1 / (1:m))
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(1.0, m * p)
    if method == "holm":
        return np.asarray(_holm(p))
    factor = 1.0 if method == "benjamini-hochberg" else float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(-p, kind="stable")  # biggest p first
    ranks = np.arange(m, 0, -1)
    adjusted = np.minimum.accumulate(factor * m / ranks * p[order])
    out = np.empty(m)
    out[order] = np.minimum(1.0, adjusted)
    return out


def _p_shown(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}"


def _welch_anova(groups):
    """Welch's one-way ANOVA (Welch 1951), as R oneway.test(var.equal = FALSE).

    w_i = n_i / s_i^2, weighted mean m' = sum(w_i m_i) / sum(w_i)
    A = sum(w_i (m_i - m')^2) / (k - 1)
    L = sum((1 - w_i / W)^2 / (n_i - 1))
    F = A / (1 + 2 (k - 2) L / (k^2 - 1)),  df1 = k - 1,  df2 = (k^2 - 1) / (3 L)
    """
    from scipy import stats

    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    w = n / variances
    total = w.sum()
    centre = (w * means).sum() / total
    a = (w * (means - centre) ** 2).sum() / (k - 1)
    lam = ((1 - w / total) ** 2 / (n - 1)).sum()
    f = a / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    df1, df2 = k - 1.0, (k ** 2 - 1) / (3 * lam)
    return float(f), df1, float(df2), float(stats.f.sf(f, df1, df2))


def _omega_squared(groups) -> float:
    """Omega-squared from the ordinary one-way sums of squares (Hays 1963)."""
    values = np.concatenate(groups)
    k, total_n = len(groups), len(values)
    grand = values.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    mean_square_within = within / (total_n - k)
    return float((between - (k - 1) * mean_square_within) / (between + within + mean_square_within))


def _has_ties(values) -> bool:
    return len(np.unique(values)) < len(values)


def _mann_whitney(a, b):
    """Mann-Whitney U like R wilcox.test(): exact below 50 values per group without ties."""
    from scipy import stats

    exact = len(a) < EXACT_LIMIT and len(b) < EXACT_LIMIT and not _has_ties(np.concatenate([a, b]))
    method = "exact" if exact else "asymptotic"
    result = stats.mannwhitneyu(a, b, alternative="two-sided", method=method, use_continuity=True)
    u = float(result.statistic)  # U of the first group, the same as R's W
    how = "exact p-value" if exact else "normal approximation with continuity correction"
    return u, float(result.pvalue), 2 * u / (len(a) * len(b)) - 1, how


def _signed_rank(changes):
    """Wilcoxon signed-rank like R wilcox.test(): zero changes left out; exact below 50 values
    without ties, otherwise the normal approximation with continuity correction.
    Gives V (the sum of the ranks of the positive changes, as R), p, the matched-pairs
    rank-biserial correlation, how p was found, and how many zero changes were left out."""
    from scipy import stats

    zeros = int((changes == 0).sum())
    nonzero = changes[changes != 0]
    if len(nonzero) == 0:
        return None
    ranks = stats.rankdata(np.abs(nonzero))
    v = float(ranks[nonzero > 0].sum())
    top = len(nonzero) * (len(nonzero) + 1) / 2
    exact = len(nonzero) < EXACT_LIMIT and not _has_ties(np.abs(nonzero)) and zeros == 0
    method = "exact" if exact else "approx"
    result = stats.wilcoxon(nonzero, zero_method="wilcox", correction=True, method=method)
    how = "exact p-value" if exact else "normal approximation with continuity correction"
    return v, float(result.pvalue), 2 * v / top - 1, how, zeros


def _shape_notes(named_values, what="the values") -> list[str]:
    """Shapiro-Wilk check of each group; plain notes, never a switch."""
    from scipy import stats

    notes = []
    for name, values in named_values:
        n = len(values)
        if n < 3:
            notes.append(f"Note: {name} has only {n} values: too few to check whether {what} are bell-shaped.")
            continue
        if n > SHAPIRO_LIMIT or np.ptp(values) == 0:
            continue
        p = float(stats.shapiro(values).pvalue)
        if p >= 0.05:
            continue
        shown = "p < 0.0001" if p < 0.0001 else f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.4f}"
        if n < CLT_SIZE:
            notes.append(
                f"Note: {what} in {name} don't look bell-shaped (Shapiro-Wilk {shown}); "
                'a safer test here: based_on="ranks".'
            )
        else:
            notes.append(
                f"Note: {what} in {name} don't look bell-shaped (Shapiro-Wilk {shown}), but with {n} values "
                "the average is close to bell-shaped anyway (the central limit theorem), so this test is "
                "usually still fine."
            )
    return notes


# ---------------------------------------------------------------------------- fit_formula reader
FIT_FUNCTIONS = {"exp": (1, 1), "log": (1, 2), "sqrt": (1, 1), "abs": (1, 1),
                 **{name: FUNCTIONS[name] for name in ("sin", "cos", *SPECIAL_FUNCTIONS)}}
# gamma and beta are also usual names for values to find: written without brackets they are values
FIT_NAMES_FREE = ("gamma", "beta")
FIT_HELP = (
    "A formula to fit has the column to fit on the left of =, and on the right:\n"
    "  column names               hours  (names with spaces go in backticks: `max temp`)\n"
    "  values to find             any other name, such as a, b, k or rate\n"
    "  numbers                    2, 100, 0.5\n"
    "  + - * / **                 add, subtract, multiply, divide, power\n"
    "  ( )                        brackets\n"
    "  exp(x)  log(x)  log(x, 10) e to the power x, natural log, log to base 10\n"
    "  sqrt(x)  abs(x)            square root, size without the minus sign\n"
    "  sin(x)  cos(x)  tan(x)     sine, cosine, tangent (x in radians); pi is 3.14159...\n"
    "  erf(x)  erfc(x)            error function, and 1 - erf(x)\n"
    "  gamma(x)  lgamma(x)        gamma function, and the log of its size (as R)\n"
    "  digamma(x)  beta(a, b)     digamma function, beta function (as R)\n"
    "  besselJ(x, nu)             Bessel function of the first kind, order nu (as R)\n"
    'Example: table.fit_formula("marks = a * hours + b")   (with sdata.example("marks"))'
)


def _fit_error(problem: str) -> SimplydataError:
    return SimplydataError(f"{problem}\n\n{FIT_HELP}")


class _FitReader(_FormulaReader):
    """The safe formula reader of add_column, turned into a NumPy function for fitting.

    The text is still read with ast and never run. Names that are columns are data; any
    other name is a value to find. Each piece becomes a small function f(values, data).
    """

    def __init__(self, text, columns):
        super().__init__(text, columns)
        self.values_to_find: list[str] = []
        self.finished = False
        self._depth = 0

    def _column(self, name):
        if name not in self.used:
            self.used.append(name)
        return lambda p, d: d[name]

    def _node(self, node):
        self._depth += 1
        try:
            out = self._fit_node(node)
        finally:
            self._depth -= 1
        if self._depth == 0:
            self.finished = True
        return out

    def _value_name(self, name):
        if (name in FIT_FUNCTIONS and name not in FIT_NAMES_FREE) or name in ("round",):
            raise _fit_error(f"{name} needs brackets after it, for example {name}(hours).")
        close = difflib.get_close_matches(name, [str(c) for c in self.columns], n=1, cutoff=0.8)
        if close and len(name) > 1:
            raise _fit_error(
                f'"{name}" is not a column, so it would be a value to find. Did you mean the column "{close[0]}"?\n'
                "If you really mean a value to find, give it a name not so like a column, such as k."
            )
        if name not in self.values_to_find:
            self.values_to_find.append(name)
        return lambda p, d: p[name]

    def _fit_node(self, node):
        if isinstance(node, ast.Constant):
            if not _is_number(node.value):
                raise _fit_error(f"{node.value!r} is not allowed in a formula: only numbers and names are.")
            value = float(node.value)
            return lambda p, d: value
        if isinstance(node, ast.Name):
            if node.id in self.placeholders:
                return self._column(self.placeholders[node.id])
            if node.id in self.columns:
                return self._column(node.id)
            if node.id in CONSTANTS:
                value = CONSTANTS[node.id]
                return lambda p, d: value
            return self._value_name(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self._node(node.operand)
            return (lambda p, d: -inner(p, d)) if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.BitXor):
                raise _fit_error("For a power use ** instead of ^, for example hours ** 2.")
            if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise _fit_error(f'The sign in "{self._snippet(node)}" is not allowed in a formula to fit.')
            left, right = self._node(node.left), self._node(node.right)
            if isinstance(op, ast.Add):
                return lambda p, d: left(p, d) + right(p, d)
            if isinstance(op, ast.Sub):
                return lambda p, d: left(p, d) - right(p, d)
            if isinstance(op, ast.Mult):
                return lambda p, d: left(p, d) * right(p, d)
            if isinstance(op, ast.Div):
                return lambda p, d: left(p, d) / right(p, d)
            return lambda p, d: np.power(left(p, d), right(p, d))
        if isinstance(node, ast.Call):
            return self._fit_call(node)
        if isinstance(node, ast.Attribute):
            raise _fit_error(f'"{self._snippet(node)}" is not allowed: a formula cannot use a dot after a name.')
        raise _fit_error(f'"{self._snippet(node)}" is not allowed in a formula.')

    def _fit_call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in FUNCTION_SPELLINGS:
            node.func.id = FUNCTION_SPELLINGS[node.func.id]
        if not isinstance(node.func, ast.Name) or node.func.id not in FIT_FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else self._snippet(node.func)
            guess = difflib.get_close_matches(name, list(FIT_FUNCTIONS), n=1)
            hint = f" Did you mean {guess[0]}()?" if guess else ""
            raise _fit_error(f'"{name}()" is not allowed in a formula to fit.{hint}')
        name = node.func.id
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise _fit_error(f"Write {name}() simply, like {name}(hours).")
        low, high = FIT_FUNCTIONS[name]
        if not low <= len(node.args) <= high:
            given = f"{len(node.args)} thing" + ("" if len(node.args) == 1 else "s")
            raise _fit_error(f"{name}() was given {given} inside the brackets.")
        if name in SPECIAL_FUNCTIONS:
            inside = [self._node(a) for a in node.args]
            return lambda p, d: special_function(name, *(f(p, d) for f in inside))
        inner = self._node(node.args[0])
        if name == "exp":
            return lambda p, d: np.exp(inner(p, d))
        if name == "sqrt":
            return lambda p, d: np.sqrt(inner(p, d))
        if name == "abs":
            return lambda p, d: np.abs(inner(p, d))
        if name == "sin":
            return lambda p, d: np.sin(inner(p, d))
        if name == "cos":
            return lambda p, d: np.cos(inner(p, d))
        if len(node.args) == 1:
            return lambda p, d: np.log(inner(p, d))
        base = node.args[1]
        if not (isinstance(base, ast.Constant) and _is_number(base.value)) or base.value <= 0 or base.value == 1:
            raise _fit_error("The base of log() must be a positive number other than 1, for example log(x, 10).")
        scale = math.log(float(base.value))
        return lambda p, d: np.log(inner(p, d)) / scale


# ---------------------------------------------------------------------------- the words
# fit_line, fit_curve and fit_formula live in fitting.py (FittingWords, a word group of Table).
class StatTestWords:
    # ------------------------------------------------------------------ helpers
    def _result_table(self, row: dict):
        """A one-row result table, on the same engine as this table."""
        frame = nw.from_dict({k: [v] for k, v in row.items()}, backend=self._frame.implementation)
        return self._new_table(frame)

    @staticmethod
    def _need_name(step: str, option: str, value, example: str):
        if not isinstance(value, str):
            raise SimplydataError(f"{step}() needs {option} as a column name in quotes.\nFor example, {example}")

    def _check_based_on(self, step, based_on):
        self._check_choice(step, "based_on", based_on, BASED_ON)

    def _number_values(self, step: str, columns: list[str]):
        """The columns as float arrays, rows with an empty cell in any of them skipped (with a note)."""
        frame = self._number_frame(step, columns)
        kept = frame.select(*columns).drop_nulls()
        _skipped_note(len(frame) - len(kept), columns)
        return [kept[c].to_numpy().astype(float) for c in columns]

    def _groups(self, step: str, column, for_each, example: str):
        """[(group name, values)] sorted by group; empty cells skipped with a note."""
        self._need_name(step, "the column", column, example)
        self._need_name(step, "for_each=", for_each, example)
        self._need_columns(column, for_each)
        if column == for_each:
            raise SimplydataError(f'"{column}" cannot be both the column to compare and the for_each= column.\n'
                                  f"For example, {example}")
        frame = self._number_frame(step, [column])
        if frame.schema[for_each].is_float():
            frame = frame.with_columns(nw.col(for_each).fill_nan(None))
        kept = frame.select(for_each, column).drop_nulls()
        _skipped_note(len(frame) - len(kept), [column, for_each])
        keys = kept[for_each].to_list()
        values = kept[column].to_numpy().astype(float)
        order = sorted(set(keys), key=lambda k: (str(type(k)), k))
        keys = np.asarray(keys, dtype=object)
        groups = [(_label(k), values[keys == k]) for k in order]
        if len(groups) < 2:
            found = f'only "{groups[0][0]}"' if groups else "none"
            raise SimplydataError(f'{step}() needs at least 2 groups in "{for_each}"; it found {found}.')
        small = [name for name, v in groups if len(v) < 2]
        if small:
            raise SimplydataError(
                f"{step}() needs at least 2 values in each group; "
                + _list_words(f'"{s}"' for s in small) + f' in "{for_each}" has fewer.\n'
                "Remove that group first, for example with table.remove_rows(...)."
            )
        return groups

    # ------------------------------------------------------------------ compare_averages
    @friendly(renamed={"between": "for_each"})
    def compare_averages(self, column=None, for_each=None, based_on="averages", test=None):
        """Are the averages of the groups really different, or could it be chance?

        table.compare_averages("marks", for_each="class")   (with sdata.example("marks"))
        Two groups: Welch's t-test (the R t.test() default). Three or more: Welch's one-way
        ANOVA (the R oneway.test() default). Neither needs the groups to have the same spread.
        based_on="ranks" uses ranks instead (Mann-Whitney U or Kruskal-Wallis): safer when
        groups are small and their values are not bell-shaped. test= chooses another test:
        "welch", "student", "welch anova", "anova", "mann-whitney" or "kruskal-wallis".
        Prints the answer in words (with the test used and why) and gives back a one-row
        table of results. The effect size says how big the difference is: Hedges' g (the
        difference in standard deviations) for 2 groups, omega-squared (the share of the
        differences explained by the groups) for 3 or more. Each group is checked for a
        bell shape (Shapiro-Wilk) and a note says if it does not look like one; with 30 or
        more values that matters little (the central limit theorem).
        To see which groups differ, use compare_each_pair().
        """
        step = "compare_averages"
        example = 'with sdata.example("marks"): table.compare_averages("marks", for_each="class")'
        self._check_based_on(step, based_on)
        if test is not None:
            self._check_choice(step, "test", test, AVERAGE_TESTS)
            if based_on == "ranks" and test not in RANK_TESTS:
                raise SimplydataError(
                    f'test="{test}" compares averages, but based_on="ranks" asks for ranks. '
                    'Use one of them: test="mann-whitney" or test="kruskal-wallis" are the rank tests.'
                )
        groups = self._groups(step, column, for_each, example)
        k = len(groups)
        if test is None:
            if based_on == "ranks":
                test = "mann-whitney" if k == 2 else "kruskal-wallis"
            else:
                test = "welch" if k == 2 else "welch anova"
            chosen = False
        else:
            chosen = True
        if test in TWO_GROUP_TESTS and k != 2:
            other = "kruskal-wallis" if test == "mann-whitney" else "welch anova"
            raise SimplydataError(
                f'test="{test}" compares exactly 2 groups, but "{for_each}" has {k}.\n'
                f'Use test="{other}" for all groups at once, or table.compare_each_pair("{column}", for_each="{for_each}").'
            )
        names = [g[0] for g in groups]
        arrays = [g[1] for g in groups]
        what = f'the average of "{column}" in each "{for_each}" ({", ".join(names)})'
        row = {"test": None, "what it compares": what}
        means = [float(v.mean()) for v in arrays]
        if test not in RANK_TESTS:
            spread = [float(v.std(ddof=1)) for v in arrays]
            if all(s == 0 for s in spread) or (test in ("welch anova",) and any(s == 0 for s in spread)):
                flat = [n for n, s in zip(names, spread) if s == 0]
                groups = _list_words('"' + str(n) + '"' for n in flat)
                raise SimplydataError(
                    f'Every value is the same within {groups}, so a test on averages '
                    f'cannot measure the spread there. Use based_on="ranks", or check the data.'
                )
        from scipy import stats

        notes = []
        if test in ("welch", "student"):
            a, b = arrays
            result = stats.ttest_ind(a, b, equal_var=(test == "student"))
            low, high = result.confidence_interval(0.95)
            t, p, df = float(result.statistic), float(result.pvalue), float(result.df)
            n1, n2 = len(a), len(b)
            pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
            g = (a.mean() - b.mean()) / pooled * _hedges_correction(n1 + n2 - 2) if pooled > 0 else math.nan
            diff = means[0] - means[1]
            name = "Welch's t-test" if test == "welch" else "Student's t-test"
            why = (
                "the usual test for two averages; it does not assume both groups have the same spread"
                if test == "welch" else "chosen with test=; it assumes both groups have the same spread"
            )
            row.update({"test": name, "statistic name": "t", "statistic": t, "degrees of freedom": df,
                        "p-value": p, "effect size name": "difference in standard deviations (Hedges' g)",
                        "effect size": float(g), "difference": diff,
                        "difference 95 % CI low": float(low), "difference 95 % CI high": float(high)})
            direction = "higher" if diff > 0 else "lower"
            first = (f'The average "{column}" is {_num(means[0])} in {names[0]} and {_num(means[1])} in {names[1]}: '
                     f"{names[0]} is {_num(abs(diff))} {direction} (95 % CI of the difference {_num(low)} to "
                     f"{_num(high)}; Hedges' g = {_num(g)}, a difference of {_num(abs(g))} standard deviations). "
                     f"A difference this big {_chance(p)}.")
            second = f"Test: {name} (t = {_num(t)}, df = {_whole(df)}), {why}."
            notes = _shape_notes([(f'group "{n}"', v) for n, v in groups])
        elif test in ("welch anova", "anova"):
            if test == "welch anova":
                f, df1, df2, p = _welch_anova(arrays)
                name = "Welch's one-way ANOVA"
                why = ("the usual test for three or more averages; it does not assume the groups have the same spread"
                       if not chosen else "it does not assume the groups have the same spread")
            else:
                result = stats.f_oneway(*arrays)
                f, p = float(result.statistic), float(result.pvalue)
                df1, df2 = k - 1.0, float(sum(len(v) for v in arrays) - k)
                name = "one-way ANOVA"
                why = "chosen with test=; it assumes the groups have the same spread"
            omega = _omega_squared(arrays)
            row.update({"test": name, "statistic name": "F", "statistic": f, "degrees of freedom 1": df1,
                        "degrees of freedom 2": df2, "p-value": p,
                        "effect size name": "share of the differences explained by the groups (omega-squared)",
                        "effect size": omega})
            averages = _list_words(f"{_num(m)} in {n}" for n, m in zip(names, means))
            share = f"about {_percent(omega)}" if omega > 0 else "about 0 % (omega-squared below 0 is read as 0)"
            first = (f'The average "{column}" is {averages}. Differences this big between the groups {_chance(p, plural=True)}. '
                     f'Omega-squared = {_num(omega)}: the groups explain {share} of the differences in "{column}".')
            second = f"Test: {name} (F = {_num(f)}, df = {_whole(df1)} and {_whole(df2)}), {why}."
            if k > 2:
                second += f' To see which groups differ: table.compare_each_pair("{column}", for_each="{for_each}").'
            notes = _shape_notes([(f'group "{n}"', v) for n, v in groups])
        elif test == "mann-whitney":
            a, b = arrays
            u, p, r, how = _mann_whitney(a, b)
            medians = [float(np.median(v)) for v in arrays]
            row.update({"test": "Mann-Whitney U test", "statistic name": "U", "statistic": u, "p-value": p,
                        "effect size name": "how far one group's values lie above the other's, -1 to 1 "
                                            "(rank-biserial correlation)",
                        "effect size": r})
            lean = f"the values in {names[0]} tend to be higher" if r > 0 else (
                f"the values in {names[0]} tend to be lower" if r < 0 else "neither group tends to be higher")
            first = (f'The median "{column}" is {_num(medians[0])} in {names[0]} and {_num(medians[1])} in {names[1]}: '
                     f"{lean} (rank-biserial correlation = {_num(r)}). A difference this big {_chance(p)}.")
            second = (f"Test: Mann-Whitney U test (U = {_whole(u)}, {how}), which compares ranks rather than "
                      "averages, so it does not need bell-shaped values.")
        else:
            result = stats.kruskal(*arrays)
            h, p = float(result.statistic), float(result.pvalue)
            n_all = sum(len(v) for v in arrays)
            eps = h / (n_all - 1)
            medians = [float(np.median(v)) for v in arrays]
            row.update({"test": "Kruskal-Wallis test", "statistic name": "H", "statistic": h,
                        "degrees of freedom": float(k - 1), "p-value": p,
                        "effect size name": "share of the rank differences explained by the groups "
                                            "(rank epsilon-squared)",
                        "effect size": eps})
            first = (f'The median "{column}" is {_list_words(f"{_num(m)} in {n}" for n, m in zip(names, medians))}. '
                     f"Differences this big between the groups {_chance(p, plural=True)}. Epsilon-squared = {_num(eps)}: the "
                     f"groups explain about {_percent(eps)} of the differences in the ranks.")
            second = (f"Test: Kruskal-Wallis test (H = {_num(h)}, df = {k - 1}), which compares ranks rather than "
                      "averages, so it does not need bell-shaped values.")
            if k > 2:
                second += (f' To see which groups differ: table.compare_each_pair("{column}", for_each="{for_each}", '
                           'based_on="ranks").')
        for name, v in groups:
            row[f"count in {name}"] = len(v)
            row[f"average in {name}"] = float(v.mean())
            if test in RANK_TESTS:
                row[f"median in {name}"] = float(np.median(v))
        print(first)
        print(second)
        for note in notes:
            print(note)
        return self._result_table(row)

    # ------------------------------------------------------------------ compare_each_pair
    @friendly(renamed={"between": "for_each"})
    def compare_each_pair(self, column=None, for_each=None, based_on="averages", test=None):
        """Which pairs of groups differ? Every pair is compared, with Holm's correction.

        table.compare_each_pair("marks", for_each="class")   (with sdata.example("marks"))
        Each pair gets Welch's t-test (or Mann-Whitney U with based_on="ranks"). Testing many
        pairs makes a false alarm more likely, so the p-values are adjusted with Holm's
        method: the chance of at least one false alarm stays at 5 %. This is the same as R
        pairwise.t.test(pool.sd = FALSE, p.adjust.method = "holm"). Gives back a table with
        one row per pair.
        test="tukey" gives Tukey's honest significant difference instead (R
        TukeyHSD(aov(marks ~ class)), the Tukey-Kramer form when the groups differ in size):
        the textbook follow-up to a one-way ANOVA, with 95 % confidence intervals that hold
        for all pairs at once. It assumes the groups have the same spread; the default does
        not. test="welch" and test="mann-whitney" name the other two choices.
        """
        step = "compare_each_pair"
        example = 'with sdata.example("marks"): table.compare_each_pair("marks", for_each="class")'
        self._check_based_on(step, based_on)
        if test is not None:
            self._check_choice(step, "test", test, PAIR_TESTS)
            if based_on == "ranks" and test != "mann-whitney":
                raise SimplydataError(
                    f'test="{test}" compares averages, but based_on="ranks" asks for ranks. Use one of them: '
                    'test="mann-whitney" is the rank test.'
                )
            if test == "mann-whitney":
                based_on = "ranks"
        groups = self._groups(step, column, for_each, example)
        if test == "tukey":
            return self._tukey_pairs(column, for_each, groups)
        from scipy import stats

        rows = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                (n1, a), (n2, b) = groups[i], groups[j]
                row = {"group 1": n1, "group 2": n2}
                if based_on == "averages":
                    if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
                        raise SimplydataError(
                            f'Every value is the same within "{n1}" and within "{n2}", so a t-test cannot '
                            'measure the spread. Use based_on="ranks", or check the data.'
                        )
                    result = stats.ttest_ind(a, b, equal_var=False)
                    low, high = result.confidence_interval(0.95)
                    row.update({"test": "Welch's t-test", "difference": float(a.mean() - b.mean()),
                                "difference 95 % CI low": float(low), "difference 95 % CI high": float(high),
                                "statistic name": "t", "statistic": float(result.statistic),
                                "degrees of freedom": float(result.df), "p-value": float(result.pvalue)})
                else:
                    u, p, r, _ = _mann_whitney(a, b)
                    row.update({"test": "Mann-Whitney U test",
                                "difference in medians": float(np.median(a) - np.median(b)),
                                "statistic name": "U", "statistic": u, "p-value": p,
                                "rank-biserial correlation": r})
                row.update({"count in group 1": len(a), "count in group 2": len(b)})
                rows.append(row)
        adjusted = _holm([r["p-value"] for r in rows])
        for row, p in zip(rows, adjusted):
            row["p-value (Holm)"] = p
        test = "Welch's t-test" if based_on == "averages" else "Mann-Whitney U test"
        pairs = "pair" if len(rows) == 1 else "pairs"
        print(f'Each pair of "{for_each}" groups compared on "{column}" with {test}; the p-values are adjusted for '
              f"{len(rows)} {pairs} with Holm's method, so the chance of any false alarm stays at 5 %.")
        for row in rows:
            if based_on == "averages":
                diff, word = row["difference"], "average"
            else:
                diff, word = row["difference in medians"], "median"
            direction = "higher" if diff > 0 else "lower"
            words = _p_words(row["p-value (Holm)"]).replace("(p ", "(Holm-adjusted p ")
            print(f"  {row['group 1']} vs {row['group 2']}: the {word} of {row['group 1']} is {_num(abs(diff))} "
                  f"{direction}; {words}.")
        frame = nw.from_dict({key: [r[key] for r in rows] for key in rows[0]}, backend=self._frame.implementation)
        return self._new_table(frame)

    def _tukey_pairs(self, column, for_each, groups):
        """compare_each_pair(test="tukey"): Tukey HSD (Tukey-Kramer for unequal sizes), as R TukeyHSD()."""
        from scipy import stats

        names = [g[0] for g in groups]
        arrays = [g[1] for g in groups]
        spreads = [float(v.std(ddof=1)) for v in arrays]
        if all(s == 0 for s in spreads):
            raise SimplydataError(
                "Every value is the same within each group, so Tukey's method cannot measure the spread. "
                'Use based_on="ranks", or check the data.'
            )
        result = stats.tukey_hsd(*arrays)
        interval = result.confidence_interval(0.95)
        rows = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                rows.append({"group 1": names[i], "group 2": names[j], "test": "Tukey HSD",
                             "difference": float(result.statistic[i, j]),
                             "difference 95 % CI low": float(interval.low[i, j]),
                             "difference 95 % CI high": float(interval.high[i, j]),
                             "p-value (Tukey)": float(result.pvalue[i, j]),
                             "count in group 1": len(arrays[i]), "count in group 2": len(arrays[j])})
        pairs = "pair" if len(rows) == 1 else "pairs"
        sizes = len({len(v) for v in arrays}) > 1
        kramer = " (the Tukey-Kramer form, as the groups differ in size)" if sizes else ""
        print(f'Each pair of "{for_each}" groups compared on "{column}" with Tukey\'s honest significant difference'
              f"{kramer}, as R TukeyHSD(aov({column} ~ {for_each})): the p-values and 95 % confidence intervals hold "
              f"for all {len(rows)} {pairs} at once, so the chance of any false alarm stays at 5 %. It assumes the "
              "groups have the same spread.")
        for row in rows:
            diff = row["difference"]
            direction = "higher" if diff > 0 else "lower"
            words = _p_words(row["p-value (Tukey)"]).replace("(p ", "(Tukey-adjusted p ")
            print(f"  {row['group 1']} vs {row['group 2']}: the average of {row['group 1']} is {_num(abs(diff))} "
                  f"{direction} (95 % CI {_num(row['difference 95 % CI low'])} to "
                  f"{_num(row['difference 95 % CI high'])}); {words}.")
        smallest = min(spreads)
        ratio = max(spreads) / smallest if smallest > 0 else math.inf
        if ratio > SPREAD_RATIO:
            shown = "many" if math.isinf(ratio) else _num(ratio)
            print(f"Note: the biggest standard deviation is {shown} times the smallest, so the groups may not have "
                  f'the same spread. Check with table.compare_spreads("{column}", for_each="{for_each}"), or leave '
                  "out test= for Welch's t-tests with Holm's correction, which do not assume it.")
        for note in _shape_notes([(f'group "{n}"', v) for n, v in groups]):
            print(note)
        frame = nw.from_dict({key: [r[key] for r in rows] for key in rows[0]}, backend=self._frame.implementation)
        return self._new_table(frame)

    # ------------------------------------------------------------------ compare_spreads
    @friendly
    def compare_spreads(self, column=None, for_each=None, test="levene"):
        """Do the groups differ in how spread out their values are, more than chance?

        table.compare_spreads("marks", for_each="class")   (with sdata.example("marks"))
        test="levene" (the default): Levene's test centred on each group's median, also
        called the Brown-Forsythe test. It is the default of R car::leveneTest() and SciPy
        levene(center="median"), and copes well with values that are not bell-shaped.
        test="bartlett": Bartlett's test (R bartlett.test()), a little stronger when the
        values are bell-shaped but easily fooled when they are not. test="fligner": the
        Fligner-Killeen test (R fligner.test()), based on ranks, the safest with outliers.
        Empty cells are skipped. Prints the standard deviation of each group and the answer
        in words; gives back a one-row table of results. compare_averages() uses Welch's
        test by default, which does not need equal spreads.
        """
        step = "compare_spreads"
        example = 'with sdata.example("marks"): table.compare_spreads("marks", for_each="class")'
        chosen = _plain_name(test)
        chosen = SPREAD_ALIASES.get(chosen, chosen)
        if chosen not in SPREAD_TESTS:
            raise choice_error(step, "test", test, list(SPREAD_TESTS))
        groups = self._groups(step, column, for_each, example)
        names = [g[0] for g in groups]
        arrays = [g[1] for g in groups]
        spreads = [float(v.std(ddof=1)) for v in arrays]
        if all(s == 0 for s in spreads):
            raise SimplydataError(f'Every value is the same within each group of "{for_each}", so there is no '
                                  "spread to compare. Check the data.")
        from scipy import stats

        what = f'the spread of "{column}" in each "{for_each}" ({", ".join(names)})'
        row = {"test": None, "what it compares": what}
        if chosen == "levene":
            result = stats.levene(*arrays, center="median")
            stat, p = float(result.statistic), float(result.pvalue)
            if not math.isfinite(stat):
                raise SimplydataError(f'Every value lies the same distance from its group\'s median in "{column}", '
                                      'so Levene\'s test cannot work here. Try test="fligner".')
            df1, df2 = float(len(arrays) - 1), float(sum(len(v) for v in arrays) - len(arrays))
            name = "Levene's test (Brown-Forsythe, centred on the medians)"
            row.update({"test": name, "statistic name": "F", "statistic": stat, "degrees of freedom 1": df1,
                        "degrees of freedom 2": df2, "p-value": p})
            second = (f"Test: Levene's test centred on the medians, the Brown-Forsythe form (F = {_num(stat)}, "
                      f"df = {_whole(df1)} and {_whole(df2)}), as R car::leveneTest() "
                      'and SciPy levene(center="median"); it copes well with values that are not bell-shaped.')
        elif chosen == "bartlett":
            flat = [n for n, s in zip(names, spreads) if s == 0]
            if flat:
                raise SimplydataError(
                    f"Every value is the same within {_list_words(repr(n) for n in flat)}, which Bartlett's test "
                    'cannot handle. Use test="levene" (the default) or test="fligner".'
                )
            result = stats.bartlett(*arrays)
            stat, p = float(result.statistic), float(result.pvalue)
            df = float(len(arrays) - 1)
            name = "Bartlett's test"
            row.update({"test": name, "statistic name": "Bartlett's K-squared", "statistic": stat,
                        "degrees of freedom": df, "p-value": p})
            second = (f"Test: {name} (K-squared = {_num(stat)}, df = {_whole(df)}), as R bartlett.test(); "
                      "it assumes bell-shaped values.")
        else:
            result = stats.fligner(*arrays, center="median")
            stat, p = float(result.statistic), float(result.pvalue)
            df = float(len(arrays) - 1)
            name = "Fligner-Killeen test"
            row.update({"test": name, "statistic name": "chi-square", "statistic": stat, "degrees of freedom": df,
                        "p-value": p})
            second = (f"Test: {name} (chi-square = {_num(stat)}, df = {_whole(df)}), as R fligner.test(); it uses "
                      "ranks, so it does not need bell-shaped values and copes with outliers.")
        for n, v, s in zip(names, arrays, spreads):
            row[f"count in {n}"] = len(v)
            row[f"standard deviation in {n}"] = s
        listed = _list_words(f"{_num(s)} in {n}" for n, s in zip(names, spreads))
        print(f'The standard deviation of "{column}" is {listed}. Differences in spread this big between the groups '
              f"{_chance(p, plural=True)}.")
        print(second)
        if chosen == "bartlett":
            for n, v in groups:
                if 3 <= len(v) <= SHAPIRO_LIMIT and np.ptp(v) > 0:
                    sw = float(stats.shapiro(v).pvalue)
                    if sw < 0.05:
                        print(f'Note: the values in group "{n}" don\'t look bell-shaped (Shapiro-Wilk {_p_shown(sw)}); '
                              "Bartlett's test is easily fooled by that, whatever the group size. A safer choice: "
                              'test="levene" (the default).')
        if p < 0.05:
            print("Note: compare_averages() uses Welch's test by default, which does not need equal spreads; "
                  'test="student", test="anova" and compare_each_pair(test="tukey") do need them.')
        return self._result_table(row)

    # ------------------------------------------------------------------ paired and one-sample
    def _one_sample(self, step, values, label, against, based_on, what, notes_label):
        """Shared by compare_before_after (changes against 0) and compare_average_to."""
        from scipy import stats

        row = {"test": None, "what it compares": what}
        if based_on == "averages":
            if values.std(ddof=1) == 0:
                raise SimplydataError(
                    f"Every {label} is the same ({_num(values[0])}), so there is no spread to test with. "
                    "Check the data."
                )
            result = stats.ttest_1samp(values, against)
            low, high = result.confidence_interval(0.95)
            t, p, df = float(result.statistic), float(result.pvalue), float(result.df)
            d = (values.mean() - against) / values.std(ddof=1)
            return row, dict(t=t, p=p, df=df, low=float(low), high=float(high), d=float(d),
                             notes=_shape_notes([(notes_label, values)], what=f"the {label}s"))
        found = _signed_rank(values - against)
        if found is None:
            raise SimplydataError(f"Every {label} is exactly {_num(against)}, so there is nothing to test.")
        v, p, r, how, zeros = found
        return row, dict(v=v, p=p, r=r, how=how, zeros=zeros, notes=[])

    @friendly
    def compare_before_after(self, before=None, after=None, based_on="averages"):
        """Did the values change from one column to the other, row by row, more than chance?

        table.compare_before_after("Maths", "Science")   (with sdata.example("marks"))
        For two measurements of the same thing, such as each child's mark before and after
        a course. Uses the change in each row (after minus before): a paired t-test, or the
        Wilcoxon signed-rank test with based_on="ranks". Rows with an empty cell in either
        column are skipped. Effect size: Cohen's d_z, the average change in standard
        deviations of the changes. Gives back a one-row table of results.
        """
        step = "compare_before_after"
        example = 'table.compare_before_after("before", "after")'
        self._need_name(step, "before", before, example)
        self._need_name(step, "after", after, example)
        self._check_based_on(step, based_on)
        if before == after:
            raise SimplydataError(f"{step}() needs two different columns, for example {example}.")
        b, a = self._number_values(step, [before, after])
        if len(a) < 2:
            raise SimplydataError(f'{step}() needs at least 2 rows with a number in both "{before}" and "{after}".')
        changes = a - b
        what = f'the change from "{before}" to "{after}" in each row'
        row, r = self._one_sample(step, changes, "change", 0.0, based_on, what, "the changes")
        mean_change = float(changes.mean())
        if based_on == "averages":
            direction = "higher" if mean_change > 0 else "lower"
            row.update({"test": "paired t-test", "statistic name": "t", "statistic": r["t"],
                        "degrees of freedom": r["df"], "p-value": r["p"],
                        "effect size name": "average change in standard deviations of the changes (Cohen's d_z)",
                        "effect size": r["d"], "average change": mean_change,
                        "average change 95 % CI low": r["low"], "average change 95 % CI high": r["high"]})
            first = (f'On average "{after}" is {_num(abs(mean_change))} {direction} than "{before}" (95 % CI of the '
                     f"average change {_num(r['low'])} to {_num(r['high'])}; Cohen's d_z = {_num(r['d'])}). "
                     f"A change this big {_chance(r['p'])}.")
            second = (f"Test: paired t-test (t = {_num(r['t'])}, df = {_whole(r['df'])}), the usual test when each row "
                      "is measured twice; it looks at the change in each row.")
        else:
            median_change = float(np.median(changes))
            row.update({"test": "Wilcoxon signed-rank test", "statistic name": "V", "statistic": r["v"],
                        "p-value": r["p"],
                        "effect size name": "how far the changes lean up rather than down, -1 to 1 "
                                            "(matched-pairs rank-biserial correlation)",
                        "effect size": r["r"], "median change": median_change})
            lean = "up" if r["r"] > 0 else "down" if r["r"] < 0 else "neither way"
            first = (f'The median change from "{before}" to "{after}" is {_num(median_change)}; the changes lean '
                     f"{lean} (rank-biserial correlation = {_num(r['r'])}). A change this big {_chance(r['p'])}.")
            second = (f"Test: Wilcoxon signed-rank test (V = {_whole(r['v'])}, {r['how']}), which uses the ranks of "
                      "the changes, so it does not need bell-shaped values.")
            if r["zeros"]:
                second += f" {r['zeros']} rows with no change were left out, as R does."
        row.update({"rows": len(changes), f"average of {before}": float(b.mean()), f"average of {after}": float(a.mean())})
        print(first)
        print(second)
        for note in r["notes"]:
            print(note)
        return self._result_table(row)

    @friendly
    def compare_average_to(self, column=None, value=None, based_on="averages"):
        """Is the average of a column different from a number you choose, or could it be chance?

        table.compare_average_to("marks", 60)   (with sdata.example("marks"))
        A one-sample t-test, or the Wilcoxon signed-rank test (on the median) with
        based_on="ranks". Empty cells are skipped. Effect size: Cohen's d, the difference
        in standard deviations. Gives back a one-row table of results.
        """
        step = "compare_average_to"
        example = 'with sdata.example("marks"): table.compare_average_to("marks", 60)'
        self._need_name(step, "the column", column, example)
        if not _is_number(value) or not math.isfinite(value):
            raise SimplydataError(f"{step}() needs a number to compare with, for example {example}.")
        self._check_based_on(step, based_on)
        (values,) = self._number_values(step, [column])
        if len(values) < 2:
            raise SimplydataError(f'{step}() needs at least 2 numbers in "{column}".')
        shown = f"{value:g}"
        what = f'the average of "{column}" against {shown}'
        row, r = self._one_sample(step, values, f'value of "{column}"', float(value), based_on, what,
                                  f'"{column}"')
        mean = float(values.mean())
        if based_on == "averages":
            diff = mean - value
            direction = "higher" if diff > 0 else "lower"
            row.update({"test": "one-sample t-test", "statistic name": "t", "statistic": r["t"],
                        "degrees of freedom": r["df"], "p-value": r["p"],
                        "effect size name": "difference in standard deviations (Cohen's d)",
                        "effect size": r["d"], "average": mean,
                        "average 95 % CI low": r["low"], "average 95 % CI high": r["high"]})
            first = (f'The average "{column}" is {_num(mean)}, which is {_num(abs(diff))} {direction} than '
                     f"{shown} (95 % CI of the average {_num(r['low'])} to {_num(r['high'])}; "
                     f"Cohen's d = {_num(r['d'])}). A difference this big {_chance(r['p'])}.")
            second = (f"Test: one-sample t-test (t = {_num(r['t'])}, df = {_whole(r['df'])}), the usual test for "
                      "one average against a fixed number.")
        else:
            median = float(np.median(values))
            row.update({"test": "Wilcoxon signed-rank test", "statistic name": "V", "statistic": r["v"],
                        "p-value": r["p"],
                        "effect size name": "how far the values lean above rather than below the number, -1 to 1 "
                                            "(rank-biserial correlation)",
                        "effect size": r["r"], "median": median, "average": mean})
            first = (f'The median "{column}" is {_num(median)}, against {shown} (rank-biserial correlation = '
                     f"{_num(r['r'])}). A difference this big {_chance(r['p'])}.")
            second = (f"Test: Wilcoxon signed-rank test (V = {_whole(r['v'])}, {r['how']}), which uses ranks, so it "
                      "does not need bell-shaped values.")
            if r["zeros"]:
                second += f" {r['zeros']} values exactly equal to {shown} were left out, as R does."
        row["rows"] = len(values)
        print(first)
        print(second)
        for note in r["notes"]:
            print(note)
        return self._result_table(row)

    # ------------------------------------------------------------------ compare_counts
    @friendly(renamed={"between": "for_each"})
    def compare_counts(self, column=None, for_each=None, test=None):
        """Does how often each value turns up differ between groups, more than chance?

        table.add_group_column("result", from_column="marks", cut_at=[60], names=["below 60", "60 or more"])
        table.compare_counts("result", for_each="class")   (with sdata.example("marks"))
        (Do the classes differ in how many children score 60 or more?) Counts the rows for
        each pair of values and uses the chi-square test of independence (as R chisq.test():
        Yates' continuity correction for 2 x 2 tables only).
        If an expected count is below 5 a note says so; for a 2 x 2 table test="fisher" then
        gives Fisher's exact test. Empty cells are skipped. Effect size: Cramer's V, the
        strength of the link from 0 (none) to 1 (complete). Gives back a one-row table.
        """
        step = "compare_counts"
        example = 'with sdata.example("fruit"): table.compare_counts("fruit", for_each="shop")'
        self._need_name(step, "the column", column, example)
        self._need_name(step, "for_each=", for_each, example)
        self._need_columns(column, for_each)
        if test is not None:
            self._check_choice(step, "test", test, COUNT_TESTS)
        if column == for_each:
            raise SimplydataError(f"{step}() needs two different columns, for example {example}.")
        frame = self._frame
        floats = [c for c in (column, for_each) if frame.schema[c].is_float()]
        if floats:
            frame = frame.with_columns(nw.col(c).fill_nan(None) for c in floats)
        kept = frame.select(column, for_each).drop_nulls()
        _skipped_note(len(frame) - len(kept), [column, for_each])
        rows_of = kept[column].to_list()
        cols_of = kept[for_each].to_list()
        order = lambda keys: sorted(set(keys), key=lambda k: (str(type(k)), k))  # noqa: E731
        row_keys, col_keys = order(rows_of), order(cols_of)
        if len(row_keys) < 2 or len(col_keys) < 2:
            few = column if len(row_keys) < 2 else for_each
            raise SimplydataError(f'{step}() needs at least 2 different values in "{few}".')
        observed = np.zeros((len(row_keys), len(col_keys)), dtype=np.int64)
        ri = {k: i for i, k in enumerate(row_keys)}
        ci = {k: i for i, k in enumerate(col_keys)}
        for r_, c_ in zip(rows_of, cols_of):
            observed[ri[r_], ci[c_]] += 1
        from scipy import stats

        chi = stats.chi2_contingency(observed, correction=True)
        expected = chi.expected_freq
        n = int(observed.sum())
        plain_chi = stats.chi2_contingency(observed, correction=False).statistic
        v = math.sqrt(plain_chi / (n * (min(observed.shape) - 1)))
        two_by_two = observed.shape == (2, 2)
        what = f'how often each "{column}" turns up in each "{for_each}"'
        row = {"test": None, "what it compares": what}
        size = f"{observed.shape[0]} x {observed.shape[1]}"
        if test == "fisher":
            if not two_by_two:
                raise SimplydataError(
                    f'test="fisher" works on 2 x 2 tables here, but "{column}" and "{for_each}" make a {size} table.\n'
                    "Use the chi-square test (leave out test=), or merge groups first."
                )
            result = stats.fisher_exact(observed)
            p = float(result.pvalue)
            row.update({"test": "Fisher's exact test", "statistic name": "odds ratio (sample)",
                        "statistic": float(result.statistic), "p-value": p})
            second = ("Test: Fisher's exact test (chosen with test=), which gives an exact p-value, so it is safe "
                      "with small counts.")
        else:
            stat, p, df = float(chi.statistic), float(chi.pvalue), int(chi.dof)
            name = "chi-square test of independence"
            row.update({"test": name, "statistic name": "chi-square", "statistic": stat,
                        "degrees of freedom": float(df), "p-value": p})
            yates = " with Yates' continuity correction (as R chisq.test() does for 2 x 2 tables)" if two_by_two else ""
            second = f"Test: {name}{yates} (chi-square = {_num(stat)}, df = {df}), the usual test for counts."
        row.update({"effect size name": "strength of the link, 0 (none) to 1 (complete) (Cramer's V)",
                    "effect size": v, "rows": n, "smallest expected count": float(expected.min())})
        first = (f'Comparing {what} ({size} table, {n} rows): the differences {_chance(p, plural=True)}. '
                 f"Cramer's V = {_num(v)} (0 means no link, 1 a complete link).")
        print(first)
        print(second)
        low = int((expected < 5).sum())
        if low and test != "fisher":
            counts = "count is" if low == 1 else "counts are"
            note = (f"Note: {low} of the {expected.size} expected {counts} below 5, so the chi-square p-value "
                    "may be off. ")
            note += ('Fisher\'s exact test is safer here: test="fisher".' if two_by_two
                     else "Merge small groups, or collect more data.")
            print(note)
        return self._result_table(row)

    # ------------------------------------------------------------------ one column of values against chosen shares
    def _filled_values(self, step, column, example):
        """The values of one column (any kind) without empty cells, which are skipped with a note."""
        self._need_name(step, "the column", column, example)
        self._need_columns(column)
        kept = self._frame.filter(~self._empty(column))
        _skipped_note(len(self._frame) - len(kept), [column])
        return [_plain(v) for v in kept[column].to_list()]

    @friendly
    def compare_counts_to(self, column=None, expected=None):
        """Does how often each value turns up match the shares you expect, more than chance?

        table.compare_counts_to("breakfast", expected={"yes": 0.6, "no": 0.4})   (with sdata.example("school"))
        The chi-square goodness-of-fit test (as R chisq.test(x, p = ...)). expected= gives
        the share of each value, adding up to 1 (or percents adding up to 100); for a ratio
        such as 9:3:3:1 write {"a": 9/16, "b": 3/16, ...}. Without expected= every value is
        expected equally often (as R chisq.test(table(x))). Empty cells are skipped. If an
        expected count is below 5 a note says so. Effect size: Cohen's w, how far the counts
        are from the expected shares (0.1 small, 0.3 medium, 0.5 large). For one value
        against one percent, compare_percent_to() gives an exact test.
        Gives back a one-row table with the count and expected count of each value.
        """
        step = "compare_counts_to"
        example = ('with sdata.example("school"): '
                   'table.compare_counts_to("breakfast", expected={"yes": 0.6, "no": 0.4})')
        values = self._filled_values(step, column, example)
        counts: dict = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        order = sorted(counts, key=_order)
        if expected is None:
            if len(order) < 2:
                raise SimplydataError(f'{step}() needs at least 2 different values in "{column}"; it found '
                                      f"{len(order)}.\nFor example, {example}")
            labels, observed = order, [counts[k] for k in order]
            shares = [1.0 / len(order)] * len(order)
            how = "the same share for every value"
        else:
            labels, observed, shares = self._expected_shares(step, column, expected, counts, order, example)
            how = "the shares in expected="
        n = sum(observed)
        if n == 0:
            raise SimplydataError(f'"{column}" has no values to count.')
        from scipy import stats

        expected_counts = np.asarray(shares) * n
        chi = stats.chisquare(np.asarray(observed, dtype=float), expected_counts)
        stat, p = float(chi.statistic), float(chi.pvalue)
        df = len(labels) - 1
        w = math.sqrt(stat / n)
        what = f'how often each "{column}" turns up, against {how}'
        row = {"test": "chi-square goodness-of-fit test", "what it compares": what, "statistic name": "chi-square",
               "statistic": stat, "degrees of freedom": float(df), "p-value": p,
               "effect size name": "distance of the counts from the expected shares (Cohen's w)", "effect size": w,
               "rows": n, "smallest expected count": float(expected_counts.min())}
        for label, o, e in zip(labels, observed, expected_counts):
            row[f"count of {_say_label(label)}"] = int(o)
            row[f"expected count of {_say_label(label)}"] = float(e)
        items = [f"{_say_label(k)} {o} (expected {_num(e)})" for k, o, e in zip(labels, observed, expected_counts)]
        shown = ", ".join(items[:GROUPS_SHOWN])
        if len(items) > GROUPS_SHOWN:
            shown += f" ... and {len(items) - GROUPS_SHOWN} more (the new table has them all)"
        print(f'Counts of "{column}" ({n} rows): {shown}. Differences this big from the expected counts '
              f"{_chance(p, plural=True)}. Cohen's w = {_num(w)}.")
        r_call = "chisq.test(table(x))" if expected is None else "chisq.test(x, p = ...)"
        print(f"Test: chi-square goodness-of-fit test (chi-square = {_num(stat)}, df = {df}), as R {r_call}; "
              "the usual test of counts against expected shares.")
        low = int((expected_counts < 5).sum())
        if low:
            counts_word = "count is" if low == 1 else "counts are"
            note = (f"Note: {low} of the {len(labels)} expected {counts_word} below 5, so the chi-square p-value may "
                    "be off. ")
            if len(labels) == 2:
                note += (f'With two values the exact binomial test is safer: table.compare_percent_to("{column}", '
                         f'value={_say_label(labels[0])!r}, percent={round(shares[0] * 100, 6):g}).')
            else:
                note += "Merge small groups, or collect more data."
            print(note)
        return self._result_table(row)

    @staticmethod
    def _expected_shares(step, column, expected, counts, order, example):
        """(labels, observed counts, shares adding up to 1) from expected={value: share}."""
        if not isinstance(expected, dict) or len(expected) < 2:
            raise SimplydataError(f"expected= needs the share of each value, at least two of them.\nFor example, {example}")
        bad = [k for k, v in expected.items() if not _is_number(v) or not math.isfinite(v) or v <= 0]
        if bad:
            raise SimplydataError(
                f"The expected share of {bad[0]!r} should be a number above 0, but it is {expected[bad[0]]!r}. "
                f"(A value that should never turn up cannot be tested; remove those rows first.)\nFor example, {example}"
            )
        total = float(sum(expected.values()))
        if math.isclose(total, 1.0, abs_tol=1e-6):
            scale = 1.0
        elif math.isclose(total, 100.0, abs_tol=1e-4):
            scale = 100.0
        else:
            raise SimplydataError(
                f"The expected shares add up to {_num(total)}; they should add up to 1 (or to 100 for percents). "
                "For a ratio such as 3:1, divide by its total: {\"yes\": 3/4, \"no\": 1/4}."
            )
        found = {}
        by_text = {str(k): k for k in order}
        for key in expected:
            if key in counts:
                found[key] = key
            elif str(key) in by_text:
                found[key] = by_text[str(key)]
            else:
                found[key] = None
        missing = [k for k in order if k not in found.values()]
        if missing:
            unknown = [k for k, v in found.items() if v is None]
            hint = ""
            if unknown:
                guesses = difflib.get_close_matches(str(unknown[0]), [str(k) for k in missing], n=1)
                hint = f' Did you mean "{guesses[0]}" for "{unknown[0]}"?' if guesses else ""
            raise SimplydataError(
                f'"{column}" has {_list_words(repr(_say_label(k)) for k in missing)}, which expected= does not '
                f"give a share for.{hint}\nGive a share for every value; values found: "
                f"{', '.join(repr(_say_label(k)) for k in order)}."
            )
        for key, value in found.items():
            if value is None:
                print(f'Note: {key!r} does not turn up in "{column}"; it counts as 0 rows.')
        labels = list(expected)
        observed = [counts[found[k]] if found[k] is not None else 0 for k in labels]
        shares = [float(expected[k]) / scale for k in labels]
        top = sum(shares)
        return labels, observed, [s / top for s in shares]

    @friendly
    def compare_percent_to(self, column=None, value=None, percent=None):
        """Is the percent of rows with one value different from a percent you choose, or could it be chance?

        table.compare_percent_to("passed", value="yes", percent=50)   (with sdata.example("school"))
        The exact binomial test (as R binom.test(x, n, p)), with the exact Clopper-Pearson
        95 % interval of the percent (as percent_of()). Percents go from 0 to 100. For a
        yes/no column (yes/no, True/False or 1/0) value= may be left out: "yes" (True, 1)
        is counted. Empty cells are skipped. Effect size: Cohen's h, the difference between
        the two shares on the arcsine scale (0.2 small, 0.5 medium, 0.8 large).
        Gives back a one-row table of results.
        """
        step = "compare_percent_to"
        example = ('with sdata.example("school"): table.compare_percent_to("passed", value="yes", percent=50)')
        if not _is_number(percent) or not math.isfinite(percent):
            raise SimplydataError(f"{step}() needs percent= as a number from 0 to 100.\nFor example, {example}")
        if not 0 < percent < 100:
            raise SimplydataError(f"percent= should be above 0 and below 100 (it is {percent:g}).\nFor example, {example}")
        values = self._filled_values(step, column, example)
        if not values:
            raise SimplydataError(f'"{column}" has no values to count.')
        value, chosen_by = self._value_to_count(step, column, value, values, example)
        hits = sum(1 for v in values if v == value or (isinstance(v, str) and isinstance(value, str)
                                                         and chosen_by and v.strip().lower() == value.lower()))
        n = len(values)
        from scipy import stats

        result = stats.binomtest(hits, n, percent / 100.0)
        p = float(result.pvalue)
        low, high = result.proportion_ci(confidence_level=0.95, method="exact")
        share = hits / n
        h = 2 * math.asin(math.sqrt(share)) - 2 * math.asin(math.sqrt(percent / 100.0))
        label = _say_label(value)
        what = f'the percent of rows where "{column}" is "{label}", against {percent:g} %'
        row = {"test": "exact binomial test", "what it compares": what,
               "statistic name": f'rows where "{column}" is "{label}"', "statistic": float(hits), "p-value": p,
               "effect size name": "difference between the shares on the arcsine scale (Cohen's h)",
               "effect size": h, "percent": share * 100, "percent 95 % CI low": float(low) * 100,
               "percent 95 % CI high": float(high) * 100, "matching rows": hits, "rows": n}
        print(f'"{column}" is "{label}" in {share * 100:.1f} % of the rows ({hits} of {n}; 95 % CI {low * 100:.1f} % '
              f"to {high * 100:.1f} %), against {percent:g} %. A difference this big {_chance(p)}. "
              f"Cohen's h = {_num(h)}.")
        print(f"Test: exact binomial test, as R binom.test({hits}, {n}, p = {percent / 100:g}); the 95 % CI is the "
              f"exact Clopper-Pearson interval.{chosen_by}")
        if percent < 1:
            print(f"Note: percent={percent:g} means {percent:g} %. For a half write percent=50.")
        return self._result_table(row)

    @staticmethod
    def _value_to_count(step, column, value, values, example):
        """(the value to count, words for the sentence when it was chosen from a yes/no column)."""
        if value is None:
            kinds = {type(v) for v in values}
            if kinds <= {bool}:
                return True, ' Counted as yes: True.'
            if all(_is_number(v) for v in values) and {float(v) for v in values} <= {0.0, 1.0}:
                return 1, " Counted as yes: 1."
            if kinds <= {str}:
                words = {v.strip().lower() for v in values}
                for yes, no in (("yes", "no"), ("true", "false"), ("y", "n")):
                    if words <= {yes, no}:
                        return yes, f' Counted as yes: "{yes}" (capital letters do not matter here).'
            shown = ", ".join(repr(v) for v in sorted({_say_label(v) for v in values})[:6])
            raise SimplydataError(
                f'{step}() needs value= to say which value of "{column}" to count, for example value={shown.split(", ")[0]}.\n'
                f"Values in \"{column}\": {shown}\nFor example, {example}"
            )
        if any(v == value for v in values):
            return value, ""
        shown = sorted({_say_label(v) for v in values})
        guess = difflib.get_close_matches(str(value), shown, n=1)
        hint = f' Did you mean "{guess[0]}"?' if guess else ""
        raise SimplydataError(
            f'value={value!r} is not in "{column}".{hint}\nValues in "{column}": '
            + ", ".join(repr(v) for v in shown[:8]) + (" ..." if len(shown) > 8 else "")
        )

    # ------------------------------------------------------------------ compare_shape_to
    @friendly
    def compare_shape_to(self, column=None, shape="normal", for_each=None, seed=1):
        """Do the values look bell-shaped (normal), or are they too far from it to be chance?

        table.compare_shape_to("marks", "normal", for_each="class")   (with sdata.example("marks"))
        Uses the Shapiro-Wilk test (as R shapiro.test()), for 3 to 5000 values in each
        group; empty cells are skipped. A small p-value says the shape is too far from a
        bell shape to be chance. A large p-value does not prove the values are normal: with
        few values the test can miss a different shape, and with many values (30 or more)
        it flags even tiny, harmless differences, while tests on averages are then usually
        fine anyway (the central limit theorem). Gives back a table with one row per group.

        Any other smooth shape works too: table.compare_shape_to("days", "gamma") fits the
        distribution (as table.fit_distribution) and runs the Anderson-Darling test, with a
        p-value from 9999 simulated samples that are each fitted again (seed=1, printed).
        """
        step = "compare_shape_to"
        example = 'with sdata.example("marks"): table.compare_shape_to("marks", "normal", for_each="class")'
        self._need_name(step, "the column", column, example)
        name = shape.strip().lower() if isinstance(shape, str) else shape
        if not isinstance(name, str) or not name:
            raise SimplydataError(f'{step}() needs a shape in quotes, such as "normal" or "gamma".\nFor example, {example}')
        if name not in SHAPES:
            return self._compare_other_shape(step, column, shape, for_each, seed)
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else []
        frame = self._number_frame(step, [column])
        values = frame.select(nw.col(column).cast(nw.Float64))[column].to_numpy()
        keys = [_key(r) for r in frame.select(groups).rows()] if groups else [()] * len(values)
        by_group: dict = {}
        for key, v in zip(keys, values):
            by_group.setdefault(key, [])
            if not math.isnan(v):
                by_group[key].append(v)
        order = sorted(by_group, key=lambda k: tuple(_order(v) for v in k))
        skipped = int(np.isnan(values).sum())
        _skipped_note(skipped, [column])
        from scipy import stats

        rows, lines, any_large_flag = [], [], False
        what = f'the shape of "{column}" with a normal (bell-shaped) distribution'
        for key in order:
            v = np.asarray(by_group[key], dtype=float)
            n = len(v)
            label = " / ".join(_say_label(x) for x in key)
            problem = None
            if n < 3:
                problem = f"only {n} value{'' if n == 1 else 's'}: Shapiro-Wilk needs at least 3"
            elif n > SHAPIRO_LIMIT:
                problem = (f"{n} values: Shapiro-Wilk works up to {SHAPIRO_LIMIT} values (as R); look at the shape "
                           f'on a chart instead, for example chart.histogram(x="{column}")')
            elif np.ptp(v) == 0:
                problem = "every value is the same, so there is no shape to test"
            if problem and not groups:
                raise SimplydataError(f'{step}() cannot test "{column}": {problem}.')
            w = p = math.nan
            if not problem:
                result = stats.shapiro(v)
                w, p = float(result.statistic), float(result.pvalue)
            row = {g: k for g, k in zip(groups, key)}
            row.update({"test": "Shapiro-Wilk test", "what it compares": what, "statistic name": "W",
                        "statistic": w, "p-value": p, "rows": n})
            rows.append(row)
            if problem:
                lines.append(f"  {label}: not tested ({problem}).")
                continue
            looks = "do not look bell-shaped" if p < 0.05 else "could well be bell-shaped"
            if not groups:
                if p < 0.05:
                    lines.append(f'The values of "{column}" do not look bell-shaped (normal): a shape this far from '
                                 f"normal {_chance(p)} (Shapiro-Wilk W = {w:.3f}, {n} values).")
                else:
                    lines.append(f'The values of "{column}" could well be bell-shaped (normal) (Shapiro-Wilk '
                                 f"W = {w:.3f}, {_p_shown(p)}, {n} values). This does not prove that they are.")
            else:
                lines.append(f"  {label}: {looks} (W = {w:.3f}, {_p_shown(p)}, {n} values)")
            any_large_flag = any_large_flag or (p < 0.05 and n >= CLT_SIZE)
        if groups:
            names = " and ".join(f'"{g}"' for g in groups)
            print(f'Do the values of "{column}" look bell-shaped (normal) in each {names}? '
                  "Shapiro-Wilk test (as R shapiro.test()); a small p-value means too far from normal to be chance:")
        for line in lines:
            print(line)
        if not groups:
            print("Test: Shapiro-Wilk test (as R shapiro.test()), the usual check of a bell shape for 3 to 5000 values.")
        else:
            print("A large p-value does not prove that the values are bell-shaped.")
        if any_large_flag:
            print(f"Note: with {CLT_SIZE} or more values Shapiro-Wilk flags even small, harmless differences from a bell "
                  "shape; tests on averages are usually still fine (the central limit theorem).")
        data = {k: [r[k] for r in rows] for k in rows[0]}
        return self._new_table(nw.from_dict(data, backend=self._frame.implementation))

    def _compare_other_shape(self, step, column, shape, for_each, seed):
        """compare_shape_to() for any shape but normal: the engine is table/distribution_words.shape_test."""
        from ..distributions import _find_family
        from .distribution_words import shape_test

        try:
            _find_family(shape, step)
        except SimplydataError as error:  # the option here is shape=: say so in the did-you-mean
            raise SimplydataError(str(error).replace('Did you mean "', 'Did you mean shape="')) from None
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SimplydataError("seed= needs a whole number, such as seed=1 (the same seed gives the same p-value).")
        groups = self._group_columns(step, for_each, [column]) if for_each is not None else []
        frame = self._number_frame(step, [column])
        values = frame.select(nw.col(column).cast(nw.Float64))[column].to_numpy()
        keys = [_key(r) for r in frame.select(groups).rows()] if groups else [()] * len(values)
        by_group: dict = {}
        for key, v in zip(keys, values):
            by_group.setdefault(key, [])
            if not math.isnan(v):
                by_group[key].append(v)
        order = sorted(by_group, key=lambda k: tuple(_order(v) for v in k))
        _skipped_note(int(np.isnan(values).sum()), [column])
        rows, method = [], None
        for key in order:
            label = " / ".join(_say_label(x) for x in key)
            result = shape_test(np.asarray(by_group[key], dtype=float), shape, seed=seed, column=column)
            sentence = result["sentence"]
            print(f"  {label}: {sentence}" if groups else sentence)
            method = result["method"]
            row = {g: k for g, k in zip(groups, key)}
            row.update({k: result[k] for k in ("test", "what it compares", "statistic name", "statistic", "p-value",
                                               "rows", "Monte Carlo samples", "seed")})
            rows.append(row)
        print(method)
        data = {k: [r[k] for r in rows] for k in rows[0]}
        return self._new_table(nw.from_dict(data, backend=self._frame.implementation))

    # ------------------------------------------------------------------ many p-values at once
    @friendly
    def add_adjusted_p_value_column(self, name=None, from_column="p-value", method="holm"):
        """Adjust p-values for testing many things at once, in a new column.

        table = sdata.table(sdata.example("marks")).correlation_between("hours", "marks", "height", show="each pair")
        table.add_adjusted_p_value_column("q", method="benjamini-hochberg")
        (table is now one row per pair of columns, with its p-value)
        The more tests you run, the more likely one of them looks real by chance; adjusted
        p-values allow for that. method="holm" (the default, as R p.adjust()): the chance
        of even one false alarm among all the tests stays at 5 %. "bonferroni": the same
        promise, simpler and stricter. "benjamini-hochberg" (also "bh" or "fdr"): about 5 %
        of the findings below 0.05 are expected to be false alarms (the false discovery
        rate); it finds more with many tests. "benjamini-yekutieli" ("by"): a false
        discovery rate that is safe however the tests depend on each other.
        from_column= is the column of p-values ("p-value" in simplydata result tables).
        Without a name the new column is called, for example, "p-value (Holm)". Empty
        cells stay empty and are not counted, as R p.adjust().
        """
        step = "add_adjusted_p_value_column"
        example = ('results.add_adjusted_p_value_column("q", from_column="p-value", method="benjamini-hochberg")\n'
                   '              (results from table.correlation_between("hours", "marks", "height", '
                   'show="each pair") with sdata.example("marks"))')
        chosen = _plain_name(method)
        chosen = ADJUST_ALIASES.get(chosen, chosen)
        if chosen not in ADJUST_METHODS:
            raise choice_error(step, "method", method, list(ADJUST_METHODS))
        label, r_name = ADJUST_METHODS[chosen]
        if not isinstance(from_column, str):
            raise SimplydataError(f"{step}() needs from_column= as one column name in quotes.\nFor example:  {example}")
        self._need_columns(from_column)
        if name is None:
            name = f"{from_column} ({label})"
        if not isinstance(name, str) or not name.strip():
            raise SimplydataError(f"{step}() needs the new column's name in quotes.\nFor example:  {example}")
        frame = self._number_frame(step, [from_column])
        p = frame.select(nw.col(from_column).cast(nw.Float64))[from_column].to_numpy()
        given = ~np.isnan(p)
        bad = p[given & ((p < 0) | (p > 1))]
        if len(bad):
            raise SimplydataError(
                f'"{from_column}" should hold p-values, which go from 0 to 1, but it has {_num(bad[0])}.\n'
                f"For example:  {example}"
            )
        m = int(given.sum())
        if m == 0:
            raise SimplydataError(f'"{from_column}" has no p-values to adjust: every cell is empty.')
        adjusted = np.full(len(p), np.nan)
        adjusted[given] = _adjust_p(p[given], chosen)
        values = [None if math.isnan(v) else float(v) for v in adjusted]
        base = nw.maybe_reset_index(self._frame)
        new = nw.new_series(name, values, nw.Float64, backend=base.implementation)
        before, after = int((p[given] < 0.05).sum()), int((adjusted[given] < 0.05).sum())
        tests = "p-value" if m == 1 else "p-values"
        print(f'Added "{name}": {m} {tests} from "{from_column}" adjusted with {label}\'s method '
              f'(as R p.adjust(method = "{r_name}")): {ADJUST_MEANING[chosen]}. '
              f"Below 0.05: {after} of {m} after adjusting ({before} before).")
        if m < len(p):
            print(f"Note: {len(p) - m} empty cell{'' if len(p) - m == 1 else 's'} in \"{from_column}\" "
                  f"{'stays' if len(p) - m == 1 else 'stay'} empty and {'was' if len(p) - m == 1 else 'were'} "
                  "not counted.")
        return self._change(f'add adjusted p-value column "{name}" from {from_column} ({label})',
                            base.with_columns(new))
