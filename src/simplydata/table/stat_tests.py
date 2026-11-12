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
  correction (R pairwise.t.test(pool.sd = FALSE, p.adjust.method = "holm")).
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
* fit_straight_line: ordinary least squares (R lm(y ~ x)), t-based 95 % intervals.
* fit_formula: nonlinear least squares (scipy curve_fit, Levenberg-Marquardt), standard
  errors as in R nls().

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
import warnings

import narwhals as nw
import numpy as np

from .._errors import SimplydataError, choice_error, friendly
from .columns import FORMULA_HELP, _formula_error, _FormulaReader, _is_number
from .summaries import _p_words

BASED_ON = ("averages", "ranks")
AVERAGE_TESTS = ("welch", "student", "welch anova", "anova", "mann-whitney", "kruskal-wallis")
RANK_TESTS = ("mann-whitney", "kruskal-wallis")
TWO_GROUP_TESTS = ("welch", "student", "mann-whitney")
COUNT_TESTS = ("chi-square", "fisher")
EXACT_LIMIT = 50  # exact p-values below 50 values without ties, as R wilcox.test()
CLT_SIZE = 30  # from this many values, a "not bell-shaped" note does not suggest switching
SHAPIRO_LIMIT = 5000  # SciPy's Shapiro-Wilk is reliable up to 5000 values


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
FIT_FUNCTIONS = {"exp": (1, 1), "log": (1, 2), "sqrt": (1, 1), "abs": (1, 1)}
FIT_HELP = (
    "A formula to fit has the column to fit on the left of =, and on the right:\n"
    "  column names               hours  (names with spaces go in backticks: `max temp`)\n"
    "  values to find             any other name, such as a, b, k or rate\n"
    "  numbers                    2, 100, 0.5\n"
    "  + - * / **                 add, subtract, multiply, divide, power\n"
    "  ( )                        brackets\n"
    "  exp(x)  log(x)  log(x, 10) e to the power x, natural log, log to base 10\n"
    "  sqrt(x)  abs(x)            square root, size without the minus sign\n"
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
        if name in FIT_FUNCTIONS or name in ("round",):
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
            raise _fit_error(f"{name}() was given {len(node.args)} things inside the brackets.")
        inner = self._node(node.args[0])
        if name == "exp":
            return lambda p, d: np.exp(inner(p, d))
        if name == "sqrt":
            return lambda p, d: np.sqrt(inner(p, d))
        if name == "abs":
            return lambda p, d: np.abs(inner(p, d))
        if len(node.args) == 1:
            return lambda p, d: np.log(inner(p, d))
        base = node.args[1]
        if not (isinstance(base, ast.Constant) and _is_number(base.value)) or base.value <= 0 or base.value == 1:
            raise _fit_error("The base of log() must be a positive number other than 1, for example log(x, 10).")
        scale = math.log(float(base.value))
        return lambda p, d: np.log(inner(p, d)) / scale


# ---------------------------------------------------------------------------- the words
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

    def _groups(self, step: str, column, between, example: str):
        """[(group name, values)] sorted by group; empty cells skipped with a note."""
        self._need_name(step, "the column", column, example)
        self._need_name(step, "between=", between, example)
        self._need_columns(column, between)
        if column == between:
            raise SimplydataError(f'"{column}" cannot be both the column to compare and the between= column.\n'
                                  f"For example, {example}")
        frame = self._number_frame(step, [column])
        if frame.schema[between].is_float():
            frame = frame.with_columns(nw.col(between).fill_nan(None))
        kept = frame.select(between, column).drop_nulls()
        _skipped_note(len(frame) - len(kept), [column, between])
        keys = kept[between].to_list()
        values = kept[column].to_numpy().astype(float)
        order = sorted(set(keys), key=lambda k: (str(type(k)), k))
        keys = np.asarray(keys, dtype=object)
        groups = [(_label(k), values[keys == k]) for k in order]
        if len(groups) < 2:
            found = f'only "{groups[0][0]}"' if groups else "none"
            raise SimplydataError(f'{step}() needs at least 2 groups in "{between}"; it found {found}.')
        small = [name for name, v in groups if len(v) < 2]
        if small:
            raise SimplydataError(
                f"{step}() needs at least 2 values in each group; "
                + _list_words(f'"{s}"' for s in small) + f' in "{between}" has fewer.\n'
                "Remove that group first, for example with table.remove_rows(...)."
            )
        return groups

    # ------------------------------------------------------------------ compare_averages
    @friendly
    def compare_averages(self, column=None, between=None, based_on="averages", test=None):
        """Are the averages of the groups really different, or could it be chance?

        table.compare_averages("marks", between="class")   (with sdata.example("marks"))
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
        example = 'with sdata.example("marks"): table.compare_averages("marks", between="class")'
        self._check_based_on(step, based_on)
        if test is not None:
            self._check_choice(step, "test", test, AVERAGE_TESTS)
            if based_on == "ranks" and test not in RANK_TESTS:
                raise SimplydataError(
                    f'test="{test}" compares averages, but based_on="ranks" asks for ranks. '
                    'Use one of them: test="mann-whitney" or test="kruskal-wallis" are the rank tests.'
                )
        groups = self._groups(step, column, between, example)
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
                f'test="{test}" compares exactly 2 groups, but "{between}" has {k}.\n'
                f'Use test="{other}" for all groups at once, or table.compare_each_pair("{column}", between="{between}").'
            )
        names = [g[0] for g in groups]
        arrays = [g[1] for g in groups]
        what = f'the average of "{column}" in each "{between}" ({", ".join(names)})'
        row = {"test": None, "what it compares": what}
        means = [float(v.mean()) for v in arrays]
        if test not in RANK_TESTS:
            spread = [float(v.std(ddof=1)) for v in arrays]
            if all(s == 0 for s in spread) or (test in ("welch anova",) and any(s == 0 for s in spread)):
                flat = [n for n, s in zip(names, spread) if s == 0]
                raise SimplydataError(
                    f'Every value is the same within {_list_words(f'"{n}"' for n in flat)}, so a test on averages '
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
                second += f' To see which groups differ: table.compare_each_pair("{column}", between="{between}").'
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
                second += (f' To see which groups differ: table.compare_each_pair("{column}", between="{between}", '
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
    @friendly
    def compare_each_pair(self, column=None, between=None, based_on="averages"):
        """Which pairs of groups differ? Every pair is compared, with Holm's correction.

        table.compare_each_pair("marks", between="class")   (with sdata.example("marks"))
        Each pair gets Welch's t-test (or Mann-Whitney U with based_on="ranks"). Testing many
        pairs makes a false alarm more likely, so the p-values are adjusted with Holm's
        method: the chance of at least one false alarm stays at 5 %. This is the same as R
        pairwise.t.test(pool.sd = FALSE, p.adjust.method = "holm"). Gives back a table with
        one row per pair.
        """
        step = "compare_each_pair"
        example = 'with sdata.example("marks"): table.compare_each_pair("marks", between="class")'
        self._check_based_on(step, based_on)
        groups = self._groups(step, column, between, example)
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
        print(f'Each pair of "{between}" groups compared on "{column}" with {test}; the p-values are adjusted for '
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
    @friendly
    def compare_counts(self, column=None, between=None, test=None):
        """Does how often each value turns up differ between groups, more than chance?

        table.add_group_column("result", from_column="marks", cut_at=[60], names=["below 60", "60 or more"])
        table.compare_counts("result", between="class")   (with sdata.example("marks"))
        (Do the classes differ in how many children score 60 or more?) Counts the rows for
        each pair of values and uses the chi-square test of independence (as R chisq.test():
        Yates' continuity correction for 2 x 2 tables only).
        If an expected count is below 5 a note says so; for a 2 x 2 table test="fisher" then
        gives Fisher's exact test. Empty cells are skipped. Effect size: Cramer's V, the
        strength of the link from 0 (none) to 1 (complete). Gives back a one-row table.
        """
        step = "compare_counts"
        example = 'with sdata.example("fruit"): table.compare_counts("fruit", between="shop")'
        self._need_name(step, "the column", column, example)
        self._need_name(step, "between=", between, example)
        self._need_columns(column, between)
        if test is not None:
            self._check_choice(step, "test", test, COUNT_TESTS)
        if column == between:
            raise SimplydataError(f"{step}() needs two different columns, for example {example}.")
        frame = self._frame
        floats = [c for c in (column, between) if frame.schema[c].is_float()]
        if floats:
            frame = frame.with_columns(nw.col(c).fill_nan(None) for c in floats)
        kept = frame.select(column, between).drop_nulls()
        _skipped_note(len(frame) - len(kept), [column, between])
        rows_of = kept[column].to_list()
        cols_of = kept[between].to_list()
        order = lambda keys: sorted(set(keys), key=lambda k: (str(type(k)), k))  # noqa: E731
        row_keys, col_keys = order(rows_of), order(cols_of)
        if len(row_keys) < 2 or len(col_keys) < 2:
            few = column if len(row_keys) < 2 else between
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
        what = f'how often each "{column}" turns up in each "{between}"'
        row = {"test": None, "what it compares": what}
        size = f"{observed.shape[0]} x {observed.shape[1]}"
        if test == "fisher":
            if not two_by_two:
                raise SimplydataError(
                    f'test="fisher" works on 2 x 2 tables here, but "{column}" and "{between}" make a {size} table.\n'
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

    # ------------------------------------------------------------------ fitting
    @friendly
    def fit_straight_line(self, x=None, y=None):
        """Fit the straight line y = slope * x + intercept that best follows the points.

        table.fit_straight_line(x="hours", y="marks")   (with sdata.example("marks"))
        Ordinary least squares (the same as R lm(y ~ x)). Prints how much y goes up for each
        1 more of x, with its 95 % confidence interval and p-value, and R squared: the share
        of the differences in y that the line explains. Rows with an empty cell in x or y
        are skipped. A link does not show that x causes y. Gives back a one-row table.
        """
        step = "fit_straight_line"
        example = 'with sdata.example("marks"): table.fit_straight_line(x="hours", y="marks")'
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
        print(f'"{y}" goes {direction} by {_num(abs(slope))} for each 1 more "{x}" (95 % CI {_num(row["slope 95 % CI low"])} '
              f'to {_num(row["slope 95 % CI high"])}); this slope {_chance(p)}. R squared = {r2:.2f}: "{x}" explains '
              f'about {round(r2 * 100)} % of the differences in "{y}".')
        print(f"Line: {y} = {_num(slope)} * {x} + {_num(intercept)}, by least squares over {n} rows. "
              "A link does not show that one thing causes the other.")
        residuals = ys - (intercept + slope * xs)
        for note in _shape_notes([("the line", residuals)], what="the distances from the line"):
            print(note.replace('a safer test here: based_on="ranks".',
                               "the p-value and confidence intervals may be off."))
        return self._result_table(row)

    @friendly
    def fit_formula(self, formula=None, start=None):
        """Find the values in a formula that make it follow the data best.

        table.fit_formula("population = a * exp(b * (year - 2000))")   (with sdata.example("growth"))
        Write the column to fit on the left of =. On the right, names that are columns are
        data; any other name (such as a, b or rate) is a value to find. Uses nonlinear least
        squares (SciPy curve_fit; standard errors as in R nls()). Every value to find starts
        at 1; give better start values with start={"a": 10, "b": 0.03} if the fit fails.
        Prints each value with its standard error, and R squared (only a rough guide for a
        curve). Rows with an empty cell are skipped. Gives back a one-row table.
        """
        step = "fit_formula"
        if not isinstance(formula, str) or formula.count("=") != 1 or "==" in formula:
            raise _fit_error(
                'fit_formula() needs a formula with one = sign: the column to fit on the left, for example\n'
                'table.fit_formula("marks = a * hours + b").'
            )
        left, right = (part.strip() for part in formula.split("="))
        if left.startswith("`") and left.endswith("`"):
            left = left[1:-1]
        self._need_columns(left)
        reader = _FitReader(right, self._columns)
        try:
            model = reader.read()
        except SimplydataError as error:
            if reader.finished and not reader.used:
                raise _fit_error(
                    f'The right side of the formula has no column in it, so there is nothing to fit "{left}" to.'
                ) from None
            raise SimplydataError(str(error).replace(FORMULA_HELP, FIT_HELP)) from None
        names = reader.values_to_find
        if not names:
            raise _fit_error("The formula has no value to find. Use a name that is not a column, such as a or b.")
        if left in reader.used:
            raise _fit_error(f'"{left}" is on both sides of the formula.')
        start = {} if start is None else start
        if not isinstance(start, dict) or not all(_is_number(v) for v in start.values()):
            raise SimplydataError('start= should look like {"a": 10, "b": 0.5}: a name and a number for each value.')
        unknown = [k for k in start if k not in names]
        if unknown:
            raise SimplydataError(f"start= has {_list_words(repr(k) for k in unknown)}, which "
                                  f"{'is' if len(unknown) == 1 else 'are'} not in the formula. "
                                  f"Values to find: {', '.join(names)}.")
        columns = [left] + list(reader.used)
        arrays = self._number_values(step, columns)
        ys = arrays[0]
        data = dict(zip(columns[1:], arrays[1:]))
        n = len(ys)
        if n <= len(names):
            raise SimplydataError(f"{step}() needs more rows ({n}) than values to find ({len(names)}).")
        p0 = [float(start.get(k, 1.0)) for k in names]

        def curve(_, *theta):
            with np.errstate(all="ignore"):
                out = model(dict(zip(names, theta)), data)
            return np.broadcast_to(np.asarray(out, dtype=float), ys.shape)

        start_text = "start={" + ", ".join(f'"{k}": {v:g}' for k, v in zip(names, p0)) + "}"
        start_hint = "start={" + ", ".join(f'"{k}": ...' for k in names) + "}"
        if not np.all(np.isfinite(curve(None, *p0))):
            raise SimplydataError(
                f"With the start values ({start_text}) the formula gives impossible numbers, such as the log of a "
                "negative number or a division by 0.\nGive start values close to what you expect, for example "
                f"table.fit_formula(\"{formula}\", {start_hint})."
            )
        from scipy import optimize

        uncertain = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                found, cov = optimize.curve_fit(curve, np.zeros(n), ys, p0=p0, maxfev=10000)
            except (RuntimeError, ValueError, optimize.OptimizeWarning):
                raise SimplydataError(
                    f"simplydata could not fit the formula: the search for {_list_words(names)} did not settle "
                    f"(it began at {start_text}).\nTry start values close to what you expect, for example "
                    f"{start_hint}, or check the formula."
                ) from None
            uncertain = any(issubclass(w.category, optimize.OptimizeWarning) for w in caught)
        errors = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(names), np.inf)
        fitted = curve(None, *found)
        if not np.all(np.isfinite(fitted)):
            raise SimplydataError("The fit ended on values that give impossible numbers. Try other start values.")
        rss = float(((ys - fitted) ** 2).sum())
        tss = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - rss / tss if tss > 0 else math.nan
        row = {"test": "formula fit (nonlinear least squares)", "formula": formula}
        for name, value, error in zip(names, found, errors):
            row[name] = float(value)
            row[f"standard error of {name}"] = float(error)
        row.update({"R squared": r2, "rows": n})
        parts = ", ".join(f"{k} = {_num(v)} +/- {_num(e)}" for k, v, e in zip(names, found, errors))
        print(f'Fitted "{formula}" to {n} rows: {parts} (value +/- standard error). R squared = {r2:.2f} '
              "(only a rough guide for a curve).")
        print(f"Values found: {', '.join(names)}. Data: {', '.join(reader.used)}. "
              "Method: nonlinear least squares, as R nls().")
        if uncertain or not np.all(np.isfinite(errors)):
            print("Note: the standard errors could not be worked out well: the data do not pin down every value. "
                  "Try a simpler formula or more data.")
        return self._result_table(row)
