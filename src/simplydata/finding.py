"""Finding the smallest or biggest value of a formula, and solving equations.

sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5]) and sdata.find_biggest(...) find where
a formula is lowest or highest; sdata.solve("x**3 - 2*x = 5", between=[0, 3]) finds where the
two sides are equal. The names in the formula (x, y, a, rate) are what to find. The formula is
read safely (never run), as in table.fit_formula.

The method is decided by what you give, and the printed sentence always says which one and
why (simplydata never switches silently). Defaults match R, so answers can be checked there:

* one name and between=[low, high]: Brent's method on the range (R optimize(), tol 1e-10).
  200 evenly spaced points are checked first; if one of them is lower than Brent's answer,
  or the curve has several dips, a note says so (search="whole range" looks everywhere).
* one name and start= only: Brent's method on a bracket found around the start.
* several names and start=: Nelder-Mead (R optim()'s default); with between= limits:
  L-BFGS-B (R optim(method = "L-BFGS-B"), the only optim() method with limits).
* rules= that are not all straight-line: SLSQP (R nloptr, NLOPT_LD_SLSQP).
* a straight-line (linear) formula with straight-line rules: linear programming (HiGHS),
  exact, as R lpSolve::lp(); also the "value of one more" of each rule (shadow prices).
  With whole_numbers=True: mixed-integer linear programming (HiGHS), as lp(int.vec =).
* search="whole range": differential evolution over between= (seed=1) and then a local
  polish, so the answer is the lowest point anywhere in the range, not only near the start.

Unlike R lpSolve::lp(), names are NOT assumed to be 0 or more: give the rule "x >= 0" (or
between=) when they must be. find_biggest finds the smallest value of minus the formula.

solve(): one name and between=: the difference of the two sides on 1000 steps across the
range, and Brent's method (R uniroot(), tol 1e-12) in every step where it changes sign, so
every solution that crosses is found (R rootSolve::uniroot.all()). One name and start=:
the secant method from the start. A polynomial without between= or start=: all its
solutions, complex ones too (R polyroot()). Several equations: solved exactly when they are
all straight-line (R solve()), otherwise Powell's hybrid method from start= (R
nleqslv::nleqslv()).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ._errors import SimplydataError, choice_error, friendly
from ._maths_formula import _SIGN_PATTERN, read_equation, read_formula

SEARCHES = ("from start", "whole range")
SCAN_POINTS = 200  # points checked before Brent's method on a range
SOLVE_STEPS = 1000  # steps across the range when solve() looks for sign changes
BRENT_TOL = 1e-10  # as R optimize(tol = 1e-10)
ROOT_TOL = 1e-12  # as R uniroot(tol = 1e-12)
SEED = 1
SHOWN_NAMES = 8  # names written out in a sentence; all of them are in the table

# method= for one search from the start: plain name -> (SciPy name, words in the sentence, R equivalent)
LOCAL_METHODS = {
    "nelder-mead": ("Nelder-Mead", "Nelder-Mead", "R optim() (its default method)"),
    "bfgs": ("BFGS", "BFGS (a quasi-Newton method)", 'R optim(method = "BFGS")'),
    "l-bfgs-b": ("L-BFGS-B", "L-BFGS-B", 'R optim(method = "L-BFGS-B")'),
    "conjugate gradient": ("CG", "conjugate gradients", 'R optim(method = "CG")'),
    "powell": ("Powell", "Powell's conjugate-direction method", "SciPy minimize(method = \"Powell\"); "
               "base R has no exact equivalent"),
    "trust region": ("trust-constr", "a trust-region method", "SciPy minimize(method = \"trust-constr\"); "
                     "base R has no exact equivalent"),
    "slsqp": ("SLSQP", "SLSQP (sequential quadratic programming)", 'R nloptr(algorithm = "NLOPT_LD_SLSQP")'),
    "cobyla": ("COBYLA", "COBYLA (linear approximations, no slopes needed)",
               'R nloptr(algorithm = "NLOPT_LN_COBYLA")'),
}
ONE_NAME_METHODS = {"brent": "Brent's method"}
GLOBAL_METHODS = {
    "differential evolution": ("differential evolution", "R DEoptim::DEoptim()"),
    "dual annealing": ("dual annealing (generalised simulated annealing)", "R GenSA::GenSA()"),
    "shgo": ("simplicial homology global optimisation (SHGO)", "SciPy shgo(); R has no common equivalent"),
    "basin hopping": ("basin hopping", "SciPy basinhopping(); R has no common equivalent"),
    "direct": ("DIRECT (dividing rectangles)", 'R nloptr(algorithm = "NLOPT_GN_DIRECT")'),
}
LINEAR_METHODS = {"linear programming": "highs", "highs": "highs", "simplex": "highs-ds",
                  "interior point": "highs-ipm"}
METHOD_SPELLINGS = {
    "nelder mead": "nelder-mead", "neldermead": "nelder-mead", "lbfgsb": "l-bfgs-b", "l-bfgs": "l-bfgs-b",
    "lbfgs": "l-bfgs-b", "cg": "conjugate gradient", "conjugate-gradient": "conjugate gradient",
    "conjugate gradients": "conjugate gradient", "trust-constr": "trust region", "trust constr": "trust region",
    "trust-region": "trust region", "brent's method": "brent", "bounded": "brent", "golden": "brent",
    "de": "differential evolution", "differential-evolution": "differential evolution",
    "dual-annealing": "dual annealing", "simulated annealing": "dual annealing",
    "basin-hopping": "basin hopping", "basinhopping": "basin hopping", "highs-ds": "simplex",
    "highs-ipm": "interior point", "lp": "linear programming", "milp": "linear programming",
}
SOLVE_ONE_METHODS = {"brent": "Brent's method", "bisection": "bisection", "ridder": "Ridders' method",
                     "toms748": "TOMS Algorithm 748", "secant": "the secant method"}
SOLVE_SYSTEM_METHODS = {"powell hybrid": ("hybr", "Powell's hybrid method (MINPACK hybrd)",
                                          "R nleqslv::nleqslv()"),
                        "levenberg-marquardt": ("lm", "Levenberg-Marquardt least squares", "R nleqslv / minpack.lm"),
                        "broyden": ("broyden1", "Broyden's method", 'R nleqslv(method = "Broyden")')}
SOLVE_SPELLINGS = {"hybr": "powell hybrid", "hybrid": "powell hybrid", "lm": "levenberg-marquardt",
                   "levenberg marquardt": "levenberg-marquardt", "broyden1": "broyden", "brentq": "brent",
                   "bisect": "bisection", "ridders": "ridder", "secant method": "secant"}


# ---------------------------------------------------------------------------- small helpers
def _show(x) -> str:
    """A number as the answer is printed: up to 6 significant digits, 2 not 2.00000."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "infinite" if x > 0 else "minus infinite"
    if abs(x) < 1e-12:
        return "0"
    text = f"{x:.6g}"
    return "0" if text in ("-0", "0") else text


def _list_words(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _places(names, values) -> str:
    pairs = [f"{n} = {_show(v)}" for n, v in zip(names, values)]
    if len(pairs) > SHOWN_NAMES:
        return ", ".join(pairs[:SHOWN_NAMES]) + f" ... ({len(pairs)} names, all in the table)"
    return ", ".join(pairs)


def _result(rows: list[dict], fast: bool):
    import pandas as pd

    from .table import table

    if not isinstance(fast, bool):
        raise SimplydataError("fast= should be True or False.")
    return table(pd.DataFrame(rows), fast=fast)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _plain_method(step, method, choices, spellings):
    if method is None:
        return None
    if not isinstance(method, str):
        raise SimplydataError(f'{step}() needs method= as a name in quotes, such as method="nelder-mead".')
    key = method.strip().lower().replace("_", " ")
    key = spellings.get(key, key)
    if key not in choices:
        raise choice_error(step, "method", method, list(choices))
    return key


def _range_words(low, high) -> str:
    if math.isfinite(low) and math.isfinite(high):
        return f"between {_show(low)} and {_show(high)}"
    if math.isfinite(low):
        return f"of at least {_show(low)}"
    if math.isfinite(high):
        return f"of at most {_show(high)}"
    return "of any size"


def _limits(step, names, between, example):
    """between= as {name: (low, high)} with -inf / inf for no limit."""
    limits = {n: (-math.inf, math.inf) for n in names}
    if between is None:
        return limits, False
    if isinstance(between, dict):
        unknown = [k for k in between if k not in names]
        if unknown:
            raise SimplydataError(f"between= has {_list_words(repr(k) for k in unknown)}, which "
                                  f"{'is' if len(unknown) == 1 else 'are'} not in the formula. "
                                  f"Names to find: {', '.join(names)}.")
        items = between.items()
    else:
        if len(names) != 1:
            raise SimplydataError(
                f"There are {len(names)} names to find ({', '.join(names)}), so between= needs a range for each, "
                f'for example between={{{", ".join(repr(n) + ": [0, 10]" for n in names[:2])}}} '
                "(None means no limit)."
            )
        items = [(names[0], between)]
    for name, pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise SimplydataError(f"between= needs two numbers for {name}, the lowest and the highest, "
                                  f"for example {example}. Use None for no limit: [0, None].")
        low, high = pair
        for value in (low, high):
            if value is not None and not _is_number(value):
                raise SimplydataError(f"between= needs numbers (or None for no limit); {name} has {value!r}.")
        low = -math.inf if low is None else float(low)
        high = math.inf if high is None else float(high)
        if math.isnan(low) or math.isnan(high) or not low < high:
            raise SimplydataError(f"between= for {name} must go from a lower number to a higher one; "
                                  f"it was [{_show(low)}, {_show(high)}].")
        limits[name] = (low, high)
    return limits, True


def _starts(step, names, start, limits, needed: bool, example):
    """start= as a list in the order of names (None when start= was not given and not needed)."""
    if start is None:
        values = {}
    elif _is_number(start):
        if len(names) != 1:
            raise SimplydataError(f"There are {len(names)} names to find ({', '.join(names)}), so start= needs a "
                                  f'value for each, for example start={{{", ".join(repr(n) + ": 1" for n in names[:2])}}}.')
        values = {names[0]: float(start)}
    elif isinstance(start, dict):
        unknown = [k for k in start if k not in names]
        if unknown:
            raise SimplydataError(f"start= has {_list_words(repr(k) for k in unknown)}, which "
                                  f"{'is' if len(unknown) == 1 else 'are'} not in the formula. "
                                  f"Names to find: {', '.join(names)}.")
        if not all(_is_number(v) and math.isfinite(v) for v in start.values()):
            raise SimplydataError('start= should look like {"x": 1, "y": 2}: a name and a number for each.')
        values = {k: float(v) for k, v in start.items()}
    else:
        raise SimplydataError(f"start= should be a number (one name) or a dict such as {{'x': 1, 'y': 2}}; "
                              f"for example {example}.")
    out, missing = [], []
    for name in names:
        if name in values:
            out.append(values[name])
            continue
        low, high = limits[name]
        if math.isfinite(low) and math.isfinite(high):
            out.append((low + high) / 2)
        else:
            missing.append(name)
            out.append(math.nan)
    if missing:
        if not needed and start is None:
            return None
        raise SimplydataError(
            f"{step}() needs a start value for {_list_words(missing)} (or a range with two ends in between=), "
            f"for example {example}."
        )
    return out


def _whole_names(step, names, whole_numbers):
    if whole_numbers is False or whole_numbers is None:
        return []
    if whole_numbers is True:
        return list(names)
    chosen = [whole_numbers] if isinstance(whole_numbers, str) else whole_numbers
    if not isinstance(chosen, (list, tuple)) or not all(isinstance(n, str) for n in chosen):
        raise SimplydataError(f'{step}() needs whole_numbers=True, or a list of names such as whole_numbers=["x"].')
    unknown = [n for n in chosen if n not in names]
    if unknown:
        raise SimplydataError(f"whole_numbers= has {_list_words(repr(n) for n in unknown)}, which "
                              f"{'is' if len(unknown) == 1 else 'are'} not in the formula or the rules. "
                              f"Names to find: {', '.join(names)}.")
    return list(chosen)


# ---------------------------------------------------------------------------- the rules
class _Rule:
    def __init__(self, text, formula, sign):
        self.text = text.strip()
        self.formula = formula
        self.sign = {"<": "<=", ">": ">="}.get(sign, sign)
        self.strict = sign in ("<", ">")

    def gap(self, values):
        """left minus right at these values."""
        return float(self.formula(values))

    def met_by(self, values, tolerance=1e-7):
        gap = self.gap(values)
        if self.sign == "<=":
            return gap <= tolerance
        if self.sign == ">=":
            return gap >= -tolerance
        return abs(gap) <= tolerance


def _read_rules(step, rules, known):
    if rules is None:
        return []
    if isinstance(rules, str):
        rules = [rules]
    if not isinstance(rules, (list, tuple)) or not rules:
        raise SimplydataError(f'{step}() needs rules= as a list of rules in quotes, for example '
                              'rules=["x + y <= 4", "x >= 0"].')
    out = []
    for text in rules:
        if not isinstance(text, str):
            raise SimplydataError(f'Each rule should be text, such as "x + y <= 4"; one was {text!r}.')
        formula, sign = read_equation(text, known=known)
        out.append(_Rule(text, formula, sign))
    if any(r.strict for r in out):
        print("Note: < and > in rules are treated as <= and >=: a smallest or biggest value under a strict "
              "rule sits on its edge, so it would never be reached exactly.")
    return out


# ---------------------------------------------------------------------------- find_smallest / find_biggest
class _Answer:
    def __init__(self, place, value, method_words, settled=True, iterations=None, extra=None, notes=None,
                 method_name=""):
        self.place = np.asarray(place, dtype=float)
        self.value = float(value)
        self.method_words = method_words
        self.settled = bool(settled)
        self.iterations = iterations
        self.extra = extra or {}
        self.notes = notes or []
        self.method_name = method_name


def _find(step, goal, formula, between, start, rules, whole_numbers, search, method, known, fast):
    biggest = goal == "biggest"
    word = "biggest" if biggest else "smallest"
    example = (f'sdata.{step}("x**2 - 4*x + 1", between=[0, 5])' if not biggest
               else f'sdata.{step}("x * (10 - x)", between=[0, 10])')
    if not isinstance(search, str) or search.strip().lower() not in SEARCHES:
        raise choice_error(step, "search", search, SEARCHES)
    search = search.strip().lower()
    if not isinstance(formula, str):
        raise SimplydataError(f"{step}() needs a formula in quotes, for example {example}.")
    goal_formula = read_formula(formula, known=known)
    rule_list = _read_rules(step, rules, known)
    names = list(goal_formula.names)
    for rule in rule_list:
        for name in rule.formula.names:
            if name not in names:
                names.append(name)
    if not names:
        raise SimplydataError(f'The formula "{formula}" has no name to find, so it has the same value everywhere.\n'
                              f"Use a name such as x for what to find, for example {example}.")
    if goal_formula.known:
        print("Using " + ", ".join(f"{k} = {_show(v)}" for k, v in goal_formula.known.items()) + " from known=.")
    limits, has_limits = _limits(step, names, between, "between=[0, 5]" if len(names) == 1 else
                                 'between={"x": [0, 5], "y": [0, None]}')
    whole = _whole_names(step, names, whole_numbers)
    sign = -1.0 if biggest else 1.0

    def objective(theta):
        value = float(goal_formula(dict(zip(names, np.atleast_1d(theta)))))
        return sign * value if math.isfinite(value) else math.inf

    linear_goal = goal_formula.linear_parts()
    all_linear = linear_goal is not None and all(r.formula.linear_parts() is not None for r in rule_list)
    linear_method = method is not None and isinstance(method, str) and \
        METHOD_SPELLINGS.get(method.strip().lower(), method.strip().lower()) in LINEAR_METHODS

    if search == "whole range":
        chosen = _plain_method(step, method, GLOBAL_METHODS, METHOD_SPELLINGS)
        answer = _whole_range(step, names, limits, objective, rule_list, whole, start, chosen or
                              "differential evolution")
    elif rule_list and (all_linear and (method is None or linear_method)) or \
            (whole and not rule_list and linear_goal is not None and method is None):
        if not all_linear:
            raise SimplydataError("whole_numbers= works with a straight-line formula and straight-line rules "
                                  "(linear programming). Use search=\"whole range\" for whole numbers in other "
                                  "formulas.")
        chosen = _plain_method(step, method, LINEAR_METHODS, METHOD_SPELLINGS)
        answer = _linear(step, names, limits, linear_goal, rule_list, whole, biggest,
                         LINEAR_METHODS[chosen or "linear programming"])
    else:
        if whole:
            raise SimplydataError(
                "whole_numbers= needs a straight-line (linear) formula and straight-line rules, such as "
                '"3*x + 2*y" with rules=["x + y <= 4"]. For other formulas use search="whole range" with '
                "a range for each name in between=."
            )
        if rule_list:
            chosen = _plain_method(step, method, {k: LOCAL_METHODS[k] for k in ("slsqp", "cobyla", "trust region")},
                                   METHOD_SPELLINGS)
            first = _starts(step, names, start, limits, True,
                            'start={"x": 1, "y": 1} (the search under rules starts there)')
            answer = _with_rules(names, limits, objective, rule_list, first, chosen or "slsqp", all_linear)
        elif len(names) == 1 and (method is None or _plain_method(step, method, {**LOCAL_METHODS, **ONE_NAME_METHODS},
                                                                  METHOD_SPELLINGS) == "brent"):
            low, high = limits[names[0]]
            if math.isfinite(low) and math.isfinite(high):
                if start is not None:
                    print("Note: start= is not used when one name has a range with two ends: Brent's method "
                          "searches the whole of between=.")
                answer = _brent_range(names[0], low, high, objective, biggest)
            else:
                if start is None:
                    raise SimplydataError(f"{step}() needs between=[low, high] or start=... for {names[0]}, for "
                                          f"example {example}.")
                first = _starts(step, names, start, limits, True, f"start=1")[0]
                answer = _brent_start(step, names[0], first, low, high, objective, word)
        else:
            chosen = _plain_method(step, method, LOCAL_METHODS, METHOD_SPELLINGS)
            if start is None and not all(math.isfinite(v) for pair in limits.values() for v in pair):
                if len(names) == 1:
                    raise SimplydataError(f"{step}() needs between=[low, high] or start=... for {names[0]}, "
                                          f"for example {example}.")
                raise SimplydataError(
                    f"{step}() needs a start for each name ({', '.join(names)}), for example "
                    f'start={{{", ".join(repr(n) + ": 1" for n in names[:2])}}}, or a range with two ends for '
                    "each in between=."
                )
            first = _starts(step, names, start, limits, True, 'start={"x": 1, "y": 1}')
            if chosen is None:
                chosen = "l-bfgs-b" if has_limits else "nelder-mead"
            answer = _local(names, limits, has_limits, objective, first, chosen)

    value = sign * answer.value
    where = ""
    if rule_list:
        count = len(rule_list)
        where = f" under the {count} rules" if count > 1 else " under the rule"
        if has_limits:
            where += " and the limits in between="
    elif len(names) == 1 and has_limits:
        where = " " + _range_words(*limits[names[0]])
    elif has_limits:
        where = " within between="
    if search == "whole range":
        where = " over the whole range" + ("" if not rule_list else where)
    print(f"The {word} value of {formula}{where} is {_show(value)}, at {_places(names, answer.place)}.")
    print(answer.method_words)
    tight = []
    for rule in rule_list:
        if rule.sign == "=":
            continue
        gap = rule.gap(dict(zip(names, answer.place)))
        scale = 1 + max(abs(float(v)) for v in answer.place)
        if abs(gap) <= 1e-7 * scale:
            tight.append(rule.text)
    if tight:
        verb = "is" if len(tight) == 1 else "are"
        print(f"{_list_words(tight)} {verb} reached exactly; the other rules have room to spare." if len(tight) <
              len([r for r in rule_list if r.sign != "="]) else f"{_list_words(tight)} {verb} reached exactly.")
    for note in answer.notes:
        print(note)
    if not answer.settled:
        print('Note: the search did not settle, so this answer may be rough; try start= nearer the answer or '
              'search="whole range".')
    row = {name: float(v) + 0.0 for name, v in zip(names, answer.place)}
    row[f"{word} value"] = value + 0.0
    row["formula"] = formula
    row["method"] = answer.method_name
    row["settled"] = "yes" if answer.settled else "no"
    row["iterations"] = answer.iterations if answer.iterations is not None else math.nan
    for rule_text, one_more in answer.extra.items():
        row[f"value of one more: {rule_text}"] = sign * one_more + 0.0
    return _result([row], fast)


def _scan(low, high, objective, points):
    xs = np.linspace(low, high, points)
    ys = np.array([objective(x) for x in xs])
    return xs, ys


def _brent_range(name, low, high, objective, biggest=False):
    """Brent's method on [low, high]. The objective is minus the formula for find_biggest, so the
    notes speak of the highest point and the peaks then."""
    from scipy import optimize

    lowest, lower, dip = ("highest", "higher", "peak") if biggest else ("lowest", "lower", "dip")

    found = optimize.minimize_scalar(objective, bounds=(low, high), method="bounded",
                                     options={"xatol": BRENT_TOL, "maxiter": 10000})
    place, value = float(found.x), float(found.fun)
    xs, ys = _scan(low, high, objective, SCAN_POINTS)
    notes = []
    for edge in (low, high):  # R optimize() never tries the ends themselves; look there too
        edge_value = objective(edge)
        if abs(place - edge) <= 1e-4 * (high - low) and edge_value <= value:
            place, value = edge, edge_value
            notes.append(f"Note: the {lowest} point is at the end of the range, {name} = {_show(edge)}; a wider "
                         f"between= may give a {lower} value.")
    finite = np.isfinite(ys)
    dips = 0
    for i in range(len(ys)):
        if not finite[i]:
            continue
        left = ys[i - 1] if i > 0 and finite[i - 1] else math.inf
        right = ys[i + 1] if i + 1 < len(ys) and finite[i + 1] else math.inf
        if ys[i] < left and ys[i] <= right:
            dips += 1
    words = "Method: Brent's method on the range, as R optimize()."
    if dips > 1:
        words += (f" It finds one {lowest} point; a curve with several {dip}s may have a {lower} one elsewhere "
                  '(try search="whole range").')
    best = int(np.nanargmin(np.where(finite, ys, np.nan))) if finite.any() else None
    if best is not None and ys[best] < value - 1e-9 * (1 + abs(value)):
        shown = -ys[best] if biggest else ys[best]
        notes.append(f"Note: of {SCAN_POINTS} evenly spaced points checked first, {name} = {_show(xs[best])} gives "
                     f"a {lower} value ({_show(shown)}) than Brent's answer: the curve has several {dip}s and "
                     f'Brent\'s method found one of them. Try search="whole range", or a narrower between=.')
    return _Answer([place], value, words, settled=bool(found.success), iterations=int(found.nit), notes=notes,
                   method_name="Brent's method on the range")


def _brent_start(step, name, first, low, high, objective, word):
    from scipy import optimize

    width = 0.1 * max(1.0, abs(first))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b, c, *_ = optimize.bracket(objective, xa=first, xb=first + width, maxiter=1000)
    except (RuntimeError, ValueError, optimize.BracketError):
        raise SimplydataError(
            f"Searching from {name} = {_show(first)}, the formula keeps getting {'bigger' if word == 'biggest' else 'smaller'}, "
            f"so it may have no {word} value. Give a range, for example between=[0, 10]."
        ) from None
    a, c = min(a, c), max(a, c)
    endless = SimplydataError(
        f"Searching from {name} = {_show(first)}, the formula keeps getting {'bigger' if word == 'biggest' else 'smaller'}, "
        f"so it may have no {word} value. Give a range, for example between=[0, 10]."
    )
    if max(abs(a), abs(c)) > 1e12 * (1 + abs(first)):
        raise endless
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        found = optimize.minimize_scalar(objective, bracket=(a, b, c), method="brent", options={"xtol": BRENT_TOL})
    if not math.isfinite(float(found.fun)):
        raise endless
    words = (f"Method: Brent's method from a bracket around the start ({name} from {_show(a)} to {_show(c)}), as R "
             "optimize() on that bracket. It finds the nearest dip to the start; between= or "
             'search="whole range" looks further.')
    notes = []
    place = float(found.x)
    if not low <= place <= high:
        notes.append(f"Note: the answer is outside between= ({_range_words(low, high)}).")
    return _Answer([place], float(found.fun), words, settled=bool(found.success), iterations=int(found.nit),
                   notes=notes, method_name="Brent's method from the start")


def _bounds(names, limits):
    return [(None if not math.isfinite(limits[n][0]) else limits[n][0],
             None if not math.isfinite(limits[n][1]) else limits[n][1]) for n in names]


def _local(names, limits, has_limits, objective, first, chosen):
    from scipy import optimize

    scipy_name, words, r_name = LOCAL_METHODS[chosen]
    options = {"nelder-mead": {"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000 * len(names),
                               "maxfev": 40000 * len(names)},
               "bfgs": {"gtol": 1e-9, "maxiter": 10000},
               "l-bfgs-b": {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 15000},
               "conjugate gradient": {"gtol": 1e-9, "maxiter": 10000},
               "powell": {"xtol": 1e-10, "ftol": 1e-12, "maxiter": 20000},
               "trust region": {"xtol": 1e-12, "gtol": 1e-10, "maxiter": 10000},
               "slsqp": {"ftol": 1e-14, "maxiter": 1000},
               "cobyla": {"tol": 1e-8, "maxiter": 5000}}[chosen]
    kwargs = {}
    if has_limits:
        if chosen in ("nelder-mead", "l-bfgs-b", "powell", "trust region", "slsqp", "cobyla"):
            kwargs["bounds"] = _bounds(names, limits)
        else:
            print(f"Note: {words} cannot keep to the limits in between=, so they were used only for the start.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = optimize.minimize(objective, np.asarray(first, dtype=float), method=scipy_name, options=options,
                                  **kwargs)
    place = np.atleast_1d(found.x)
    start_words = _places(names, first)
    limit_words = " with the limits in between=" if "bounds" in kwargs else ""
    sentence = f"Method: {words} from {start_words}{limit_words}, as {r_name}."
    if chosen == "nelder-mead":
        sentence += " It finds the nearest dip to the start; search=\"whole range\" looks everywhere."
    iterations = int(getattr(found, "nit", getattr(found, "nfev", 0)) or 0)
    return _Answer(place, float(found.fun), sentence, settled=bool(found.success), iterations=iterations,
                   method_name=words)


def _constraints(names, rule_list):
    out = []
    for rule in rule_list:
        def gap(theta, rule=rule):
            return rule.gap(dict(zip(names, np.atleast_1d(theta))))

        if rule.sign == "=":
            out.append({"type": "eq", "fun": gap})
        elif rule.sign == "<=":
            out.append({"type": "ineq", "fun": lambda theta, g=gap: -g(theta)})
        else:
            out.append({"type": "ineq", "fun": gap})
    return out


def _with_rules(names, limits, objective, rule_list, first, chosen, all_linear):
    from scipy import optimize

    scipy_name, words, r_name = LOCAL_METHODS[chosen]
    options = {"slsqp": {"ftol": 1e-14, "maxiter": 1000}, "cobyla": {"tol": 1e-8, "maxiter": 5000},
               "trust region": {"xtol": 1e-12, "gtol": 1e-10, "maxiter": 10000}}[chosen]
    bounds = _bounds(names, limits)
    kwargs = {"bounds": bounds} if any(b != (None, None) for b in bounds) else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = optimize.minimize(objective, np.asarray(first, dtype=float), method=scipy_name,
                                  constraints=_constraints(names, rule_list), options=options, **kwargs)
    place = np.atleast_1d(found.x)
    values = dict(zip(names, place))
    broken = [r.text for r in rule_list if not r.met_by(values, 1e-6 * (1 + max(abs(float(v)) for v in place)))]
    reason = ("you chose this method" if all_linear else "the formula or a rule is not straight-line (linear)")
    sentence = (f"Method: {words} under the rules, from {_places(names, first)}, as {r_name}, because {reason}. "
                "It finds the nearest best point to the start.")
    notes = []
    settled = bool(found.success)
    if broken:
        settled = False
        notes.append(f"Note: the answer does not keep to {_list_words(broken)}; the rules may be impossible to "
                     "meet together, or the search needs a start nearer the answer.")
    iterations = int(getattr(found, "nit", 0) or 0)
    return _Answer(place, float(found.fun), sentence, settled=settled, iterations=iterations, notes=notes,
                   method_name=words)


def _linear(step, names, limits, linear_goal, rule_list, whole, biggest, highs_method):
    from scipy import optimize

    coefficients, _constant = linear_goal
    c = np.array([coefficients.get(n, 0.0) for n in names]) * (-1.0 if biggest else 1.0)
    upper_rows, upper_right, equal_rows, equal_right, kinds = [], [], [], [], []
    for rule in rule_list:
        parts, constant = rule.formula.linear_parts()
        row = [parts.get(n, 0.0) for n in names]
        if rule.sign == "<=":
            upper_rows.append(row)
            upper_right.append(-constant)
            kinds.append(("upper", len(upper_rows) - 1, 1.0))
        elif rule.sign == ">=":
            upper_rows.append([-v for v in row])
            upper_right.append(constant)
            kinds.append(("upper", len(upper_rows) - 1, -1.0))
        else:
            equal_rows.append(row)
            equal_right.append(-constant)
            kinds.append(("equal", len(equal_rows) - 1, 1.0))
    bounds = _bounds(names, limits)
    goal_constant = linear_goal[1]
    integer = [1 if n in whole else 0 for n in names]
    if whole:
        lows = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        highs = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        constraints = []
        if upper_rows:
            constraints.append(optimize.LinearConstraint(np.array(upper_rows), -np.inf, np.array(upper_right)))
        if equal_rows:
            constraints.append(optimize.LinearConstraint(np.array(equal_rows), np.array(equal_right),
                                                         np.array(equal_right)))
        found = optimize.milp(c, integrality=np.array(integer), bounds=optimize.Bounds(lows, highs),
                              constraints=constraints)
        status = found.status
    else:
        found = optimize.linprog(c, A_ub=np.array(upper_rows) if upper_rows else None,
                                 b_ub=np.array(upper_right) if upper_rows else None,
                                 A_eq=np.array(equal_rows) if equal_rows else None,
                                 b_eq=np.array(equal_right) if equal_rows else None,
                                 bounds=bounds, method=highs_method)
        status = found.status
    if status == 2:
        raise SimplydataError("No values meet every rule at once, so there is no answer. Check the rules: two of "
                              "them may contradict each other (such as x >= 5 and x <= 3).")
    if status == 3:
        raise SimplydataError(f"The formula can grow without end under these rules, so it has no "
                              f"{'biggest' if biggest else 'smallest'} value. Is a rule missing, such as x >= 0?")
    if found.x is None:
        raise SimplydataError(f"Linear programming stopped without an answer ({found.message}).")
    place = np.asarray(found.x, dtype=float)
    value = float(c @ place) + (-goal_constant if biggest else goal_constant)
    if whole:
        whole_words = _list_words(whole) + (" must" if len(whole) > 1 else " must")
        sentence = (f"Method: mixed-integer linear programming (HiGHS), as R lpSolve::lp(int.vec = ...), because the "
                    f"formula and every rule are straight-line (linear) and {whole_words} be whole numbers.")
        extra = {}
        iterations = int(getattr(found, "mip_node_count", 0) or 0)
        name = "mixed-integer linear programming (HiGHS)"
    else:
        sentence = ("Method: linear programming (HiGHS), as R lpSolve::lp(), because the formula and every rule "
                    "are straight-line (linear).")
        extra = {}
        upper = getattr(found, "ineqlin", None)
        equal = getattr(found, "eqlin", None)
        for rule, (kind, index, direction) in zip(rule_list, kinds):
            marginals = (upper if kind == "upper" else equal)
            if marginals is None or marginals.marginals is None:
                continue
            extra[rule.text] = float(direction * marginals.marginals[index]) + 0.0
        iterations = int(getattr(found, "nit", 0) or 0)
        name = "linear programming (HiGHS)"
        if extra:
            changes = [f"{text}: {_show(-v if biggest else v)}" for text, v in extra.items() if abs(v) > 1e-12]
            if changes:
                sentence += (" Value of one more on the right of each rule that limits the answer (shadow price): "
                             + "; ".join(changes) + ".")
    return _Answer(place, value * (-1.0 if biggest else 1.0) if False else float(c @ place) +
                   (-goal_constant if biggest else goal_constant), sentence, settled=status == 0,
                   iterations=iterations, extra=extra, method_name=name)


def _whole_range(step, names, limits, objective, rule_list, whole, start, chosen):
    from scipy import optimize

    open_ended = [n for n in names if not all(math.isfinite(v) for v in limits[n])]
    if open_ended:
        raise SimplydataError(
            f'search="whole range" needs a range with two ends for every name; {_list_words(open_ended)} '
            f"{'has' if len(open_ended) == 1 else 'have'} none. For example between="
            + ("[-10, 10]" if len(names) == 1 else "{" + ", ".join(f'"{n}": [-10, 10]' for n in names[:2]) + "}")
            + "."
        )
    bounds = [limits[n] for n in names]
    words, r_name = GLOBAL_METHODS[chosen]
    if whole and chosen != "differential evolution":
        raise SimplydataError('whole_numbers= with search="whole range" works with method="differential evolution".')
    if rule_list and chosen not in ("differential evolution", "shgo"):
        raise SimplydataError(f'rules= with search="whole range" works with method="differential evolution" or '
                              f'"shgo", not "{chosen}".')
    constraints = []
    if rule_list:
        for rule in rule_list:
            def gap(theta, rule=rule):
                return rule.gap(dict(zip(names, np.atleast_1d(theta))))
            low, high = {"<=": (-np.inf, 0.0), ">=": (0.0, np.inf), "=": (0.0, 0.0)}[rule.sign]
            constraints.append(optimize.NonlinearConstraint(gap, low, high))
    polish_words = " then a local polish (L-BFGS-B)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if chosen == "differential evolution":
            integrality = [n in whole for n in names] if whole else None
            found = optimize.differential_evolution(objective, bounds, rng=SEED, tol=1e-10, maxiter=3000,
                                                    constraints=tuple(constraints), integrality=integrality,
                                                    polish=not whole)
            if whole:
                polish_words = " (no polish, because some names are whole numbers)"
            elif constraints:
                polish_words = " then a local polish (trust-constr)"
        elif chosen == "dual annealing":
            found = optimize.dual_annealing(objective, bounds, rng=SEED)
            polish_words = " with local searches (L-BFGS-B)"
        elif chosen == "shgo":
            shgo_constraints = _constraints(names, rule_list) if rule_list else None
            found = optimize.shgo(objective, bounds, constraints=shgo_constraints)
            polish_words = " with local searches (SLSQP)"
        elif chosen == "basin hopping":
            first = _starts(step, names, start, limits, True, 'start={"x": 1}')
            found = optimize.basinhopping(objective, first, rng=SEED, niter=200,
                                          minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds})
            found.success = bool(found.lowest_optimization_result.success)
            polish_words = " with local searches (L-BFGS-B)"
        else:
            found = optimize.direct(objective, bounds, maxfun=20000 * len(names), len_tol=1e-8)
            polished = optimize.minimize(objective, found.x, method="L-BFGS-B", bounds=bounds,
                                         options={"ftol": 1e-15, "gtol": 1e-10})
            if polished.fun <= found.fun:
                found.x, found.fun = polished.x, polished.fun
    place = np.atleast_1d(found.x)
    ranges = ", ".join(f"{n} {_range_words(*limits[n])}" for n in names[:SHOWN_NAMES])
    seed_words = f", seed={SEED}" if chosen in ("differential evolution", "dual annealing", "basin hopping") else ""
    sentence = (f"Method: {words} over the whole range ({ranges}{seed_words}){polish_words}, as {r_name}. It looks "
                "everywhere in between=, so it finds the lowest dip, not only the nearest one; it is a search, "
                "not a proof, so a very narrow dip can still be missed.")
    if r_name.startswith("SciPy"):
        sentence = sentence.replace(f", as {r_name}.", f" ({r_name}).")
    iterations = int(getattr(found, "nit", 0) or 0)
    settled = bool(getattr(found, "success", True))
    if rule_list:
        values = dict(zip(names, place))
        if not all(r.met_by(values, 1e-6 * (1 + max(abs(float(v)) for v in place))) for r in rule_list):
            settled = False
    return _Answer(place, float(found.fun), sentence, settled=settled, iterations=iterations,
                   method_name=words)


@friendly
def find_smallest(formula=None, between=None, start=None, rules=None, whole_numbers=False, search="from start",
                  method=None, known=None, fast=False):
    """Find where a formula is smallest, and its value there.

    sdata.find_smallest("x**2 - 4*x + 1", between=[0, 5])   (the smallest value is -3, at x = 2)
    The names in the formula (x, y, a, rate) are what to find. Give a range with between=
    ([0, 5] for one name; {"x": [0, 5], "y": [0, None]} for several, None = no limit) or
    starting guesses with start= (a number, or {"x": 1, "y": 2}). rules=["x + y <= 4", "x >= 0"]
    adds rules the answer must keep (=, <= or >=); whole_numbers=True asks for whole numbers
    (with a straight-line formula and rules). search="whole range" looks everywhere in
    between= instead of near the start. known={"a": 2.1} (or a one-row result table from
    table.fit_formula) fixes some names. The method is decided by what you give and printed
    (R optimize(), optim(), lpSolve::lp(); see the module notes); method= lets experts choose
    ("nelder-mead", "bfgs", "l-bfgs-b", "conjugate gradient", "powell", "trust region",
    "slsqp", "cobyla"; with search="whole range": "differential evolution", "dual annealing",
    "shgo", "basin hopping", "direct"). Gives back a one-row table: the value of each name,
    the smallest value, the method, whether the search settled, and for linear programs the
    value of one more on the right of each rule (shadow price, R lp(compute.sens = TRUE)$duals).
    fast=True gives the table on the fast engine (Polars).
    """
    return _find("find_smallest", "smallest", formula, between, start, rules, whole_numbers, search, method,
                 known, fast)


@friendly
def find_biggest(formula=None, between=None, start=None, rules=None, whole_numbers=False, search="from start",
                 method=None, known=None, fast=False):
    """Find where a formula is biggest, and its value there.

    sdata.find_biggest("3*x + 2*y", rules=["x + y <= 4", "x + 3*y <= 6", "x >= 0", "y >= 0"])   (12, at x = 4, y = 0)
    The same options as find_smallest(): between=, start=, rules=, whole_numbers=,
    search="whole range", method=, known= (a dict, or a one-row result table from
    table.fit_formula, so a fitted curve can be handed straight over). It works by finding
    the smallest value of minus the formula, and prints the biggest. Gives back a one-row
    table: each name, the biggest value, the method, whether it settled, and for linear
    programs the value of one more on the right of each rule (shadow price).
    """
    return _find("find_biggest", "biggest", formula, between, start, rules, whole_numbers, search, method,
                 known, fast)


# ---------------------------------------------------------------------------- solve
def _solve_example():
    return 'sdata.solve("x**3 - 2*x = 5", between=[0, 3])'


def _read_solve_equation(text, known):
    if not isinstance(text, str):
        raise SimplydataError(f"solve() needs an equation in quotes, for example {_solve_example()}.")
    signs = _SIGN_PATTERN.findall(text)
    if not signs:
        raise SimplydataError(f'solve() needs an = sign, for example {_solve_example()}; "{text}" has none.\n'
                              'To find where a formula is smallest, use sdata.find_smallest("...").')
    formula, sign = read_equation(text, known=known)
    if sign != "=":
        raise SimplydataError(f'solve() works with = only; "{text}" is a rule. To keep to rules, use '
                              'sdata.find_smallest(formula, rules=["..."]).')
    return formula


def _complex_words(z) -> str:
    real, imaginary = _show(z.real), _show(abs(z.imag))
    sign = "+" if z.imag >= 0 else "-"
    return f"{real} {sign} {imaginary}i" if real != "0" else (f"{'-' if z.imag < 0 else ''}{imaginary}i")


def _solve_one_range(text, name, gap, low, high, chosen):
    from scipy import optimize

    xs = np.linspace(low, high, SOLVE_STEPS + 1)
    with np.errstate(all="ignore"):
        ys = np.array([gap(x) for x in xs])
    roots, jumps = [], 0
    scale = np.nanmax(np.abs(ys[np.isfinite(ys)])) if np.isfinite(ys).any() else 1.0
    finder = {"brent": optimize.brentq, "bisection": optimize.bisect, "ridder": optimize.ridder,
              "toms748": optimize.toms748}[chosen]
    for i in range(len(xs)):
        if ys[i] == 0:
            roots.append(float(xs[i]))
            continue
        if i + 1 < len(xs) and np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and ys[i] * ys[i + 1] < 0:
            root = finder(gap, xs[i], xs[i + 1], xtol=ROOT_TOL, maxiter=1000)
            if abs(gap(root)) <= 1e-6 * (1 + scale):
                roots.append(float(root))
            else:
                jumps += 1
    roots = sorted(set(roots))
    return roots, jumps


@friendly
def solve(equations=None, between=None, start=None, method=None, known=None, fast=False):
    """Find the values that make an equation true.

    sdata.solve("x**3 - 2*x = 5", between=[0, 3])   (x = 2.09455)
    One equation with one name: between=[low, high] finds every solution in the range
    where the two sides cross (Brent's method, R uniroot() on each sign change, as
    rootSolve::uniroot.all()); start= finds one solution by searching from the start (secant
    method); a polynomial (only + - * and whole powers) needs neither and gives all its
    solutions, complex ones too (R polyroot()). Several equations in a list:
    sdata.solve(["x + y = 3", "x * y = 2"], start={"x": 0, "y": 1}) (Powell's hybrid method, as
    R nleqslv::nleqslv()); straight-line equations are solved exactly (R solve()).
    known= fixes some names (a dict, or a one-row result table from table.fit_formula).
    Gives back a table with one row per solution: each name, "left minus right" at the
    solution (the check: close to 0) and the method. fast=True gives a Polars table.
    """
    if isinstance(equations, str):
        texts = [equations]
    elif isinstance(equations, (list, tuple)) and equations:
        texts = list(equations)
    else:
        raise SimplydataError(f"solve() needs an equation in quotes, for example {_solve_example()}.")
    formulas = [_read_solve_equation(t, known) for t in texts]
    names: list[str] = []
    for f in formulas:
        for n in f.names:
            if n not in names:
                names.append(n)
    used_known = {}
    for f in formulas:
        used_known.update(f.known)
    if used_known:
        print("Using " + ", ".join(f"{k} = {_show(v)}" for k, v in used_known.items()) + " from known=.")
    if not names:
        raise SimplydataError(f'"{texts[0]}" has no name to find, such as x. For example {_solve_example()}.')
    if len(names) > len(formulas):
        raise SimplydataError(
            f"There are more names to find ({', '.join(names)}) than equations ({len(formulas)}), so there are "
            "endless solutions. Give one equation for each name, or fix some names with known={...}."
        )
    if len(names) < len(formulas):
        raise SimplydataError(
            f"There are more equations ({len(formulas)}) than names to find ({', '.join(names)}). Give one equation "
            "for each name. (To find the values that come closest to meeting them all, use table.fit_formula.)"
        )
    limits, has_limits = _limits("solve", names, between, "between=[0, 3]" if len(names) == 1 else
                                 'between={"x": [0, 3], "y": [0, 3]}')
    if len(names) == 1:
        return _solve_one(texts[0], names[0], formulas[0], limits, has_limits, start, method, fast)
    return _solve_system(texts, names, formulas, limits, start, method, fast)


def _solve_one(text, name, formula, limits, has_limits, start, method, fast):
    from scipy import optimize

    chosen = _plain_method("solve", method, SOLVE_ONE_METHODS, SOLVE_SPELLINGS)

    def gap(x):
        return float(formula({name: x}))

    low, high = limits[name]
    rows = []
    if has_limits and math.isfinite(low) and math.isfinite(high):
        if chosen == "secant":
            raise SimplydataError('method="secant" searches from start=; with between= use method="brent".')
        if start is not None:
            print("Note: start= is not used when between= gives a range with two ends: every solution in the "
                  "range is searched for.")
        roots, jumps = _solve_one_range(text, name, gap, low, high, chosen or "brent")
        if not roots:
            raise SimplydataError(
                f"The two sides of {text} never cross between {_show(low)} and {_show(high)}; try a wider between=.\n"
                f"(A solution where the curve only touches, like x**2 = 0, has no crossing: "
                f'sdata.find_smallest("abs({formula.left_text} - ({formula.right_text}))", between=...) finds it.)'
            )
        values = ", ".join(_show(r) for r in roots)
        if len(roots) == 1:
            print(f"{text} between {_show(low)} and {_show(high)} when {name} = {values}.")
        else:
            print(f"{text} has {len(roots)} solutions between {_show(low)} and {_show(high)}: {name} = {values}.")
        words = SOLVE_ONE_METHODS[chosen or "brent"]
        r_words = "R uniroot()" if (chosen or "brent") == "brent" else "R uniroot() (Brent's method) for the same answer"
        print(f"Method: {words}, as {r_words}; every sign change on {SOLVE_STEPS} steps across the range was "
              "searched, so all solutions that cross were found (one where the curve only touches, like x**2 = 0, "
              "can be missed).")
        if jumps:
            print(f"Note: {jumps} sign change{'s were jumps' if jumps > 1 else ' was a jump'} (such as a division by "
                  "0), not a solution, and left out.")
        method_name = words
        for r in roots:
            rows.append({name: r, "left minus right": gap(r), "method": method_name})
        return _result(rows, fast)
    if start is not None:
        if chosen not in (None, "secant"):
            raise SimplydataError(f'method="{chosen}" needs between=[low, high] (a range where the sides cross).')
        first = _starts("solve", [name], start, limits, True, "start=1")[0]
        step = 1e-4 * max(1.0, abs(first))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = optimize.root_scalar(gap, x0=first, x1=first + step, method="secant", xtol=ROOT_TOL,
                                         maxiter=1000)
        root = float(found.root)
        check = gap(root)
        scale = 1 + abs(gap(first))
        if not found.converged or not math.isfinite(check) or abs(check) > 1e-8 * scale:
            raise SimplydataError(
                f"Searching from {name} = {_show(first)}, simplydata found no solution of {text}. Try another start, "
                "or between=[low, high] to search a whole range."
            )
        print(f"{text} when {name} = {_show(root)}: the solution found by searching from {name} = {_show(first)} "
              "(usually the one nearest the start; between= finds every one in a range).")
        print("Method: the secant method from the start, as R uniroot() on an interval around the answer.")
        if has_limits and not low <= root <= high:
            print(f"Note: the solution is outside between= ({_range_words(low, high)}).")
        return _result([{name: root, "left minus right": check, "method": "the secant method"}], fast)
    coefficients = formula.polynomial()
    if coefficients is None or has_limits:
        raise SimplydataError(
            f"solve() needs between=[low, high] or start= for {text}" +
            (" (it is not a polynomial, so its solutions cannot all be listed)" if coefficients is None else "") +
            f". For example {_solve_example()}."
        )
    return _solve_polynomial(text, name, coefficients, gap, fast)


def _solve_polynomial(text, name, coefficients, gap, fast):
    coefficients = list(coefficients)
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    if len(coefficients) == 1:
        if coefficients[0] == 0:
            raise SimplydataError(f"{text} is true for every {name}: the two sides are the same.")
        raise SimplydataError(f"{text} is never true: {name} cancels out and the two sides always differ.")
    roots = np.polynomial.polynomial.polyroots(coefficients)
    size = max(1.0, float(np.max(np.abs(roots)))) if len(roots) else 1.0
    real, other = [], []
    for z in roots:
        z = complex(z)
        if abs(z.imag) <= 1e-9 * size:
            real.append(z.real)
        else:
            other.append(z)
    real.sort()
    other.sort(key=lambda z: (z.real, -z.imag))
    degree = len(coefficients) - 1
    if real:
        values = ", ".join(_show(r) for r in real)
        print(f"{text} when {name} = {values}." if len(real) == 1 else
              f"{text} has {len(real)} real solutions: {name} = {values}.")
    else:
        print(f"{text} has no solution among the real numbers.")
    if other:
        more = "more " if real else ""
        print(f"{len(other)} {more}solution{'s are' if len(other) > 1 else ' is'} not real number"
              f"{'s' if len(other) > 1 else ''}: {_list_words(_complex_words(z) for z in other)}.")
    print(f"Method: all {degree} solutions of a polynomial of degree {degree}, from the eigenvalues of its companion "
          "matrix, as R polyroot(). Solutions that appear twice are listed twice.")
    rows = []
    for r in real:
        rows.append({name: r, "solution": _show(r), "real number": "yes", "left minus right": gap(r),
                     "method": "polynomial roots"})
    for z in other:
        value = np.polynomial.polynomial.polyval(z, coefficients)
        rows.append({name: math.nan, "solution": _complex_words(z), "real number": "no",
                     "left minus right": abs(complex(value)), "method": "polynomial roots"})
    if not other:
        for row in rows:
            del row["solution"], row["real number"]
    return _result(rows, fast)


def _solve_system(texts, names, formulas, limits, start, method, fast):
    from scipy import optimize

    linear = [f.linear_parts() for f in formulas]
    chosen = _plain_method("solve", method, SOLVE_SYSTEM_METHODS, SOLVE_SPELLINGS)
    if all(part is not None for part in linear) and chosen is None:
        matrix = np.array([[parts.get(n, 0.0) for n in names] for parts, _ in linear])
        right = np.array([-constant for _, constant in linear])
        if np.linalg.matrix_rank(matrix) < len(names):
            raise SimplydataError("These equations do not pin down one answer: some say the same thing, or "
                                  "contradict each other. Check them, or drop one name with known={...}.")
        place = np.linalg.solve(matrix, right)
        print(f"{_list_words(texts)} when {_places(names, place)}.")
        print("Method: the equations are straight-line (linear), so they were solved exactly (Gaussian "
              "elimination, LU), as R solve().")
        method_name = "linear equations, solved exactly"
    else:
        chosen = chosen or "powell hybrid"
        scipy_name, words, r_name = SOLVE_SYSTEM_METHODS[chosen]
        first = _starts("solve", names, start, limits, True,
                        "start={" + ", ".join(f'"{n}": 1' for n in names[:3]) + "}")

        def gaps(theta):
            values = dict(zip(names, theta))
            return np.array([float(f(values)) for f in formulas])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = optimize.root(gaps, np.asarray(first, dtype=float), method=scipy_name,
                                  options={"xtol": 1e-13} if scipy_name in ("hybr", "lm") else {})
        place = np.asarray(found.x, dtype=float)
        check = gaps(place)
        if not np.all(np.isfinite(check)) or np.max(np.abs(check)) > 1e-8 * (1 + np.max(np.abs(gaps(first)))):
            raise SimplydataError(
                f"Searching from {_places(names, first)}, simplydata found no values that make every equation true. "
                "Try other start values, or check that the equations can all be true at once."
            )
        print(f"{_list_words(texts)} when {_places(names, place)}: the solution found by searching from "
              f"{_places(names, first)} (other solutions may exist; another start= may find them).")
        print(f"Method: {words} from the start, as {r_name}.")
        for name in names:
            low, high = limits[name]
            if not low <= place[names.index(name)] <= high:
                print(f"Note: {name} is outside between= ({_range_words(low, high)}).")
        method_name = words
    values = dict(zip(names, place))
    row = {n: float(v) for n, v in values.items()}
    for text, f in zip(texts, formulas):
        row[f"left minus right: {text}"] = float(f(values))
    row["method"] = method_name
    return _result([row], fast)
