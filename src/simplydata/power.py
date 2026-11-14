"""Power: how many values a study needs, and its chance to find a real difference.

Before a study, two questions: how many values do I need (how_many_needed), and with the
values I can get, what is the chance to find a real difference of the size I expect
(chance_to_find, the "power")? test= names the simplydata test the study will use.
Numbers come from SciPy's noncentral t, F and chi-square distributions; the formulas are
the ones in R, so results can be checked there:

* compare_averages, 2 groups: two-sample t-test power, R power.t.test() (type = "two.sample").
  Effect size: Cohen's d, the difference in standard deviations (or difference= / spread=).
  It uses the usual equal-spread formula; simplydata's compare_averages uses Welch's
  t-test, which gives almost the same power when the groups are the same size.
* compare_averages, 3 or more groups: one-way ANOVA power for groups of the same size,
  R pwr::pwr.anova.test(). Effect size: Cohen's f, the spread (standard deviation) of the
  group averages divided by the spread within the groups. Again the equal-spread formula,
  close to Welch's ANOVA when the groups are the same size.
* compare_before_after: paired t-test power, R power.t.test(type = "paired"). Effect size:
  Cohen's d_z, the average change divided by the standard deviation of the changes.
* compare_average_to: one-sample t-test power, R power.t.test(type = "one.sample").
* compare_counts: chi-square test power, R pwr::pwr.chisq.test(), with Cohen's w. groups=3
  means 3 groups compared on a yes/no outcome (2 degrees of freedom); groups=(3, 4) means
  3 groups and an outcome with 4 kinds: (3 - 1) x (4 - 1) = 6 degrees of freedom.
  values_per_group times the number of groups is R's N. The formula has no Yates
  correction, so for 2 x 2 tables it is a little hopeful compared with compare_counts.
* correlation_between: R pwr::pwr.r.test(), Fisher's arctanh approximation. Effect: r.

As in R power.t.test(), a two-sided t-test counts only the chance of finding the difference
in the right direction (strict = FALSE); the other direction adds almost nothing. One-sided
tests (sides=1) look in the direction of the expected difference, so the sign of the effect
does not matter. The needed number is always rounded UP to a whole number; the exact value
is in the result too.

Cohen's rough words for effect sizes are only a hint (sizes from your own field are better):
d 0.2 small, 0.5 medium, 0.8 large; f 0.1, 0.25, 0.4; w 0.1, 0.3, 0.5; r 0.1, 0.3, 0.5.
"""

from __future__ import annotations

import math

from ._errors import SimplydataError, choice_error, friendly

TESTS = ("compare_averages", "compare_before_after", "compare_average_to", "compare_counts",
         "correlation_between")
# Cohen's (1988) rough sizes: small, medium, large
COHEN = {"d": (0.2, 0.5, 0.8), "f": (0.1, 0.25, 0.4), "w": (0.1, 0.3, 0.5), "r": (0.1, 0.3, 0.5)}
EFFECT_NAMES = {"d": "Cohen's d", "f": "Cohen's f", "w": "Cohen's w", "r": "correlation r"}
LARGEST = 1e9  # the most values per group searched (as R pwr; power.t.test stops at 1e7)


def _num(x) -> str:
    """3 significant digits without trailing zeros: 0.5, 0.05, 0.681, 10."""
    from .table.stat_tests import _num as num

    text = num(x)
    if "." in text and "e" not in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _percent(share: float) -> str:
    return f"{share * 100:.3g} %"


def _a(text: str) -> str:
    """'an 80 % chance', 'a 90 % chance'."""
    spoken_an = text.startswith("8") or (text[:2] in ("11", "18") and not text[:3].isdigit())
    return "an" if spoken_an else "a"


def _number(step, option, value, example):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SimplydataError(f"{step}() needs {option}= as a number, for example {example}.")
    return float(value)


def _whole_or_num(x) -> str:
    return str(int(x)) if float(x).is_integer() else _num(x)


def _cohen_words(kind: str, size: float) -> str:
    small, medium, large = COHEN[kind]
    word = ("very small" if size < small else "small" if size < medium else
            "medium" if size < large else "large")
    return (f"Cohen's rough guide calls this a {word} effect "
            f"({EFFECT_NAMES[kind]}: {_num(small)} small, {_num(medium)} medium, {_num(large)} large); "
            "it is only a hint, sizes from your own field are better.")


def _plan(step, test, effect_size, difference, spread, significance, groups, sides):
    """Check the inputs and describe the calculation: power(n) for n values per group, and the words."""
    from scipy import stats

    if not isinstance(test, str) or test not in TESTS:
        raise choice_error(step, "test", test, TESTS)
    if isinstance(sides, bool) or sides not in (1, 2):
        raise SimplydataError(f"{step}() needs sides=2 (a difference in either direction, the usual) "
                              "or sides=1 (only in the direction you expect).")
    significance = _number(step, "significance", significance, "significance=0.05")
    if not 0 < significance < 1:
        raise SimplydataError(f"{step}() needs significance= between 0 and 1, for example 0.05.")

    # ------------------------------------------------ groups and the kind of calculation
    df = None
    if test == "compare_counts":
        k, kinds = (tuple(groups) if isinstance(groups, (tuple, list)) and len(groups) == 2 else (groups, 2))
        for value in (k, kinds):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise SimplydataError(
                    f"{step}() needs groups= as a whole number of 2 or more (groups compared on a yes/no "
                    "outcome), or as (groups, kinds of outcome), for example groups=(3, 4).")
        kind, df = "w", (k - 1) * (kinds - 1)
    elif test == "compare_averages":
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 2:
            raise SimplydataError(f"{step}() needs groups= as a whole number of 2 or more, for example groups=3.")
        k = groups
        kind = "d" if k == 2 else "f"
        if kind == "f":
            df = k - 1
    else:
        if isinstance(groups, bool) or groups not in (1, 2):
            raise SimplydataError(f'groups= is not used with test="{test}"; leave it out.')
        k = 1
        kind = "r" if test == "correlation_between" else "d"
    if kind in ("f", "w") and sides == 1:
        what = "The F test (ANOVA)" if kind == "f" else "The chi-square test"
        raise SimplydataError(f"{what} looks for differences in every direction at once, so sides=1 "
                              "does not apply; leave sides= out.")

    # ------------------------------------------------ the effect size
    if effect_size is not None and (difference is not None or spread is not None):
        raise SimplydataError(f"{step}() needs effect_size= OR difference= and spread=, not both.")
    if effect_size is None and difference is None and spread is None:
        example = {"d": "effect_size=0.5 (Cohen's d), or difference=5, spread=10",
                   "f": "effect_size=0.25 (Cohen's f)", "w": "effect_size=0.3 (Cohen's w)",
                   "r": "effect_size=0.3 (the correlation r)"}[kind]
        raise SimplydataError(f"{step}() needs the size of the difference you hope to find, for example "
                              f"{example}.")
    given = {}
    if effect_size is None:
        if kind != "d":
            raise SimplydataError("difference= and spread= work for the t-tests (2 groups, before/after, "
                                  f"average to a value). Here give effect_size= ({EFFECT_NAMES[kind]}).")
        if difference is None or spread is None:
            raise SimplydataError(f"{step}() needs both difference= and spread= (the standard deviation), "
                                  "for example difference=5, spread=10.")
        difference = _number(step, "difference", difference, "difference=5")
        spread = _number(step, "spread", spread, "spread=10")
        if spread <= 0:
            raise SimplydataError("spread= (the standard deviation) must be more than 0.")
        effect = difference / spread
        given = {"difference": difference, "spread": spread}
    else:
        effect = _number(step, "effect_size", effect_size, "effect_size=0.5")
    if kind in ("f", "w") and effect < 0:
        raise SimplydataError(f"{EFFECT_NAMES[kind]} cannot be below 0.")
    if kind == "r" and not -1 < effect < 1:
        raise SimplydataError("A correlation r must be between -1 and 1.")
    effect = abs(effect)
    if effect == 0:
        raise SimplydataError("A difference of 0 can never be found; give the smallest difference that matters.")

    # ------------------------------------------------ power for n values per group
    if kind == "d":
        tsample = 2 if test == "compare_averages" else 1

        def power(n):
            nu = (n - 1) * tsample
            return float(stats.nct.sf(stats.t.isf(significance / sides, nu), nu,
                                      math.sqrt(n / tsample) * effect))
        lowest = 2.0
        method = {"compare_averages": "two-sample t-test power, as R power.t.test()",
                  "compare_before_after": 'paired t-test power, as R power.t.test(type = "paired")',
                  "compare_average_to": 'one-sample t-test power, as R power.t.test(type = "one.sample")'}[test]
    elif kind == "f":
        def power(n):
            d1, d2 = k - 1, (n - 1) * k
            return float(stats.ncf.sf(stats.f.isf(significance, d1, d2), d1, d2, k * n * effect ** 2))
        lowest = 2.0
        method = "one-way ANOVA power for groups of the same size, as R pwr::pwr.anova.test()"
    elif kind == "w":
        def power(n):
            return float(stats.ncx2.sf(stats.chi2.isf(significance, df), df, n * k * effect ** 2))
        lowest = 1.0 / k
        method = "chi-square test power, as R pwr::pwr.chisq.test()"
    else:
        def power(n):
            ttt = stats.t.isf(significance / sides, n - 2)
            rc = math.sqrt(ttt ** 2 / (ttt ** 2 + n - 2))
            zr = math.atanh(effect) + effect / (2 * (n - 1))
            zrc = math.atanh(rc)
            p = stats.norm.cdf((zr - zrc) * math.sqrt(n - 3))
            if sides == 2:
                p += stats.norm.cdf((-zr - zrc) * math.sqrt(n - 3))
            return float(p)
        lowest = 4.0
        method = "correlation power (Fisher's arctanh approximation), as R pwr::pwr.r.test()"

    # ------------------------------------------------ words
    shown = _num(effect)
    if kind == "d":
        if given:
            sd_words = "a standard deviation of the changes" if test == "compare_before_after" else \
                "a standard deviation"
            sizes = f"{_num(abs(difference))} (with {sd_words} of {_num(spread)}: Cohen's d = {shown})"
        elif test == "compare_before_after":
            sizes = f"{shown} standard deviations of the changes (Cohen's d)"
        else:
            sizes = f"{shown} standard deviations (Cohen's d)"
        effect_words = {"compare_averages": f"a difference of {sizes}",
                        "compare_before_after": f"an average change of {sizes}",
                        "compare_average_to": f"a difference from your value of {sizes}"}[test]
    elif kind == "f":
        effect_words = (f"differences between the {k} group averages of Cohen's f = {shown} (the spread of "
                        "the group averages, in standard deviations)")
    elif kind == "w":
        effect_words = f"a difference in counts of Cohen's w = {shown}"
    else:
        effect_words = f"a correlation of r = {shown}"
    if kind in ("d", "r"):
        test_words = f"a {'two' if sides == 2 else 'one'}-sided test at {_num(significance)}"
    elif kind == "f":
        test_words = f"the F test at {_num(significance)}"
    else:
        test_words = (f"the chi-square test at {_num(significance)} ({df} degree{'' if df == 1 else 's'} "
                      "of freedom)")

    def amount(n):
        if test in ("compare_averages", "compare_counts"):
            return f"{_whole_or_num(n)} values in each group ({_whole_or_num(n * k)} in all)"
        if test == "compare_before_after":
            return f"{_whole_or_num(n)} pairs (a before and an after value each)"
        if test == "correlation_between":
            return f"{_whole_or_num(n)} pairs of values"
        return f"{_whole_or_num(n)} values"

    notes = []
    if test == "compare_averages":
        notes.append("Note: the calculation uses the usual equal-spread formula; compare_averages uses Welch's "
                     f"{'t-test' if k == 2 else 'ANOVA'}, which is very close to it when the groups are "
                     "the same size.")
    if kind == "w" and df == 1:
        notes.append("Note: the formula has no Yates correction, so for a 2 x 2 table compare_counts is a "
                     "little less likely to find the difference than this says.")
    notes.append(_cohen_words(kind, effect))

    method_name, r_function = method.split(", as R ")
    row = {"test": test, "method": method_name, "R function": r_function,
           "effect size name": EFFECT_NAMES[kind], "effect size": effect, **given, "groups": k}
    if df is not None:
        row["degrees of freedom"] = df
    if kind in ("d", "r"):
        row["sides"] = sides
    row["significance"] = significance
    return {"power": power, "lowest": lowest, "method": method, "effect_words": effect_words,
            "test_words": test_words, "amount": amount, "notes": notes, "row": row, "k": k}


def _result(row: dict):
    import pandas as pd

    from .table import table

    return table(pd.DataFrame([row]))


@friendly
def how_many_needed(test="compare_averages", effect_size=None, difference=None, spread=None, power=0.8,
                    significance=0.05, groups=2, sides=2):
    """How many values (in each group) does a study need to find a real difference?

    sdata.how_many_needed(effect_size=0.5)       (two groups, a medium difference: 64 in each)
    test= is the simplydata test the study will use: "compare_averages" (2 groups: t-test;
    groups=3 or more: ANOVA with Cohen's f), "compare_before_after" (paired, Cohen's d of the
    changes), "compare_average_to" (one sample), "compare_counts" (chi-square with Cohen's w;
    groups=(groups, kinds of outcome)) or "correlation_between" (r; the answer is pairs).
    The size of the difference: effect_size= (Cohen's d, f, w or r, the sentence says which),
    or difference= and spread= (the standard deviation) for the t-tests.
    power= is the chance you want to find it (0.8 is usual); significance= is the p-value
    limit (0.05); sides=1 looks only in the direction you expect.
    The t-tests use the usual equal-spread formula (R power.t.test()), very close to Welch's
    t-test when the groups are the same size; the others match R pwr. The number is rounded
    UP to a whole number; the exact value is in the result. Gives back a one-row table.
    """
    step = "how_many_needed"
    plan = _plan(step, test, effect_size, difference, spread, significance, groups, sides)
    target = _number(step, "power", power, "power=0.8")
    if not 0 < target < 1:
        raise SimplydataError(f"{step}() needs power= between 0 and 1, for example 0.8 (an 80 % chance).")
    if target <= plan["row"]["significance"]:
        raise SimplydataError("power= must be more than significance=: even with no real difference, a test "
                              "finds one that often by chance.")
    from scipy.optimize import brentq

    chance, lowest, k = plan["power"], plan["lowest"], plan["k"]
    notes = list(plan["notes"])
    if chance(lowest) >= target:
        exact = lowest
        notes.insert(0, "Note: the difference is so large that even the smallest possible study reaches this "
                        "chance.")
    else:
        if chance(LARGEST) < target:
            raise SimplydataError("The difference is too small to find with any practical study (more than "
                                  f"{LARGEST:,.0f} values in each group). Check the effect size.")
        exact = brentq(lambda n: chance(n) - target, lowest, LARGEST, xtol=1e-12, rtol=1e-14, maxiter=500)
    needed = max(math.ceil(exact - 1e-9), math.ceil(lowest))
    reached = chance(needed)
    print(f"To find {plan['effect_words']} with {_a(_percent(target))} {_percent(target)} chance, using "
          f"{plan['test_words']}, you need {plan['amount'](needed)}. Method: {plan['method']}.")
    for note in notes:
        print(note)
    row = dict(plan["row"])
    row.update({"power wanted": target, "values per group (exact)": exact, "values per group": needed,
                "values in all": needed * k, "power with that many": reached})
    return _result(row)


@friendly
def chance_to_find(test="compare_averages", effect_size=None, difference=None, spread=None,
                   values_per_group=None, significance=0.05, groups=2, sides=2):
    """What chance does a study have to find a real difference of a given size (its power)?

    sdata.chance_to_find(effect_size=0.5, values_per_group=30)   (two groups of 30: about 48 %)
    The same options as how_many_needed(), with values_per_group= instead of power=: the
    number in each group (pairs for compare_before_after and correlation_between; all the
    values for compare_average_to). The t-tests use the usual equal-spread formula (R
    power.t.test()), very close to Welch's t-test when the groups are the same size; the
    others match R pwr. Gives back a one-row table.
    """
    step = "chance_to_find"
    plan = _plan(step, test, effect_size, difference, spread, significance, groups, sides)
    if values_per_group is None:
        raise SimplydataError(f"{step}() needs values_per_group=, the number of values in each group, "
                              "for example values_per_group=30.")
    n = _number(step, "values_per_group", values_per_group, "values_per_group=30")
    least = 4 if test == "correlation_between" else 2
    if n < least:
        raise SimplydataError(f"{step}() needs values_per_group= of at least {least}"
                              f"{' (pairs of values)' if least == 4 else ''}.")
    chance = plan["power"](n)
    k = plan["k"]
    shown = _percent(chance)
    print(f"With {plan['amount'](n)}, a study has {_a(shown)} {shown} chance (power = {_num(chance)}) to find "
          f"{plan['effect_words']}, using {plan['test_words']}. Method: {plan['method']}.")
    if chance < 0.8:
        print("Note: this is below the usual 80 %; how_many_needed() tells you how many values you need.")
    for note in plan["notes"]:
        print(note)
    row = dict(plan["row"])
    row.update({"values per group": n, "values in all": n * k, "power": chance})
    return _result(row)
