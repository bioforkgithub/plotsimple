"""Tests for the power words: how_many_needed and chance_to_find.

Reference values come from R 4.6.2: stats::power.t.test() (strict = FALSE, the default) and
pwr 1.3 (pwr.anova.test, pwr.chisq.test, pwr.r.test). The exact sample sizes were solved in R
with a tight tolerance, because R's own uniroot() stops at about 1e-4:
power.t.test(..., tol = 1e-13), and for pwr
uniroot(function(n) pwr.anova.test(k = k, n = n, f = f)$power - power, c(2 + 1e-10, 1e9), tol = 1e-13).
These words take no table, so there is nothing to run on two engines.
"""

import pytest

from simplydata import SimplydataError
from simplydata.power import chance_to_find, how_many_needed

TOL = 1e-6


def row(result):
    return result._frame.to_pandas().iloc[0]


# ------------------------------------------------------------------ how_many_needed
@pytest.mark.parametrize("kwargs, exact, needed, in_all", [
    # power.t.test(delta = 0.5, power = 0.8)$n
    (dict(effect_size=0.5), 63.765763724846, 64, 128),
    # power.t.test(delta = 5, sd = 10, power = 0.9, alternative = "one.sided")$n
    (dict(difference=5, spread=10, power=0.9, sides=1), 69.197821844886, 70, 140),
    # power.t.test(delta = 0.3, power = 0.8, type = "paired")$n
    (dict(test="compare_before_after", effect_size=0.3), 89.149577266243, 90, 90),
    # power.t.test(delta = 0.8, power = 0.9, sig.level = 0.01, type = "one.sample", alternative = "one.sided")$n
    (dict(test="compare_average_to", effect_size=0.8, power=0.9, significance=0.01, sides=1),
     23.138957670919, 24, 24),
    # power.t.test(delta = 3, power = 0.8)$n
    (dict(effect_size=3), 3.070009746949, 4, 8),
    # pwr.anova.test(k = 3, f = 0.25, power = 0.8)$n  (R's default answer: 52.3966016)
    (dict(groups=3, effect_size=0.25), 52.396597460113, 53, 159),
    # pwr.r.test(r = 0.3, power = 0.8)$n
    (dict(test="correlation_between", effect_size=0.3), 84.073637744296, 85, 85),
    # pwr.r.test(r = 0.25, power = 0.9, alternative = "greater")$n
    (dict(test="correlation_between", effect_size=0.25, power=0.9, sides=1), 133.280329608771, 134, 134),
    # a negative difference is the same size
    (dict(difference=-5, spread=10), 63.765763724846, 64, 128),
])
def test_how_many_needed_matches_r(kwargs, exact, needed, in_all):
    got = row(how_many_needed(**kwargs))
    assert got["values per group (exact)"] == pytest.approx(exact, abs=TOL)
    assert got["values per group"] == needed
    assert got["values in all"] == in_all


@pytest.mark.parametrize("w, groups, total, needed, in_all", [
    # pwr.chisq.test(w = 0.3, df = 1, power = 0.8)$N
    (0.3, 2, 87.209561214736, 44, 88),
    # pwr.chisq.test(w = 0.2, df = 6, power = 0.9)$N  (3 groups, outcome with 4 kinds)
    (0.2, (3, 4), 435.470660723361, 146, 438),
])
def test_how_many_needed_counts_matches_r(w, groups, total, needed, in_all):
    power = 0.9 if groups == (3, 4) else 0.8
    got = row(how_many_needed("compare_counts", effect_size=w, groups=groups, power=power))
    k = 3 if groups == (3, 4) else 2
    assert got["values per group (exact)"] * k == pytest.approx(total, abs=TOL)
    assert got["values per group"] == needed
    assert got["values in all"] == in_all
    assert got["degrees of freedom"] == (6 if k == 3 else 1)


def test_how_many_needed_sentence_and_table(capsys):
    got = row(how_many_needed(effect_size=0.5))
    out = capsys.readouterr().out
    assert ("To find a difference of 0.5 standard deviations (Cohen's d) with an 80 % chance, using a "
            "two-sided test at 0.05, you need 64 values in each group (128 in all). Method: two-sample "
            "t-test power, as R power.t.test().") in out
    assert "equal-spread formula" in out and "Welch" in out
    assert "medium effect" in out
    # power.t.test(n = 64, delta = 0.5)$power
    assert got["power with that many"] == pytest.approx(0.801458623368, abs=TOL)
    assert got["test"] == "compare_averages"
    assert got["effect size name"] == "Cohen's d"
    assert got["power wanted"] == 0.8 and got["significance"] == 0.05 and got["sides"] == 2


def test_cohen_words(capsys):
    how_many_needed(effect_size=0.1)
    assert "very small effect" in capsys.readouterr().out
    how_many_needed("compare_counts", effect_size=0.5)
    assert "a large effect" in capsys.readouterr().out
    how_many_needed(groups=4, effect_size=0.15)
    assert "a small effect" in capsys.readouterr().out


def test_words_for_each_test(capsys):
    how_many_needed("compare_before_after", difference=2, spread=4)
    out = capsys.readouterr().out
    assert "an average change of 2 (with a standard deviation of the changes of 4: Cohen's d = 0.5)" in out
    assert "pairs" in out
    how_many_needed("compare_counts", effect_size=0.3)
    out = capsys.readouterr().out
    assert "(1 degree of freedom)" in out and "Yates" in out
    how_many_needed("correlation_between", effect_size=-0.3)
    assert "85 pairs of values" in capsys.readouterr().out


# ------------------------------------------------------------------ chance_to_find
@pytest.mark.parametrize("kwargs, power", [
    # power.t.test(n = 64, delta = 0.5)$power
    (dict(effect_size=0.5, values_per_group=64), 0.801458623368),
    # power.t.test(n = 30, delta = 0.5)$power
    (dict(effect_size=0.5, values_per_group=30), 0.477840985941),
    # power.t.test(n = 40, delta = 5, sd = 10, alternative = "one.sided")$power
    (dict(difference=5, spread=10, values_per_group=40, sides=1), 0.716254868650),
    # power.t.test(n = 30, delta = 0.3, type = "paired")$power
    (dict(test="compare_before_after", effect_size=0.3, values_per_group=30), 0.355270660769),
    # power.t.test(n = 15, delta = 0.8, sig.level = 0.01, type = "one.sample", alternative = "one.sided")$power
    (dict(test="compare_average_to", effect_size=0.8, values_per_group=15, significance=0.01, sides=1),
     0.680525774379),
    # power.t.test(n = 20, delta = 0.5, type = "one.sample")$power
    (dict(test="compare_average_to", effect_size=0.5, values_per_group=20), 0.564482896679),
    # power.t.test(n = 4, delta = 3)$power
    (dict(effect_size=3, values_per_group=4), 0.938935739925),
    # pwr.anova.test(k = 4, n = 30, f = 0.2)$power
    (dict(groups=4, effect_size=0.2, values_per_group=30), 0.411516309981),
    # pwr.chisq.test(w = 0.3, N = 100, df = 1)$power
    (dict(test="compare_counts", effect_size=0.3, values_per_group=50), 0.850838768327),
    # pwr.chisq.test(w = 0.25, N = 150, df = 2)$power  (3 groups, yes/no outcome)
    (dict(test="compare_counts", effect_size=0.25, groups=3, values_per_group=50), 0.788400516851),
    # pwr.r.test(n = 50, r = 0.3)$power
    (dict(test="correlation_between", effect_size=0.3, values_per_group=50), 0.571555841860),
    # pwr.r.test(n = 50, r = 0.3, alternative = "greater")$power
    (dict(test="correlation_between", effect_size=0.3, values_per_group=50, sides=1), 0.691139485380),
])
def test_chance_to_find_matches_r(kwargs, power):
    assert row(chance_to_find(**kwargs))["power"] == pytest.approx(power, abs=TOL)


def test_chance_to_find_sentence(capsys):
    got = row(chance_to_find(effect_size=0.5, values_per_group=30))
    out = capsys.readouterr().out
    assert ("With 30 values in each group (60 in all), a study has a 47.8 % chance (power = 0.478) to find a "
            "difference of 0.5 standard deviations (Cohen's d), using a two-sided test at 0.05.") in out
    assert "below the usual 80 %" in out
    assert got["values in all"] == 60
    chance_to_find(effect_size=0.5, values_per_group=64)
    out = capsys.readouterr().out
    assert "an 80.1 % chance" in out and "below the usual" not in out


# ------------------------------------------------------------------ friendly errors
@pytest.mark.parametrize("call, words", [
    (lambda: how_many_needed(), "needs the size of the difference"),
    (lambda: how_many_needed(effect_size=0.5, difference=5, spread=10), "not both"),
    (lambda: how_many_needed(difference=5), "both difference= and spread="),
    (lambda: how_many_needed(difference=5, spread=0), "more than 0"),
    (lambda: how_many_needed(effect_size=0.5, power=1.2), "between 0 and 1"),
    (lambda: how_many_needed(effect_size=0.5, power=0), "between 0 and 1"),
    (lambda: how_many_needed(effect_size=0.5, power=0.03), "more than significance"),
    (lambda: how_many_needed(effect_size=0.5, significance=5), "between 0 and 1"),
    (lambda: how_many_needed(effect_size=0), "can never be found"),
    (lambda: how_many_needed("compare_avarages", effect_size=0.5), 'Did you mean test="compare_averages"'),
    (lambda: how_many_needed(effect_size=0.5, sides=3), "sides=2"),
    (lambda: how_many_needed(groups=3, effect_size=0.25, sides=1), "sides=1"),
    (lambda: how_many_needed(groups=3, difference=1, spread=2), "effect_size="),
    (lambda: how_many_needed(groups=1, effect_size=0.5), "groups="),
    (lambda: how_many_needed("compare_counts", effect_size=0.3, groups=(1, 2)), "groups="),
    (lambda: how_many_needed("compare_average_to", effect_size=0.3, groups=3), "groups= is not used"),
    (lambda: how_many_needed("correlation_between", effect_size=1.2), "between -1 and 1"),
    (lambda: how_many_needed(effect_size="medium"), "as a number"),
    (lambda: how_many_needed(effect_size=1e-6), "too small"),
    (lambda: how_many_needed(effect_size=0.5, powr=0.8), "power="),
    (lambda: chance_to_find(effect_size=0.5), "needs values_per_group="),
    (lambda: chance_to_find(effect_size=0.5, values_per_group=1), "at least 2"),
    (lambda: chance_to_find("correlation_between", effect_size=0.3, values_per_group=3), "at least 4"),
])
def test_friendly_errors(call, words):
    with pytest.raises(SimplydataError) as caught:
        call()
    assert words in str(caught.value)
