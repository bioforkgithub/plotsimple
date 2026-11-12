"""Tests for the v0.4.x statistics: compare_spreads, compare_each_pair(test="tukey"),
compare_counts_to, compare_percent_to, the new summary measures (summary_of show=,
percentile_of, most_common_of) and the special functions in formulas.

Every number is checked against R 4.6.2 (reference values hard-coded below, with the R call
that made them), on pandas and Polars tables. The R data are the practice tables, read in R
with read.csv(".../data/marks.csv") and read.csv(".../data/school.csv"); ss is school
without the rows where sleep is empty. car and e1071 are not needed: Levene's test is
worked out in R as anova(lm(abs(y - ave(y, g, FUN = median)) ~ g)), which is what
car::leveneTest() does by default, and skewness / kurtosis by their formulas (e1071 type 2).
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table

ENGINES = ["pandas", "polars"]


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


@pytest.fixture
def marks():
    return sdata.example("marks")


@pytest.fixture
def school():
    return sdata.example("school")


def row(result):
    frame = result._frame.to_pandas()
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


def rows(result):
    return result._frame.to_pandas()


def unchanged(t, before):
    pd.testing.assert_frame_equal(t._frame.to_pandas(), before)
    assert t._history == []


# ================================================================ compare_spreads
def test_levene_brown_forsythe_matches_r(marks, engine, capsys):
    t = make(marks, engine)
    before = t._frame.to_pandas().copy()
    got = row(t.compare_spreads("marks", for_each="class"))
    # R: g <- factor(m$class); anova(lm(abs(m$marks - ave(m$marks, g, FUN = median)) ~ g))
    #    (= car::leveneTest(marks ~ class, data = m), center = median)
    assert got["statistic"] == pytest.approx(0.09374281994333, rel=1e-9)
    assert got["p-value"] == pytest.approx(0.910673418418, rel=1e-9)
    assert got["degrees of freedom 1"] == 2 and got["degrees of freedom 2"] == 51
    assert got["test"].startswith("Levene's test")
    # R: tapply(m$marks, m$class, sd)
    assert got["standard deviation in A"] == pytest.approx(13.7423775663628)
    assert got["standard deviation in B"] == pytest.approx(14.8633206469347)
    assert got["standard deviation in C"] == pytest.approx(14.1800974396755)
    assert got["count in A"] == 18
    out = capsys.readouterr().out
    assert "could easily be chance (p = 0.911)" in out
    assert "car::leveneTest()" in out and "Brown-Forsythe" in out
    unchanged(t, before)
    assert t.compare_spreads("marks", for_each="class", test="Brown-Forsythe")._frame.to_pandas()["statistic"][0] \
        == pytest.approx(got["statistic"])


def test_bartlett_and_fligner_match_r(marks, engine):
    t = make(marks, engine)
    got = row(t.compare_spreads("marks", for_each="class", test="bartlett"))
    # R: bartlett.test(marks ~ class, data = m)
    assert got["statistic"] == pytest.approx(0.103698845381851, rel=1e-9)
    assert got["degrees of freedom"] == 2
    assert got["p-value"] == pytest.approx(0.949471824991942, rel=1e-9)
    got = row(t.compare_spreads("marks", for_each="class", test="fligner"))
    # R: fligner.test(marks ~ class, data = m)
    assert got["statistic"] == pytest.approx(0.282004280539934, rel=1e-9)
    assert got["p-value"] == pytest.approx(0.868487452898067, rel=1e-9)
    assert got["test"] == "Fligner-Killeen test"


def test_spreads_with_empty_cells_and_twelve_groups_match_r(school, engine, capsys):
    t = make(school, engine)
    lev = row(t.compare_spreads("sleep", for_each="school"))
    assert "19 rows with an empty cell" in capsys.readouterr().out
    # R: ss <- s[!is.na(s$sleep), ]; Levene as above on sleep ~ school
    assert lev["statistic"] == pytest.approx(1.127607010742, rel=1e-9)
    assert lev["p-value"] == pytest.approx(0.343747588311, rel=1e-9)
    assert lev["degrees of freedom 1"] == 11 and lev["degrees of freedom 2"] == 149
    bart = row(t.compare_spreads("sleep", for_each="school", test="bartlett"))
    # R: bartlett.test(sleep ~ school, data = s)
    assert bart["statistic"] == pytest.approx(17.7445739625315, rel=1e-9)
    assert bart["p-value"] == pytest.approx(0.0876997805625571, rel=1e-9)
    flig = row(t.compare_spreads("sleep", for_each="school", test="fligner"))
    # R: fligner.test(sleep ~ school, data = s)
    assert flig["statistic"] == pytest.approx(10.9594999689304, rel=1e-9)
    assert flig["p-value"] == pytest.approx(0.446662475634758, rel=1e-9)


def test_spreads_notes_and_errors(engine, capsys):
    rng = np.random.default_rng(3)
    skewed = np.round(rng.exponential(1.0, 40), 3)
    data = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40, "y": np.r_[skewed, np.round(rng.normal(0, 8, 40), 3)]})
    t = make(data, engine)
    got = row(t.compare_spreads("y", for_each="g", test="bartlett"))
    out = capsys.readouterr().out
    assert got["p-value"] < 0.05
    assert 'group "a" don\'t look bell-shaped' in out and 'test="levene"' in out
    assert "does not need equal spreads" in out
    with pytest.raises(SimplydataError, match='did you mean|Did you mean test="levene"'):
        t.compare_spreads("y", for_each="g", test="levine")
    with pytest.raises(SimplydataError, match="cannot be both"):
        t.compare_spreads("y", for_each="y")
    flat = make(pd.DataFrame({"g": list("aaabbb"), "y": [1, 1, 1, 2, 3, 5]}), engine)
    with pytest.raises(SimplydataError, match="Bartlett's test"):
        flat.compare_spreads("y", for_each="g", test="bartlett")
    same = make(pd.DataFrame({"g": list("aabb"), "y": [1, 1, 2, 2]}), engine)
    with pytest.raises(SimplydataError, match="no spread"):
        same.compare_spreads("y", for_each="g")


# ================================================================ compare_each_pair(test="tukey")
def test_tukey_matches_r(marks, engine, capsys):
    t = make(marks, engine)
    got = rows(t.compare_each_pair("marks", for_each="class", test="tukey"))
    # R: TukeyHSD(aov(marks ~ class, data = m)); R gives B-A etc., here A minus B (sign turned)
    r = {("A", "B"): (-7.55555555555556, -19.0375599447001, 3.92644883358901, 0.2598051600910130),
         ("A", "C"): (-13.22222222222221, -24.7042266113668, -1.74021783307765, 0.0204169133370182),
         ("B", "C"): (-5.66666666666666, -17.1486710558112, 5.81533772247791, 0.4637063381572426)}
    for _, pair in got.iterrows():
        diff, low, high, p = r[(pair["group 1"], pair["group 2"])]
        assert pair["difference"] == pytest.approx(-diff, rel=1e-9)
        assert pair["difference 95 % CI low"] == pytest.approx(-high, rel=1e-6)
        assert pair["difference 95 % CI high"] == pytest.approx(-low, rel=1e-6)
        assert pair["p-value (Tukey)"] == pytest.approx(p, rel=1e-6)
    out = capsys.readouterr().out
    assert "TukeyHSD(aov(marks ~ class))" in out and "Tukey-adjusted p = 0.020" in out
    assert "Tukey-Kramer" not in out


def test_tukey_kramer_unequal_sizes_matches_r(school, engine, capsys):
    t = make(school, engine)
    got = rows(t.compare_each_pair("sleep", for_each="school", test="tukey"))
    out = capsys.readouterr().out
    assert len(got) == 66 and "Tukey-Kramer" in out
    # R: TukeyHSD(aov(sleep ~ school, data = ss))$school
    r = {("School A", "School B"): (0.525274725274721, -0.699564735359526, 1.75011418590897, 0.956966569733648),
         ("School K", "School L"): (0.212179487179488, -1.060856517124496, 1.48521549148347, 0.999992073484830),
         ("School A", "School C"): (0.317582417582414, -0.907257043051833, 1.54242187821666, 0.999365481805768)}
    for (g1, g2), (diff, low, high, p) in r.items():
        pair = got[(got["group 1"] == g1) & (got["group 2"] == g2)].iloc[0]
        assert pair["difference"] == pytest.approx(-diff, rel=1e-9)
        assert pair["difference 95 % CI low"] == pytest.approx(-high, rel=1e-6)
        assert pair["difference 95 % CI high"] == pytest.approx(-low, rel=1e-6)
        assert pair["p-value (Tukey)"] == pytest.approx(p, rel=1e-6)
    assert "2.58 times the smallest" in out  # sd 1.33 in School B, 0.513 in School J


def test_each_pair_test_choices(marks, engine):
    t = make(marks, engine)
    default = rows(t.compare_each_pair("marks", for_each="class"))
    welch = rows(t.compare_each_pair("marks", for_each="class", test="welch"))
    pd.testing.assert_frame_equal(default, welch)
    assert "p-value (Holm)" in default.columns
    ranks = rows(t.compare_each_pair("marks", for_each="class", based_on="ranks"))
    mw = rows(t.compare_each_pair("marks", for_each="class", test="mann-whitney"))
    pd.testing.assert_frame_equal(ranks, mw)
    with pytest.raises(SimplydataError, match="based_on=\"ranks\" asks for ranks"):
        t.compare_each_pair("marks", for_each="class", based_on="ranks", test="tukey")
    with pytest.raises(SimplydataError, match='test="tukey"'):
        t.compare_each_pair("marks", for_each="class", test="turkey")


# ================================================================ compare_counts_to
def mendel():
    kinds = ["round yellow"] * 315 + ["round green"] * 108 + ["wrinkled yellow"] * 101 + ["wrinkled green"] * 32
    return pd.DataFrame({"pea": kinds})


def test_counts_to_shares_matches_r(engine, capsys):
    t = make(mendel(), engine)
    before = t._frame.to_pandas().copy()
    expected = {"round yellow": 9 / 16, "round green": 3 / 16, "wrinkled yellow": 3 / 16, "wrinkled green": 1 / 16}
    got = row(t.compare_counts_to("pea", expected=expected))
    # R: chisq.test(c(315, 108, 101, 32), p = c(9, 3, 3, 1) / 16)
    assert got["statistic"] == pytest.approx(0.470023980815348, rel=1e-9)
    assert got["degrees of freedom"] == 3
    assert got["p-value"] == pytest.approx(0.925425895103616, rel=1e-9)
    assert got["expected count of round yellow"] == pytest.approx(312.75)
    assert got["expected count of wrinkled green"] == pytest.approx(34.75)
    assert got["count of round green"] == 108
    # R: pwr::ES.w1(c(9, 3, 3, 1) / 16, c(315, 108, 101, 32) / 556)
    assert got["effect size"] == pytest.approx(0.0290751934124272, rel=1e-9)
    assert got["rows"] == 556
    out = capsys.readouterr().out
    assert "chisq.test(x, p = ...)" in out and "could easily be chance (p = 0.925)" in out
    unchanged(t, before)
    as_percent = row(t.compare_counts_to("pea", expected={k: v * 100 for k, v in expected.items()}))
    assert as_percent["statistic"] == pytest.approx(got["statistic"])


def test_counts_to_equal_shares_matches_r(school, engine, capsys):
    t = make(school, engine)
    got = row(t.compare_counts_to("breakfast"))
    # R: chisq.test(table(s$breakfast))
    assert got["statistic"] == pytest.approx(13.8888888888889, rel=1e-9)
    assert got["p-value"] == pytest.approx(0.000193941629103719, rel=1e-9)
    assert got["expected count of yes"] == 90
    # R: pwr::ES.w1(c(0.5, 0.5), c(65, 115) / 180)
    assert got["effect size"] == pytest.approx(0.277777777777778, rel=1e-9)
    assert "chisq.test(table(x))" in capsys.readouterr().out
    # R: chisq.test(table(s$school)) -- 15 pupils in every school
    even = row(t.compare_counts_to("school"))
    assert even["statistic"] == 0 and even["p-value"] == pytest.approx(1.0) and even["degrees of freedom"] == 11


def test_counts_to_notes_and_errors(engine, capsys):
    t = make(pd.DataFrame({"colour": ["red"] * 6 + ["white"] * 2 + [None]}), engine)
    t.compare_counts_to("colour", expected={"red": 0.5, "white": 0.4, "pink": 0.1})
    out = capsys.readouterr().out
    assert "1 row with an empty cell" in out
    assert "'pink' does not turn up" in out and "below 5" in out
    t.compare_counts_to("colour", expected={"red": 0.7, "white": 0.3})
    assert 'compare_percent_to("colour", value=\'red\', percent=70)' in capsys.readouterr().out
    with pytest.raises(SimplydataError, match="add up to 0.9"):
        t.compare_counts_to("colour", expected={"red": 0.5, "white": 0.4})
    with pytest.raises(SimplydataError, match='Did you mean "white"'):
        t.compare_counts_to("colour", expected={"red": 0.5, "whit": 0.5})
    with pytest.raises(SimplydataError, match="above 0"):
        t.compare_counts_to("colour", expected={"red": 1.0, "white": 0})
    with pytest.raises(SimplydataError, match="share of each value"):
        t.compare_counts_to("colour", expected=[0.5, 0.5])
    with pytest.raises(SimplydataError, match="Did you mean"):
        t.compare_counts_to("color")
    one = make(pd.DataFrame({"colour": ["red"] * 3}), engine)
    with pytest.raises(SimplydataError, match="at least 2 different values"):
        one.compare_counts_to("colour")


# ================================================================ compare_percent_to
def test_percent_to_matches_r(school, engine, capsys):
    t = make(school, engine)
    got = row(t.compare_percent_to("passed", percent=50))
    # R: binom.test(sum(s$passed == "yes"), nrow(s), p = 0.5)
    assert got["statistic"] == 78 and got["rows"] == 180
    assert got["p-value"] == pytest.approx(0.0861873133286952, rel=1e-9)
    assert got["percent"] == pytest.approx(43.3333333333333)
    assert got["percent 95 % CI low"] == pytest.approx(35.9804690177120, rel=1e-9)
    assert got["percent 95 % CI high"] == pytest.approx(50.9094395662734, rel=1e-9)
    # R: pwr::ES.h(78 / 180, 0.5)
    assert got["effect size"] == pytest.approx(-0.133731589409942, rel=1e-9)
    out = capsys.readouterr().out
    assert 'Counted as yes: "yes"' in out and "binom.test(78, 180, p = 0.5)" in out
    got = row(t.compare_percent_to("breakfast", value="yes", percent=70))
    # R: binom.test(115, 180, p = 0.7)
    assert got["p-value"] == pytest.approx(0.0872191378806251, rel=1e-9)
    assert got["percent 95 % CI low"] == pytest.approx(56.4106553004983, rel=1e-9)
    assert got["percent 95 % CI high"] == pytest.approx(70.9013605267408, rel=1e-9)


def test_percent_to_trial_and_other_kinds(engine, capsys):
    trial = sdata.example("trial")
    t = make(trial, engine)
    got = row(t.compare_percent_to("died", "yes", 40))
    # R: binom.test(59, 120, p = 0.4); pwr::ES.h(59 / 120, 0.4)
    assert got["p-value"] == pytest.approx(0.0499034630935699, rel=1e-9)
    assert got["percent 95 % CI low"] == pytest.approx(39.9258582538410, rel=1e-9)
    assert got["effect size"] == pytest.approx(0.184690482422259, rel=1e-9)
    assert "unlikely to be chance (p = 0.050)" in capsys.readouterr().out
    flags = make(pd.DataFrame({"ok": [True] * 59 + [False] * 61, "n": [1] * 59 + [0] * 61}), engine)
    assert row(flags.compare_percent_to("ok", percent=40))["p-value"] == pytest.approx(got["p-value"])
    assert row(flags.compare_percent_to("n", percent=40))["p-value"] == pytest.approx(got["p-value"])
    assert "Counted as yes: 1." in capsys.readouterr().out
    flags.compare_percent_to("ok", percent=0.5)
    assert "For a half write percent=50" in capsys.readouterr().out


def test_percent_to_errors(school, engine):
    t = make(school, engine)
    with pytest.raises(SimplydataError, match="percent= as a number"):
        t.compare_percent_to("passed")
    with pytest.raises(SimplydataError, match="above 0 and below 100"):
        t.compare_percent_to("passed", percent=150)
    with pytest.raises(SimplydataError, match="needs value="):
        t.compare_percent_to("school", percent=10)
    with pytest.raises(SimplydataError, match='Did you mean "School A"'):
        t.compare_percent_to("school", value="School  A", percent=10)
    with pytest.raises(SimplydataError, match="Did you mean"):
        t.compare_percent_to("pased", percent=50)


# ================================================================ summary measures
MEASURE_LIST = ["geometric mean", "skewness", "kurtosis", "coefficient of variation", "median absolute deviation",
                "trimmed average", "trimmed average 20", "percentile 90", "2.5th percentile"]
COLUMNS = ["geometric mean", "skewness", "kurtosis", "coefficient of variation", "median absolute deviation",
           "trimmed average 10 %", "trimmed average 20 %", "percentile 90", "percentile 2.5"]
# R (functions as in the module docstring):
#   gm <- function(x) exp(mean(log(x)))
#   skew2 <- function(x) {n; g1 = m3 / m2^1.5; g1 * sqrt(n * (n - 1)) / (n - 2)}           (e1071 type 2)
#   kurt2 <- function(x) {n; g2 = m4 / m2^2 - 3; ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))}
#   sd(x) / mean(x); mad(x); mean(x, trim = 0.1); mean(x, trim = 0.2); quantile(x, c(0.9, 0.025), type = 7)
R_HOURS = [4.386690692333822, 0.706492023293421, 0.140790721026474, 0.488959655607050, 2.446290000000000,
           4.802083333333333, 4.711111111111111, 8.5, 1.5475]
R_SLEEP = [7.8791217991321503, -0.0644226772951041, 0.0459614570191429, 0.1187296597039522, 0.8895600000000008,
           7.9503875968992253, 7.9391752577319590, 9.2, 5.9]
R_MARKS_BY_SCHOOL = {  # school: measures in the order of COLUMNS
    "School A": [43.1854545874519, 0.941866079342785, 0.493907964908428, 0.265267119732767, 10.3782,
                 44.0, 42.1111111111111, 62.6, 29.45],
    "School C": [33.4502181080901, -0.121786013125471, -1.267165860656977, 0.278432876052025, 14.8260,
                 34.9230769230769, 35.2222222222222, 46.6, 20.35],
    "School K": [52.0234794004621, 0.390735098893825, -1.201363620663964, 0.179560751535695, 11.8608,
                 52.4615384615385, 51.8888888888889, 65.2, 41.0],
}
R_SLEEP_BY_SCHOOL = {
    "School A": [7.73332173953221, 0.210565495412103, -0.352899787387668, 0.161687165414919, 1.40847,
                 7.79166666666667, 7.81, 9.14, 5.93],
    "School B": [8.24793516669716, -0.611126195663873, 0.695340443878767, 0.158702933017477, 0.74130,
                 8.41818181818182, 8.44444444444444, 9.78, 5.86],
}


def test_new_measures_match_r(school, engine, capsys):
    t = make(school, engine)
    got = rows(t.summary_of(["hours", "sleep"], show=MEASURE_LIST))
    assert list(got.columns) == ["column", *COLUMNS]
    for i, reference in enumerate((R_HOURS, R_SLEEP)):
        for name, value in zip(COLUMNS, reference):
            assert got[name][i] == pytest.approx(value, rel=1e-9), name
    out = capsys.readouterr().out
    assert "19 empty cells" in out and "trim = 0.2" in out


def test_new_measures_by_group_match_r(school, engine):
    t = make(school, engine)
    for column, reference in (("marks", R_MARKS_BY_SCHOOL), ("sleep", R_SLEEP_BY_SCHOOL)):
        got = rows(t.summary_of(column, for_each="school", show=MEASURE_LIST)).set_index("school")
        for school_name, values in reference.items():
            for name, value in zip(COLUMNS, values):
                assert got.loc[school_name, name] == pytest.approx(value, rel=1e-9), (column, school_name, name)


def test_measures_agree_with_scipy_and_small_r_example(engine):
    x = [2.0, 4.0, 4.0, 8.0, 16.0, 3.0, 7.0, 7.0, 1.0]
    t = make(pd.DataFrame({"x": x}), engine)
    got = rows(t.summary_of("x", show=["trimmed average", "trimmed average 25", "mad", "percentile 10",
                                       "percentile 33", "90th percentile", "gmean", "skew", "cv", "mode"]))
    # R: mean(x, trim = 0.1); mean(x, trim = 0.25); mad(x); quantile(x, c(0.1, 0.33, 0.9), type = 7)
    assert got["trimmed average 10 %"][0] == pytest.approx(5.77777777777778)
    assert got["trimmed average 25 %"][0] == pytest.approx(5.0)
    assert got["median absolute deviation"][0] == pytest.approx(4.4478)
    assert got["percentile 10"][0] == pytest.approx(1.8)
    assert got["percentile 33"][0] == pytest.approx(3.64)
    assert got["percentile 90"][0] == pytest.approx(9.6)
    values = np.array(x)
    assert got["geometric mean"][0] == pytest.approx(stats.gmean(values))
    assert got["skewness"][0] == pytest.approx(stats.skew(values, bias=False))
    assert got["coefficient of variation"][0] == pytest.approx(stats.variation(values, ddof=1))
    assert got["trimmed average 10 %"][0] == pytest.approx(stats.trim_mean(values, 0.1))
    assert got["most common"][0] == 4.0  # 4 and 7 both twice: the smallest


def test_measure_notes_for_empty_results(engine, capsys):
    data = pd.DataFrame({"g": ["a", "a", "b", "b", "b", "b"], "y": [0.0, 2.0, 1.0, 1.0, 3.0, 3.0]})
    got = rows(make(data, engine).summary_of("y", for_each="g", show=["geometric mean", "kurtosis", "most common"]))
    assert pd.isna(got["geometric mean"][0]) and got["geometric mean"][1] == pytest.approx(3 ** 0.5)
    assert pd.isna(got["kurtosis"][0]) and not pd.isna(got["kurtosis"][1])
    out = capsys.readouterr().out
    assert "the geometric mean needs numbers above 0, so it is empty for a" in out
    assert "the kurtosis needs at least 4 values, so it is empty for a" in out
    assert "most common in a (0, 2), b (1, 3)" in out


def test_measure_name_errors(marks, engine):
    t = make(marks, engine)
    with pytest.raises(SimplydataError, match='needs the percent, for example "percentile 90"'):
        t.summary_of("marks", show="percentile")
    with pytest.raises(SimplydataError, match="from 0 to 100"):
        t.summary_of("marks", show="percentile 120")
    with pytest.raises(SimplydataError, match="below 50"):
        t.summary_of("marks", show="trimmed average 60")
    with pytest.raises(SimplydataError, match='Did you mean show="skewness"'):
        t.summary_of("marks", show="skewnes")


# ================================================================ percentile_of and most_common_of
def test_percentile_of_matches_r(marks, engine, capsys):
    t = make(marks, engine)
    got = rows(t.percentile_of("marks", 90, for_each="class"))
    # R: tapply(m$marks, m$class, quantile, 0.9)
    assert got["percentile 90 of marks"].tolist() == pytest.approx([86.3, 78.2, 73.2])
    assert "The 90th percentile of \"marks\" for each \"class\"" in capsys.readouterr().out
    one = t.percentile_of("marks", 50)
    assert one == pytest.approx(float(np.median(marks["marks"])))
    cross = rows(t.percentile_of("marks", 25, for_each="class", across="subject"))
    a_maths = marks[(marks["class"] == "A") & (marks["subject"] == "Maths")]["marks"]
    assert cross.loc[cross["class"] == "A", "Maths"].item() == pytest.approx(np.percentile(a_maths, 25))
    with pytest.raises(SimplydataError, match="from 0 to 100"):
        t.percentile_of("marks", 101)
    with pytest.raises(SimplydataError, match="percent as a number"):
        t.percentile_of("marks")


def test_most_common_of(school, engine, capsys):
    t = make(school, engine)
    # R: table(s$days_absent) -- 2 turns up 44 times
    assert t.most_common_of("days_absent") == 2
    assert "44 of 180" in capsys.readouterr().out
    got = rows(t.most_common_of("days_absent", for_each="school"))
    # R: tapply(s$days_absent, s$school, function(x) names(which.max(table(x))))
    assert got["most common days_absent"].tolist() == [3, 2, 2, 4, 2, 4, 2, 4, 3, 1, 2, 4]
    assert set(got.columns) == {"school", "most common days_absent", "count", "percent", "values with that count"}
    text = rows(t.most_common_of("passed", for_each="school"))
    assert text["most common passed"][0] == "no"
    ties = make(pd.DataFrame({"tree": ["oak", "ash", "ash", "oak", None, "elm"]}), engine)
    assert ties.most_common_of("tree") == "ash"
    out = capsys.readouterr().out
    assert '"ash", "oak"' in out and "1 empty cell" in out
    with pytest.raises(SimplydataError, match="one column name"):
        t.most_common_of(["marks"])


# ================================================================ special functions in formulas
X = [0.3, 1.0, 2.5, 7.2]
# R: 2 * pnorm(x * sqrt(2)) - 1; 2 * pnorm(-x * sqrt(2)); gamma(x); lgamma(x); digamma(x); beta(x, 1.7);
#    besselJ(x, 0); besselJ(x, 1.5); tan(x)
R_SPECIAL = {
    "erf(x)": [0.3286267594591274, 0.842700792949715, 0.999593047982554994, 1.0],
    "erfc(x)": [0.6713732405408726, 0.157299207050285, 0.000406952017444959, 2.37779456632628e-24],
    "gamma(x)": [2.9915689876875908, 1.0, 1.329340388179137022, 1.05031781666268e+03],
    "lgamma(x)": [1.0957979948180756, 0.0, 0.284682870472919181, 6.95684807988834e+00],
    "digamma(x)": [-3.5025242222001323, -0.577215664901532, 0.703156640645243414, 1.90303214427017e+00],
    "beta(x, 1.7)": [2.7182554542156532, 0.588235294117647, 0.155722381342194127, 2.93023299688231e-02],
    "besselJ(x, 0)": [0.9776262465382962, 0.765197686557967, -0.048383776468198039, 2.95070691400958e-01],
    "besselJ(x, 1.5)": [0.0433098819183783, 0.240297839123427, 0.525080264664003038, -1.48117885037337e-01],
    "tan(x)": [0.3093362496096232, 1.557407724654902, -0.747022297238660204, 1.30462094005564e+00],
}


def test_special_functions_match_r(engine, capsys):
    t = make(pd.DataFrame({"x": X, "nu": [1.5] * 4}), engine)
    for i, formula in enumerate(R_SPECIAL):
        t.add_column(f"f{i}", formula)
    t.add_column("j", "besselj(x, nu)")
    t.add_column("trig", "sin(x) ** 2 + cos(x) ** 2 + exp(0) + pi - pi")
    got = t._frame.to_pandas()
    for i, (formula, reference) in enumerate(R_SPECIAL.items()):
        assert got[f"f{i}"].tolist() == pytest.approx(reference, rel=1e-12, abs=1e-300), formula
    assert got["j"].tolist() == pytest.approx(R_SPECIAL["besselJ(x, 1.5)"], rel=1e-12)
    assert got["trig"].tolist() == pytest.approx([2.0] * 4)
    assert len(t._history) == len(R_SPECIAL) + 2
    t.undo()
    assert "trig" not in t._frame.columns


def test_special_functions_with_summaries_groups_and_empty_cells(engine):
    data = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, None, 4.0], "y": [2.0, 3.0, 4.0, 5.0]})
    t = make(data, engine)
    t.add_column("share", "gamma(x) / total(gamma(y))", for_each="g")
    t.add_column("inside", "lgamma(total(y))")
    t.add_column("both", "beta(x, y)")
    t.add_column("bad", "gamma(x - x - 1)")  # gamma(-1) is not a number: an empty cell, as R gives NaN
    got = t._frame.to_pandas()
    assert got["share"].tolist()[:2] == pytest.approx([1 / (1 + 2), 1 / 3])
    assert pd.isna(got["share"][2]) and got["share"][3] == pytest.approx(6 / (6 + 24))
    assert got["inside"].tolist() == pytest.approx([math.lgamma(14.0)] * 4)
    assert got["both"][0] == pytest.approx(special.beta(1, 2)) and pd.isna(got["both"][2])
    assert t._frame["bad"].null_count() == 4


def test_special_function_errors(marks, engine):
    t = make(marks, engine)
    with pytest.raises(SimplydataError, match="it takes two things"):
        t.add_column("j", "besselJ(hours)")
    with pytest.raises(SimplydataError, match="was given 2 things inside the brackets; it takes one thing"):
        t.add_column("g", "gamma(hours, 2)")
    with pytest.raises(SimplydataError, match="called lgamma"):
        t.add_column("g", "gammaln(hours)")
    with pytest.raises(SimplydataError, match="needs brackets"):
        t.add_column("g", "gamma * hours")
    with pytest.raises(SimplydataError, match="Add it as a column first"):
        t.keep_rows("gamma(hours) > 2")
    assert len(t.keep_rows("hours > pi")._frame) == int((marks["hours"] > math.pi).sum())


def test_fit_formula_with_special_functions(engine, capsys):
    x = np.linspace(0.5, 10, 40)
    noise = np.sin(np.arange(40) * 1.7) * 0.01
    data = pd.DataFrame({"x": x, "y": 2 * special.jv(1, 1.3 * x) + noise})
    t = make(data, engine)
    got = row(t.fit_formula("y = a * besselJ(beta * x, 1)", start={"a": 1.5, "beta": 1.2}))
    assert got["a"] == pytest.approx(2.0, abs=0.02) and got["beta"] == pytest.approx(1.3, abs=0.005)
    assert "Values found: a, beta" in capsys.readouterr().out
    wave = make(pd.DataFrame({"x": x, "y": 3 * np.sin(x / math.pi) + noise}), engine)
    got = row(wave.fit_formula("y = gamma * sin(x / pi)", start={"gamma": 2}))
    assert got["gamma"] == pytest.approx(3.0, abs=0.01)
    with pytest.raises(SimplydataError, match="not allowed in a formula to fit"):
        wave.fit_formula("y = a * besel(x, 1)")
