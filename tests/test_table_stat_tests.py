"""Tests for the statistical test words.

Every number is checked against SciPy and against R 4.6.2 (reference values hard-coded below,
with the R call that made them), on pandas and Polars tables. The R data is the practice
table sdata.example("marks"), read in R with read.csv(".../data/marks.csv").
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats.contingency import association

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.stat_tests import _hedges_correction, _holm, _num, _welch_anova

ENGINES = ["pandas", "polars"]
REL = 1e-9  # SciPy and R agree to about this many digits for closed-form tests


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


def group(df, name, column="marks", by="class"):
    return df.loc[df[by] == name, column].to_numpy(float)


def row(result):
    """The one-row result table as a plain dict."""
    frame = result._frame.to_pandas()
    assert len(frame) == 1
    return frame.iloc[0].to_dict()


def unchanged(t, before):
    pd.testing.assert_frame_equal(t._frame.to_pandas(), before)
    assert t._history == []


def wide(marks):
    """One row per child, a column per subject (like R reshape(..., direction = "wide"))."""
    return marks.pivot(index="name", columns="subject", values="marks").reset_index()


# ---------------------------------------------------------------- compare_averages: 2 groups
def test_welch_t_test_matches_r_and_scipy(marks, engine, capsys):
    ab = marks[marks["class"].isin(["A", "B"])].reset_index(drop=True)
    t = make(ab, engine)
    before = t._frame.to_pandas().copy()
    got = row(t.compare_averages("marks", for_each="class"))
    # R: t.test(marks ~ class, data = subset(m, class %in% c("A", "B")))
    assert got["statistic"] == pytest.approx(1.583550500481361, rel=REL)
    assert got["degrees of freedom"] == pytest.approx(33.793073095025697, rel=REL)
    assert got["p-value"] == pytest.approx(0.122611215801451, rel=1e-8)
    assert got["difference 95 % CI low"] == pytest.approx(-2.143030164683023, rel=1e-8)
    assert got["difference 95 % CI high"] == pytest.approx(17.254141275794137, rel=1e-8)
    assert got["average in A"] == pytest.approx(71.833333333333329)
    assert got["average in B"] == pytest.approx(64.277777777777771)
    assert got["count in A"] == 18 and got["count in B"] == 18
    a, b = group(ab, "A"), group(ab, "B")
    scipy = stats.ttest_ind(a, b, equal_var=False)
    assert got["statistic"] == pytest.approx(scipy.statistic)
    assert got["p-value"] == pytest.approx(scipy.pvalue)
    assert got["test"] == "Welch's t-test"
    out = capsys.readouterr().out
    assert "Welch's t-test" in out and "does not assume both groups have the same spread" in out
    assert "could easily be chance (p = 0.123)" in out
    assert "significant" not in out and "prove" not in out
    unchanged(t, before)


def test_hedges_g_is_pooled_sd_difference_with_exact_correction(marks):
    ab = marks[marks["class"].isin(["A", "B"])]
    a, b = group(ab, "A"), group(ab, "B")
    got = row(make_table(ab).compare_averages("marks", for_each="class"))
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / pooled
    df = n1 + n2 - 2
    exact = math.gamma(df / 2) / (math.sqrt(df / 2) * math.gamma((df - 1) / 2))
    assert got["effect size"] == pytest.approx(d * exact)
    # the exact correction is close to Hedges' approximation 1 - 3 / (4 df - 1)
    assert _hedges_correction(df) == pytest.approx(1 - 3 / (4 * df - 1), abs=1e-4)
    assert "Hedges' g" in got["effect size name"]


def test_student_t_test_with_test_option_matches_r(marks, engine, capsys):
    ab = marks[marks["class"].isin(["A", "B"])].reset_index(drop=True)
    got = row(make(ab, engine).compare_averages("marks", for_each="class", test="student"))
    # R: t.test(marks ~ class, data = ab, var.equal = TRUE)
    assert got["statistic"] == pytest.approx(1.583550500481361, rel=REL)
    assert got["degrees of freedom"] == 34
    assert got["p-value"] == pytest.approx(0.122554793925191, rel=1e-8)
    assert got["difference 95 % CI low"] == pytest.approx(-2.140842688741232, rel=1e-8)
    assert "assumes both groups have the same spread" in capsys.readouterr().out


# ---------------------------------------------------------------- compare_averages: 3+ groups
def test_welch_anova_matches_r_oneway_test(marks, engine, capsys):
    got = row(make(marks, engine).compare_averages("marks", for_each="class"))
    # R: oneway.test(marks ~ class, data = m)          (var.equal = FALSE is the default)
    assert got["test"] == "Welch's one-way ANOVA"
    assert got["statistic"] == pytest.approx(3.9943188986448011, rel=REL)
    assert got["degrees of freedom 1"] == 2
    assert got["degrees of freedom 2"] == pytest.approx(33.9653259624436430, rel=REL)
    assert got["p-value"] == pytest.approx(0.0276731022463206, rel=1e-8)
    out = capsys.readouterr().out
    assert "Welch's one-way ANOVA (F = 3.99, df = 2 and 34.0)" in out
    assert 'table.compare_each_pair("marks", for_each="class")' in out
    assert "are unlikely to be chance (p = 0.028)" in out


def test_welch_anova_formula_on_unequal_groups_matches_r_and_scipy():
    x = [10.1, 11.4, 9.8, 12.0, 10.7]
    y = [13.2, 12.9, 14.8, 11.9]
    z = [9.0, 8.7, 10.2, 9.9, 8.1, 9.4]
    f, df1, df2, p = _welch_anova([np.array(v) for v in (x, y, z)])
    # R: oneway.test(v ~ g, data = data.frame(v = c(x, y, z), g = rep(c("x", "y", "z"), c(5, 4, 6))))
    assert f == pytest.approx(16.5370012871309484, rel=REL)
    assert df2 == pytest.approx(6.6502681032443043, rel=REL)
    assert p == pytest.approx(0.0026240684447844, rel=1e-8)
    # SciPy only has Welch's ANOVA from version 1.16 (not available for Python 3.10);
    # the R values above are the main check.
    try:
        scipy = stats.f_oneway(x, y, z, equal_var=False)
    except TypeError:
        return
    assert f == pytest.approx(scipy.statistic) and p == pytest.approx(scipy.pvalue)


def test_welch_anova_on_two_groups_equals_welch_t_squared(marks):
    ab = marks[marks["class"].isin(["A", "B"])]
    got = row(make_table(ab).compare_averages("marks", for_each="class", test="welch anova"))
    # R: oneway.test(marks ~ class, data = ab)
    assert got["statistic"] == pytest.approx(2.507632187574768, rel=REL)
    assert got["statistic"] == pytest.approx(1.583550500481361 ** 2, rel=REL)
    assert got["p-value"] == pytest.approx(0.122611215801451, rel=1e-8)


def test_classic_anova_with_test_option_matches_r(marks, engine):
    got = row(make(marks, engine).compare_averages("marks", for_each="class", test="anova"))
    # R: oneway.test(marks ~ class, data = m, var.equal = TRUE)
    assert got["statistic"] == pytest.approx(3.8900486306902953, rel=REL)
    assert got["degrees of freedom 2"] == 51
    assert got["p-value"] == pytest.approx(0.0267711379274287, rel=1e-8)


def test_omega_squared_from_sums_of_squares(marks):
    got = row(make_table(marks).compare_averages("marks", for_each="class"))
    groups = [group(marks, c) for c in "ABC"]
    allv = marks["marks"].to_numpy(float)
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msw = ssw / (len(allv) - 3)
    assert got["effect size"] == pytest.approx((ssb - 2 * msw) / (ssb + ssw + msw))
    # omega-squared is smaller than eta-squared (it removes the upward bias)
    assert got["effect size"] < ssb / (ssb + ssw)
    assert "omega-squared" in got["effect size name"]


# ---------------------------------------------------------------- compare_averages: ranks
def test_mann_whitney_matches_r(marks, engine, capsys):
    ab = marks[marks["class"].isin(["A", "B"])].reset_index(drop=True)
    got = row(make(ab, engine).compare_averages("marks", for_each="class", based_on="ranks"))
    # The marks have ties, so the normal approximation with continuity correction is used.
    # R: wilcox.test(marks ~ class, data = ab, exact = FALSE)
    assert got["statistic"] == 213.5
    assert got["p-value"] == pytest.approx(0.106305448775518, rel=1e-8)
    assert got["effect size"] == pytest.approx(2 * 213.5 / (18 * 18) - 1)
    assert got["median in A"] == np.median(group(ab, "A"))
    out = capsys.readouterr().out
    assert "Mann-Whitney U test (U = 213.5, normal approximation with continuity correction)" in out


def test_mann_whitney_exact_without_ties_matches_r():
    data = pd.DataFrame({"g": ["x"] * 5 + ["y"] * 6,
                         "v": [1.1, 2.3, 3.8, 4.0, 5.9, 2.0, 6.5, 7.1, 8.4, 9.9, 10.3]})
    got = row(make_table(data).compare_averages("v", for_each="g", based_on="ranks"))
    # R: wilcox.test(c(1.1, 2.3, 3.8, 4.0, 5.9), c(2.0, 6.5, 7.1, 8.4, 9.9, 10.3))
    assert got["statistic"] == 4
    assert got["p-value"] == pytest.approx(0.051948051948052, rel=1e-10)


def test_kruskal_wallis_matches_r(marks, engine, capsys):
    got = row(make(marks, engine).compare_averages("marks", for_each="class", based_on="ranks"))
    # R: kruskal.test(marks ~ class, data = m)
    assert got["test"] == "Kruskal-Wallis test"
    assert got["statistic"] == pytest.approx(7.5376653944020422, rel=REL)
    assert got["degrees of freedom"] == 2
    assert got["p-value"] == pytest.approx(0.0230789877366583, rel=1e-8)
    assert got["effect size"] == pytest.approx(got["statistic"] / (54 - 1))
    assert "based_on=\"ranks\")" in capsys.readouterr().out


def test_kruskal_wallis_small_groups_match_r():
    data = pd.DataFrame({"g": list("xxxxxyyyyzzzzzz"),
                         "v": [10.1, 11.4, 9.8, 12.0, 10.7, 13.2, 12.9, 14.8, 11.9,
                               9.0, 8.7, 10.2, 9.9, 8.1, 9.4]})
    got = row(make_table(data).compare_averages("v", for_each="g", test="kruskal-wallis"))
    # R: kruskal.test(list(x, y, z))
    assert got["statistic"] == pytest.approx(10.4025, rel=REL)
    assert got["p-value"] == pytest.approx(0.00550967302325559, rel=1e-8)


# ---------------------------------------------------------------- compare_each_pair
def test_holm_matches_r_p_adjust():
    # R: p.adjust(c(0.01, 0.04, 0.03, 0.2, 0.005), "holm")
    assert _holm([0.01, 0.04, 0.03, 0.2, 0.005]) == pytest.approx([0.040, 0.090, 0.090, 0.200, 0.025])


def test_each_pair_welch_holm_matches_r(marks, engine, capsys):
    result = make(marks, engine).compare_each_pair("marks", for_each="class")
    frame = result._frame.to_pandas()
    assert list(zip(frame["group 1"], frame["group 2"])) == [("A", "B"), ("A", "C"), ("B", "C")]
    # R: pairwise.t.test(m$marks, m$class, pool.sd = FALSE, p.adjust.method = "none")
    assert frame["p-value"].tolist() == pytest.approx([0.12261121580145126, 0.00755147193154552,
                                                       0.250021255346916], rel=1e-8)
    # R: pairwise.t.test(m$marks, m$class, pool.sd = FALSE, p.adjust.method = "holm")
    assert frame["p-value (Holm)"].tolist() == pytest.approx([0.2452224316029025, 0.0226544157946366,
                                                              0.250021255346916], rel=1e-8)
    out = capsys.readouterr().out
    assert "Holm's method" in out and "A vs C" in out and "Holm-adjusted p = 0.023" in out


def test_each_pair_on_ranks_matches_r(marks):
    frame = make_table(marks).compare_each_pair("marks", for_each="class", based_on="ranks")._frame.to_pandas()
    # R: pairwise.wilcox.test(m$marks, m$class, p.adjust.method = "holm", exact = FALSE)
    assert frame["p-value (Holm)"].tolist() == pytest.approx([0.2126108975510370, 0.0246143303939027,
                                                              0.22893038037713], rel=1e-8)


def test_each_pair_unequal_groups_matches_r():
    data = pd.DataFrame({"g": list("xxxxxyyyyzzzzzz"),
                         "v": [10.1, 11.4, 9.8, 12.0, 10.7, 13.2, 12.9, 14.8, 11.9,
                               9.0, 8.7, 10.2, 9.9, 8.1, 9.4]})
    frame = make_table(data).compare_each_pair("v", for_each="g")._frame.to_pandas()
    # R: pairwise.t.test(v, g, pool.sd = FALSE, p.adjust.method = "holm")
    assert frame["p-value (Holm)"].tolist() == pytest.approx([0.0306973234457488, 0.0306973234457488,
                                                              0.00764108783563924], rel=1e-8)


# ---------------------------------------------------------------- compare_before_after
def test_paired_t_test_matches_r(marks, engine, capsys):
    w = wide(marks)
    t = make(w, engine)
    before = t._frame.to_pandas().copy()
    got = row(t.compare_before_after("Maths", "Science"))
    # R: t.test(w$marks.Science, w$marks.Maths, paired = TRUE)
    assert got["statistic"] == pytest.approx(1.455821935271032, rel=REL)
    assert got["degrees of freedom"] == 17
    assert got["p-value"] == pytest.approx(0.163661440714291, rel=1e-8)
    assert got["average change 95 % CI low"] == pytest.approx(-3.094670734448396, rel=1e-8)
    assert got["average change 95 % CI high"] == pytest.approx(16.872448512226175, rel=1e-8)
    assert got["average change"] == pytest.approx(6.888888888888889)
    change = (w["Science"] - w["Maths"]).to_numpy(float)
    assert got["effect size"] == pytest.approx(change.mean() / change.std(ddof=1))
    scipy = stats.ttest_rel(w["Science"], w["Maths"])
    assert got["p-value"] == pytest.approx(scipy.pvalue)
    assert "paired t-test" in capsys.readouterr().out
    unchanged(t, before)


def test_signed_rank_paired_matches_r(marks, engine):
    got = row(make(wide(marks), engine).compare_before_after("Maths", "Science", based_on="ranks"))
    # The changes have ties, so the normal approximation with continuity correction is used.
    # R: wilcox.test(w$marks.Science, w$marks.Maths, paired = TRUE, exact = FALSE)
    assert got["statistic"] == 121
    assert got["p-value"] == pytest.approx(0.12735292104833, rel=1e-8)
    assert got["effect size"] == pytest.approx(2 * 121 / (18 * 19 / 2) - 1)


def test_before_after_skips_rows_with_an_empty_cell(engine, capsys):
    data = pd.DataFrame({"before": [5.0, 6.0, np.nan, 8.0, 7.5, 6.2],
                         "after": [6.1, 6.9, 7.0, np.nan, 8.8, 7.0]})
    got = row(make(data, engine).compare_before_after("before", "after"))
    kept = data.dropna()
    assert got["rows"] == 4
    assert got["p-value"] == pytest.approx(stats.ttest_rel(kept["after"], kept["before"]).pvalue)
    assert 'Note: 2 rows with an empty cell in "before" or "after" were skipped.' in capsys.readouterr().out


def test_signed_rank_leaves_out_zero_changes_like_r(capsys):
    data = pd.DataFrame({"before": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "after": [1.0, 3.5, 2.5, 6.0, 8.0, 9.5]})
    got = row(make_table(data).compare_before_after("before", "after", based_on="ranks"))
    d = (data["after"] - data["before"]).to_numpy()
    scipy = stats.wilcoxon(d, zero_method="wilcox", correction=True, method="approx")
    assert got["p-value"] == pytest.approx(scipy.pvalue)
    assert "1 rows with no change were left out" in capsys.readouterr().out


# ---------------------------------------------------------------- compare_average_to
def test_one_sample_t_test_matches_r(marks, engine, capsys):
    got = row(make(marks, engine).compare_average_to("marks", 60))
    # R: t.test(m$marks, mu = 60)
    assert got["statistic"] == pytest.approx(2.3997521795531536, rel=REL)
    assert got["degrees of freedom"] == 53
    assert got["p-value"] == pytest.approx(0.0199527762801669, rel=1e-8)
    assert got["average 95 % CI low"] == pytest.approx(60.8057285596217696, rel=1e-10)
    assert got["average 95 % CI high"] == pytest.approx(69.0090862551930400, rel=1e-10)
    v = marks["marks"].to_numpy(float)
    assert got["effect size"] == pytest.approx((v.mean() - 60) / v.std(ddof=1))
    out = capsys.readouterr().out
    assert "4.91 higher than 60 " in out and "is unlikely to be chance (p = 0.020)" in out


def test_one_sample_signed_rank_matches_r(marks, engine):
    got = row(make(marks, engine).compare_average_to("marks", 60, based_on="ranks"))
    # R: wilcox.test(m$marks, mu = 60)        (ties, so R uses the normal approximation too)
    assert got["statistic"] == 1009.5
    assert got["p-value"] == pytest.approx(2.17090135520022e-02, rel=1e-8)


def test_one_sample_signed_rank_exact_matches_r():
    data = pd.DataFrame({"x": [1.5, -0.3, 2.2, 0.8, 3.1, -1.4, 2.7]})
    got = row(make_table(data).compare_average_to("x", 0, based_on="ranks"))
    # R: wilcox.test(c(1.5, -0.3, 2.2, 0.8, 3.1, -1.4, 2.7), mu = 0)
    assert got["statistic"] == 24
    assert got["p-value"] == pytest.approx(0.109375, rel=1e-12)


# ---------------------------------------------------------------- compare_counts
def with_result(marks):
    out = marks.copy()
    out["result"] = np.where(out["marks"] >= 60, "60 or more", "below 60")
    return out


def test_chi_square_matches_r(marks, engine, capsys):
    t = make(with_result(marks), engine)
    got = row(t.compare_counts("result", for_each="class"))
    # R: chisq.test(table(m$result, m$class))
    assert got["statistic"] == pytest.approx(4.207792207792208, rel=REL)
    assert got["degrees of freedom"] == 2
    assert got["p-value"] == pytest.approx(0.121980253501261, rel=1e-8)
    observed = np.array([[14, 11, 8], [4, 7, 10]])
    assert got["effect size"] == pytest.approx(association(observed, method="cramer"))
    assert got["rows"] == 54
    out = capsys.readouterr().out
    assert "Yates" not in out and "Note:" not in out


def test_chi_square_two_by_two_uses_yates_like_r(marks, engine, capsys):
    data = with_result(marks)
    data["class A"] = np.where(data["class"] == "A", "yes", "no")
    got = row(make(data, engine).compare_counts("result", for_each="class A"))
    # R: chisq.test(table(m$result, m$class == "A"))      (correct = TRUE is the default)
    assert got["statistic"] == pytest.approx(2.191558441558442, rel=REL)
    assert got["p-value"] == pytest.approx(0.138768846789353, rel=1e-8)
    assert "Yates' continuity correction" in capsys.readouterr().out
    # Cramer's V uses the chi-square without the correction
    observed = np.array([[19, 14], [17, 4]])
    assert got["effect size"] == pytest.approx(association(observed, method="cramer", correction=False))
    fisher = row(make(data, engine).compare_counts("result", for_each="class A", test="fisher"))
    # R: fisher.test(table(m$result, m$class == "A"))
    assert fisher["p-value"] == pytest.approx(0.137562740161948, rel=1e-8)
    assert fisher["test"] == "Fisher's exact test"


def test_small_counts_note_suggests_fisher_and_never_switches(engine, capsys):
    data = pd.DataFrame({"a": ["x"] * 4 + ["y"] * 5, "b": ["p", "p", "p", "q", "p", "q", "q", "q", "q"]})
    got = row(make(data, engine).compare_counts("a", for_each="b"))
    # R: chisq.test(matrix(c(3, 1, 1, 4), 2))
    assert got["test"] == "chi-square test of independence"
    assert got["statistic"] == pytest.approx(0.950625, rel=REL)
    assert got["p-value"] == pytest.approx(0.329560259960621, rel=1e-8)
    out = capsys.readouterr().out
    assert "expected counts are below 5" in out and 'test="fisher"' in out
    fisher = row(make(data, engine).compare_counts("a", for_each="b", test="fisher"))
    # R: fisher.test(matrix(c(3, 1, 1, 4), 2))
    assert fisher["p-value"] == pytest.approx(0.206349206349207, rel=1e-8)
    assert "below 5" not in capsys.readouterr().out


def test_fisher_needs_two_by_two(marks):
    with pytest.raises(SimplydataError, match="2 x 2"):
        make_table(with_result(marks)).compare_counts("result", for_each="class", test="fisher")


# ---------------------------------------------------------------- fit_straight_line
def test_straight_line_matches_r_lm(marks, engine, capsys):
    got = row(make(marks, engine).fit_straight_line(x="hours", y="marks"))
    # R: fit <- lm(marks ~ hours, data = m); summary(fit)$coefficients; confint(fit)
    assert got["slope"] == pytest.approx(5.20319382959493, rel=1e-12)
    assert got["intercept"] == pytest.approx(37.16667586036332, rel=1e-12)
    assert got["slope standard error"] == pytest.approx(0.482463745779896, rel=1e-10)
    assert got["intercept standard error"] == pytest.approx(2.816614795937894, rel=1e-10)
    assert got["statistic"] == pytest.approx(10.7846317471669, rel=1e-10)
    assert got["p-value"] == pytest.approx(7.17654229238716e-15, rel=1e-6)
    assert got["slope 95 % CI low"] == pytest.approx(4.23505949556761, rel=1e-10)
    assert got["slope 95 % CI high"] == pytest.approx(6.17132816362226, rel=1e-10)
    assert got["intercept 95 % CI low"] == pytest.approx(31.51472477900506, rel=1e-10)
    assert got["R squared"] == pytest.approx(0.691043129867498, rel=1e-10)
    assert got["rows"] == 54
    out = capsys.readouterr().out
    assert '"marks" goes up by 5.20 for each 1 more "hours" (95 % CI 4.24 to 6.17)' in out
    assert "very unlikely to be chance (p < 0.0001)" in out and "about 69 %" in out


# ---------------------------------------------------------------- fit_formula
def growth_applia(engine):
    t = make(sdata.example("growth"), engine)
    t.keep_rows("country", equal_to="Applia")
    return t


def test_fit_formula_matches_r_nls(engine, capsys):
    t = growth_applia(engine)
    got = row(t.fit_formula("population = a * exp(b * (year - 2000))", start={"a": 10, "b": 0.01}))
    # R: nls(population ~ a * exp(b * (year - 2000)), data = subset(g, country == "Applia"),
    #        start = list(a = 10, b = 0.01))
    assert got["a"] == pytest.approx(11.9771183381250896, rel=1e-6)
    assert got["b"] == pytest.approx(0.0297361849443211, rel=1e-6)
    assert got["standard error of a"] == pytest.approx(0.078701859505909202, rel=1e-4)
    assert got["standard error of b"] == pytest.approx(0.000391979899616625, rel=1e-4)
    assert got["rows"] == 13
    out = capsys.readouterr().out
    assert "Values found: a, b. Data: year." in out and "rough guide" in out


def test_fit_formula_default_start_works_on_the_example(engine):
    got = row(growth_applia(engine).fit_formula("population = a * exp(b * (year - 2000))"))
    assert got["a"] == pytest.approx(11.9771183381250896, rel=1e-6)


def test_fit_formula_straight_line_equals_fit_straight_line(marks, engine, capsys):
    t = make(marks, engine)
    line = row(t.fit_straight_line(x="hours", y="marks"))
    curve = row(t.fit_formula("marks = a * hours + b"))
    assert curve["a"] == pytest.approx(line["slope"], rel=1e-6)
    assert curve["b"] == pytest.approx(line["intercept"], rel=1e-6)
    assert curve["standard error of a"] == pytest.approx(line["slope standard error"], rel=1e-5)
    assert curve["R squared"] == pytest.approx(line["R squared"], rel=1e-8)


def test_fit_formula_matches_scipy_curve_fit(marks):
    from scipy.optimize import curve_fit

    got = row(make_table(marks).fit_formula("marks = top - k / hours", start={"top": 80, "k": 10}))
    x, y = marks["hours"].to_numpy(float), marks["marks"].to_numpy(float)
    found, cov = curve_fit(lambda x, top, k: top - k / x, x, y, p0=[80, 10])
    assert got["top"] == pytest.approx(found[0], rel=1e-6)
    assert got["standard error of k"] == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-4)


@pytest.mark.parametrize("formula, message", [
    ("a * hours + b", "one = sign"),
    ("marks == a * hours", "one = sign"),
    ("marks = a * hourz + b", 'Did you mean the column "hours"'),
    ("marks = a * sine(hours)", "not allowed in a formula to fit"),
    ("marks = a + b", "no column in it"),
    ("marks = 2 * hours", "no value to find"),
    ("marks = a * hours ^ 2", "use \\*\\*"),
    ("marks = a * hours // 2", "not allowed in a formula to fit"),
    ("marks = __import__('os')", "not allowed"),
    ("nothing = a * hours", 'Column "nothing" not found'),
])
def test_fit_formula_friendly_errors(marks, formula, message):
    with pytest.raises(SimplydataError, match=message):
        make_table(marks).fit_formula(formula)


def test_fit_formula_failed_fit_is_explained(marks):
    t = make_table(sdata.example("growth"))
    with pytest.raises(SimplydataError, match="impossible numbers"):
        t.fit_formula("population = a * exp(b * year)")
    with pytest.raises(SimplydataError, match="not in the formula"):
        make_table(marks).fit_formula("marks = a * hours", start={"c": 1})


# ---------------------------------------------------------------- assumption notes
def test_skewed_small_group_gets_a_note_suggesting_ranks(capsys):
    data = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10,
                         "v": [1, 1, 1, 1, 1, 1, 1, 2, 2, 40] + list(range(10, 20))})
    p = stats.shapiro(data["v"][:10]).pvalue
    assert p < 0.05
    result = row(make_table(data).compare_averages("v", for_each="g"))
    out = capsys.readouterr().out
    assert f'the values in group "a" don\'t look bell-shaped (Shapiro-Wilk' in out
    assert 'a safer test here: based_on="ranks"' in out
    assert result["test"] == "Welch's t-test"  # never switched


def test_skewed_large_group_note_explains_central_limit_theorem(capsys):
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200,
                         "v": np.concatenate([rng.exponential(1, 200), rng.exponential(1.3, 200)])})
    make_table(data).compare_averages("v", for_each="g")
    out = capsys.readouterr().out
    assert "central limit theorem" in out and "based_on=" not in out


def test_tiny_group_note_and_error(capsys):
    data = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "v": [1.0, 2.0, 3.0, 4.5, 5.0]})
    make_table(data).compare_averages("v", for_each="g")
    assert 'group "a" has only 2 values: too few to check' in capsys.readouterr().out
    single = pd.DataFrame({"g": ["a", "b", "b"], "v": [1.0, 3.0, 4.0]})
    with pytest.raises(SimplydataError, match="at least 2 values in each group"):
        make_table(single).compare_averages("v", for_each="g")


def test_ranks_give_no_bell_shape_note(capsys):
    data = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10,
                         "v": [1, 1, 1, 1, 1, 1, 1, 2, 2, 40] + list(range(10, 20))})
    make_table(data).compare_averages("v", for_each="g", based_on="ranks")
    assert "bell-shaped (Shapiro" not in capsys.readouterr().out


# ---------------------------------------------------------------- empty cells, engines, saving
def test_empty_cells_skipped_with_a_note(engine, capsys):
    data = pd.DataFrame({"g": ["a", "a", "a", None, "b", "b", "b"],
                         "v": [1.0, np.nan, 3.0, 4.0, 5.0, 6.5, 7.0]})
    got = row(make(data, engine).compare_averages("v", for_each="g"))
    assert got["count in a"] == 2 and got["count in b"] == 3
    assert got["p-value"] == pytest.approx(stats.ttest_ind([1.0, 3.0], [5.0, 6.5, 7.0], equal_var=False).pvalue)
    assert 'Note: 2 rows with an empty cell in "v" or "g" were skipped.' in capsys.readouterr().out


def test_result_is_a_table_on_the_same_engine_and_can_be_saved(marks, engine, tmp_path):
    result = make(marks, engine).compare_averages("marks", for_each="class")
    assert getattr(result, "_is_simplydata_table", False)
    assert result._engine == ("Polars" if engine == "polars" else "pandas")
    result.save(tmp_path / "test.csv")
    back = pd.read_csv(tmp_path / "test.csv")
    assert back["p-value"].iloc[0] == pytest.approx(0.0276731022463206, rel=1e-8)


def test_numbers_are_written_plainly():
    assert _num(0.0967) == "0.0967"
    assert _num(71.8333) == "71.8"
    assert _num(1234.5) == "1234"
    assert _num(-2.143) == "-2.14"
    assert _num(0.000392) == "0.000392"
    assert _num(0) == "0"


# ---------------------------------------------------------------- friendly errors
@pytest.mark.parametrize("call, message", [
    (lambda t: t.compare_averages("marks", for_each="class", based_on="median"), 'based_on="median"'),
    (lambda t: t.compare_averages("marks", for_each="class", test="welsh"), 'Did you mean test="welch"'),
    (lambda t: t.compare_averages("marks", for_each="class", test="welch"), "compares exactly 2 groups"),
    (lambda t: t.compare_averages("marks", for_each="class", based_on="ranks", test="welch"), "Use one of them"),
    (lambda t: t.compare_averages("name", for_each="class"), "needs a column of numbers"),
    (lambda t: t.compare_averages("marks", for_each="marks"), "cannot be both"),
    (lambda t: t.compare_averages("marks", for_each="klass"), 'Did you mean "class"'),
    (lambda t: t.compare_averages("marks"), "for_each= as a column name"),
    (lambda t: t.compare_averages("marks", by="class"), "has no option called by="),
    (lambda t: t.compare_average_to("marks", "60"), "needs a number"),
    (lambda t: t.compare_before_after("marks", "marks"), "two different columns"),
    (lambda t: t.compare_counts("subject", for_each="class", test="g-test"), 'test="g-test"'),
    (lambda t: t.fit_straight_line(x="hours", y="hours"), "two different columns"),
])
def test_friendly_errors(marks, call, message):
    with pytest.raises(SimplydataError, match=message):
        call(make_table(marks))


def test_one_group_only_is_explained(marks):
    t = make_table(marks)
    t.keep_rows("class", equal_to="A")
    with pytest.raises(SimplydataError, match='at least 2 groups in "class"; it found only "A"'):
        t.compare_averages("marks", for_each="class")


def test_all_values_the_same_is_explained():
    data = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 1.0, 2.0, 2.0]})
    with pytest.raises(SimplydataError, match="Every value is the same"):
        make_table(data).compare_averages("v", for_each="g")
