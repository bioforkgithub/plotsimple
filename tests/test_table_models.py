"""Tests for the model words: explain() and add_predicted_column().

Reference values come from R 4.6.2 (lme4 and mice for mixed models and multiple imputation),
with the R call that made them written above each block, on pandas and Polars tables. The R
data is read with read.csv(".../data/school.csv") and read.csv(".../data/marks.csv"), with
breakfast and passed made factors (levels "no", "yes").
"""

import math

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.stat_tests import _num as sdata_num

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
def school():
    return sdata.example("school")


@pytest.fixture
def class_marks():
    return sdata.example("marks")


def rows(result):
    """The result table as {term: row dict}."""
    frame = result._frame.to_pandas()
    return {r["term"]: r for r in frame.to_dict("records")}


def check(got, expected, rel=1e-6):
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=rel), key


# ---------------------------------------------------------------- linear model
def test_linear_model_matches_r_lm(school, engine, capsys):
    t = make(school, engine)
    before = t._frame.to_pandas().copy()
    got = rows(t.explain("marks", using=["hours", "breakfast", "sleep"]))
    # R: f1 <- lm(marks ~ hours + breakfast + sleep, data = s); summary(f1); confint(f1)
    check(got["(intercept)"], {"effect": 14.73606572977855, "standard error": 5.857203006036914,
                               "statistic": 2.51588782471606, "p-value": 1.28782439691068e-02,
                               "95 % CI low": 3.166981826314831, "95 % CI high": 26.30514963324227})
    check(got["hours"], {"effect": 2.68933256410568, "standard error": 0.274280835897071,
                         "statistic": 9.80503269690672, "p-value": 5.29245785465905e-18,
                         "95 % CI low": 2.147576029402201, "95 % CI high": 3.23108909880917})
    check(got["breakfast (yes)"], {"effect": 3.23928543083062, "standard error": 1.390551008712815,
                                   "p-value": 2.11065939337892e-02, "95 % CI low": 0.492684147715850,
                                   "95 % CI high": 5.98588671394539})
    check(got["sleep"], {"effect": 2.28935369184562, "p-value": 1.66303770320158e-03,
                         "95 % CI low": 0.876264837106532})
    assert got["hours"]["degrees of freedom"] == 157 and got["hours"]["statistic name"] == "t"
    out = capsys.readouterr().out
    # R: summary(f1)$r.squared, $adj.r.squared, $fstatistic  0.4296 0.4187 39.41 on 3 and 157 df
    assert "R squared = 0.430 (adjusted 0.419)" in out
    assert "F = 39.4, df = 3 and 157" in out and "very unlikely to be chance (p < 0.0001)" in out
    assert "Note: 19 rows with an empty cell" in out and "(161 rows)" in out
    assert "Method: linear model by least squares, as R lm(marks ~ hours + breakfast + sleep)." in out
    assert 'breakfast "yes" against "no": "marks" is 3.24 higher (95 % CI 0.493 to 5.99)' in out
    assert "bell-shaped" not in out  # R: shapiro.test(residuals(f1))$p.value 0.433
    pd.testing.assert_frame_equal(t._frame.to_pandas(), before)
    assert t._history == []


def test_model_values_match_r_summary(school):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        make_table(school).explain("marks", using=["hours", "breakfast", "sleep"])
    # the printed values come from these, checked here to more digits
    from simplydata.table.models import _least_squares

    s = school.dropna(subset=["sleep"])
    X = np.column_stack([np.ones(len(s)), s.hours, (s.breakfast == "yes"), s.sleep]).astype(float)
    y = s.marks.to_numpy(float)
    rss = _least_squares(X, y)[3]
    r2 = 1 - rss / ((y - y.mean()) ** 2).sum()
    assert r2 == pytest.approx(0.429555885024079, rel=1e-10)
    assert 1 - (1 - r2) * 160 / 157 == pytest.approx(0.418655679005431, rel=1e-10)
    assert (r2 / 3) / ((1 - r2) / 157) == pytest.approx(39.408051947751410, rel=1e-10)


def test_interaction_and_whole_column_test_match_r(school, engine, capsys):
    got = rows(make(school, engine).explain("marks", using=["hours * breakfast", "school"]))
    # R: f2 <- lm(marks ~ hours * breakfast + school, data = s); summary(f2); drop1(f2, test = "F")
    check(got["(intercept)"], {"effect": 25.109385007796540, "standard error": 2.830752226235134})
    check(got["hours"], {"effect": 2.990964011806588, "p-value": 4.53202163587603e-12})
    check(got["breakfast (yes)"], {"effect": 4.248909480074885, "p-value": 1.04103649718300e-01})
    check(got["hours : breakfast (yes)"], {"effect": -0.259859431998816, "standard error": 0.470053024016599,
                                           "statistic": -0.552830039850225, "p-value": 5.81127983109754e-01})
    check(got["school (whole column)"], {"statistic": 10.46741, "p-value": 1.9926e-14}, rel=1e-4)
    assert got["school (whole column)"]["degrees of freedom"] == 11
    assert list(got)[-2] == "hours : breakfast (yes)"  # main effects first, then the interaction, as R
    out = capsys.readouterr().out
    assert "school, all groups together: F test (F = 10.5, df = 11 and 165)" in out
    assert "(interaction)" in out and "as R lm(marks ~ hours * breakfast + school)" in out


def test_interaction_only_uses_r_coding(school):
    got = rows(make_table(school).explain("marks", using="hours : breakfast"))
    # R: summary(lm(marks ~ hours:breakfast, data = s))  (full coding of breakfast, as model.matrix())
    check(got["hours : breakfast (no)"], {"effect": 2.51249216124761, "standard error": 0.321449628693693})
    check(got["hours : breakfast (yes)"], {"effect": 2.94112751457657, "statistic": 10.69924515642394})
    check(got["(intercept)"], {"effect": 34.53434595366022})


def test_group_column_with_three_groups_matches_drop1(class_marks, engine, capsys):
    got = rows(make(class_marks, engine).explain("marks", using=["hours", "subject"]))
    # R: l2 <- lm(marks ~ hours + subject, data = mk); summary(l2); drop1(l2, test = "F")
    check(got["subject (Maths)"], {"effect": -1.26295852168258, "standard error": 2.809245971355325,
                                   "p-value": 6.54961956171916e-01})
    check(got["subject (Science)"], {"effect": 2.86775332521903, "p-value": 3.16966967965089e-01})
    check(got["subject (whole column)"], {"statistic": 1.12392, "p-value": 0.33307}, rel=1e-4)
    assert "hours (whole column)" not in got  # one column: its t-test is the same test
    assert 'subject "Maths" against "English"' in capsys.readouterr().out  # the baseline is the first group


# ---------------------------------------------------------------- logistic regression
def test_logistic_regression_matches_r_glm(school, engine, capsys):
    got = rows(make(school, engine).explain("passed", using=["hours", "breakfast"], outcome="yes or no"))
    # R: g1 <- glm(passed ~ hours + breakfast, family = binomial, data = s); summary(g1); exp(confint.default(g1))
    check(got["(intercept)"], {"log odds ratio": -3.092800457622370, "standard error": 0.552342416354540,
                               "statistic": -5.59942594674306, "p-value": 2.15062777983353e-08,
                               "95 % CI low": 0.0153694777458107, "95 % CI high": 0.13395796605019})
    check(got["hours"], {"log odds ratio": 0.509493433436877, "standard error": 0.089727444785042,
                         "p-value": 1.36092408724107e-08, "odds ratio": math.exp(0.509493433436877),
                         "95 % CI low": 1.3960277372504717, "95 % CI high": 1.98447817141263})
    check(got["breakfast (yes)"], {"log odds ratio": 0.415554886072377, "p-value": 2.49568930524936e-01,
                                   "95 % CI low": 0.7469008520298389, "95 % CI high": 3.07385539596500})
    assert got["hours"]["statistic name"] == "z"
    out = capsys.readouterr().out
    assert 'counted as yes: "passed" = "yes"' in out
    # R: deviance(g1), g1$null.deviance, AIC(g1): 198.963 246.323 204.963
    assert "Residual deviance 199 on 177 df (with no columns: 246 on 179 df)" in out
    assert "likelihood-ratio chi-square = 47.4, df = 2" in out and "AIC = 205." in out
    assert "multiplies the odds of yes by 1.66 (95 % CI 1.40 to 1.98)" in out


def test_yes_no_deviance_and_aic_to_more_digits(school):
    from simplydata.table.models import _irls, _log_likelihood

    X = np.column_stack([np.ones(len(school)), school.hours, school.breakfast == "yes"]).astype(float)
    y = (school.passed == "yes").to_numpy(float)
    beta, cov, mu, deviance, converged = _irls(X, y, "binomial")
    assert converged
    assert deviance == pytest.approx(198.963040749707, rel=1e-10)
    assert -2 * _log_likelihood("binomial", y, mu) + 6 == pytest.approx(204.963040749707, rel=1e-10)


def test_other_two_values_count_the_second_as_yes(school, capsys):
    school = school.assign(result=np.where(school.passed == "yes", "pass", "fail"))
    got = rows(make_table(school).explain("result", using="hours", outcome="yes or no"))
    assert '"result" = "pass" (the second value in sorted order, as R; "fail" is no)' in capsys.readouterr().out
    assert got["hours"]["odds ratio"] > 1


# ---------------------------------------------------------------- counts
def test_poisson_regression_matches_r_glm(school, engine, capsys):
    got = rows(make(school, engine).explain("days_absent", using=["hours", "school"], outcome="counts"))
    # R: p1 <- glm(days_absent ~ hours + school, family = poisson, data = s); exp(confint.default(p1))
    check(got["(intercept)"], {"log rate ratio": 1.977333679777313, "standard error": 0.1740583639128035,
                               "95 % CI low": 5.135542859400666, "95 % CI high": 10.160237361749651})
    check(got["hours"], {"log rate ratio": -0.157048456917664, "statistic": -7.591325804954122,
                         "p-value": 3.16648649405036e-14, "95 % CI high": 0.890029249548635})
    check(got["school (School B)"], {"log rate ratio": -0.049941978846956, "p-value": 8.09846041642287e-01,
                                     "95 % CI low": 0.633348022042962})
    # R: drop1(p1, test = "Chisq")  school: LRT 6.06666875815 on 11 df, p 0.86888
    check(got["school (whole column)"], {"statistic": 6.06666875815, "p-value": 0.86888}, rel=1e-5)
    out = capsys.readouterr().out
    # R: deviance(p1) 187.406, null 255.330, AIC 707.449
    assert "Residual deviance 187 on 167 df (with no columns: 255 on 179 df)" in out and "AIC = 707." in out
    assert "vary more than a Poisson" not in out  # R: deviance / df.residual = 1.12


def test_quasi_poisson_matches_r(school, engine, capsys):
    got = rows(make(school, engine).explain("days_absent", using=["hours", "school"],
                                            outcome="counts (overdispersed)"))
    # R: q1 <- glm(days_absent ~ hours + school, family = quasipoisson, data = s); exp(confint.default(q1))
    check(got["hours"], {"log rate ratio": -0.157048456917664, "standard error": 0.0212550932366633,
                         "statistic": -7.388744672583623, "p-value": 6.75163198035203e-12,
                         "95 % CI low": 0.819789503972954, "95 % CI high": 0.89101925634466})
    check(got["(intercept)"], {"standard error": 0.1788306144130610, "95 % CI low": 5.087731819385348})
    # R: drop1(q1, test = "F")  school: F 0.49146 on 11 df, p 0.90672
    check(got["school (whole column)"], {"statistic": 0.49146, "p-value": 0.90672}, rel=1e-4)
    out = capsys.readouterr().out
    assert "quasi-Poisson" in out and "dispersion 1.06" in out  # R: summary(q1)$dispersion 1.0556


def test_overdispersion_note_without_switching(capsys):
    od = pd.DataFrame({"x": np.arange(1, 21), "y": [0, 12, 1, 0, 25, 3, 0, 1, 30, 2, 0, 7, 44, 1, 0, 19, 2, 60, 0, 5]})
    got = rows(make_table(od).explain("y", using="x", outcome="counts"))
    # R: po <- glm(y ~ x, family = poisson, data = od); deviance(po) / df.residual(po)  21.5988
    check(got["x"], {"log rate ratio": 0.062580774014284, "standard error": 0.012376742079836,
                     "p-value": 4.27422914340600e-07})
    out = capsys.readouterr().out
    assert "residual deviance / df = 21.6, above 1.5" in out
    assert 'outcome="counts (overdispersed)"' in out and got["x"]["statistic name"] == "z"


# ---------------------------------------------------------------- mixed model
def test_mixed_model_matches_lme4(school, engine, capsys):
    pytest.importorskip("statsmodels")
    got = rows(make(school, engine).explain("marks", using=["hours", "breakfast"], grouped_by="school"))
    # R: m1 <- lmer(marks ~ hours + breakfast + (1 | school), data = s, REML = TRUE)
    #    fixef(m1); sqrt(diag(vcov(m1))); confint(m1, method = "Wald")
    check(got["(intercept)"], {"effect": 32.54247952940091, "standard error": 2.094689297228810,
                               "95 % CI low": 28.436963948030925, "95 % CI high": 36.64799511077089}, rel=1e-5)
    check(got["hours"], {"effect": 2.80599100405524, "standard error": 0.216296745071906,
                         "95 % CI low": 2.382057173741058}, rel=1e-5)
    check(got["breakfast (yes)"], {"effect": 2.97464502138913, "standard error": 1.100184824074570,
                                   "95 % CI high": 5.13096765291282}, rel=1e-5)
    assert got["hours"]["statistic name"] == "z"
    out = capsys.readouterr().out
    # R: as.data.frame(VarCorr(m1))$vcov  30.1106 46.7956;  ICC 0.39152
    assert "Variance between groups 30.1, within groups 46.8" in out and "ICC = 0.392" in out
    assert "12 groups" in out and "lme4::lmer(marks ~ hours + breakfast + (1 | school))" in out
    assert "Wald z-tests" in out and "Satterthwaite" in out


def test_mixed_model_variances_and_wald_test(class_marks):
    pytest.importorskip("statsmodels")
    from simplydata.table.models import ModelWords

    t = make_table(class_marks)
    got = rows(t.explain("marks", using=["hours", "subject"], grouped_by="name"))
    # R: m2 <- lmer(marks ~ hours + subject + (1 | name), data = mk); fixef(m2); sqrt(diag(vcov(m2)))
    check(got["hours"], {"effect": 5.01289086016590, "standard error": 0.451652669862403}, rel=1e-5)
    check(got["subject (Maths)"], {"effect": -1.22666240739045, "standard error": 2.319811609351706}, rel=1e-5)
    # R: b <- fixef(m2)[3:4]; V <- vcov(m2)[3:4, 3:4]; t(b) %*% solve(V) %*% b  -> 3.3995, p 0.18273
    check(got["subject (whole column)"], {"statistic": 3.399526729646236, "p-value": 0.182726758515958}, rel=1e-4)
    # R: as.data.frame(VarCorr(m2))$vcov  22.4282 48.2159
    X = np.column_stack([np.ones(len(class_marks)), class_marks.hours, class_marks.subject == "Maths",
                         class_marks.subject == "Science"]).astype(float)
    fit = ModelWords._fit_one("mixed", X, class_marks.marks.to_numpy(float), class_marks.name.to_numpy(object))
    assert fit["between"] == pytest.approx(22.4281945684373, rel=1e-4)
    assert fit["within"] == pytest.approx(48.2159169937685, rel=1e-4)


def test_mixed_model_not_yet_for_yes_no(school):
    with pytest.raises(SimplydataError, match="not in simplydata yet"):
        make_table(school).explain("passed", using="hours", outcome="yes or no", grouped_by="school")


# ---------------------------------------------------------------- multiple imputation
def test_multiple_imputation_close_to_r_mice(school, engine, capsys):
    pytest.importorskip("statsmodels")
    t = make(school, engine)
    got = rows(t.explain("marks", using=["hours", "breakfast", "sleep"], empty_cells="multiple imputation"))
    # R: imp <- mice(s[, c("marks", "hours", "breakfast", "sleep")], m = 20, seed = 1)
    #    summary(pool(with(imp, lm(marks ~ hours + breakfast + sleep))), conf.int = TRUE)
    # The filling-in is random, so the pooled estimates are compared within a fifth of R's standard error.
    r = {"(intercept)": (14.85148833370181, 5.684466868992708), "hours": (2.73829190154065, 0.259069287481892),
         "breakfast (yes)": (3.27564860285531, 1.308614226105913), "sleep": (2.24952148173618, 0.701113591071218)}
    for term, (estimate, se) in r.items():
        assert abs(got[term]["effect"] - estimate) < 0.2 * se, term
        assert got[term]["standard error"] == pytest.approx(se, rel=0.05), term
    # sleep has the empty cells: part of its variance comes from the filling-in, and it uses all 180 rows
    assert got["sleep"]["share of variance from empty cells"] > 0.01
    complete = rows(make(school, engine).explain("marks", using=["hours", "breakfast", "sleep"]))
    assert got["hours"]["standard error"] < complete["hours"]["standard error"]  # 180 rows instead of 161
    out = capsys.readouterr().out
    assert "(180 rows, 19 of them with empty cells filled in)" in out
    assert "filled in 20 times" in out and "Rubin's rules" in out and "seed=1" in out
    again = rows(make(school, engine).explain("marks", using=["hours", "breakfast", "sleep"],
                                              empty_cells="multiple imputation"))
    assert again["sleep"]["effect"] == got["sleep"]["effect"]  # the same seed gives the same answer
    other = rows(make(school, engine).explain("marks", using=["hours", "breakfast", "sleep"],
                                              empty_cells="multiple imputation", seed=7))
    assert other["sleep"]["effect"] != got["sleep"]["effect"]


def test_multiple_imputation_logistic_close_to_r_mice(school):
    pytest.importorskip("statsmodels")
    got = rows(make_table(school).explain("passed", using=["hours", "sleep"], outcome="yes or no",
                                          empty_cells="multiple imputation"))
    # R: imp2 <- mice(s[, c("passed", "hours", "sleep")], m = 20, seed = 1)
    #    summary(pool(with(imp2, glm(passed ~ hours + sleep, family = binomial))))
    r = {"(intercept)": (-8.593662062047756, 1.9522023190134896), "hours": (0.531541589859232, 0.0945977569416381),
         "sleep": (0.711304413207242, 0.2237741038460201)}
    for term, (estimate, se) in r.items():
        assert abs(got[term]["log odds ratio"] - estimate) < 0.2 * se, term
        assert got[term]["standard error"] == pytest.approx(se, rel=0.08), term


def test_multiple_imputation_needs_no_groups(school):
    with pytest.raises(SimplydataError, match="cannot be used with grouped_by"):
        make_table(school).explain("marks", using="hours", grouped_by="school", empty_cells="multiple imputation")


# ---------------------------------------------------------------- notes
def test_strongly_linked_columns_are_named(capsys):
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    data = pd.DataFrame({"x": x, "x2": x + rng.normal(scale=0.1, size=60), "z": rng.normal(size=60)})
    data["y"] = data.x + data.z + rng.normal(size=60)
    make_table(data).explain("y", using=["x", "x2", "z"])
    note = [line for line in capsys.readouterr().out.splitlines() if "variance inflation factor" in line][0]
    # variance inflation factor of x: 1 / (1 - R squared of x on x2 and z)
    others = np.column_stack([np.ones(60), data.x2, data.z])
    fitted = others @ np.linalg.lstsq(others, data.x, rcond=None)[0]
    r2 = 1 - ((data.x - fitted) ** 2).sum() / ((data.x - data.x.mean()) ** 2).sum()
    assert f'"x" ({sdata_num(1 / (1 - r2))})' in note and '"x2"' in note and '"z"' not in note


def test_too_few_rows_per_effect(capsys):
    data = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "b": [2.0, 1, 4, 3, 6, 5, 8, 9],
                         "y": [1.0, 3, 2, 5, 4, 7, 6, 9]})
    make_table(data).explain("y", using=["a", "b"])
    assert "only 8 rows for 2 effects" in capsys.readouterr().out


# ---------------------------------------------------------------- add_predicted_column
def test_add_predicted_column_matches_r_fitted(school, engine, capsys):
    t = make(school, engine)
    t.add_predicted_column("marks", using=["hours", "sleep"])
    values = t._frame.to_pandas()["predicted marks"]
    # R: head(fitted(lm(marks ~ hours + sleep, data = s)), 3)  rows 1, 3, 4 (row 2 has no sleep)
    assert values[0] == pytest.approx(46.8798096605902, rel=1e-9)
    assert values[2] == pytest.approx(50.7194844507314, rel=1e-9)
    assert pd.isna(values[1])
    assert "19 rows with an empty cell stay empty" in capsys.readouterr().out
    t.add_predicted_column("passed", using=["hours", "breakfast"], outcome="yes or no", name="chance")
    # R: head(fitted(g1), 3)
    assert t._frame.to_pandas()["chance"][:3].tolist() == pytest.approx(
        [0.288644044969628, 0.279897943502409, 0.442338408816744], rel=1e-8)
    t.undo()
    t.undo()
    assert "predicted marks" not in t._columns


def test_add_predicted_column_mixed_model(school):
    pytest.importorskip("statsmodels")
    t = make_table(school).add_predicted_column("marks", using=["hours", "breakfast"], grouped_by="school")
    # R: head(fitted(m1), 3)  (with each school's own level)
    assert t._frame.to_pandas()["predicted marks"][:3].tolist() == pytest.approx(
        [38.6549726182820, 39.1042257360214, 43.0326131416988], rel=1e-5)
    with pytest.raises(SimplydataError, match="already a column"):
        t.add_predicted_column("marks", using="hours")


# ---------------------------------------------------------------- friendly errors
def test_friendly_errors(school):
    t = make_table(school)
    with pytest.raises(SimplydataError, match='Did you mean "hours"'):
        t.explain("marks", using=["hour"])
    with pytest.raises(SimplydataError, match='Did you mean outcome="yes or no"'):
        t.explain("passed", using="hours", outcome="yes or not")
    with pytest.raises(SimplydataError, match='use outcome="yes or no"'):
        t.explain("passed", using="hours")
    with pytest.raises(SimplydataError, match="exactly 2 different values"):
        t.explain("school", using="hours", outcome="yes or no")
    with pytest.raises(SimplydataError, match="needs using="):
        t.explain("marks")
    with pytest.raises(SimplydataError, match="both be explained"):
        t.explain("marks", using=["marks", "hours"])
    with pytest.raises(SimplydataError, match="whole numbers of 0 or more"):
        t.explain("hours", using="marks", outcome="counts")
    with pytest.raises(SimplydataError, match="cannot separate"):
        make_table(school.assign(hours2=school.hours * 2)).explain("marks", using=["hours", "hours2"])
    with pytest.raises(SimplydataError, match="only one group"):
        make_table(school.assign(one="x")).explain("marks", using=["hours", "one"])
    with pytest.raises(SimplydataError, match="statsmodels|grouped_by"):
        t.explain("marks", using="hours", grouped_by="hours")
