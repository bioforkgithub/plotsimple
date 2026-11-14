"""Tests for the fitting words: fit_line, fit_curve, fit_formula and add_predicted_column with a formula.

Every printed number is checked against R 4.6.2 (reference values written in below, with the R call
that made them), on pandas and Polars tables. R read the practice tables with
read.csv(".../src/simplydata/data/growth.csv") and read.csv(".../data/marks.csv").

R's nls() stops by its own rule (relative offset below 1e-5), so its estimates are the least-squares
answer to about 7 digits; nls(algorithm = "port") and minpack.lm::nlsLM() agree with each other and with
simplydata to about 9. The tests use rel=1e-6 for estimates and 1e-5 for standard errors.
"""

import math

import numpy as np
import pandas as pd
import pytest

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.table import table as make_table
from simplydata.table.fitting import FitResult, fit_curve_core, fit_formula_core

ENGINES = ["pandas", "polars"]
EST, SE = 1e-6, 1e-5
GROWTH = "population = a * exp(b * (year - 2000))"


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


def make(data, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return make_table(pl.from_pandas(data))
    return make_table(data)


def rows(result):
    return result._frame.to_pandas().to_dict("records") if hasattr(result._frame, "to_pandas") else \
        result.as_pandas().to_dict("records")


def applia():
    g = sdata.example("growth")
    return g[g["country"] == "Applia"].reset_index(drop=True)


# R: for (cn in unique(g$country)) { d <- subset(g, country == cn)
#      fit <- nls(population ~ a * exp(b * (year - 2000)), data = d, start = list(a = d$population[1], b = 0.01))
#      summary(fit); sigma(fit); AIC(fit); coef +/- qt(0.975, 11) * se }
R_GROWTH = {
    "Applia": dict(a=11.977118106213835, b=0.029736186214539329, se_a=0.07870185852257694,
                   se_b=0.00039197990044479748, t_a=152.18342147254373, t_b=75.861507645663266,
                   p_a=1.2358303397059646e-19, p_b=2.5977661289806771e-16, sigma=0.1792681360589157,
                   aic=-3.9699895435871841, low_a=11.80389648353401, high_b=0.030598928158472524),
    "Bananaland": dict(a=29.905389357088307, b=0.012244130655053522, se_a=0.1817519044952820,
                       se_b=0.0003977940499822586, t_a=164.53962031448538, t_b=30.780074904588453,
                       p_a=5.2381212915761771e-20, p_b=5.0422722708243408e-12, sigma=0.37112177178653077,
                       aic=14.948847608578184, low_a=29.505356112475798, high_b=0.013119669455837226),
    "Cherria": dict(a=4.993722565282843, b=0.044332967575763015, se_a=0.05016738276078677,
                    se_b=0.00056306493569430118, t_a=99.541221615933594, t_b=78.735088557942518,
                    p_a=1.3139158772056848e-17, p_b=1.7269753989175140e-16, sigma=0.12676881035861282,
                    aic=-12.979447537766568, low_a=4.883304900305714, high_b=0.045572265143394124),
    "Dateland": dict(a=44.730266904114735, b=0.0040590962571448574, se_a=0.18860077552125573,
                     se_b=0.00029028019935384075, t_a=237.16905076603746, t_b=13.983372845203851,
                     p_a=9.3967421874717196e-22, p_b=2.3801637085861004e-08, sigma=0.36775025212733992,
                     aic=14.711566681133229, low_a=44.315159396010678, high_b=0.0046979986681911032),
}


def check_growth_row(got, want):
    assert got["a"] == pytest.approx(want["a"], rel=EST)
    assert got["b"] == pytest.approx(want["b"], rel=EST)
    assert got["standard error of a"] == pytest.approx(want["se_a"], rel=SE)
    assert got["standard error of b"] == pytest.approx(want["se_b"], rel=SE)
    assert got["t of a"] == pytest.approx(want["t_a"], rel=SE)
    assert got["t of b"] == pytest.approx(want["t_b"], rel=SE)
    assert got["p-value of a"] == pytest.approx(want["p_a"], rel=1e-3)
    assert got["p-value of b"] == pytest.approx(want["p_b"], rel=1e-3)
    assert got["95 % CI low of a"] == pytest.approx(want["low_a"], rel=EST)
    assert got["95 % CI high of b"] == pytest.approx(want["high_b"], rel=EST)
    assert got["residual standard error"] == pytest.approx(want["sigma"], rel=1e-7)
    assert got["degrees of freedom"] == 11
    assert got["AIC"] == pytest.approx(want["aic"], abs=1e-6)
    assert got["rows"] == 13 and got["converged"] == "yes"


# ---------------------------------------------------------------- fit_formula
def test_fit_formula_for_each_country_matches_r_nls(engine, capsys):
    result = make(sdata.example("growth"), engine).fit_formula(GROWTH, for_each="country")
    got = rows(result)
    assert [r["country"] for r in got] == list(R_GROWTH)
    for row in got:
        check_growth_row(row, R_GROWTH[row["country"]])
    out = capsys.readouterr().out
    assert f'Fitted "{GROWTH}" for each "country":' in out
    assert ("  Applia (13 rows): a = 12.0 (95 % CI 11.8 to 12.2), b = 0.0297 (95 % CI 0.0289 to 0.0306). "
            "Residual standard error 0.179 on 11 degrees of freedom; R squared = 1.00 (only a rough guide") in out
    assert "  Dateland (13 rows): a = 44.7 (95 % CI 44.3 to 45.1), b = 0.00406 (95 % CI 0.00342 to 0.00470)" in out
    assert ("Values found: a, b. Data: year. Method: nonlinear least squares (Levenberg-Marquardt), as R nls(); "
            "intervals are estimate +/- t x standard error (R confint() on nls profiles") in out


def test_fit_formula_one_fit_matches_nlslm_from_the_same_start(engine, capsys):
    # every value starts at 1, as R: minpack.lm::nlsLM(population ~ a * exp(b * (year - 2000)), data = d,
    #   start = list(a = 1, b = 1)); coef(summary(fit)); AIC(fit)
    got = rows(make(applia(), engine).fit_formula(GROWTH))[0]
    assert got["a"] == pytest.approx(11.977118347280268, rel=1e-8)
    assert got["b"] == pytest.approx(0.029736184895048149, rel=1e-8)
    assert got["standard error of a"] == pytest.approx(0.078701858955756029, rel=1e-7)
    assert got["standard error of b"] == pytest.approx(0.00039197989739835672, rel=1e-7)
    assert got["AIC"] == pytest.approx(-3.969989543601212, abs=1e-8)
    assert got["R squared"] == pytest.approx(0.998205, abs=1e-6)
    out = capsys.readouterr().out
    assert (f'Fitted "{GROWTH}" to 13 rows: a = 12.0 (95 % CI 11.8 to 12.2), b = 0.0297 (95 % CI 0.0289 to '
            "0.0306). Residual standard error 0.179 on 11 degrees of freedom") in out


def test_weighted_fit_matches_r_nls_weights(engine, capsys):
    data = applia().assign(w=lambda d: 1 / d["population"])
    got = rows(make(data, engine).fit_formula(GROWTH, weights="w"))[0]
    # R: nls(f, data = d, start = list(a = 10, b = 0.01), weights = 1 / population); summary; AIC
    #    (minpack.lm::nlsLM(..., weights = 1 / population) gives the same to 9 digits)
    assert got["a"] == pytest.approx(11.961958831982569, rel=EST)
    assert got["b"] == pytest.approx(0.029818122764469341, rel=EST)
    assert got["standard error of a"] == pytest.approx(0.06618254543014179, rel=SE)
    assert got["standard error of b"] == pytest.approx(0.00035647271288790277, rel=SE)
    assert got["residual standard error"] == pytest.approx(0.039729918572425721, rel=1e-7)
    assert got["AIC"] == pytest.approx(-6.2323880942658896, abs=1e-6)
    assert got["95 % CI low of a"] == pytest.approx(11.816292033185869, rel=EST)
    assert got["95 % CI high of b"] == pytest.approx(0.030602713908309258, rel=EST)
    out = capsys.readouterr().out
    assert "Method: weighted nonlinear least squares (Levenberg-Marquardt), as R nls(weights = w);" in out


def test_bounded_fit_at_its_limit_matches_r_port(engine, capsys):
    t = make(applia(), engine)
    got = rows(t.fit_formula(GROWTH, start={"a": 10, "b": 0.01}, between={"a": [0, 100], "b": [0, 0.025]}))[0]
    # R: nls(f, data = d, start = list(a = 10, b = 0.01), algorithm = "port", lower = c(0, 0),
    #        upper = c(100, 0.025)); summary(fit); sigma(fit); AIC(fit)
    assert got["a"] == pytest.approx(12.849508139378134, rel=EST)
    assert got["b"] == pytest.approx(0.025, rel=1e-12)
    assert got["standard error of a"] == pytest.approx(0.3095781364755425535, rel=SE)
    assert got["standard error of b"] == pytest.approx(0.0014707861476336314, rel=SE)
    assert got["residual standard error"] == pytest.approx(0.68354082090978363, rel=1e-7)
    assert got["AIC"] == pytest.approx(30.828507330163529, abs=1e-6)
    assert got["method"] == "trust region reflective"
    out = capsys.readouterr().out
    assert ('Method: nonlinear least squares within the limits of between= (trust region reflective), as R '
            'nls(algorithm = "port", lower =, upper =);') in out
    assert "Note: b ended at its limit (0.025): the best fit may lie beyond it" in out


def test_bounded_fit_with_room_equals_the_free_fit(engine, capsys):
    got = rows(make(applia(), engine).fit_formula(GROWTH, between={"a": [0, 100], "b": [-1, 1]}))[0]
    # R: nls(..., algorithm = "port", lower = c(0, -1), upper = c(100, 1))
    assert got["a"] == pytest.approx(11.977118351786306, rel=1e-8)
    assert got["b"] == pytest.approx(0.029736184869825329, rel=1e-8)
    assert got["standard error of b"] == pytest.approx(0.00039197989770340693, rel=SE)
    assert "ended at its limit" not in capsys.readouterr().out


def test_weighted_and_bounded_fit_matches_r_port(engine):
    data = applia().assign(w=lambda d: 1 / d["population"])
    got = rows(make(data, engine).fit_formula(GROWTH, between={"b": [0, 0.025]}, weights="w",
                                              start={"a": 10, "b": 0.01}))[0]
    # R: nls(f, d, start = list(a = 10, b = 0.01), weights = 1 / population, algorithm = "port",
    #        lower = c(0, 0), upper = c(100, 0.025))
    assert got["a"] == pytest.approx(12.750910579139045, rel=EST)
    assert got["standard error of a"] == pytest.approx(0.2870503004953723, rel=SE)
    assert got["standard error of b"] == pytest.approx(0.0014922310369434129, rel=SE)
    assert got["residual standard error"] == pytest.approx(0.16753974614874628, rel=1e-7)
    assert got["AIC"] == pytest.approx(31.184630330382518, abs=1e-6)


def test_whole_range_search_finds_the_r_answer_without_start_values(engine, capsys):
    g = sdata.example("growth")
    t = make(g[g["country"] == "Bananaland"].reset_index(drop=True), engine)
    got = rows(t.fit_formula(GROWTH, between={"a": [0, 100], "b": [-1, 1]}, search="whole range"))[0]
    want = R_GROWTH["Bananaland"]  # R nls() from the good start a = 30, b = 0.01
    assert got["a"] == pytest.approx(want["a"], rel=EST)
    assert got["b"] == pytest.approx(want["b"], rel=EST)
    assert got["standard error of b"] == pytest.approx(want["se_b"], rel=SE)
    assert got["AIC"] == pytest.approx(want["aic"], abs=1e-6)
    assert got["method"] == "differential evolution, then trust region reflective"
    out = capsys.readouterr().out
    assert "Method: differential evolution over the whole range of between= (seed 1), then nonlinear least" in out
    again = rows(t.fit_formula(GROWTH, between={"a": [0, 100], "b": [-1, 1]}, search="whole range"))[0]
    assert again["a"] == got["a"]  # seeded: the same every time


def test_whole_range_finds_what_a_bad_start_misses():
    # y = sin(k * x) with k = 3: from k = 1 the local search stops in a wrong dip (R nls() from start
    # k = 1 fails or stops there too); the whole-range search over 0 to 5 finds k = 3 (exact data).
    x = np.linspace(0, 6, 40)
    data = pd.DataFrame({"x": x, "y": np.sin(3 * x)})
    t = make_table(data)
    near = rows(t.fit_formula("y = sin(k * x)", start={"k": 1}))[0]["k"]
    assert abs(near - 3) > 0.5
    found = rows(t.fit_formula("y = sin(k * x)", between={"k": [0, 5]}, search="whole range"))[0]
    assert found["k"] == pytest.approx(3, abs=1e-9)


def test_method_choice_is_used_and_printed(capsys):
    t = make_table(applia())
    got = rows(t.fit_formula(GROWTH, method="dogbox", between={"a": [0, 100], "b": [0, 1]}))[0]
    assert got["a"] == pytest.approx(11.977118347280268, rel=EST) and got["method"] == "dogbox"
    got = rows(t.fit_formula(GROWTH, method="trf"))[0]  # SciPy's own name is accepted
    assert got["b"] == pytest.approx(0.029736184895048149, rel=EST)
    assert "(trust region reflective)" in capsys.readouterr().out


def test_scipy_option_names_are_explained():
    with pytest.raises(SimplydataError, match="has no option called p0="):
        make_table(applia()).fit_formula(GROWTH, p0={"a": 10, "b": 0.01})


def test_a_group_that_cannot_be_fitted_gets_a_note_and_an_empty_row(capsys):
    g = sdata.example("growth")
    small = pd.DataFrame({"year": [2000, 2002], "country": ["Tiny", "Tiny"], "population": [1.0, 1.1],
                          "forest": [1.0, 1.0]})
    got = rows(make_table(pd.concat([g, small])).fit_formula(GROWTH, for_each="country"))
    tiny = got[-1]
    assert tiny["country"] == "Tiny" and math.isnan(tiny["a"]) and tiny["converged"] == "no"
    check_growth_row(got[0], R_GROWTH["Applia"])
    assert 'Note: group "Tiny" was not fitted: fit_formula() needs more rows (2) than values to find (2).' in \
        capsys.readouterr().out


def test_empty_cells_are_skipped_with_a_note(engine, capsys):
    data = applia()
    data.loc[3, "population"] = None
    got = rows(make(data, engine).fit_formula(GROWTH))[0]
    assert got["rows"] == 12
    assert 'Note: 1 row with an empty cell in "population" or "year" was skipped.' in capsys.readouterr().out


@pytest.mark.parametrize("options, message", [
    (dict(search="whole range"), 'search="whole range" needs a low and a high limit for every value'),
    (dict(search="whole range", between={"a": [0, 100]}), "b has none"),
    (dict(method="levenberg-marquardt", between={"b": [0, 1]}), "Levenberg-Marquardt cannot keep values"),
    (dict(method="newton"), 'does not know method="newton"'),
    (dict(search="everywhere"), 'does not know search="everywhere"'),
    (dict(between=[0, 1]), "between=\\[low, high\\] is for one value to find"),
    (dict(between={"c": [0, 1]}), "'c', which is not in the formula"),
    (dict(between={"b": [1, 0]}), "low limit for b in between= should be below"),
    (dict(between={"b": [0, 1]}, start={"b": 5}), "start value for b is outside its limits"),
    (dict(start=2), "start= is one number, but the formula has 2 values"),
    (dict(weights="year"), None),  # positive weights are fine
])
def test_fit_formula_option_errors(options, message):
    t = make_table(applia())
    if message is None:
        t.fit_formula(GROWTH, **options)
        return
    with pytest.raises(SimplydataError, match=message):
        t.fit_formula(GROWTH, **options)


def test_weights_must_be_above_zero():
    data = applia().assign(w=1.0)
    data.loc[0, "w"] = 0
    with pytest.raises(SimplydataError, match="weights= needs a number above 0"):
        make_table(data).fit_formula(GROWTH, weights="w")


def test_a_fit_that_fails_is_explained():
    data = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8], "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    with pytest.raises(SimplydataError, match="impossible numbers"):
        make_table(data).fit_formula("y = a * log(x - b)", start={"a": 1, "b": 10})
    with pytest.raises(SimplydataError, match=r"impossible numbers \(too big to work with\)"):
        make_table(data).fit_formula("y = a * exp(b * x) + c * exp(d * x) + e", start={"b": 50})


def test_values_the_data_cannot_pin_down_get_a_note(capsys):
    # b and c only appear as b * c, so the data fix their product but not each one (R nls(): "singular
    # gradient"); simplydata gives the fit with a note instead of failing
    data = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8], "y": [1.1, 2.0, 2.9, 4.2, 5.0, 5.9, 7.1, 8.0]})
    got = rows(make_table(data).fit_formula("y = a + b * c * x"))[0]
    assert got["b"] * got["c"] == pytest.approx(float(np.polyfit(data.x, data.y, 1)[0]), rel=1e-8)
    assert "Note: the standard errors could not be worked out well" in capsys.readouterr().out
    assert not math.isfinite(got["standard error of b"])


def test_the_gradient_rank_check_does_not_depend_on_scipy_noticing():
    # SciPy's own check depends on rounding (it missed b * c on macOS); this one uses R qr()'s tolerance
    from simplydata.table.fitting import _not_pinned_down

    x, w = np.arange(1.0, 9.0), np.ones(8)
    assert _not_pinned_down(lambda _, a, b, c: a + b * c * x, [0.1, 1.3, 0.75], w)
    assert _not_pinned_down(lambda _, a, b: a + 0 * b * x, [0.1, 1.3], w)  # b does nothing
    assert not _not_pinned_down(lambda _, a, b: a * np.exp(b * x), [2.0, 0.1], w)
    assert not _not_pinned_down(lambda _, a, b: a + b * x, [1e6, 1e-6], w)


def test_a_start_that_overflows_is_explained_before_the_search():
    data = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8], "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    with pytest.raises(SimplydataError, match=r'With the start values \(start=\{.*"b": 50.*too big to work with'):
        make_table(data).fit_formula("y = a * exp(b * x) + c * exp(d * x) + e", start={"b": 50})


def test_a_single_number_start_and_between_list_for_one_value(capsys):
    data = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2]})
    one = rows(make_table(data).fit_formula("y = k * x", start=1.5))[0]
    # R: coef(lm(y ~ x - 1)) = sum(x * y) / sum(x^2)
    assert one["k"] == pytest.approx(float((data.x * data.y).sum() / (data.x ** 2).sum()), rel=1e-10)
    limited = rows(make_table(data).fit_formula("y = k * x", between=[0, 1.5]))[0]
    assert limited["k"] == pytest.approx(1.5)


# ---------------------------------------------------------------- fit_formula_core (used by chart.fit_formula)
def test_fit_formula_core_works_on_plain_arrays():
    d = applia()
    data = {"year": d["year"].to_numpy(float), "population": d["population"].to_numpy(float)}
    fit = fit_formula_core(GROWTH, data, ["year", "population"], start={"a": 10, "b": 0.01})
    assert isinstance(fit, FitResult) and fit.names == ["a", "b"] and fit.df == 11
    assert fit.values[0] == pytest.approx(11.977118347280268, rel=1e-8)
    assert fit.sigma == pytest.approx(0.1792681360589157, rel=1e-7)
    assert fit.cov.shape == (2, 2)
    # R: predict(fit, newdata = data.frame(year = c(2030, 2001)))
    assert fit.predict({"year": np.array([2030.0, 2001.0])}) == pytest.approx(
        [29.226726235424668, 12.338620351991871], rel=1e-7)
    assert fit.method_sentence.startswith("Method: nonlinear least squares (Levenberg-Marquardt), as R nls()")


# ---------------------------------------------------------------- add_predicted_column with a formula
R_FITTED = [11.977118338125090, 12.711033480063289, 13.489920327244779, 14.316534585550141, 15.193800820698819,
            16.124822805378439, 17.112894500409396, 18.161511708796056, 19.274384443896697, 20.455450055470862,
            21.708887160044110, 23.039130424875889, 24.450886257836324]  # R: fitted(nls(...))


def test_add_predicted_column_with_a_formula_matches_r_fitted_and_predict(engine, capsys):
    data = pd.concat([applia(), pd.DataFrame({"year": [2030], "country": ["Applia"], "population": [None]})],
                     ignore_index=True)
    data["population"] = data["population"].astype(float)
    t = make(data, engine)
    same = t.add_predicted_column(GROWTH)
    assert same is t and t.column_names()[-1] == "predicted population"
    got = t.as_pandas()["predicted population"].tolist()
    assert got[:13] == pytest.approx(R_FITTED, rel=1e-7)
    assert got[13] == pytest.approx(29.226726235424668, rel=1e-7)  # R: predict(fit, data.frame(year = 2030))
    out = capsys.readouterr().out
    assert (f'Added "predicted population": "{GROWTH}" with a = 12.0, b = 0.0297 (fitted to 13 rows by nonlinear '
            "least squares, as R nls()). Undo with table.undo().") in out
    t.undo()
    assert "predicted population" not in t.column_names()


def test_add_predicted_column_for_each_group(engine):
    t = make(sdata.example("growth"), engine)
    t.add_predicted_column(GROWTH, for_each="country", name="trend")
    frame = t.as_pandas()
    for country, want in R_GROWTH.items():
        part = frame[frame["country"] == country]
        expected = want["a"] * np.exp(want["b"] * (part["year"] - 2000))
        assert part["trend"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-6)


def test_add_predicted_column_using_a_fit_result(engine, capsys):
    t = make(sdata.example("growth"), engine)
    fit = t.fit_formula(GROWTH, for_each="country")
    t.add_predicted_column(GROWTH, using=fit, for_each="country")
    frame = t.as_pandas()
    part = frame[frame["country"] == "Cherria"]
    want = R_GROWTH["Cherria"]
    assert part["predicted population"].to_numpy() == pytest.approx(
        (want["a"] * np.exp(want["b"] * (part["year"] - 2000))).to_numpy(), rel=1e-6)
    assert 'with values found for each "country" (4 groups) (values from using=, fitted to 52 rows)' in \
        capsys.readouterr().out
    one = make(applia(), engine)
    one.add_predicted_column(GROWTH, using=one.fit_formula(GROWTH))
    assert one.as_pandas()["predicted population"].tolist() == pytest.approx(R_FITTED, rel=1e-7)


@pytest.mark.parametrize("call, message", [
    (lambda t: t.add_predicted_column("population", using=["year"], start={"a": 1}), "start= is for a formula"),
    (lambda t: t.add_predicted_column(GROWTH, outcome="counts"), "outcome= and grouped_by= are for a model"),
    (lambda t: t.add_predicted_column(GROWTH, using=t.fit_formula("population = a * year + b")),
     "using= has no column for 'b'|not of"),
    (lambda t: t.add_predicted_column(GROWTH, using=["year"]), "takes the result of table.fit_formula"),
    (lambda t: t.add_predicted_column(GROWTH, name="year"), 'already a column called "year"'),
])
def test_add_predicted_column_formula_errors(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(make_table(applia()))


# ---------------------------------------------------------------- fit_curve
def test_fit_curve_matches_r_lm_poly(engine, capsys):
    got = rows(make(sdata.example("marks"), engine).fit_curve(x="hours", y="marks", degree=2))
    # R: fit <- lm(marks ~ poly(hours, 2, raw = TRUE), data = m); summary(fit); confint(fit)
    assert [r["term"] for r in got] == ["(intercept)", "hours", "hours**2"]
    want = [(30.99604022005571, 5.069361940938749, 6.1143868954671312, 1.3570612382675401e-07, 20.81887146157029,
             41.17320897854113),
            (8.25633459990281, 2.148962708203755, 3.8420092486407089, 3.3919467112689106e-04, 3.942111943898987,
             12.570557255906635),
            (-0.29655292686764861, 0.20351532325443125, -1.4571528183993467, 1.5120425970960663e-01,
             -0.7051269868438258, 0.11202113310852857)]
    for row, (b, se, t, p, low, high) in zip(got, want):
        assert row["estimate"] == pytest.approx(b, rel=1e-10)
        assert row["standard error"] == pytest.approx(se, rel=1e-10)
        assert row["statistic"] == pytest.approx(t, rel=1e-10)
        assert row["p-value"] == pytest.approx(p, rel=1e-8)
        assert row["95 % CI low"] == pytest.approx(low, rel=1e-10)
        assert row["95 % CI high"] == pytest.approx(high, rel=1e-10)
        assert row["R squared"] == pytest.approx(0.70339188159064259, rel=1e-10)
        assert row["adjusted R squared"] == pytest.approx(0.69176019067262851, rel=1e-10)
        assert row["degrees of freedom"] == 51 and row["rows"] == 54
    out = capsys.readouterr().out
    assert ("Curve: marks = 31.0 + 8.26 * hours - 0.297 * hours**2 (each with its 95 % CI in the table); "
            "R squared = 0.70, adjusted 0.69.") in out
    assert "Method: least squares on powers of hours, as R lm(marks ~ poly(hours, 2, raw = TRUE))." in out
    assert "wiggle" not in out


def test_weighted_fit_curve_matches_r_lm_weights(engine, capsys):
    data = sdata.example("marks").assign(w=lambda d: d["height"] / 100)
    got = rows(make(data, engine).fit_curve(x="hours", y="marks", weights="w"))
    # R: lm(marks ~ poly(hours, 2, raw = TRUE), data = m, weights = height / 100); summary; confint
    assert got[0]["estimate"] == pytest.approx(31.087086851390552, rel=1e-10)
    assert got[1]["standard error"] == pytest.approx(2.1455552317402207, rel=1e-10)
    assert got[2]["p-value"] == pytest.approx(0.15620947309570846, rel=1e-8)
    assert got[2]["95 % CI high"] == pytest.approx(0.11539379255719912, rel=1e-9)
    assert got[0]["R squared"] == pytest.approx(0.70480745584068416, rel=1e-10)
    assert got[0]["adjusted R squared"] == pytest.approx(0.69323127763835801, rel=1e-10)
    assert 'weighted least squares (weights "w")' in capsys.readouterr().out


def test_fit_curve_for_each_class_matches_r(engine, capsys):
    got = rows(make(sdata.example("marks"), engine).fit_curve(x="hours", y="marks", degree=3, for_each="class"))
    assert len(got) == 12 and [r["class"] for r in got[::4]] == ["A", "B", "C"]
    # R: for each class: lm(marks ~ poly(hours, 3, raw = TRUE), data = subset(m, class == cl))
    want = {"A": (-14.313681748012252, 0.22610380478963801, 0.81006977728818552, 0.76937044384993958),
            "B": (26.258231205997554, 0.12537651433833227, 0.74875627011428147, 0.69491832799591324),
            "C": (19.300521520662254, 0.13891589912655039, 0.83941865635127266, 0.80500836842654544)}
    for i, cls in enumerate("ABC"):
        intercept, se_cube, r2, adjusted = want[cls]
        block = got[4 * i: 4 * i + 4]
        assert block[0]["estimate"] == pytest.approx(intercept, rel=1e-8)
        assert block[3]["standard error"] == pytest.approx(se_cube, rel=1e-8)
        assert block[3]["R squared"] == pytest.approx(r2, rel=1e-10)
        assert block[3]["adjusted R squared"] == pytest.approx(adjusted, rel=1e-10)
        assert block[3]["rows"] == 18
    out = capsys.readouterr().out
    assert 'Curves of degree 3 for each "class"' in out
    assert "  A (18 rows): marks = -14.3 + 40.0 * hours - 6.55 * hours**2 + 0.382 * hours**3; R squared = 0.81" in out


def test_high_degree_gets_a_note(capsys):
    make_table(sdata.example("marks")).fit_curve(x="hours", y="marks", degree=4)
    assert "Note: high degrees wiggle between points" in capsys.readouterr().out


def test_fit_curve_degree_one_is_the_straight_line(capsys):
    t = make_table(sdata.example("marks"))
    curve = rows(t.fit_curve(x="hours", y="marks", degree=1))
    line = rows(t.fit_line(x="hours", y="marks"))[0]
    assert curve[1]["estimate"] == pytest.approx(line["slope"], rel=1e-12)
    assert curve[0]["standard error"] == pytest.approx(line["intercept standard error"], rel=1e-10)


def test_fit_curve_core_is_steady_with_big_x():
    # x near 2000: the matrix of powers is badly scaled, QR still gives R's answer
    x = np.arange(2000.0, 2025.0)
    y = 3 + 0.5 * (x - 2000) - 0.02 * (x - 2000) ** 2
    fit = fit_curve_core(x, y, 2)
    assert fit["fitted"] == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("call, message", [
    (lambda t: t.fit_curve(x="hours", y="marks", degree=11), "degree= should be a whole number from 1 to 10"),
    (lambda t: t.fit_curve(x="hours", y="marks", degree=2.5), "degree= should be a whole number"),
    (lambda t: t.fit_curve(x="hours", y="hours"), "two different columns"),
    (lambda t: t.fit_curve(x="hours", y="name"), "text"),
    (lambda t: t.fit_curve(x="hours", y="marks", for_each="hours"), "both a column of the fit"),
    (lambda t: t.fit_curve(x="hours", y="marks", degree=9, for_each="name"), "could not fit a curve in any group"),
    (lambda t: t.fit_curve(x="hourz", y="marks"), 'Did you mean "hours"'),
])
def test_fit_curve_friendly_errors(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(make_table(sdata.example("marks")))


# ---------------------------------------------------------------- fit_line and its old name
def test_fit_line_is_the_main_name_and_fit_straight_line_still_works(engine, capsys):
    t = make(sdata.example("marks"), engine)
    line = rows(t.fit_line(x="hours", y="marks"))[0]
    # R: lm(marks ~ hours, data = m)
    assert line["slope"] == pytest.approx(5.20319382959493, rel=1e-12)
    old = rows(t.fit_straight_line(x="hours", y="marks"))[0]
    assert old["slope"] == line["slope"]
    out = capsys.readouterr().out
    assert 'Note: "fit_straight_line" is now called "fit_line". The old name will stop working' in out
    assert "fit_line" in t._word_names() and "fit_straight_line" not in t._word_names()
