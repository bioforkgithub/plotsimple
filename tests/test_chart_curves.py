"""Tests for the curve chart words: curve, fit_formula, fit_distribution, lines(joined_by=),
smooth_line(method="spline") and qq_dots(distribution=).

The drawn line data is checked against the formula, the fit engines and table.values_at
(whose numbers are checked against R in test_between.py); the printed fit values against
R 4.6.2 (values written in below, with the R call that made them).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.chart import Chart
from simplydata.table.between import smoothing_spline
from simplydata.table.table import Table


def drawn(chart):
    return chart._render().axes[0]


def long_lines(ax, at_least=50):
    """The lines drawn from many points (curves), not legend handles or two-point lines."""
    return [line for line in ax.lines if len(line.get_xdata()) >= at_least]


# ====================================================================== curve
def test_curve_of_a_formula():
    ax = drawn(Chart().curve("y = 100 * exp(-0.3 * x)", between=[0, 10], label="decay"))
    line = long_lines(ax)[0]
    xs = np.asarray(line.get_xdata())
    assert len(xs) == 200 and xs[0] == 0 and xs[-1] == 10
    assert np.allclose(line.get_ydata(), 100 * np.exp(-0.3 * xs))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["decay"]
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"


def test_curve_over_dots_uses_the_chart_range():
    chart = Chart(sdata.example("marks")).dots(x="hours", y="marks").curve("40 + 5 * hours", dashed=True,
                                                                                 steps=50)
    ax = drawn(chart)
    line = long_lines(ax, 50)[0]
    hours = sdata.example("marks")["hours"].dropna()
    assert line.get_xdata()[0] == pytest.approx(hours.min()) and line.get_xdata()[-1] == pytest.approx(hours.max())
    assert np.allclose(line.get_ydata(), 40 + 5 * np.asarray(line.get_xdata()))
    assert line.get_linestyle() == "--"


def test_curve_friendly_errors():
    with pytest.raises(SimplydataError, match="one name"):
        Chart().curve("a * exp(b * x)", between=[0, 1])
    with pytest.raises(SimplydataError, match=r"\*\* instead of \^|use \*\*"):
        Chart().curve("x ^ 2", between=[0, 1])
    with pytest.raises(SimplydataError, match="between="):
        Chart().curve("x ** 2", between=[3, 1])
    with pytest.raises(SimplydataError, match="needs between="):
        Chart().curve("x ** 2")._render()
    with pytest.raises(SimplydataError, match="does not know the colour"):
        Chart().curve("x ** 2", between=[0, 1], colour="redd")
    with pytest.raises(SimplydataError, match="distribution"):
        Chart().curve("x ** 2", between=[0, 1], show="cumulative")


def test_curve_of_a_distribution():
    dist = sdata.distribution("normal", average=50, standard_deviation=10) if hasattr(sdata, "distribution") \
        else pytest.importorskip("simplydata.distributions").distribution("normal", average=50, standard_deviation=10)
    ax = drawn(Chart().curve(dist))
    line = long_lines(ax)[0]
    xs = np.asarray(line.get_xdata())
    # from the 0.1 % to the 99.9 % point: R qnorm(c(0.001, 0.999), 50, 10)
    assert xs[0] == pytest.approx(19.097676938322, abs=1e-9) and xs[-1] == pytest.approx(80.902323061678, abs=1e-9)
    assert np.allclose(line.get_ydata(), stats.norm(50, 10).pdf(xs))
    assert ax.get_ylabel() == "density"
    ax = drawn(Chart().curve(dist, show="cumulative", between=[20, 80]))
    line = long_lines(ax)[0]
    assert np.allclose(line.get_ydata(), stats.norm(50, 10).cdf(np.asarray(line.get_xdata())))


def test_curve_of_a_count_distribution_is_bars():
    distributions = pytest.importorskip("simplydata.distributions")
    dist = distributions.distribution("binomial", tries=10, chance=0.3)
    ax = drawn(Chart().curve(dist, between=[0, 10]))
    heights = [p.get_height() for p in ax.patches]
    assert np.allclose(heights, stats.binom(10, 0.3).pmf(np.arange(11)))  # R dbinom(0:10, 10, 0.3)


def test_curve_of_a_distribution_on_a_histogram_is_scaled(capsys):
    distributions = pytest.importorskip("simplydata.distributions")
    marks = sdata.example("marks")
    dist = distributions.distribution("normal", average=65, standard_deviation=15)
    chart = Chart(marks).histogram(x="marks", bins=10).curve(dist)
    ax = drawn(chart)
    values = marks["marks"].dropna()
    width = (values.max() - values.min()) / 10
    line = long_lines(ax)[0]
    xs = np.asarray(line.get_xdata())
    assert np.allclose(line.get_ydata(), stats.norm(65, 15).pdf(xs) * len(values) * width)
    assert f"density x {len(values)} values x bar width" in capsys.readouterr().out
    chart = Chart(marks).histogram(x="marks", show="percent").curve(dist)
    line = long_lines(drawn(chart))[0]
    assert np.allclose(line.get_ydata(), stats.norm(65, 15).pdf(np.asarray(line.get_xdata())) * 100 * width)
    chart = Chart(marks).histogram(x="marks", show="density").curve(dist)
    line = long_lines(drawn(chart))[0]
    assert np.allclose(line.get_ydata(), stats.norm(65, 15).pdf(np.asarray(line.get_xdata())))


# ====================================================================== fit_formula
def test_fit_formula_on_dots_matches_r(capsys):
    growth = sdata.example("growth")
    applia = growth[growth["country"] == "Applia"]
    chart = Chart(applia).dots(x="year", y="population").fit_formula(
        "population = a * exp(b * (year - 2000))", start={"a": 10, "b": 0.03}, show_equation=True)
    ax = drawn(chart)
    out = capsys.readouterr().out
    # R: nls(population ~ a * exp(b * (year - 2000)), data = Applia, start = list(a = 10, b = 0.03))
    #    a = 11.9771184809804 (se 0.0787018596), b = 0.0297361841598 (se 0.000391979899), sigma 0.179268136
    assert 'Fitted "population = a * exp(b * (year - 2000))" to 13 rows: a = 12.0' in out
    assert "b = 0.0297" in out and "Residual standard error 0.179 on 11 degrees of freedom" in out
    assert "as R nls()" in out
    line = long_lines(ax)[0]
    xs = np.asarray(line.get_xdata())
    assert np.allclose(line.get_ydata(), 11.9771184809804 * np.exp(0.0297361841598 * (xs - 2000)), rtol=1e-6)
    assert any("population = 12.0 * exp(0.0297 * (year - 2000))" in t.get_text() for t in ax.texts)
    drawn(chart)
    assert capsys.readouterr().out == ""  # the values are printed once, not at every drawing


def test_fit_formula_same_values_as_the_table_engine(capsys):
    from simplydata.table.fitting import fit_formula_core

    growth = sdata.example("growth")
    chart = Chart(growth).dots(x="year", y="population", colour="country").fit_formula(
        "population = a * exp(b * (year - 2000))", start={"a": 10, "b": 0.03})
    lines = long_lines(drawn(chart))
    assert len(lines) == 4
    for line, (country, rows) in zip(lines, growth.groupby("country", sort=False)):
        fit = fit_formula_core("population = a * exp(b * (year - 2000))",
                               {"year": rows["year"].to_numpy(float), "population": rows["population"].to_numpy(float)},
                               list(growth.columns), start={"a": 10, "b": 0.03})
        assert np.allclose(line.get_ydata(), fit.predict({"year": np.asarray(line.get_xdata())}))
    out = capsys.readouterr().out
    assert out.count("Fitted ") == 4 and 'for "Cherria"' in out


def test_fit_formula_friendly_errors():
    growth = sdata.example("growth")
    with pytest.raises(SimplydataError, match="one = sign"):
        Chart(growth).dots(x="year", y="population").fit_formula("a * exp(b * year)")
    with pytest.raises(SimplydataError, match='should start with\\s+"population'):
        Chart(growth).dots(x="year", y="population").fit_formula("forest = a * year")._render()
    with pytest.raises(SimplydataError, match="works on dots"):
        Chart(growth).lines(x="year", y="population").fit_formula("population = a * year")


# ====================================================================== fit_distribution
def test_fit_distribution_on_a_histogram(capsys):
    pytest.importorskip("simplydata.distributions")
    marks = sdata.example("marks")
    values = marks["marks"].dropna().to_numpy(float)
    ax = drawn(Chart(marks).histogram(x="marks").fit_distribution("normal"))
    out = capsys.readouterr().out
    # R: MASS::fitdistr(x, "normal"): mean 64.90740740741 (2.02594090971), sd 14.88756443344 (1.43255655554)
    assert 'Fitted a normal distribution to "marks" (54 values): average = 64.9 (standard error 2.03), ' \
           "standard deviation = 14.9 (1.43)" in out
    width = (values.max() - values.min()) / 10
    line = long_lines(ax)[0]
    xs = np.asarray(line.get_xdata())
    expected = stats.norm(64.90740740741, 14.88756443344).pdf(xs) * 54 * width
    assert np.allclose(line.get_ydata(), expected, rtol=1e-9)
    assert "density x 54 values" in out


def test_fit_distribution_groups_on_density(capsys):
    distributions = pytest.importorskip("simplydata.distributions")
    marks = sdata.example("marks")
    ax = drawn(Chart(marks).density(x="marks", colour="class").fit_distribution("gamma"))
    fitted = ax.lines[-3:]  # drawn after the three density curves
    assert len(ax.lines) == 6 and all(len(line.get_xdata()) == 200 for line in fitted)
    for line, (name, rows) in zip(fitted, marks.dropna(subset=["marks"]).groupby("class", sort=True)):
        dist = distributions.fit(rows["marks"].to_numpy(float), "gamma")
        assert np.allclose(line.get_ydata(), dist._pdf(np.asarray(line.get_xdata())))
    assert capsys.readouterr().out.count("Fitted gamma") == 3
    with pytest.raises(SimplydataError, match="histogram or density"):
        Chart(marks).dots(x="hours", y="marks").fit_distribution()


# ====================================================================== lines(joined_by=)
@pytest.mark.parametrize("how", ["smooth curve", "curve without overshoot", "akima"])
def test_lines_joined_by_a_curve(how):
    weather = sdata.example("weather")
    ax = drawn(Chart(weather).lines(x="month_number", y="temperature", colour="city", joined_by=how))
    curves = long_lines(ax, 100)
    assert len(curves) == weather["city"].nunique()
    city = weather["city"].iloc[0]
    line = curves[0]
    xs = np.asarray(line.get_xdata())
    table = Table(weather).keep_rows("city", equal_to=city)
    expected = table.values_at("temperature", at={"month_number": list(xs)}, joined_by=how)
    assert np.allclose(line.get_ydata(), expected._frame["temperature"].to_numpy())
    assert ax.get_legend() is not None


def test_lines_steps_and_errors():
    weather = sdata.example("weather")
    ax = drawn(Chart(weather).lines(x="month_number", y="temperature", colour="city", joined_by="steps"))
    assert ax.lines[0].get_drawstyle() == "steps-post"
    with pytest.raises(SimplydataError, match='Did you mean joined_by="smooth curve"'):
        Chart(weather).lines(x="month_number", y="temperature", joined_by="smoth curve")
    with pytest.raises(SimplydataError, match="error="):
        Chart(weather).lines(x="month_number", y="temperature", joined_by="smooth curve",
                                  error="standard error")
    with pytest.raises(SimplydataError, match="numbers on x"):
        Chart(weather).lines(x="month", y="temperature", joined_by="smooth curve")._render()


# ====================================================================== smooth_line(method="spline")
def test_smooth_line_spline():
    marks = sdata.example("marks")
    ax = drawn(Chart(marks).dots(x="hours", y="marks").smooth_line(method="spline"))
    line = long_lines(ax)[0]
    rows = marks.dropna(subset=["hours", "marks"])
    expected = smoothing_spline(rows["hours"], rows["marks"])(np.asarray(line.get_xdata()))
    assert np.allclose(line.get_ydata(), expected)
    # R: smooth.spline(m$hours, m$marks, all.knots = TRUE, lambda = 0.01); predict(f, 2)$y = 46.3672143365
    ax = drawn(Chart(marks).dots(x="hours", y="marks").smooth_line(method="spline", smoothness=0.01))
    line = long_lines(ax)[0]
    spline = smoothing_spline(rows["hours"], rows["marks"], 0.01)
    assert np.allclose(line.get_ydata(), spline(np.asarray(line.get_xdata())))
    assert spline([2])[0] == pytest.approx(46.3672143365, abs=1e-4)
    with pytest.raises(SimplydataError, match='method="spline"'):
        Chart(marks).dots(x="hours", y="marks").smooth_line(smoothness=0.5)
    with pytest.raises(SimplydataError, match='Did you mean method="spline"'):
        Chart(marks).dots(x="hours", y="marks").smooth_line(method="splin")


def test_smooth_line_lowess_unchanged():
    from simplydata.words.stats import lowess

    marks = sdata.example("marks")
    ax = drawn(Chart(marks).dots(x="hours", y="marks").smooth_line())
    rows = marks.dropna(subset=["hours", "marks"])
    xs, ys = lowess(rows["hours"].to_numpy(float), rows["marks"].to_numpy(float))
    line = [line for line in ax.lines if len(line.get_xdata()) == len(xs)][0]
    assert np.allclose(line.get_ydata(), ys)


# ====================================================================== qq_dots(distribution=)
def test_qq_dots_against_a_fitted_gamma(capsys):
    distributions = pytest.importorskip("simplydata.distributions")
    trial = sdata.example("trial")
    ax = drawn(Chart(trial).qq_dots("days", distribution="gamma"))
    values = trial["days"].dropna().to_numpy(float)
    dist = distributions.fit(values, "gamma")
    (theory, ordered), (slope, intercept, _) = stats.probplot(values, dist=dist._frozen)
    dots = ax.collections[0].get_offsets()
    assert np.allclose(dots[:, 0], theory) and np.allclose(dots[:, 1], ordered)
    assert ax.get_xlabel() == "gamma quantiles (fitted)"
    out = capsys.readouterr().out
    assert "fitted by maximum likelihood" in out and "gamma" in out
    # the normal Q-Q plot is unchanged
    ax = drawn(Chart(trial).qq_dots("days"))
    (theory, ordered), _ = stats.probplot(values, dist="norm")
    assert np.allclose(ax.collections[0].get_offsets()[:, 0], theory)
    assert ax.get_xlabel() == "normal quantiles"
