"""Tests for the statistics words: numbers are checked against SciPy, NumPy, pandas
and (for LOWESS) reference values from statsmodels 0.15.0."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from scipy import stats

import plotsimple as ps
from plotsimple import PlotsimpleError
from plotsimple._stats import t_interval
from plotsimple.words.stats import (equation_text, group_errors, line_band, linear_fit, lowess,
                                    nice_number, poly_fit)


@pytest.fixture
def xy():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 30)
    y = 2.3 * x + 0.5 + rng.normal(0, 1.5, 30)
    return pd.DataFrame({"x": x, "y": y, "g": np.where(np.arange(30) % 2, "a", "b")})


def _fit_lines(ax):
    """Drawn fit lines (seaborn also keeps small legend lines on the axes)."""
    return [ln for ln in ax.lines if len(ln.get_xdata()) > 2]


def _bar_halves(container):
    """Half-widths of the vertical error bars in an errorbar container."""
    segs = container.lines[2][0].get_segments()
    return np.array([(s[1][1] - s[0][1]) / 2 for s in segs]), np.array([(s[1][1] + s[0][1]) / 2 for s in segs])


# ------------------------------------------------------------------ error bars
@pytest.mark.parametrize("kind", ["sd", "se", "ci"])
def test_group_errors_match_libraries(marks, kind):
    out = group_errors(marks, "name", "marks", kind=kind)
    g = marks.groupby("name", sort=False)["marks"]
    assert np.allclose(out["mean"], g.mean())
    if kind == "sd":
        expected = g.std(ddof=1)
    elif kind == "se":
        expected = g.apply(lambda v: stats.sem(v))
    else:
        expected = g.apply(lambda v: np.diff(stats.t.interval(0.95, len(v) - 1, v.mean(), stats.sem(v)))[0] / 2)
        assert np.allclose(out["half"], g.apply(lambda v: np.diff(t_interval(v))[0] / 2))
    assert np.allclose(out["half"], expected)


def test_group_errors_skip_empty_cells():
    t = pd.DataFrame({"x": ["a"] * 4, "y": [1.0, 2.0, np.nan, 4.0]})
    out = group_errors(t, "x", "y", kind="sd")
    assert out["n"].iloc[0] == 3
    assert np.isclose(out["half"].iloc[0], np.std([1, 2, 4], ddof=1))


@pytest.mark.parametrize("kind", ["sd", "se", "ci"])
def test_error_bars_drawn_on_dots(marks, kind):
    fig = ps.chart(marks).dots(x="name", y="marks").error_bars(kind)._render()
    halves, centres = _bar_halves(fig.axes[0].containers[0])
    expected = group_errors(marks, "name", "marks", kind=kind)
    assert np.allclose(halves, expected["half"])
    assert np.allclose(centres, expected["mean"])


def test_error_bars_per_colour_group_on_lines(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="class").error_bars("se")._render()
    containers = fig.axes[0].containers
    assert len(containers) == 2
    for cont, cls in zip(containers, ["A", "B"]):
        halves, centres = _bar_halves(cont)
        part = marks[marks["class"] == cls].groupby("test")["marks"]
        assert np.allclose(sorted(centres), sorted(part.mean()))
        assert np.allclose(sorted(halves), sorted(part.apply(stats.sem)))


def test_error_bars_from_column():
    t = pd.DataFrame({"day": [1, 2, 3], "temp": [3.0, 5.0, 4.0], "error": [0.5, 1.0, 0.25]})
    fig = ps.chart(t).lines(x="day", y="temp").error_bars(column="error")._render()
    halves, centres = _bar_halves(fig.axes[0].containers[0])
    assert np.allclose(halves, [0.5, 1.0, 0.25])
    assert np.allclose(centres, [3, 5, 4])


def test_error_bars_column_checks(marks):
    with pytest.raises(PlotsimpleError, match='Did you mean "marks"'):
        ps.chart(marks).dots(x="name", y="marks").error_bars(column="mraks")
    t = pd.DataFrame({"x": [1, 2], "y": [1.0, 2.0], "e": [1.0, -1.0]})
    with pytest.raises(PlotsimpleError, match="below 0"):
        ps.chart(t).dots(x="x", y="y").error_bars(column="e")._render()


def test_error_bars_need_two_values():
    t = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(PlotsimpleError, match="only one"):
        ps.chart(t).dots(x="x", y="y").error_bars()._render()


def test_error_bars_bad_kind_and_layer(marks):
    with pytest.raises(PlotsimpleError, match='Did you mean kind="sd"'):
        ps.chart(marks).error_bars("sdd")
    with pytest.raises(PlotsimpleError, match="last chart type is bars"):
        ps.chart(marks).bars(x="name", y="marks").error_bars()
    with pytest.raises(PlotsimpleError, match="median"):
        ps.chart(marks).lines(x="test", y="marks", show="median").error_bars()
    with pytest.raises(PlotsimpleError, match="none yet"):
        ps.chart(marks).error_bars()._render()


# ------------------------------------------------------------------ fit_line
def test_linear_fit_matches_linregress(xy):
    fit = linear_fit(xy.x, xy.y)
    ref = stats.linregress(xy.x, xy.y)
    assert np.isclose(fit["slope"], ref.slope)
    assert np.isclose(fit["intercept"], ref.intercept)
    assert np.isclose(fit["r2"], ref.rvalue ** 2)


def test_band_matches_matrix_formula(xy):
    """Confidence band of the mean, written independently with the hat matrix."""
    fit = linear_fit(xy.x, xy.y)
    xs = np.array([-1.0, 0.0, 3.3, 10.0, 12.0])
    low, high = line_band(fit, xs)
    X = np.column_stack([np.ones(len(xy)), xy.x])
    beta, *_ = np.linalg.lstsq(X, xy.y, rcond=None)
    resid = xy.y - X @ beta
    sigma2 = resid @ resid / (len(xy) - 2)
    X0 = np.column_stack([np.ones(len(xs)), xs])
    se = np.sqrt(sigma2 * np.einsum("ij,jk,ik->i", X0, np.linalg.inv(X.T @ X), X0))
    t = stats.t.ppf(0.975, len(xy) - 2)
    assert np.allclose(low, X0 @ beta - t * se)
    assert np.allclose(high, X0 @ beta + t * se)


def test_fit_line_drawn(xy):
    fig = ps.chart(xy).dots(x="x", y="y").fit_line(band=True, show_equation=True)._render()
    ax = fig.axes[0]
    line = ax.lines[-1]
    ref = stats.linregress(xy.x, xy.y)
    assert np.allclose(line.get_ydata(), ref.intercept + ref.slope * line.get_xdata())
    assert line.get_xdata().min() == xy.x.min() and line.get_xdata().max() == xy.x.max()
    assert any(isinstance(c, PolyCollection) for c in ax.collections)  # the band
    texts = [t.get_text() for t in ax.texts]
    assert texts == [equation_text([ref.slope, ref.intercept], ref.rvalue ** 2)]


def test_fit_line_per_group(xy):
    fig = ps.chart(xy).dots(x="x", y="y", colour="g").fit_line(show_equation=True)._render()
    ax = fig.axes[0]
    assert len(_fit_lines(ax)) == 2
    assert [t.get_text().split(":")[0] for t in ax.texts] == ["b", "a"]  # seaborn's order
    for line, g in zip(_fit_lines(ax), ["b", "a"]):
        part = xy[xy.g == g]
        ref = stats.linregress(part.x, part.y)
        assert np.allclose(line.get_ydata(), ref.intercept + ref.slope * line.get_xdata())


def test_fit_line_order_does_not_matter_and_color_spelling(xy):
    chart = ps.chart(xy).fit_line(color="red").dots(x="x", y="y")
    line = chart._render().axes[0].lines[-1]
    assert line.get_color() == "red"


def test_fit_line_errors(marks, xy):
    with pytest.raises(PlotsimpleError, match="needs numbers on x"):
        ps.chart(marks).dots(x="name", y="marks").fit_line()
    with pytest.raises(PlotsimpleError, match="works on dots"):
        ps.chart(marks).bars(x="name", y="marks").fit_line()
    with pytest.raises(PlotsimpleError, match="none yet"):
        ps.chart(xy).fit_line()._render()
    with pytest.raises(PlotsimpleError, match="colour"):
        ps.chart(xy).fit_line(colour="notacolour")
    with pytest.raises(PlotsimpleError, match="2 or more different x"):
        ps.chart().dots(x=[1, 1, 1], y=[1, 2, 3]).fit_line()._render()


def test_fit_line_skips_empty_cells():
    t = pd.DataFrame({"x": [1, 2, 3, 4, np.nan], "y": [2.0, 4.1, np.nan, 8.2, 9.0]})
    line = ps.chart(t).dots(x="x", y="y").fit_line()._render().axes[0].lines[-1]
    ref = stats.linregress([1, 2, 4], [2.0, 4.1, 8.2])
    assert np.allclose(line.get_ydata(), ref.intercept + ref.slope * line.get_xdata())


# ------------------------------------------------------------------ fit_curve
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_poly_fit_matches_least_squares(xy, degree):
    fit = poly_fit(xy.x, xy.y, degree)
    V = np.vander(xy.x, degree + 1)
    coefs, *_ = np.linalg.lstsq(V, xy.y, rcond=None)
    assert np.allclose(fit["coefs"], coefs)
    assert np.allclose(fit["coefs"], np.polyfit(xy.x, xy.y, degree))
    pred = V @ coefs
    r2 = 1 - ((xy.y - pred) ** 2).sum() / ((xy.y - xy.y.mean()) ** 2).sum()
    assert np.isclose(fit["r2"], r2)
    if degree == 1:
        assert np.isclose(fit["r2"], stats.linregress(xy.x, xy.y).rvalue ** 2)


def test_fit_curve_drawn():
    x = np.linspace(-3, 3, 25)
    y = 1.5 * x ** 2 - 2 * x + 0.25
    fig = ps.chart().dots(x=list(x), y=list(y)).fit_curve(degree=2, show_equation=True)._render()
    ax = fig.axes[0]
    line = ax.lines[-1]
    assert np.allclose(line.get_ydata(), 1.5 * line.get_xdata() ** 2 - 2 * line.get_xdata() + 0.25)
    assert ax.texts[0].get_text() == "y = 1.5x² − 2x + 0.25, R² = 1.00"


def test_fit_curve_checks():
    with pytest.raises(PlotsimpleError, match="whole number"):
        ps.chart().fit_curve(degree=2.5)
    with pytest.raises(PlotsimpleError, match="whole number"):
        ps.chart().fit_curve(degree=0)
    with pytest.raises(PlotsimpleError, match="4 or more"):
        ps.chart().dots(x=[1, 2, 3], y=[1, 2, 3]).fit_curve(degree=3)._render()


# ------------------------------------------------------------------ LOWESS
def _lowess_data(case):
    i = np.arange(40, dtype=float)
    if case == "plain":
        x = (i * 7) % 41 + i / 50
        y = np.sin(x / 6) * 3 + ((i * 37) % 11 - 5) / 4
        y[5] += 8  # an outlier
    elif case == "dense":
        i = np.arange(300, dtype=float)
        x = i / 10
        y = np.cos(x / 4) + ((i * 37) % 11 - 5) / 10
    else:  # ties in x
        x = np.floor(i / 3)
        y = 0.5 * x + ((i * 13) % 7 - 3) / 3
    return x, y


# statsmodels 0.15.0: lowess(y, x, frac=2/3, it=..., delta=...)[:, 1]
STATSMODELS = {
    ("plain", 0): [1.4194973551, 1.5315413190, 1.6281894900, 1.7102387554, 1.7791953562, 1.8368496136,
                   1.8848188380, 1.8961695161, 1.9333893705, 1.9628789052, 1.9837671857, 1.9874487614,
                   1.9305767273, 1.6736284995, 1.5964481592, 1.2833213800, 0.9345619555, 0.5555083161,
                   0.1548383350, -0.2547751665, -0.6552459761, -0.7593672234, -1.1321691056, -1.4674598125,
                   -1.7260332301, -1.8835829472, -1.9378289834, -1.8028137970, -1.7491619099, -1.5415957407,
                   -1.3360894751, -1.1304605342, -0.9195316632, -0.6994971765, -0.4033841221, -0.1548535397,
                   0.1052728076, 0.3755189517, 0.6549875499, 0.9429797230],
    ("plain", 3): [1.8620115747, 1.9192713968, 1.9627367951, 1.9928192296, 2.0107844196, 2.0182781959,
                   2.0167824185, 2.0149869773, 2.0034757413, 1.9847274926, 1.9580714092, 1.9182747768,
                   1.8409193129, 1.5949913982, 1.5215724603, 1.2234669298, 0.8902667813, 0.5259402059,
                   0.1381350792, -0.2597243046, -0.6481725180, -0.7490646995, -1.1099136365, -1.4447515815,
                   -1.7493056176, -2.0022487309, -2.1806332655, -2.1820900019, -2.1536079918, -2.0135771115,
                   -1.8650717087, -1.7192066324, -1.5734413056, -1.4246502151, -1.2277492496, -1.0636036423,
                   -0.8901321250, -0.7060292778, -0.5102971711, -0.3026191442],
    ("ties", 3): [0.0922337929] * 3 + [0.5580384594] * 3 + [1.0289271727] * 3 + [1.5021130183] * 3
                 + [1.9928992018] * 3 + [2.5101226140] * 3 + [3.0188031805] * 3 + [3.5285331902] * 3
                 + [4.0024353979] * 3 + [4.5020131456] * 3 + [5.0420381803] * 3 + [5.5806733884] * 3
                 + [6.1201965427] * 3 + [6.6692836542],
}
# dense case with delta = 1 % of the x range, every 25th value
STATSMODELS_DENSE_DELTA = [1.0523060119, 0.6470079986, 0.2589020163, -0.1081551800, -0.4572626712,
                           -0.5707205426, -0.4746640608, -0.1942171305, 0.1693253470, 0.4444211945,
                           0.7022611182, 0.9470224198]


@pytest.mark.parametrize("case,iterations", list(STATSMODELS))
def test_lowess_matches_statsmodels(case, iterations):
    x, y = _lowess_data(case)
    xs, fitted = lowess(x, y, iterations=iterations)
    assert np.all(np.diff(xs) >= 0)
    assert np.allclose(fitted, STATSMODELS[(case, iterations)], atol=1e-6)


def test_lowess_delta_matches_statsmodels():
    x, y = _lowess_data("dense")
    _, fitted = lowess(x, y, delta=0.01 * np.ptp(x))
    assert np.allclose(fitted[::25], STATSMODELS_DENSE_DELTA, atol=1e-6)


def test_lowess_straight_line_is_exact():
    x = np.array([5, 1, 4, 2, 3, 8, 7, 6], dtype=float)
    _, fitted = lowess(x, 3 * x - 1)
    assert np.allclose(fitted, 3 * np.sort(x) - 1)


def test_smooth_line_drawn(xy):
    fig = ps.chart(xy).dots(x="x", y="y", colour="g").smooth_line()._render()
    ax = fig.axes[0]
    assert len(_fit_lines(ax)) == 2
    part = xy[xy.g == "b"]
    xs, fitted = lowess(part.x, part.y)
    assert np.allclose(_fit_lines(ax)[0].get_ydata(), fitted)


# ------------------------------------------------------------------ side histograms
def test_side_histograms(xy):
    fig = ps.chart(xy).dots(x="x", y="y").side_histograms()._render()
    ax = fig.axes[0]
    assert len(ax.child_axes) == 2
    top, right = ax.child_axes
    counts, _ = np.histogram(xy.x, bins=np.histogram_bin_edges(xy.x, bins="auto"))
    heights = top.patches[0].get_path().vertices[:, 1]
    assert np.isclose(heights.max(), counts.max())
    assert top.get_xlim() == ax.get_xlim()
    assert right.get_ylim() == ax.get_ylim()


def test_side_histograms_needs_numbers(marks):
    with pytest.raises(PlotsimpleError, match="needs numbers"):
        ps.chart(marks).dots(x="name", y="marks").side_histograms()


# ------------------------------------------------------------------ reference lines and shading
def test_reference_lines(marks):
    chart = ps.chart(marks).bars(x="name", y="marks")
    chart.line_at_y(50, label="pass mark").line_at_x(1.5, dashed=False, colour="red")
    ax = chart._render().axes[0]
    hline = [ln for ln in ax.lines if list(ln.get_ydata()) == [50, 50]][0]
    assert hline.get_linestyle() == "--"
    vline = [ln for ln in ax.lines if list(ln.get_xdata()) == [1.5, 1.5]][0]
    assert vline.get_linestyle() == "-" and vline.get_color() == "red"
    assert "pass mark" in ax.texts[0].get_text()


def test_shading(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").shade_between_y(60, 40).shade_between_x(0.5, 1.5)
    ax = chart._render().axes[0]
    spans = [p for p in ax.patches if p.get_zorder() == 0]
    assert len(spans) == 2
    assert spans[0].get_y() == 40 and spans[0].get_height() == 20  # low and high may be given either way round


def test_reference_line_checks():
    with pytest.raises(PlotsimpleError, match="needs a number"):
        ps.chart().line_at_y("fifty")
    with pytest.raises(PlotsimpleError, match="needs a number"):
        ps.chart().line_at_x(True)
    with pytest.raises(PlotsimpleError, match="two different values"):
        ps.chart().shade_between_x(3, 3)
    with pytest.raises(PlotsimpleError, match="colour"):
        ps.chart().shade_between_y(1, 2, colour="blu")
    with pytest.raises(PlotsimpleError, match="dashd"):
        ps.chart().line_at_y(1, dashd=True)


def test_lines_work_on_each_panel_and_save(marks, tmp_path):
    chart = ps.chart(marks).dots(x="height", y="marks").fit_line(band=True).line_at_y(50)
    chart.save(tmp_path / "fit.png")
    assert (tmp_path / "fit.png").stat().st_size > 1000


# ------------------------------------------------------------------ text helpers
def test_number_and_equation_text():
    assert nice_number(2.3149) == "2.31"
    assert nice_number(0.48) == "0.48"
    assert nice_number(-1234.5) == "−1234"  # 3 significant figures, no rounding into 1.23e+03
    assert nice_number(0.000123) == "0.000123"
    assert equation_text([2.31, 0.48], 0.9412) == "y = 2.31x + 0.48, R² = 0.94"
    assert equation_text([-1, -3], 0.5) == "y = −x − 3, R² = 0.50"


# ------------------------------------------------------------------ ps.pairs
def test_pairs_grid(marks):
    chart = ps.pairs(marks)
    numbers = ["test", "marks", "height"]
    assert repr(chart) == "<plotsimple pairs chart: test, marks, height>"
    fig = chart.title_top("Pairs")._render()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 9 + 3  # 3 x 3 cells and a twin for each histogram
    assert fig.get_suptitle() == "Pairs"
    cells = fig.axes[:9]
    assert cells[6].get_xlabel() == "test" and cells[3].get_ylabel() == "marks"
    # a scatter cell holds the right numbers: row "marks", column "height"
    offsets = cells[1 * 3 + 2].collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], marks["height"]) and np.allclose(offsets[:, 1], marks["marks"])
    assert len(numbers) == 3


def test_pairs_colour_and_save(marks, tmp_path):
    chart = ps.pairs(marks, colour="class").size(width=7, height=7)
    chart.save(tmp_path / "pairs.png")
    fig = chart._render()
    assert fig.legends and fig.legends[0].get_title().get_text() == "class"
    assert chart._repr_png_()[:4] == b"\x89PNG"


def test_pairs_from_file_and_polars(marks, tmp_path):
    path = tmp_path / "marks.csv"
    marks.to_csv(path, index=False)
    assert ps.pairs(str(path), columns=["marks", "height"])._columns == ["marks", "height"]
    pl = pytest.importorskip("polars")
    assert ps.pairs(pl.from_pandas(marks))._columns == ["test", "marks", "height"]


def test_pairs_errors(marks):
    with pytest.raises(PlotsimpleError, match="at least 2 columns"):
        ps.pairs(marks[["name", "marks"]])
    with pytest.raises(PlotsimpleError, match='Did you mean "class"'):
        ps.pairs(marks, colour="clas")
    with pytest.raises(PlotsimpleError, match="Not numbers: name"):
        ps.pairs(marks, columns=["name", "marks"])
    with pytest.raises(PlotsimpleError, match="already names"):
        ps.pairs(marks).label_x("x")
