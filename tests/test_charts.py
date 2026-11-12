"""Chart-type words: each one is checked against the library that does the sums."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from scipy import stats
from scipy.cluster import hierarchy
from scipy.interpolate import griddata

import plotsimple as ps
from plotsimple import PlotsimpleError


def draw(chart):
    """Draw a chart's panel steps and return the drawing context (ax, store, last)."""
    fig = Figure()
    ax = fig.add_subplot(projection=chart._settings["projection"])
    return chart._draw_panel(fig, ax)


@pytest.fixture
def big():
    rng = np.random.default_rng(7)
    return pd.DataFrame({"class": rng.choice(["A", "B", "C"], 600),
                         "marks": rng.normal(60, 12, 600).round()})


@pytest.fixture
def sales():
    return pd.DataFrame({"year": [2020, 2021, 2022] * 2 + [2020],
                         "shop": ["N"] * 3 + ["S"] * 3 + ["N"],
                         "sales": [10.0, 20, 30, 5, 6, 7, 30]})


@pytest.fixture
def grid():
    xs, ys = np.meshgrid(np.linspace(0, 3, 7), np.linspace(-1, 1, 5))
    return pd.DataFrame({"lon": xs.ravel(), "lat": ys.ravel(), "depth": (xs * 10 + ys).ravel()})


# ---------------------------------------------------------------- area
def test_area_heights_are_averages(sales):
    ctx = draw(ps.chart(sales).area(x="year", y="sales", colour="shop"))
    expected = sales.groupby(["year", "shop"])["sales"].mean().unstack("shop")
    north = ctx.ax.lines[0]
    assert list(north.get_xdata()) == [2020, 2021, 2022]
    assert np.allclose(north.get_ydata(), expected["N"])  # 2020: (10 + 30) / 2 = 20


def test_area_stacked_top_is_total(sales):
    ctx = draw(ps.chart(sales).area(x="year", y="sales", colour="shop", stacked=True))
    top = ctx.ax.collections[-1].get_paths()[0].vertices[:, 1].max()
    totals = sales.groupby(["year", "shop"])["sales"].mean().groupby("year").sum()
    assert np.isclose(top, totals.max())


def test_area_stacked_needs_colour(sales):
    with pytest.raises(PlotsimpleError, match="stacked"):
        ps.chart(sales).area(x="year", y="sales", stacked=True)


# ---------------------------------------------------------------- histogram
def test_histogram_counts_match_numpy(big):
    ax = draw(ps.chart(big).histogram(x="marks", bins=10)).ax
    counts, edges = np.histogram(big["marks"], bins=10)
    assert [p.get_height() for p in ax.patches] == list(counts)
    assert np.isclose(ax.patches[0].get_x(), edges[0])


def test_histogram_edges_list(big):
    ax = draw(ps.chart(big).histogram(x="marks", bins=[0, 50, 70, 120])).ax
    counts, _ = np.histogram(big["marks"], bins=[0, 50, 70, 120])
    assert [p.get_height() for p in ax.patches] == list(counts)


@pytest.mark.parametrize("bins", [0, -3, 2.5, True, [5, 3], "ten"])
def test_histogram_bad_bins(big, bins):
    with pytest.raises(PlotsimpleError, match="bins="):
        ps.chart(big).histogram(x="marks", bins=bins)


def test_histogram_needs_numbers(big):
    with pytest.raises(PlotsimpleError, match="needs numbers"):
        ps.chart(big).histogram(x="class")


# ---------------------------------------------------------------- count bars
def test_count_bars_match_value_counts(marks):
    fruit = pd.DataFrame({"fruit": ["apple", "pear", "apple", "kiwi", "apple", "pear"]})
    ax = draw(ps.chart(fruit).count_bars(x="fruit")).ax
    expected = fruit["fruit"].value_counts(sort=False)
    assert [p.get_height() for p in ax.patches] == list(expected)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(expected.index)


def test_count_bars_sideways(marks):
    ax = draw(ps.chart(marks).count_bars(x="name", sideways=True)).ax
    assert [p.get_width() for p in ax.patches] == list(marks["name"].value_counts(sort=False))


# ---------------------------------------------------------------- boxes and violins
def test_box_medians_match_pandas(marks):
    ax = draw(ps.chart(marks).boxes(x="class", y="marks")).ax
    flat = {round(line.get_ydata()[0], 6) for line in ax.lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]}
    for median in marks.groupby("class")["marks"].median():
        assert round(median, 6) in flat


def test_violins_stop_at_data(marks):
    ax = draw(ps.chart(marks).violins(x="class", y="marks")).ax
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bodies) == 2
    for body, (_, rows) in zip(bodies, marks.groupby("class", sort=False)):
        ys = body.get_paths()[0].vertices[:, 1]
        assert np.isclose(ys.min(), rows["marks"].min()) and np.isclose(ys.max(), rows["marks"].max())


def test_boxes_need_y():
    with pytest.raises(PlotsimpleError, match="needs y="):
        ps.chart().boxes()


# ---------------------------------------------------------------- all points
def test_all_dots_shows_every_value(marks):
    ax = draw(ps.chart(marks).all_dots(x="class", y="marks")).ax
    ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
    assert sorted(ys) == sorted(marks["marks"])


def test_all_dots_strip_is_same_every_time(big):
    state = np.random.get_state()
    a = draw(ps.chart(big).all_dots(x="class", y="marks")).ax.collections[0].get_offsets()
    b = draw(ps.chart(big).all_dots(x="class", y="marks")).ax.collections[0].get_offsets()
    assert np.array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], state[1])  # user's random numbers untouched


# ---------------------------------------------------------------- density
def test_density_matches_scipy_kde(big):
    ax = draw(ps.chart(big).density(x="marks")).ax
    line = ax.lines[0]
    expected = stats.gaussian_kde(big["marks"])(line.get_xdata())
    assert np.allclose(line.get_ydata(), expected)


def test_density_groups_each_have_area_one(big):
    ax = draw(ps.chart(big).density(x="marks", colour="class")).ax
    for line in ax.lines:
        assert np.isclose(np.trapezoid(line.get_ydata(), line.get_xdata()), 1, atol=0.02)


# ---------------------------------------------------------------- heatmap
@pytest.fixture
def rain():
    return pd.DataFrame({"month": ["Jan", "Feb", "Mar"] * 2 + ["Jan"],
                         "city": ["Bergen"] * 3 + ["Oslo"] * 3 + ["Bergen"],
                         "rain": [190.0, 150, 170, 50, 40, 45, 210]})


def test_heatmap_pivot_values(rain):
    ctx = draw(ps.chart(rain).heatmap(x="month", y="city", value="rain"))
    expected = rain.pivot_table(index="city", columns="month", values="rain", aggfunc="mean")
    drawn = ctx.store["heatmap_table"]
    assert list(drawn.columns) == ["Jan", "Feb", "Mar"]  # order of appearance, not a-z
    pd.testing.assert_frame_equal(drawn, expected.loc[drawn.index, drawn.columns])
    cells = np.asarray(ctx.ax.collections[0].get_array()).reshape(drawn.shape)
    assert np.allclose(cells, drawn.to_numpy())
    assert ctx.store["colour_bar"].ax.get_ylabel() == "rain"


def test_heatmap_numbers_written(rain):
    ax = draw(ps.chart(rain).heatmap(x="month", y="city", value="rain", numbers=True)).ax
    texts = [t.get_text() for t in ax.texts]
    assert "200" in texts and "150" in texts  # Bergen in Jan: (190 + 210) / 2


def test_heatmap_whole_table_uses_number_columns():
    table = pd.DataFrame({"gene": ["g1", "g2"], "s1": [1.0, 2], "s2": [3.0, 4], "ok": [True, False]})
    ctx = draw(ps.chart(table).heatmap())
    drawn = ctx.store["heatmap_table"]
    assert list(drawn.columns) == ["s1", "s2"] and list(drawn.index) == ["g1", "g2"]


def test_heatmap_cluster_order_matches_scipy():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(8, 5))
    table = pd.DataFrame(values, columns=list("abcde"))
    ctx = draw(ps.chart(table).heatmap(cluster=True))
    rows = hierarchy.leaves_list(hierarchy.linkage(values, "average", "euclidean"))
    cols = hierarchy.leaves_list(hierarchy.linkage(values.T, "average", "euclidean"))
    drawn = ctx.store["heatmap_table"]
    assert list(drawn.index) == list(rows)
    assert list(drawn.columns) == [list("abcde")[i] for i in cols]
    assert len(ctx.ax.child_axes) == 2  # two trees, drawn beside the map on the same figure
    assert len(ctx.fig.axes) == 2  # map and colour bar (trees live inside the map): no new figure


def test_heatmap_needs_all_three(rain):
    with pytest.raises(PlotsimpleError, match="all three"):
        ps.chart(rain).heatmap(x="month", value="rain")


# ---------------------------------------------------------------- pie
def test_pie_counts_rows():
    fruit = pd.DataFrame({"fruit": ["apple", "pear", "apple", "kiwi"]})
    ax = draw(ps.chart(fruit).pie(labels="fruit")).ax
    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    sizes = [abs(w.theta2 - w.theta1) for w in wedges]
    assert np.allclose(np.array(sizes) / 360, [0.5, 0.25, 0.25])


def test_pie_values_are_summed(rain):
    ctx = draw(ps.chart(rain).pie(labels="city", values="rain"))
    pd.testing.assert_series_equal(ctx.store["pie_sizes"], rain.groupby("city", sort=False)["rain"].sum())


def test_pie_refuses_negative():
    with pytest.raises(PlotsimpleError, match="below 0"):
        ps.chart({"a": ["x", "y"], "v": [3, -1]}).pie(labels="a", values="v")._render()


# ---------------------------------------------------------------- polar and 3D
def test_polar_uses_degrees():
    table = pd.DataFrame({"direction": [0.0, 90, 180], "speed": [1.0, 2, 3]})
    chart = ps.chart(table).polar(angle="direction", distance="speed")
    ax = chart._render().axes[0]
    assert ax.name == "polar"
    assert np.allclose(ax.collections[0].get_offsets()[:, 0], np.deg2rad([0, 90, 180]))


def test_polar_and_flat_words_do_not_mix(marks):
    with pytest.raises(PlotsimpleError, match="cannot share"):
        ps.chart(marks).bars(x="name", y="marks").polar(angle="marks", distance="height")
    with pytest.raises(PlotsimpleError, match="cannot share"):
        ps.chart(marks).polar(angle="marks", distance="height").lines(x="test", y="marks")
    with pytest.raises(PlotsimpleError, match="cannot share"):
        ps.chart(marks).dots_3d(x="test", y="marks", z="height").polar(angle="marks", distance="height")


def test_3d_words_share_a_chart(grid):
    chart = ps.chart(grid).surface(x="lon", y="lat", z="depth").dots_3d(x="lon", y="lat", z="depth")
    ax = chart._render().axes[0]
    assert ax.name == "3d"


def test_dots_3d_every_point(marks):
    ax = ps.chart(marks).dots_3d(x="test", y="marks", z="height", colour="class")._render().axes[0]
    assert sum(len(c._offsets3d[0]) for c in ax.collections) == len(marks)


# ---------------------------------------------------------------- contour and surface
def test_contour_full_grid_is_used_as_is(grid):
    ctx = draw(ps.chart(grid).contour(x="lon", y="lat", z="depth"))
    gx, gy, gz = ctx.store["grid"]
    expected = grid.pivot(index="lat", columns="lon", values="depth")
    assert np.allclose(gz, expected.to_numpy())
    assert ctx.store["colour_bar"].ax.get_ylabel() == "depth"


def test_contour_scattered_matches_griddata():
    rng = np.random.default_rng(5)
    t = pd.DataFrame({"x": rng.uniform(0, 1, 50), "y": rng.uniform(0, 1, 50)})
    t["z"] = t.x + 2 * t.y
    gx, gy, gz = draw(ps.chart(t).contour(x="x", y="y", z="z", filled=False)).store["grid"]
    expected = griddata((t.x, t.y), t.z, (gx, gy), method="linear")
    assert np.allclose(gz, expected, equal_nan=True)


def test_contour_needs_numbers(marks):
    with pytest.raises(PlotsimpleError, match="needs numbers"):
        ps.chart(marks).contour(x="name", y="test", z="marks")


def test_surface_draws_on_3d(grid):
    ctx = draw(ps.chart(grid).surface(x="lon", y="lat", z="depth"))
    assert ctx.ax.name == "3d" and "colour_bar" in ctx.store


# ---------------------------------------------------------------- standard names
def test_standard_name_aliases(marks):
    chart = ps.chart(marks)
    assert chart.bar.__func__ is chart.bars.__func__
    assert chart.line.__func__ is chart.lines.__func__
    assert chart.box.__func__ is chart.boxes.__func__
    assert chart.violin.__func__ is chart.violins.__func__


def test_every_new_word_saves(marks, grid, tmp_path):
    charts = [
        ps.chart(marks).area(x="test", y="marks"),
        ps.chart(marks).histogram(x="marks"),
        ps.chart(marks).box(x="class", y="marks", colour="name"),
        ps.chart(marks).violin(x="class", y="marks").all_dots(x="class", y="marks"),
        ps.chart(marks).count_bars(x="name", colour="class"),
        ps.chart(marks).density(x="marks", colour="class"),
        ps.chart(marks).heatmap(x="test", y="name", value="marks", numbers=True, cluster=True),
        ps.chart(marks).pie(labels="name"),
        ps.chart(marks).polar(angle="marks", distance="height", colour="class"),
        ps.chart(grid).contour(x="lon", y="lat", z="depth"),
        ps.chart(grid).surface(x="lon", y="lat", z="depth"),
        ps.chart(marks).dots_3d(x="test", y="marks", z="height", colour="marks"),
    ]
    for i, chart in enumerate(charts):
        chart.save(tmp_path / f"{i}.png")
        assert (tmp_path / f"{i}.png").stat().st_size > 0
