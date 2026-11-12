"""Chart words and options of v0.4.x: colour_scale, transparency=, thickness=, tick_format_x/y,
ticks_x/y(every=, how_many=), histogram options, line_with_slope, qq_dots, cumulative_share
and font_name. Each is checked against the matplotlib objects it draws and against the
library that does the sums (SciPy, NumPy)."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import colors as mcolors
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.figure import Figure
from scipy import stats

import simplydata as sdata
from simplydata import SimplydataError
from simplydata.words.axes import date_format

pl = pytest.importorskip("polars")
ENGINES = ["pandas", "polars"]


def as_engine(frame, engine):
    return pl.from_pandas(frame) if engine == "polars" else frame


def draw(chart):
    """Draw a chart's panel steps and return the drawing context (ax, store, last)."""
    fig = Figure(layout="constrained")
    ax = fig.add_subplot(projection=chart._settings["projection"])
    return chart._draw_panel(fig, ax)


@pytest.fixture
def marks():
    return sdata.example("marks")


@pytest.fixture
def weather():
    return sdata.example("weather")


@pytest.fixture
def island():
    return sdata.example("island")


@pytest.fixture
def shop():
    return sdata.example("shop")


def rain_heatmap(weather):
    return sdata.chart(weather).heatmap(x="month", y="city", value="rain")


def mesh_of(ctx):
    return ctx.ax.collections[0]


# ====================================================================== colour_scale
def test_colour_scale_named_map_on_heatmap(weather):
    ctx = draw(rain_heatmap(weather).colour_scale("magma"))
    mesh = mesh_of(ctx)
    assert mesh.get_cmap().name == "magma"
    values = ctx.store["heatmap_table"].to_numpy(dtype=float)
    assert np.isclose(mesh.norm.vmin, values.min()) and np.isclose(mesh.norm.vmax, values.max())


@pytest.mark.parametrize("name, cmap", [("blue to red", "RdBu_r"), ("Purple to Orange", "PuOr_r"),
                                        ("blues", "Blues"), ("Blues", "Blues"), ("rocket", "rocket"),
                                        ("TURBO", "turbo")])
def test_colour_scale_plain_and_library_names(weather, name, cmap):
    ctx = draw(rain_heatmap(weather).colour_scale(name))
    assert mesh_of(ctx).get_cmap().name == cmap


def test_colour_scale_own_colours(weather):
    ctx = draw(rain_heatmap(weather).colour_scale("white", "navy"))
    cmap = mesh_of(ctx).get_cmap()
    assert np.allclose(cmap(0.0), mcolors.to_rgba("white"))
    assert np.allclose(cmap(1.0), mcolors.to_rgba("navy"))


def test_colour_scale_middle_goes_equally_far_both_ways(weather):
    ctx = draw(sdata.chart(weather).heatmap(x="month", y="city", value="temperature")
               .colour_scale("blue to red", middle=10))
    t = weather["temperature"]
    wide = ctx.store["heatmap_table"].to_numpy(dtype=float)
    half = max(wide.max() - 10, 10 - wide.min())
    norm = mesh_of(ctx).norm
    assert np.isclose(norm.vmin, 10 - half) and np.isclose(norm.vmax, 10 + half)
    assert np.isclose(norm(10), 0.5)
    assert t.min() < 10 < t.max()


def test_colour_scale_middle_with_between_is_two_slope(weather):
    ctx = draw(sdata.chart(weather).heatmap(x="month", y="city", value="temperature")
               .colour_scale(middle=0, between=[-20, 40]))
    norm = mesh_of(ctx).norm
    assert isinstance(norm, mcolors.TwoSlopeNorm)
    assert (norm.vmin, norm.vcenter, norm.vmax) == (-20, 0, 40)
    assert mesh_of(ctx).get_cmap().name == "RdBu_r"  # a middle without a name: blue to red


def test_colour_scale_log(weather):
    ctx = draw(rain_heatmap(weather).colour_scale(log=True))
    norm = mesh_of(ctx).norm
    assert isinstance(norm, mcolors.LogNorm)
    wide = ctx.store["heatmap_table"].to_numpy(dtype=float)
    assert np.isclose(norm.vmin, wide.min()) and np.isclose(norm.vmax, wide.max())
    assert np.isclose(norm(np.sqrt(norm.vmin * norm.vmax)), 0.5)  # the middle colour at the geometric mean


def test_colour_scale_steps_as_edges_and_arrow(weather):
    ctx = draw(rain_heatmap(weather).colour_scale(steps=[0, 50, 100, 150]))
    norm = mesh_of(ctx).norm
    assert isinstance(norm, mcolors.BoundaryNorm)
    assert list(norm.boundaries) == [0, 50, 100, 150]
    assert weather["rain"].max() > 150
    assert ctx.store["colour_bar"].extend == "max"  # values above 150: an arrow at the top


def test_colour_scale_steps_as_a_number(weather):
    ctx = draw(rain_heatmap(weather).colour_scale(steps=4))
    wide = ctx.store["heatmap_table"].to_numpy(dtype=float)
    assert np.allclose(mesh_of(ctx).norm.boundaries, np.linspace(wide.min(), wide.max(), 5))
    mesh = mesh_of(ctx)
    assert len(np.unique(mesh.norm(mesh.get_array().compressed()))) == 4  # 4 colours


def test_colour_scale_between_fixes_the_ends(weather):
    ctx = draw(rain_heatmap(weather).colour_scale(between=[0, 100]))
    norm = mesh_of(ctx).norm
    assert (norm.vmin, norm.vmax) == (0, 100)
    assert ctx.store["colour_bar"].extend == "max"


def test_colour_scale_on_contour_steps_are_the_levels(island):
    ctx = draw(sdata.chart(island).contour(x="east", y="north", z="height")
               .colour_scale("Blues", steps=[0, 25, 50, 100]))
    shapes = ctx.store["colour_bar"].mappable
    assert list(shapes.levels) == [0, 25, 50, 100]
    assert shapes.get_cmap().name == "Blues"
    assert shapes.extend in ("max", "both")  # the island is higher than 100


def test_colour_scale_on_surface_and_dots_3d(island, marks):
    ctx = draw(sdata.chart(island).surface(x="east", y="north", z="height").colour_scale("cividis"))
    assert ctx.store["colour_bar"].mappable.get_cmap().name == "cividis"
    ctx = draw(sdata.chart(marks).dots_3d(x="hours", y="height", z="marks", colour="marks")
               .colour_scale("magma", log=True))
    assert isinstance(ctx.store["colour_bar"].mappable.norm, mcolors.LogNorm)


@pytest.mark.parametrize("engine", ENGINES)
def test_colour_scale_on_dots_coloured_by_a_number_column(marks, engine):
    chart = sdata.chart(as_engine(marks, engine)).dots(x="hours", y="marks", colour="height")
    ctx = draw(chart.colour_scale("cividis", between=[140, 160]))
    bar = ctx.store["colour_bar"]
    assert bar.mappable.get_cmap().name == "cividis"
    assert (bar.mappable.norm.vmin, bar.mappable.norm.vmax) == (140, 160)
    dots = ctx.ax.collections[0]
    cmap = matplotlib.colormaps["cividis"]
    norm = mcolors.Normalize(140, 160)
    expected = cmap(norm(marks["height"].to_numpy(dtype=float)))
    assert np.allclose(dots.get_facecolors()[:, :3], expected[:, :3], atol=1e-6)
    assert ctx.ax.get_legend() is None  # the colour bar is the key


def test_colour_scale_default_heatmap_unchanged_without_it(weather):
    ctx = draw(sdata.chart(weather).heatmap(x="month", y="city", value="temperature"))
    assert mesh_of(ctx).get_cmap().name != "viridis"  # both sides of 0: blue to red, as before


def test_colour_scale_unknown_name_says_did_you_mean():
    with pytest.raises(SimplydataError, match='Did you mean "magma"'):
        sdata.chart().colour_scale("magam")


@pytest.mark.parametrize("call, message", [
    (lambda c: c.colour_scale("navy"), "at least two colours"),
    (lambda c: c.colour_scale("white", "nvy"), '"nvy" is not a colour'),
    (lambda c: c.colour_scale(log=True, middle=0), "log=True and middle="),
    (lambda c: c.colour_scale(between=[10, 0]), "low end must be smaller"),
    (lambda c: c.colour_scale(between=[0]), "two numbers in between="),
    (lambda c: c.colour_scale(steps=1), "steps="),
    (lambda c: c.colour_scale(steps=[0, 10, 5]), "going up"),
    (lambda c: c.colour_scale(steps=3, middle=0), "steps= and middle="),
    (lambda c: c.colour_scale(middle=50, between=[0, 30]), "should lie between"),
    (lambda c: c.colour_scale(log=True, between=[0, 10]), "cannot show 0"),
    (lambda c: c.colour_scale(log="yes"), "True or False"),
    (lambda c: c.colour_scale(3), "name of a colour scale"),
])
def test_colour_scale_wrong_options(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(sdata.chart())


def test_colour_scale_log_with_zero_values_is_explained(island):
    chart = sdata.chart(island).contour(x="east", y="north", z="height").colour_scale(log=True)
    with pytest.raises(SimplydataError, match="cannot colour zero or negative"):
        chart._render()


def test_colour_scale_on_a_chart_without_number_colours_is_explained(marks):
    chart = sdata.chart(marks).bars(x="class", y="marks").colour_scale("magma")
    with pytest.raises(SimplydataError, match="nothing coloured by numbers"):
        chart._render()


def test_colour_scale_american_spelling(weather):
    ctx = draw(rain_heatmap(weather).color_scale("magma"))
    assert mesh_of(ctx).get_cmap().name == "magma"


# ====================================================================== transparency=
def alpha_of(artist):
    """The see-through-ness drawn: the artist's alpha, or the alpha inside its colour."""
    if artist.get_alpha() is not None:
        return artist.get_alpha()
    colour = artist.get_color() if hasattr(artist, "get_xydata") else artist.get_facecolor()
    return float(np.atleast_2d(mcolors.to_rgba_array(colour))[0, 3])


def _alpha_of_faces(artists):
    return [float(np.atleast_2d(mcolors.to_rgba_array(a.get_facecolor()))[0, 3]) for a in artists]


TRANSPARENT = [
    ("dots", lambda c: c.dots(x="hours", y="marks", transparency=0.3), lambda ax: [ax.collections[0].get_alpha()]),
    ("bars", lambda c: c.bars(x="class", y="marks", transparency=0.3), lambda ax: [p.get_alpha() for p in ax.patches]),
    ("stacked bars", lambda c: c.bars(x="class", y="marks", colour="subject", stacked=True, transparency=0.3),
     lambda ax: [p.get_alpha() for p in ax.patches]),
    ("count_bars", lambda c: c.count_bars(x="class", transparency=0.3), lambda ax: [p.get_alpha() for p in ax.patches]),
    ("lines", lambda c: c.lines(x="hours", y="marks", transparency=0.3), lambda ax: [ax.lines[0].get_alpha()]),
    ("histogram", lambda c: c.histogram(x="marks", transparency=0.3), lambda ax: [alpha_of(p) for p in ax.patches]),
    ("boxes", lambda c: c.boxes(x="class", y="marks", transparency=0.3), lambda ax: _alpha_of_faces(ax.patches)),
    ("violins", lambda c: c.violins(x="class", y="marks", transparency=0.3),
     lambda ax: _alpha_of_faces([a for a in ax.collections if len(a.get_paths()) and a.get_facecolor().size])[:1]),
    ("all_dots", lambda c: c.all_dots(x="class", y="marks", transparency=0.3), lambda ax: [ax.collections[0].get_alpha()]),
    ("density", lambda c: c.density(x="marks", transparency=0.3), lambda ax: [alpha_of(ax.lines[0])]),
    ("pie", lambda c: c.pie(labels="class", transparency=0.3), lambda ax: _alpha_of_faces(ax.patches)),
    ("area", lambda c: c.area(x="hours", y="marks", transparency=0.3), lambda ax: [ax.collections[0].get_alpha()]),
    ("heatmap", lambda c: c.heatmap(x="subject", y="class", value="marks", transparency=0.3),
     lambda ax: [ax.collections[0].get_alpha()]),
    ("cumulative_share", lambda c: c.cumulative_share(x="marks", transparency=0.3), lambda ax: [ax.lines[0].get_alpha()]),
    ("qq_dots", lambda c: c.qq_dots("marks", transparency=0.3), lambda ax: [ax.collections[0].get_alpha()]),
]


@pytest.mark.parametrize("name, make, alphas", TRANSPARENT, ids=[t[0] for t in TRANSPARENT])
def test_transparency_is_one_minus_alpha(marks, name, make, alphas):
    ctx = draw(make(sdata.chart(marks)))
    found = alphas(ctx.ax)
    assert found and all(a is not None and np.isclose(a, 0.7) for a in found), found


def test_transparency_on_boxes_keeps_outlines_solid(marks):
    ctx = draw(sdata.chart(marks).boxes(x="class", y="marks", transparency=0.5))
    box = ctx.ax.patches[0]
    assert np.isclose(box.get_facecolor()[3], 0.5)
    assert np.isclose(box.get_edgecolor()[3], 1.0)


def test_transparency_on_band_contour_polar_surface(marks, island, weather):
    table = pd.DataFrame({"x": [1, 2, 3], "low": [1, 2, 3], "high": [2, 4, 5]})
    ctx = draw(sdata.chart(table).band(x="x", between=["low", "high"], transparency=0.1))
    assert np.isclose(ctx.ax.collections[0].get_alpha(), 0.9)
    ctx = draw(sdata.chart(island).contour(x="east", y="north", z="height", transparency=0.4))
    assert np.isclose(ctx.store["colour_bar"].mappable.get_alpha(), 0.6)
    ctx = draw(sdata.chart(weather).polar(angle="wind_direction", distance="wind_speed", transparency=0.4))
    assert np.isclose(ctx.ax.collections[0].get_alpha(), 0.6)
    ctx = draw(sdata.chart(island).surface(x="east", y="north", z="height", transparency=0.4))
    assert np.isclose(ctx.store["colour_bar"].mappable.get_alpha(), 0.6)
    ctx = draw(sdata.chart(marks).dots_3d(x="hours", y="height", z="marks", transparency=0.4))
    assert np.isclose(ctx.ax.collections[0].get_alpha(), 0.6)


def test_no_transparency_keeps_the_old_look(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks"))
    assert ctx.ax.collections[0].get_alpha() is None
    table = pd.DataFrame({"x": [1, 2, 3], "low": [1, 2, 3], "high": [2, 4, 5]})
    ctx = draw(sdata.chart(table).band(x="x", between=["low", "high"]))
    assert np.isclose(ctx.ax.collections[0].get_alpha(), 0.25)


@pytest.mark.parametrize("bad, message", [(1, "invisible"), (1.5, "from 0"), (-0.1, "from 0"),
                                          ("half", "from 0"), (True, "from 0")])
def test_transparency_wrong_values(marks, bad, message):
    with pytest.raises(SimplydataError, match=message):
        sdata.chart(marks).dots(x="hours", y="marks", transparency=bad)


# ====================================================================== thickness=
THICK = [
    ("lines", lambda c: c.lines(x="hours", y="marks", thickness=4), lambda ax: ax.lines[0]),
    ("density", lambda c: c.density(x="marks", thickness=4), lambda ax: ax.lines[0]),
    ("cumulative_share", lambda c: c.cumulative_share(x="marks", thickness=4), lambda ax: ax.lines[0]),
    ("area", lambda c: c.area(x="hours", y="marks", thickness=4), lambda ax: ax.lines[0]),
    ("fit_line", lambda c: c.dots(x="hours", y="marks").fit_line(thickness=4), lambda ax: ax.lines[0]),
    ("fit_curve", lambda c: c.dots(x="hours", y="marks").fit_curve(thickness=4), lambda ax: ax.lines[0]),
    ("smooth_line", lambda c: c.dots(x="hours", y="marks").smooth_line(thickness=4), lambda ax: ax.lines[0]),
    ("line_at_x", lambda c: c.dots(x="hours", y="marks").line_at_x(5, thickness=4), lambda ax: ax.lines[0]),
    ("line_at_y", lambda c: c.dots(x="hours", y="marks").line_at_y(50, thickness=4), lambda ax: ax.lines[0]),
    ("line_with_slope", lambda c: c.dots(x="hours", y="marks").line_with_slope(1, thickness=4),
     lambda ax: ax.lines[0]),
    ("qq_dots", lambda c: c.qq_dots("marks", thickness=4), lambda ax: ax.lines[0]),
    ("boxes", lambda c: c.boxes(x="class", y="marks", thickness=4), lambda ax: ax.patches[0]),
    ("histogram steps", lambda c: c.histogram(x="marks", steps=True, thickness=4),
     lambda ax: (ax.patches or ax.lines)[0]),
]


@pytest.mark.parametrize("name, make, artist", THICK, ids=[t[0] for t in THICK])
def test_thickness_sets_the_line_width(marks, name, make, artist):
    ctx = draw(make(sdata.chart(marks)))
    assert np.isclose(artist(ctx.ax).get_linewidth(), 4)


def test_thickness_on_contour_lines(island):
    ctx = draw(sdata.chart(island).contour(x="east", y="north", z="height", filled=False, thickness=3))
    assert np.allclose(ctx.store["colour_bar"].mappable.get_linewidths(), 3)


@pytest.mark.parametrize("bad", [0, -1, 30, "thick", True])
def test_thickness_wrong_values(marks, bad):
    with pytest.raises(SimplydataError, match="thickness="):
        sdata.chart(marks).lines(x="hours", y="marks", thickness=bad)


# ====================================================================== tick_format_x / y
def formatted(axis, value):
    return axis.get_major_formatter().format_data(value) if False else axis.get_major_formatter()(value, 0)


def test_tick_format_percent(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("percent"))
    assert isinstance(ctx.ax.yaxis.get_major_formatter(), ticker.PercentFormatter)
    assert formatted(ctx.ax.yaxis, 50).replace("\\", "") == "50%"
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("percent", out_of=1, decimals=1))
    assert formatted(ctx.ax.yaxis, 0.25).replace("\\", "") == "25.0%"


def test_tick_format_thousands_and_decimals(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("thousands"))
    assert formatted(ctx.ax.yaxis, 12500) == "12,500"
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_x(decimals=2))
    assert formatted(ctx.ax.xaxis, 3) == "3.00"
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("thousands", decimals=1))
    assert formatted(ctx.ax.yaxis, 1234.56) == "1,234.6"


def test_tick_format_scientific_and_plain(marks):
    chart = sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("scientific")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = chart._render()
    ax = fig.axes[0]
    FigureCanvasAgg(fig).draw()
    assert isinstance(ax.yaxis.get_major_formatter(), ticker.ScalarFormatter)
    assert "10" in ax.yaxis.get_offset_text().get_text()  # "×10¹" above the axis
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").tick_format_y("plain"))
    fmt = ctx.ax.yaxis.get_major_formatter()
    assert isinstance(fmt, ticker.ScalarFormatter) and not fmt.get_useOffset()


@pytest.mark.parametrize("example, code", [("Jan 2026", "%b %Y"), ("31.01.2026", "%d.%m.%Y"),
                                           ("2026-01-31", "%Y-%m-%d"), ("31 Jan", "%d %b"),
                                           ("January", "%B"), ("13:45", "%H:%M"), ("Sat 31", "%a %d"),
                                           ("%d %b", "%d %b"), ("'26", "'%y")])
def test_date_formats_from_an_example(example, code):
    assert date_format(example) == code


def test_tick_format_dates(shop):
    ctx = draw(sdata.chart(shop).lines(x="date", y="sales").tick_format_x("Jan 2026"))
    fmt = ctx.ax.xaxis.get_major_formatter()
    assert isinstance(fmt, mdates.DateFormatter)
    assert fmt(mdates.date2num(pd.Timestamp("2026-03-15"))) == "Mar 2026"


@pytest.mark.parametrize("call, message", [
    (lambda c: c.tick_format_y("percnet"), 'Did you mean "percent"'),
    (lambda c: c.tick_format_y(), "needs a format"),
    (lambda c: c.tick_format_y(decimals=-1), "decimals="),
    (lambda c: c.tick_format_y("thousands", out_of=1), 'goes with "percent"'),
    (lambda c: c.tick_format_y("scientific", decimals=1), "leave out decimals"),
    (lambda c: c.tick_format_x("Jan 2026", decimals=1), "date format"),
    (lambda c: c.tick_format_y(5), "format in words"),
])
def test_tick_format_wrong_options(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(sdata.chart())


def test_tick_format_date_on_numbers_and_number_on_dates_are_explained(marks, shop):
    with pytest.raises(SimplydataError, match="writes dates"):
        sdata.chart(marks).dots(x="hours", y="marks").tick_format_x("Jan 2026")._render()
    with pytest.raises(SimplydataError, match="shows dates"):
        sdata.chart(shop).lines(x="date", y="sales").tick_format_x("percent")._render()
    with pytest.raises(SimplydataError, match="groups"):
        sdata.chart(marks).bars(x="class", y="marks").tick_format_x("percent")._render()


# ====================================================================== ticks_x / ticks_y every= how_many=
def visible_ticks(ax, axis="x"):
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    return [t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]


def test_ticks_every_number(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").ticks_y(every=20))
    assert isinstance(ctx.ax.yaxis.get_major_locator(), ticker.MultipleLocator)
    ticks = visible_ticks(ctx.ax, "y")
    assert ticks and all(np.isclose(t % 20, 0) for t in ticks)


def test_ticks_every_month_and_weeks(shop):
    ctx = draw(sdata.chart(shop).lines(x="date", y="sales").ticks_x(every="month"))
    locator = ctx.ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.MonthLocator)
    days = [mdates.num2date(t).day for t in visible_ticks(ctx.ax)]
    assert days and set(days) == {1}
    assert isinstance(ctx.ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    ctx = draw(sdata.chart(shop).lines(x="date", y="sales").ticks_x(every="2 weeks"))
    ticks = visible_ticks(ctx.ax)
    assert isinstance(ctx.ax.xaxis.get_major_locator(), mdates.WeekdayLocator)
    assert all(mdates.num2date(t).weekday() == 0 for t in ticks)  # Mondays
    assert np.allclose(np.diff(ticks), 14)


def test_ticks_how_many_and_whole_numbers(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").ticks_y(how_many=4))
    assert isinstance(ctx.ax.yaxis.get_major_locator(), ticker.MaxNLocator)
    assert 2 <= len(visible_ticks(ctx.ax, "y")) <= 4
    table = pd.DataFrame({"x": [0, 1, 2], "y": [0.0, 1.0, 2.0]})
    ctx = draw(sdata.chart(table).dots(x="x", y="y").ticks_y(whole_numbers=True))
    assert all(float(t).is_integer() for t in visible_ticks(ctx.ax, "y"))


def test_ticks_every_replaces_places_and_back(marks):
    chart = sdata.chart(marks).dots(x="hours", y="marks").ticks_x(0, 5, 10).ticks_x(every=2)
    assert [s.name for s in chart._steps].count("ticks_x") == 0
    chart.ticks_x(1, 2)
    assert [s.name for s in chart._steps].count("ticks_x_spacing") == 0


@pytest.mark.parametrize("call, message", [
    (lambda c: c.ticks_x(1, 2, every=5), "not both"),
    (lambda c: c.ticks_x(every=5, how_many=3), "every= or how_many="),
    (lambda c: c.ticks_x(every="monht"), 'Did you mean every="month"'),
    (lambda c: c.ticks_x(every=0), "bigger than 0"),
    (lambda c: c.ticks_x(how_many=1), "how_many="),
    (lambda c: c.ticks_x(every=2, whole_numbers=True), "whole_numbers"),
    (lambda c: c.ticks_x(), "needs the places"),
])
def test_ticks_wrong_options(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(sdata.chart())


def test_ticks_every_on_the_wrong_axis_is_explained(marks, shop):
    with pytest.raises(SimplydataError, match="needs dates"):
        sdata.chart(marks).dots(x="hours", y="marks").ticks_x(every="month")._render()
    with pytest.raises(SimplydataError, match="needs a time"):
        sdata.chart(shop).lines(x="date", y="sales").ticks_x(every=5)._render()
    with pytest.raises(SimplydataError, match="more than"):
        sdata.chart(marks).dots(x="hours", y="marks").ticks_x(every=0.001)._render()
    with pytest.raises(SimplydataError, match="groups"):
        sdata.chart(marks).bars(x="class", y="marks").ticks_x(every=1)._render()


# ====================================================================== histogram options
def bar_heights(ax):
    return np.array([p.get_height() for p in ax.patches])


def test_histogram_count_matches_numpy(marks):
    ctx = draw(sdata.chart(marks).histogram(x="marks"))
    counts, _ = np.histogram(marks["marks"], bins=10)
    assert np.allclose(bar_heights(ctx.ax), counts)
    assert ctx.ax.get_ylabel() == "count"


def test_histogram_percent_and_density(marks):
    ctx = draw(sdata.chart(marks).histogram(x="marks", show="percent"))
    assert np.isclose(bar_heights(ctx.ax).sum(), 100)
    assert ctx.ax.get_ylabel() == "percent"
    ctx = draw(sdata.chart(marks).histogram(x="marks", show="density"))
    widths = np.array([p.get_width() for p in ctx.ax.patches])
    assert np.isclose((bar_heights(ctx.ax) * widths).sum(), 1)
    counts, edges = np.histogram(marks["marks"], bins=10, density=True)
    assert np.allclose(bar_heights(ctx.ax), counts)


def test_histogram_percent_per_group_and_stacked(marks):
    ctx = draw(sdata.chart(marks).histogram(x="marks", colour="class", show="percent"))
    by_colour = {}
    for p in ctx.ax.patches:
        by_colour.setdefault(mcolors.to_hex(p.get_facecolor()), []).append(p.get_height())
    assert len(by_colour) == 3 and all(np.isclose(sum(h), 100) for h in by_colour.values())
    ctx = draw(sdata.chart(marks).histogram(x="marks", colour="class", show="percent", stacked=True))
    assert np.isclose(bar_heights(ctx.ax).sum(), 100)  # the whole stack makes 100 %
    tops = {}
    for p in ctx.ax.patches:
        tops[round(p.get_x(), 6)] = max(tops.get(round(p.get_x(), 6), 0), p.get_y() + p.get_height())
    counts, _ = np.histogram(marks["marks"], bins=10)
    assert np.allclose(sorted(tops.values()), sorted(counts / len(marks) * 100))


def test_histogram_cumulative_bin_width_and_steps(marks):
    ctx = draw(sdata.chart(marks).histogram(x="marks", cumulative=True))
    assert np.isclose(bar_heights(ctx.ax).max(), len(marks))
    assert ctx.ax.get_ylabel() == "cumulative count"
    ctx = draw(sdata.chart(marks).histogram(x="marks", bin_width=5))
    assert np.allclose([p.get_width() for p in ctx.ax.patches], 5)
    ctx = draw(sdata.chart(marks).histogram(x="marks", steps=True))
    outline = (ctx.ax.patches or ctx.ax.lines)[0]
    assert len(ctx.ax.patches) <= 1  # one step outline, not ten bars
    assert not getattr(outline, "get_fill", lambda: False)()


@pytest.mark.parametrize("call, message", [
    (lambda c: c.histogram(x="marks", stacked=True), "needs groups"),
    (lambda c: c.histogram(x="marks", bins=5, bin_width=5), "not both"),
    (lambda c: c.histogram(x="marks", bin_width=0), "bigger than 0"),
    (lambda c: c.histogram(x="marks", bin_width=0.001), "more than 1000"),
    (lambda c: c.histogram(x="marks", show="percnt"), 'Did you mean show="percent"'),
    (lambda c: c.histogram(x="marks", cumulative="yes"), "True or False"),
])
def test_histogram_wrong_options(marks, call, message):
    with pytest.raises(SimplydataError, match=message):
        call(sdata.chart(marks))


# ====================================================================== line_with_slope
def test_line_with_slope_through_a_point(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks").line_with_slope(1, through=(0, 0)))
    line = ctx.ax.lines[0]
    assert isinstance(line, matplotlib.lines.AxLine)
    assert line.get_slope() == 1 and tuple(line.get_xy1()) == (0, 0)
    assert line.get_linestyle() == "--"


def test_line_with_slope_intercept_and_label(marks):
    ctx = draw(sdata.chart(marks).dots(x="hours", y="marks", colour="class")
               .line_with_slope(5, intercept=40, label="guess", dashed=False, colour="red"))
    line = [ln for ln in ctx.ax.lines if isinstance(ln, matplotlib.lines.AxLine)][0]
    assert tuple(line.get_xy1()) == (0, 40) and line.get_slope() == 5
    assert mcolors.to_hex(line.get_color()) == "#ff0000" and line.get_linestyle() == "-"
    labels = [t.get_text() for t in ctx.ax.get_legend().get_texts()]
    assert "guess" in labels and "A" in labels  # added to the groups' legend


@pytest.mark.parametrize("call, message", [
    (lambda c: c.line_with_slope("one"), "number for the slope"),
    (lambda c: c.line_with_slope(1, through=(0, 0), intercept=3), "not both"),
    (lambda c: c.line_with_slope(1, through=5), "point"),
    (lambda c: c.line_with_slope(1, colour="bleu"), "colour"),
])
def test_line_with_slope_wrong_options(call, message):
    with pytest.raises(SimplydataError, match=message):
        call(sdata.chart())


def test_line_with_slope_on_a_log_scale_is_explained(marks):
    chart = sdata.chart(marks).dots(x="hours", y="marks").log_scale_y().line_with_slope(1)
    with pytest.raises(SimplydataError, match="log scale"):
        chart._render()


# ====================================================================== qq_dots
@pytest.mark.parametrize("engine", ENGINES)
def test_qq_dots_match_scipy_probplot(marks, engine):
    ctx = draw(sdata.chart(as_engine(marks, engine)).qq_dots("marks"))
    (osm, osr), (slope, intercept, r) = stats.probplot(marks["marks"].to_numpy(dtype=float), dist="norm")
    dots = ctx.ax.collections[0].get_offsets()
    assert np.allclose(dots[:, 0], osm) and np.allclose(dots[:, 1], osr)
    line = ctx.ax.lines[0]
    assert np.allclose(line.get_ydata(), intercept + slope * np.asarray(line.get_xdata()))
    assert np.allclose(line.get_xdata(), [osm[0], osm[-1]])
    assert ctx.ax.get_xlabel() == "normal quantiles" and ctx.ax.get_ylabel() == "marks"
    assert np.isclose(ctx.store["qq"][None]["r"], r)


def test_qq_dots_for_each_group(marks):
    ctx = draw(sdata.chart(marks).qq_dots("marks", for_each="class"))
    assert set(ctx.store["qq"]) == {"A", "B", "C"}
    for i, group in enumerate(["A", "B", "C"]):
        values = marks.loc[marks["class"] == group, "marks"].to_numpy(dtype=float)
        (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
        assert np.allclose(ctx.ax.collections[i].get_offsets()[:, 1], osr)
        assert np.isclose(ctx.store["qq"][group]["slope"], slope)
    assert [t.get_text() for t in ctx.ax.get_legend().get_texts()] == ["A", "B", "C"]


def test_qq_dots_leave_out_empty_cells_with_a_note(capsys):
    table = pd.DataFrame({"v": [1.0, 2.5, np.nan, 3.0, 4.2, 5.1]})
    chart = sdata.chart(table).qq_dots("v")
    chart._render()
    assert '1 row with empty "v" was left out' in capsys.readouterr().out
    ctx = draw(chart)
    assert len(ctx.ax.collections[0].get_offsets()) == 5


def test_qq_dots_wrong_input(marks):
    with pytest.raises(SimplydataError, match="needs a column"):
        sdata.chart(marks).qq_dots()
    with pytest.raises(SimplydataError, match='Did you mean "marks"'):
        sdata.chart(marks).qq_dots("mark")
    with pytest.raises(SimplydataError, match="needs numbers"):
        sdata.chart(marks).qq_dots("class")
    with pytest.raises(SimplydataError, match="at least 3"):
        sdata.chart(pd.DataFrame({"v": [1.0, 2.0]})).qq_dots("v")._render()


# ====================================================================== cumulative_share
@pytest.mark.parametrize("engine", ENGINES)
def test_cumulative_share_is_the_ecdf(marks, engine):
    ctx = draw(sdata.chart(as_engine(marks, engine)).cumulative_share(x="marks"))
    line = ctx.ax.lines[0]
    xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
    values = np.sort(marks["marks"].to_numpy(dtype=float))
    ecdf = stats.ecdf(values).cdf
    for v in np.unique(values):  # the top of each step is the share at or below that value
        assert np.isclose(ys[xs == v].max(), ecdf.evaluate(v))
    assert np.isclose(ys.max(), 1) and ctx.ax.get_ylabel() == "share at or below"


def test_cumulative_share_per_group(marks):
    ctx = draw(sdata.chart(marks).cumulative_share(x="marks", colour="class"))
    assert len(ctx.ax.lines) == 3
    assert all(np.isclose(np.nanmax(ln.get_ydata()), 1) for ln in ctx.ax.lines)


def test_cumulative_share_wrong_input(marks):
    with pytest.raises(SimplydataError, match="needs x="):
        sdata.chart(marks).cumulative_share()
    with pytest.raises(SimplydataError, match="needs numbers"):
        sdata.chart(marks).cumulative_share(x="class")


# ====================================================================== font_name
def test_font_name_is_used_for_all_text(marks):
    chart = sdata.chart(marks).dots(x="hours", y="marks").title_top("Marks").font_name("dejavu serif")
    assert chart._settings["font_name"] == "DejaVu Serif"  # the installed font's own spelling
    fig = chart._render()
    ax = fig.axes[0]
    assert ax.title.get_fontname() == "DejaVu Serif"
    assert ax.xaxis.label.get_fontname() == "DejaVu Serif"


def test_font_name_kinds_of_font(marks):
    fig = sdata.chart(marks).dots(x="hours", y="marks").font_name("monospace")._render()
    assert "Mono" in fig.axes[0].xaxis.label.get_fontname()


def test_font_name_missing_lists_close_fonts():
    with pytest.raises(SimplydataError, match='similar name: .*"DejaVu Sans"'):
        sdata.chart().font_name("DejaVu Sanz")
    with pytest.raises(SimplydataError, match="name of a font"):
        sdata.chart().font_name(12)


# ====================================================================== more combinations
def test_ticks_how_many_on_dates_and_every_on_a_log_scale(shop, marks):
    ctx = draw(sdata.chart(shop).lines(x="date", y="sales").ticks_x(how_many=4))
    assert isinstance(ctx.ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
    assert len(visible_ticks(ctx.ax)) <= 4
    with pytest.raises(SimplydataError, match="log scale"):
        sdata.chart(marks).dots(x="hours", y="marks").log_scale_y().ticks_y(every=10)._render()


def test_colour_scale_works_on_every_panel(weather):
    chart = (sdata.chart(weather).heatmap(x="month", y="city", value="rain")
             .colour_scale("magma", between=[0, 200]).one_chart_per("city"))
    fig = chart._render()
    meshes = [ax.collections[0] for ax in fig.axes if ax.get_title().startswith("city =")]
    assert len(meshes) == 3
    assert all(m.get_cmap().name == "magma" and (m.norm.vmin, m.norm.vmax) == (0, 200) for m in meshes)


def test_docstring_examples_draw_and_save(tmp_path, marks, weather):
    chart = sdata.chart(weather).heatmap(x="month", y="city", value="rain")
    chart.colour_scale("magma")
    chart.save(tmp_path / "a.png")
    chart = sdata.chart(marks).dots(x="hours", y="marks")
    chart.line_with_slope(1, through=(0, 0))
    chart.tick_format_x("thousands")
    chart.tick_format_y("percent")
    chart.ticks_x(every=2)
    chart.font_name("DejaVu Serif")
    chart.save(tmp_path / "b.png")
    sdata.chart(marks).qq_dots("marks").save(tmp_path / "c.png")
    sdata.chart(marks).cumulative_share(x="marks", colour="class").save(tmp_path / "d.png")
    assert all((tmp_path / f"{n}.png").stat().st_size > 0 for n in "abcd")
