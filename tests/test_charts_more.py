"""More chart words: order_x, order_y, order_legend, calendar order, label_legend, band,
names_on_dots. Each one is checked against plain pandas, on pandas and on Polars tables."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

import simplydata as sdata
from simplydata import SimplydataError
from simplydata._style import COLOURS
from simplydata.words.charts import calendar_order

pl = pytest.importorskip("polars")

ENGINES = ["pandas", "polars"]


def as_engine(frame, engine):
    return pl.from_pandas(frame) if engine == "polars" else frame


def draw(chart):
    fig = Figure()
    ax = fig.add_subplot(projection=chart._settings["projection"])
    return chart._draw_panel(fig, ax)


def xnames(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def ynames(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def bar_heights(ax):
    return [p.get_height() for p in ax.patches if p.get_height() == p.get_height()]


@pytest.fixture
def marks():
    return sdata.example("marks")


@pytest.fixture
def weather():
    return sdata.example("weather")


@pytest.fixture
def ranges():
    return pd.DataFrame({"day": [1, 2, 3, 4, 1, 2, 3, 4],
                         "site": ["N"] * 4 + ["S"] * 4,
                         "low": [1.0, 2, 1, 3, 5, 6, 4, 5],
                         "high": [4.0, 5, 6, 6, 9, 8, 9, 7]})


# ---------------------------------------------------------------- calendar order
def test_calendar_order_months_and_weekdays():
    assert calendar_order(["Mar", "Jan", "Feb", "Jan"]) == ["Jan", "Feb", "Mar"]
    assert calendar_order(["December", "april", "Sept"]) == ["april", "Sept", "December"]
    assert calendar_order(["Sun", "Mon", "Wed", "Tues"]) == ["Mon", "Tues", "Wed", "Sun"]
    assert calendar_order(["Mon", "Jan"]) is None          # months and weekdays mixed
    assert calendar_order(["Jan", "apple"]) is None        # not every value is a month
    assert calendar_order(["Ma", "Ju"]) is None            # too short to be sure
    assert calendar_order(["Jan"]) is None                 # one value: nothing to order


@pytest.mark.parametrize("engine", ENGINES)
def test_months_are_in_calendar_order_by_themselves(weather, engine):
    shuffled = weather.sample(frac=1, random_state=4)
    ax = draw(sdata.chart(as_engine(shuffled, engine)).bars(x="month", y="rain")).ax
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert xnames(ax) == months
    expected = shuffled.groupby("month")["rain"].mean().reindex(months)
    assert np.allclose(bar_heights(ax), expected)


def test_weekdays_in_order_for_colour_groups_too():
    t = pd.DataFrame({"day": ["Wed", "Mon", "Tue", "Mon", "Wed"], "n": [1, 2, 3, 4, 5]})
    ax = draw(sdata.chart(t).count_bars(x="n", colour="day")).ax
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Mon", "Tue", "Wed"]


def test_category_column_keeps_its_order():
    t = pd.DataFrame({"size": pd.Categorical(["M", "S", "L", "S"], categories=["S", "M", "L"]),
                      "v": [2.0, 1, 3, 1]})
    ax = draw(sdata.chart(t).bars(x="size", y="v")).ax
    assert xnames(ax) == ["S", "M", "L"]


# ---------------------------------------------------------------- order_x / order_y
@pytest.mark.parametrize("engine", ENGINES)
def test_order_x_puts_bars_in_order(marks, engine):
    chart = sdata.chart(as_engine(marks, engine)).bars(x="class", y="marks").order_x("C", "A", "B")
    ax = draw(chart).ax
    assert xnames(ax) == ["C", "A", "B"]
    expected = marks.groupby("class")["marks"].mean().reindex(["C", "A", "B"])
    assert np.allclose(bar_heights(ax), expected)


def test_order_x_before_the_chart_type_and_as_a_list(marks):
    chart = sdata.chart(marks).order_x(["B", "C"]).boxes(x="class", y="marks")
    ax = draw(chart).ax
    assert xnames(ax) == ["B", "C", "A"]  # groups left out come last


@pytest.mark.parametrize("engine", ENGINES)
def test_order_x_biggest_first_uses_what_the_bars_show(marks, engine):
    chart = sdata.chart(as_engine(marks, engine)).bars(x="name", y="marks", show="median")
    ax = draw(chart.order_x(biggest_first=True)).ax
    expected = marks.groupby("name")["marks"].median().sort_values(ascending=False, kind="stable")
    assert xnames(ax) == list(expected.index)
    assert np.allclose(bar_heights(ax), expected)
    ax = draw(chart.order_x(biggest_first=False)).ax
    assert xnames(ax) == list(expected.index[::-1]) or np.all(np.diff(bar_heights(ax)) >= 0)


def test_order_x_biggest_first_on_count_bars_and_boxes(marks):
    fruit = sdata.example("fruit")
    ax = draw(sdata.chart(fruit).count_bars(x="fruit").order_x(biggest_first=True)).ax
    counts = fruit["fruit"].value_counts()
    assert [int(h) for h in bar_heights(ax)] == sorted(counts, reverse=True)
    ax = draw(sdata.chart(marks).boxes(x="class", y="marks").order_x(biggest_first=True)).ax
    medians = marks.groupby("class")["marks"].median().sort_values(ascending=False, kind="stable")
    assert xnames(ax) == list(medians.index)


def test_order_y_on_sideways_bars_first_is_at_the_top(marks):
    ax = draw(sdata.chart(marks).bars(x="class", y="marks", sideways=True).order_y("B", "A", "C")).ax
    assert ynames(ax) == ["B", "A", "C"]
    assert ax.yaxis_inverted()  # position 0 (the first name) is at the top
    widths = [p.get_width() for p in ax.patches]
    assert np.allclose(widths, marks.groupby("class")["marks"].mean().reindex(["B", "A", "C"]))


def test_order_x_stacked_bars_and_lines(marks):
    chart = sdata.chart(marks).bars(x="class", y="marks", colour="subject", stacked=True, show="total")
    ax = draw(chart.order_x("C", "B", "A")).ax
    assert xnames(ax) == ["C", "B", "A"]
    t = pd.DataFrame({"level": ["High", "Low", "Medium", "Low"], "v": [3.0, 1, 2, 1.5]})
    ax = draw(sdata.chart(t).lines(x="level", y="v").order_x("Low", "Medium", "High")).ax
    assert xnames(ax) == ["Low", "Medium", "High"]
    assert np.allclose(ax.lines[0].get_ydata(), [1.25, 2, 3])


def test_order_x_with_one_chart_per_and_lists(marks):
    chart = sdata.chart(marks).bars(x="class", y="marks").order_x("C", "B", "A").one_chart_per("subject")
    fig = chart._render()
    for ax in fig.axes:
        assert xnames(ax) == ["C", "B", "A"]
    ax = draw(sdata.chart().bars(x=["b", "a", "c"], y=[2, 1, 3]).order_x("a", "b", "c")).ax
    assert xnames(ax) == ["a", "b", "c"]
    assert np.allclose(bar_heights(ax), [1, 2, 3])


def test_second_y_list_still_meets_its_row_after_ordering():
    t = pd.DataFrame({"month": ["Mar", "Jan", "Feb"], "temp": [9.0, 3, 5]})
    chart = sdata.chart(t).bars(x="month", y="temp").second_y([30, 10, 20], label="rain")
    fig = chart._render()
    twin = [a for a in fig.axes if a is not fig.axes[0]][0]
    line = twin.lines[0]
    # Jan, Feb, Mar on the axis: rain 10, 20, 30 above them
    assert xnames(fig.axes[0]) == ["Jan", "Feb", "Mar"]
    assert list(line.get_ydata()) == [10, 20, 30]


def test_order_words_errors(marks):
    chart = sdata.chart(marks).bars(x="class", y="marks")
    with pytest.raises(SimplydataError, match='Did you mean "B"'):
        chart.order_x("A", "b")
    with pytest.raises(SimplydataError, match="needs the names"):
        chart.order_x()
    with pytest.raises(SimplydataError, match="not both"):
        chart.order_x("A", biggest_first=True)
    with pytest.raises(SimplydataError, match="twice"):
        chart.order_x("A", "A")
    with pytest.raises(SimplydataError, match="True or False"):
        chart.order_x(biggest_first="yes")
    with pytest.raises(SimplydataError, match="holds numbers"):
        chart.order_y("A")
    with pytest.raises(SimplydataError, match="holds numbers"):
        sdata.chart(marks).dots(x="hours", y="marks").order_x("A")
    with pytest.raises(SimplydataError, match="no size for a group"):
        sdata.chart(marks).dots(x="class", y="marks").order_x(biggest_first=True)._render()
    with pytest.raises(SimplydataError, match="option called"):
        chart.order_x("A", biggest=True)
    # a name not in the column is also caught when the chart type comes later
    with pytest.raises(SimplydataError, match="no such value"):
        sdata.chart(marks).order_x("D").bars(x="class", y="marks")._render()


# ---------------------------------------------------------------- order_legend
@pytest.mark.parametrize("engine", ENGINES)
def test_order_legend_orders_groups_and_colours(marks, engine):
    chart = sdata.chart(as_engine(marks, engine)).bars(x="class", y="marks", colour="subject")
    ax = draw(chart.order_legend("Science", "English")).ax
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Science", "English", "Maths"]
    first = ax.containers[0]  # the first group's bars get the first colour
    assert np.allclose(first.patches[0].get_facecolor()[:3], matplotlib_rgb(COLOURS[0]))
    expected = marks[marks["subject"] == "Science"].groupby("class")["marks"].mean()
    assert np.allclose(list(first.datavalues), expected.reindex(xnames(ax)))


def matplotlib_rgb(code):
    from matplotlib.colors import to_rgb
    return to_rgb(code)


def test_order_legend_on_a_pie_and_errors():
    fruit = sdata.example("fruit")
    ctx = draw(sdata.chart(fruit).pie(labels="fruit", values="count").order_legend("pear", "apple"))
    assert list(ctx.store["pie_sizes"].index[:2]) == ["pear", "apple"]
    assert np.isclose(ctx.store["pie_sizes"]["pear"], fruit.loc[fruit["fruit"] == "pear", "count"].sum())
    with pytest.raises(SimplydataError, match="needs the names"):
        sdata.chart(fruit).order_legend()
    with pytest.raises(SimplydataError, match="option called"):
        sdata.chart(fruit).order_legend("pear", biggest_first=True)


# ---------------------------------------------------------------- label_legend
@pytest.mark.parametrize("engine", ENGINES)
def test_label_legend_names_the_legend(marks, engine):
    fig = sdata.chart(as_engine(marks, engine)).dots(x="hours", y="marks", colour="class") \
        .label_legend("Class")._render()
    assert fig.axes[0].get_legend().get_title().get_text() == "Class"


def test_label_legend_with_legend_places_and_panels(marks):
    for order in ("before", "after"):
        chart = sdata.chart(marks).lines(x="hours", y="marks", colour="class")
        if order == "before":
            chart.label_legend("Class").legend_bottom()
        else:
            chart.legend_bottom().label_legend("Class")
        fig = chart._render()
        assert fig.legends[0].get_title().get_text() == "Class"
    fig = sdata.chart(marks).dots(x="hours", y="marks", colour="class").one_chart_per("subject") \
        .label_legend("Group")._render()
    titles = {ax.get_legend().get_title().get_text() for ax in fig.axes if ax.get_legend()}
    assert titles == {"Group"}
    fig = sdata.chart(marks).dots(x="hours", y="marks", colour="class").label_legend("")._render()
    assert fig.axes[0].get_legend().get_title().get_text() == ""


def test_label_legend_on_a_page_and_with_hidden_legend(marks):
    a = sdata.chart(marks).bars(x="class", y="marks", colour="subject").label_legend("Subject")
    b = sdata.chart(marks).dots(x="hours", y="marks")
    fig = sdata.side_by_side(a, b, letters=True)._render()
    titles = [ax.get_legend().get_title().get_text() for ax in fig.axes if ax.get_legend()]
    assert titles == ["Subject"]
    sdata.chart(marks).dots(x="hours", y="marks", colour="class").hide_legend().label_legend("x")._render()


def test_label_legend_errors(marks):
    with pytest.raises(SimplydataError, match="no legend"):
        sdata.chart(marks).dots(x="hours", y="marks").label_legend("Class")._render()
    with pytest.raises(SimplydataError, match="needs some words"):
        sdata.chart(marks).label_legend(None)


# ---------------------------------------------------------------- band
def band_edges(collection):
    """Bottom and top edge of a fill_between polygon, as {x: (low, high)}."""
    verts = collection.get_paths()[0].vertices
    edges = {}
    for x, y in verts:
        lo, hi = edges.get(x, (np.inf, -np.inf))
        edges[x] = (min(lo, y), max(hi, y))
    return edges


@pytest.mark.parametrize("engine", ENGINES)
def test_band_follows_the_low_and_high_columns(ranges, engine):
    one = ranges[ranges["site"] == "N"].sample(frac=1, random_state=1)  # x not sorted in the table
    ax = draw(sdata.chart(as_engine(one, engine)).band(x="day", between=["low", "high"])).ax
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 1
    edges = band_edges(polys[0])
    for _, row in one.iterrows():
        assert edges[row["day"]] == (row["low"], row["high"])
    assert ax.get_xlabel() == "day"


def test_band_per_group_with_lines_and_legend(ranges):
    chart = sdata.chart(ranges).band(x="day", between=["low", "high"], colour="site")
    chart.lines(x="day", y="high", colour="site")
    fig = chart._render()
    ax = fig.axes[0]
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 2
    south = band_edges(polys[1])
    assert south[3.0] == (4.0, 9.0)
    assert np.allclose(polys[0].get_facecolor()[0][:3], matplotlib_rgb(COLOURS[0]))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["N", "S"]  # one entry per group, not two


def test_band_label_joins_the_legend(ranges):
    one = ranges[ranges["site"] == "N"]
    chart = sdata.chart(one).band(x="day", between=["low", "high"], label="range").lines(x="day", y="low")
    ax = chart._render().axes[0]
    assert "range" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_band_gap_for_empty_cells_and_lists():
    t = pd.DataFrame({"x": [1, 2, 3, 4], "lo": [1.0, np.nan, 1, 1], "hi": [2.0, 2, 2, 2]})
    ax = draw(sdata.chart(t).band(x="x", between=["lo", "hi"])).ax
    poly = [c for c in ax.collections if isinstance(c, PolyCollection)][0]
    assert len(poly.get_paths()) == 2  # the empty cell splits the band in two
    ax = draw(sdata.chart().band(x=[1, 2, 3], between=[[0, 1, 0], [2, 3, 2]])).ax
    assert band_edges([c for c in ax.collections if isinstance(c, PolyCollection)][0])[2.0] == (1.0, 3.0)


def test_band_errors(ranges, marks):
    with pytest.raises(SimplydataError, match="needs x= and between="):
        sdata.chart(ranges).band(x="day")
    with pytest.raises(SimplydataError, match="two columns"):
        sdata.chart(ranges).band(x="day", between="low")
    with pytest.raises(SimplydataError, match="swap"):
        sdata.chart(ranges).band(x="day", between=["high", "low"], colour="site")
    with pytest.raises(SimplydataError, match="more than one row"):
        sdata.chart(ranges).band(x="day", between=["low", "high"])._render()
    with pytest.raises(SimplydataError, match='Did you mean "high"'):
        sdata.chart(ranges).band(x="day", between=["low", "hihg"])
    with pytest.raises(SimplydataError, match="needs numbers"):
        sdata.chart(ranges).band(x="day", between=["site", "high"])
    with pytest.raises(SimplydataError, match="two different"):
        sdata.chart(ranges).band(x="day", between=["low", "low"])
    with pytest.raises(SimplydataError, match="label= should be words"):
        sdata.chart(ranges).band(x="day", between=["low", "high"], label=3)


# ---------------------------------------------------------------- names_on_dots
def label_boxes(fig, ax):
    renderer = fig._get_renderer()
    return [t.get_window_extent(renderer) for t in ax.texts]


@pytest.mark.parametrize("engine", ENGINES)
def test_names_on_dots_writes_each_name_at_its_dot(engine):
    t = pd.DataFrame({"x": [1.0, 5, 9, 3], "y": [1.0, 9, 2, 6], "who": ["Ann", "Bo", "Cy", "Di"]})
    chart = sdata.chart(as_engine(t, engine)).dots(x="x", y="y").names_on_dots("who")
    fig = chart._render()
    ax = fig.axes[0]
    found = {a.get_text(): a.xy for a in ax.texts}
    assert found == {r.who: (r.x, r.y) for r in t.itertuples()}


def test_names_on_dots_do_not_cover_each_other(marks, capsys):
    chart = sdata.chart(marks).dots(x="hours", y="marks").names_on_dots("name")
    fig = chart._render()
    ax = fig.axes[0]
    boxes = label_boxes(fig, ax)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b) or a.intersection(a, b).width * a.intersection(a, b).height < 4
    shown = len(ax.texts)
    out = capsys.readouterr().out
    if shown < len(marks):
        assert f"{len(marks) - shown} names were left out" in out


def test_names_on_dots_list_follows_calendar_order_and_errors(marks):
    t = pd.DataFrame({"m": ["Mar", "Jan", "Feb"], "v": [3.0, 1, 2]})
    ax = draw(sdata.chart(t).dots(x="m", y="v").names_on_dots(["c", "a", "b"])).ax
    texts = {a.get_text(): a.xy[1] for a in ax.texts}
    assert texts == {"a": 1.0, "b": 2.0, "c": 3.0}
    with pytest.raises(SimplydataError, match='Did you mean "name"'):
        sdata.chart(marks).dots(x="hours", y="marks").names_on_dots("nmae")
    with pytest.raises(SimplydataError, match="last chart type is bars"):
        sdata.chart(marks).bars(x="name", y="marks").names_on_dots("name")
    with pytest.raises(SimplydataError, match="no dots yet"):
        sdata.chart(marks).names_on_dots("name")._render()
    with pytest.raises(SimplydataError, match="one name for each dot"):
        sdata.chart(t).dots(x="m", y="v").names_on_dots(["a"])._render()
    with pytest.raises(SimplydataError, match="does not know the colour"):
        sdata.chart(t).dots(x="m", y="v").names_on_dots("m", colour="bleu")


def test_new_words_work_with_styles_and_saving(tmp_path, marks):
    chart = sdata.chart(marks[marks["subject"] == "Maths"]).dots(x="hours", y="marks", colour="class")
    chart.names_on_dots("name").label_legend("Class").order_legend("C", "B", "A").style("dark")
    chart.save(tmp_path / "x.png")
    assert (tmp_path / "x.png").stat().st_size > 1000


# ---------------------------------------------------------------- panel and frame order
def _panel_titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]


@pytest.mark.parametrize("engine", ENGINES)
def test_panels_and_frames_follow_calendar_order(engine):
    t = pd.DataFrame({"month": ["Mar", "Jan", "Feb", "Mar", "Jan", "Feb"], "x": [1, 2, 3, 4, 5, 6],
                      "y": [1.0, 2, 3, 4, 5, 6]})
    chart = sdata.chart(as_engine(t, engine)).dots(x="x", y="y").one_chart_per("month")
    assert _panel_titles(chart._render()) == ["month = Jan", "month = Feb", "month = Mar"]
    chart = sdata.chart(as_engine(t, engine)).dots(x="x", y="y").one_chart_per(down="month", across="month")
    titles = [ax.get_title() for ax in chart._render().axes]
    assert titles[0] == "month = Jan, month = Jan" and titles[-1] == "month = Mar, month = Mar"
    frames = sdata.chart(as_engine(t, engine)).dots(x="x", y="y").animate(over="month")._frames()
    assert [value for value, _ in frames] == ["Jan", "Feb", "Mar"]  # not A to Z (Feb, Jan, Mar)


def test_panels_follow_order_x_and_category_order(marks):
    chart = sdata.chart(marks).bars(x="class", y="marks").order_x("C", "A", "B").one_chart_per("class")
    assert _panel_titles(chart._render()) == ["class = C", "class = A", "class = B"]
    chart = sdata.chart(marks).bars(x="class", y="marks").order_x("B", "C", "A").animate(over="class")
    assert [value for value, _ in chart._frames()] == ["B", "C", "A"]
    chart = (sdata.chart(marks).bars(x="name", y="marks", colour="subject")
             .order_legend("Science", "Maths").one_chart_per("subject"))
    titles = _panel_titles(chart._render())
    assert titles[:2] == ["subject = Science", "subject = Maths"]
    cat = marks.assign(level=pd.Categorical(np.where(marks["marks"] > 60, "high", "low"), categories=["low", "high"]))
    chart = sdata.chart(cat).dots(x="height", y="marks").one_chart_per("level")
    assert _panel_titles(chart._render()) == ["level = low", "level = high"]


def test_panels_of_numbers_go_from_small_to_big():
    t = pd.DataFrame({"year": [2003, 2001, 2002], "x": [1, 2, 3], "y": [1.0, 2, 3]})
    fig = sdata.chart(t).dots(x="x", y="y").one_chart_per("year")._render()
    assert _panel_titles(fig) == ["year = 2001", "year = 2002", "year = 2003"]


# ---------------------------------------------------------------- empty cells on charts
def test_chart_says_once_which_rows_empty_cells_left_out(capsys):
    shop = sdata.example("shop")
    n_sales = int(shop["sales"].isna().sum())
    sdata.chart(shop).lines(x="date", y="sales", colour="shop").one_chart_per("shop")._render()
    out = capsys.readouterr().out
    assert out.count("left out of this chart") == 1  # once, not once per panel
    assert f'Note: {n_sales} rows with empty "sales" were left out of this chart.' in out
    both = int(shop[["customers", "sales"]].isna().any(axis=1).sum())
    sdata.chart(shop).dots(x="customers", y="sales").animate(over="shop")._render()
    out = capsys.readouterr().out
    assert out.count("left out") == 1
    assert f'Note: {both} rows with empty "customers" or "sales" were left out of this chart.' in out
    sdata.chart(sdata.example("marks")).bars(x="class", y="marks")._render()
    assert "left out" not in capsys.readouterr().out  # nothing empty, nothing said


def test_empty_cells_chart_marks_every_empty_cell(capsys):
    shop = sdata.example("shop")
    chart = sdata.chart(shop).empty_cells()
    ax = chart._render().axes[0]
    grid = ax.images[0].get_array()
    assert grid.shape == (len(shop), 4)
    assert np.array_equal(np.asarray(grid, dtype=bool), shop.isna().to_numpy())
    assert xnames(ax) == [f"{c}\n{int(shop[c].isna().sum())} empty" for c in shop.columns]
    assert "left out" not in capsys.readouterr().out  # this chart shows empty cells on purpose
    ax = sdata.chart(pd.DataFrame({"a": ["x", "  ", None]})).empty_cells(columns="a")._render().axes[0]
    assert list(np.asarray(ax.images[0].get_array(), dtype=bool).ravel()) == [False, True, True]
    with pytest.raises(SimplydataError, match='Did you mean "sales"'):
        sdata.chart(shop).empty_cells(columns=["sale"])
    with pytest.raises(SimplydataError, match="needs a table"):
        sdata.chart().empty_cells()
