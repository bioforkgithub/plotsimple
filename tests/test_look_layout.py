"""Tests for colours, legends, the grid and the several-charts words."""

import numpy as np
import pytest
from matplotlib.colors import to_hex
from PIL import Image

import plotsimple as ps
from plotsimple import PlotsimpleError


def _axes(fig):
    """All plotting panels of a figure or sub-figure (the root figure also lists its sub-figures' panels)."""
    return list(fig.axes)


def _bbox(artist, fig):
    return artist.get_window_extent(fig.canvas.get_renderer())


def _draw(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig).draw()


# ---------------------------------------------------------------- colours
def test_colour_applies(marks):
    fig = ps.chart(marks).dots(x="height", y="marks").colour("red")._render()
    face = fig.axes[0].collections[0].get_facecolor()[0]
    assert to_hex(face) == "#ff0000"


def test_color_spelling_and_option(marks):
    chart = ps.chart(marks).color("green")
    assert chart._settings["colour"] == "green"


def test_colours_apply_to_groups(marks):
    fig = ps.chart(marks).bars(x="name", y="marks", colour="class").colours("crimson", "teal")._render()
    faces = {to_hex(p.get_facecolor()) for p in fig.axes[0].patches if p.get_height() > 0}
    assert faces == {to_hex("crimson"), to_hex("teal")}


def test_colours_accepts_a_list(marks):
    assert ps.chart(marks).colors(["red", "#00ff00"])._settings["colours"] == ["red", "#00ff00"]


def test_misspelt_colour_suggests(marks):
    with pytest.raises(PlotsimpleError, match='Did you mean "blue"'):
        ps.chart(marks).colour("bleu")
    with pytest.raises(PlotsimpleError, match="purple"):
        ps.chart(marks).colours("red", "purpel")


def test_colours_needs_one(marks):
    with pytest.raises(PlotsimpleError, match="at least one"):
        ps.chart(marks).colours()


# ---------------------------------------------------------------- grid
def test_show_and_hide_grid(marks):
    shown = ps.chart(marks).lines(x="test", y="marks").show_grid()._render().axes[0]
    assert any(line.get_visible() for line in shown.xaxis.get_gridlines())
    hidden = ps.chart(marks).lines(x="test", y="marks").hide_grid()._render().axes[0]
    assert not any(line.get_visible() for line in hidden.yaxis.get_gridlines())


# ---------------------------------------------------------------- legends
@pytest.mark.parametrize("where", ["top", "bottom", "left", "right"])
def test_legend_outside_plot(marks, where):
    chart = ps.chart(marks).lines(x="test", y="marks", colour="name").title_top("T").label_x("Test")
    getattr(chart, f"legend_{where}")()
    fig = chart._render()
    _draw(fig)
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert len(fig.legends) == 1
    leg, plot = _bbox(fig.legends[0], fig), _bbox(ax, fig)
    assert not leg.overlaps(plot)
    side = {"top": leg.y0 >= plot.y1, "bottom": leg.y1 <= plot.y0,
            "left": leg.x1 <= plot.x0, "right": leg.x0 >= plot.x1}
    assert side[where]
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert {"Asha", "Ben", "Chen", "Dina"} <= set(labels)


def test_legend_bottom_is_below_x_label_and_one_row(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="name").label_x("Test").legend_bottom()._render()
    _draw(fig)
    leg = fig.legends[0]
    assert _bbox(leg, fig).y1 <= _bbox(fig.axes[0].xaxis.label, fig).y0
    ys = {round(t.get_window_extent().y0) for t in leg.get_texts()}
    assert len(ys) == 1


def test_legend_top_keeps_title_above(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="name").title_top("Marks").legend_top()._render()
    assert fig.legends[0].get_title().get_text() == "Marks"


def test_one_legend_for_all_panels_without_overlap(marks):
    chart = (ps.chart(marks).bars(x="name", y="marks", colour="class").one_chart_per("test")
             .title_top("Big title").label_x("Student").label_y("Marks"))
    for where in ("top", "bottom", "left", "right"):
        getattr(chart, f"legend_{where}")()
        fig = chart._render()
        _draw(fig)
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert len(fig.legends) == 1
        leg = _bbox(fig.legends[0], fig)
        for ax in fig.axes:
            assert not leg.overlaps(_bbox(ax, fig))
        texts = [t for t in (fig._suptitle, fig._supxlabel, fig._supylabel) if t is not None and t.get_visible()]
        for t in texts:
            assert not leg.overlaps(_bbox(t, fig)), where


def test_hide_legend(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="name").hide_legend()._render()
    assert fig.axes[0].get_legend() is None and not fig.legends


def test_default_legend_kept(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="name")._render()
    assert fig.axes[0].get_legend() is not None and not fig.legends


def test_last_legend_word_wins(marks):
    fig = ps.chart(marks).lines(x="test", y="marks", colour="name").legend_top().hide_legend()._render()
    assert not fig.legends


def test_legend_takes_twin_axis_entries(marks):
    from plotsimple._core import PHASES

    def add_twin(ctx):
        twin = ctx.ax.twinx()
        twin.plot([1, 2, 3], [5, 6, 7], label="rain")
        twin.legend()
        ctx.store["second_y"] = twin

    chart = ps.chart(marks).lines(x="test", y="marks", colour="class").legend_right()
    chart._add("second_y", "extra", add_twin)
    fig = chart._render()
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "rain" in labels and "A" in labels
    assert PHASES["finish"] > PHASES["extra"]


# ---------------------------------------------------------------- colour bar label
def test_label_colour_bar(marks):
    def fake_heatmap(ctx):
        image = ctx.ax.imshow(np.arange(9).reshape(3, 3))
        ctx.store["colour_bar"] = ctx.fig.colorbar(image, ax=ctx.ax)

    chart = ps.chart(marks).label_colour_bar("Depth (m)")
    chart._add("heatmap", "draw", fake_heatmap)
    fig = chart._render()
    assert any(ax.get_ylabel() == "Depth (m)" for ax in fig.axes)


def test_label_colour_bar_without_bar(marks):
    with pytest.raises(PlotsimpleError, match="colour bar"):
        ps.chart(marks).bars(x="name", y="marks").label_colour_bar("x")._render()


# ---------------------------------------------------------------- one_per
def test_one_chart_per_panels_and_titles(marks):
    fig = ps.chart(marks).bars(x="name", y="marks").one_chart_per("test")._render()
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert [ax.get_title() for ax in visible] == ["test = 1", "test = 2", "test = 3"]


def test_one_chart_per_columns_and_size(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").one_chart_per("name", per_row=2)
    fig = chart._render()
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    assert chart._settings["height"] > 6  # two rows: grew taller


def test_one_chart_per_keeps_user_size(marks):
    chart = ps.chart(marks).size(5, 4).bars(x="name", y="marks").one_chart_per("test")
    assert (chart._settings["width"], chart._settings["height"]) == (5.0, 4.0)


def test_one_chart_per_row_column(marks):
    fig = ps.chart(marks).dots(x="height", y="marks").one_chart_per(down="class", across="test")._render()
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title() == "class = A, test = 1"


def test_one_chart_per_errors(marks):
    chart = ps.chart(marks)
    with pytest.raises(PlotsimpleError, match="needs a column"):
        chart.one_chart_per()
    with pytest.raises(PlotsimpleError, match='Did you mean "class"'):
        chart.one_chart_per("clas")
    with pytest.raises(PlotsimpleError, match="whole number"):
        chart.one_chart_per("class", per_row=0)
    with pytest.raises(PlotsimpleError, match="not both"):
        chart.one_chart_per("class", down="test")
    with pytest.raises(PlotsimpleError, match="True or"):
        chart.one_chart_per("class", same_y="yes")


# ---------------------------------------------------------------- pages
def test_side_by_side(marks, tmp_path):
    a = ps.chart(marks).bars(x="name", y="marks").title_top("Bars")
    b = ps.chart(marks).lines(x="test", y="marks").title_top("Lines").label_y("Marks")
    page = ps.side_by_side(a, b).title_top("Both")
    fig = page._render()
    axes = _axes(fig)
    assert [ax.get_title() for ax in axes] == ["Bars", "Lines"]
    assert axes[1].get_ylabel() == "Marks"
    assert fig.get_suptitle() == "Both"
    _draw(fig)
    left, right = (_bbox(ax, fig) for ax in axes)
    assert left.x1 < right.x0
    page.save(tmp_path / "both.png")
    assert (tmp_path / "both.png").stat().st_size > 0
    assert page._repr_png_()[:4] == b"\x89PNG"


def test_one_above_other_and_same_x(marks):
    a = ps.chart(marks).lines(x="test", y="marks")
    b = ps.chart(marks[marks["test"] < 3]).lines(x="test", y="height")
    fig = ps.one_above_other(a, b, same_x=True)._render()
    top, bottom = _axes(fig)
    _draw(fig)
    assert _bbox(top, fig).y0 > _bbox(bottom, fig).y1
    assert top.get_xlim() == bottom.get_xlim()


def test_grid_same_y_and_legends_stay_with_their_chart(marks):
    charts = [ps.chart(marks).bars(x="name", y="marks", colour="class").legend_bottom(),
              ps.chart(marks).bars(x="class", y="height"),
              ps.chart(marks).dots(x="height", y="marks"),
              ps.chart(marks).bars(x="name", y="marks").one_chart_per("test")]
    page = ps.grid(*charts, per_row=2, same_y=True)
    fig = page._render()
    assert len(fig.subfigs) == 4
    assert len(fig.subfigs[0].legends) == 1 and not fig.subfigs[1].legends
    assert len({ax.get_ylim() for ax in _axes(fig)}) == 1
    assert len(_axes(fig.subfigs[3])) == 3


def test_page_style_and_size(marks):
    a = ps.chart(marks).bars(x="name", y="marks")
    page = ps.side_by_side(a, a).size(10, 4).style("dark")
    fig = page._render()
    assert tuple(fig.get_size_inches()) == (10, 4)
    assert to_hex(_axes(fig)[0].get_facecolor()) == "#1e1e1e"


def test_page_errors(marks):
    a = ps.chart(marks).bars(x="name", y="marks")
    with pytest.raises(PlotsimpleError, match="needs some charts"):
        ps.side_by_side()
    with pytest.raises(PlotsimpleError, match="ps.chart"):
        ps.grid(a, "b")
    with pytest.raises(PlotsimpleError, match="already a page"):
        ps.grid(ps.side_by_side(a, a), a)
    with pytest.raises(PlotsimpleError, match="whole number"):
        ps.grid(a, a, per_row=0)
    with pytest.raises(PlotsimpleError, match="title_top"):
        ps.side_by_side(a, a).title_topp("x")


def test_page_accepts_a_list(marks):
    a = ps.chart(marks).bars(x="name", y="marks")
    assert len(_axes(ps.side_by_side([a, a])._render())) == 2


# ---------------------------------------------------------------- zoom box
def test_zoom_box(marks):
    fig = ps.chart(marks).lines(x="height", y="marks").zoom_box(x_between=[145, 155], y_between=[40, 80])._render()
    ax = fig.axes[0]
    inset = ax.child_axes[0]
    assert inset.get_xlim() == (145, 155) and inset.get_ylim() == (40, 80)
    assert len(inset.lines) == 1  # the chart's line, drawn again
    assert inset.get_legend() is None


def test_zoom_box_errors(marks):
    chart = ps.chart(marks)
    with pytest.raises(PlotsimpleError, match="two different numbers"):
        chart.zoom_box(x_between=[2, 2], y_between=[0, 1])
    with pytest.raises(PlotsimpleError, match='where="top left"'):
        chart.zoom_box(x_between=[1, 2], y_between=[0, 1], where="top lft")
    with pytest.raises(PlotsimpleError, match="numbers in x_between"):
        chart.zoom_box(x_between=["a", 2], y_between=[0, 1])
    with pytest.raises(PlotsimpleError, match="two numbers"):
        chart.zoom_box(x_between=2, y_between=[0, 1])
    with pytest.raises(PlotsimpleError, match=r"needs y_between=\[low, high\]"):
        chart.zoom_box(x_between=[1, 2])
    with pytest.raises(PlotsimpleError, match="has no option called x_too"):
        chart.zoom_box(x_between=[1, 2], x_too=3)


def test_zoom_box_span_order_does_not_matter(marks):
    fig = ps.chart(marks).lines(x="height", y="marks").zoom_box(x_between=(155, 145), y_between=[80, 40])._render()
    assert fig.axes[0].child_axes[0].get_xlim() == (145, 155)


# ---------------------------------------------------------------- animation
def _frames(path):
    with Image.open(path) as im:
        return im.n_frames, im.info.get("duration")


def test_animate_gif_frames(marks, tmp_path):
    chart = ps.chart(marks).dots(x="height", y="marks", colour="class").animate(over="test", seconds_per_frame=0.5)
    path = tmp_path / "growth.gif"
    chart.save(path)
    n, duration = _frames(path)
    assert n == 3
    assert duration == 500


def test_animate_keep_builds_up(marks):
    chart = ps.chart(marks).dots(x="height", y="marks").animate(over="test", keep=True)
    frames = list(chart._frames())
    assert [len(t) for _, t in frames] == [8, 16, 24]
    chart2 = ps.chart(marks).dots(x="height", y="marks").animate(over="test")
    assert [len(t) for _, t in chart2._frames()] == [8, 8, 8]


def test_animate_fixed_ranges_and_last_frame(marks):
    still = ps.chart(marks).dots(x="height", y="marks")._render().axes[0]
    chart = ps.chart(marks).dots(x="height", y="marks").animate(over="test").title_top("Growth")
    ax = chart._render().axes[0]
    assert np.allclose(ax.get_xlim(), still.get_xlim()) and np.allclose(ax.get_ylim(), still.get_ylim())
    assert ax.get_title() == "Growth  (test: 3)"
    assert len(ax.collections[0].get_offsets()) == 8


def test_animate_with_one_chart_per(marks, tmp_path):
    chart = ps.chart(marks).bars(x="name", y="marks").one_chart_per("class").animate(over="test")
    chart.save(tmp_path / "a.gif")
    assert _frames(tmp_path / "a.gif")[0] == 3


def test_gif_without_animate(marks, tmp_path):
    with pytest.raises(PlotsimpleError, match="animate"):
        ps.chart(marks).bars(x="name", y="marks").save(tmp_path / "x.gif")


def test_animate_errors(marks):
    chart = ps.chart(marks)
    with pytest.raises(PlotsimpleError, match='Did you mean "test"'):
        chart.animate(over="tset")
    with pytest.raises(PlotsimpleError, match="bigger than 0"):
        chart.animate(over="test", seconds_per_frame=0)
    with pytest.raises(PlotsimpleError, match="nothing to animate"):
        ps.chart(marks[marks["test"] == 1]).animate(over="test")


def test_dark_style_uses_dark_colours():
    import plotsimple as ps
    from plotsimple._style import palette
    chart = ps.chart().bars(x=["a", "b"], y=[1, 2]).style("dark")
    face = chart._render().axes[0].patches[0].get_facecolor()
    from matplotlib.colors import to_rgba
    assert np.allclose(face, to_rgba(palette({"style": "dark"})[0]))
