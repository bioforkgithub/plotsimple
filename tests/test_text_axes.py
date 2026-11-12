"""Tests for the text words (caption, note, numbers on bars, font size) and the axes words."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.text import Annotation

import plotsimple as ps
from plotsimple import PlotsimpleError


@pytest.fixture
def weather():
    return pd.DataFrame({
        "month": list(range(1, 13)),
        "temperature": [1, 2, 4, 8, 12, 15, 17, 16, 13, 8, 4, 2],
        "rain": [90, 70, 65, 50, 45, 60, 70, 85, 110, 130, 120, 100],
    })


def texts(ax):
    return [t.get_text() for t in ax.texts]


def draw(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # runs constrained layout, so positions are final
    return canvas.get_renderer()


# ================================================================ caption_bottom
def test_caption_under_x_label(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").label_x("Student")
    chart.caption_bottom("Data from 2026").rotate_labels_x()
    fig = chart._render()
    ax = fig.axes[0]
    assert "Data from 2026" in texts(ax)
    r = draw(fig)
    caption = [t for t in ax.texts if t.get_text() == "Data from 2026"][0]
    cap, lab = caption.get_window_extent(r), ax.xaxis.label.get_window_extent(r)
    assert cap.y1 <= lab.y0 + 0.5            # below the x label, no overlap
    assert cap.y0 >= 0                       # still inside the figure (layout made room)


def test_caption_order_does_not_matter(marks):
    chart = ps.chart(marks).caption_bottom("first").bars(x="name", y="marks")
    assert "first" in texts(chart._render().axes[0])


def test_caption_replaces_earlier(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").caption_bottom("a").caption_bottom("b")
    t = texts(chart._render().axes[0])
    assert "b" in t and "a" not in t


def test_caption_several_panels_below_shared_label(marks):
    chart = ps.chart(marks).lines(x="test", y="marks").one_per("class")
    chart.label_x("Test").caption_bottom("Made-up marks")
    fig = chart._render()
    r = draw(fig)
    caption = [a for a in fig.artists if isinstance(a, Annotation)][0]
    assert caption.get_text() == "Made-up marks"
    assert fig.get_supxlabel().startswith("Test")
    for ax in fig.axes:  # below every panel
        assert caption.get_window_extent(r).y1 <= ax.get_tightbbox(r).y0 + 1
    assert caption.get_window_extent(r).y0 >= 0


def test_caption_several_panels_without_label(marks):
    chart = ps.chart(marks).lines(x="test", y="marks").one_per("class").caption_bottom("Only caption")
    assert chart._render().get_supxlabel() == "Only caption"


def test_caption_needs_words(marks):
    with pytest.raises(PlotsimpleError, match="needs some words"):
        ps.chart(marks).caption_bottom("  ")


# ================================================================ note
def test_note_at_place(weather):
    chart = ps.chart(weather).lines(x="month", y="temperature").note("Warmest", x=7, y=17)
    ax = chart._render().axes[0]
    note = [t for t in ax.texts if t.get_text() == "Warmest"][0]
    assert note.xy == (7, 17)
    assert note.arrow_patch is None


def test_note_with_arrow_points_at_place_and_stays_inside(weather):
    chart = ps.chart(weather).lines(x="month", y="temperature").note("Warmest", x=7, y=17, arrow=True)
    fig = chart._render()
    ax = fig.axes[0]
    note = [t for t in ax.texts if t.get_text() == "Warmest"][0]
    assert note.xy == (7, 17)
    assert note.arrow_patch is not None
    r = draw(fig)
    box, inside = note.get_window_extent(r), ax.get_window_extent(r)
    assert box.y1 <= inside.y1 and box.x0 >= inside.x0  # high point: text goes down, into the chart


def test_note_on_bar_name(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").note("Best", x="Ben", y=90)
    note = [t for t in chart._render().axes[0].texts if t.get_text() == "Best"][0]
    assert note.xy == (1, 90)


def test_note_unknown_bar_name(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").note("Best", x="Bne", y=90)
    with pytest.raises(PlotsimpleError, match="Names you have: Asha, Ben"):
        chart._render()


def test_note_needs_place(marks):
    with pytest.raises(PlotsimpleError, match=r"x= and y="):
        ps.chart(marks).note("Hi")
    with pytest.raises(PlotsimpleError, match="arrow=True"):
        ps.chart(marks).note("Hi", x=1, y=2, arrow="yes")
    with pytest.raises(PlotsimpleError, match="needs a number for x="):
        ps.chart(marks).note("Hi", x=[1], y=2)


def test_several_notes_kept(weather):
    chart = ps.chart(weather).lines(x="month", y="temperature")
    chart.note("a", x=1, y=1).note("b", x=2, y=2)
    t = texts(chart._render().axes[0])
    assert "a" in t and "b" in t


# ================================================================ numbers_on_bars
def bar_numbers(ax):
    return sorted(t.get_text() for t in ax.texts if t.get_text())


def test_numbers_on_bars(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").numbers_on_bars(decimals=1)
    ax = chart._render().axes[0]
    expected = sorted(f"{v:.1f}" for v in marks.groupby("name")["marks"].mean())
    assert bar_numbers(ax) == expected


def test_numbers_on_bars_stay_inside(marks):
    chart = ps.chart(marks).bars(x="name", y="marks").numbers_on_bars()
    fig = chart._render()
    ax = fig.axes[0]
    r = draw(fig)
    top = ax.get_window_extent(r).y1
    assert all(t.get_window_extent(r).y1 <= top for t in ax.texts if t.get_text())


def test_numbers_on_sideways_bars(marks):
    chart = ps.chart(marks).bars(x="name", y="marks", sideways=True).numbers_on_bars()
    ax = chart._render().axes[0]
    expected = sorted(f"{v:.0f}" for v in marks.groupby("name")["marks"].mean())
    assert bar_numbers(ax) == expected
    assert all(t.get_horizontalalignment() == "left" for t in ax.texts if t.get_text())


def test_numbers_on_stacked_bars(marks):
    chart = ps.chart(marks).bars(x="name", y="marks", colour="class", stacked=True, show="total")
    chart.numbers_on_bars()
    ax = chart._render().axes[0]
    expected = sorted(f"{v:,.0f}" for v in marks.groupby(["name", "class"])["marks"].sum())
    assert bar_numbers(ax) == expected
    # each number sits in the middle of its own piece
    assert all(t.get_verticalalignment() == "center" for t in ax.texts if t.get_text())


def test_numbers_on_grouped_bars(marks):
    chart = ps.chart(marks).bars(x="name", y="marks", colour="class").numbers_on_bars()
    ax = chart._render().axes[0]
    assert len(bar_numbers(ax)) == 8


def test_numbers_go_past_error_bars(marks):
    chart = ps.chart(marks).bars(x="name", y="marks", error="sd").numbers_on_bars()
    ax = chart._render().axes[0]
    sd_top = (marks.groupby("name", sort=False)["marks"].mean()
              + marks.groupby("name", sort=False)["marks"].std()).to_numpy()
    tops = sorted(t.xy[1] for t in ax.texts if t.get_text())
    assert np.allclose(tops, sorted(sd_top))


def test_numbers_need_bars(weather):
    chart = ps.chart(weather).lines(x="month", y="rain").numbers_on_bars()
    with pytest.raises(PlotsimpleError, match="no bars"):
        chart._render()


def test_numbers_decimals_checked(marks):
    with pytest.raises(PlotsimpleError, match="whole number"):
        ps.chart(marks).numbers_on_bars(decimals=1.5)


# ================================================================ font_size
def test_font_size_scales_all_text(marks):
    small = ps.chart(marks).bars(x="name", y="marks").title_top("T").label_x("X")
    big = ps.chart(marks).bars(x="name", y="marks").title_top("T").label_x("X").font_size(24)
    a, b = small._render().axes[0], big._render().axes[0]
    assert b.title.get_fontsize() == pytest.approx(2 * a.title.get_fontsize())
    assert b.xaxis.label.get_fontsize() == pytest.approx(2 * a.xaxis.label.get_fontsize())
    assert b.get_xticklabels()[0].get_fontsize() == pytest.approx(2 * a.get_xticklabels()[0].get_fontsize())


def test_font_size_checked(marks):
    with pytest.raises(PlotsimpleError, match="bigger than 0"):
        ps.chart(marks).font_size(-3)
    with pytest.raises(PlotsimpleError, match="very big"):
        ps.chart(marks).font_size(500)
    with pytest.raises(PlotsimpleError, match="bigger than 0"):
        ps.chart(marks).font_size("big")


# ================================================================ range
def test_range_y_both_ends(weather):
    ax = ps.chart(weather).lines(x="month", y="rain").range_y(0, 200)._render().axes[0]
    assert ax.get_ylim() == (0, 200)


def test_range_one_end_only(weather):
    ax = ps.chart(weather).lines(x="month", y="rain").range_y(0, None)._render().axes[0]
    low, high = ax.get_ylim()
    assert low == 0 and high >= 130
    ax = ps.chart(weather).lines(x="month", y="rain").range_x(None, 6)._render().axes[0]
    low, high = ax.get_xlim()
    assert high == 6 and low <= 1


def test_range_and_flip_in_any_order(weather):
    for chart in (ps.chart(weather).lines(x="month", y="rain").range_y(0, 200).flip_y(),
                  ps.chart(weather).lines(x="month", y="rain").flip_y().range_y(0, 200)):
        assert chart._render().axes[0].get_ylim() == (200, 0)


def test_range_errors(weather):
    chart = ps.chart(weather)
    with pytest.raises(PlotsimpleError, match="at least one end"):
        chart.range_y()
    with pytest.raises(PlotsimpleError, match=r"(?s)range_y\(100, 0\).*flip_y"):
        chart.range_y(100, 0)
    with pytest.raises(PlotsimpleError, match="needs numbers"):
        chart.range_x("0", 10)


def test_range_one_end_past_other_end(weather):
    chart = ps.chart(weather).lines(x="month", y="rain").range_y(500, None)
    with pytest.raises(PlotsimpleError, match="empty"):
        chart._render()


# ================================================================ log scale
def test_log_scale(weather):
    ax = ps.chart(weather).dots(x="month", y="rain").log_scale_x().log_scale_y()._render().axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"


def test_log_scale_with_ticks_and_range(weather):
    chart = ps.chart(weather).dots(x="month", y="rain").ticks_y(50, 100).range_y(10, 1000).log_scale_y()
    ax = chart._render().axes[0]
    assert ax.get_yscale() == "log"
    assert list(ax.get_yticks()) == [50, 100]   # the log scale did not wipe the ticks
    assert ax.get_ylim() == pytest.approx((10, 1000))


def test_log_scale_zero_is_friendly():
    chart = ps.chart(pd.DataFrame({"a": [1, 2, 3], "b": [0, 5, -1]})).dots(x="a", y="b").log_scale_y()
    with pytest.raises(PlotsimpleError, match=r'(?s)"b" has 2 of them.*remove those rows'):
        chart._render()


def test_log_scale_on_bars_checks_bar_values():
    data = pd.DataFrame({"name": ["a", "b"], "v": [3, 0]})
    chart = ps.chart(data).bars(x="name", y="v").log_scale_y()
    with pytest.raises(PlotsimpleError, match="zero or negative"):
        chart._render()
    with pytest.raises(PlotsimpleError, match="holds words"):
        ps.chart(data).bars(x="name", y="v").log_scale_x()._render()


def test_log_scale_and_range_from_zero(weather):
    with pytest.raises(PlotsimpleError, match="above 0"):
        ps.chart(weather).range_y(0, 10).log_scale_y()
    with pytest.raises(PlotsimpleError, match="above 0"):
        ps.chart(weather).log_scale_y().range_y(0, 10)


# ================================================================ ticks
def test_ticks(weather):
    ax = ps.chart(weather).lines(x="month", y="rain").ticks_x(0, 6, 12)._render().axes[0]
    assert list(ax.get_xticks()) == [0, 6, 12]


def test_ticks_with_labels_and_list(weather):
    chart = ps.chart(weather).lines(x="month", y="rain").ticks_x([1, 7], labels=["Jan", "Jul"])
    ax = chart._render().axes[0]
    assert list(ax.get_xticks()) == [1, 7]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Jan", "Jul"]


def test_ticks_by_bar_name(marks):
    ax = ps.chart(marks).bars(x="name", y="marks").ticks_x("Asha", "Dina")._render().axes[0]
    assert list(ax.get_xticks()) == [0, 3]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Asha", "Dina"]


def test_ticks_errors(weather):
    chart = ps.chart(weather)
    with pytest.raises(PlotsimpleError, match=r"ticks_y\(0, 10, 20, 30\)"):
        chart.ticks_y()
    with pytest.raises(PlotsimpleError, match="3 ticks but 2 labels"):
        chart.ticks_x(1, 2, 3, labels=["a", "b"])
    with pytest.raises(PlotsimpleError, match="should be a list"):
        chart.ticks_x(1, labels="a")


# ================================================================ rotate
def test_rotate_labels_x(marks):
    ax = ps.chart(marks).bars(x="name", y="marks").rotate_labels_x()._render().axes[0]
    for label in ax.get_xticklabels():
        assert label.get_rotation() == 45
        assert label.get_horizontalalignment() == "right"
        assert label.get_rotation_mode() == "anchor"


def test_rotate_after_ticks_whatever_the_order(weather):
    chart = ps.chart(weather).lines(x="month", y="rain").rotate_labels_x(90).ticks_x(1, 6, 12)
    ax = chart._render().axes[0]
    assert [t.get_rotation() for t in ax.get_xticklabels()] == [90, 90, 90]


def test_rotate_labels_y(weather):
    ax = ps.chart(weather).lines(x="month", y="rain").rotate_labels_y(30)._render().axes[0]
    assert all(t.get_rotation() == 30 for t in ax.get_yticklabels())


def test_rotate_checked(marks):
    with pytest.raises(PlotsimpleError, match="angle in degrees"):
        ps.chart(marks).rotate_labels_x("45")


# ================================================================ flip
def test_flip(weather):
    chart = ps.chart(weather).lines(x="month", y="rain").flip_x().flip_y().flip_y()
    ax = chart._render().axes[0]
    assert ax.xaxis_inverted() and ax.yaxis_inverted()  # calling twice does not flip back


def test_flip_shared_panels(marks):
    chart = ps.chart(marks).lines(x="test", y="marks").one_per("class").flip_y()
    assert all(ax.yaxis_inverted() for ax in chart._render().axes)


# ================================================================ second_y
def second(fig):
    main = fig.axes[0]
    twins = [a for a in fig.axes[1:] if a.get_shared_x_axes().joined(a, main)]
    assert len(twins) == 1
    return main, twins[0]


def test_second_y_lines(weather):
    chart = ps.chart(weather).bars(x="month", y="temperature")
    chart.second_y("rain", label="Rain (mm)")
    main, twin = second(chart._render())
    line = twin.lines[0]
    assert list(line.get_ydata()) == weather["rain"].tolist()
    assert list(line.get_xdata()) == list(range(12))       # above each bar
    assert twin.get_ylabel() == "Rain (mm)"
    assert to_hex(line.get_color()) != to_hex(main.patches[0].get_facecolor())
    leg = main.get_legend() or twin.get_legend()
    assert [t.get_text() for t in leg.texts] == ["temperature", "Rain (mm)"]


def test_second_y_bars_under_lines(weather):
    chart = ps.chart(weather).lines(x="month", y="temperature").second_y("rain", chart="bars")
    main, twin = second(chart._render())
    assert [p.get_height() for p in twin.patches] == weather["rain"].tolist()
    assert main.get_zorder() > twin.get_zorder()   # the line is drawn in front of the bars
    assert [t.get_text() for t in main.get_legend().texts] == ["temperature", "rain"]


def test_second_y_bars_next_to_bars(weather):
    chart = ps.chart(weather).bars(x="month", y="temperature").second_y("rain", chart="bars")
    main, twin = second(chart._render())
    for a, b in zip(main.patches, twin.patches):
        assert a.get_x() + a.get_width() == pytest.approx(b.get_x())  # side by side, not on top


def test_second_y_dots_and_colour(weather):
    chart = ps.chart(weather).lines(x="month", y="temperature").second_y("rain", chart="dots", colour="red")
    _, twin = second(chart._render())
    pts = twin.collections[0].get_offsets()
    assert list(pts[:, 1]) == weather["rain"].tolist()
    assert to_hex(twin.collections[0].get_facecolor()[0]) == to_hex("red")


def test_second_y_keeps_groups_in_legend(marks):
    chart = ps.chart(marks).lines(x="test", y="marks", colour="class").second_y("height")
    main, twin = second(chart._render())
    leg = main.get_legend() or twin.get_legend()
    assert [t.get_text() for t in leg.texts] == ["A", "B", "height"]
    # a colour the groups do not use
    used = {to_hex(line.get_color()) for line in main.lines}
    assert to_hex(twin.lines[0].get_color()) not in used


def test_second_y_errors(weather):
    with pytest.raises(PlotsimpleError, match='Did you mean "rain"'):
        ps.chart(weather).second_y("rian")
    with pytest.raises(PlotsimpleError, match='Did you mean chart="bars"'):
        ps.chart(weather).second_y("rain", chart="bar")
    with pytest.raises(PlotsimpleError, match="one colour name"):
        ps.chart(weather).second_y("rain", colour=3)
    with pytest.raises(PlotsimpleError, match="needs a chart to sit on"):
        ps.chart(weather).second_y("rain")._render()
    with pytest.raises(PlotsimpleError, match="sideways"):
        ps.chart(weather).bars(x="month", y="temperature", sideways=True).second_y("rain")._render()


def test_second_y_stored_in_store(weather):
    seen = {}
    chart = ps.chart(weather).lines(x="month", y="temperature").second_y("rain")
    chart._add("peek", "finish", lambda ctx: seen.update(ctx.store))
    chart._render()
    assert "second_y" in seen


# ================================================================ a full chart saves
def test_everything_together_saves(tmp_path, weather):
    chart = ps.chart(weather).lines(x="month", y="temperature", dots=True)
    chart.second_y("rain", chart="bars", label="Rain (mm)").title_top("Weather").label_x("Month")
    chart.caption_bottom("Made-up numbers").note("Warmest", x=7, y=17, arrow=True)
    chart.ticks_x(range(1, 13), labels=list("JFMAMJJASOND")).rotate_labels_x(0).range_y(0, None).font_size(14)
    chart.save(tmp_path / "weather.png")
    assert (tmp_path / "weather.png").stat().st_size > 0
